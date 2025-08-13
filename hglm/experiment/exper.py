import pathlib
import re
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.linalg

import hglm.effect
import hglm.mask
from .load_image import load_image_color, load_image_nii
from .permute import Permuter
from .sigma import stretch_sigma
from ..mask import get_mask_idx


class ExperimentImageOnly:
    """ contains all imaging data of an experiment

    Attributes:
        y (np.array): (b, num_img, num_vox) image intensities
        mask_idx (np.array): same shape as image.  -1 where voxel not
            included in analysis, otherwise contains voxel index
    """

    def __init__(self, *, y, mask_idx, **kwargs):
        self.y = y
        self.mask_idx = mask_idx

    @classmethod
    def from_gauss(cls, b=None, num_img=10, shape=(2, 3, 4), seed=None,
                   mu=None, cov=None, **kwargs):
        """ generates gaussian data, outputs sample mu & cov as given

        Args:
            b (int): dimension of imaging features (default b=1)
            num_img (int): number of "images" to sample
            shape (tuple): shape of images
            seed (int): inits random number generator
            mu (np.array): output sample mean (default to np.zeros(b))
            cov (np.array): output sample cov, with bessel's (default to
                np.eye(b))

        Returns:
            ExperimentImageOnly
        """
        if b is None:
            if mu is not None:
                b = mu.size
            elif cov is not None:
                b = cov.shape[0]
            else:
                b = 1

        num_vox = np.prod(shape)
        rng = np.random.default_rng(seed=seed)
        y = rng.multivariate_normal(np.zeros(b), np.eye(b), num_img * num_vox)

        # reshape, de-mean, reshape
        y = y.reshape((b, num_img, num_vox))
        y = y - y.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]
        y = y.reshape((b, -1))

        if cov is not None:
            # project to proper covariance
            _cov = y @ y.T / (num_img * num_vox - 1)
            p = np.linalg.inv(scipy.linalg.sqrtm(_cov))
            p = scipy.linalg.sqrtm(cov) @ p
            y = p @ y

        if mu is not None:
            # add to proper mean
            y = y + mu[:, np.newaxis]

        return cls(y=y.reshape((b, num_img, num_vox)),
                   mask_idx=get_mask_idx(np.ones(shape)),
                   **kwargs)

    @classmethod
    def from_search(cls, folder, sbj_regex, img_glob_dict, **kwargs):
        """ searches path for matching image files

        Args:
            folder (str): folder to search (recursively) for image files
            sbj_regex (str): regex string which matches on subject names,
                must match once on each image's full path
            img_glob_dict (dict): keys are str names of each image type
                (e.g. 'FA', 'MD'), values are glob strs (e.g. '*_FA.nii.gz')

        Returns:
            experiment (Experiment): experiment generated from all images found
        """
        # find all images
        folder = pathlib.Path(folder)
        df = pd.DataFrame()
        for y_feat, y_glob in img_glob_dict.items():
            for file in folder.glob(y_glob):
                # extract sbj from full path of file
                sbj_list = re.findall(sbj_regex, str(file))
                assert len(sbj_list) == 1, \
                    f'unique sbj not found in file: {file}'
                sbj = sbj_list[0]

                df.loc[sbj, y_feat] = file

        assert df.size, 'no images found'
        assert len(set(df.values.flatten())) == np.prod(df.shape), \
            'file repeated for more than one subject-feature pair'

        # check if any subject is missing any imaging feature
        s_missing = df.isna().mean(axis=1)
        if s_missing.any():
            print('some sbj missing files:')
            print(df.loc[s_missing, :].notnull().astype('int'))

        nii_in_file = ['.nii' in str(file) for file in df.values.flatten()]
        if all(nii_in_file):
            feat_sbj_img, mask_idx = load_image_nii(df)
        elif not any(nii_in_file):
            feat_sbj_img, mask_idx = load_image_color(df)
        else:
            raise TypeError('may not mix nifti and color images in input')

        # ensure all input has same data type
        dtype = None
        for _, sbj_img in feat_sbj_img.items():
            for _, img in sbj_img.items():
                if dtype is None:
                    dtype = img.dtype
                else:
                    assert dtype == img.dtype, 'dtype mismatch'

        # mask into each image, store as y
        mask = mask_idx >= 0
        y_names = sorted(feat_sbj_img.keys())
        y = np.empty((len(feat_sbj_img), df.shape[0], mask.sum()))
        for sbj_idx, sbj in enumerate(sorted(df.index)):
            for feat_idx, feat in enumerate(y_names):
                y[feat_idx, sbj_idx, :] = feat_sbj_img[feat][sbj][mask]

        return cls(y=y, mask_idx=mask_idx, **kwargs)

    def bootstrap_img(self, n, seed=None, noise_scale=0):
        """ bootstrap resamples images

        Args:
            n (int): number of images in the resulting Experiment
            seed: used for random number generator
            noise_scale (float): scales noise std_dev.  noise is drawn
                independently, per voxel, from a normal distribution whose
                covariance is the (b, b) sample covariance of self.y

        Returns:
            exp_out (Experiment): y has been bootstrap resampled from self
        """
        # bootstrap sample images
        rng = np.random.default_rng(seed=seed)
        b, num_img_init, num_vox = self.y.shape
        img_idx = rng.choice(num_img_init, n, replace=True)
        self.y = self.y[:, img_idx, :]

        # add noise
        assert noise_scale >= 0, 'snr cannot be negative'
        if noise_scale > 0:
            cov = np.atleast_2d(np.cov(self.y.reshape((b, -1))))
            noise = rng.multivariate_normal(mean=np.zeros(b),
                                            cov=cov * (noise_scale ** 2),
                                            size=num_vox * n)
            self.y += noise.T.reshape((b, n, num_vox))

    def sample_x(self, a=None, contrast=None, seed=None, **kwargs):
        """ generates (or replaces) x with an arbitrary std normal noise

        Args:
            a (int): number of x features in output
            contrast (np.array): (a) True for each corresponding feature in x
                which is "of interest"
            seed: used for random number generator

        Returns:
            exp_out (Experiment): x has been replaced with noise from self
        """
        assert (a is None) != (contrast is None), 'a xor contrast required'

        if a is None:
            # contrast specified, extract a from it
            a = contrast.size
            contrast = contrast
        else:
            # default contrast: all x of interest but bias term
            contrast = np.ones(a, dtype=bool)

        # sample x
        num_img = self.y.shape[1]
        rng = np.random.default_rng(seed=seed)
        x = rng.standard_normal(size=(a, num_img))

        return Experiment(x=x, contrast=contrast, y=self.y,
                          mask_idx=self.mask_idx, **kwargs)

    def apply_mask(self, mask):
        """ applies boolean mask to experiment

        Args:
            mask (np.array): True where voxels included, false otherwise

        Returns:
            exp (Experiment): experiment corresponding to intersection of mask
                and self
        """
        # apply mask to data
        mask = np.logical_and(mask, self.mask_idx > -1)
        assert mask.sum(), 'mask has no intersection with mask_idx'
        mask_idx = hglm.mask.get_mask_idx(mask)
        y = self.y[:, :, self.mask_idx[mask]]

        # build new object identical as self
        d = deepcopy(self.__dict__)
        d['mask_idx'] = mask_idx
        d['y'] = y
        return type(self)(**d)

    def add_offset(self, offset, mask=None, vox_idx=None, sigma_scale=None):
        """ returns new experiment with constant offset added to y

        Args:
            offset (np.array): (b, num_img) offset to apply to each voxel
            mask (np.array): boolean area of locations to apply offset to
            vox_idx (list): list of voxel index to apply effect to
            sigma_scale (float): sigma scaling fator (see stretch_sigma())

        Returns:
            exp (Experiment): new experiment, with offset applied
        """
        assert (mask is None) != (vox_idx is None), \
            'either mask xor vox_idx required'

        if vox_idx is None:
            vox_idx = self.mask_idx[mask]

        # build new y
        y = deepcopy(self.y)
        y[:, :, vox_idx] += offset[..., np.newaxis]

        if sigma_scale is not None:
            # scale sigma within mask, if passed
            y[:, :, vox_idx] = stretch_sigma(y=y[:, :, vox_idx],
                                             scale=sigma_scale)

        # build new object
        d = deepcopy(self.__dict__)
        d['y'] = y
        return type(self)(**d)


class Experiment(ExperimentImageOnly):
    """ contains complete set of data needed to run an experiment

    Attributes:
        x (np.array): (a, num_img) explanatory variables
        contrast (np.array): (a) True for each corresponding feature in x which
            is "of interest" (other x features form the reduced model in
            computing f statistic)
    """

    @classmethod
    def from_gauss(cls, a=1, contrast=None, seed=None, add_bias=True,
                   **kwargs):
        exp = ExperimentImageOnly.from_gauss(seed=seed, **kwargs)
        return exp.sample_x(a=a, contrast=contrast, seed=seed, add_bias=True)

    def __init__(self, *, x, contrast=None, add_bias=False, **kwargs):
        super().__init__(**kwargs)

        self.x = x
        self.contrast = contrast

        if add_bias:
            # append row of ones (bias term) to x
            num_img = x.shape[1]
            self.x = np.vstack([np.ones(num_img), x])

            # append leading False to contrast (it's not of interest)
            self.contrast = np.insert(self.contrast, 0, values=False)

        if not np.any(np.all(self.x == 1, axis=1)):
            warnings.warn('no bias term: regression constrained to '
                          'origin (consider add_bias=True)')

    def impose_effect(self, hotel_tr, extenter=None, mask=None, seed=None,
                      **kwargs):
        """ builds experiment with effect imposed

        Args:
            hotel_tr (float): hotelling's trace
            extenter (ExtenterSphere or ExtenterMinVar): identifies volume to
                impose effect on
            mask (np.array): is passed, will impose effect on
            seed: seed of random number generator (for extent)
            **kwargs: passed to compute_offset(), either pval or f_stat

        Returns:
            exp (Experiment): an experiment
            effect (Effect): encapsulates
            **kwargs: passed to compute_offset()
        """
        assert self.x is not None, 'x/contrast needed, call .sample_x()'
        assert (mask is None) != (extenter is None), \
            'either mask xor extenter needed'

        if mask is None:
            # sample effect space
            mask = extenter(y=self.y, mask_idx=self.mask_idx, seed=seed)

        # extract y of effect region (before effect applied)
        effect_idx = self.mask_idx[mask]
        y = self.y[:, :, effect_idx]

        # get offset which imposes desired effect strength
        offset = hglm.effect.compute_offset(x=self.x,
                                            y=y,
                                            contrast=self.contrast,
                                            hotel_tr=hotel_tr)

        # impose effect on y, build new experiment
        exp = self.add_offset(offset, mask=mask)

        effect = hglm.effect.Effect.from_exp_mask(exp=exp,
                                                  mask=mask,
                                                  seed=seed,
                                                  hotel_tr=hotel_tr)

        return exp, effect

    def permute(self, perm_idx, block_exchange=True):
        """ gets new experiment whose y features were permuted (freedman lane)

        Args:
            perm_idx (int): permutation index (0 is no permutation)
            block_exchange (bool): toggles block exchange permuting.  all
                voxels are permuted with same permutation matrix.  setting
                to False will give each voxel its own permutation matrix

        Returns:
            exp (Experiment): new experiment whose y features have been
                permuted
        """
        # build a permutation object
        perm = Permuter(x=self.x[~self.contrast, :])

        if perm_idx == 0:
            # perm_idx = 0 is reserved for unpermuted data
            y = deepcopy(self.y)
        elif block_exchange:
            y = perm(self.y, n_perm=1, perm_idx_min=perm_idx, keep_orig=False)
            y = y[:, :, :, 0]
        else:
            y = np.empty_like(self.y)
            for vox_idx in range(y.shape[2]):
                # each voxel gets its own permutation index
                y[:, :, vox_idx] = perm(self.y[:, :, vox_idx],
                                        n_perm=1,
                                        perm_idx_min=perm_idx + vox_idx,
                                        keep_orig=False)[:, :, 0]

        return Experiment(x=self.x, y=y, contrast=self.contrast,
                          mask_idx=self.mask_idx)


class ExperimentScaled(Experiment):
    """ pre-process (zero mean, scale_normalize, and pca, in that order)

    y_out = pre_scale @ (y_in - mean_orig)

    Attributes:
        mean_orig (np.array): (b, 1, 1) original average y value (across vox &
            image)
        pre_scale (np.array): (b, b) left multiplies y to
    """

    @classmethod
    def from_exp(cls, exp):
        return cls(y=exp.y, mask_idx=exp.mask_idx, x=exp.x,
                   contrast=exp.contrast)

    def prep(self, y):
        return np.einsum('ij,jkl->ikl',
                         self.pre_scale,
                         y - self.mean_orig)

    def prep_inv(self, y):
        return np.einsum('ij,jkl->ikl',
                         np.linalg.inv(self.pre_scale),
                         y) + self.mean_orig

    def __init__(self, y, *args, **kwargs):
        # zero mean (and make new copy)
        self.mean_orig = y.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]

        # scale normalize
        b, num_img, num_vox = y.shape
        cov = np.cov(y.reshape((b, -1), order='F'))
        cov = np.atleast_2d(cov)
        self.pre_scale = np.diag(1 / np.diag(cov) ** .5)

        cov_scale = self.pre_scale @ cov @ self.pre_scale.T
        evals, evecs = np.linalg.eig(cov_scale)
        self.pre_scale = evecs.T @ self.pre_scale

        super().__init__(y=self.prep(y), *args, **kwargs)
