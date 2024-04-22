import pathlib
import re
from copy import copy

import numpy as np
import pandas as pd

from hglm.effect import Effect
from hglm.effect import compute_offset
from hglm.mask import get_mask_idx
from .load_image import load_image_color, load_image_nii
from .permute import Permuter


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

                # todo: check that duplicate file doesn't already exist
                df.loc[sbj, y_feat] = file

        assert df.size, 'no images found'

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

    def sample_x(self, a=None, contrast=None, seed=None, add_bias=True):
        """ generates (or replaces) x with an arbitrary std normal noise

        Args:
            a (int): number of x features in output
            contrast (np.array): (a) True for each corresponding feature in x
                which is "of interest" (other x features form the reduced
                model in computing f statistic)
            seed: used for random number generator
            add_bias (bool): if True, first explanatory feature is bias term
                (constant row of ones)

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

        if add_bias:
            contrast[0] = False
            x[0, :] = 1

        return Experiment(x=x, contrast=contrast, y=self.y,
                          mask_idx=self.mask_idx,
                          add_bias=False)

    def impose_effect(self, extenter=None, mask=None, seed=None, **kwargs):
        """ builds experiment with effect imposed

        Args:
            extenter (ExtenterSphere or ExtenterMinVar): identifies volume to
                impose effect on
            seed: seed of random number generator (for extent)
            **kwargs: passed to compute_offset(), either p_val or f_stat

        Returns:
            exp (Experiment): an experiment
            effect (Effect): encapsulates
        """
        assert self.x is not None, 'x/contrast needed, call .sample_x()'
        assert (mask is None) != (extenter is None), \
            'either mask xor extenter needed'

        if mask is None:
            # sample effect space
            mask = extenter(y=self.y, mask_idx=self.mask_idx, seed=seed)

        # get offset which imposes desired effect strength
        effect_idx = self.mask_idx[mask]
        y_effect = self.y[:, :, effect_idx]
        offset = compute_offset(x=self.x, y=y_effect, contrast=self.contrast,
                                **kwargs)

        # impose effect on y, build new experiment
        exp = self.add_offset(offset, mask=mask)

        effect = Effect.from_exp_mask(exp=exp, mask=mask)

        return exp, effect

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
        mask_idx = get_mask_idx(mask)
        y = self.y[:, :, self.mask_idx[mask]]

        # build new object identical as self (references where possible) but
        # replace y & mask_idx with above
        d = copy(self.__dict__)
        d['mask_idx'] = mask_idx
        d['y'] = y
        return type(self)(**d)

    def add_offset(self, offset, mask=None, vox_idx=None):
        """ returns new experiment with constant offset added to y

        Args:
            offset (np.array): (b, num_img) offset to apply to each voxel
            mask (np.array): boolean area of locations to apply offset to
            vox_idx (list): list of voxel index to apply effect to

        Returns:
            exp (Experiment): new experiment, with offset applied
        """
        assert (mask is None) != (vox_idx is None), \
            'either mask xor vox_idx required'

        if vox_idx is None:
            vox_idx = self.mask_idx[mask]

        # build new y
        y = copy(self.y)
        y[:, :, vox_idx] += offset[..., np.newaxis]

        # build new object identical as self (references where possible) but
        # replace y with above
        d = copy(self.__dict__)
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

    def __init__(self, *, x, contrast=None, add_bias=False, **kwargs):
        super().__init__(**kwargs)

        self.x = x
        self.contrast = contrast

        if add_bias:
            # append row of ones (bias term) to x
            num_img = x.shape[1]
            self.x = np.vstack([np.ones(num_img), x])

            # append leading False to contrast (its not of interest)
            self.contrast = np.insert(self.contrast, 0, values=False)

        # build residual forming arrays
        x = self.x[~self.contrast, :], self.x
        self.h = tuple(np.linalg.pinv(_x) @ _x for _x in x)

    def rm_effect(self, effect):
        """ builds a new experiment which has the effect removed

        Args:
            effect (Effect): effect to add to experiment

        Returns:
            exp_out (Experiment): experiment whose y features have had the
                effect subtracted away
        """
        num_img = self.x.shape[1]
        offset = -effect.y_mean @ self.h[1] @ (np.eye(num_img) - self.h[0])

        return self.add_offset(offset=offset, mask=effect.mask)

    def permute(self, perm_idx, block_exchange=False):
        """ gets new experiment whose y features were permuted (freedman lane)

        Args:
            perm_idx (int): permutation index (0 is no permutation)
            block_exchange (bool): toggles block exchange permuting.  all
                voxels are permuted with same permutation matrix (faster, but
                resulting experiment may still contain shadows of effects in
                original data).  setting to False will give each voxel its own
                permutation matrix, more computationally expensive.

        Returns:
            exp (Experiment): new experiment whose y features have been
                permuted
        """
        # build a permutation object
        perm = Permuter(x=self.x[~self.contrast, :])

        if perm_idx == 0:
            # perm_idx = 0 is reserved for unpermuted data
            y = copy(self.y)
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


class ExperimentWhitened(Experiment):
    """ imaging data whitened (ZCA)

    Attributes:
        to_white (np.array): (b, b) transform from original to white data
        from_white (np.array): (b, b) from whitened back to original
        y_orig (np.array): (b, num_img, num_vox) original imaging features
    """

    @classmethod
    def from_exp(cls, exp):
        return ExperimentWhitened(y=exp.y, mask_idx=exp.mask_idx, x=exp.x,
                                  contrast=exp.contrast)

    def __init__(self, y, mask_idx, **kwargs):
        # computing zca whitening transform (and inverse)
        _y = y.reshape((y.shape[0], -1), order='F')
        cov = np.atleast_2d(np.cov(_y))
        assert np.linalg.matrix_rank(cov) == cov.shape[0], \
            'redundant imaging feature'
        u, s, _ = np.linalg.svd(cov)
        self.to_white = np.diag(s ** -.5) @ u.T
        self.from_white = np.linalg.inv(self.to_white)

        # store original
        self.y_orig = y

        # whiten & build experiment
        y = np.einsum('ab,bnr->anr', self.to_white, y)
        super().__init__(y=y, mask_idx=mask_idx, **kwargs)
