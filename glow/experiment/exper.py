import pathlib
import re
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.linalg

import glow.effect
import glow.mask
from .load_image import load_image_color, load_image_nii
from .permute import get_freed_lane
from .sigma import stretch_sigma
from ..mask import get_mask_idx


class NoBiasTermWarning(UserWarning):
    """raised when regression is constrained to origin without bias term."""
    pass


class ExperimentImageOnly:
    """imaging data for an experiment (no design matrix).

    Attributes:
        y (np.array): (b, num_img, num_vox) image intensities
        mask_idx (np.array): voxel index array (-1 outside analysis)
        meta (dict): optional metadata (subjects, features, affine, etc.)
            not used by analysis — propagated for export / display
    """

    def __init__(self, *, y, mask_idx, meta=None, **kwargs):
        self.y = y
        self.mask_idx = mask_idx
        self.meta = meta if meta is not None else {}

    def _hash(self):
        """rolling SHA-256 hash over data arrays (16-char hex digest)."""
        import hashlib
        h = hashlib.sha256()
        for arr in (self.y, self.mask_idx):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()[:16]

    @classmethod
    def from_gauss(cls, b=None, num_img=10, shape=(2, 3, 4), seed=None,
                   mu=None, cov=None, **kwargs):
        """generate Gaussian imaging data with prescribed mean and covariance.

        Args:
            b (int): number of imaging features (default 1)
            num_img (int): number of images to sample
            shape (tuple): spatial shape of each image
            seed (int): random seed
            mu (np.array): target sample mean (default zeros)
            cov (np.array): target sample covariance (default identity)

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

        meta = kwargs.pop('meta', {})
        meta.setdefault('features', [f'feat_{i}' for i in range(b)])
        meta.setdefault('subjects', [f'subject_{i:03d}' for i in range(num_img)])
        return cls(y=y.reshape((b, num_img, num_vox)),
                   mask_idx=get_mask_idx(np.ones(shape)),
                   meta=meta, **kwargs)

    @classmethod
    def from_search(cls, folder, sbj_regex, img_glob_dict, **kwargs):
        """search a folder for images and build an experiment.

        Args:
            folder (str): root folder to search recursively
            sbj_regex (str): regex extracting the subject id from file paths
            img_glob_dict (dict): feature_name -> glob pattern

        Returns:
            Experiment built from discovered images
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
        affine = None
        if all(nii_in_file):
            feat_sbj_img, mask_idx, affine = load_image_nii(df)
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
        subjects = sorted(df.index)
        y = np.empty((len(feat_sbj_img), df.shape[0], mask.sum()))
        for sbj_idx, sbj in enumerate(subjects):
            for feat_idx, feat in enumerate(y_names):
                y[feat_idx, sbj_idx, :] = feat_sbj_img[feat][sbj][mask]

        meta = kwargs.pop('meta', {})
        meta.setdefault('subjects', [str(s) for s in subjects])
        meta.setdefault('features', list(y_names))
        if affine is not None:
            meta.setdefault('affine', affine)

        return cls(y=y, mask_idx=mask_idx, meta=meta, **kwargs)

    def bootstrap_img(self, n, seed=None, noise_scale=0):
        """return a new experiment with bootstrap-resampled images.

        Args:
            n (int): number of images in the output
            seed: random seed
            noise_scale (float): std-dev multiplier for additive noise
                (drawn per-voxel from the sample covariance)

        Returns:
            ExperimentImageOnly: new object with resampled y
        """
        # bootstrap sample images
        rng = np.random.default_rng(seed=seed)
        b, num_img_init, num_vox = self.y.shape
        img_idx = rng.choice(num_img_init, n, replace=True)
        y = self.y[:, img_idx, :].copy()

        # add noise
        assert noise_scale >= 0, 'snr cannot be negative'
        if noise_scale > 0:
            cov = np.atleast_2d(np.cov(y.reshape((b, -1))))
            noise = rng.multivariate_normal(mean=np.zeros(b),
                                            cov=cov * (noise_scale ** 2),
                                            size=num_vox * n)
            y = y + noise.T.reshape((b, n, num_vox))

        exp = deepcopy(self)
        exp.y = y
        return exp

    def sample_x(self, a=None, contrast=None, seed=None, **kwargs):
        """return a new Experiment with random standard-normal design matrix.

        Args:
            a (int): number of features (exactly one of a / contrast required)
            contrast (np.array): (a,) boolean, True for features of interest
            seed: random seed

        Returns:
            Experiment with sampled x
        """
        assert (a is None) != (contrast is None), 'a xor contrast required'

        if a is None:
            # contrast specified, extract a from it
            a = contrast.size
        else:
            # default contrast: all x of interest but bias term
            contrast = np.ones(a, dtype=bool)

        # sample x
        num_img = self.y.shape[1]
        rng = np.random.default_rng(seed=seed)
        x = rng.standard_normal(size=(a, num_img))

        return Experiment(x=x, contrast=contrast, y=self.y,
                          mask_idx=self.mask_idx,
                          meta=self.meta, **kwargs)

    def apply_mask(self, mask):
        """return a new experiment restricted to voxels where mask is True.

        Args:
            mask (np.array): boolean mask, same shape as self.mask_idx

        Returns:
            new experiment restricted to the intersection of mask and self
        """
        # apply mask to data
        mask = np.logical_and(mask, self.mask_idx > -1)
        assert mask.sum(), 'mask has no intersection with mask_idx'
        mask_idx = glow.mask.get_mask_idx(mask)
        y = self.y[:, :, self.mask_idx[mask]]

        # build new object identical as self
        d = deepcopy(self.__dict__)
        d['mask_idx'] = mask_idx
        d['y'] = y
        return type(self)(**d)

    def add_offset(self, offset, mask=None, vox_idx=None, sigma_scale=None):
        """return a new experiment with a constant offset added to y.

        Args:
            offset (np.array): (b, num_img) offset per voxel
            mask (np.array): boolean region to apply offset (xor vox_idx)
            vox_idx (list): voxel indices to apply offset (xor mask)
            sigma_scale (float): optional sigma stretch factor

        Returns:
            new experiment with offset applied
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
    """imaging data plus design matrix and contrast.

    Attributes:
        x (np.array): (a, num_img) design matrix
        contrast (np.array): (a,) boolean, True for features of interest
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
                          'origin (consider add_bias=True)',
                          NoBiasTermWarning)

    def _hash(self):
        """rolling SHA-256 hash over all data arrays (16-char hex digest)."""
        import hashlib
        h = hashlib.sha256()
        for arr in (self.y, self.mask_idx, self.x, self.contrast):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()[:16]

    def impose_effect(self, effect_llr, extenter=None, mask=None, seed=None,
                      roughness=None, **kwargs):
        """return a new experiment with a synthetic effect imposed.

        Args:
            effect_llr (float): target size-normalized LLR
            extenter: ExtenterSphere or ExtenterMinVar (xor mask)
            mask (np.array): boolean effect region (xor extenter)
            seed: random seed for extent sampling
            roughness (float | None): target Tr(sigma)/Tr(E) in (0,1).
                When set, the spatial covariance within the effect region
                is scaled so the post-imposition roughness matches.

        Returns:
            exp (Experiment): experiment with effect
            effect (Effect): the imposed effect
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

        # get offset (and sigma_scale when roughness is requested)
        offset, sigma_scale = glow.effect.compute_offset(
            x=self.x, y=y, contrast=self.contrast,
            effect_llr=effect_llr, roughness=roughness,
        )

        # impose effect on y, build new experiment
        exp = self.add_offset(offset, mask=mask, sigma_scale=sigma_scale)

        effect = glow.effect.Effect.from_exp_mask(exp=exp,
                                                  mask=mask,
                                                  seed=seed,
                                                  effect_llr=effect_llr)

        return exp, effect

    def permute(self, perm_idx):
        """return a new experiment with Freedman-Lane permuted images.

        Args:
            perm_idx (int): permutation index (0 = unpermuted)

        Returns:
            Experiment with permuted y
        """
        if perm_idx == 0:
            # perm_idx = 0 is reserved for unpermuted data
            y = deepcopy(self.y)
        else:
            freed_lane = get_freed_lane(self.x, self.contrast, perm_idx)
            y = np.einsum('abc,bd->adc', self.y, freed_lane, optimize=True)

        return Experiment(x=self.x, y=y, contrast=self.contrast,
                          mask_idx=self.mask_idx, meta=self.meta)


class ExperimentScaled(Experiment):
    """pre-processed experiment: zero-mean, variance-normalise, then PCA.

    y_out = pre_scale @ (y_in - mean_orig)

    Attributes:
        mean_orig (np.array): (b, 1, 1) original grand mean
        pre_scale (np.array): (b, b) pre-processing matrix
    """

    @classmethod
    def from_exp(cls, exp):
        return cls(y=exp.y, mask_idx=exp.mask_idx, x=exp.x,
                   contrast=exp.contrast, meta=getattr(exp, 'meta', None))

    def prep(self, y):
        """apply pre-processing: y_out = pre_scale @ (y - mean_orig)."""
        return np.einsum('ij,jkl->ikl',
                         self.pre_scale,
                         y - self.mean_orig)

    def prep_inv(self, y):
        """invert pre-processing: y_out = pre_scale^-1 @ y + mean_orig."""
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
        variances = np.diag(cov)
        if np.any(variances == 0):
            zero_feats = np.where(variances == 0)[0]
            raise ValueError(
                f'zero-variance feature(s) at index {zero_feats.tolist()}. '
                f'ExperimentScaled requires all features to have non-zero '
                f'variance. Remove constant features before analysis.'
            )
        self.pre_scale = np.diag(1 / variances ** .5)

        cov_scale = self.pre_scale @ cov @ self.pre_scale.T
        evals, evecs = np.linalg.eigh(cov_scale)
        self.pre_scale = evecs.T @ self.pre_scale

        super().__init__(y=self.prep(y), *args, **kwargs)
