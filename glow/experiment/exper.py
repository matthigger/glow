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
from .regen import (REGEN_REGISTRY, STEP_REPLAY, FullPickleNotice,
                    compute_pickle_status)
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

    def __init__(self, *, y, mask_idx, meta=None, dtype=None, **kwargs):
        """
        Args:
            y (np.array): (b, num_img, num_vox) imaging features.
            mask_idx (np.array): voxel index array (-1 outside analysis).
            meta (dict): optional metadata.
            dtype (np.dtype | None): if not None and ``y.dtype`` differs,
                cast ``y`` to ``dtype`` (no copy when already matching).
                Default ``None`` preserves ``y.dtype`` — used by internal
                constructors and recipe-replay paths so dtype is set
                exactly once by the public factory at top of the chain.
        """
        if dtype is not None and y is not None and y.dtype != dtype:
            y = y.astype(dtype, copy=False)
        self.y = y
        self.mask_idx = mask_idx
        self.meta = meta if meta is not None else {}

    @property
    def dtype(self):
        """dtype of the underlying ``y`` array, or None when y is slim."""
        return self.y.dtype if self.y is not None else None

    def _full_pickle_msg(self, src):
        y_mb = self.y.nbytes / 1024**2 if self.y is not None else 0.0
        if src is None:
            head = 'Pickling Experiment without recipe tagged.'
        else:
            head = (f'Pickling Experiment with source={src!r} '
                    f'(not registered).')
        return (f'{head} Storing y ({y_mb:.1f} MB) inline. '
                f'For slim pickles, see glow.experiment.regen.')

    def __getstate__(self):
        # x is always preserved (small; no general regen interface for
        # user-supplied designs).  Only y is droppable on slim pickle.
        state = self.__dict__.copy()
        if state.get('y') is None:
            return state
        recipe = self.meta.get('recipe') if self.meta else None
        src = recipe.get('source') if recipe else None
        if src and src in REGEN_REGISTRY:
            state['y'] = None
            return state
        msg = self._full_pickle_msg(src)
        warnings.warn(msg, FullPickleNotice, stacklevel=2)
        return state

    def rehydrate(self):
        """reconstruct ``y`` from the recipe.

        Builds the base experiment from ``recipe['source']`` then replays
        each step in ``recipe['steps']``.  No-op if ``y`` is already
        loaded.  ``x`` is preserved through pickling and is not modified
        here.  Raises ``RuntimeError`` if the recipe source or any step
        op is unregistered.
        """
        if self.y is not None:
            return
        recipe = self.meta.get('recipe') if self.meta else None
        src = recipe.get('source') if recipe else None
        if not src or src not in REGEN_REGISTRY:
            raise RuntimeError(
                f'cannot rehydrate Experiment: source={src!r} not in '
                f'REGEN_REGISTRY')
        base = REGEN_REGISTRY[src](**recipe['args'])
        for step in recipe.get('steps', []):
            op = step['op']
            if op not in STEP_REPLAY:
                raise RuntimeError(
                    f'cannot replay step: op={op!r} not in STEP_REPLAY')
            base = STEP_REPLAY[op](base, step['args'])
        self.y = base.y

    def pickle_status(self):
        return compute_pickle_status(self)

    def _hash(self):
        """rolling SHA-256 hash over data arrays (16-char hex digest)."""
        import hashlib
        h = hashlib.sha256()
        for arr in (self.y, self.mask_idx):
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()[:16]

    @classmethod
    def from_gauss(cls, b=None, num_img=10, shape=(2, 3, 4), seed=None,
                   mu=None, cov=None, dtype=np.float32, **kwargs):
        """generate Gaussian imaging data with prescribed mean and covariance.

        Args:
            b (int): number of imaging features (default 1)
            num_img (int): number of images to sample
            shape (tuple): spatial shape of each image
            seed (int): random seed
            mu (np.array): target sample mean (default zeros)
            cov (np.array): target sample covariance (default identity)
            dtype: numpy dtype for the generated ``y`` array.  Default
                ``np.float32`` matches the HCP loader and keeps the
                ``compute_llr_batched`` hot loop in float32.

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
        meta['recipe'] = {
            'source': 'gauss',
            'args': {
                'b': b,
                'num_img': num_img,
                'shape': tuple(shape),
                'seed': seed,
                'mu': mu,
                'cov': cov,
                'dtype': dtype,
            },
        }
        # cast to target dtype here (single cast at the public factory;
        # the constructor would also handle this but doing it explicitly
        # documents the contract).
        y = y.reshape((b, num_img, num_vox)).astype(dtype, copy=False)
        return cls(y=y,
                   mask_idx=get_mask_idx(np.ones(shape)),
                   meta=meta, **kwargs)

    @classmethod
    def _search_files(cls, folder, sbj_regex, img_glob_dict):
        """scan folder for feature files and return a (subject x feature)
        DataFrame of file paths.  No image data is loaded."""
        folder = pathlib.Path(folder)
        df = pd.DataFrame()
        for y_feat, y_glob in img_glob_dict.items():
            for file in folder.glob(y_glob):
                sbj_list = re.findall(sbj_regex, str(file))
                assert len(sbj_list) == 1, \
                    f'unique sbj not found in file: {file}'
                sbj = sbj_list[0]
                df.loc[sbj, y_feat] = file
        return df

    @classmethod
    def list_subjects(cls, folder, sbj_regex, img_glob_dict):
        """return the sorted list of subject ids discovered under folder.

        Canonical across regex rewrites that select the same files — useful
        as a stable identity for caching / hashing."""
        df = cls._search_files(folder, sbj_regex, img_glob_dict)
        assert df.size, 'no images found'
        return sorted(df.index)

    @classmethod
    def from_search(cls, folder, sbj_regex, img_glob_dict,
                    dtype=np.float32, **kwargs):
        """search a folder for images and build an experiment.

        Args:
            folder (str): root folder to search recursively
            sbj_regex (str): regex extracting the subject id from file paths
            img_glob_dict (dict): feature_name -> glob pattern
            dtype: numpy dtype for the loaded ``y`` array (default
                ``np.float32``; see ``from_paths``).

        Returns:
            Experiment built from discovered images
        """
        df = cls._search_files(folder, sbj_regex, img_glob_dict)
        return cls.from_paths(df, dtype=dtype, **kwargs)

    @classmethod
    def from_paths(cls, paths, *, channel_names=None,
                   dtype=np.float32, **kwargs):
        """build an experiment from an explicit (subject x feature) path map.

        Args:
            paths: either a ``pandas.DataFrame`` indexed by subject with
                feature columns whose values are file paths, or a
                ``dict`` of the form ``{subject: {feature: path}}``.
            channel_names: optional ``{feature: [name0, ...]}`` overriding
                the default ``feat0``/``feat1``/... naming when a non-NIfTI
                image splits into multiple channels (e.g. RGB →
                ``{'rgb': ['red', 'green', 'blue']}``).

        Returns:
            Experiment built from the listed images
        """
        if isinstance(paths, dict):
            df = pd.DataFrame.from_dict(paths, orient='index')
        else:
            df = paths

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
        subjects = sorted(df.index)
        if all(nii_in_file):
            # NIfTI path streams to y directly (no per-image dict held in
            # memory).  The loader controls dtype; we pass the public
            # factory's choice (float32 by default).
            y, y_names, mask_idx, affine = load_image_nii(df, dtype=dtype)
        elif not any(nii_in_file):
            feat_sbj_img, mask_idx = load_image_color(
                df, channel_names=channel_names)

            # ensure all input has same data type (PNG/JPG dtype is
            # whatever PIL reads — typically uint8; cast happens below)
            src_dtype = None
            for _, sbj_img in feat_sbj_img.items():
                for _, img in sbj_img.items():
                    if src_dtype is None:
                        src_dtype = img.dtype
                    else:
                        assert src_dtype == img.dtype, 'dtype mismatch'

            # mask into each image, store as y.  Preserve feature insertion
            # order so channel_names (e.g. RGB → red, green, blue) keep the
            # caller's intended order rather than alphabetical.  Allocate
            # at the requested dtype so the per-image copy casts in place.
            mask = mask_idx >= 0
            y_names = list(feat_sbj_img.keys())
            y = np.empty((len(feat_sbj_img), df.shape[0], mask.sum()),
                         dtype=dtype)
            for sbj_idx, sbj in enumerate(subjects):
                for feat_idx, feat in enumerate(y_names):
                    y[feat_idx, sbj_idx, :] = feat_sbj_img[feat][sbj][mask]
        else:
            raise TypeError('may not mix nifti and color images in input')

        meta = kwargs.pop('meta', {})
        meta.setdefault('subjects', [str(s) for s in subjects])
        meta.setdefault('features', list(y_names))
        if affine is not None:
            meta.setdefault('affine', affine)

        # serializable form of the path table for recipe regen
        recipe_paths = {str(sbj): {str(feat): str(p)
                                    for feat, p in row.items()
                                    if isinstance(p, (str, pathlib.Path))}
                        for sbj, row in df.iterrows()}
        recipe_args = {'paths': recipe_paths}
        if channel_names:
            recipe_args['channel_names'] = channel_names
        meta['recipe'] = {
            'source': 'image_paths',
            'args': recipe_args,
        }

        return cls(y=y, mask_idx=mask_idx, meta=meta, **kwargs)

    def _propagate_recipe(self, recipe_step):
        """Return a fresh meta dict for a derived experiment.

        - If ``recipe_step`` is given and the current recipe exists, append
          the step to ``meta['recipe']['steps']``.
        - If ``recipe_step`` is None and a recipe exists, drop it (legacy
          fallback so ``__getstate__`` does a full pickle).
        - Otherwise pass meta through unchanged.
        """
        meta = dict(self.meta) if self.meta else {}
        recipe = meta.get('recipe')
        if recipe_step is not None and recipe is not None:
            new_recipe = dict(recipe)
            new_recipe['steps'] = list(recipe.get('steps', [])) + [recipe_step]
            meta['recipe'] = new_recipe
        elif recipe_step is None and recipe is not None:
            meta = {k: v for k, v in meta.items() if k != 'recipe'}
        return meta

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
            # multivariate_normal returns float64 unconditionally; cast
            # back so the addition doesn't silently promote y.
            noise = noise.T.reshape((b, n, num_vox)).astype(y.dtype,
                                                            copy=False)
            y = y + noise

        # bootstrap_img is fully described by its scalar args (n, seed,
        # noise_scale) plus the upstream recipe — self-recipe so replay
        # reproduces y exactly.
        step = {'op': 'bootstrap_img',
                'args': {'n': n, 'seed': seed, 'noise_scale': noise_scale}}
        new_meta = self._propagate_recipe(step)

        exp = deepcopy(self)
        exp.y = y
        exp.meta = new_meta
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

        # sample x — match y's dtype so downstream decompose() / einsums
        # don't silently upcast (numpy promotes float32 @ float64 to
        # float64, eliminating the bandwidth win in compute_llr_batched).
        num_img = self.y.shape[1]
        rng = np.random.default_rng(seed=seed)
        x = rng.standard_normal(size=(a, num_img))
        if self.y is not None and self.y.dtype != x.dtype:
            x = x.astype(self.y.dtype, copy=False)

        return Experiment(x=x, contrast=contrast, y=self.y,
                          mask_idx=self.mask_idx,
                          meta=self.meta, **kwargs)

    def apply_mask(self, mask, *, recipe_step=None):
        """return a new experiment restricted to voxels where mask is True.

        Args:
            mask (np.array): boolean mask, same shape as self.mask_idx
            recipe_step (dict | None): if given, append this step to the
                propagated recipe instead of clearing it.

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
        d['meta'] = self._propagate_recipe(recipe_step)
        return type(self)(**d)

    def add_offset(self, offset, mask=None, vox_idx=None, sigma_scale=None,
                   *, recipe_step=None):
        """return a new experiment with a constant offset added to y.

        Args:
            offset (np.array): (b, num_img) offset per voxel
            mask (np.array): boolean region to apply offset (xor vox_idx)
            vox_idx (list): voxel indices to apply offset (xor mask)
            sigma_scale (float): optional sigma stretch factor
            recipe_step (dict | None): if given, append this step to the
                propagated recipe instead of clearing it.

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
        d['meta'] = self._propagate_recipe(recipe_step)
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
        exp = exp.sample_x(a=a, contrast=contrast, seed=seed,
                           add_bias=add_bias)
        # overwrite the inner image-only recipe with one that includes
        # x-sampling kwargs so a single regen reproduces both y and x.
        if exp.meta is not None and 'recipe' in exp.meta:
            inner_args = dict(exp.meta['recipe']['args'])
            inner_args['a'] = a
            inner_args['contrast'] = contrast
            inner_args['add_bias'] = add_bias
            exp.meta['recipe'] = {'source': 'gauss', 'args': inner_args}
        return exp

    def __init__(self, *, x, contrast=None, add_bias=False, **kwargs):
        super().__init__(**kwargs)

        self.x = x
        self.contrast = contrast

        if add_bias:
            # append row of ones (bias term) to x.  np.ones defaults to
            # float64, which would silently promote x if it's float32 —
            # match x's dtype to keep the design matrix in the same
            # precision as y (decompose() propagates x.dtype through to
            # the q matrices used in compute_llr_batched's einsums).
            num_img = x.shape[1]
            self.x = np.vstack([np.ones(num_img, dtype=x.dtype), x])

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
            # get_freed_lane uses np.eye / rng.permutation which return
            # float64 — cast to y.dtype so the einsum preserves dtype.
            # This is the inner loop of AnalysisGLOW: every permutation
            # goes through here, and a silent f32 -> f64 promotion would
            # erase the memory + speed gains of float32 y.
            if freed_lane.dtype != self.y.dtype:
                freed_lane = freed_lane.astype(self.y.dtype, copy=False)
            y = np.einsum('abc,bd->adc', self.y, freed_lane, optimize=True)

        # permute is fully described by perm_idx + the upstream recipe;
        # self-recipe so replay reproduces y exactly.
        step = {'op': 'permute', 'args': {'perm_idx': perm_idx}}
        meta = self._propagate_recipe(step)

        return Experiment(x=self.x, y=y, contrast=self.contrast,
                          mask_idx=self.mask_idx, meta=meta)


class ExperimentScaled(Experiment):
    """pre-processed experiment: zero-mean, variance-normalise, then PCA.

    y_out = pre_scale @ (y_in - mean_orig)

    Attributes:
        mean_orig (np.array): (b, 1, 1) original grand mean
        pre_scale (np.array): (b, b) pre-processing matrix
    """

    @classmethod
    def from_exp(cls, exp):
        new = cls(y=exp.y, mask_idx=exp.mask_idx, x=exp.x,
                  contrast=exp.contrast,
                  meta=dict(exp.meta) if getattr(exp, 'meta', None) else None)
        # Record a 'scale' step in the recipe so a slim rehydrate replays
        # the exact same prep transform (mean_orig + pre_scale survive
        # the slim pickle as ndarray attributes).
        if new.meta and new.meta.get('recipe') is not None:
            recipe = dict(new.meta['recipe'])
            recipe['steps'] = list(recipe.get('steps', [])) + [
                {'op': 'scale',
                 'args': {'mean_orig': new.mean_orig,
                          'pre_scale': new.pre_scale}},
            ]
            new.meta['recipe'] = recipe
        return new

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
        # Preserve y.dtype through the prep transform.  np.cov / eigh /
        # mean(axis=...) all use float64 accumulators internally and
        # return float64 regardless of input dtype, so cast back at the
        # end — otherwise self.prep(y) silently promotes y to float64.
        y_dtype = y.dtype

        # zero mean (and make new copy).  Use dtype=float64 accumulator
        # for numerical stability (b is small, no memory cost), then
        # cast the (b,) result back to match y.
        self.mean_orig = y.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]
        self.mean_orig = self.mean_orig.astype(y_dtype, copy=False)

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
        self.pre_scale = (evecs.T @ self.pre_scale).astype(y_dtype, copy=False)

        super().__init__(y=self.prep(y), *args, **kwargs)
