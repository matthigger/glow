"""Experiment classes: imaging data, design matrix, and pre-scaling.

ExperimentImageOnly holds the imaging data (y plus a voxel mask) and the
factories that build it (from Gaussian samples, from a folder search, or
from an explicit path map).  Experiment adds a design matrix x and a
contrast, plus Freedman-Lane permutation.  ExperimentScaled pre-processes
y (zero-mean, variance-normalise, PCA) before analysis.
"""

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
    """Warn when regression is constrained to origin without a bias term."""
    pass


class ConstantVoxelWarning(UserWarning):
    """Warn when voxels leave an experiment for not varying across images."""
    pass


class ExperimentImageOnly:
    """Imaging data for an experiment (no design matrix).

    Attributes:
        y (np.array): (b, num_img, num_vox) image intensities
        mask_idx (np.array): voxel index array (-1 outside analysis)
        mask_dead (np.array): (X, Y, Z) boolean, True where
            drop_constant_vox removed a voxel, or None where it never ran.
            Image-space, like every other mask here (an effect's support,
            an analysis's mask_active), so it stays readable after
            mask_idx renumbers the voxels that remain.
        meta (dict): optional metadata (subjects, features, affine, etc.)
            not used by analysis — propagated for export / display
    """

    def __init__(self, *, y, mask_idx, mask_dead=None, meta: dict = None,
                 dtype=None, **kwargs):
        """Store imaging data, optionally casting y to a target dtype.

        Args:
            y (np.array): (b, num_img, num_vox) imaging features
            mask_idx (np.array): voxel index array (-1 outside analysis)
            mask_dead (np.array): (X, Y, Z) boolean of dropped voxels, or
                None. Named here rather than swept into kwargs so it
                survives _copy_with, which rebuilds through __init__.
            meta (dict): optional metadata
            dtype: if not None and y.dtype differs, cast y to dtype (no
                copy when already matching).  Default None preserves
                y.dtype — used by internal constructors so dtype is set
                exactly once by the public factory at the top of the chain.
        """
        if dtype is not None and y is not None and y.dtype != dtype:
            y = y.astype(dtype, copy=False)
        self.y = y
        self.mask_idx = mask_idx
        self.mask_dead = mask_dead
        self.meta = meta if meta is not None else {}

    @property
    def dtype(self):
        """Return the dtype of the underlying y array, or None when unset."""
        return self.y.dtype if self.y is not None else None

    def _repr_dropped(self) -> str:
        """Render the screened-voxel count, or '' if never screened.

        The recorder stores an opaque output as its repr, so this string
        is how many voxels drop_constant_vox took shows up in the built
        experiment's record -- once, where the drop happened, rather than
        copied onto every leaf that later reads the experiment. Shown
        even at zero, since screened-and-clean is worth telling apart
        from never-screened.
        """
        if self.mask_dead is None:
            return ''
        return f', num_vox_dropped={self.num_vox_dropped}'

    def __repr__(self):
        """A compact identity string: class name + the y dimensions.

        Cheap and human-readable (no array hashing): the (b, num_img,
        num_vox) shape names what the object is in a log line, traceback, or
        recorded DataFrame cell. y is None only for a half-built instance.
        """
        if self.y is None:
            return f'{type(self).__name__}(empty)'
        b, num_img, num_vox = self.y.shape
        return (f'{type(self).__name__}(b={b}, num_img={num_img}, '
                f'num_vox={num_vox}{self._repr_dropped()})')

    @classmethod
    def from_gauss(cls, b: int = None, num_img: int = 10,
                   shape: tuple = (2, 3, 4), seed: int = None,
                   mu=None, cov=None, dtype=np.float32, **kwargs):
        """Generate Gaussian imaging data with prescribed mean and covariance.

        Args:
            b (int): number of imaging features (default 1)
            num_img (int): number of images to sample
            shape (tuple): spatial shape of each image
            seed (int): random seed
            mu (np.array): (b,) target sample mean (default zeros)
            cov (np.array): (b, b) target sample covariance (default identity)
            dtype: numpy dtype for the generated y array.  Default
                np.float32 matches the HCP loader and keeps the
                compute_llr_batched hot loop in float32.

        Returns:
            ExperimentImageOnly with sampled y
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
        meta.setdefault('subjects',
                        [f'subject_{i:03d}' for i in range(num_img)])
        # single cast at the public factory; the constructor would also
        # handle this but doing it explicitly documents the contract.
        y = y.reshape((b, num_img, num_vox)).astype(dtype, copy=False)
        return cls(y=y,
                   mask_idx=get_mask_idx(np.ones(shape)),
                   meta=meta, **kwargs)

    @classmethod
    def _search_files(cls, folder, sbj_regex: str, img_glob_dict: dict):
        """Scan a folder for feature files into a subject x feature table.

        Returns a DataFrame of file paths; no image data is loaded.

        Args:
            folder: root folder to search
            sbj_regex (str): regex extracting the subject id from file paths
            img_glob_dict (dict): feature_name -> glob pattern

        Returns:
            df (pd.DataFrame): index=subject, columns=feature, values=paths
        """
        folder = pathlib.Path(folder)
        df = pd.DataFrame()
        for y_feat, y_glob in img_glob_dict.items():
            for file in folder.glob(y_glob):
                # dedupe before asserting: BIDS-derivatives paths repeat the
                # subject id in both the directory and the filename (e.g.
                # sub-100307/dwi/sub-100307_..._param-fa_dwimap.nii.gz), so a
                # natural regex like sub-\d+ matches more than once. Collapse
                # identical matches to one; still reject a path that yields two
                # genuinely different ids.
                sbj_set = set(re.findall(sbj_regex, str(file)))
                assert len(sbj_set) == 1, \
                    f'unique sbj not found in file: {file}'
                sbj = sbj_set.pop()
                df.loc[sbj, y_feat] = file
        return df

    @classmethod
    def list_subjects(cls, folder, sbj_regex: str,
                      img_glob_dict: dict) -> list:
        """Return the sorted list of subject ids discovered under folder.

        Canonical across regex rewrites that select the same files — useful
        as a stable identity for caching / hashing.

        Args:
            folder: root folder to search
            sbj_regex (str): regex extracting the subject id from file paths
            img_glob_dict (dict): feature_name -> glob pattern

        Returns:
            subjects (list): sorted subject ids
        """
        df = cls._search_files(folder, sbj_regex, img_glob_dict)
        assert df.size, 'no images found'
        return sorted(df.index)

    @classmethod
    def from_search(cls, folder, sbj_regex: str, img_glob_dict: dict,
                    dtype=np.float32, mask=None, **kwargs):
        """Search a folder for images and build an experiment.

        Args:
            folder: root folder to search recursively
            sbj_regex (str): regex extracting the subject id from file paths
            img_glob_dict (dict): feature_name -> glob pattern
            dtype: numpy dtype for the loaded y array (default np.float32;
                see from_paths)
            mask (path): optional brain-mask NIfTI defining the analysis
                support (NIfTI inputs only); see from_paths

        Returns:
            Experiment built from discovered images
        """
        df = cls._search_files(folder, sbj_regex, img_glob_dict)
        return cls.from_paths(df, dtype=dtype, mask=mask, **kwargs)

    @classmethod
    def from_paths(cls, paths, *, channel_names: dict = None,
                   dtype=np.float32, mask=None, **kwargs):
        """Build an experiment from an explicit (subject x feature) path map.

        Args:
            paths: either a pandas.DataFrame indexed by subject with
                feature columns whose values are file paths, or a dict of
                the form {subject: {feature: path}}
            channel_names (dict): optional {feature: [name0, ...]} overriding
                the default feat0/feat1/... naming when a non-NIfTI image
                splits into multiple channels (e.g. RGB ->
                {'rgb': ['red', 'green', 'blue']})
            dtype: numpy dtype for the loaded y array (default np.float32)
            mask (path): optional path to a brain-mask NIfTI on the images'
                grid (NIfTI inputs only).  When given, its nonzero voxels
                are the analysis support; when None the support is inferred
                as the voxels nonzero in every image (see load_image_nii).

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

        # warn rather than fail: a subject missing a feature is often a
        # recoverable data-collection gap the caller wants to see, not stop on.
        s_missing = df.isna().mean(axis=1)
        if s_missing.any():
            missing = df.index[s_missing > 0].tolist()
            warnings.warn(f'some subjects missing imaging files: {missing}')

        nii_in_file = ['.nii' in str(file) for file in df.values.flatten()]
        affine = None
        subjects = sorted(df.index)
        if all(nii_in_file):
            # NIfTI path streams to y directly (no per-image dict held in
            # memory).  The loader controls dtype; we pass the public
            # factory's choice (float32 by default).
            y, y_names, mask_idx, affine = load_image_nii(
                df, dtype=dtype, mask=mask)
        elif not any(nii_in_file):
            assert mask is None, 'explicit mask only supported for NIfTI'
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

        return cls(y=y, mask_idx=mask_idx, meta=meta, **kwargs)

    def bootstrap_img(self, n: int, seed: int = None,
                      noise_scale: float = 0):
        """Return a new experiment with bootstrap-resampled images.

        Args:
            n (int): number of images in the output
            seed (int): random seed
            noise_scale (float): std-dev multiplier for additive noise
                (drawn per-voxel from the sample covariance)

        Returns:
            exp: new experiment over n draws of this one's images. A
                draw takes the image whole, so every per-image attribute
                follows it -- the design matrix column on an Experiment,
                the subject label in meta -- repeats and all.
        """
        rng = np.random.default_rng(seed=seed)
        b, num_img_init, num_vox = self.y.shape
        img_idx = rng.choice(num_img_init, n, replace=True)
        exp = self._take_img(img_idx)

        assert noise_scale >= 0, 'snr cannot be negative'
        if noise_scale > 0:
            cov = np.atleast_2d(np.cov(exp.y.reshape((b, -1))))
            noise = rng.multivariate_normal(mean=np.zeros(b),
                                            cov=cov * (noise_scale ** 2),
                                            size=num_vox * n)
            # multivariate_normal returns float64 unconditionally; cast
            # back so the addition doesn't silently promote y.
            noise = noise.T.reshape((b, n, num_vox)).astype(exp.y.dtype,
                                                            copy=False)
            exp.y = exp.y + noise

        return exp

    def _take_img(self, img_idx, **overrides):
        """Return a new experiment over the images img_idx selects.

        Slices every per-image attribute; subclasses cooperate by adding
        their own to overrides (Experiment adds the design matrix).

        Args:
            img_idx (np.array): image indices to keep, in output order
            **overrides: further attributes for _copy_with

        Returns:
            exp: a new experiment over those images, same voxels
        """
        meta = deepcopy(self.meta)
        subjects = meta.get('subjects')
        if subjects is not None and len(subjects) == self.y.shape[1]:
            meta['subjects'] = [subjects[i] for i in img_idx]

        return self._copy_with(y=self.y[:, img_idx, :], meta=meta,
                               **overrides)

    def get_img_segment(self, frac_segment: float = .5, *, seed: int = None,
                        group=None):
        """Return which images the segmentation fold takes.

        The partition split_img cuts, as a mask instead of a pair of
        experiments, for a caller that needs to know which images built
        something rather than the fold itself (the viewer's regression
        panel, labelling each image by fold).

        The partition is a uniform random draw, so each fold's share of
        the design's information is right on average (the two folds'
        second-moment matrices sum to the whole design's, whatever the
        draw). It is not balanced against an unlucky draw; the split
        depends on the images alone, never on y.

        Args:
            frac_segment (float): fraction of the images going to the
                segmentation fold, in (0, 1)
            seed (int): RNG seed for the partition
            group (np.array): (num_img,) labels held together, so every
                image sharing a label lands in the same fold. For related
                subjects (siblings, repeat scans), whose images correlate
                and would otherwise leave the folds dependent. The
                realized fraction then only approximates frac_segment,
                since whole groups move at a time.

        Returns:
            in_segment (np.array): (num_img,) boolean, True for the images
                in the segmentation fold

        Raises:
            ValueError: if either fold would come out empty
        """
        num_img = self.y.shape[1]
        assert 0 < frac_segment < 1, 'frac_segment must lie in (0, 1)'

        # with no group argument each image is its own group, which walks
        # the same path below and lands on n_segment exactly
        group = np.arange(num_img) if group is None else np.asarray(group)
        assert group.shape == (num_img,), \
            f'group needs one label per image (num_img={num_img})'
        label, group_idx = np.unique(group, return_inverse=True)

        rng = np.random.default_rng(seed=seed)
        order = rng.permutation(label.size)

        # cut the shuffled groups wherever the running image count comes
        # closest to the target, so a group is never broken across folds
        n_cum = np.concatenate([[0], np.cumsum(np.bincount(group_idx)[order])])
        n_segment = int(round(frac_segment * num_img))
        k = int(np.argmin(np.abs(n_cum - n_segment)))

        if not 0 < n_cum[k] < num_img:
            raise ValueError(
                f'split leaves a fold empty ({n_cum[k]} of {num_img} images '
                f'in the segmentation fold): frac_segment={frac_segment} is '
                f'too extreme for num_img={num_img}, or one group is too '
                f'large a share of the images')

        return np.isin(group_idx, order[:k])

    def split_img(self, frac_segment: float = .5, *, seed: int = None,
                  group=None):
        """Partition the images into a segmentation fold and a test fold.

        The two folds are disjoint in images and identical in voxels --
        both keep this experiment's mask_idx and num_vox -- so a Ward tree
        built on exp_segment indexes the leaves of exp_test unchanged.
        That is what lets GLOW segment on one fold and compute LLR / inner
        perms / FWER on the other, which removes the selection bias of
        choosing the tree with the same images that then test it.

        Run drop_constant_vox before splitting, not after: screening each
        fold separately renumbers mask_idx differently in each, and the
        tree stops transferring.

        Args:
            frac_segment (float): fraction of the images going to the
                segmentation fold, in (0, 1)
            seed (int): RNG seed for the partition
            group (np.array): (num_img,) labels held together; see
                get_img_segment, which draws the partition

        Returns:
            exp_segment: fold the Ward tree is built on
            exp_test: fold the statistics are computed on

        Raises:
            ValueError: if either fold would come out empty
        """
        in_segment = self.get_img_segment(frac_segment=frac_segment, seed=seed,
                                          group=group)
        img_idx = np.arange(self.y.shape[1])
        return (self._take_img(img_idx[in_segment]),
                self._take_img(img_idx[~in_segment]))

    def sample_x(self, a: int = None, contrast=None, seed: int = None,
                 **kwargs):
        """Return a new Experiment with random standard-normal design matrix.

        Args:
            a (int): number of features (exactly one of a / contrast required)
            contrast (np.array): (a,) boolean, True for features of interest
            seed (int): random seed

        Returns:
            Experiment with sampled x
        """
        assert (a is None) != (contrast is None), 'a xor contrast required'

        if a is None:
            a = contrast.size
        else:
            # default contrast: all x of interest but bias term
            contrast = np.ones(a, dtype=bool)

        # match y's dtype so downstream decompose() / einsums don't
        # silently upcast (numpy promotes float32 @ float64 to float64,
        # eliminating the bandwidth win in compute_llr_batched).
        num_img = self.y.shape[1]
        rng = np.random.default_rng(seed=seed)
        x = rng.standard_normal(size=(a, num_img))
        if self.y is not None and self.y.dtype != x.dtype:
            x = x.astype(self.y.dtype, copy=False)

        return Experiment(x=x, contrast=contrast, y=self.y,
                          mask_idx=self.mask_idx, mask_dead=self.mask_dead,
                          meta=self.meta, **kwargs)

    def _copy_with(self, **overrides):
        """Return a deep copy of this experiment with attributes overridden.

        Clones every instance attribute and feeds them back to
        type(self)(**d), so callers that derive a new experiment by
        swapping one or two fields (y, mask_idx) don't repeat the
        deepcopy-and-reconstruct dance.
        """
        d = deepcopy(self.__dict__)
        d.update(overrides)
        return type(self)(**d)

    def apply_mask(self, mask):
        """Return a new experiment restricted to voxels where mask is True.

        Args:
            mask (np.array): boolean mask, same shape as self.mask_idx

        Returns:
            new experiment restricted to the intersection of mask and self
        """
        mask = np.logical_and(mask, self.mask_idx > -1)
        assert mask.sum(), 'mask has no intersection with mask_idx'
        mask_idx = glow.mask.get_mask_idx(mask)
        y = self.y[:, :, self.mask_idx[mask]]

        return self._copy_with(mask_idx=mask_idx, y=y)

    @property
    def num_vox_dropped(self) -> int:
        """How many voxels drop_constant_vox removed (0 if it never ran)."""
        return 0 if self.mask_dead is None else int(self.mask_dead.sum())

    def drop_constant_vox(self, rtol: float = 1e-6):
        """Return a new experiment with the constant voxels removed.

        A voxel whose intensities barely move across images has no signal
        to test: once the design is projected out almost nothing is left,
        so its MANCOVA error matrix E is rank-deficient in all but name
        and det(E), a product of b tiny eigenvalues, underflows. In
        float32 the cliff is sharp -- below roughly 1e-8 relative
        variation slogdet(E) returns (0, -inf) and every statistic on
        that voxel comes back NaN, or +-inf where the cancellation also
        leaves E with a negative eigenvalue.

        This belongs to pre-processing, ahead of ExperimentScaled, for
        two reasons. The test is per feature and ExperimentScaled mixes
        the features (y_out = pre_scale @ y), so by the time an analysis
        sees the data one flat feature has been smeared over all b and no
        longer stands out. And dropping here hands every method the same
        voxels, so GLOW and the voxel-wise arms control FWER over one
        family rather than each pruning its own.

        The dropped voxels leave y and mask_idx is renumbered over what
        remains, so nothing downstream needs to know this happened;
        mask_dead records which ones went.

        Args:
            rtol (float): variation floor, as a fraction of each feature's
                median across-image standard deviation over voxels.
                Median, so the dead voxels do not set the scale they are
                then measured against.

        Returns:
            exp: a new experiment over the surviving voxels, carrying
                mask_dead and leaving this one untouched

        Raises:
            ValueError: if every voxel is constant across images
        """
        # float64 accumulator: the sums that underflow in float32 are the
        # ones this is here to catch
        std = self.y.std(axis=1, dtype=np.float64)
        vox_dead = np.any(std <= rtol * np.median(std, axis=1, keepdims=True),
                          axis=0)
        if vox_dead.all():
            raise ValueError('every voxel is constant across images')

        # scatter the per-voxel flags back into image space. Indexed
        # through mask_idx rather than assigned positionally, so this
        # holds whatever order the index array numbers its voxels in.
        mask_active = self.mask_idx > -1
        mask_dead = np.zeros(self.mask_idx.shape, dtype=bool)
        mask_dead[mask_active] = vox_dead[self.mask_idx[mask_active]]

        if not vox_dead.any():
            return self._copy_with(mask_dead=mask_dead)

        warnings.warn(f'dropped {int(vox_dead.sum())} of {vox_dead.size} '
                      f'voxels constant across images (rtol={rtol:g})',
                      ConstantVoxelWarning)
        exp = self.apply_mask(~mask_dead)
        exp.mask_dead = mask_dead
        return exp

    def add_offset(self, offset, mask=None, vox_idx=None,
                   sigma_scale: float = None):
        """Return a new experiment with a constant offset added to y.

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

        y = deepcopy(self.y)
        y[:, :, vox_idx] += offset[..., np.newaxis]

        if sigma_scale is not None:
            y[:, :, vox_idx] = stretch_sigma(y=y[:, :, vox_idx],
                                             scale=sigma_scale)

        return self._copy_with(y=y)


class Experiment(ExperimentImageOnly):
    """Imaging data plus design matrix and contrast.

    Attributes:
        y (np.array): (b, num_img, num_vox) image intensities
        mask_idx (np.array): voxel index array (-1 outside analysis)
        x (np.array): (a, num_img) design matrix
        contrast (np.array): (a,) boolean, True for features of interest
    """

    @classmethod
    def from_gauss(cls, a: int = 1, contrast=None, seed: int = None,
                   add_bias: bool = True, **kwargs):
        """Generate Gaussian imaging data and a sampled design matrix.

        Args:
            a (int): number of design-matrix features
            contrast (np.array): (a,) boolean, True for features of interest
            seed (int): random seed (shared by y and x sampling)
            add_bias (bool): prepend a bias (all-ones) row to x

        Returns:
            Experiment with sampled y and x
        """
        exp = ExperimentImageOnly.from_gauss(seed=seed, **kwargs)
        return exp.sample_x(a=a, contrast=contrast, seed=seed,
                            add_bias=add_bias)

    def __init__(self, *, x, contrast=None, add_bias: bool = False,
                 **kwargs):
        """Store the design matrix and contrast, optionally adding a bias row.

        Args:
            x (np.array): (a, num_img) design matrix
            contrast (np.array): (a,) boolean, True for features of interest
            add_bias (bool): prepend a bias (all-ones) row to x and a
                leading False to contrast

        Raises:
            NoBiasTermWarning: if x has no all-ones row (regression
                constrained to origin)
        """
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

    def __repr__(self):
        """Extend the image-only repr with the design width a (== x rows)."""
        a = self.x.shape[0]
        if self.y is None:
            return f'{type(self).__name__}(a={a})'
        b, num_img, num_vox = self.y.shape
        return (f'{type(self).__name__}(b={b}, num_img={num_img}, '
                f'num_vox={num_vox}, a={a}{self._repr_dropped()})')

    def _take_img(self, img_idx, **overrides):
        """Extend the image subset to the design matrix's columns."""
        overrides.setdefault('x', self.x[:, img_idx])
        return super()._take_img(img_idx, **overrides)

    def _assert_design(self, fold: str, seed: int):
        """Raise unless this fold's design still supports a MANCOVA fit.

        A fold whose x lost rank (every subject at one level of a rare
        covariate went to the other fold) gives mancova.decompose a
        degenerate q0 / q1 and fails silently rather than loudly, so
        check it here where the seed that produced it is still in hand.

        Args:
            fold (str): fold name, for the message
            seed (int): the split's seed, for the message

        Raises:
            ValueError: if x is rank-deficient or leaves no residual space
        """
        a, num_img = self.x.shape
        detail = (f'{fold} fold of a split_img(seed={seed}); pass group= to '
                  f'hold related images together, or try another seed')

        if num_img <= a:
            raise ValueError(f'design has no residual space: num_img='
                             f'{num_img} <= a={a} in the {detail}')
        if np.linalg.matrix_rank(self.x) < a:
            raise ValueError(f'design lost rank: rank(x) < a={a} in the '
                             f'{detail}')

    def split_img(self, frac_segment: float = .5, *, seed: int = None,
                  group=None):
        """Split the images, checking both folds keep a usable design.

        See ExperimentImageOnly.split_img; this adds the design matrix to
        what each fold carries, and the check that each fold's x survived
        the partition.

        Raises:
            ValueError: if either fold's design is rank-deficient or
                leaves no residual space
        """
        exp_segment, exp_test = super().split_img(
            frac_segment=frac_segment, seed=seed, group=group)

        exp_segment._assert_design('segmentation', seed)
        exp_test._assert_design('test', seed)
        return exp_segment, exp_test

    def permute(self, perm_idx: int):
        """Return a new experiment with Freedman-Lane permuted images.

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

        return Experiment(x=self.x, y=y, contrast=self.contrast,
                          mask_idx=self.mask_idx, mask_dead=self.mask_dead,
                          meta=deepcopy(self.meta))


class ExperimentScaled(Experiment):
    """Pre-processed experiment: zero-mean, variance-normalise, then PCA.

    y_out = pre_scale @ (y_in - mean_orig)

    Attributes:
        y (np.array): (b, num_img, num_vox) pre-processed image intensities
        mask_idx (np.array): voxel index array (-1 outside analysis)
        x (np.array): (a, num_img) design matrix
        contrast (np.array): (a,) boolean, True for features of interest
        mean_orig (np.array): (b, 1, 1) original grand mean
        pre_scale (np.array): (b, b) pre-processing matrix
    """

    @classmethod
    def from_exp(cls, exp):
        """Build an ExperimentScaled from an existing Experiment.

        Idempotent: an exp that is already an ExperimentScaled is returned
        unchanged (never re-scaled), so each Analysis.fit can pass whatever
        it was handed -- raw or already-scaled -- through this one call.

        Args:
            exp: source Experiment (provides y, mask_idx, x, contrast, meta)

        Returns:
            ExperimentScaled with pre-processing applied to exp.y, or exp
            itself when it is already an ExperimentScaled
        """
        if isinstance(exp, cls):
            return exp
        return cls(y=exp.y, mask_idx=exp.mask_idx, x=exp.x,
                   contrast=exp.contrast,
                   mask_dead=getattr(exp, 'mask_dead', None),
                   meta=dict(exp.meta) if getattr(exp, 'meta', None) else None)

    def split_img(self, *args, **kwargs):
        """Refuse the split: pre-scaling was fit on every image.

        pre_scale and mean_orig come from all the images at once, so a
        fold cut out afterwards carries a transform the other fold helped
        choose -- a leak, small but free to avoid. Split the raw
        Experiment instead; AnalysisGLOW.fit runs from_exp on whatever it
        is handed, so each fold gets its own transform.

        Raises:
            TypeError: always.
        """
        raise TypeError('split before scaling: ExperimentScaled fits '
                        'pre_scale on every image, so both folds of a later '
                        'split share a transform the test fold helped '
                        'choose. Split the Experiment, then scale each fold.')

    def prep(self, y):
        """Apply pre-processing: y_out = pre_scale @ (y - mean_orig).

        Args:
            y (np.array): (b, num_img, num_vox) raw image intensities

        Returns:
            y (np.array): (b, num_img, num_vox) pre-processed intensities
        """
        return np.einsum('ij,jkl->ikl',
                         self.pre_scale,
                         y - self.mean_orig)

    def prep_inv(self, y):
        """Invert pre-processing: y_out = pre_scale^-1 @ y + mean_orig.

        Args:
            y (np.array): (b, num_img, num_vox) pre-processed intensities

        Returns:
            y (np.array): (b, num_img, num_vox) raw image intensities
        """
        return np.einsum('ij,jkl->ikl',
                         np.linalg.inv(self.pre_scale),
                         y) + self.mean_orig

    def __init__(self, y, **kwargs):
        """Fit the pre-processing transform on y, then store the scaled y.

        Args:
            y (np.array): (b, num_img, num_vox) raw image intensities

        Raises:
            ValueError: if any feature has zero variance
        """
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

        super().__init__(y=self.prep(y), **kwargs)
