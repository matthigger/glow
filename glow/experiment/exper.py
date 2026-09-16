"""Experiment classes: imaging data, design matrix, and pre-scaling.

ExperimentImageOnly holds the imaging data (y plus a voxel mask) and the
factories that build it (from Gaussian samples, from a folder search, or
from an explicit path map).  Experiment adds a design matrix x and a
contrast, plus Freedman-Lane permutation.  ExperimentScaled pre-processes
y (zero-mean, variance-normalise, PCA) before analysis.

Each factory also attaches the source its images came from
(glow.experiment.source), so an experiment cropped down to its analysis
support can read voxels outside it again.
"""

import pathlib
import re
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.ndimage

import glow.effect
import glow.mask
from .load_image import load_image_color, load_image_nii
from .permute import get_freed_lane
from .sigma import stretch_sigma
from .smooth import halo_vox_needed, smooth_y
from .source import SourceGauss, SourceNifti
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
        source (ImageSource | None): where the images came from, able to
            read voxels this experiment no longer holds
            (glow.experiment.source). None where they cannot be read
            again, either because the factory attaches no source or
            because an operation since then cannot be replayed onto
            freshly loaded voxels.
        patch_list (list): what has been added to y since it was loaded,
            as records {offset (b, num_img), mask (X, Y, Z) bool}, one per
            add_offset call. inflate replays them onto the voxels it
            loads, so a planted experiment can still grow. Offsets are in
            y's own space, which for an ExperimentScaled means scaled.
        meta (dict): optional metadata (subjects, features, affine, etc.)
            not used by analysis — propagated for export / display
    """

    def __init__(self, *, y, mask_idx, mask_dead=None, source=None,
                 patch_list: list = None, meta: dict = None, dtype=None,
                 **kwargs):
        """Store imaging data, optionally casting y to a target dtype.

        Args:
            y (np.array): (b, num_img, num_vox) imaging features
            mask_idx (np.array): voxel index array (-1 outside analysis)
            mask_dead (np.array): (X, Y, Z) boolean of dropped voxels, or
                None. Named here rather than swept into kwargs so it
                survives _copy_with, which rebuilds through __init__.
            source (ImageSource): the images' source, or None; named
                here for the same reason as mask_dead
            patch_list (list): offsets added to y since it was loaded
                (see the class Attributes); None is none
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
        self.source = source
        self.patch_list = list(patch_list) if patch_list else []
        self.meta = meta if meta is not None else {}

    @property
    def dtype(self):
        """Return the dtype of the underlying y array, or None when unset."""
        return self.y.dtype if self.y is not None else None

    def _repr_dropped(self) -> str:
        """Render the screened-voxel count, or '' if never screened.

        The recorder stores an opaque output as its repr, so this is how
        the dropped-voxel count reaches a record. Shown even at zero, since
        screened-and-clean is worth telling apart from never-screened.
        """
        if self.mask_dead is None:
            return ''
        return f', num_vox_dropped={self.num_vox_dropped}'

    def __repr__(self):
        """A compact identity string: class name + the y dimensions.

        The (b, num_img, num_vox) shape, with no array hashing, so it is
        cheap enough for a log line or a recorded DataFrame cell.
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
            ExperimentImageOnly with sampled y, drawn through the
            SourceGauss it carries (so an inflate redraws the same box)
        """
        if b is None:
            if mu is not None:
                b = mu.size
            elif cov is not None:
                b = cov.shape[0]
            else:
                b = 1

        source = SourceGauss(shape=shape, b=b, num_img=num_img, seed=seed,
                             mu=mu, cov=cov, dtype=dtype)

        meta = kwargs.pop('meta', {})
        for name, value in source.get_meta().items():
            meta.setdefault(name, value)
        return cls(y=source.load(),
                   mask_idx=get_mask_idx(np.ones(shape)),
                   source=source, meta=meta, **kwargs)

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
                # dedupe before asserting: BIDS-derivatives paths repeat
                # the subject id in the directory and the filename, so a
                # natural regex matches twice. Two genuinely different ids
                # are still rejected.
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
        source = None
        subjects = sorted(df.index)
        if all(nii_in_file):
            # NIfTI path streams to y directly, holding no per-image dict
            y, y_names, mask_idx, affine = load_image_nii(
                df, dtype=dtype, mask=mask)
            # the realized support, not the mask argument: it may have been
            # inferred here, and a source has to know what it holds
            source = SourceNifti(df, mask=mask_idx > -1, dtype=dtype,
                                 affine=affine)
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

            # feature insertion order, not alphabetical, so channel_names
            # (RGB -> red, green, blue) keeps the caller's order
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

        return cls(y=y, mask_idx=mask_idx, source=source, meta=meta,
                   **kwargs)

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
            exp: a new experiment over those images, same voxels. It
                carries no source: a source reloads the images it was
                given, so it would hand back the full set in the original
                order, not this subset (see glow.experiment.source).
        """
        meta = deepcopy(self.meta)
        subjects = meta.get('subjects')
        if subjects is not None and len(subjects) == self.y.shape[1]:
            meta['subjects'] = [subjects[i] for i in img_idx]

        overrides.setdefault('source', None)
        overrides.setdefault('patch_list', None)
        return self._copy_with(y=self.y[:, img_idx, :], meta=meta,
                               **overrides)

    def get_img_segment(self, frac_segment: float = .5, *, seed: int = None,
                        group=None):
        """Return which images the segmentation fold takes.

        The partition split_img cuts, as a mask instead of a pair of
        experiments, for a caller that needs to know which images built
        something rather than the fold itself (the viewer's regression
        panel, labelling each image by fold).

        A uniform random draw, so each fold's share of the design's
        information is right on average but not balanced against an unlucky
        draw. The split depends on the images alone, never on y.

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

        The two folds are disjoint in images and identical in voxels, so a
        Ward tree built on exp_segment indexes the leaves of exp_test
        unchanged -- which is what lets AnalysisGLOWSplit segment on one
        fold and test on the other.

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

        # match y's dtype: numpy promotes float32 @ float64, which would
        # cost the bandwidth win of a float32 y
        num_img = self.y.shape[1]
        rng = np.random.default_rng(seed=seed)
        x = rng.standard_normal(size=(a, num_img))
        if self.y is not None and self.y.dtype != x.dtype:
            x = x.astype(self.y.dtype, copy=False)

        return Experiment(x=x, contrast=contrast, y=self.y,
                          mask_idx=self.mask_idx, mask_dead=self.mask_dead,
                          source=self.source, patch_list=self.patch_list,
                          meta=self.meta, **kwargs)

    def _copy_with(self, **overrides):
        """Return a deep copy of this experiment with attributes overridden.

        Clones every instance attribute and feeds them back to
        type(self)(**d), so a caller swapping one field need not repeat the
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

    def _to_y_space(self, y):
        """Return images the source read, in the space this y is in.

        The identity here, where y is the images themselves.
        ExperimentScaled overrides it, since a source reads raw images and
        its own y is pre-processed. This is the only thing inflate needs
        to know about what a subclass did to y.

        Args:
            y (np.array): (b, num_img, num_vox) images from the source

        Returns:
            y (np.array): (b, num_img, num_vox) in self.y's space
        """
        return y

    def inflate(self, halo_vox: int = None, mask=None):
        """Return a copy grown to more voxels, reloaded from the source.

        The voxels already held keep their values; the added ones are read
        from the source and carry every offset recorded since the load
        (see patch_list in the class Attributes). Growth is bounded by the
        source's own support, so a support against the brain's edge simply
        gets a thinner ring: that is all the context there is.

        Voxels drop_constant_vox screened come back as context, which is
        what a filter reading the parent images would see there; the screen
        is not re-run, and mask_dead still names them, so a later
        apply_mask over what was analysed leaves them behind again.

        Args:
            halo_vox (int): grow by this many voxels along each axis, xor
                mask. A box rather than a ball, because a separable kernel
                reads the cube of its radius (glow.experiment.smooth).
            mask (np.array): (X, Y, Z) boolean to grow to, xor halo_vox.
                Voxels already held are kept whether or not it names them.

        Returns:
            exp: a new experiment over the union, this one untouched

        Raises:
            ValueError: this experiment has no source, so its images
                cannot be read again
        """
        assert (halo_vox is None) != (mask is None), \
            'halo_vox xor mask required'
        if self.source is None:
            raise ValueError(
                'cannot inflate an experiment with no source: either it '
                'was built without one, or an operation since then cannot '
                'be replayed onto freshly loaded voxels (image selection, '
                'a sigma stretch, a permutation). Inflate before those, or '
                'rebuild from the source.')

        mask_held = self.mask_idx > -1
        if halo_vox is not None:
            mask = scipy.ndimage.maximum_filter(
                mask_held, size=2 * int(halo_vox) + 1, mode='constant')
        mask_out = (np.asarray(mask, dtype=bool) | mask_held)
        mask_out &= self.source.mask
        mask_add = mask_out & ~mask_held

        mask_idx_out = get_mask_idx(mask_out)
        y = np.empty((*self.y.shape[:2], int(mask_out.sum())),
                     dtype=self.y.dtype, order='F')
        # gather through both index arrays rather than assuming either is
        # numbered canonically, so a caller's own numbering survives
        y[:, :, mask_idx_out[mask_held]] = self.y[:, :, self.mask_idx[
            mask_held]]

        if mask_add.any():
            y_add = self._to_y_space(self.source.load(mask_add))
            mask_idx_add = get_mask_idx(mask_add)
            for patch in self.patch_list:
                sel = patch['mask'] & mask_add
                if sel.any():
                    y_add[:, :, mask_idx_add[sel]] += \
                        patch['offset'][..., np.newaxis]
            y[:, :, mask_idx_out[mask_add]] = y_add

        return self._copy_with(y=y, mask_idx=mask_idx_out)

    def smooth(self, fwhm: float):
        """Return a copy Gaussian-smoothed against its real neighbours.

        A kernel reads past the voxels it is given, so smoothing a crop on
        its own would dilute its whole edge against nothing. Context is
        borrowed from the source for the kernel and dropped again: what
        comes back holds exactly the voxels that went in, in the same
        numbering, so a smoothed fit tests what an unsmoothed one does and
        costs the same.

        Exact, not approximate. The halo is as deep as the kernel's own
        truncation radius, so within the kernel's reach of any voxel held
        here the mask is the one a whole-volume smooth would see there,
        numerator and denominator alike.

        The voxel size comes from meta['affine'] when there is one, so
        fwhm is in mm; without one the kernel is sized in voxels.

        Args:
            fwhm (float): kernel full width at half maximum, in mm. 0 or
                None returns this experiment untouched.

        Returns:
            exp: a new experiment over these voxels, smoothed

        Raises:
            ValueError: fwhm is set and there is no source to read the
                context from (see inflate)
        """
        if not fwhm:
            return self

        affine = self.meta.get('affine')
        exp = self.inflate(halo_vox=halo_vox_needed(fwhm, affine=affine))
        y = smooth_y(exp.y, mask_idx=exp.mask_idx, fwhm=fwhm, affine=affine)

        # gather the voxels held here back out of the halo, into the
        # numbering they already had rather than a fresh canonical one, so
        # a caller's own voxel order survives a smooth
        mask_held = self.mask_idx > -1
        col = np.empty(self.y.shape[2], dtype=int)
        col[self.mask_idx[mask_held]] = exp.mask_idx[mask_held]
        return self._copy_with(y=np.asfortranarray(y[:, :, col]))

    @property
    def num_vox_dropped(self) -> int:
        """How many voxels drop_constant_vox removed (0 if it never ran)."""
        return 0 if self.mask_dead is None else int(self.mask_dead.sum())

    def drop_constant_vox(self, rtol: float = 1e-6):
        """Return a new experiment with the constant voxels removed.

        A voxel whose intensities barely move across images has no signal
        to test: with the design projected out, its MANCOVA error matrix E
        is rank-deficient in all but name and det(E) underflows, after which
        every statistic on that voxel is NaN or +-inf.

        It runs ahead of ExperimentScaled for two reasons. The test is per
        feature, and ExperimentScaled mixes the features
        (y_out = pre_scale @ y), so one flat feature would be smeared over
        all b. And dropping here hands every method the same voxels, so
        GLOW and the voxel-wise arms control FWER over one family.

        The dropped voxels leave y and mask_idx is renumbered over what
        remains; mask_dead records which ones went.

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

        # scatter the flags back into image space through mask_idx, not
        # positionally, so any voxel numbering holds
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
            new experiment with offset applied, and the offset recorded in
                patch_list so an inflate replays it (see the class
                Attributes). A sigma stretch cannot be replayed -- it
                rescales deviations from the support's own voxel mean, so
                it does not act on one voxel at a time -- and drops the
                source instead.
        """
        assert (mask is None) != (vox_idx is None), \
            'either mask xor vox_idx required'

        if mask is None:
            mask = np.isin(self.mask_idx, vox_idx) & (self.mask_idx > -1)
        else:
            # what was applied, not what was asked for: the offset reaches
            # only voxels this experiment holds, and a patch records the act
            mask = np.asarray(mask, dtype=bool) & (self.mask_idx > -1)
        vox_idx = self.mask_idx[mask]

        y = deepcopy(self.y)
        y[:, :, vox_idx] += offset[..., np.newaxis]

        if sigma_scale is not None:
            y[:, :, vox_idx] = stretch_sigma(y=y[:, :, vox_idx],
                                             scale=sigma_scale)
            return self._copy_with(y=y, source=None)

        patch_list = self.patch_list + [{'offset': offset, 'mask': mask}]
        return self._copy_with(y=y, patch_list=patch_list)


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
            # match x's dtype: np.ones is float64, which would promote a
            # float32 x and, through decompose, the q matrices with it
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

        A fold whose x lost rank -- every subject at one level of a rare
        covariate landing in the other fold -- gives mancova.decompose a
        degenerate q0 / q1 and fails silently, so it is checked here, where
        the seed that produced it is still in hand.

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
            # get_freed_lane returns float64; cast so the einsum preserves
            # y.dtype. Every permutation of a fit comes through here, so a
            # silent f32 -> f64 promotion would cost the whole float32 win.
            if freed_lane.dtype != self.y.dtype:
                freed_lane = freed_lane.astype(self.y.dtype, copy=False)
            y = np.einsum('abc,bd->adc', self.y, freed_lane, optimize=True)

        # no source: it reloads the images unpermuted, and the permutation
        # mixes them, so freshly loaded voxels could not be brought into
        # line with the ones already here
        return Experiment(x=self.x, y=y, contrast=self.contrast,
                          mask_idx=self.mask_idx, mask_dead=self.mask_dead,
                          meta=deepcopy(self.meta))


class ExperimentScaled(Experiment):
    """Pre-processed experiment: zero-mean, variance-normalise, then PCA.

    y_out = pre_scale @ (y_in - mean_orig)

    The transform is fit once, by from_exp, and carried from there on: a
    copy, a crop or a screen keeps the transform its values were produced
    with (see __init__). Both coefficients are constant across images and
    across voxels, so the transform commutes with anything acting on the
    voxel axis alone, and applies to voxels it was not fit on.

    A source reads the images, which are not pre-processed, so what it
    returns has to go through prep before it lines up with y.

    Attributes:
        y (np.array): (b, num_img, num_vox) pre-processed image intensities
        mask_idx (np.array): voxel index array (-1 outside analysis)
        x (np.array): (a, num_img) design matrix
        contrast (np.array): (a,) boolean, True for features of interest
        mean_orig (np.array): (b, 1, 1) original grand mean
        pre_scale (np.array): (b, b) pre-processing matrix
    """

    @staticmethod
    def fit_pre_scale(y):
        """Fit the pre-processing transform on y.

        Args:
            y (np.array): (b, num_img, num_vox) raw image intensities

        Returns:
            pre_scale (np.array): (b, b) pre-processing matrix
            mean_orig (np.array): (b, 1, 1) grand mean, one per feature

        Raises:
            ValueError: if any feature has zero variance
        """
        # np.cov / eigh / mean return float64 whatever they are given, so
        # cast back at the end or prep(y) promotes y
        y_dtype = y.dtype

        # zero mean, accumulated in float64 then cast back to y.dtype
        mean_orig = y.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]
        mean_orig = mean_orig.astype(y_dtype, copy=False)

        b = y.shape[0]
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
        pre_scale = np.diag(1 / variances ** .5)

        cov_scale = pre_scale @ cov @ pre_scale.T
        evals, evecs = np.linalg.eigh(cov_scale)
        pre_scale = (evecs.T @ pre_scale).astype(y_dtype, copy=False)

        return pre_scale, mean_orig

    @staticmethod
    def apply_pre_scale(y, pre_scale, mean_orig):
        """Apply a fitted transform: y_out = pre_scale @ (y - mean_orig).

        Args:
            y (np.array): (b, num_img, num_vox) raw image intensities
            pre_scale (np.array): (b, b) pre-processing matrix
            mean_orig (np.array): (b, 1, 1) grand mean, one per feature

        Returns:
            y (np.array): (b, num_img, num_vox) pre-processed intensities
        """
        return np.einsum('ij,jkl->ikl', pre_scale, y - mean_orig)

    @classmethod
    def from_exp(cls, exp):
        """Fit the pre-processing on exp and return the scaled experiment.

        The one place the transform is fit.  Idempotent: an exp that is
        already an ExperimentScaled is returned unchanged (never
        re-scaled), so each Analysis.fit can pass whatever it was handed --
        raw or already-scaled -- through this one call.

        Args:
            exp: source Experiment (provides y, mask_idx, x, contrast,
                source, meta)

        Returns:
            ExperimentScaled with pre-processing applied to exp.y, or exp
            itself when it is already an ExperimentScaled
        """
        if isinstance(exp, cls):
            return exp

        pre_scale, mean_orig = cls.fit_pre_scale(exp.y)

        # an offset recorded in raw space becomes pre_scale @ offset here:
        # prep is affine, so prep(y + offset) == prep(y) + pre_scale @ offset,
        # which keeps every patch in the space of the y beside it
        patch_list = [{'offset': pre_scale @ patch['offset'],
                       'mask': patch['mask']}
                      for patch in getattr(exp, 'patch_list', None) or ()]

        return cls(y=cls.apply_pre_scale(exp.y, pre_scale, mean_orig),
                   pre_scale=pre_scale, mean_orig=mean_orig,
                   mask_idx=exp.mask_idx, x=exp.x, contrast=exp.contrast,
                   mask_dead=getattr(exp, 'mask_dead', None),
                   source=getattr(exp, 'source', None),
                   patch_list=patch_list,
                   meta=dict(exp.meta) if getattr(exp, 'meta', None) else None)

    def __init__(self, *, pre_scale, mean_orig, **kwargs):
        """Carry a fitted transform and the pre-processed y it produced.

        Fitting belongs to from_exp alone. This is what _copy_with rebuilds
        through, so every crop, screen and offset downstream of a scaling
        arrives here: a transform refit on the voxels that happen to be
        left would rotate the features differently and pre-process values
        that already are.

        Args:
            pre_scale (np.array): (b, b) pre-processing matrix
            mean_orig (np.array): (b, 1, 1) grand mean, one per feature
            **kwargs: Experiment's arguments, y among them and already
                pre-processed
        """
        super().__init__(**kwargs)
        self.pre_scale = pre_scale
        self.mean_orig = mean_orig

    def split_img(self, *args, **kwargs):
        """Refuse the split: pre-scaling was fit on every image.

        pre_scale and mean_orig come from all the images at once, so a fold
        cut out afterwards carries a transform the other fold helped choose.
        Split the raw Experiment instead; AnalysisGLOWSplit.fit scales each
        fold it cuts.

        Raises:
            TypeError: always.
        """
        raise TypeError('split before scaling: ExperimentScaled fits '
                        'pre_scale on every image, so both folds of a later '
                        'split share a transform the test fold helped '
                        'choose. Split the Experiment, then scale each fold.')

    def _to_y_space(self, y):
        """Pre-process images the source read, so they join a scaled y.

        Args:
            y (np.array): (b, num_img, num_vox) images from the source

        Returns:
            y (np.array): (b, num_img, num_vox) pre-processed
        """
        return self.prep(y)

    def prep(self, y):
        """Apply this experiment's pre-processing to y.

        Args:
            y (np.array): (b, num_img, num_vox) raw image intensities

        Returns:
            y (np.array): (b, num_img, num_vox) pre-processed intensities
        """
        return self.apply_pre_scale(y, self.pre_scale, self.mean_orig)

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
