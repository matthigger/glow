"""DataSource classes for the benchmark refactor.

Hashable + memoising data builders. Public surface is the .exp
property; the build implementation is _get().

Hierarchy:
    DataSource (ABC, DataclassJSON)
    ├── DataSourceWGN                 (synthetic gaussian noise)
    └── DataSourceHCP                 (HCP-YA open dataset)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import glow
from glow.benchmark import hcp
from glow.util import DataclassJSON


@dataclass(frozen=True, slots=True, kw_only=True)
class DataSource(DataclassJSON, ABC):
    """Build an Experiment, owning design-matrix construction and crop.

    Abstract base of the data-source hierarchy (DataSource ->
    DataSourceWGN / DataSourceHCP). Subclasses implement _get to assemble
    the image-only Experiment; this base adds the shared design matrix X
    (interest + nuisance + optional bias) and the optional extenter crop.
    Instances are frozen and hashable (native dataclass __hash__), so equal
    sources share one memoised build via the class-level cache.

    Attributes:
        a (int): number of design-matrix features of interest.
        a_nuisance (int): number of nuisance design-matrix features.
        has_bias (bool): whether to append a bias (constant) column.
        seed (int): RNG seed for X sampling.
        extenter (Extenter | None): a frozen extenter producing a boolean
            support mask to crop the experiment, or None for no crop. The
            extenter carries its own seed / contiguity, so the crop is a
            pure function of the source's identity.
    """

    a: int = 1
    a_nuisance: int = 0
    has_bias: bool = True
    seed: int = 0
    extenter: object = None

    # Memoised Experiment cache. Class-level (not a field -- fields are
    # identity-only) and keyed by self, so equal sources share a build.
    _exp_cache = {}

    def __post_init__(self):
        object.__setattr__(self, 'a', int(self.a))
        object.__setattr__(self, 'a_nuisance', int(self.a_nuisance))
        object.__setattr__(self, 'has_bias', bool(self.has_bias))
        object.__setattr__(self, 'seed', int(self.seed))

    @property
    def exp(self):
        """Return the built Experiment, memoised and read-only.

        Builds via _get on first access for this identity, freezes its
        arrays writeable=False so the shared instance can't be mutated,
        and caches it keyed by self so equal sources reuse one build.

        Returns:
            The fully built Experiment (image data y of shape
            (b, num_img, num_vox), design matrix X, and contrast).
        """
        if self not in self._exp_cache:
            exp = self._get()
            # freeze so cached experiment arrays can't be overwritten by
            # one consumer and corrupt another sharing the same identity
            for val in exp.__dict__.values():
                if isinstance(val, np.ndarray):
                    val.flags.writeable = False
            self._exp_cache[self] = exp
        return self._exp_cache[self]

    def _sample_x_and_crop(self, exp_img_only):
        """Sample the design matrix X onto an image-only Experiment, crop.

        Shared by subclass _get: builds the contrast (a_nuisance False
        columns then a True columns), samples X (with optional bias),
        and applies the extenter support mask if one was given. The
        extenter carries its own seed and contiguity.

        Args:
            exp_img_only: image-only Experiment with y of shape
                (b, num_img, num_vox), prior to design-matrix sampling.

        Returns:
            The Experiment with X and contrast attached, cropped to the
            extenter support when set.
        """
        contrast = np.array([False] * self.a_nuisance + [True] * self.a,
                            dtype=bool)
        exp = exp_img_only.sample_x(contrast=contrast,
                                    add_bias=self.has_bias,
                                    seed=self.seed)
        if self.extenter is not None:
            mask = self.extenter(mask_idx=exp.mask_idx)
            exp = exp.apply_mask(mask)
        return exp

    @abstractmethod
    def _get(self):
        """Build and return the full Experiment (image data + X)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DataSourceWGN(DataSource):
    """Build a synthetic white-Gaussian-noise Experiment.

    Generates image data y of shape (b, num_img, num_vox) from i.i.d.
    Gaussian noise (no planted effect), then attaches X via the base.

    Attributes:
        shape (tuple[int]): spatial shape of each image, e.g. (5, 5, 5);
            num_vox is its product.
        b (int): number of imaging features per voxel (y channels);
            distinct from the design-matrix a.
        num_img (int): number of images (observations).
    """

    shape: tuple = (5, 5, 5)
    b: int = 2
    num_img: int = 100

    def __post_init__(self):
        DataSource.__post_init__(self)
        object.__setattr__(self, 'shape', tuple(int(x) for x in self.shape))
        object.__setattr__(self, 'b', int(self.b))
        object.__setattr__(self, 'num_img', int(self.num_img))

    def _get(self):
        """Build the WGN image-only Experiment, then sample X / crop."""
        exp_img = glow.experiment.ExperimentImageOnly.from_gauss(
            seed=self.seed, shape=self.shape, b=self.b,
            num_img=self.num_img)
        return self._sample_x_and_crop(exp_img)


@dataclass(frozen=True, slots=True, kw_only=True)
class DataSourceHCP(DataSource):
    """Build an Experiment from glow's reference HCP-YA open dataset.

    Identity is the requested feature subset (plus the base design /
    crop). The per-subject image dataframe is derived from hcp_feats at
    build time, not stored, so no field holds a DataFrame.

    hcp.ensure_hcp_data downloads the published Zenodo dataset on first
    .exp access (gated on the HCP Data Use Terms); the brain mask shipped
    with the dataset defines the analysis support, in place of from_paths's
    default every-image-nonzero rule (which wrongly excludes in-brain
    voxels where NODDI isovf is legitimately zero -- dense tissue / low
    free water).

    Attributes:
        hcp_feats (tuple[str]): subset of hcp.HCP_FEATS to load
            (DKI fa/md/mk, NODDI icvf/isovf/od).
    """

    hcp_feats: tuple = hcp.HCP_FEATS

    def __post_init__(self):
        DataSource.__post_init__(self)
        object.__setattr__(self, 'hcp_feats', tuple(self.hcp_feats))

    def _get(self):
        """Load the HCP maps over the dataset brain mask, then sample X / crop."""
        folder = hcp.ensure_hcp_data()
        glob_dict = {feat: hcp.IMG_GLOB_DICT[feat] for feat in self.hcp_feats}
        df = glow.experiment.ExperimentImageOnly._search_files(
            folder, hcp.SBJ_REGEX, glob_dict)
        assert df.size, f'no HCP maps found under {folder}'
        # column order = hcp_feats; sort by subject id for deterministic
        # subject iteration
        df = df[list(self.hcp_feats)].sort_index()
        mask = next(folder.glob(hcp.MASK_GLOB))
        exp_img = glow.experiment.ExperimentImageOnly.from_paths(df, mask=mask)
        return self._sample_x_and_crop(exp_img)
