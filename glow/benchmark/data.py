"""DataSource classes for the benchmark refactor.

Hashable + memoising data builders. Public surface is the .exp
property; the build implementation is _get().

Hierarchy:
    DataSource (ABC, HashBySlots)
    ├── DataSourceWGN                 (synthetic gaussian noise)
    └── DataSourceDataFrame           (real data via a per-subject df)
        └── DataSourceHCP             (HCP-YA open dataset)
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import glow
from glow.benchmark import hcp
from glow.util import HashBySlots, hash_array


class DataSource(HashBySlots, ABC):
    """Build an Experiment, owning design-matrix construction and crop.

    Abstract base of the data-source hierarchy (DataSource ->
    DataSourceWGN / DataSourceDataFrame -> DataSourceHCP). Subclasses
    implement _get to assemble the image-only Experiment; this base adds
    the shared design matrix X (interest + nuisance + optional bias) and
    the optional extenter crop. Instances are hashable (HashBySlots), so
    equal sources share one memoised build via the class-level cache.

    Attributes:
        a (int): number of design-matrix features of interest.
        a_nuisance (int): number of nuisance design-matrix features.
        has_bias (bool): whether to append a bias (constant) column.
        seed (int): RNG seed for X sampling and the extenter crop.
        extenter (Callable | None): callable producing a boolean support
            mask to crop the experiment, or None for no crop.
    """

    __slots__ = ('a', 'a_nuisance', 'has_bias', 'seed', 'extenter')

    # Memoised Experiment cache. Class-level (not a slot — slots are
    # identity-only) and keyed by self, so equal sources share a build.
    _exp_cache = {}

    def __init__(self, *, a: int = 1, a_nuisance: int = 0,
                 has_bias: bool = True, seed: int = 0, extenter=None):
        self.a = int(a)
        self.a_nuisance = int(a_nuisance)
        self.has_bias = bool(has_bias)
        self.seed = int(seed)
        self.extenter = extenter

    @staticmethod
    def _canon(v):
        """Canonicalise one slot value for hashing / equality.

        Extends HashBySlots._canon with a DataFrame case so a per-subject
        image dataframe contributes a stable, order-sensitive identity
        (columns, index, and a content hash) rather than failing to
        JSON-serialise.
        """
        if isinstance(v, pd.DataFrame):
            return {'df_columns': [str(c) for c in v.columns],
                    'df_index': [str(i) for i in v.index],
                    'df_hash': hash_array(
                        pd.util.hash_pandas_object(v, index=True).to_numpy())}
        return HashBySlots._canon(v)

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
        and applies the extenter support mask if one was given.

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
            mask = self.extenter(mask_idx=exp.mask_idx,
                                 seed=self.seed,
                                 contiguous=True)
            exp = exp.apply_mask(mask)
        return exp

    @abstractmethod
    def _get(self):
        """Build and return the full Experiment (image data + X)."""


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

    __slots__ = ('shape', 'b', 'num_img')

    def __init__(self, *, shape=(5, 5, 5), b: int = 2, num_img: int = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.shape = tuple(int(x) for x in shape)
        self.b = int(b)
        self.num_img = int(num_img)

    def _get(self):
        """Build the WGN image-only Experiment, then sample X / crop."""
        exp_img = glow.experiment.ExperimentImageOnly.from_gauss(
            seed=self.seed, shape=self.shape, b=self.b,
            num_img=self.num_img)
        return self._sample_x_and_crop(exp_img)


class DataSourceDataFrame(DataSource):
    """Build a real-data Experiment from a per-subject image dataframe.

    The dataframe has index = subject_id, columns = imaging features, and
    values = file paths (see ExperimentImageOnly.from_paths).

    Attributes:
        df (pd.DataFrame): per-subject image-path dataframe, sorted by
            index for deterministic identity and subject iteration.
    """

    __slots__ = ('df',)

    def __init__(self, *, df: pd.DataFrame, **kwargs):
        super().__init__(**kwargs)
        # sort by index so identity is order-invariant and downstream
        # subject iteration is deterministic
        self.df = df.sort_index()

    def _get(self):
        """Load images from the dataframe paths, then sample X / crop."""
        exp_img = glow.experiment.ExperimentImageOnly.from_paths(self.df)
        return self._sample_x_and_crop(exp_img)


class DataSourceHCP(DataSourceDataFrame):
    """Build an Experiment from glow's reference HCP-YA open dataset.

    Searches the hcp module's local maps (hcp.ensure_hcp_data downloads
    the published Zenodo dataset on first use, gated on the HCP Data Use
    Terms) for the requested features, then defers to DataSourceDataFrame
    for loading and design-matrix construction.

    Attributes:
        hcp_feats (tuple[str]): subset of hcp.HCP_FEATS to load
            (DKI fa/md/mk, NODDI icvf/isovf/od).
    """

    __slots__ = ('hcp_feats',)

    def __init__(self, *, hcp_feats=hcp.HCP_FEATS, **kwargs):
        self.hcp_feats = tuple(hcp_feats)
        folder = hcp.ensure_hcp_data()
        glob_dict = {feat: hcp.IMG_GLOB_DICT[feat] for feat in self.hcp_feats}
        df = glow.experiment.ExperimentImageOnly._search_files(
            folder, hcp.SBJ_REGEX, glob_dict)
        assert df.size, f'no HCP maps found under {folder}'
        super().__init__(df=df[list(self.hcp_feats)], **kwargs)

    def _get(self):
        """Load the HCP maps over the dataset brain mask, then sample X / crop.

        The brain mask shipped with the dataset defines the analysis
        support, in place of from_paths's default every-image-nonzero rule.
        That rule wrongly excludes in-brain voxels where NODDI isovf is
        legitimately zero (dense tissue / low free water).
        """
        folder = hcp.ensure_hcp_data()
        mask = next(folder.glob(hcp.MASK_GLOB))
        exp_img = glow.experiment.ExperimentImageOnly.from_paths(
            self.df, mask=mask)
        return self._sample_x_and_crop(exp_img)
