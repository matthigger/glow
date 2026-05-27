"""DataSource classes for the benchmark refactor.

Hashable + memoising data builders. Public surface is the ``.exp``
property; the build implementation is ``_get()``.

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
from glow.util import HashBySlots, hash_array


class DataSource(HashBySlots, ABC):
    """owns X (design matrix) construction and the optional crop."""

    __slots__ = ('a', 'a_nuisance', 'has_bias', 'seed', 'extenter')

    # Memoised Experiment cache. Class-level (not a slot — slots are
    # identity-only) and keyed by `self`, so equal sources share a build.
    _exp_cache = {}

    def __init__(self, *, a=1, a_nuisance=0, has_bias=True, seed=0,
                 extenter=None):
        self.a = int(a)
        self.a_nuisance = int(a_nuisance)
        self.has_bias = bool(has_bias)
        self.seed = int(seed)
        self.extenter = extenter

    @staticmethod
    def _canon(v):
        if isinstance(v, pd.DataFrame):
            return {'df_columns': [str(c) for c in v.columns],
                    'df_index': [str(i) for i in v.index],
                    'df_hash': hash_array(
                        pd.util.hash_pandas_object(v, index=True).to_numpy())}
        return HashBySlots._canon(v)

    @property
    def exp(self):
        if self not in self._exp_cache:
            exp = self._get()
            # ensure experiment arrays aren't overwritten
            for val in exp.__dict__.values():
                if isinstance(val, np.ndarray):
                    val.flags.writeable = False
            self._exp_cache[self] = exp
        return self._exp_cache[self]

    def _sample_x_and_crop(self, exp_img_only):
        """Shared X-sampling + optional crop. Used by subclass _get()."""
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
        """Build and return the full Experiment."""


class DataSourceWGN(DataSource):
    """White-Gaussian-noise data source.

    Attributes:
        shape (tuple[int]): spatial shape of each image, e.g. (5, 5, 5).
        b (int): number of imaging features per voxel (y channels);
            distinct from the design-matrix ``a``.
        num_img (int): number of images (observations).
    """

    __slots__ = ('shape', 'b', 'num_img')

    def __init__(self, *, shape=(5, 5, 5), b=2, num_img=100, **kwargs):
        super().__init__(**kwargs)
        self.shape = tuple(int(x) for x in shape)
        self.b = int(b)
        self.num_img = int(num_img)

    def _get(self):
        exp_img = glow.experiment.ExperimentImageOnly.from_gauss(
            seed=self.seed, shape=self.shape, b=self.b,
            num_img=self.num_img)
        return self._sample_x_and_crop(exp_img)


class DataSourceDataFrame(DataSource):
    """real-data source identified by a per-subject image dataframe.

    index = subject_id, columns = imaging features, values = file paths
    (see ``ExperimentImageOnly.from_paths``).

    Attributes:
        df (pd.DataFrame): per-subject image-path dataframe.
    """

    __slots__ = ('df',)

    def __init__(self, *, df, **kwargs):
        super().__init__(**kwargs)
        # sort by index so identity is order-invariant and downstream
        # subject iteration is deterministic
        self.df = df.sort_index()

    def _get(self):
        exp_img = glow.experiment.ExperimentImageOnly.from_paths(self.df)
        return self._sample_x_and_crop(exp_img)


class DataSourceHCP(DataSourceDataFrame):
    """HCP-YA open-dataset source.

    Attributes:
        hcp_feats (tuple[str]): subset of ('fa', 'md') to load.
    """

    __slots__ = ('hcp_feats',)

    def __init__(self, *, hcp_feats=('fa', 'md'), **kwargs):
        from brainjar import hcp_ya_open
        self.hcp_feats = tuple(hcp_feats)
        super().__init__(
            df=hcp_ya_open.get_df_image()[list(self.hcp_feats)],
            **kwargs)
