"""Analysis ABC and shared FWER / effect-discovery machinery."""

from abc import ABC, abstractmethod
from bisect import bisect_left
from typing import Callable

import numpy as np
from scipy.ndimage import label

import glow.effect
import glow.graph
from glow.experiment.exper import ExperimentScaled


def _canon(v):
    """Canonicalize one recorded config value to a JSON-friendly form.

    A spec exposing to_record uses it; a bare callable (a stat function)
    records as its __name__; everything else passes through unchanged --
    scalars, bools, and a StrEnum like ClusterMode, which is already a
    JSON-native str serialising to its value (e.g. 'Focus').
    """
    hook = getattr(v, 'to_record', None)
    if callable(hook):
        return hook()
    if callable(v):
        return getattr(v, '__name__', repr(v))
    return v


class Analysis(ABC):
    """Perform effect discovery (GLOW or TFCE) and compute FWER p-values.

    Attributes:
        exp (Experiment): source data
        effect_list (list): discovered Effect objects (populated by fit)
        pval (np.array): (num_reg,) FWER-controlled p-values (set by fit)
    """

    # the __init__ config knobs recorded by to_record; subclasses declare
    # their own. Never includes exp, the fitted arrays, or pval -- only the
    # immutable recipe (see to_record).
    RECORD_FIELDS = ()

    def __init__(self, exp):
        if not isinstance(exp, ExperimentScaled):
            exp = ExperimentScaled.from_exp(exp)
        self.exp = exp
        self.effect_list = None
        self.pval = None

    def to_record(self) -> dict:
        """Build a JSON-friendly recipe dict: class name + config knobs.

        The benchmark Recorder serialises any value exposing to_record (see
        glow._extra.benchmark.recorder). This records only the configuration subset
        declared in RECORD_FIELDS (the __init__ knobs) -- never exp, the
        fitted outputs (effect_list, pval, and any per-region array), nor the
        large data arrays. exp is omitted deliberately: it is recorded as its
        own input arg wherever an Analysis is built, so nesting it here would
        duplicate it.

        Returns:
            a dict of {kind, <each RECORD_FIELDS knob, canonicalized>}
        """
        out = {'kind': type(self).__name__}
        for name in self.RECORD_FIELDS:
            out[name] = _canon(getattr(self, name))
        return out

    @abstractmethod
    def fit(self):
        """Run the analysis computation and return self."""

    @classmethod
    def get_pval(cls, stat, reg_active=None, *, stat_null=None):
        """Compute FWER-adjusted p-values via Westfall-Young permutation.

        Westfall & Young 1993: the max-statistic null over the active
        comparison set controls the family-wise error rate.

        Two modes, sharing the same max-stat bisect:

        * Single tree (VBA / CET): pass stat as the full
          (n_perm+1, num_reg) matrix (row 0 observed). The max-stat null
          is built column-wise as sort(nanmax(stat[:, reg_active])) and
          the observed per-region statistic is stat[0].

        * Per-permutation trees (GLOW): the null cannot be read off a
          single matrix because each permutation has its own tree, so
          pass the precomputed max-stat null as stat_null (one entry per
          permutation incl. observed) and the observed per-region
          statistic as stat (a (num_reg,) 1-D array).

        Args:
            stat (np.array): (n_perm+1, num_reg) statistics per region
                (row 0 observed), or (num_reg,) observed statistics when
                stat_null is given.
            reg_active (np.array): (num_reg,) boolean mask. Only active
                regions have a p-value computed; inactive get np.nan.
                Discarding a-priori small regions from the comparison
                set preserves power for larger regions. Defaults to all
                regions active.
            stat_null (np.array): optional (n_perm+1,) precomputed
                max-stat null (one entry per permutation incl. observed).

        Returns:
            pval (np.array): (num_reg,) FWER-controlled p-values
        """
        stat_obs = stat[0] if stat_null is None else stat
        num_reg = stat_obs.shape[0]

        if reg_active is None:
            reg_active = np.ones(num_reg, dtype=bool)
        elif not reg_active.any():
            return np.full(num_reg, fill_value=np.nan)

        # max stat per permutation, sorted low to high (bisect_left below
        # needs an ascending array)
        if stat_null is None:
            null_sorted = np.sort(np.nanmax(stat[:, reg_active], axis=1))
        else:
            null_sorted = np.sort(stat_null)
        n_null = len(null_sorted)

        # p-value: fraction of permuted-or-observed max-stats >= the
        # region's observed value
        pval = np.full(num_reg, fill_value=-1.0)
        for reg_idx, z in enumerate(stat_obs):
            if np.isnan(z):
                pval[reg_idx] = np.nan
                continue
            pval[reg_idx] = max(1 - bisect_left(null_sorted, z) / n_null,
                                1 / n_null)

        # inactive regions must stay NaN: assigning them a p-value would
        # expand the comparison set and break FWER control
        pval[~reg_active] = np.nan

        return pval

    @classmethod
    def z_score_stat(cls, stat):
        """Z-score each voxel across permutations (observed row included).

        For each voxel, the mean and std are computed from **all** rows
        — the observed row (0) together with the permutation null (1:) —
        then every row is standardized by that voxel's empirical mean and
        std.  This equalizes per-voxel scale so max-stat FWER is not
        biased by regional heterogeneity.

        Under H0 the observed row is exchangeable with the permuted rows
        (Phipson & Smyth 2010; Winkler et al. 2014), so it must contribute
        to the standardization on equal footing — otherwise row 0 is
        divided by a std it did not contribute to while rows 1: are
        divided by a std they did, and max-stat FWER drifts above
        nominal at finite B (see test_stat_reliability.py).

        Args:
            stat (np.array): (n_perm+1, num_vox) statistics.
                Row 0 is the observed (unpermuted) statistic.

        Returns:
            z (np.array): same shape, voxel-wise z-scored
        """
        mu = np.nanmean(stat, axis=0)
        std = np.nanstd(stat, axis=0, ddof=1)
        std[std < 1e-12] = 1.0
        return (stat - mu) / std

    @classmethod
    def discover_mask(cls, mask, exp):
        """Split a boolean mask into connected-component effects.

        Args:
            mask (np.array): boolean mask, same shape as exp.mask_idx
            exp (Experiment): experiment for constructing Effect objects

        Returns:
            effect_list (list): discovered Effect objects
        """
        # one effect per connected component: adjacent voxels belong to
        # the same effect
        mask_est, num_effect = label(mask.astype(bool))

        effect_list = list()
        for eff_idx in range(1, num_effect + 1):
            _mask = mask_est == eff_idx
            eff = glow.effect.EffectEstimate.from_exp_mask(exp=exp, mask=_mask)
            effect_list.append(eff)

        return effect_list


class AnalysisVoxel(Analysis):
    """Analysis with a pluggable per-region stat function.

    VBA and CET compute one stat per voxel via get_stat (Wilks,
    Hotelling-Lawley-trace, etc.). AnalysisGLOW does not subclass
    this -- its inner kernel hard-codes LLR.

    Attributes:
        get_stat (Callable): per-region stat function f(e, h, n) -> float
        stat (np.array): (n_perm+1, num_reg) statistics (set by fit)
    """

    def __init__(self, exp, get_stat: Callable = None):
        super().__init__(exp)
        if get_stat is None:
            from .mancova import get_wilks
            get_stat = get_wilks
        self.get_stat = get_stat
        self.stat = None

    @classmethod
    def get_stat_perm_multi(cls, exp, get_stat_list: list, children=None) -> dict:
        """Compute multiple test statistics from a single tree walk.

        Avoids redundant E/H computation when comparing stat functions.
        Computes one stat value per region per stat function, on the
        given (possibly permuted) experiment. For permutation nulls,
        callers must loop externally over exp.permute(k).

        Args:
            exp (Experiment): experiment data (already permuted if
                this is a permutation draw)
            get_stat_list (list): stat functions (each accepts e, h, n)
            children (np.array): (num_leaf - 1, 2) child index array

        Returns:
            dict mapping each stat function to a (num_reg,) array
        """
        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox
        if children is not None:
            num_reg += children.shape[0]

        result = {fn: np.full(num_reg, fill_value=np.nan)
                  for fn in get_stat_list}

        for reg_idx, size, e, h in glow.graph.iter_mancova(
                exp=exp, children=children):
            for fn in get_stat_list:
                try:
                    result[fn][reg_idx] = fn(e=e, h=h, n=size)
                except np.linalg.LinAlgError:
                    pass

        return result

    def get_stat_perm(self, exp, children=None):
        """Compute the test statistic for each region.

        Computes one stat per region on the given (possibly permuted)
        experiment. For permutation nulls, callers must loop
        externally over exp.permute(k).

        Args:
            exp (Experiment): experiment to evaluate (already permuted
                if this is a permutation draw)
            children (np.array): (num_reg, 2) child index array. If None,
                only iterates through individual voxels.

        Returns:
            stat (np.array): (num_reg,) test statistics
        """
        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox
        if children is not None:
            num_reg += children.shape[0]

        stat = np.full(num_reg, fill_value=np.nan)
        for reg_idx, size, e, h in glow.graph.iter_mancova(exp=exp,
                                                           children=children):
            try:
                stat[reg_idx] = self.get_stat(e=e, h=h, n=size)
            except np.linalg.LinAlgError:
                pass

        return stat

    def build_stat_matrix(self, _stat=None):
        """Per-voxel stat matrix for the FWER walk, (n_perm_fwer+1, num_vox).

        If ``_stat`` is None, runs the Freedman-Lane permutation walk
        (row 0 observed, rows 1: permuted).  If provided, validates its
        permutation count against ``self.n_perm_fwer`` and returns it
        unchanged (the caller owns the copy).
        """
        if _stat is None:
            num_vox = self.exp.y.shape[2]
            _stat = np.full((self.n_perm_fwer + 1, num_vox), np.nan)
            for k in range(self.n_perm_fwer + 1):
                _exp = self.exp.permute(k) if k else self.exp
                _stat[k, :] = self.get_stat_perm(_exp, children=None)
        elif _stat.shape[0] - 1 != self.n_perm_fwer:
            raise ValueError(
                f'_stat has {_stat.shape[0] - 1} permutations but '
                f'n_perm_fwer={self.n_perm_fwer}')
        return _stat
