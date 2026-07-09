"""Analysis ABC and shared FWER / effect-discovery machinery."""

from abc import ABC, abstractmethod
from bisect import bisect_left
from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import label
from tqdm import tqdm

import glow.effect
import glow.graph


class Analysis(ABC):
    """Perform effect discovery (GLOW or TFCE) and compute FWER p-values.

    An Analysis is a reusable recipe -- its config knobs only. The
    experiment is not stored; it is passed to fit(), which scales it
    (ExperimentScaled) on the way in, so the same recipe can be fit
    against any experiment.

    Attributes:
        effect_list (list): discovered Effect objects (populated by fit)
        pval (np.array): (num_reg,) FWER-controlled p-values (set by fit)
    """

    # the __init__ config knobs that identify the recipe -- the fields __repr__
    # renders. Subclasses declare their own. Never includes exp, the fitted
    # arrays, or pval -- only the immutable recipe.
    RECORD_FIELDS = ()

    def __init__(self):
        self.effect_list = None
        self.pval = None

    def __repr__(self):
        """A compact recipe string: class name + the RECORD_FIELDS knobs.

        Reuses RECORD_FIELDS (the config subset that identifies the recipe) as
        the single source of truth, so the repr tracks the recipe automatically
        and never shows fitted arrays or the experiment. A stat-function knob
        renders as its __name__, so it stays an address-free name rather than
        '<function ... at 0x...>'.
        """
        parts = []
        for name in self.RECORD_FIELDS:
            v = getattr(self, name)
            if callable(v) and not isinstance(v, type):
                v = getattr(v, '__name__', v)
            parts.append(f'{name}={v}')
        return f'{type(self).__name__}({", ".join(parts)})'

    @abstractmethod
    def fit(self, exp):
        """Run the analysis computation on exp and return self.

        Implementations scale exp with ExperimentScaled.from_exp(exp)
        first (idempotent -- a raw exp is scaled, an already-scaled one
        passes through), then compute.
        """

    @classmethod
    def get_pval(cls, stat, reg_active=None, *, stat_null=None):
        """Compute FWER-adjusted p-values via Westfall-Young permutation.

        Westfall & Young 1993: the max-statistic null over the active
        comparison set controls the family-wise error rate.

        Two modes, sharing the same max-stat bisect:

        - Single tree (VBA / CET): pass stat as the full
          (n_perm+1, num_reg) matrix (row 0 observed). The max-stat null
          is built column-wise as sort(nanmax(stat[:, reg_active])) and
          the observed per-region statistic is stat[0].

        - Per-permutation trees (GLOW): the null cannot be read off a
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

        For each voxel, the mean and std are computed from all rows
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

    def __init__(self, get_stat: Callable = None):
        super().__init__()
        if get_stat is None:
            from .mancova import get_wilks
            get_stat = get_wilks
        self.get_stat = get_stat
        self.stat = None

    @classmethod
    def get_stat_perm_multi(cls, exp, get_stat_list: list,
                            children=None) -> dict:
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

    @staticmethod
    def _walk_stat(exp, get_stat: Callable, children=None):
        """Compute one stat per region for a fixed (already-permuted) exp.

        The shared MANCOVA tree-walk behind get_stat_perm and the joblib
        worker _stat_row. get_stat is passed explicitly (not read off
        self) so a joblib task carries no instance state.

        Args:
            exp (Experiment): experiment to evaluate (already permuted if
                this is a permutation draw)
            get_stat (Callable): per-region stat function (e, h, n)
            children (np.array): (num_reg - num_vox, 2) child index array.
                If None, only iterates through individual voxels.

        Returns:
            stat (np.array): (num_reg,) test statistics
        """
        num_vox = exp.y.shape[2]
        num_reg = num_vox
        if children is not None:
            num_reg += children.shape[0]

        stat = np.full(num_reg, fill_value=np.nan)
        for reg_idx, size, e, h in glow.graph.iter_mancova(exp=exp,
                                                           children=children):
            try:
                stat[reg_idx] = get_stat(e=e, h=h, n=size)
            except np.linalg.LinAlgError:
                pass

        return stat

    def get_stat_perm(self, exp, children=None):
        """Compute the test statistic for each region.

        Computes one stat per region on the given (possibly permuted)
        experiment. For permutation nulls, callers must loop
        externally over exp.permute(k).

        Args:
            exp (Experiment): experiment to evaluate (already permuted
                if this is a permutation draw)
            children (np.array): (num_reg - num_vox, 2) child index array.
                If None, only iterates through individual voxels.

        Returns:
            stat (np.array): (num_reg,) test statistics
        """
        return self._walk_stat(exp, self.get_stat, children=children)

    @classmethod
    def _stat_row(cls, exp, k: int, get_stat: Callable, children=None):
        """Per-region stat row for outer perm k (pure, for joblib workers).

        Permutes exp by outer-perm index k (k=0 = observed data), then
        computes one stat per region. Kept free of instance state so
        joblib workers can run it; k seeds the permutation, so the row
        is a deterministic function of k alone.
        """
        _exp = exp.permute(k) if k else exp
        return cls._walk_stat(_exp, get_stat, children=children)

    def build_stat_matrix(self, exp, _stat=None, *, n_jobs: int = 1,
                          verbose: bool = False):
        """Per-voxel stat matrix for the FWER walk, (n_perm_fwer+1, num_vox).

        If _stat is None, runs the Freedman-Lane permutation walk on exp
        (row 0 observed, rows 1: permuted), parallelised over
        permutations with joblib. If provided, validates its permutation
        count against self.n_perm_fwer and returns it unchanged (the
        caller owns the copy).

        Row k depends only on k (exp.permute(k) is seeded by k), so the
        matrix is identical regardless of n_jobs.

        Args:
            exp (Experiment): experiment to walk (already scaled).
            _stat (np.array): optional precomputed (n_perm_fwer+1,
                num_vox) stat matrix; returned unchanged after a shape
                check.
            n_jobs (int): permutation-level parallelism via joblib. 1
                (default) runs in-process; -1 uses all cores.
            verbose (bool): show a tqdm bar over permutations.
        """
        if _stat is None:
            n_total = self.n_perm_fwer + 1
            num_vox = exp.y.shape[2]
            _stat = np.full((n_total, num_vox), np.nan)
            rows = Parallel(n_jobs=n_jobs, return_as='generator')(
                delayed(self._stat_row)(exp, k, self.get_stat, children=None)
                for k in range(n_total))
            for k, row in enumerate(tqdm(rows, total=n_total,
                                         desc='fwer perms',
                                         disable=not verbose)):
                _stat[k, :] = row
        elif _stat.shape[0] - 1 != self.n_perm_fwer:
            raise ValueError(
                f'_stat has {_stat.shape[0] - 1} permutations but '
                f'n_perm_fwer={self.n_perm_fwer}')
        return _stat
