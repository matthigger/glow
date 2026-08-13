"""Analysis ABC and shared FWER / effect-discovery machinery."""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Callable, NamedTuple

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import label
from tqdm import tqdm

import glow.effect
import glow.graph


def resolve_n_jobs(n_jobs: int) -> int:
    """Clamp a requested worker count to the cores this machine has.

    A benchmark config names one worker count for every machine it may run
    on (glow._extra.benchmark.config.GLOW_FIT_N_JOBS) and joblib takes
    n_jobs literally, so 32 there is 32 processes on a 4-core Batch
    container, each holding its own copy of y. Clamping makes the config's
    figure an upper bound rather than a demand. Negative counts keep
    joblib's reading (-1 all cores, -2 all but one).

    Args:
        n_jobs (int): requested workers.

    Returns:
        n_jobs (int): the request, capped at os.cpu_count(), floored at 1.
    """
    n_cpu = os.cpu_count() or 1
    if n_jobs < 0:
        return max(1, n_cpu + 1 + n_jobs)
    return max(1, min(n_jobs, n_cpu))


def reject_gpu(gpu, name: str) -> None:
    """Raise if a device fit was demanded of an analysis with no backend.

    Every fit takes gpu so one fit_params dict can be handed to any recipe
    (glow._extra.benchmark.run.run_ana). gpu='auto' asks for a device only
    where one helps, so it is a silent no-op here; an explicit gpu=True is
    an error rather than a silent hour on the CPU.

    Args:
        gpu: the fit(gpu=...) argument.
        name (str): the analysis class name, for the message.

    Raises:
        ValueError: gpu names a device explicitly.
    """
    if gpu and gpu != 'auto':
        raise ValueError(f'{name} has no GPU backend (only AnalysisGLOW '
                         f"has one); pass gpu=False or gpu='auto'")


class MaxStatPermResult(NamedTuple):
    """One Westfall-Young max-stat permutation test over a comparison set.

    Self-contained: pval follows from stat_obs, max_stat and reg_active
    alone, so a stored result stays inspectable once the
    (n_perm+1, num_reg) matrix behind it is gone -- which AnalysisGLOW
    drops as soon as this is built, that matrix running to gigabytes at
    full-brain num_vox.

    max_stat is kept in draw order, not sorted. Sorting is what the
    comparison needs and get_fwer_from_max does it internally; the
    per-draw correspondence is the one thing a discarded matrix leaves
    behind.

    Attributes:
        stat_obs (np.array): (num_reg,) statistic tested per region --
            the source matrix's row 0, copied rather than sliced so the
            result does not pin the matrix alive
        max_stat (np.array): (n_perm+1,) max statistic per draw over
            reg_active, in draw order. max_stat[0] is the observed draw.
            A draw with no finite active region is NaN here and sits out
            of the comparison set.
        reg_active (np.array): (num_reg,) boolean comparison set
        pval (np.array): (num_reg,) FWER p-values, NaN off reg_active
        reg_sig (np.array): (num_reg,) boolean, True where pval <= alpha
    """

    stat_obs: np.ndarray
    max_stat: np.ndarray
    reg_active: np.ndarray
    pval: np.ndarray
    reg_sig: np.ndarray


class Analysis(ABC):
    """Perform effect discovery (GLOW or TFCE) and compute FWER p-values.

    An Analysis is a reusable recipe -- its config knobs only. The
    experiment is not stored; it is passed to fit(), which scales it
    (ExperimentScaled) on the way in, so the same recipe can be fit
    against any experiment.

    Attributes:
        effect_list (list): discovered Effect objects (populated by fit)
        fwer (MaxStatPermResult): the max-stat test the discoveries came
            out of -- p-values, the selected regions, the comparison set
            and the per-draw null (set by fit)
    """

    # the __init__ config knobs that identify the recipe -- the fields __repr__
    # renders. Subclasses declare their own. Never includes exp, the fitted
    # arrays, or fwer -- only the immutable recipe.
    RECORD_FIELDS = ()

    def __init__(self):
        self.effect_list = None
        self.fwer = None

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
    def fit(self, exp, *, n_jobs: int = 1, gpu=False):
        """Run the analysis computation on exp and return self.

        Implementations scale exp with ExperimentScaled.from_exp(exp)
        first (idempotent -- a raw exp is scaled, an already-scaled one
        passes through), then compute.

        n_jobs and gpu are the execution contract every recipe honours, so
        one fit_params dict reaches any of them (see
        glow._extra.benchmark.run.run_ana). Neither changes the result: a
        fit is identical at any n_jobs (permutations are seeded by index)
        and on either device (AnalysisGLOW._fit_gpu draws the same
        permutations in float64), which is why neither enters a recipe's
        RECORD_FIELDS or a benchmark cache key.

        Args:
            exp (Experiment): experiment to analyze.
            n_jobs (int): joblib workers for the permutation walk. 1
                (default) runs in-process; -1 uses all cores.
            gpu: False (default) to stay on the CPU, True to require a
                device, 'auto' to take one when visible. Only AnalysisGLOW
                has a backend; the rest reject an explicit True
                (reject_gpu).

        Returns:
            self
        """

    @classmethod
    def get_fwer(cls, stat, *, alpha: float, reg_active=None):
        """Run the Westfall-Young max-stat test over the active regions.

        Westfall & Young 1993: the max-statistic null over a comparison
        set fixed in advance controls the family-wise error rate. Each
        region's p-value is the share of per-draw maxima at least as
        large as its observed statistic.

        The observed draw is one of its own null draws: row 0 enters the
        per-draw maxima on equal footing with rows 1:, so the denominator
        is n_perm+1 and the smallest attainable p-value is 1/(n_perm+1)
        rather than 0 (Phipson & Smyth 2010). That is not a guard against
        zero but the randomization argument itself -- the identity
        permutation belongs to the permutation group, so under H0 the
        observed statistic is exchangeable with the permuted ones and its
        rank among all n_perm+1 of them is uniform, which is what makes
        the test exact (Lehmann & Romano Thm 15.2.1; Hemerik & Goeman
        2018). It is the same exchangeability z_score_stat relies on to
        put row 0 inside the standardizing moments.

        reg_active is the comparison set and must be fixed with respect
        to the permutation group -- known a priori, or read off data the
        permutations never touch (GLOW's size >= min_vox comes from the
        segmentation fold). An inactive region leaves both the per-draw
        maxima and the tested family, so a set chosen from the observed
        statistics voids FWER control silently: discarding whatever looks
        null lowers the maxima the survivors are compared against.

        Args:
            stat (np.array): (n_perm+1, num_reg) statistics per region.
                Row 0 is the observed draw, rows 1: the permutation null.
            alpha (float): family-wise error rate, the reg_sig cutoff.
            reg_active (np.array): (num_reg,) boolean comparison set.
                Defaults to every region.

        Returns:
            MaxStatPermResult: see the class docstring.
        """
        # a copy, not the row-0 view: the result outlives stat, and a view
        # would hold the whole (n_perm+1, num_reg) matrix alive for one row
        stat_obs = np.array(stat[0])
        num_reg = stat_obs.shape[0]

        if reg_active is None:
            reg_active = np.ones(num_reg, dtype=bool)

        # a draw with no finite active region has no max to contribute;
        # nanmax reports NaN (and warns), and NaN would sort to the top of
        # the null and silently raise every p-value
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            max_stat = (np.nanmax(stat[:, reg_active], axis=1)
                        if reg_active.any()
                        else np.full(stat.shape[0], fill_value=np.nan))

        return cls.get_fwer_from_max(stat_obs, max_stat, alpha=alpha,
                                     reg_active=reg_active)

    @classmethod
    def get_fwer_from_max(cls, stat_obs, max_stat, *, alpha: float,
                          reg_active=None):
        """Assemble a max-stat test from an observed row and its null.

        The primitive get_fwer reduces to, and the way in for a null that
        cannot be read off an (n_perm+1, num_reg) matrix: CET tests
        cluster sizes, and its clusters reform in every draw, so it
        accumulates max_stat a draw at a time rather than ever holding
        the matrix (AnalysisCET._get_fwer_cet). Both arms share this
        bisect so they cannot drift onto different p-value conventions --
        see get_fwer for which convention, and why.

        Args:
            stat_obs (np.array): (num_reg,) observed statistic per region.
            max_stat (np.array): (n_perm+1,) max statistic per draw over
                reg_active, draw order, entry 0 the observed draw.
            alpha (float): family-wise error rate, the reg_sig cutoff.
            reg_active (np.array): (num_reg,) boolean comparison set.
                Defaults to every region.

        Returns:
            MaxStatPermResult: see the class docstring.
        """
        num_reg = stat_obs.shape[0]
        if reg_active is None:
            reg_active = np.ones(num_reg, dtype=bool)

        null_sorted = np.sort(max_stat[np.isfinite(max_stat)])
        n_null = len(null_sorted)

        # only a finite observed statistic is tested, matching what may
        # enter the null. n_null is then nonzero whenever anything is
        # tested at all, the observed draw's own max being one of those
        # entries, so the guard below only skips empty work.
        pval = np.full(num_reg, fill_value=np.nan)
        reg_test = reg_active & np.isfinite(stat_obs)
        if n_null:
            # side='left' counts the strictly smaller maxima, so
            # n_null - k is the count at least as large as the observed.
            # Spelled (n_null - k) / n_null, never 1 - k / n_null: the
            # latter cancels, and a p-value that should land exactly on
            # alpha comes back an ulp above it and fails the reg_sig
            # cutoff.
            k = np.searchsorted(null_sorted, stat_obs[reg_test], side='left')
            pval[reg_test] = np.maximum((n_null - k) / n_null, 1 / n_null)

        # NaN <= alpha is False, so a region off the comparison set is
        # never selected
        return MaxStatPermResult(stat_obs=stat_obs, max_stat=max_stat,
                                 reg_active=reg_active, pval=pval,
                                 reg_sig=pval <= alpha)

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
            mu (np.array): (num_vox,) per-voxel mean over the rows
            std (np.array): (num_vox,) per-voxel std over the rows, as
                measured -- a degenerate column keeps its 0 or NaN here
                and only the divisor is replaced
        """
        # a dropped voxel is NaN in every row, which nanmean / nanstd
        # report on rather than skip; the column is meant to stay NaN
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mu = np.nanmean(stat, axis=0)
            std = np.nanstd(stat, axis=0, ddof=1)
        # negated so the NaN std of an all-NaN column lands here too:
        # NaN > x is False, where NaN < x would have been False as well
        # and left the division to propagate it as a warning
        denom = np.where(std > 1e-12, std, 1.0)
        return (stat - mu) / denom, mu, std

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
        count against self.n_perm_fwer and takes it as the walk's output.

        Row k depends only on k (exp.permute(k) is seeded by k), so the
        matrix is identical regardless of n_jobs.

        A region with no usable statistic is left NaN, which every reader
        of this matrix skips: nanmean / nanstd in z_score_stat, the
        sub-threshold blank in apply_tfce_stat, nanquantile for the CET
        threshold, nanmax and the isfinite guard in get_fwer. Nothing here
        repairs it, because there is nothing left to repair -- no stat
        function can return +-inf (see mancova) and the voxels with no
        variance to test are gone before an analysis sees the data
        (Experiment.drop_constant_vox).

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
