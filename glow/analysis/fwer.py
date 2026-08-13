"""The Westfall-Young max-stat permutation test, as one result object.

Westfall & Young 1993: the max-statistic null over a comparison set fixed
in advance controls the family-wise error rate. Each region's p-value is
the share of per-draw maxima at least as large as its observed statistic.

The observed draw is one of its own null draws -- row 0 enters the
per-draw maxima on equal footing with rows 1:, so the denominator is
n_perm+1 and the smallest attainable p-value is 1/(n_perm+1) rather than
0 (Phipson & Smyth 2010). That is not a guard against zero but the
randomization argument itself: the identity permutation belongs to the
permutation group, so under H0 the observed statistic is exchangeable
with the permuted ones and its rank among all n_perm+1 of them is
uniform, which is what makes the test exact (Lehmann & Romano
Thm 15.2.1; Hemerik & Goeman 2018). It is the same exchangeability
Analysis.z_score_stat leans on to put row 0 inside the standardizing
moments.

The comparison set must be fixed with respect to the permutation group --
known a priori, or read off data the permutations never touch (GLOW's
size >= min_vox comes from the segmentation fold). An inactive region
leaves both the per-draw maxima and the tested family, so a set chosen
from the observed statistics voids FWER control silently: discarding
whatever looks null lowers the maxima the survivors are compared against.
"""

import warnings
from dataclasses import dataclass

import numpy as np


# eq=False: the generated __eq__ compares fields pairwise, which on array
# fields raises on the ambiguous truth value rather than answering. Nothing
# compares two results, so identity is the honest fallback.
@dataclass(frozen=True, eq=False)
class MaxStatPerm:
    """One max-stat permutation test: what went in, and what it decided.

    Self-contained: pval and reg_sig both follow from stat_obs, max_stat,
    reg_active and alpha, so a stored result stays checkable once the
    (n_perm+1, num_reg) matrix behind it is gone -- which AnalysisGLOW
    drops as soon as this is built, that matrix running to gigabytes at
    full-brain num_vox.

    max_stat is kept in draw order, not sorted. Sorting is what the
    comparison needs and from_max does it internally; the per-draw
    correspondence is the one thing a discarded matrix leaves behind.

    frozen stops the fields being rebound, not the arrays being written
    into -- this is a record, not a deep-immutable value.

    The test (as passed to from_stat / from_max):
        stat_obs (np.array): (num_reg,) statistic tested per region --
            the source matrix's row 0, copied rather than sliced so the
            result does not pin that matrix alive
        max_stat (np.array): (n_perm+1,) max statistic per draw over
            reg_active, in draw order. max_stat[0] is the observed draw.
            A draw with no finite active region is NaN here and sits out
            of the comparison set.
        reg_active (np.array): (num_reg,) boolean comparison set
        alpha (float): family-wise error rate, the reg_sig cutoff

    What it decided:
        pval (np.array): (num_reg,) FWER p-values, NaN off reg_active
        reg_sig (np.array): (num_reg,) boolean, True where pval <= alpha
    """

    stat_obs: np.ndarray
    max_stat: np.ndarray
    reg_active: np.ndarray
    alpha: float
    pval: np.ndarray
    reg_sig: np.ndarray

    @classmethod
    def from_stat(cls, stat, *, alpha: float, reg_active=None):
        """Test every region of a permutation stat matrix.

        The way in for an arm whose whole family sits in one matrix: VBA
        and GLOW both build (n_perm+1, num_reg) and hand it here. See the
        module docstring for the convention, and for what reg_active has
        to satisfy for the result to mean anything.

        Args:
            stat (np.array): (n_perm+1, num_reg) statistics per region.
                Row 0 is the observed draw, rows 1: the permutation null.
            alpha (float): family-wise error rate, the reg_sig cutoff.
            reg_active (np.array): (num_reg,) boolean comparison set.
                Defaults to every region.

        Returns:
            MaxStatPerm: see the class docstring.
        """
        # a copy, not the row-0 view: the result outlives stat, and a view
        # would hold the whole (n_perm+1, num_reg) matrix alive for one row
        stat_obs = np.array(stat[0])

        if reg_active is None:
            reg_active = np.ones(stat_obs.shape[0], dtype=bool)

        # a draw with no finite active region has no max to contribute;
        # nanmax reports NaN (and warns), and NaN would sort to the top of
        # the null and silently raise every p-value
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            max_stat = (np.nanmax(stat[:, reg_active], axis=1)
                        if reg_active.any()
                        else np.full(stat.shape[0], fill_value=np.nan))

        return cls.from_max(stat_obs, max_stat, alpha=alpha,
                            reg_active=reg_active)

    @classmethod
    def from_max(cls, stat_obs, max_stat, *, alpha: float, reg_active=None):
        """Test an observed row against a null accumulated per draw.

        The primitive from_stat reduces to, and the way in for a null that
        cannot be read off one matrix: CET tests cluster sizes, and its
        clusters reform in every draw, so it accumulates max_stat a draw
        at a time rather than ever holding the matrix
        (AnalysisCET._get_fwer_cet). Every arm arrives here, so none can
        drift onto a p-value convention of its own.

        Args:
            stat_obs (np.array): (num_reg,) observed statistic per region.
            max_stat (np.array): (n_perm+1,) max statistic per draw over
                reg_active, draw order, entry 0 the observed draw.
            alpha (float): family-wise error rate, the reg_sig cutoff.
            reg_active (np.array): (num_reg,) boolean comparison set.
                Defaults to every region.

        Returns:
            MaxStatPerm: see the class docstring.
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
        return cls(stat_obs=stat_obs, max_stat=max_stat,
                   reg_active=reg_active, alpha=alpha, pval=pval,
                   reg_sig=pval <= alpha)
