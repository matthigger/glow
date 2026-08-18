"""The Westfall-Young max-stat permutation test, as one result object.

Westfall & Young 1993: the max-statistic null over a comparison set fixed
in advance controls the family-wise error rate. Each region's p-value is
the share of per-draw maxima at least as large as its observed statistic.

The observed draw is one of its own null draws: row 0 enters the per-draw
maxima with rows 1:, so the denominator is n_perm+1 and the smallest
attainable p-value is 1/(n_perm+1) (Phipson & Smyth 2010). That is the
randomization argument, not a guard against zero -- the identity
permutation belongs to the group, so under H0 the observed statistic's rank
among all n_perm+1 draws is uniform and the test is exact (Lehmann &
Romano Thm 15.2.1; Hemerik & Goeman 2018).

The comparison set must be fixed with respect to the permutation group --
known a priori, or read off data the permutations never touch. An inactive
region leaves both the per-draw maxima and the tested family, so choosing
the set from the observed statistics voids FWER control silently: dropping
whatever looks null lowers the maxima the survivors face.
"""

import warnings
from dataclasses import dataclass

import numpy as np


def max_over_active(stat, reg_active):
    """Reduce a stat matrix to one max per draw over the comparison set.

    The only place the max-stat convention lives: an inactive region leaves
    the family, and a draw with no finite active region contributes NaN
    rather than a number (from_max drops it, where a -inf or a 0 would move
    every p-value). Streaming backends reproduce this per chunk.

    Args:
        stat (np.array): (n_perm+1, num_reg) statistics per region
        reg_active (np.array): (num_reg,) boolean comparison set

    Returns:
        max_stat (np.array): (n_perm+1,) max over reg_active per draw, in
            draw order, NaN for a draw with no finite active region
    """
    # nanmax warns on an all-NaN slice and reports NaN, which is what the
    # convention wants; an empty comparison set has no slice to take at all
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if not reg_active.any():
            return np.full(stat.shape[0], fill_value=np.nan)
        return np.nanmax(stat[:, reg_active], axis=1)


# eq=False: a generated __eq__ compares array fields pairwise and raises on
# the ambiguous truth value; nothing compares two results anyway
@dataclass(frozen=True, eq=False)
class MaxStatPerm:
    """One max-stat permutation test: what went in, and what it decided.

    Self-contained: pval and reg_sig follow from stat_obs, max_stat,
    reg_active and alpha, so a stored result stays checkable once the
    (n_perm+1, num_reg) matrix behind it is gone.

    max_stat is kept in draw order, not sorted (from_max sorts internally):
    the per-draw correspondence is the one thing a discarded matrix leaves
    behind. frozen stops the fields being rebound, not the arrays being
    written into.

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

        The way in for an arm whose whole family sits in one matrix. See
        the module docstring for the convention, and for what reg_active
        must satisfy.

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

        return cls.from_max(stat_obs, max_over_active(stat, reg_active),
                            alpha=alpha, reg_active=reg_active)

    @classmethod
    def from_max(cls, stat_obs, max_stat, *, alpha: float, reg_active=None):
        """Test an observed row against a null accumulated per draw.

        The primitive from_stat reduces to, and the way in for a null that
        cannot be read off one matrix -- CET's clusters reform in every
        draw, so it accumulates max_stat a draw at a time. Every arm
        arrives here, so none can drift onto its own p-value convention.

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
        # enter the null; the guard below therefore only skips empty work
        pval = np.full(num_reg, fill_value=np.nan)
        reg_test = reg_active & np.isfinite(stat_obs)
        if n_null:
            # side='left' counts the strictly smaller maxima, so n_null -
            # k is the count at least as large as the observed. Spelled
            # (n_null - k) / n_null, never 1 - k / n_null, which cancels
            # and lands an exactly-alpha p-value an ulp above the cutoff.
            k = np.searchsorted(null_sorted, stat_obs[reg_test], side='left')
            pval[reg_test] = np.maximum((n_null - k) / n_null, 1 / n_null)

        # NaN <= alpha is False, so a region off the comparison set is
        # never selected
        return cls(stat_obs=stat_obs, max_stat=max_stat,
                   reg_active=reg_active, alpha=alpha, pval=pval,
                   reg_sig=pval <= alpha)
