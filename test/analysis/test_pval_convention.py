"""The p-value convention every arm shares, pinned by rank enumeration.

Two defects hid in this arithmetic, and they cancelled each other at one
permutation count -- the benchmark's own (config.N_PERM_FWER = 500) --
so no rejection-rate test could see either. CET dropped the observed
draw from its null while VBA and GLOW kept it, and both spelled the tail
1 - k/n, whose cancellation pushes a p-value that should land exactly on
alpha an ulp above it. Each is worth about 1.2x in size, and the rate
test in test/test_fwer_calibration.py is sized for 1.5x-2x, so it stayed
green throughout.

These take the rate exactly rather than estimating it. Under H0 the
observed draw is exchangeable with the n_perm permuted ones, so its rank
among all n_perm+1 per-draw maxima is uniform (see glow.analysis.fwer);
sweeping that rank over every value it can take enumerates the test's
null distribution instead of sampling it. The true FWER is then a closed
form, floor(alpha * n) / n, and a one-rank error is a hard failure.
"""
import numpy as np
import pytest

from glow.analysis import AnalysisCET, MaxStatPerm


def _fwer_at_rank(rank: int, n: int, alpha: float):
    """Run the max-stat test with the observed draw at a given rank.

    Builds n distinct per-draw maxima, puts the observed draw's max at
    rank (1 being the largest), and tests a one-region family whose
    observed statistic is that max -- the configuration a real fit is in
    for its top region.

    Args:
        rank (int): the observed draw's rank among all n maxima, 1-based.
        n (int): draws counting the observed, i.e. n_perm + 1.
        alpha (float): family-wise error rate.

    Returns:
        MaxStatPerm: the one-region test at that rank.
    """
    value = np.arange(n, dtype=float)[::-1]
    obs = value[rank - 1]
    # draw 0 is the observed one, so its max leads max_stat
    max_stat = np.concatenate([[obs], np.delete(value, rank - 1)])
    return MaxStatPerm.from_max(np.array([obs]), max_stat, alpha=alpha)


@pytest.mark.parametrize('n_perm', [49, 99, 500, 999, 1000])
def test_true_fwer_is_the_exact_randomization_rate(n_perm):
    """Sweeping the observed rank rejects floor(alpha * n) times, no more.

    n_perm = 49 is what test_fwer_calibration.py runs and 500 the
    benchmark's config.N_PERM_FWER. 99 and 999 put alpha * n exactly on
    an integer, which is the whole point of choosing such a count -- and
    where the 1 - k/n spelling used to drop the last rejection rank.
    """
    alpha = 0.05
    n = n_perm + 1
    n_reject = sum(bool(_fwer_at_rank(rank, n, alpha).reg_sig[0])
                   for rank in range(1, n + 1))

    assert n_reject == int(np.floor(alpha * n))
    assert n_reject / n <= alpha


def test_a_boundary_pval_compares_equal_to_alpha():
    """A p-value of exactly alpha must clear the cutoff, not miss by an ulp.

    At n = 100 the rank-5 region's p-value is 5/100. Spelled 1 - 95/100
    it is 0.050000000000000044, which fails pval <= alpha and quietly
    costs the boundary rejection rank; spelled (100 - 95)/100 it is 0.05.
    """
    res = _fwer_at_rank(rank=5, n=100, alpha=0.05)

    assert res.pval[0] == 0.05
    assert res.reg_sig[0]


def test_the_observed_draw_is_one_of_its_own_null_draws():
    """The denominator is n_perm+1, so no p-value can reach 0.

    Phipson & Smyth 2010. The identity permutation belongs to the
    permutation group, so the observed draw's max sits in the null
    alongside the permuted ones and the floor is 1/(n_perm+1).
    """
    # three permuted draws plus the observed, whose region beats them all
    stat = np.array([[10.0], [1.0], [2.0], [3.0]])
    res = MaxStatPerm.from_stat(stat, alpha=0.05)

    assert res.max_stat.shape == (4,)
    assert res.pval[0] == 1 / 4


def test_an_inactive_region_leaves_the_maxima_too():
    """Discarding a region drops it from the null, not just the family.

    A region barred from a p-value but still allowed to set a draw's
    maximum would raise the bar for every region that survived. The
    comparison set is one set, used for both halves of the test.
    """
    # region 1 is the largest thing in every draw, and discarded
    stat = np.array([[1.0, 9.0], [0.5, 8.0], [0.6, 7.0], [0.7, 6.0]])
    res = MaxStatPerm.from_stat(stat, alpha=0.05,
                            reg_active=np.array([True, False]))

    np.testing.assert_allclose(res.max_stat, [1.0, 0.5, 0.6, 0.7])
    assert np.isnan(res.pval[1])
    assert res.pval[0] == 1 / 4


def test_cet_shares_the_convention():
    """CET's null keeps the observed draw, like every other arm.

    Its comparison runs through MaxStatPerm.from_max, so what is
    pinned here is the wiring rather than a second implementation: one
    null entry per draw, the observed leading them in draw order, and
    the same 1/(n_perm+1) floor.
    """
    mask_idx = np.arange(1000).reshape((10, 10, 10))
    stat = np.zeros((201, 1000))
    # the observed draw clears the threshold over a 60-voxel slab
    stat[0, 400:460] = 1.0
    # every permuted draw clears it on one voxel, so the null max is 1
    stat[1:, 0] = 1.0

    res = AnalysisCET._get_fwer_cet(stat, mask_idx, cft=0.5, alpha=0.05)

    assert res.max_stat.shape == (201,)
    assert res.max_stat[0] == 60
    np.testing.assert_array_equal(res.max_stat[1:], np.ones(200))
    assert np.nanmin(res.pval) == 1 / 201
