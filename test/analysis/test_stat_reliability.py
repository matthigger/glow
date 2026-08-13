"""Property tests for the two reductions every arm shares.

Verifies:
- z_score_stat standardizes per voxel, observed row included
- get_fwer is monotone in the observed stat and propagates NaN

Both are pure array math, checked against closed forms on hand-built
inputs -- no experiment is fit here. The rate-based claims that used to
live alongside them (FWER calibration, exchangeability) moved to
test/test_fwer_calibration.py, which needs enough null trials to mean
anything and so runs only under --runslow.
"""

import numpy as np

from glow.analysis import Analysis


# ---------------------------------------------------------------------------
# z-score standardization
# ---------------------------------------------------------------------------

class TestZScoreStatReliability:
    """z_score_stat uses all rows (observed + null) to equalize voxel scales.

    Including the observed row in mu/sigma preserves exchangeability at
    finite B — each row is standardized by parameters it contributed to.
    Phipson & Smyth (2010) / Winkler et al. (2014).
    """

    def test_observed_contributes_to_standardization(self):
        """Extreme observed row inflates sigma, deflating its own z.

        A column of n rows holding one outlier and n-1 equal values has a
        closed-form z regardless of how extreme the outlier is. Writing the
        gap as d, the mean sits at v + d/n, so deviations are d(n-1)/n at
        the outlier and -d/n elsewhere; the sum of squares is d^2 (n-1)/n
        and the ddof=1 std is therefore d/sqrt(n). Both z's lose d:

            z0 = (n-1)/sqrt(n),   z_null = -1/sqrt(n)

        That independence from d is the property under test -- the observed
        row cannot outrun a sigma it contributes to, however extreme it is.
        """
        n = 51
        stat = np.ones((n, 10))
        stat[0, :] = 100

        z, _, _ = Analysis.z_score_stat(stat)

        z0_exp = (n - 1) / np.sqrt(n)
        z_null_exp = -1 / np.sqrt(n)
        np.testing.assert_allclose(z[0, :], z0_exp, rtol=1e-12)
        np.testing.assert_allclose(z[1:, :], z_null_exp, rtol=1e-12)

        # a 100x larger outlier lands on exactly the same z
        stat_bigger = np.ones((n, 10))
        stat_bigger[0, :] = 10_000
        np.testing.assert_allclose(Analysis.z_score_stat(stat_bigger)[0], z,
                                   rtol=1e-12)

    def test_heterogeneous_voxels_equalized(self):
        """Voxels with 10x different std should all have std=1 after."""
        rng = np.random.default_rng(42)
        stat = rng.standard_normal((201, 50))
        stat[:, :10] *= 10

        z, _, _ = Analysis.z_score_stat(stat)
        stds = z.std(axis=0, ddof=1)
        np.testing.assert_allclose(stds, 1.0, atol=1e-10)

    def test_wrong_axis_leaves_heterogeneity(self):
        """Normalizing across voxels (wrong) does NOT equalize per-voxel stds.

        Demonstrates the specific failure mode that correct axis=0
        standardization avoids.
        """
        rng = np.random.default_rng(42)
        null = rng.standard_normal((200, 50))
        null[:, :10] *= 10
        stat = np.vstack([rng.standard_normal((1, 50)), null])

        # wrong: normalize each row across voxels (axis=1)
        mu = stat.mean(axis=1, keepdims=True)
        sigma = stat.std(axis=1, keepdims=True, ddof=1)
        sigma[sigma < 1e-12] = 1.0
        z_wrong = (stat - mu) / sigma

        # per-voxel null stds should remain heterogeneous
        stds = z_wrong[1:, :].std(axis=0, ddof=1)
        ratio = stds[:10].mean() / stds[10:].mean()
        assert ratio > 3, (
            f'Wrong-axis z-score should leave heterogeneous stds '
            f'(ratio={ratio:.1f}, expected > 3)')


# ---------------------------------------------------------------------------
# get_fwer monotonicity
# ---------------------------------------------------------------------------

class TestGetFwerProperties:
    """Properties that get_fwer must satisfy for any valid null."""

    def test_monotone_in_observed_stat(self):
        """Higher observed stat -> lower (or equal) p-value."""
        rng = np.random.default_rng(0)
        stat = rng.standard_normal((51, 20))
        stat[0, :] = np.linspace(0, 5, 20)

        pval = Analysis.get_fwer(stat, alpha=.05).pval

        for i in range(len(pval) - 1):
            assert pval[i] >= pval[i + 1] - 1e-10, (
                f'p-value not monotone: pval[{i}]={pval[i]:.4f} < '
                f'pval[{i + 1}]={pval[i + 1]:.4f}')

    def test_all_nan_region_gives_nan_pval(self):
        """Regions that are entirely NaN should get NaN p-values."""
        stat = np.random.default_rng(0).standard_normal((10, 5))
        stat[:, 2] = np.nan

        pval = Analysis.get_fwer(stat, alpha=.05).pval
        assert np.isnan(pval[2])
        assert not np.isnan(pval[0])

    def test_inactive_regions_get_nan(self):
        """Inactive regions should get NaN, not spurious p-values."""
        stat = np.random.default_rng(0).standard_normal((10, 5))
        reg_active = np.array([True, True, False, True, False])

        pval = Analysis.get_fwer(stat, reg_active=reg_active, alpha=.05).pval
        assert np.isnan(pval[2]) and np.isnan(pval[4])
        assert not np.isnan(pval[0]) and not np.isnan(pval[1])
