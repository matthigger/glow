"""Statistical reliability tests for the core analysis pipeline.

Verifies:
- FWER control under the null for VBA and CET (with and without z-scoring)
- z_score_stat standardization correctness
- Power monotonicity: stronger effects detected more often
- Size adjustment: GLOW regression decorrelates stat from region size
- Permutation exchangeability: observed rank uniform under H0

Small synthetic experiments keep runtime manageable.
"""

import numpy as np
from scipy import stats as sp_stats

from glow.effect import ExtenterSphere
from glow.experiment import Experiment
from glow.analysis import (
    Analysis, AnalysisVBA, AnalysisCET, AnalysisGLOW,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _null_rejection_rate(cls, K, n_perm, alpha, **kw):
    """Fraction of K null experiments where H0 is rejected."""
    hits = 0
    for seed in range(K):
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                    num_img=30, seed=seed)
        ana = cls(exp, n_perm_fwer=n_perm, alpha_fwer=alpha, **kw)
        if len(ana.effect_list) > 0:
            hits += 1
    return hits / K


# ---------------------------------------------------------------------------
# FWER calibration under the null
# ---------------------------------------------------------------------------

class TestFWERCalibration:
    """Under H0, rejection rate must be close to alpha.

    K=40 null seeds, 49 permutations (50 total rows), alpha=0.05.
    Exact FWER with discrete p-values ~ 0.04.  Threshold at 0.20 catches
    any systematic inflation at <0.1% false-alarm probability.

    P(rate > 0.20 | true_rate=0.04, K=40) < 0.001
    P(rate > 0.20 | true_rate=0.30, K=40) > 0.90   (catches 30% inflation)
    """

    K = 40
    N_PERM = 49
    ALPHA = 0.05
    MAX_RATE = 0.20

    def _check(self, rate, label):
        assert rate <= self.MAX_RATE, (
            f'{label} FWER inflated: {rate:.0%} '
            f'({int(rate * self.K)}/{self.K} rejected, '
            f'expected ~ {self.ALPHA:.0%})')

    def test_vba(self):
        self._check(
            _null_rejection_rate(AnalysisVBA, self.K, self.N_PERM,
                                 self.ALPHA),
            'VBA')

    def test_cet(self):
        self._check(
            _null_rejection_rate(AnalysisCET, self.K, self.N_PERM,
                                 self.ALPHA),
            'CET')

    def test_vba_z(self):
        """VBA with z-scoring should also control FWER."""
        self._check(
            _null_rejection_rate(AnalysisVBA, self.K, self.N_PERM,
                                 self.ALPHA, z_flag=True),
            'VBA+z')

    def test_cet_z(self):
        """CET with z-scoring should also control FWER."""
        self._check(
            _null_rejection_rate(AnalysisCET, self.K, self.N_PERM,
                                 self.ALPHA, z_flag=True),
            'CET+z')


# ---------------------------------------------------------------------------
# z-score standardization
# ---------------------------------------------------------------------------

class TestZScoreStatReliability:
    """z_score_stat must use null rows only and equalize voxel variances."""

    def test_observed_excluded_from_standardization(self):
        """Extreme observed row must not affect null z-scores."""
        stat = np.ones((51, 10))
        stat[0, :] = 100  # extreme observed row

        z = Analysis.z_score_stat(stat)

        # null rows are constant → sigma clipped to 1 → z = (1-1)/1 = 0
        np.testing.assert_allclose(z[1:, :], 0, atol=1e-12)
        # observed z should reflect its extremity: (100-1)/1 = 99
        assert np.all(z[0, :] > 50), \
            'Observed z too small — may include observed row in stats'

    def test_heterogeneous_null_equalized(self):
        """Voxels with 10x different null std should all have std=1 after."""
        rng = np.random.default_rng(42)
        null = rng.standard_normal((200, 50))
        null[:, :10] *= 10

        obs = rng.standard_normal((1, 50))
        obs[:, :10] *= 10
        stat = np.vstack([obs, null])

        z = Analysis.z_score_stat(stat)
        null_stds = z[1:, :].std(axis=0, ddof=1)
        np.testing.assert_allclose(null_stds, 1.0, atol=1e-10)

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
# Power monotonicity
# ---------------------------------------------------------------------------

class TestPowerMonotonicity:
    """Detection rate should increase with effect strength."""

    def test_vba_power_increases(self):
        K = 20
        n_perm = 25
        alpha = 0.1

        rates = {}
        for llr in [0.0, 0.3, 0.8]:
            hits = 0
            for seed in range(K):
                exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                            num_img=50, seed=seed)
                if llr > 0:
                    exp, _ = exp.impose_effect(
                        seed=seed,
                        extenter=ExtenterSphere(radius=2),
                        effect_llr=llr)
                ana = AnalysisVBA(exp, n_perm_fwer=n_perm,
                                  alpha_fwer=alpha)
                if len(ana.effect_list) > 0:
                    hits += 1
            rates[llr] = hits / K

        # null should rarely reject
        assert rates[0.0] <= 0.30, f'Null rate too high: {rates[0.0]}'
        # power should increase with effect size
        assert rates[0.8] >= rates[0.3], (
            f'Power not monotonic: {rates}')
        # strong effect should be detected reliably
        assert rates[0.8] >= 0.5, (
            f'Strong effect power too low: {rates[0.8]}')


# ---------------------------------------------------------------------------
# GLOW size adjustment
# ---------------------------------------------------------------------------

class TestSizeAdjustment:
    """GLOW regression should remove the size-stat confound."""

    def test_adjustment_reduces_size_correlation(self):
        """Raw LLR correlates with size; adjustment should reduce this."""
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                    num_img=50, seed=42)
        ana = AnalysisGLOW(exp, n_perm_fwer=10, alpha_fwer=0.05,
                           n_perm_fwer_size_adjust=25)

        valid = (np.isfinite(ana.stat)
                 & np.isfinite(ana.llr_adjusted_0)
                 & (ana.size > 0))

        r_raw, _ = sp_stats.pearsonr(np.log(ana.size[valid]),
                                      ana.stat[valid])
        r_adj, _ = sp_stats.pearsonr(np.log(ana.size[valid]),
                                      ana.llr_adjusted_0[valid])

        # raw LLR should positively correlate with size (the confound)
        assert r_raw > 0.3, (
            f'Expected positive raw size-stat correlation, got r={r_raw:.3f}')
        # adjustment should reduce correlation
        assert abs(r_adj) < abs(r_raw), (
            f'Adjustment did not reduce correlation: '
            f'raw={r_raw:.3f}, adj={r_adj:.3f}')
        assert abs(r_adj) < 0.5, (
            f'Adjusted correlation still too high: r={r_adj:.3f}')

    def test_predict_null_mean_monotone(self):
        """Predicted null mean should increase with region size for LLR."""
        sizes = np.array([1, 5, 10, 50, 100, 500])
        beta = np.array([0.5, 0.3])
        pred = AnalysisGLOW.predict_null_mean(sizes, 'power_law', beta)
        assert np.all(np.diff(pred) > 0), (
            f'Predicted null mean not increasing with size: {pred}')

    def test_accumulate_regression_symmetric(self):
        """Accumulating in one batch vs two should give same result."""
        rng = np.random.default_rng(7)
        sizes = rng.integers(1, 100, size=200).astype(float)
        stats = 0.5 * np.log(sizes) + rng.standard_normal(200) * 0.3

        # one-shot
        XtX_1, Xty_1 = AnalysisGLOW.accumulate_regression(
            sizes, stats, 'power_law')

        # two-batch
        XtX_2, Xty_2 = AnalysisGLOW.accumulate_regression(
            sizes[:100], stats[:100], 'power_law')
        XtX_2, Xty_2 = AnalysisGLOW.accumulate_regression(
            sizes[100:], stats[100:], 'power_law', XtX_2, Xty_2)

        np.testing.assert_allclose(XtX_1, XtX_2)
        np.testing.assert_allclose(Xty_1, Xty_2)


# ---------------------------------------------------------------------------
# Permutation exchangeability
# ---------------------------------------------------------------------------

class TestPermutationExchangeability:
    """Under H0, observed max-stat rank should be approximately uniform."""

    def test_observed_rank_uniform(self):
        """Rank of observed max-stat among all permutations ~ Uniform."""
        K = 50
        n_perm = 49
        ranks = []

        for seed in range(K):
            exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                        num_img=30, seed=seed)
            from glow.analysis.mancova import get_wilks
            ana = Analysis(exp, get_stat=get_wilks)
            stat = ana.get_stat_perm(exp, n_perm=n_perm)

            max_stats = np.nanmax(stat, axis=1)
            # rank = number of permutations with max-stat >= observed
            rank = int(np.sum(max_stats >= max_stats[0]))
            ranks.append(rank)

        # Normalize to [0,1] for KS test against Uniform
        ranks_norm = (np.array(ranks) - 0.5) / (n_perm + 1)
        _, ks_p = sp_stats.kstest(ranks_norm, 'uniform')

        assert ks_p > 0.01, (
            f'Observed max-stat rank not uniform (KS p={ks_p:.3f}), '
            f'suggesting broken exchangeability')


# ---------------------------------------------------------------------------
# get_pval monotonicity
# ---------------------------------------------------------------------------

class TestGetPvalProperties:
    """Properties that get_pval must satisfy for any valid null."""

    def test_monotone_in_observed_stat(self):
        """Higher observed stat -> lower (or equal) p-value."""
        rng = np.random.default_rng(0)
        stat = rng.standard_normal((51, 20))
        stat[0, :] = np.linspace(0, 5, 20)

        pval = Analysis.get_pval(stat)

        for i in range(len(pval) - 1):
            assert pval[i] >= pval[i + 1] - 1e-10, (
                f'p-value not monotone: pval[{i}]={pval[i]:.4f} < '
                f'pval[{i + 1}]={pval[i + 1]:.4f}')

    def test_all_nan_region_gives_nan_pval(self):
        """Regions that are entirely NaN should get NaN p-values."""
        stat = np.random.default_rng(0).standard_normal((10, 5))
        stat[:, 2] = np.nan

        pval = Analysis.get_pval(stat)
        assert np.isnan(pval[2])
        assert not np.isnan(pval[0])

    def test_inactive_regions_get_nan(self):
        """Inactive regions should get NaN, not spurious p-values."""
        stat = np.random.default_rng(0).standard_normal((10, 5))
        reg_active = np.array([True, True, False, True, False])

        pval = Analysis.get_pval(stat, reg_active=reg_active)
        assert np.isnan(pval[2]) and np.isnan(pval[4])
        assert not np.isnan(pval[0]) and not np.isnan(pval[1])
