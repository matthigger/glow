"""Statistical reliability tests for the core analysis pipeline.

Verifies:
- FWER control under the null for VBA and CET (with and without z-scoring)
- z_score_stat standardization correctness
- Power monotonicity: stronger effects detected more often
- Permutation exchangeability: observed rank uniform under H0

Small synthetic experiments keep runtime manageable.
"""

import numpy as np
import pytest
from scipy import stats as sp_stats

from glow.effect import ExtenterSphere, EffectSynthetic
from glow.experiment import Experiment
from glow.analysis import (
    Analysis, AnalysisVBA, AnalysisCET,
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
        ana = cls(exp, n_perm_fwer=n_perm, alpha_fwer=alpha, **kw).fit()
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
# VBA+z FWER at small n_perm — tighter regression than TestFWERCalibration
# ---------------------------------------------------------------------------

class TestVbaZFwerSmallNPerm:
    """VBA+z must control FWER at small n_perm.

    Historical context: when ``z_score_stat`` computed mu/sigma from the
    null rows (1:) only, the observed row (0) was standardized by
    parameters it did not contribute to — an asymmetry that broke
    exchangeability at finite B and inflated rejection to ~11% at B=49
    / ~6–8% at B=250 against a nominal 5%. Moving mu/sigma to all rows
    (Phipson & Smyth 2010; Winkler et al. 2014) restored exchangeability.

    This test uses a tighter budget than ``TestFWERCalibration.test_vba_z``
    (which allows up to 20%) so that any regression reintroducing the
    small-B drift would fail loudly here.

    Gated by ``--runslow`` (K=200 trials × 49 perms).
    """

    K = 200
    N_PERM = 49
    ALPHA = 0.05

    # K=200, true rate ≈ ALPHA: 99% binomial upper bound ≈ 0.09
    MAX_RATE = 0.09

    @pytest.mark.slow
    def test_vba_z_controls_fwer(self):
        """At n_perm=49, VBA+z rejection stays within binomial CI of nominal."""
        rate = _null_rejection_rate(
            AnalysisVBA, self.K, self.N_PERM, self.ALPHA, z_flag=True)
        assert rate <= self.MAX_RATE, (
            f'VBA+z rate {rate:.3f} exceeds {self.MAX_RATE} — exchangeability '
            f'drift may have reappeared (check z_score_stat includes observed '
            f'row in mu/sigma)')

    @pytest.mark.slow
    def test_vba_no_z_controls_fwer(self):
        """Reference: same setup without z-scoring also controls FWER."""
        rate = _null_rejection_rate(
            AnalysisVBA, self.K, self.N_PERM, self.ALPHA, z_flag=False)
        assert rate <= self.MAX_RATE, (
            f'VBA (no z) rate {rate:.3f} exceeds {self.MAX_RATE}')


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
        """Extreme observed row inflates sigma, deflating its own z."""
        stat = np.ones((51, 10))
        stat[0, :] = 100  # extreme observed row

        z = Analysis.z_score_stat(stat)

        # all-rows mean = (100 + 50*1) / 51 ≈ 2.94
        # all-rows std dominated by the single outlier — observed z is
        # bounded by sqrt(n-1) ≈ 7.07 for a single-outlier column
        assert np.all(z[0, :] < 10), \
            'Observed z too large — observed row not contributing to sigma'
        # null rows contribute symmetrically, so their z is small (< 1)
        assert np.all(np.abs(z[1:, :]) < 1), \
            'Null z too large — unexpected scale'

    def test_heterogeneous_voxels_equalized(self):
        """Voxels with 10x different std should all have std=1 after."""
        rng = np.random.default_rng(42)
        stat = rng.standard_normal((201, 50))
        stat[:, :10] *= 10

        z = Analysis.z_score_stat(stat)
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
                    exp = EffectSynthetic(
                        extenter=ExtenterSphere(radius=2, seed=seed),
                        effect_llr=llr).fit(exp)[0]
                ana = AnalysisVBA(exp, n_perm_fwer=n_perm,
                                  alpha_fwer=alpha).fit()
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
            ana = AnalysisVBA(exp, n_perm_fwer=n_perm, get_stat=get_wilks)
            num_vox = exp.y.shape[2]
            stat = np.full((n_perm + 1, num_vox), np.nan)
            for k in range(n_perm + 1):
                _exp = exp.permute(k) if k else exp
                stat[k, :] = ana.get_stat_perm(_exp)

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
