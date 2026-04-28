"""Tests for the two-stage size-adjustment GAM and the studentized
score path in ``AnalysisGLOW``.

Covers:
    T1  Recovery of mu / sigma on heteroscedastic synthetic data.
    T2  Studentized null is approximately homoscedastic across size.
    T3  Reduces to mean-adj when the null is homoscedastic.
    T4  Edge cases: zero residuals at some size do not produce NaN.
    T5  Backward compat: legacy pickles restore as ``score_method='mean_adj'``.
    T6  ``z_score`` analysis runs end-to-end and the threshold differs
        from ``mean_adj`` on heteroscedastic data.
    T7  ``sig_reg_list`` always contains region indices that are a valid
        antichain after pruning (consistency check; doesn't depend on
        whether the test detects an effect).
"""

import warnings

import numpy as np
import pytest

from glow.analysis import AnalysisGLOW
from glow.experiment.exper import Experiment


warnings.filterwarnings('ignore', category=RuntimeWarning)


# ---------------------------------------------------------------------------
# T1, T3, T4: fit_size_gam unit tests on synthetic data
# ---------------------------------------------------------------------------

class TestFitSizeGam:

    def test_recovers_mean_homoscedastic(self):
        """T3a: GAM recovers the conditional mean when sigma is constant."""
        rng = np.random.default_rng(0)
        n = 5000
        size = 10 ** rng.uniform(0, 3, n)
        true_mu = 0.5 * np.log10(size)
        stat = true_mu + rng.normal(0, 1.0, n)

        fit = AnalysisGLOW.fit_size_gam(size, stat, score_method='z_score')

        assert fit.size_adjusted
        assert fit.mu_gam is not None
        assert fit.sigma_gam is not None

        # mu_fn within ±0.1 of the truth across the size range
        check_sizes = np.array([2, 10, 100, 800])
        mu_pred = fit.mu_fn(check_sizes)
        mu_true = 0.5 * np.log10(check_sizes)
        assert np.max(np.abs(mu_pred - mu_true)) < 0.1, \
            f'mu mis-fit: {mu_pred} vs {mu_true}'

    def test_sigma_constant_when_homoscedastic(self):
        """T3b: sigma_fn is approximately constant when null is homoscedastic.

        Absolute value carries the digamma-bias factor (~0.53); we only
        check that the ratio max/min across sizes stays close to 1.
        """
        rng = np.random.default_rng(1)
        n = 5000
        size = 10 ** rng.uniform(0, 3, n)
        stat = 0.5 * np.log10(size) + rng.normal(0, 1.0, n)

        fit = AnalysisGLOW.fit_size_gam(size, stat, score_method='z_score')

        sigmas = fit.sigma_fn(np.array([2, 10, 100, 800]))
        ratio = sigmas.max() / sigmas.min()
        assert ratio < 1.2, \
            f'sigma_fn varies by {ratio:.2f}x on homoscedastic data'

    def test_sigma_tracks_heteroscedastic(self):
        """T1: sigma_fn captures the relative scale of a known
        size-dependent SD (tolerant on the absolute log-bias factor)."""
        rng = np.random.default_rng(2)
        n = 8000
        size = 10 ** rng.uniform(0, 3, n)
        true_sd = 0.5 + 2.5 * (np.log10(size) / 3)  # 0.5 -> 3
        stat = 0.5 * np.log10(size) + rng.normal(0, 1.0, n) * true_sd

        fit = AnalysisGLOW.fit_size_gam(size, stat, score_method='z_score')

        sizes_test = np.array([2, 10, 100, 800])
        sigma_pred = fit.sigma_fn(sizes_test)
        sigma_true = 0.5 + 2.5 * (np.log10(sizes_test) / 3)

        # The ratio (largest size / smallest size) should match the true ratio
        true_ratio = sigma_true[-1] / sigma_true[0]
        pred_ratio = sigma_pred[-1] / sigma_pred[0]
        rel_err = abs(pred_ratio - true_ratio) / true_ratio
        assert rel_err < 0.20, \
            f'sigma ratio mis-fit: pred {pred_ratio:.2f} vs true {true_ratio:.2f}'

    def test_mean_adj_skips_sigma_stage(self):
        """score_method='mean_adj' fits mu_gam only, sigma_gam stays None."""
        rng = np.random.default_rng(3)
        n = 1000
        size = 10 ** rng.uniform(0, 3, n)
        stat = rng.normal(0, 1, n)
        fit = AnalysisGLOW.fit_size_gam(size, stat, score_method='mean_adj')
        assert fit.mu_gam is not None
        assert fit.sigma_gam is None
        # sigma_fn should return ones (identity scale)
        assert np.allclose(fit.sigma_fn(np.array([1, 10, 100])), 1.0)

    def test_zero_residuals_no_nan(self):
        """T4: residuals exactly at the GAM mean don't produce NaN/Inf."""
        rng = np.random.default_rng(4)
        n = 1000
        size = 10 ** rng.uniform(0, 3, n)
        # residuals: half are exactly zero, half are normal
        stat = 0.5 * np.log10(size) + np.where(
            rng.uniform(size=n) < 0.5, 0.0, rng.normal(0, 1, n))
        fit = AnalysisGLOW.fit_size_gam(size, stat, score_method='z_score')

        check_sizes = np.array([2, 10, 100, 800])
        mu = fit.mu_fn(check_sizes)
        sigma = fit.sigma_fn(check_sizes)
        assert np.all(np.isfinite(mu)), 'mu_fn produced non-finite values'
        assert np.all(np.isfinite(sigma)), 'sigma_fn produced non-finite values'
        assert np.all(sigma > 0), 'sigma_fn produced non-positive values'

    def test_invalid_score_method_raises(self):
        size = np.array([1, 10, 100] * 50)
        stat = np.zeros(150)
        with pytest.raises(ValueError):
            AnalysisGLOW.fit_size_gam(size, stat, score_method='nonsense')


# ---------------------------------------------------------------------------
# T2: studentized null is approximately homoscedastic across size
# ---------------------------------------------------------------------------

class TestStudentizedHomoscedasticity:

    def test_studentized_sd_uniform_across_sizes(self):
        """After studentization the residual SD per size band should be ~1
        across the size range (the actual value depends on the bias
        factor, but it should be uniform)."""
        rng = np.random.default_rng(5)
        n = 8000
        size = 10 ** rng.uniform(0, 3, n)
        true_sd = 0.5 + 2.5 * (np.log10(size) / 3)
        stat = 0.5 * np.log10(size) + rng.normal(0, 1.0, n) * true_sd

        fit = AnalysisGLOW.fit_size_gam(size, stat, score_method='z_score')

        z = (stat - fit.mu_fn(size)) / fit.sigma_fn(size)

        # bin by log-size deciles, compute SD in each bin
        log_sz = np.log10(size)
        bins = np.quantile(log_sz, np.linspace(0, 1, 11))
        sds = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (log_sz >= lo) & (log_sz < hi)
            if m.sum() < 30:
                continue
            sds.append(z[m].std())
        sds = np.asarray(sds)

        # uniform within ±25% across deciles (some noise is expected)
        ratio = sds.max() / sds.min()
        assert ratio < 1.5, \
            f'studentized SD varies by {ratio:.2f}x; expected near 1'


# ---------------------------------------------------------------------------
# T5, T6, T7: AnalysisGLOW end-to-end
# ---------------------------------------------------------------------------

class TestAnalysisGlowEndToEnd:

    def _make_exp(self, seed=0):
        return Experiment.from_gauss(b=1, num_img=20, shape=(8, 8), a=2,
                                      seed=seed)

    def test_runs_both_modes(self):
        exp = self._make_exp()
        for sm in ('mean_adj', 'z_score'):
            ana = AnalysisGLOW(exp=exp, n_perm_fwer=20,
                               n_perm_fwer_size_adjust=15,
                               score_method=sm, verbose=False)
            assert ana.score_method == sm
            assert ana.mu_gam is not None
            if sm == 'z_score':
                assert ana.sigma_gam is not None
            else:
                assert ana.sigma_gam is None
            assert ana.adj_crit is not None
            assert np.isfinite(np.nanmin(ana.llr_adjusted_0))

    def test_z_score_threshold_differs(self):
        """The FWER threshold should differ in scale between the two
        modes -- the z_score threshold lives in z units (typically of
        order ~3-15 for small permutation counts) while mean_adj lives
        in raw stat units (often <1 for WGN at low effect)."""
        exp = self._make_exp(seed=1)
        ana_m = AnalysisGLOW(exp=exp, n_perm_fwer=20,
                             n_perm_fwer_size_adjust=15,
                             score_method='mean_adj', verbose=False)
        ana_z = AnalysisGLOW(exp=exp, n_perm_fwer=20,
                             n_perm_fwer_size_adjust=15,
                             score_method='z_score', verbose=False)
        # not strictly guaranteed in pathological cases, but for default WGN
        # the z_score threshold is dramatically larger numerically
        assert ana_z.adj_crit != ana_m.adj_crit

    def test_pruning_produces_valid_antichain(self):
        """T7: regardless of mode, prune_info should report a disjoint
        set (no parent-child pairs in the selected list)."""
        from glow.graph import get_parent
        exp = self._make_exp(seed=2)
        ana = AnalysisGLOW(exp=exp, n_perm_fwer=20,
                           n_perm_fwer_size_adjust=15, verbose=False)

        selected = [eff.region_idx for eff in ana.effect_list]
        if not selected:
            return  # nothing to verify, but no crash either
        num_vox = exp.y.shape[2]
        parent = get_parent(ana.children, num_vox)

        for i in selected:
            anc = set()
            p = parent[i]
            while p != -1:
                anc.add(int(p))
                p = parent[p]
            assert anc.isdisjoint(selected), \
                f'region {i} has ancestor in selected: {anc & set(selected)}'

    def test_legacy_pickle_restores_as_mean_adj(self):
        """T5: a pre-z_score AnalysisGLOW pickle has no sigma_gam and no
        score_method.  __setstate__ should restore it as mean_adj."""
        # construct fresh, then mimic the old pickle layout
        exp = self._make_exp(seed=3)
        ana = AnalysisGLOW(exp=exp, n_perm_fwer=20,
                           n_perm_fwer_size_adjust=15,
                           score_method='mean_adj', verbose=False)
        # simulate old state: drop the new attributes
        state = ana.__dict__.copy()
        state.pop('sigma_gam', None)
        state.pop('score_method', None)
        state.pop('mu_gam', None)
        # adj_gam stays (it was the legacy field)

        new_ana = AnalysisGLOW.__new__(AnalysisGLOW)
        new_ana.__setstate__(state)

        assert new_ana.score_method == 'mean_adj'
        assert new_ana.sigma_gam is None
        assert new_ana.mu_gam is new_ana.adj_gam   # alias restored
