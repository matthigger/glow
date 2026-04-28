"""Silent-failure guardrails + statistical sanity (Sprints 1b, 1c).

Covers:
- GAM low-data fallback warns and flags size_adjusted=False (1b)
- GAM robustness: high noise, extreme sizes, monotone shape (1c)
- _sanitize_adjusted_stat finite/nan/inf boundary behaviour (1c)
- get_stat_perm LinAlgError rate tracking (>1% warn, >10% raise) (1b)
- _estimate_perm_sec raises FileNotFoundError when runtime model missing (1b)
"""
import warnings

import numpy as np
import pytest

from glow.analysis import Analysis, AnalysisGLOW
from glow.analysis._base import (
    _check_linalg_rate, _sanitize_adjusted_stat,
    _LINALG_WARN_RATE, _LINALG_FAIL_RATE)


# ---------------------------------------------------------------------------
# Fix 2: GAM low-data warning
# ---------------------------------------------------------------------------

class TestFitSizeGamLowData:
    """fit_size_gam warns + flags size_adjusted=False when data is thin."""

    def test_warns_below_threshold(self):
        sizes = np.arange(1, 10).astype(float)  # only 9 points
        stats = np.log10(sizes) + 0.1
        with pytest.warns(RuntimeWarning, match='fit_size_gam'):
            fit = AnalysisGLOW.fit_size_gam(sizes, stats)
        assert fit.mu_gam is None
        assert fit.sigma_gam is None
        assert fit.r2 is None
        assert fit.size_adjusted is False
        # fallback mu_fn returns zeros, sigma_fn returns ones (identity)
        np.testing.assert_array_equal(fit.mu_fn(sizes), np.zeros_like(sizes))
        np.testing.assert_array_equal(fit.sigma_fn(sizes), np.ones_like(sizes))

    def test_no_warn_above_threshold(self):
        rng = np.random.default_rng(1)
        sizes = rng.integers(1, 500, size=100).astype(float)
        stats = 0.5 * np.log10(sizes) + rng.standard_normal(100) * 0.3
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            fit = AnalysisGLOW.fit_size_gam(sizes, stats,
                                            score_method='z_score')
        assert fit.size_adjusted is True
        assert fit.mu_gam is not None
        assert fit.sigma_gam is not None

    def test_threshold_boundary(self):
        # exactly at threshold - 1 should warn; at threshold should not.
        n_ok = AnalysisGLOW._MIN_GAM_FIT_POINTS
        rng = np.random.default_rng(2)

        # below: 1 short
        sizes_short = rng.integers(1, 100, size=n_ok - 1).astype(float)
        stats_short = np.log10(sizes_short)
        with pytest.warns(RuntimeWarning):
            fit = AnalysisGLOW.fit_size_gam(sizes_short, stats_short)
        assert fit.size_adjusted is False

        # at threshold: should fit (no warning)
        sizes_ok = rng.integers(1, 100, size=n_ok).astype(float)
        stats_ok = np.log10(sizes_ok)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            fit = AnalysisGLOW.fit_size_gam(sizes_ok, stats_ok)
        assert fit.size_adjusted is True


# ---------------------------------------------------------------------------
# Fix 3: LinAlgError rate tracking
# ---------------------------------------------------------------------------

class TestLinAlgRateHelper:
    """Unit tests for the _check_linalg_rate threshold logic."""

    def test_zero_total_is_noop(self):
        # should not warn or raise on an empty run
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            _check_linalg_rate(0, 0, fn_name='test')

    def test_below_warn_is_silent(self):
        # 0.5% failure rate -> silent
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            _check_linalg_rate(5, 1000, fn_name='test')

    def test_above_warn_warns(self):
        # 5% failure rate -> warn (> 1%, < 10%)
        with pytest.warns(RuntimeWarning, match='LinAlgError rate'):
            _check_linalg_rate(50, 1000, fn_name='test')

    def test_above_fail_raises(self):
        # 20% failure rate -> raise
        with pytest.raises(RuntimeError, match='LinAlgError rate'):
            _check_linalg_rate(200, 1000, fn_name='test')

    def test_message_includes_fn_name(self):
        with pytest.raises(RuntimeError, match='my_fn'):
            _check_linalg_rate(200, 1000, fn_name='my_fn')


class TestLinAlgRateIntegration:
    """End-to-end: force LinAlgError via rank-deficient design and verify."""

    def _tiny_exp(self, num_img):
        # rank-deficient setup: num_img <= b makes E singular under
        # some permutations, which raises LinAlgError in get_hotel_tr
        # / get_pillai / get_roys_root.
        from glow.experiment.exper import Experiment
        return Experiment.from_gauss(a=2, b=3, shape=(4, 4),
                                      num_img=num_img, seed=0)

    def test_raises_when_all_cells_fail(self):
        from glow.analysis.mancova import get_hotel_tr
        # num_img == b => E is guaranteed singular at all regions/perms
        exp = self._tiny_exp(num_img=3)
        with pytest.raises(RuntimeError, match='LinAlgError rate'):
            Analysis.get_stat_perm_multi(
                exp=exp, get_stat_list=[get_hotel_tr], n_perm=1,
                children=None)

    def test_healthy_run_is_silent(self):
        from glow.analysis.mancova import get_llr
        exp = self._tiny_exp(num_img=20)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            result = Analysis.get_stat_perm_multi(
                exp=exp, get_stat_list=[get_llr], n_perm=2,
                children=None)
        assert get_llr in result


# ---------------------------------------------------------------------------
# Fix 4: _estimate_perm_sec raises when runtime model missing
# ---------------------------------------------------------------------------

class TestEstimatePermSec:
    """Magic fallback removed -- missing model must raise FileNotFoundError."""

    def test_raises_when_model_missing(self, monkeypatch):
        from glow.experiment.exper import Experiment
        import glow.benchmark.runtime as rt_mod

        # stub load_runtime_model to simulate missing model file
        monkeypatch.setattr(rt_mod, 'load_runtime_model',
                            lambda atype: None)

        exp = Experiment.from_gauss(a=2, b=1, shape=(4, 4),
                                    num_img=20, seed=0)
        with pytest.raises(FileNotFoundError,
                           match='GLOW runtime model not found'):
            AnalysisGLOW._estimate_perm_sec(exp)

    def test_error_points_to_profile_cli(self, monkeypatch):
        from glow.experiment.exper import Experiment
        import glow.benchmark.runtime as rt_mod

        monkeypatch.setattr(rt_mod, 'load_runtime_model',
                            lambda atype: None)

        exp = Experiment.from_gauss(a=2, b=1, shape=(4, 4),
                                    num_img=20, seed=1)
        with pytest.raises(FileNotFoundError,
                           match=r'python -m glow\.benchmark\.runtime'):
            AnalysisGLOW._estimate_perm_sec(exp)


# ---------------------------------------------------------------------------
# Sprint 1c: _sanitize_adjusted_stat boundary behaviour
# ---------------------------------------------------------------------------

class TestSanitizeAdjustedStat:
    """nan -> 0.0, posinf -> 0.0, neginf -> nan; finite values untouched."""

    def test_finite_passthrough(self):
        x = np.array([-3.0, -0.1, 0.0, 0.5, 7.2])
        np.testing.assert_array_equal(_sanitize_adjusted_stat(x), x)

    def test_nan_maps_to_zero(self):
        x = np.array([np.nan, 1.0])
        out = _sanitize_adjusted_stat(x)
        assert out[0] == 0.0 and out[1] == 1.0

    def test_posinf_maps_to_zero(self):
        x = np.array([np.inf, 2.0])
        out = _sanitize_adjusted_stat(x)
        assert out[0] == 0.0 and out[1] == 2.0

    def test_neginf_maps_to_nan(self):
        x = np.array([-np.inf, 2.0])
        out = _sanitize_adjusted_stat(x)
        assert np.isnan(out[0]) and out[1] == 2.0

    def test_no_clip_to_magic_negative(self):
        # Regression: legacy code clipped neginf to -30; now neginf -> nan
        # so nanmax can ignore the region rather than silently biasing
        # the max-stat null with a magic constant.
        x = np.array([-np.inf, -1e9, -30.0, 0.0, 5.0])
        out = _sanitize_adjusted_stat(x)
        assert np.isnan(out[0])
        # non-infinite values are untouched regardless of magnitude
        assert out[1] == -1e9
        assert out[2] == -30.0

    def test_mixed_array_preserves_shape(self):
        x = np.array([[np.nan, np.inf], [-np.inf, 1.0]])
        out = _sanitize_adjusted_stat(x)
        assert out.shape == x.shape
        assert out[0, 0] == 0.0 and out[0, 1] == 0.0
        assert np.isnan(out[1, 0]) and out[1, 1] == 1.0


# ---------------------------------------------------------------------------
# Sprint 1c: GAM robustness on unusual data shapes
# ---------------------------------------------------------------------------

class TestFitSizeGamRobustness:
    """GAM should stay well-behaved under noisy / extreme / monotone inputs."""

    def _fit(self, sizes, stats):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return AnalysisGLOW.fit_size_gam(sizes, stats)

    def test_high_noise_still_fits(self):
        # fit should succeed (no nan/inf) even when signal buried in noise
        rng = np.random.default_rng(0)
        sizes = rng.integers(1, 2000, size=400).astype(float)
        stats = 0.3 * np.log10(sizes) + rng.standard_normal(400) * 3.0
        fit = self._fit(sizes, stats)
        assert fit.size_adjusted is True
        pred = fit.mu_fn(sizes)
        assert np.all(np.isfinite(pred))

    def test_extreme_sizes_do_not_overflow(self):
        # sizes spanning 1..1e6 shouldn't break the GAM or its predictions
        rng = np.random.default_rng(1)
        sizes = rng.integers(1, 1_000_000, size=200).astype(float)
        stats = 0.5 * np.log10(sizes) + rng.standard_normal(200) * 0.5
        fit = self._fit(sizes, stats)
        assert fit.size_adjusted is True
        pred = fit.mu_fn(np.array([1.0, 1e3, 1e6]))
        assert np.all(np.isfinite(pred))

    def test_monotone_size_stat_recovered(self):
        # clean monotone relationship: GAM should recover the trend
        rng = np.random.default_rng(2)
        sizes = np.unique(rng.integers(1, 5000, size=300)).astype(float)
        stats = 2.0 * np.log10(sizes) + rng.standard_normal(len(sizes)) * 0.05
        fit = self._fit(sizes, stats)
        assert fit.size_adjusted is True
        # monotone: mu(small) < mu(large)
        pred_small = fit.mu_fn(np.array([5.0]))[0]
        pred_large = fit.mu_fn(np.array([4000.0]))[0]
        assert pred_small < pred_large, \
            f'expected monotone increase, got {pred_small:.3f} vs {pred_large:.3f}'

    def test_constant_stats_fit(self):
        # zero-variance y: GAM should still return a fit (degenerate but valid)
        rng = np.random.default_rng(3)
        sizes = rng.integers(1, 500, size=120).astype(float)
        stats = np.ones_like(sizes) * 4.2
        fit = self._fit(sizes, stats)
        assert fit.size_adjusted is True
        pred = fit.mu_fn(sizes)
        assert np.all(np.isfinite(pred))
