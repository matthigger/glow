"""Sprint 1b silent-failure guardrails.

Tests the loud-failure paths added in Sprint 1b:

- GAM low-data fallback warns and flags size_adjusted=False
- get_stat_perm LinAlgError rate tracking (>1% warn, >10% raise)
- _estimate_perm_sec raises FileNotFoundError when runtime model missing
"""
import warnings

import numpy as np
import pytest

from glow.analysis import Analysis, AnalysisGLOW
from glow.analysis._base import (
    _check_linalg_rate, _LINALG_WARN_RATE, _LINALG_FAIL_RATE)


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
        assert fit.gam is None
        assert fit.r2 is None
        assert fit.size_adjusted is False
        # fallback mu_fn returns zeros (identity)
        pred = fit.mu_fn(sizes)
        np.testing.assert_array_equal(pred, np.zeros_like(sizes))

    def test_no_warn_above_threshold(self):
        rng = np.random.default_rng(1)
        sizes = rng.integers(1, 500, size=100).astype(float)
        stats = 0.5 * np.log10(sizes) + rng.standard_normal(100) * 0.3
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            fit = AnalysisGLOW.fit_size_gam(sizes, stats)
        assert fit.size_adjusted is True
        assert fit.gam is not None

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
