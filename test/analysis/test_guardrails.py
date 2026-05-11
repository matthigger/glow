"""Silent-failure guardrails + statistical sanity.

Covers:
- _sanitize_adjusted_stat finite/nan/inf boundary behaviour
- get_stat_perm LinAlgError rate tracking (>1% warn, >10% raise)
- _estimate_perm_sec raises FileNotFoundError when runtime model missing
"""
import warnings

import numpy as np
import pytest

from glow.analysis import Analysis, AnalysisGLOW
from glow.analysis._base import (
    _check_linalg_rate, _sanitize_adjusted_stat,
    _LINALG_WARN_RATE, _LINALG_FAIL_RATE)


# ---------------------------------------------------------------------------
# LinAlgError rate tracking
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
        # Force float64 here: under the new float32 default, E is
        # numerically near-zero (rather than exactly zero), so np.linalg
        # .solve does not consistently raise LinAlgError and we cannot
        # exercise the guardrail.  This test is specifically about the
        # singular-matrix branch, so float64 is the appropriate dtype.
        import numpy as np
        from glow.experiment.exper import Experiment
        return Experiment.from_gauss(a=2, b=3, shape=(4, 4),
                                      num_img=num_img, seed=0,
                                      dtype=np.float64)

    def test_raises_when_all_cells_fail(self):
        from glow.analysis.mancova import get_hotel_tr
        # num_img == b => E is guaranteed singular at all regions/perms
        exp = self._tiny_exp(num_img=3)
        with pytest.raises(RuntimeError, match='LinAlgError rate'):
            Analysis.get_stat_perm_multi(
                exp=exp, get_stat_list=[get_hotel_tr], children=None)

    def test_healthy_run_is_silent(self):
        from glow.analysis.mancova import get_llr
        exp = self._tiny_exp(num_img=20)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            result = Analysis.get_stat_perm_multi(
                exp=exp, get_stat_list=[get_llr], children=None)
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
        # (signature: load_runtime_model(analysis_type, platform))
        monkeypatch.setattr(rt_mod, 'load_runtime_model',
                            lambda *a, **kw: None)

        exp = Experiment.from_gauss(a=2, b=1, shape=(4, 4),
                                    num_img=20, seed=0)
        with pytest.raises(FileNotFoundError,
                           match='runtime model not found'):
            AnalysisGLOW._estimate_perm_sec(exp)

    def test_error_points_to_profile_cli(self, monkeypatch):
        from glow.experiment.exper import Experiment
        import glow.benchmark.runtime as rt_mod

        monkeypatch.setattr(rt_mod, 'load_runtime_model',
                            lambda *a, **kw: None)

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


