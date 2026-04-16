"""Sprint 1b silent-failure guardrails.

Tests the loud-failure paths added in Sprint 1b:

- GAM low-data fallback warns and flags size_adjusted=False
- get_stat_perm LinAlgError rate tracking (>1% warn, >10% raise)
- _estimate_perm_sec raises FileNotFoundError when runtime model missing
"""
import warnings

import numpy as np
import pytest

from glow.analysis import AnalysisGLOW


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
