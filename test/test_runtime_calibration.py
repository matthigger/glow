"""Heavyweight runtime/FWER calibration tests (Sprint 1c/1d).

Gated by ``--runslow`` — not part of default pytest runs.

Two tests live here:

- ``test_fpr_bounded_under_size_adjustment``
    Generates a modest grid of H0 (no effect) synthetic experiments and
    verifies the observed family-wise error rate stays at or below the
    nominal alpha. This is the calibration check that the GAM size
    adjustment doesn't systematically inflate false positives.

- ``test_runtime_estimate_covers_actual``
    Asserts the safety factor stored in ``runtime_glow.json`` covers the
    max observed ``actual/predicted`` ratio from the training grid.
    Cheap smoke check that the Sprint 2b calibration isn't silently
    undercounting.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import glow
from glow.benchmark.runtime import RUNTIME_MODEL_PATHS


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# FPR calibration
# ---------------------------------------------------------------------------

def _run_h0_trial(seed, alpha):
    """Run AnalysisGLOW on pure-noise data at seed; return whether any
    effect was declared significant."""
    exp = glow.experiment.Experiment.from_gauss(
        seed=seed, shape=(12, 12), a=2, b=2, num_img=25)
    ana = glow.analysis.AnalysisGLOW(
        exp, n_perm_fwer=50, n_perm_fwer_size_adjust=50,
        alpha_fwer=alpha, min_size=1, verbose=False)
    return len(ana.effect_list) > 0


def test_fpr_bounded_under_size_adjustment():
    """Calibration: observed FPR under H0 should be <= nominal alpha."""
    alpha = 0.10            # relaxed for a small trial count
    n_trials = 200           # keep cheap; opt-in via --runslow
    # Binomial upper bound for alpha=0.10, n=200 at ~95% confidence: ~0.15
    budget = 0.15
    rng = np.random.default_rng(0)
    seeds = rng.integers(0, 2**31 - 1, size=n_trials).tolist()
    n_rej = sum(_run_h0_trial(int(s), alpha) for s in seeds)
    observed = n_rej / n_trials
    assert observed <= budget, \
        f'observed FPR {observed:.3f} exceeds budget {budget:.3f} ' \
        f'({n_rej}/{n_trials} under H0)'


# ---------------------------------------------------------------------------
# Safety factor sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('analysis_type', ['GLOW', 'VBA', 'VBA-TFCE'])
def test_runtime_safety_factor_covers_max_ratio(analysis_type):
    """safety_factor (99.99% Gaussian) should cover residual_max_ratio.

    Pre-Sprint-2b models may not have these fields — in which case the
    test xfails rather than erroring, so the suite stays green until the
    models are refit.
    """
    path = RUNTIME_MODEL_PATHS[analysis_type]
    if not Path(path).exists():
        pytest.skip(f'no fitted model at {path}')
    with open(path) as f:
        model = json.load(f)
    if 'safety_factor' not in model or 'residual_max_ratio' not in model:
        pytest.xfail(f'{analysis_type} model predates Sprint 2b safety-factor '
                     f'calibration; refit with '
                     f'`python -m glow.benchmark.runtime --profile experiment`')
    assert model['safety_factor'] >= model['residual_max_ratio'], \
        (f'{analysis_type}: safety_factor={model["safety_factor"]} < '
         f'residual_max_ratio={model["residual_max_ratio"]} — '
         f'residuals have heavier-than-Gaussian tails, refit needed')
