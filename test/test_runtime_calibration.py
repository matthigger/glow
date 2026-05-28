"""Heavyweight FWER calibration test.

Gated by ``--runslow`` — not part of default pytest runs.

``test_fpr_bounded_under_size_adjustment``
    Generates a modest grid of H0 (no effect) synthetic experiments and
    verifies the observed family-wise error rate stays at or below the
    nominal alpha.  Catches systematic FPR inflation in the analysis
    pipeline.
"""
import numpy as np
import pytest

import glow


pytestmark = pytest.mark.slow


def _run_h0_trial(seed, alpha):
    """Run AnalysisGLOW on pure-noise data at seed; return whether any
    effect was declared significant."""
    exp = glow.experiment.Experiment.from_gauss(
        seed=seed, shape=(12, 12), a=2, b=2, num_img=25)
    ana = glow.analysis.AnalysisGLOW(
        exp, n_perm_fwer=50,
        alpha_fwer=alpha, min_vox=1).fit()
    return len(ana.effect_list) > 0


def test_fpr_bounded_under_size_adjustment():
    """Calibration: observed FPR under H0 should be <= nominal alpha."""
    alpha = 0.10            # relaxed for a small trial count
    n_trials = 200           # keep cheap; opt-in via --runslow
    # Budget derivation (one-sided binomial bound on the H0 rejection rate):
    # under correct calibration each trial rejects with prob p = alpha = 0.10,
    # so the rejection count is ~Binomial(n=200, p=0.10):
    #     mean = n*p              = 20
    #     std  = sqrt(n*p*(1-p))  = sqrt(200*0.10*0.90) = sqrt(18) ~= 4.24
    # a normal approximation gives a ~95% one-sided upper bound at
    #     mean + 1.645*std ~= 20 + 7.0 = 27 rejections  (rate ~0.135).
    # We set budget = 0.15 (= 30 rejections), leaving ~0.7 std of headroom
    # above that bound so the test fails only on genuine FPR inflation,
    # not on ordinary sampling noise.
    budget = 0.15
    rng = np.random.default_rng(0)
    seeds = rng.integers(0, 2**31 - 1, size=n_trials).tolist()
    n_rej = sum(_run_h0_trial(int(s), alpha) for s in seeds)
    observed = n_rej / n_trials
    assert observed <= budget, \
        f'observed FPR {observed:.3f} exceeds budget {budget:.3f} ' \
        f'({n_rej}/{n_trials} under H0)'
