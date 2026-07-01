"""Regression: float64 accumulation in iter_llr_perm survives near-constant
HCP voxels (the sweep_llr / GLOW-Focus / a55e7237 FWER collapse).

Real HCP-YA data, after ExperimentScaled's global mean-subtraction, contains
near-constant background/edge voxels sitting at a large DC offset (a55e7237's
worst was mean -0.76, across-subject std ~5e-4 down to ~6e-8). For such a voxel
each region's error matrix E is built by cancelling two terms of magnitude
~num_img*mean(y)^2 down to the residual ~num_img*var(y); in float32 that loses
every significant digit, so E goes indefinite -- NaN where it crosses zero,
and where it stays barely positive the per-region inner-null std collapses
~3000x (5e-3 -> 1.7e-6) and the standardized z = (llr-mu)/std explodes (~4 ->
76286), poisoning the Westfall-Young max-z null so a perfectly segmented effect
(dice ~0.98) never reaches significance.

iter_llr_perm now accumulates in float64 (acc_dtype default); float32 is exposed
only so this test can exhibit the old collapse. The collapse is a fragile
float32 coincidence (the exact bad voxel/permutation that triggered a55e7237 is
not reproduced by a clean rebuild -- the cohort data has since drifted), so
rather than replay one trial we plant a block of voxels with a55e7237's
pathological profile on top of real HCP data and compare the two accumulation
dtypes directly.

Skipped unless the HCP reference dataset is present (glow._extra.benchmark.hcp) AND
--runslow is passed (it loads the full HCP brain).
"""
import numpy as np
import pytest

import glow.graph
from glow.analysis.mancova import decompose
from glow.experiment import permute
from glow._extra.benchmark import hcp
from glow._extra.benchmark.data import data_factory_hcp


def _caterpillar_children(num_vox):
    """A minimal valid binary tree over num_vox leaves.

    Topology is irrelevant here: the test asserts on the size-1 leaf regions
    (the planted voxels), where the float32 cancellation bit. Returns the
    (num_vox - 1, 2) children array in bottom-up order.
    """
    children = np.empty((num_vox - 1, 2), dtype=np.int64)
    children[0] = (0, 1)
    for j in range(1, num_vox - 1):
        children[j] = (num_vox + j - 1, j + 1)
    return children


def _inner_draws(y, q0, q1, children, acc_dtype, *, n_perm=128, base_seed=100_000):
    """Stack iter_llr_perm's inner-FL draws into (n_perm, num_reg), at acc_dtype."""
    num_vox, num_img = y.shape[2], y.shape[1]
    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=num_vox)
    perms = np.stack([permute._perm_indices(base_seed + i, num_img)
                      for i in range(n_perm)])
    return np.vstack(list(glow.graph.iter_llr_perm(
        y=y, q0=q0, q1=q1, perms=perms, leaf_ord=leaf_ord,
        region_l=region_l, region_h=region_h, min_size=1,
        acc_dtype=acc_dtype)))


@pytest.mark.slow
def test_iter_llr_perm_float64_survives_near_constant_voxels():
    """float64 keeps E positive-definite for near-constant HCP voxels; float32
    does not -- it NaNs roughly half the inner-FL draws and collapses the rest.
    """
    if not hcp.is_present():
        pytest.skip('HCP reference dataset not present (see glow._extra.benchmark.hcp)')

    # a55e7237's design: intercept-only nuisance (a0 = 1), one contrast (a1 = 1)
    exp = data_factory_hcp(hcp_feats=('fa',), a=1, seed=0)
    q0, q1, _ = decompose(exp.x, exp.contrast)
    assert q0.shape[0] == 1, 'expected intercept-only nuisance'
    num_img = exp.y.shape[1]

    # a real block of HCP voxels, with the first n_planted overwritten to match
    # a55e7237's pathological background profile: a large DC offset with a tiny
    # across-subject spread (mean -0.7611, std 2e-4).  This is the float32
    # cancellation regime; the trailing voxels stay healthy so the tree is real.
    n_planted = 30
    y = np.ascontiguousarray(exp.y[:, :, :500]).astype(np.float32)
    rng = np.random.default_rng(0)
    for v in range(n_planted):
        y[0, :, v] = (-0.7611 + 2e-4 * rng.standard_normal(num_img)
                      ).astype(np.float32)
    children = _caterpillar_children(y.shape[2])

    draws32 = _inner_draws(y, q0, q1, children, np.float32)[:, :n_planted]
    draws64 = _inner_draws(y, q0, q1, children, np.float64)[:, :n_planted]

    frac_finite32 = np.isfinite(draws32).mean()
    frac_finite64 = np.isfinite(draws64).mean()

    # float64 computes E accurately -> essentially every draw is finite
    assert frac_finite64 > 0.95, (
        f'float64 accumulation should keep E PD for near-constant voxels; '
        f'only {frac_finite64:.2f} of draws finite')
    # float32 cancellation drives E indefinite -> a large share of draws NaN
    # (empirically ~0.4-0.6 finite across planting seeds; 0.85 leaves margin)
    assert frac_finite32 < 0.85, (
        f'float32 accumulation unexpectedly stable ({frac_finite32:.2f} finite) '
        f'-- the near-constant cancellation regime is not being exercised')

    # and where float64 is finite, the inner-null std is at the real ~1e-2
    # scale, not the ~1e-6 noise floor float32 collapsed a55e7237 to
    std64 = np.nanstd(draws64, axis=0, ddof=1)
    assert np.nanmedian(std64) > 1e-3, (
        f'float64 inner-null std collapsed (median {np.nanmedian(std64):.2e})')
