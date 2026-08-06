"""GPU inner-perm backend equivalence, anchored on cpu_reliable.

Mirrors test_inner_perm.py's anchoring strategy one level out: there,
cpu_perm is validated against the per-region iter_mancova + get_llr
trust anchor; here the device backend is validated against that same
anchor (per-draw) and against cpu_perm (moments), across b and both
nuisance regimes. The two paths in inner_perm_gpu take opposite
permutation directions, so a convention slip in either shows up as a
per-draw mismatch rather than a subtle moment drift.

Every test skips when no CUDA device is visible.

Run:
    ~/venv_glow/bin/pytest test/analysis/test_inner_perm_gpu.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

import glow.graph
import glow.mask
from glow.analysis import inner_perm, inner_perm_gpu
from glow.analysis.cluster import cluster
from glow.analysis.mancova import decompose
from glow.experiment.exper import Experiment


requires_cuda = pytest.mark.skipif(
    not inner_perm_gpu.is_available(),
    reason='no CUDA device visible')

B_LIST = [1, 2, 3, 6]


# ---------------------------------------------------------------------------
# Synthetic experiment builders (b-parametrised versions of
# test_inner_perm.py's, which fix b = 2)

def _intercept_only_exp(b, seed=0, n_img=24, shape=(2, 3, 5)):
    """Build an exp with intercept-only nuisance (Q0 = span(1))."""
    rng = np.random.default_rng(seed)
    num_vox = int(np.prod(shape))
    mask_idx = np.arange(num_vox, dtype=np.int64).reshape(shape)
    y = rng.standard_normal((b, n_img, num_vox)).astype(np.float64)
    x = np.vstack([
        np.ones(n_img, dtype=np.float64),
        rng.standard_normal(n_img).astype(np.float64),
        rng.standard_normal(n_img).astype(np.float64),
    ])
    contrast = np.array([False, True, True])
    return Experiment(x=x, y=y, contrast=contrast, mask_idx=mask_idx)


def _general_q0_exp(b, seed=0, n_img=24, shape=(3, 4, 4)):
    """Build an exp whose nuisance columns are non-constant."""
    rng = np.random.default_rng(seed)
    rows = [np.ones(n_img, dtype=np.float64)]
    contrast = [False]
    for is_interest in (False, True, True):
        c = rng.standard_normal(n_img).astype(np.float64)
        c -= c.mean()
        c /= c.std() + 1e-9
        rows.append(c)
        contrast.append(is_interest)
    x = np.vstack(rows)

    num_vox = int(np.prod(shape))
    mask = np.ones(shape, dtype=bool)
    y = rng.standard_normal((b, n_img, num_vox)).astype(np.float64)
    return Experiment(x=x, y=y, contrast=np.array(contrast),
                      mask_idx=glow.mask.get_mask_idx(mask))


def _prep(b, path, *, min_vox=2):
    """Build the kwargs every inner_perm backend wants, for one path."""
    exp = (_intercept_only_exp(b) if path == 'intercept'
           else _general_q0_exp(b))
    q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)
    children = cluster(exp=exp, mode='Focus')
    return dict(exp=exp, q0=q0, q1=q1, children=children, min_vox=min_vox)


def _call(fn, prep, *, n_perm, base_seed=12_345, **kwargs):
    return fn(exp=prep['exp'], base_seed=base_seed, n_perm=n_perm,
              q0=prep['q0'], q1=prep['q1'], children=prep['children'],
              min_vox=prep['min_vox'], **kwargs)


def _assert_cells_match(got, ref, *, atol, label):
    """Assert identical NaN masks and per-cell agreement within atol."""
    assert got.shape == ref.shape
    nan_g, nan_r = np.isnan(got), np.isnan(ref)
    assert (nan_g == nan_r).all(), \
        f'{label}: NaN masks differ in {(nan_g != nan_r).sum()} cells'
    finite = ~nan_g
    assert finite.any(), f'{label}: no finite cells to compare'
    diff = float(np.abs(got[finite] - ref[finite]).max())
    assert diff < atol, f'{label}: max abs diff {diff:.3e} > {atol:.3e}'


# ---------------------------------------------------------------------------
# Per-draw equivalence against the trust anchor

@requires_cuda
@pytest.mark.parametrize('path', ['intercept', 'general'])
@pytest.mark.parametrize('b', B_LIST)
def test_gpu_draws_match_reliable(b, path):
    """gpu_perm_full draws match cpu_reliable_full cell-by-cell."""
    prep = _prep(b, path)
    got = _call(inner_perm_gpu.gpu_perm_full, prep, n_perm=6)
    ref = _call(inner_perm.cpu_reliable_full, prep, n_perm=6)
    _assert_cells_match(got, ref, atol=1e-9, label=f'b={b} {path}')


@requires_cuda
@pytest.mark.parametrize('path', ['intercept', 'general'])
@pytest.mark.parametrize('b', B_LIST)
def test_gpu_moments_match_cpu_perm(b, path):
    """gpu_perm (mu, std) match cpu_perm's, which share the Chan reduction."""
    prep = _prep(b, path)
    mu_g, std_g = _call(inner_perm_gpu.gpu_perm, prep, n_perm=20)
    mu_c, std_c = _call(inner_perm.cpu_perm, prep, n_perm=20)
    _assert_cells_match(mu_g, mu_c, atol=1e-9, label=f'mu b={b} {path}')
    _assert_cells_match(std_g, std_c, atol=1e-9, label=f'std b={b} {path}')


# ---------------------------------------------------------------------------
# The path dispatch, and the knobs that must not change the answer

@requires_cuda
def test_dispatch_picks_intercept_path():
    """An intercept-only design dispatches to the intercept-only path."""
    prep = _prep(2, 'intercept')
    _, chunk_llr = inner_perm_gpu._dispatch_state(
        exp=prep['exp'], q0=prep['q0'], q1=prep['q1'],
        children=prep['children'], min_vox=prep['min_vox'],
        device='cuda', acc_dtype=np.float64)
    assert chunk_llr is inner_perm_gpu._intercept_chunk_llr


@requires_cuda
def test_dispatch_picks_general_path():
    """A non-constant nuisance column dispatches to the general path."""
    prep = _prep(2, 'general')
    _, chunk_llr = inner_perm_gpu._dispatch_state(
        exp=prep['exp'], q0=prep['q0'], q1=prep['q1'],
        children=prep['children'], min_vox=prep['min_vox'],
        device='cuda', acc_dtype=np.float64)
    assert chunk_llr is inner_perm_gpu._general_chunk_llr


@requires_cuda
def test_forced_paths_agree():
    """On an intercept-only exp both paths compute the same draws.

    The general-Q0 algebra must reduce to the intercept-only one when Q0
    happens to be span(1) -- the two take opposite permutation
    directions, so this pins the convention rather than the algebra.
    """
    prep = _prep(2, 'intercept')
    got = _call(inner_perm_gpu.gpu_perm_full, prep, n_perm=6,
                force_path='general')
    ref = _call(inner_perm_gpu.gpu_perm_full, prep, n_perm=6,
                force_path='intercept')
    _assert_cells_match(got, ref, atol=1e-9, label='forced-path')


@requires_cuda
@pytest.mark.parametrize('path', ['intercept', 'general'])
@pytest.mark.parametrize('perm_chunk', [1, 3, 32])
def test_perm_chunk_invariance(path, perm_chunk):
    """perm_chunk is a memory knob only: draws are unchanged by it.

    Uses raw draws rather than moments: the Chan combine folds one chunk
    at a time, so its round-off legitimately depends on chunk size, but
    the draws themselves must not.
    """
    prep = _prep(2, path)
    got = _call(inner_perm_gpu.gpu_perm_full, prep, n_perm=7,
                perm_chunk=perm_chunk)
    ref = _call(inner_perm_gpu.gpu_perm_full, prep, n_perm=7, perm_chunk=8)
    _assert_cells_match(got, ref, atol=1e-12,
                        label=f'perm_chunk={perm_chunk} {path}')


# ---------------------------------------------------------------------------
# float32 safety: T_u via the cancellation-free split
#
# The a55e7237 FWER collapse (see test_inner_perm_hcp.py) came from forming
# E as a difference of two terms ~num_img * mean(y)^2 whose residual is
# ~num_img * var(y). For a near-constant voxel on a large DC offset that
# ratio is below float32 eps. These tests plant that profile synthetically
# (no HCP dataset needed) and pin both halves of the claim: the centred
# split is an algebraic identity, and it is what makes float32 safe.

def _caterpillar_children(num_vox):
    """Build a minimal valid binary tree over num_vox leaves."""
    children = np.empty((num_vox - 1, 2), dtype=np.int64)
    children[0] = (0, 1)
    for j in range(1, num_vox - 1):
        children[j] = (num_vox + j - 1, j + 1)
    return children


def _poisoned_prep(b=1, num_vox=500, n_planted=30, num_img=100, seed=0):
    """Healthy WGN voxels with a planted near-constant DC-offset block.

    Profile is a55e7237's worst background voxel: mean -0.7611 with an
    across-image std of 2e-4.
    """
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((b, num_img, num_vox))
    for v in range(n_planted):
        y[0, :, v] = -0.7611 + 2e-4 * rng.standard_normal(num_img)
    x = np.vstack([np.ones(num_img), rng.standard_normal(num_img)])
    contrast = np.array([False, True])
    mask_idx = np.arange(num_vox, dtype=np.int64).reshape(1, 1, num_vox)
    exp = Experiment(x=x, y=y, contrast=contrast, mask_idx=mask_idx)
    q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)
    return dict(exp=exp, q0=q0, q1=q1,
                children=_caterpillar_children(num_vox), min_vox=1), n_planted


@requires_cuda
@pytest.mark.parametrize('path', ['intercept'])
@pytest.mark.parametrize('b', [1, 2])
def test_centering_is_an_identity(b, path):
    """center=True reproduces the direct T_u form in float64.

    Guards the algebra itself (T_u = W_r + S_r, cross terms vanishing
    because Q0 w = 0) independently of any float32 question.
    """
    prep = _prep(b, path)
    got = _call(inner_perm_gpu.gpu_perm_full, prep, n_perm=6, center=True)
    ref = _call(inner_perm_gpu.gpu_perm_full, prep, n_perm=6, center=False)
    _assert_cells_match(got, ref, atol=1e-9, label=f'centered b={b}')


@requires_cuda
def test_float32_centered_survives_near_constant_voxels():
    """Centred float32 recovers float64; direct float32 collapses.

    Both halves are asserted: without the collapse in the direct form the
    test would pass vacuously on data that never entered the cancellation
    regime.
    """
    prep, n_planted = _poisoned_prep()
    sl = slice(0, n_planted)
    llr_obs, _ = glow.graph.compute_llr_batched(
        prep['exp'], children=prep['children'],
        q0=prep['q0'], q1=prep['q1'])

    def summarize(**kw):
        draws = _call(inner_perm_gpu.gpu_perm_full, prep, n_perm=128,
                      base_seed=100_000, **kw)
        mu, std = _call(inner_perm_gpu.gpu_perm, prep, n_perm=128,
                        base_seed=100_000, **kw)
        with np.errstate(invalid='ignore', divide='ignore'):
            z = (llr_obs - mu) / np.where(std < 1e-12, np.nan, std)
        return (float(np.isfinite(draws[:, sl]).mean()),
                float(np.nanmedian(std[sl])),
                float(np.nanmax(np.abs(z))))

    fin64, std64, z64 = summarize(acc_dtype=np.float64, center=False)
    fin32, std32, z32 = summarize(acc_dtype=np.float32, center=False)
    fin32c, std32c, z32c = summarize(acc_dtype=np.float32, center=True)

    # float64 reference: every draw finite, inner-null std at its real scale
    assert fin64 > 0.99, f'float64 reference unhealthy ({fin64:.2f} finite)'
    assert std64 > 1e-3, f'float64 std collapsed ({std64:.2e})'

    # direct float32 must exhibit the collapse, else nothing is being tested
    assert fin32 < 0.9, (
        f'direct float32 unexpectedly stable ({fin32:.2f} finite) -- the '
        f'cancellation regime is not being exercised')
    assert z32 > 3 * z64, (
        f'direct float32 max|z| {z32:.3g} did not inflate over the float64 '
        f'{z64:.3g}')

    # centred float32 recovers the reference on all three signatures
    assert fin32c > 0.99, f'centred float32 NaNed draws ({fin32c:.2f} finite)'
    assert abs(std32c - std64) < 1e-2 * std64, (
        f'centred float32 std {std32c:.4e} != float64 {std64:.4e}')
    assert abs(z32c - z64) < 1e-2 * z64, (
        f'centred float32 max|z| {z32c:.5g} != float64 {z64:.5g}')


@requires_cuda
@pytest.mark.parametrize('path', ['intercept', 'general'])
def test_min_vox_drops_small_regions(path):
    """Regions below min_vox come back NaN, larger ones finite."""
    prep = _prep(2, path, min_vox=4)
    mu, std = _call(inner_perm_gpu.gpu_perm, prep, n_perm=8)
    size = np.diff(np.stack(
        glow.graph.build_dfs_preorder(
            children=prep['children'],
            num_vox=prep['exp'].y.shape[2])[1:]), axis=0)[0]
    small = size < 4
    assert small.any() and (~small).any()
    assert np.isnan(mu[small]).all()
    assert np.isfinite(mu[~small]).all()
    assert np.isfinite(std[~small]).all()
