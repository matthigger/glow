"""GPU draw-matrix equivalence, anchored on cpu_reliable.

Mirrors test_draws.py's anchoring strategy one level out: there the
batched CPU kernel is validated against the per-region iter_mancova +
get_llr trust anchor; here the device backend is validated against that
same anchor, cell by cell, across b and both nuisance regimes. There is
one code path for both regimes -- for intercept-only nuisance the rho
correction comes out identically zero rather than being branched around --
so the design axis below exercises that collapse.

The float32 tests are the load-bearing ones: the backend's own acc_dtype
default is float32, and it is only safe because T = E + H is assembled
from two cancellation-free pieces with the DC-carrying scans held at
scan_dtype=float64. Both halves of that claim are asserted. (A fit takes
float64 regardless, to reproduce the CPU path's p-values -- see
_fit_gpu.)

Run:
    ~/venv_glow/bin/pytest test/analysis/test_draws_gpu.py -v
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import glow.graph
import glow.mask
from glow.analysis import draws, draws_gpu
from glow.analysis.cluster import cluster
from glow.analysis.mancova import decompose
from glow.experiment.exper import Experiment


requires_cuda = pytest.mark.skipif(
    not draws_gpu.is_available(),
    reason='no CUDA device visible')

B_LIST = [1, 2, 3, 6]
DESIGNS = ['intercept', 'general']


# ---------------------------------------------------------------------------
# Synthetic experiment builders (b-parametrised versions of
# test_draws.py's, which fix b = 2)

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
    y = rng.standard_normal((b, n_img, num_vox)).astype(np.float64)
    return Experiment(x=x, y=y, contrast=np.array(contrast),
                      mask_idx=glow.mask.get_mask_idx(np.ones(shape, bool)))


def _prep(b, design, *, min_vox=2):
    """Build the kwargs every draws backend wants, for one design."""
    exp = (_intercept_only_exp(b) if design == 'intercept'
           else _general_q0_exp(b))
    q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)
    children = cluster(exp=exp, mode='Focus')
    return dict(exp=exp, q0=q0, q1=q1, children=children, min_vox=min_vox)


def _call(fn, prep, *, n_perm, base_seed=12_345, **kwargs):
    return fn(exp=prep['exp'], base_seed=base_seed, n_perm=n_perm,
              q0=prep['q0'], q1=prep['q1'], children=prep['children'],
              min_vox=prep['min_vox'], **kwargs)


def _reg_active(prep):
    """The comparison set a fit would use: size >= min_vox."""
    _, region_l, region_h = glow.graph.build_dfs_preorder(
        children=prep['children'], num_vox=prep['exp'].y.shape[2])
    return (region_h - region_l) >= prep['min_vox']


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
# Equivalence against the trust anchor, at float64 so only the algorithm
# (not the dtype) is under test

@requires_cuda
@pytest.mark.parametrize('design', DESIGNS)
@pytest.mark.parametrize('b', B_LIST)
def test_gpu_draws_match_reliable(b, design):
    """gpu_perm draws match cpu_reliable cell by cell."""
    prep = _prep(b, design)
    got = _call(draws_gpu.gpu_perm, prep, n_perm=6,
                acc_dtype=np.float64)
    ref = _call(draws.cpu_reliable, prep, n_perm=6)
    _assert_cells_match(got, ref, atol=1e-9, label=f'b={b} {design}')


@requires_cuda
@pytest.mark.parametrize('design', DESIGNS)
@pytest.mark.parametrize('b', B_LIST)
def test_gpu_row_0_is_the_observed_draw(b, design):
    """Row 0 at base_seed = 0 is the unpermuted LLR, not a null draw.

    The equivalence test above runs at base_seed = 12_345, where every row
    is permuted. AnalysisGLOWSplit.fit reads row 0 of a base_seed = 0 matrix as
    the observed LLR, so a device backend that permuted at seed 0 would
    pass every other test here and still swap the observed statistic for a
    null one -- see permute._perm_indices on the reserved-0 convention.
    """
    prep = _prep(b, design)
    llr_obs, _ = glow.graph.compute_llr_batched(
        prep['exp'], children=prep['children'], q0=prep['q0'],
        q1=prep['q1'], min_size=prep['min_vox'])
    got = _call(draws_gpu.gpu_perm, prep, n_perm=3, base_seed=0,
                acc_dtype=np.float64)
    _assert_cells_match(got[:1], llr_obs[None], atol=1e-9,
                        label=f'row 0 b={b} {design}')


@requires_cuda
@pytest.mark.parametrize('design', DESIGNS)
@pytest.mark.parametrize('b', B_LIST)
def test_gpu_draws_match_reliable_at_base_seed_0(b, design):
    """Every row agrees with the trust anchor at base_seed = 0."""
    prep = _prep(b, design)
    got = _call(draws_gpu.gpu_perm, prep, n_perm=4, base_seed=0,
                acc_dtype=np.float64)
    ref = _call(draws.cpu_reliable, prep, n_perm=4, base_seed=0)
    _assert_cells_match(got, ref, atol=1e-9, label=f'b={b} {design}')


@requires_cuda
@pytest.mark.parametrize('design', DESIGNS)
@pytest.mark.parametrize('perm_chunk', [1, 3, 32])
def test_perm_chunk_invariance(design, perm_chunk):
    """perm_chunk is a throughput and memory knob only, never a result.

    _fit_gpu.resolve_perm_chunk picks it from free device memory, so it
    varies with the card and the problem size. Nothing a fit reports may
    move with it.
    """
    prep = _prep(2, design)
    got = _call(draws_gpu.gpu_perm, prep, n_perm=7,
                perm_chunk=perm_chunk, acc_dtype=np.float64)
    ref = _call(draws_gpu.gpu_perm, prep, n_perm=7, perm_chunk=8,
                acc_dtype=np.float64)
    _assert_cells_match(got, ref, atol=1e-12,
                        label=f'perm_chunk={perm_chunk} {design}')


@requires_cuda
@pytest.mark.parametrize('design', DESIGNS)
def test_min_vox_drops_small_regions(design):
    """Regions below min_vox come back NaN, larger ones finite."""
    prep = _prep(2, design, min_vox=4)
    got = _call(draws_gpu.gpu_perm, prep, n_perm=8)
    _, region_l, region_h = glow.graph.build_dfs_preorder(
        children=prep['children'], num_vox=prep['exp'].y.shape[2])
    small = (region_h - region_l) < 4
    assert small.any() and (~small).any()
    assert np.isnan(got[:, small]).all()
    assert np.isfinite(got[:, ~small]).all()


# ---------------------------------------------------------------------------
# One path for both nuisance regimes

@requires_cuda
def test_rho_vanishes_for_intercept_only():
    """The nuisance correction is identically zero for intercept-only Q0.

    This is why one code path suffices: rho = Q0[:, pi^-1] u is zero when
    Q0's rows are constant, so W_r collapses to reg_sum(T_v) and S_r to the
    spatial scatter of s0 -- exactly what a hoisted intercept-only fast
    path would compute (cf. glow.graph.iter_llr_perm's rho / X_v terms).
    """
    import torch
    prep = _prep(2, 'intercept')
    state = draws_gpu.prep_tree(
        draws_gpu.prep_shared(
            prep['exp'], q0=prep['q0'], q1=prep['q1'],
            acc_dtype=np.float64),
        children=prep['children'], min_vox=prep['min_vox'])
    perms = draws_gpu._build_perms(7, 4, prep['exp'].y.shape[1])
    src = draws_gpu._build_perm_inv_tensor(perms, state['dev'])
    q01_perm = state['Q01'][:, src].permute(1, 0, 2).contiguous()
    alpha = torch.einsum('pkn,bnv->pkbv', q01_perm, state['U'])
    rho = alpha[:, :state['a0']]
    assert float(rho.abs().max()) < 1e-12, \
        f'rho should vanish for intercept-only, got {float(rho.abs().max()):.2e}'


# ---------------------------------------------------------------------------
# float32 safety
#
# The a55e7237 FWER collapse (see test_draws_hcp.py) came from forming
# T as a difference of two terms ~num_img * mean(y)^2 whose residual is
# ~num_img * var(y). For a near-constant voxel on a large DC offset that
# ratio is below float32 eps. The backend avoids it by splitting T into
# W_r (built on the nuisance residual, DC-free) and S_r (the DC-carrying
# group, scanned at scan_dtype). These tests plant that profile
# synthetically -- no HCP dataset, no --runslow.

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
@pytest.mark.parametrize('design', DESIGNS)
def test_float32_matches_float64_on_wellconditioned_data(design):
    """float32 costs only round-off when nothing is ill-conditioned."""
    prep = _prep(2, design)
    got32 = _call(draws_gpu.gpu_perm, prep, n_perm=20, acc_dtype=np.float32)
    got64 = _call(draws_gpu.gpu_perm, prep, n_perm=20, acc_dtype=np.float64)
    fin = np.isfinite(got64) & np.isfinite(got32)
    assert fin.any()
    assert (np.abs(got32[fin] - got64[fin]).max()
            < 1e-4 * np.abs(got64[fin]).max())


@requires_cuda
def test_float32_survives_near_constant_voxels():
    """float32 recovers float64 as long as the s_star scans stay float64.

    Both halves are asserted: scan_dtype=float32 must exhibit the collapse,
    else the test would pass vacuously on data that never entered the
    cancellation regime.
    """
    prep, n_planted = _poisoned_prep()
    sl = slice(0, n_planted)
    llr_obs, _ = glow.graph.compute_llr_batched(
        prep['exp'], children=prep['children'],
        q0=prep['q0'], q1=prep['q1'])

    def summarize(**kw):
        got = _call(draws_gpu.gpu_perm, prep, n_perm=128,
                    base_seed=100_000, **kw)
        # the moments the fit would take, off the same matrix: this is
        # Analysis.z_score_stat's reduction, so the collapse is measured
        # where it actually reaches a p-value
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mu = np.nanmean(got, axis=0)
            std = np.nanstd(got, axis=0, ddof=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            z = (llr_obs - mu) / np.where(std < 1e-12, np.nan, std)
        return (float(np.isfinite(got[:, sl]).mean()),
                float(np.nanmedian(std[sl])),
                float(np.nanmax(np.abs(z))))

    fin64, std64, z64 = summarize(acc_dtype=np.float64)
    fin32, std32, z32 = summarize(acc_dtype=np.float32,
                                  scan_dtype=np.float32)
    fin32c, std32c, z32c = summarize(acc_dtype=np.float32,
                                     scan_dtype=np.float64)

    # float64 reference: every draw finite, inner-null std at its real scale
    assert fin64 > 0.99, f'float64 reference unhealthy ({fin64:.2f} finite)'
    assert std64 > 1e-3, f'float64 std collapsed ({std64:.2e})'

    # an all-float32 scan must exhibit the collapse, else nothing is tested
    assert fin32 < 0.9 or z32 > 3 * z64, (
        f'float32 scans unexpectedly stable ({fin32:.2f} finite, '
        f'max|z| {z32:.3g} vs {z64:.3g}) -- the cancellation regime is not '
        f'being exercised')

    # the shipped configuration recovers the reference on all three signatures
    assert fin32c > 0.99, f'float32 NaNed draws ({fin32c:.2f} finite)'
    assert abs(std32c - std64) < 1e-2 * std64, (
        f'float32 std {std32c:.4e} != float64 {std64:.4e}')
    assert abs(z32c - z64) < 1e-2 * z64, (
        f'float32 max|z| {z32c:.5g} != float64 {z64:.5g}')


# ---------------------------------------------------------------------------
# Streaming reduction: two passes must equal reducing the whole matrix
#
# gpu_summarize never forms the (n_perm, num_reg) matrix, so its only
# reference is the same backend's gpu_perm run through summarize_draws. That
# pins the pieces a matrix-free path could get wrong on its own: the Chan
# moments against nanmean / nanstd(ddof=1), the Z_STD_FLOOR guard, the
# all-NaN row convention in the max, and that pass 2 re-draws bit-identically
# to pass 1.

def _assert_summaries_match(got, ref, *, atol, label):
    """Assert two DrawSummary objects agree field by field."""
    for name in ('llr', 'mu', 'std', 'z_obs', 'max_stat'):
        _assert_cells_match(getattr(got, name), getattr(ref, name),
                            atol=atol, label=f'{label} {name}')


@requires_cuda
@pytest.mark.parametrize('design', DESIGNS)
@pytest.mark.parametrize('b', B_LIST)
def test_gpu_summarize_matches_the_matrix_reduction(b, design):
    """Streaming the reduction equals materializing then reducing."""
    prep = _prep(b, design)
    reg_active = _reg_active(prep)
    got = _call(draws_gpu.gpu_summarize, prep, n_perm=12, base_seed=0,
                acc_dtype=np.float64, reg_active=reg_active)
    ref = draws.summarize_draws(
        _call(draws_gpu.gpu_perm, prep, n_perm=12, base_seed=0,
              acc_dtype=np.float64), reg_active=reg_active)
    _assert_summaries_match(got, ref, atol=1e-9, label=f'b={b} {design}')


@requires_cuda
@pytest.mark.parametrize('design', DESIGNS)
def test_gpu_summarize_matches_the_cpu_anchor(design):
    """And equals the trust anchor's matrix, reduced the same way."""
    prep = _prep(2, design)
    reg_active = _reg_active(prep)
    got = _call(draws_gpu.gpu_summarize, prep, n_perm=12, base_seed=0,
                acc_dtype=np.float64, reg_active=reg_active)
    ref = draws.summarize_draws(
        _call(draws.cpu_reliable, prep, n_perm=12, base_seed=0),
        reg_active=reg_active)
    _assert_summaries_match(got, ref, atol=1e-9, label=design)


@requires_cuda
@pytest.mark.parametrize('perm_chunk', [1, 5, 32])
def test_gpu_summarize_is_chunk_invariant(perm_chunk):
    """Chunking splits both passes, so the moments must survive it.

    The Chan combine folds a chunk at a time, so its round-off does depend
    on chunk size -- but only at round-off, and the max-z null and the
    observed row must not move at all beyond that.
    """
    prep = _prep(2, 'general')
    reg_active = _reg_active(prep)
    kw = dict(n_perm=9, base_seed=0, acc_dtype=np.float64,
              reg_active=reg_active)
    got = _call(draws_gpu.gpu_summarize, prep, perm_chunk=perm_chunk, **kw)
    ref = _call(draws_gpu.gpu_summarize, prep, perm_chunk=4, **kw)
    _assert_summaries_match(got, ref, atol=1e-10,
                            label=f'perm_chunk={perm_chunk}')


@requires_cuda
def test_gpu_summarize_default_comparison_set_is_min_vox():
    """reg_active=None takes size >= min_vox, the set prep_tree derived."""
    prep = _prep(2, 'general', min_vox=4)
    got = _call(draws_gpu.gpu_summarize, prep, n_perm=8, base_seed=0,
                acc_dtype=np.float64)
    ref = _call(draws_gpu.gpu_summarize, prep, n_perm=8, base_seed=0,
                acc_dtype=np.float64, reg_active=_reg_active(prep))
    # bit-identical, not merely close: it is the same set either way
    for name in ('llr', 'mu', 'std', 'z_obs', 'max_stat'):
        np.testing.assert_array_equal(getattr(got, name), getattr(ref, name),
                                      err_msg=f'default reg_active {name}')


@requires_cuda
def test_gpu_summarize_empty_comparison_set_gives_nan_null():
    """No active region means no max to take, so the null is all NaN.

    fwer.max_over_active's convention on an empty set: NaN rather than a
    number, so from_max finds nothing to test against.
    """
    prep = _prep(2, 'general')
    num_reg = 2 * prep['exp'].y.shape[2] - 1
    got = _call(draws_gpu.gpu_summarize, prep, n_perm=6, base_seed=0,
                acc_dtype=np.float64,
                reg_active=np.zeros(num_reg, dtype=bool))
    assert np.isnan(got.max_stat).all()
    assert got.max_stat.shape == (6,)
