"""Draw-matrix equivalence, anchored on cpu_reliable.

cpu_reliable is the trust anchor: a thin wrapper over iter_mancova +
get_llr per region, per draw. Slow but unambiguous, and an independent
code path from the batched compute_llr_batched that iter_llr_perm rides.
Both intercept-only and general-Q0 designs are exercised, and the two
agree cell by cell to fp64 round-off (~1e-10).

Nothing here fits with the anchor -- AnalysisGLOW.fit takes cpu_summary,
the streamed batched kernel -- which is exactly why these comparisons
carry the weight: they are what says the fast path computes the anchor's
statistic. Three layers of it, each against the anchor or against a
materialized reference: cpu_batched's matrix, cpu_summary's reduction of
it, and (in test_fit_gpu.py) the device's.

The reserved-0 convention gets its own section: base_seed = 0 must put
the observed draw in row 0, which the equivalence cases above cannot see
because they run at base_seed = 12_345 where every row is permuted.

Run:
    ~/venv_glow/bin/pytest test/analysis/test_draws.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

import glow.graph
import glow.mask
from glow.analysis import draws, fwer
from glow.analysis._base import Analysis
from glow.analysis.cluster import cluster
from glow.analysis.mancova import decompose
from glow.experiment import permute
from glow.experiment.exper import Experiment


# ---------------------------------------------------------------------------
# Synthetic experiment builders

def _intercept_only_exp(seed=0, n_img=24, shape=(2, 3, 5)):
    """Intercept-only nuisance (Q0 = span(1))."""
    rng = np.random.default_rng(seed)
    V = int(np.prod(shape))
    mask_idx = np.arange(V, dtype=np.int64).reshape(shape)
    y = rng.standard_normal((2, n_img, V)).astype(np.float64)
    x = np.vstack([
        np.ones(n_img, dtype=np.float64),
        rng.standard_normal(n_img).astype(np.float64),
        rng.standard_normal(n_img).astype(np.float64),
    ])
    contrast = np.array([False, True, True])
    return Experiment(x=x, y=y, contrast=contrast, mask_idx=mask_idx)


def _general_q0_exp(seed=0, n_img=24, shape=(3, 4, 4)):
    """Non-constant nuisance columns (Q0 does not commute with P)."""
    rng = np.random.default_rng(seed)
    rows = [np.ones(n_img, dtype=np.float64)]
    contrast = [False]
    for _ in range(2):
        c = rng.standard_normal(n_img).astype(np.float64)
        c -= c.mean(); c /= c.std() + 1e-9
        rows.append(c); contrast.append(False)
    for _ in range(2):
        c = rng.standard_normal(n_img).astype(np.float64)
        c -= c.mean(); c /= c.std() + 1e-9
        rows.append(c); contrast.append(True)
    x = np.vstack(rows)
    contrast = np.array(contrast)

    V = int(np.prod(shape))
    mask = np.zeros(shape, dtype=bool); mask.reshape(-1)[:V] = True
    y = rng.standard_normal((2, n_img, V)).astype(np.float64)
    return Experiment(x=x, y=y, contrast=contrast,
                      mask_idx=glow.mask.get_mask_idx(mask))


def _prep(exp, *, min_vox=2):
    """Build the kwargs every draws backend wants."""
    q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)
    children = cluster(exp=exp, mode='Focus')
    return dict(exp=exp, q0=q0, q1=q1, children=children, min_vox=min_vox)


# ---------------------------------------------------------------------------
# Fixtures

@pytest.fixture(scope='module')
def prep_intercept_fp64():
    return _prep(_intercept_only_exp(seed=0))


@pytest.fixture(scope='module')
def prep_general_fp64():
    return _prep(_general_q0_exp(seed=0))


# ---------------------------------------------------------------------------
# Helpers

def _draws(fn, prep, *, n_perm=6, base_seed=12_345):
    return fn(exp=prep['exp'], base_seed=base_seed, n_perm=n_perm,
              q0=prep['q0'], q1=prep['q1'],
              children=prep['children'], min_vox=prep['min_vox'])



def _materialize_iter_llr_perm(prep, *, n_perm, base_seed):
    """Stack the chunks from iter_llr_perm into a (n_perm, num_reg) matrix.

    The batched CPU route, in the same shape cpu_reliable returns, so the
    two can be compared cell by cell.
    """
    exp = prep['exp']
    num_vox = exp.y.shape[2]
    num_img = exp.y.shape[1]
    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=prep['children'], num_vox=num_vox)
    perms = np.empty((n_perm, num_img), dtype=np.int64)
    for i in range(n_perm):
        perms[i] = permute._perm_indices(base_seed + i, num_img)
    return np.vstack(list(glow.graph.iter_llr_perm(
        y=exp.y, q0=prep['q0'], q1=prep['q1'], perms=perms,
        leaf_ord=leaf_ord, region_l=region_l, region_h=region_h,
        min_size=prep['min_vox'])))


def _assert_draws_match(draws, ref, *, atol):
    """Per-cell agreement + identical NaN masks."""
    assert draws.shape == ref.shape
    nan_d, nan_r = np.isnan(draws), np.isnan(ref)
    assert (nan_d == nan_r).all(), \
        f'NaN masks differ in {(nan_d != nan_r).sum()} cells'
    finite = ~nan_d
    assert finite.any(), 'no finite cells to compare'
    diff = np.abs(draws[finite] - ref[finite]).max()
    assert diff < atol, f'max abs diff {diff:.3e} > {atol:.3e}'


# ---------------------------------------------------------------------------
# Per-draw equivalence -- iter_llr_perm chunks (stacked) against cpu_reliable
# row by row. Two independent implementations of the same statistic: the
# batched perm-LLR kernel with its hoisted Phase-1 statistics and
# cumsum-and-diff region scan, against the per-region iter_mancova walk.

def test_iter_llr_perm_matches_reliable_intercept(prep_intercept_fp64):
    """iter_llr_perm matches cpu_reliable cell by cell, intercept-only."""
    prep = prep_intercept_fp64
    draws_perm = _materialize_iter_llr_perm(prep, n_perm=6, base_seed=12_345)
    draws_ref = _draws(draws.cpu_reliable, prep, n_perm=6)
    _assert_draws_match(draws_perm, draws_ref, atol=1e-10)


def test_iter_llr_perm_matches_reliable_general(prep_general_fp64):
    """Same on general Q0."""
    prep = prep_general_fp64
    draws_perm = _materialize_iter_llr_perm(prep, n_perm=6, base_seed=12_345)
    draws_ref = _draws(draws.cpu_reliable, prep, n_perm=6)
    _assert_draws_match(draws_perm, draws_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# The reserved-0 convention: base_seed = 0 puts the OBSERVED draw in row 0.
#
# The equivalence tests above run at base_seed = 12_345, where every row is a
# permuted draw. That leaves row 0 of a base_seed = 0 matrix -- the one row
# AnalysisGLOW.fit reads as the observed LLR, and the one row
# Analysis.z_score_stat standardizes -- unchecked. A backend that permuted at
# seed 0 would pass every test above and silently replace the observed
# statistic with a null one.

def _observed_llr(prep):
    """Compute the unpermuted per-region LLR, no permutation machinery."""
    llr, _ = glow.graph.compute_llr_batched(
        prep['exp'], children=prep['children'], q0=prep['q0'],
        q1=prep['q1'], min_size=prep['min_vox'])
    return llr


def test_row_0_is_the_observed_draw_intercept(prep_intercept_fp64):
    """Row 0 at base_seed = 0 is the unpermuted LLR, on both CPU paths."""
    prep = prep_intercept_fp64
    ref = _observed_llr(prep)
    draws_perm = _materialize_iter_llr_perm(prep, n_perm=3, base_seed=0)
    draws_rel = _draws(draws.cpu_reliable, prep, n_perm=3,
                       base_seed=0)
    _assert_draws_match(draws_perm[:1], ref[None], atol=1e-10)
    _assert_draws_match(draws_rel[:1], ref[None], atol=1e-10)


def test_row_0_is_the_observed_draw_general(prep_general_fp64):
    """Same on general Q0, where the seed-0 gather is not a no-op."""
    prep = prep_general_fp64
    ref = _observed_llr(prep)
    draws_perm = _materialize_iter_llr_perm(prep, n_perm=3, base_seed=0)
    draws_rel = _draws(draws.cpu_reliable, prep, n_perm=3,
                       base_seed=0)
    _assert_draws_match(draws_perm[:1], ref[None], atol=1e-10)
    _assert_draws_match(draws_rel[:1], ref[None], atol=1e-10)


def test_backends_agree_at_base_seed_0(prep_general_fp64):
    """Every row agrees at base_seed = 0, not just the null rows."""
    prep = prep_general_fp64
    draws_perm = _materialize_iter_llr_perm(prep, n_perm=4, base_seed=0)
    draws_ref = _draws(draws.cpu_reliable, prep, n_perm=4,
                       base_seed=0)
    _assert_draws_match(draws_perm, draws_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# min_vox NaN handling -- small regions must drop out of every backend.

def test_min_vox_drops_small_regions_cpu_reliable(prep_intercept_fp64):
    """Regions with size < min_vox come back NaN in every draw."""
    prep = {**prep_intercept_fp64, 'min_vox': 4}
    got = _draws(draws.cpu_reliable, prep)
    _, size = glow.graph.compute_llr_batched(
        prep['exp'], children=prep['children'],
        q0=prep['q0'], q1=prep['q1'], min_size=1)
    small = size < 4
    assert small.any(), 'fixture has no size<4 regions; raise min_vox'
    assert np.isnan(got[:, small]).all(), \
        'cpu_reliable: size<min_vox cells leaked finite values'


def test_min_vox_drops_small_regions_cpu_batched(prep_intercept_fp64):
    """Same of the batched path, which derives NaN a different way.

    cpu_reliable skips a small region; the kernel masks it after a scan
    that ran over it anyway, so this is not the same code answering twice.
    """
    prep = {**prep_intercept_fp64, 'min_vox': 4}
    got = _draws(draws.cpu_batched, prep)
    _, size = glow.graph.compute_llr_batched(
        prep['exp'], children=prep['children'],
        q0=prep['q0'], q1=prep['q1'], min_size=1)
    small = size < 4
    assert small.any(), 'fixture has no size<4 regions; raise min_vox'
    assert np.isnan(got[:, small]).all(), \
        'cpu_batched: size<min_vox cells leaked finite values'


# ---------------------------------------------------------------------------
# The two fast CPU entry points against the anchor. A fit's draws come from
# cpu_summary, which streams cpu_batched's chunks, so the anchor holds both:
# cpu_batched cell by cell, cpu_summary against the same reduction taken over
# a materialized matrix (summarize_draws). This is the comparison that keeps
# cpu_reliable in the tree -- it is never fitted with, only compared to.

def _reg_active(prep):
    """The comparison set a fit would derive: size >= min_vox."""
    _, size = glow.graph.compute_llr_batched(
        prep['exp'], children=prep['children'], q0=prep['q0'],
        q1=prep['q1'], min_size=1)
    return size >= prep['min_vox']


def _summary(prep, *, n_perm=6, base_seed=0, **over):
    return draws.cpu_summary(
        exp=prep['exp'], base_seed=base_seed, n_perm=n_perm,
        q0=prep['q0'], q1=prep['q1'], children=prep['children'],
        min_vox=prep['min_vox'], reg_active=_reg_active(prep), **over)


def _assert_summaries_match(got, ref, *, atol):
    """Every DrawSummary field agrees, NaN masks included."""
    for field in ('llr', 'mu', 'std', 'z_obs', 'max_stat'):
        g, r = getattr(got, field), getattr(ref, field)
        assert g.shape == r.shape, f'{field}: {g.shape} != {r.shape}'
        nan_g, nan_r = np.isnan(g), np.isnan(r)
        assert (nan_g == nan_r).all(), f'{field}: NaN masks differ'
        finite = ~nan_g
        if finite.any():
            diff = np.abs(g[finite] - r[finite]).max()
            assert diff < atol, f'{field}: max abs diff {diff:.3e}'


def test_cpu_batched_matches_reliable_intercept(prep_intercept_fp64):
    """cpu_batched matches the anchor cell by cell, intercept-only."""
    prep = prep_intercept_fp64
    got = _draws(draws.cpu_batched, prep, n_perm=6)
    ref = _draws(draws.cpu_reliable, prep, n_perm=6)
    _assert_draws_match(got, ref, atol=1e-10)


def test_cpu_batched_matches_reliable_general(prep_general_fp64):
    """Same on general Q0, where the FL re-projection is not a no-op."""
    prep = prep_general_fp64
    got = _draws(draws.cpu_batched, prep, n_perm=6)
    ref = _draws(draws.cpu_reliable, prep, n_perm=6)
    _assert_draws_match(got, ref, atol=1e-10)


def test_cpu_batched_row_0_is_the_observed_draw(prep_general_fp64):
    """base_seed = 0 leaves row 0 unpermuted on the batched path too."""
    prep = prep_general_fp64
    got = _draws(draws.cpu_batched, prep, n_perm=3, base_seed=0)
    _assert_draws_match(got[:1], _observed_llr(prep)[None], atol=1e-10)


def test_cpu_summary_matches_the_materialized_reduction(prep_general_fp64):
    """Streaming the reduction changes nothing about it.

    Held against summarize_draws over cpu_batched's own matrix, so the
    only difference under test is streaming -- same draws, same
    conventions, two ways of folding them.
    """
    prep = prep_general_fp64
    ref = draws.summarize_draws(
        _draws(draws.cpu_batched, prep, n_perm=6, base_seed=0),
        reg_active=_reg_active(prep))
    _assert_summaries_match(_summary(prep, n_perm=6), ref, atol=1e-12)


def test_cpu_summary_matches_the_anchors_summary(prep_general_fp64):
    """The whole CPU fast path against the whole anchor path.

    What AnalysisGLOW.fit(cpu_anchor=True) and fit() respectively compute,
    compared at the one object a fit actually keeps.
    """
    prep = prep_general_fp64
    ref = draws.summarize_draws(
        _draws(draws.cpu_reliable, prep, n_perm=6, base_seed=0),
        reg_active=_reg_active(prep))
    _assert_summaries_match(_summary(prep, n_perm=6), ref, atol=1e-9)


@pytest.mark.parametrize('perm_chunk', [1, 2, 3, 5, 16])
def test_cpu_summary_is_chunk_invariant(prep_general_fp64, perm_chunk):
    """perm_chunk is a throughput knob and moves no number.

    Chunking sets how the Chan accumulators are combined, so agreement is
    to round-off rather than bitwise -- and a chunk size that does not
    divide n_perm is the case that catches an off-by-one in the tail.
    """
    prep = prep_general_fp64
    ref = _summary(prep, n_perm=6, perm_chunk=8)
    _assert_summaries_match(_summary(prep, n_perm=6, perm_chunk=perm_chunk),
                            ref, atol=1e-11)


def test_cpu_summary_max_stat_is_in_draw_order(prep_general_fp64):
    """max_stat[i] is draw i's max, not a sorted or reordered null.

    Entry 0 must be the observed draw's: MaxStatPerm.from_max reads the
    p-value off the null's rank against it, so a permuted order would
    still look plausible while being wrong.
    """
    prep = prep_general_fp64
    summary = _summary(prep, n_perm=5)
    active = _reg_active(prep)
    matrix = _draws(draws.cpu_batched, prep, n_perm=5, base_seed=0)
    z, _, _ = Analysis.z_score_stat(matrix)
    np.testing.assert_allclose(summary.max_stat,
                               fwer.max_over_active(z, active),
                               rtol=0, atol=1e-11)
    np.testing.assert_allclose(summary.max_stat[0],
                               np.nanmax(summary.z_obs[active]),
                               rtol=0, atol=1e-11)


def test_cpu_summary_with_an_empty_comparison_set(prep_general_fp64):
    """No active region leaves max_stat all-NaN, not empty or zero.

    fwer.max_over_active's convention, reached per chunk here, so this
    pins that streaming did not turn a NaN draw into a number.
    """
    prep = prep_general_fp64
    summary = draws.cpu_summary(
        exp=prep['exp'], base_seed=0, n_perm=4, q0=prep['q0'],
        q1=prep['q1'], children=prep['children'], min_vox=prep['min_vox'],
        reg_active=np.zeros(prep['children'].shape[0]
                            + prep['exp'].y.shape[2], dtype=bool))
    assert summary.max_stat.shape == (4,)
    assert np.isnan(summary.max_stat).all()


