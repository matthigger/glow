"""Draw-matrix equivalence, anchored on cpu_reliable.

cpu_reliable is the trust anchor: a thin wrapper over iter_mancova +
get_llr per region, per draw. Slow but unambiguous, and an independent
code path from the batched compute_llr_batched that iter_llr_perm rides.
Both intercept-only and general-Q0 designs are exercised, and the two
agree cell by cell to fp64 round-off (~1e-10).

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
from glow.analysis import draws
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


