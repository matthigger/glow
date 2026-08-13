"""Inner-perm backend equivalence, anchored on ``cpu_reliable``.

``cpu_reliable`` is the trust anchor: a thin wrapper over
``iter_mancova`` + ``get_llr`` per region, per draw.  Slow but
unambiguous, and an independent code path from the batched
``compute_llr_batched`` that ``cpu_perm`` shares.  Both intercept-only
and general-Q0 inputs to ``cpu_perm`` are validated here -- moments
agree to fp64 round-off (~1e-10).

Run:
    ~/venv_glow/bin/pytest test/analysis/test_inner_perm.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

import glow.graph
import glow.mask
from glow.analysis import inner_perm
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
    """Build the kwargs every inner_perm backend wants."""
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


def _moments(fn, prep, *, n_perm=20, base_seed=12_345):
    return fn(exp=prep['exp'], base_seed=base_seed, n_perm=n_perm,
              q0=prep['q0'], q1=prep['q1'],
              children=prep['children'], min_vox=prep['min_vox'])


def _materialize_iter_llr_perm(prep, *, n_perm, base_seed):
    """Stack the chunks from ``iter_llr_perm`` into a (n_perm, num_reg)
    draws matrix.  Mirrors the old ``cpu_perm_full`` shape for tests
    that want per-draw equivalence checks."""
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


def _assert_moments_match(got, ref, *, atol):
    """Per-region (mu, std) agreement + identical NaN masks."""
    mu_g, std_g = got
    mu_r, std_r = ref
    assert mu_g.shape == mu_r.shape == std_g.shape == std_r.shape
    nan_mu_g, nan_mu_r = np.isnan(mu_g), np.isnan(mu_r)
    nan_std_g, nan_std_r = np.isnan(std_g), np.isnan(std_r)
    assert (nan_mu_g == nan_mu_r).all(), 'mu NaN masks differ'
    assert (nan_std_g == nan_std_r).all(), 'std NaN masks differ'
    fin_mu = ~nan_mu_g
    fin_std = ~nan_std_g
    assert fin_mu.any() and fin_std.any(), 'no finite cells to compare'
    d_mu = float(np.abs(mu_g[fin_mu] - mu_r[fin_mu]).max())
    d_std = float(np.abs(std_g[fin_std] - std_r[fin_std]).max())
    assert d_mu < atol, f'max |mu - ref| = {d_mu:.3e} > {atol:.3e}'
    assert d_std < atol, f'max |std - ref| = {d_std:.3e} > {atol:.3e}'


# ---------------------------------------------------------------------------
# cpu_perm vs cpu_reliable -- moments agree to fp64 round-off.
# Both backends now feed their per-draw output through the same Welford /
# Chan-parallel accumulator (_welford_moments), so differences trace back
# only to the underlying LLR computation (batched perm-LLR vs.
# per-region iter_mancova).

def test_cpu_perm_matches_reliable_intercept(prep_intercept_fp64):
    """``cpu_perm`` moments match ``cpu_reliable`` on intercept-only nuisance."""
    moments_perm = _moments(inner_perm.cpu_perm, prep_intercept_fp64)
    moments_ref = _moments(inner_perm.cpu_reliable, prep_intercept_fp64)
    _assert_moments_match(moments_perm, moments_ref, atol=1e-10)


def test_cpu_perm_matches_reliable_general(prep_general_fp64):
    """``cpu_perm`` moments match ``cpu_reliable`` on general Q0 too."""
    moments_perm = _moments(inner_perm.cpu_perm, prep_general_fp64)
    moments_ref = _moments(inner_perm.cpu_reliable, prep_general_fp64)
    _assert_moments_match(moments_perm, moments_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# Per-draw equivalence -- iter_llr_perm chunks (stacked) vs. cpu_reliable_full
# row-by-row.  Anchors that the underlying LLR computations agree before
# moments reduction; if this fails, the moments tests above can't isolate
# whether the divergence is in the LLR or in the accumulator.
#
# Both layers are retained on purpose: the per-draw layer localizes a
# divergence to the LLR kernel, while the moment layer above also exercises
# the Welford accumulator -- one without the other can't tell them apart.

def test_iter_llr_perm_matches_reliable_intercept(prep_intercept_fp64):
    """Per-draw output of ``iter_llr_perm`` matches ``cpu_reliable_full``
    cell-by-cell on intercept-only nuisance."""
    prep = prep_intercept_fp64
    draws_perm = _materialize_iter_llr_perm(prep, n_perm=6, base_seed=12_345)
    draws_ref = _draws(inner_perm.cpu_reliable_full, prep, n_perm=6)
    _assert_draws_match(draws_perm, draws_ref, atol=1e-10)


def test_iter_llr_perm_matches_reliable_general(prep_general_fp64):
    """Same on general Q0."""
    prep = prep_general_fp64
    draws_perm = _materialize_iter_llr_perm(prep, n_perm=6, base_seed=12_345)
    draws_ref = _draws(inner_perm.cpu_reliable_full, prep, n_perm=6)
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
    draws_rel = _draws(inner_perm.cpu_reliable_full, prep, n_perm=3,
                       base_seed=0)
    _assert_draws_match(draws_perm[:1], ref[None], atol=1e-10)
    _assert_draws_match(draws_rel[:1], ref[None], atol=1e-10)


def test_row_0_is_the_observed_draw_general(prep_general_fp64):
    """Same on general Q0, where the seed-0 gather is not a no-op."""
    prep = prep_general_fp64
    ref = _observed_llr(prep)
    draws_perm = _materialize_iter_llr_perm(prep, n_perm=3, base_seed=0)
    draws_rel = _draws(inner_perm.cpu_reliable_full, prep, n_perm=3,
                       base_seed=0)
    _assert_draws_match(draws_perm[:1], ref[None], atol=1e-10)
    _assert_draws_match(draws_rel[:1], ref[None], atol=1e-10)


def test_backends_agree_at_base_seed_0(prep_general_fp64):
    """Every row agrees at base_seed = 0, not just the null rows."""
    prep = prep_general_fp64
    draws_perm = _materialize_iter_llr_perm(prep, n_perm=4, base_seed=0)
    draws_ref = _draws(inner_perm.cpu_reliable_full, prep, n_perm=4,
                       base_seed=0)
    _assert_draws_match(draws_perm, draws_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# min_vox NaN handling -- small regions must drop out of every backend.

def test_min_vox_drops_small_regions_cpu_perm(prep_intercept_fp64):
    """Regions with size < min_vox return NaN moments under ``cpu_perm``."""
    prep = {**prep_intercept_fp64, 'min_vox': 4}
    mu, std = _moments(inner_perm.cpu_perm, prep)
    _, size = glow.graph.compute_llr_batched(
        prep['exp'], children=prep['children'],
        q0=prep['q0'], q1=prep['q1'], min_size=1)
    small = size < 4
    assert small.any(), 'fixture has no size<4 regions; raise min_vox'
    assert np.isnan(mu[small]).all(), \
        'cpu_perm: size<min_vox cells leaked finite mu'
    assert np.isnan(std[small]).all(), \
        'cpu_perm: size<min_vox cells leaked finite std'


def test_min_vox_drops_small_regions_cpu_reliable_full(prep_intercept_fp64):
    """Regions with size < min_vox return NaN per-draw under
    ``cpu_reliable_full``."""
    prep = {**prep_intercept_fp64, 'min_vox': 4}
    draws = _draws(inner_perm.cpu_reliable_full, prep)
    _, size = glow.graph.compute_llr_batched(
        prep['exp'], children=prep['children'],
        q0=prep['q0'], q1=prep['q1'], min_size=1)
    small = size < 4
    assert small.any(), 'fixture has no size<4 regions; raise min_vox'
    assert np.isnan(draws[:, small]).all(), \
        'cpu_reliable_full: size<min_vox cells leaked finite values'


