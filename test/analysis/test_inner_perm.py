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


# ---------------------------------------------------------------------------
# cpu_perm_race vs cpu_perm -- the survivor race reproduces the full run on the
# outputs the FWER test consumes.  The race draws the same seeds as cpu_perm
# (burn-in identical, survivors drawn to full n_perm through the low-rank
# kernel), so (a) survivor moments match cpu_perm to fp round-off regardless of
# what gets trimmed, and (b) with a real effect the arg-max region survives, so
# the per-perm max-z -- the only number the outer FWER loop reads -- is
# reproduced exactly.  The trim never touches validity (see cpu_perm_race).

def _planted_exp(*, intercept_only, seed=1, n_img=30, shape=(6, 6, 6),
                 beta=1.5):
    """A builder exp with a strong effect planted along the first interest
    column into a contiguous voxel block, giving a stable dominant max-z
    region (so the arg-max reliably survives the trim)."""
    exp = (_intercept_only_exp(seed=seed, n_img=n_img, shape=shape)
           if intercept_only else
           _general_q0_exp(seed=seed, n_img=n_img, shape=shape))
    interest = int(np.where(exp.contrast)[0][0])
    y = exp.y.copy()
    blk = slice(0, max(8, y.shape[2] // 4))
    y[:, :, blk] += beta * exp.x[interest][None, :, None]
    return Experiment(x=exp.x, y=y, contrast=exp.contrast,
                      mask_idx=exp.mask_idx)


def _race_survivors(prep, *, base_seed, race_init, p_keep_thresh):
    """Reproduce the race burn-in + trim to recover the survivor mask and the
    observed LLR (so a test can check survivor moments against cpu_perm)."""
    exp = prep['exp']
    num_vox, num_img = exp.y.shape[2], exp.y.shape[1]
    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=prep['children'], num_vox=num_vox)
    llr_obs, size = glow.graph.compute_llr_batched(
        exp, children=prep['children'], q0=prep['q0'], q1=prep['q1'],
        min_size=prep['min_vox'])
    perms = np.stack([permute._perm_indices(base_seed + i, num_img)
                      for i in range(race_init)])
    n = np.zeros(size.size)
    mean = np.zeros(size.size)
    M2 = np.zeros(size.size)
    for chunk in glow.graph.iter_llr_perm(
            y=exp.y, q0=prep['q0'], q1=prep['q1'], perms=perms,
            leaf_ord=leaf_ord, region_l=region_l, region_h=region_h,
            min_size=prep['min_vox']):
        n, mean, M2 = inner_perm._welford_combine(chunk, n, mean, M2)
    mu_bi, std_bi = inner_perm._welford_finalize(n, mean, M2)
    keep = inner_perm._race_keep(
        llr_obs=llr_obs, mu=mu_bi, std=std_bi, n=n, size=size,
        min_vox=prep['min_vox'], p_keep_thresh=p_keep_thresh)
    return keep, llr_obs, size


@pytest.mark.parametrize('intercept_only', [False, True],
                         ids=['general', 'intercept'])
def test_race_reproduces_cpu_perm_maxz(intercept_only):
    """The race's per-perm max-z equals the full cpu_perm's, and the full
    run's arg-max region survives -- for general and intercept-only Q0."""
    prep = _prep(_planted_exp(intercept_only=intercept_only), min_vox=2)
    base_seed, n_perm, race_init, pk = 777, 40, 10, 1e-6

    mu_f, std_f = inner_perm.cpu_perm(
        exp=prep['exp'], base_seed=base_seed, n_perm=n_perm,
        q0=prep['q0'], q1=prep['q1'], children=prep['children'],
        min_vox=prep['min_vox'])
    keep, llr_obs, size = _race_survivors(
        prep, base_seed=base_seed, race_init=race_init, p_keep_thresh=pk)
    mu_r, std_r = inner_perm.cpu_perm_race(
        exp=prep['exp'], llr_obs=llr_obs, base_seed=base_seed, n_perm=n_perm,
        q0=prep['q0'], q1=prep['q1'], children=prep['children'],
        min_vox=prep['min_vox'], race_init=race_init, p_keep_thresh=pk)

    with np.errstate(divide='ignore', invalid='ignore'):
        z_f = (llr_obs - mu_f) / std_f
        z_r = (llr_obs - mu_r) / std_r
    active = (size >= prep['min_vox']) & np.isfinite(z_f) & np.isfinite(z_r)
    assert active.any()
    amax = int(np.argmax(np.where(active, z_f, -np.inf)))

    assert keep[amax], 'the full-run arg-max region was trimmed (recall miss)'
    assert keep.sum() >= 1
    np.testing.assert_allclose(z_r[amax], z_f[amax], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        float(np.nanmax(np.where(active, z_r, np.nan))),
        float(np.nanmax(np.where(active, z_f, np.nan))),
        rtol=1e-6, atol=1e-8)


def test_race_survivor_moments_match_cpu_perm(prep_general_fp64):
    """Survivor moments equal cpu_perm to fp round-off (independent of what
    gets trimmed): survivors are drawn to full n_perm on the same seeds."""
    prep = prep_general_fp64
    base_seed, n_perm, race_init, pk = 4242, 40, 10, 1e-6
    keep, llr_obs, _ = _race_survivors(
        prep, base_seed=base_seed, race_init=race_init, p_keep_thresh=pk)
    mu_f, std_f = inner_perm.cpu_perm(
        exp=prep['exp'], base_seed=base_seed, n_perm=n_perm,
        q0=prep['q0'], q1=prep['q1'], children=prep['children'],
        min_vox=prep['min_vox'])
    mu_r, std_r = inner_perm.cpu_perm_race(
        exp=prep['exp'], llr_obs=llr_obs, base_seed=base_seed, n_perm=n_perm,
        q0=prep['q0'], q1=prep['q1'], children=prep['children'],
        min_vox=prep['min_vox'], race_init=race_init, p_keep_thresh=pk)
    assert keep.sum() >= 1
    fin = keep & np.isfinite(mu_f) & np.isfinite(mu_r)
    assert fin.any()
    np.testing.assert_allclose(mu_r[fin], mu_f[fin], rtol=1e-7, atol=1e-9)
    fin_s = keep & np.isfinite(std_f) & np.isfinite(std_r)
    np.testing.assert_allclose(std_r[fin_s], std_f[fin_s],
                               rtol=1e-7, atol=1e-9)


def test_race_keep_band():
    """_race_keep is a scale-free k_sigma band around the interim leader:
    keeps the leader and near rivals, drops far-below, small, and non-finite
    regions."""
    n = np.full(5, 50.0)
    std = np.ones(5)
    mu = np.zeros(5)
    # z_hat = llr_obs here (mu=0, std=1). Region 3 has the highest z but is
    # too small; region 4 is non-finite. Leader is region 0 (z=10).
    llr_obs = np.array([10.0, 9.5, 2.0, 100.0, np.nan])
    size = np.array([10, 10, 10, 1, 10])
    keep = inner_perm._race_keep(
        llr_obs=llr_obs, mu=mu, std=std, n=n, size=size, min_vox=2,
        p_keep_thresh=1e-6)
    assert keep.tolist() == [True, True, False, False, False]
