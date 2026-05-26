"""Inner-perm backend equivalence, anchored on ``cpu_reliable``.

``cpu_reliable`` is the trust anchor: a thin wrapper over
``iter_mancova`` + ``get_llr`` per region, per draw.  Slow but
unambiguous, and an independent code path from the batched
``compute_llr_batched`` that the production backends share.  Both
production backends (``cpu_fast``, ``cpu_slow``) are validated
against it here -- agreement is fp64 round-off (~1e-10).

Run:
    ~/venv_glow/bin/pytest test/analysis/test_inner_perm.py -v
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import glow.graph
import glow.mask
from glow.analysis import inner_perm
from glow.analysis.cluster import cluster
from glow.analysis.mancova import decompose, is_intercept_only_nuisance
from glow.experiment.exper import Experiment


# ---------------------------------------------------------------------------
# Synthetic experiment builders

def _intercept_only_exp(seed=0, n_img=24, shape=(2, 3, 5)):
    """Intercept-only nuisance (Q0 = span(1)) -- exercises the fast path."""
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
    """Non-constant nuisance columns -- forces the general-Q0 path."""
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
# Pin the dispatcher preconditions cpu_fast / cpu_slow assume.

def test_intercept_only_fixture_picks_fast_path(prep_intercept_fp64):
    exp = prep_intercept_fp64['exp']
    assert is_intercept_only_nuisance(exp.x, exp.contrast)


def test_general_q0_fixture_picks_slow_path(prep_general_fp64):
    exp = prep_general_fp64['exp']
    assert not is_intercept_only_nuisance(exp.x, exp.contrast)


# ---------------------------------------------------------------------------
# CPU backends vs cpu_reliable -- fp64 round-off

def test_cpu_fast_matches_reliable(prep_intercept_fp64):
    """``cpu_fast`` (intercept-only Phase-1 hoist) matches ``cpu_reliable``."""
    draws_fast = _draws(inner_perm.cpu_fast_full, prep_intercept_fp64)
    draws_ref = _draws(inner_perm.cpu_reliable_full, prep_intercept_fp64)
    _assert_draws_match(draws_fast, draws_ref, atol=1e-10)


def test_cpu_slow_matches_reliable_intercept(prep_intercept_fp64):
    """``cpu_slow`` (batched einsum) matches ``cpu_reliable`` on intercept-only."""
    draws_slow = _draws(inner_perm.cpu_slow_full, prep_intercept_fp64)
    draws_ref = _draws(inner_perm.cpu_reliable_full, prep_intercept_fp64)
    _assert_draws_match(draws_slow, draws_ref, atol=1e-10)


def test_cpu_slow_matches_reliable_general(prep_general_fp64):
    """``cpu_slow`` matches ``cpu_reliable`` on general-Q0 too."""
    draws_slow = _draws(inner_perm.cpu_slow_full, prep_general_fp64)
    draws_ref = _draws(inner_perm.cpu_reliable_full, prep_general_fp64)
    _assert_draws_match(draws_slow, draws_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# moments_from_draws wrapper agrees with manual nanmean / nanstd

def test_moments_wrapper_matches_draws(prep_general_fp64):
    """``cpu_reliable`` (mu, std) == nanmean / nanstd(ddof=1) of draws."""
    draws = _draws(inner_perm.cpu_reliable_full, prep_general_fp64, n_perm=8)
    mu, std = _moments(inner_perm.cpu_reliable, prep_general_fp64, n_perm=8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu_manual = np.nanmean(draws, axis=0)
        std_manual = np.nanstd(draws, axis=0, ddof=1)
    finite = np.isfinite(mu) & np.isfinite(mu_manual)
    assert finite.any()
    assert np.allclose(mu[finite], mu_manual[finite], atol=1e-12)
    assert np.allclose(std[finite], std_manual[finite], atol=1e-12)


# ---------------------------------------------------------------------------
# min_vox NaN handling -- small regions must drop out of every backend.

@pytest.mark.parametrize('backend_full', [
    inner_perm.cpu_fast_full,
    inner_perm.cpu_slow_full,
    inner_perm.cpu_reliable_full,
])
def test_min_vox_drops_small_regions(prep_intercept_fp64, backend_full):
    """Regions with size < min_vox return NaN draws on every CPU backend."""
    prep = {**prep_intercept_fp64, 'min_vox': 4}
    draws = _draws(backend_full, prep)
    _, size = glow.graph.compute_llr_batched(
        prep['exp'], children=prep['children'],
        q0=prep['q0'], q1=prep['q1'], min_size=1)
    small = size < 4
    assert small.any(), 'fixture has no size<4 regions; raise min_vox'
    assert np.isnan(draws[:, small]).all(), \
        f'{backend_full.__name__}: size<min_vox cells leaked finite values'
