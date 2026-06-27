"""Tests for glow._extra.benchmark.run: run_ana fit + detection scoring.

run_ana shares data.py's MEMORY / RECORDER, so the cache + recorder decorator
*machinery* (args-hash keying, miss/hit, hit-does-not-record) is already
covered by test_data.py / test_recorder.py and not re-tested here. The score
dict's own schema / arithmetic is covered by test_score.py. These cover what is
novel to run_ana: that it returns a score dict uniform across heterogeneous
methods, that it fits a private copy (the caller's recipe is left un-fitted,
which is what keeps the cache key stable), and that it joins the shared
provenance DAG (a run_ana leaf carries the build that produced its exp).
"""
import random

import numpy as np
import pytest

from glow._extra.benchmark import data
from glow._extra.benchmark.run import run_ana
from glow.analysis import AnalysisGLOW, AnalysisVBA


def _fresh_seed() -> int:
    """A seed unlikely to already be in the on-disk cache, so a call misses."""
    return random.randrange(2 ** 31)


@pytest.fixture(autouse=True)
def _records_to_tmp(monkeypatch, tmp_path):
    """Mirror the shared recorder's per-hash files to a tmp dir, not the real one."""
    monkeypatch.setattr(data.RECORDER, 'folder', tmp_path)


def _exp(shape=(5, 5, 5), num_img=20, seed=None):
    """A small clean WGN experiment (fresh seed by default, so run_ana misses)."""
    return data.data_factory_wgn(shape=shape, b=2, num_img=num_img, a=2,
                                 seed=_fresh_seed() if seed is None else seed)


_SCORE_KEYS = {'num_vox', 'min_pval', 'n_pred', 'pred', 'target'}


# ---------------------------------------------------------------------------
# return contract: a score dict, uniform across heterogeneous methods
# ---------------------------------------------------------------------------

class TestContract:
    def test_vba_returns_score_dict(self):
        exp = _exp()
        score = run_ana(exp, AnalysisVBA(n_perm_fwer=15), [])
        assert _SCORE_KEYS <= set(score)
        # null target ([]) -> everything is background; num_vox is the volume
        assert score['num_vox'] == exp.y.shape[2]
        assert score['target']['fn'] == 0  # no target voxels to miss

    def test_glow_returns_score_dict(self):
        exp = _exp(shape=(4, 4, 4), num_img=16)
        score = run_ana(exp, AnalysisGLOW(n_perm_fwer=8, n_perm_inner=8), [])
        assert _SCORE_KEYS <= set(score)
        assert score['num_vox'] == exp.y.shape[2]

    def test_does_not_mutate_caller_recipe(self):
        # fit runs on a private copy: the caller's recipe stays un-fitted, which
        # is what keeps its joblib hash (the cache key) stable on reuse.
        exp = _exp()
        ana = AnalysisVBA(n_perm_fwer=15)
        run_ana(exp, ana, [])
        assert ana.pval is None
        assert ana.effect_list is None


# ---------------------------------------------------------------------------
# cache: keyed on the (exp, ana, target) triple (ana is a recipe arg)
# ---------------------------------------------------------------------------

class TestCacheKeyedOnRecipe:
    def test_miss_then_hit_on_reused_recipe(self):
        exp = _exp()
        ana = AnalysisVBA(n_perm_fwer=15)
        assert not run_ana.check_call_in_cache(exp, ana, [])
        run_ana(exp, ana, [])
        # the recipe is un-mutated, so the same object still hashes the same ->
        # the second call is a hit (the copy-fit fix; a mutating fit would miss)
        assert run_ana.check_call_in_cache(exp, ana, [])

    def test_distinct_recipe_is_a_distinct_key(self):
        exp = _exp()
        run_ana(exp, AnalysisVBA(n_perm_fwer=15), [])
        # a different recipe is not served by the first's cache entry
        assert not run_ana.check_call_in_cache(exp, AnalysisVBA(n_perm_fwer=16), [])

    def test_cached_result_matches_compute(self):
        exp = _exp()
        ana = AnalysisVBA(n_perm_fwer=15)
        s0 = run_ana(exp, ana, [])   # computed
        s1 = run_ana(exp, ana, [])   # served from cache
        assert s0 == s1


# ---------------------------------------------------------------------------
# provenance DAG: run_ana joins data.py's recorder graph via its exp input
# ---------------------------------------------------------------------------

class TestProvenanceDAG:
    def test_run_ana_leaf_carries_its_build_ancestor(self):
        # a fresh seed so the build is a cache miss and therefore records (a hit
        # would not, leaving run_ana an ancestor-less leaf)
        seed = _fresh_seed()
        data.RECORDER.records.clear()
        exp = data.data_factory_wgn(shape=(5, 5, 5), b=2, num_img=20, a=2,
                                    seed=seed)
        run_ana(exp, AnalysisVBA(n_perm_fwer=15), [])

        df = data.RECORDER.flatten_to_df()
        # one leaf -- the run_ana call; the build feeds its exp, so the build is
        # an ancestor, not a leaf
        assert len(df) == 1
        (row,) = df.to_dict('records')
        assert row['run_ana.function'] == 'run_ana'
        # the score dict is the leaf's recorded output
        assert isinstance(row['run_ana.out.score'], dict)
        # the build chained in (only possible via the shared-exp DAG edge),
        # carrying the swept seed onto the run_ana row
        assert row['data_factory_wgn.function'] == 'data_factory_wgn'
        assert row['data_factory_wgn.in.seed'] == seed
