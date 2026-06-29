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
from glow._extra.benchmark.run import run_ana, run_segment
from glow.analysis import AnalysisGLOW, AnalysisVBA
from glow.analysis.cluster import ClusterMode
from glow.effect import ExtenterMinVar


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

    def test_label_ignored_in_cache_key(self):
        # label is recorded metadata, not a cache axis: a call differing only in
        # label is served from the first's entry (run_ana's ignore=['label']),
        # so renaming a method never invalidates its cached fit
        exp = _exp()
        ana = AnalysisVBA(n_perm_fwer=15)
        run_ana(exp, ana, [], label='VBA')
        assert run_ana.check_call_in_cache(exp, ana, [], label='DIFFERENT')


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

    def test_label_recorded_as_input_column(self):
        # the method label is recorded beside the score, as the in.label column
        seed = _fresh_seed()
        data.RECORDER.records.clear()
        exp = data.data_factory_wgn(shape=(5, 5, 5), b=2, num_img=20, a=2,
                                    seed=seed)
        run_ana(exp, AnalysisVBA(n_perm_fwer=15), [], label='VBA')
        row = data.RECORDER.flatten_to_df().iloc[0]
        assert row['run_ana.in.label'] == 'VBA'


# ---------------------------------------------------------------------------
# run_segment: oracle best-Dice region of one Ward tree (segmentation quality)
# ---------------------------------------------------------------------------

class TestRunSegment:
    _N_VOX_EFF = 20

    def _planted(self):
        """A clean WGN exp with one planted effect; returns (exp, [mask])."""
        exp = data.data_factory_wgn(shape=(7, 7, 7), b=2, num_img=20, a=1,
                                    seed=_fresh_seed())
        return data.effect_factory(
            exp, effect_llr=0.1, extenter_cls=ExtenterMinVar,
            n_vox=self._N_VOX_EFF, seed=0)

    def test_returns_oracle_confusion_counts(self):
        exp, mask = self._planted()
        score = run_segment(exp, [mask], ClusterMode.FOCUS)
        assert set(score) == {'tp', 'fp', 'tn', 'fn'}
        # the best region's counts vs the planted support: tp + fn is exactly
        # the support size (every target voxel is hit or missed)
        assert score['tp'] + score['fn'] == self._N_VOX_EFF

    def test_cluster_mode_is_a_cache_axis(self):
        exp, mask = self._planted()
        run_segment(exp, [mask], ClusterMode.FOCUS)
        # a different Ward mode is its own segmentation -> distinct cache entry
        assert not run_segment.check_call_in_cache(
            exp, [mask], ClusterMode.NAIVE)

    def test_label_ignored_in_cache_key(self):
        exp, mask = self._planted()
        run_segment(exp, [mask], ClusterMode.FOCUS, label='Focus')
        assert run_segment.check_call_in_cache(
            exp, [mask], ClusterMode.FOCUS, label='DIFFERENT')
