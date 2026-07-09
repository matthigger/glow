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
import json
import random

import numpy as np
import pytest

from glow._extra.benchmark import data
from glow._extra.benchmark.run import (glow_fit_for_prune, run_ana,
                                       run_inner_edge, run_min_size,
                                       run_perm_fwer, run_perm_inner, run_prune,
                                       run_segment, run_segment_time, run_stat,
                                       voxel_stat_walk)
from glow.analysis import AnalysisGLOW, AnalysisVBA
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import get_hotel_tr, get_wilks, stat_dict_inv
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
        # run_ana declares recurse_out_list=['score'], so the score dict is
        # expanded into out.score.<path> columns, not kept as one dict cell
        assert 'run_ana.out.score' not in row
        assert row['run_ana.out.score.num_vox'] == int((exp.mask_idx > -1).sum())
        assert {'run_ana.out.score.target.tp',
                'run_ana.out.score.min_pval'} <= set(row)
        # the build chained in (only possible via the shared-exp DAG edge),
        # carrying the swept seed onto the run_ana row
        assert row['data_factory_wgn.function'] == 'data_factory_wgn'
        assert row['data_factory_wgn.in.seed'] == seed

    def test_recipe_recorded_as_input_column(self):
        # the recipe is recorded as the in.ana column (its address-free repr) --
        # the key results / plot recover the method label from (no label stored)
        seed = _fresh_seed()
        data.RECORDER.records.clear()
        ana = AnalysisVBA(n_perm_fwer=15)
        exp = data.data_factory_wgn(shape=(5, 5, 5), b=2, num_img=20, a=2,
                                    seed=seed)
        run_ana(exp, ana, [])
        row = data.RECORDER.flatten_to_df().iloc[0]
        assert row['run_ana.in.ana'] == repr(ana)


# ---------------------------------------------------------------------------
# run_segment: oracle best-Dice region of one Ward tree (segmentation quality)
# ---------------------------------------------------------------------------

class TestRunSegment:
    def _planted(self):
        """A clean WGN exp with one planted effect; returns (exp, mask)."""
        exp = data.data_factory_wgn(shape=(7, 7, 7), b=2, num_img=20, a=1,
                                    seed=_fresh_seed())
        exp_eff, (mask,) = data.effect_factory(
            exp, effect_llr=0.1, extenter_cls=ExtenterMinVar,
            n_vox_frac=0.1, seed=0)
        return exp_eff, mask

    def test_returns_oracle_confusion_counts(self):
        exp, mask = self._planted()
        score = run_segment(exp, [mask], ClusterMode.FOCUS)
        assert set(score) == {'tp', 'fp', 'tn', 'fn'}
        # the best region's counts vs the planted support: tp + fn is exactly
        # the support size (every target voxel is hit or missed)
        assert score['tp'] + score['fn'] == int(mask.sum())

    def test_cluster_mode_is_a_cache_axis(self):
        exp, mask = self._planted()
        run_segment(exp, [mask], ClusterMode.FOCUS)
        # a different Ward mode is its own segmentation -> distinct cache entry
        assert not run_segment.check_call_in_cache(
            exp, [mask], ClusterMode.NAIVE)


# ---------------------------------------------------------------------------
# run_min_size: capture GLOW's per-perm (size -> max-z) staircases (no score)
# ---------------------------------------------------------------------------

class TestRunMinSize:
    def _planted(self):
        exp = data.data_factory_wgn(shape=(6, 6, 6), b=2, num_img=20, a=1,
                                    seed=_fresh_seed())
        exp_eff, (mask,) = data.effect_factory(
            exp, effect_llr=0.1, extenter_cls=ExtenterMinVar, n_vox_frac=0.1,
            seed=0)
        return exp_eff, mask

    def test_returns_one_curve_per_outer_perm(self):
        exp, mask = self._planted()
        curve = run_min_size(exp, [mask], n_perm_fwer=4, n_perm_inner=20)
        parsed = json.loads(curve)
        # n_perm_fwer + 1 staircases (k=0 observed); each lists [size, max_z]
        assert len(parsed) == 5
        for staircase in parsed:
            assert all(len(corner) == 2 for corner in staircase)

    def test_min_vox_floor_is_a_cache_axis(self):
        # the lower bound changes which regions get a z -> distinct curves
        exp, mask = self._planted()
        run_min_size(exp, [mask], n_perm_fwer=4, n_perm_inner=20,
                     min_vox_floor=1)
        assert not run_min_size.check_call_in_cache(
            exp, [mask], n_perm_fwer=4, n_perm_inner=20, min_vox_floor=3)


# ---------------------------------------------------------------------------
# run_inner_edge: capture max-z vs num_inner_perm from one inner sampling
# ---------------------------------------------------------------------------

class TestRunInnerEdge:
    def _planted(self):
        exp = data.data_factory_wgn(shape=(6, 6, 6), b=2, num_img=20, a=1,
                                    seed=_fresh_seed())
        exp_eff, (mask,) = data.effect_factory(
            exp, effect_llr=0.1, extenter_cls=ExtenterMinVar, n_vox_frac=0.1,
            seed=0)
        return exp_eff, mask

    def test_grid_and_matrix_shape(self):
        exp, mask = self._planted()
        curve = json.loads(run_inner_edge(
            exp, [mask], cluster_mode=ClusterMode.FOCUS, max_inner_perm=30,
            n_perm_fwer=4))
        grid = curve['num_inner_perm']
        # ascending, ends at max_inner_perm; one max-z row per outer perm
        assert grid == sorted(grid)
        assert grid[-1] == 30
        assert np.array(curve['max_z_null']).shape == (5, len(grid))

    def test_prefix_snapshot_equals_real_fit(self):
        # the whole point: the num_inner_perm=max snapshot reproduces a real
        # AnalysisGLOW fit at n_perm_inner=max (nested seeds -> shared draws), so
        # one sampling stands in for a fit at every num_inner_perm <= it. The
        # edge curve is built on the exact cpu_perm draw prefixes, so it models
        # the full (use_race=False) inner null -- the race trims per perm and
        # has no exact-prefix property to snapshot.
        exp, mask = self._planted()
        curve = json.loads(run_inner_edge(
            exp, [mask], cluster_mode=ClusterMode.FOCUS, max_inner_perm=30,
            n_perm_fwer=4))
        mz_max = np.array(curve['max_z_null'])[:, -1]
        ana = AnalysisGLOW(n_perm_fwer=4, n_perm_inner=30,
                           cluster_mode=ClusterMode.FOCUS).fit(
                               exp, use_race=False)
        np.testing.assert_allclose(mz_max, ana.max_z_null, rtol=1e-6, atol=1e-9)

    def test_cluster_mode_is_a_cache_axis(self):
        exp, mask = self._planted()
        run_inner_edge(exp, [mask], cluster_mode=ClusterMode.FOCUS,
                       max_inner_perm=30, n_perm_fwer=4)
        # a different Ward projection is its own edge -> distinct cache entry
        assert not run_inner_edge.check_call_in_cache(
            exp, [mask], cluster_mode=ClusterMode.GLM_ERROR, max_inner_perm=30,
            n_perm_fwer=4)

    def test_max_inner_perm_is_a_cache_axis(self):
        exp, mask = self._planted()
        run_inner_edge(exp, [mask], cluster_mode=ClusterMode.FOCUS,
                       max_inner_perm=30, n_perm_fwer=4)
        # a deeper sampling is a distinct capture (though its prefix agrees)
        assert not run_inner_edge.check_call_in_cache(
            exp, [mask], cluster_mode=ClusterMode.FOCUS, max_inner_perm=40,
            n_perm_fwer=4)


# ---------------------------------------------------------------------------
# run_stat: one VBA / CET MANCOVA-stat variant reading the shared voxel-walk
# ---------------------------------------------------------------------------

class TestRunStat:
    def _planted(self, seed=None):
        # b=2 so the multivariate stats differ
        seed = _fresh_seed() if seed is None else seed
        exp = data.data_factory_wgn(shape=(6, 6, 6), b=2, num_img=24, a=1,
                                    seed=seed)
        exp_eff, (mask,) = data.effect_factory(
            exp, effect_llr=0.15, extenter_cls=ExtenterMinVar, n_vox_frac=0.1,
            seed=0)
        return exp_eff, mask

    def test_matches_standalone_fit(self):
        # the shared walk is just a precompute of the same stat matrix, so a
        # variant reading it equals the standalone run_ana fit of that recipe
        exp, mask = self._planted()
        ana = AnalysisVBA(get_stat=get_wilks, n_perm_fwer=15, z_flag=True)
        assert run_stat(exp, [mask], ana, stat_dict_inv[get_wilks]) == run_ana(
            exp, ana, [mask])

    def test_variants_share_one_walk(self):
        # the first variant computes voxel_stat_walk; the rest are cache hits
        exp, mask = self._planted()
        assert not voxel_stat_walk.check_call_in_cache(exp, 15)
        run_stat(exp, [mask], AnalysisVBA(get_stat=get_wilks, n_perm_fwer=15),
                 stat_dict_inv[get_wilks])
        assert voxel_stat_walk.check_call_in_cache(exp, 15)

    def test_recipe_is_a_cache_axis(self):
        exp, mask = self._planted()
        ana = AnalysisVBA(get_stat=get_wilks, n_perm_fwer=15)
        run_stat(exp, [mask], ana, stat_dict_inv[get_wilks])
        # a different stat is a different variant -> distinct cache entry
        other = AnalysisVBA(get_stat=get_hotel_tr, n_perm_fwer=15)
        assert not run_stat.check_call_in_cache(
            exp, [mask], other, stat_dict_inv[get_hotel_tr])


# ---------------------------------------------------------------------------
# run_prune: one pruning rule scored on a shared GLOW fit
# ---------------------------------------------------------------------------

class TestRunPrune:
    _GLOW = dict(n_perm_fwer=4, n_perm_inner=8, alpha_fwer=0.05)

    def _planted(self):
        exp = data.data_factory_wgn(shape=(6, 6, 6), b=2, num_img=24, a=1,
                                    seed=_fresh_seed())
        exp_eff, (mask,) = data.effect_factory(
            exp, effect_llr=0.2, extenter_cls=ExtenterMinVar, n_vox_frac=0.1,
            seed=0)
        return exp_eff, mask

    def test_returns_prune_score(self):
        exp, mask = self._planted()
        score = run_prune(exp, [mask], 'maxllr', **self._GLOW)
        # per-effect confusion counts plus the output-region count
        assert {'n_selected', 'tp', 'fp', 'tn', 'fn'} <= set(score)

    def test_rules_share_one_fit(self):
        # the first rule fits GLOW; the other rules are glow_fit_for_prune hits
        exp, mask = self._planted()
        assert not glow_fit_for_prune.check_call_in_cache(
            exp, cluster_mode=ClusterMode.FOCUS, **self._GLOW)
        run_prune(exp, [mask], 'greedy', **self._GLOW)
        assert glow_fit_for_prune.check_call_in_cache(
            exp, cluster_mode=ClusterMode.FOCUS, **self._GLOW)

    def test_rule_is_a_cache_axis(self):
        exp, mask = self._planted()
        run_prune(exp, [mask], 'greedy', **self._GLOW)
        # a different rule is its own selection -> distinct cache entry
        assert not run_prune.check_call_in_cache(
            exp, [mask], 'dp', **self._GLOW)

    def test_bad_rule_raises(self):
        exp, mask = self._planted()
        with pytest.raises(ValueError):
            run_prune(exp, [mask], 'nope', **self._GLOW)


# ---------------------------------------------------------------------------
# runtime leaves: time one piece of GLOW, return the analyzed voxel count
# ---------------------------------------------------------------------------

class TestRuntimeLeaves:
    def _planted(self):
        exp = data.data_factory_wgn(shape=(6, 6, 6), b=2, num_img=20, a=1,
                                    seed=_fresh_seed())
        exp_eff, (mask,) = data.effect_factory(
            exp, effect_llr=0.1, extenter_cls=ExtenterMinVar, n_vox_frac=0.1,
            seed=0)
        return exp_eff, mask

    def test_perm_fwer_returns_num_vox(self):
        # the recorded measurement is time_sec; the return is the analyzed
        # voxel count, the sweep's size context
        exp, mask = self._planted()
        num_vox = run_perm_fwer(exp, [mask], n_perm_fwer=3, n_perm_inner=8)
        assert num_vox == int((exp.mask_idx > -1).sum())

    def test_perm_fwer_n_perm_is_a_cache_axis(self):
        exp, mask = self._planted()
        run_perm_fwer(exp, [mask], n_perm_fwer=3, n_perm_inner=8)
        # a different n_perm_fwer is its own timing -> distinct cache entry
        assert not run_perm_fwer.check_call_in_cache(
            exp, [mask], n_perm_fwer=5, n_perm_inner=8)

    def test_perm_inner_returns_num_vox(self):
        exp, mask = self._planted()
        num_vox = run_perm_inner(exp, [mask], n_perm_inner=8)
        assert num_vox == int((exp.mask_idx > -1).sum())

    def test_perm_inner_n_perm_is_a_cache_axis(self):
        exp, mask = self._planted()
        run_perm_inner(exp, [mask], n_perm_inner=8)
        assert not run_perm_inner.check_call_in_cache(
            exp, [mask], n_perm_inner=16)

    def test_segment_time_returns_num_vox(self):
        exp, mask = self._planted()
        num_vox = run_segment_time(exp, [mask], ClusterMode.FOCUS)
        assert num_vox == int((exp.mask_idx > -1).sum())

    def test_segment_time_mode_is_a_cache_axis(self):
        exp, mask = self._planted()
        run_segment_time(exp, [mask], ClusterMode.FOCUS)
        # a different Ward mode is its own segmentation -> distinct cache entry
        assert not run_segment_time.check_call_in_cache(
            exp, [mask], ClusterMode.NAIVE)

    def test_label_ignored_in_cache_key(self):
        exp, mask = self._planted()
        run_perm_inner(exp, [mask], n_perm_inner=8, label='GLOW-Focus')
        assert run_perm_inner.check_call_in_cache(
            exp, [mask], n_perm_inner=8, label='DIFFERENT')
