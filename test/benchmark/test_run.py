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

from glow._extra.benchmark import data, run
from glow._extra.benchmark.run import (glow_fit_for_prune, run_ana,
                                       run_ana_time_1perm, run_inner_edge,
                                       run_prune, run_segment, run_stat,
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
    """Build a small clean WGN experiment and name it.

    Returns (exp, parent_uid): every leaf requires its exp's declared uid, and
    the driver names it from the cell's kwargs -- so do the same here rather
    than inventing one, or two distinct experiments could share a cache entry
    (exp itself is ignored by the key; see glow._extra.benchmark.recipe).

    A fresh seed by default, so a leaf call misses the on-disk cache.
    """
    kwargs = dict(source='wgn', shape=shape, b=2, num_img=num_img, a=2,
                  seed=_fresh_seed() if seed is None else seed)
    return data.data_factory(**kwargs), data.data_recipe(kwargs).uid


def _planted_cell(kwargs_data, kwargs_effect):
    """Build one (data, effect) cell; return (exp, mask, parent_uid).

    The driver's chain in miniature: name the data cell, plant on it with that
    uid as the parent, and hand back the planted exp's own uid for the leaf.

    Args:
        kwargs_data (dict): one data_factory cell (including its source).
        kwargs_effect (dict): one effect_factory cell (kind defaults to
            'single').

    Returns:
        exp: the planted Experiment.
        mask (np.array): (X, Y, Z) bool, its single realized support.
        parent_uid (str): the planted exp's declared uid.
    """
    uid_data = data.data_recipe(kwargs_data).uid
    exp = data.data_factory(**kwargs_data)
    exp_eff, (mask,) = data.effect_factory(exp, parent_uid=uid_data,
                                           **kwargs_effect)
    return exp_eff, mask, data.effect_recipe(kwargs_effect, uid_data).uid


_SCORE_KEYS = {'num_vox', 'min_pval', 'n_pred', 'pred', 'target'}


class _SpyVBA(AnalysisVBA):
    """A VBA recipe that records the fit kwargs it was called with.

    Module level, not a local class: run_ana's key hashes the recipe, and
    joblib cannot pickle a class defined inside a test. The log is a class
    attribute so it survives run_ana's deepcopy of the recipe.
    """

    seen = {}

    def fit(self, exp, _stat=None, **kwargs):
        """Log the fit kwargs, then fit as AnalysisVBA does."""
        type(self).seen = dict(kwargs)
        return super().fit(exp, _stat, **kwargs)


# ---------------------------------------------------------------------------
# return contract: a score dict, uniform across heterogeneous methods
# ---------------------------------------------------------------------------

class TestContract:
    def test_vba_returns_score_dict(self):
        exp, uid = _exp()
        score = run_ana(exp, AnalysisVBA(n_perm_fwer=15), [], parent_uid=uid)
        assert _SCORE_KEYS <= set(score)
        # null target ([]) -> everything is background; num_vox is the volume
        assert score['num_vox'] == exp.y.shape[2]
        assert score['target']['fn'] == 0  # no target voxels to miss

    def test_glow_returns_score_dict(self):
        exp, uid = _exp(shape=(4, 4, 4), num_img=16)
        score = run_ana(exp, AnalysisGLOW(n_perm_fwer=8, n_perm_inner=8), [],
                        parent_uid=uid)
        assert _SCORE_KEYS <= set(score)
        assert score['num_vox'] == exp.y.shape[2]

    def test_does_not_mutate_caller_recipe(self):
        # fit runs on a private copy: the caller's recipe stays un-fitted, which
        # is what keeps its joblib hash (the cache key) stable on reuse.
        exp, uid = _exp()
        ana = AnalysisVBA(n_perm_fwer=15)
        run_ana(exp, ana, [], parent_uid=uid)
        assert ana.pval is None
        assert ana.effect_list is None


# ---------------------------------------------------------------------------
# cache: keyed on the (exp, ana, target) triple (ana is a recipe arg)
# ---------------------------------------------------------------------------

class TestCacheKeyedOnRecipe:
    def test_miss_then_hit_on_reused_recipe(self):
        exp, uid = _exp()
        ana = AnalysisVBA(n_perm_fwer=15)
        assert not run_ana.check_call_in_cache(exp, ana, [], parent_uid=uid)
        run_ana(exp, ana, [], parent_uid=uid)
        # the recipe is un-mutated, so the same object still hashes the same ->
        # the second call is a hit (the copy-fit fix; a mutating fit would miss)
        assert run_ana.check_call_in_cache(exp, ana, [], parent_uid=uid)

    def test_distinct_recipe_is_a_distinct_key(self):
        exp, uid = _exp()
        run_ana(exp, AnalysisVBA(n_perm_fwer=15), [], parent_uid=uid)
        # a different recipe is not served by the first's cache entry
        assert not run_ana.check_call_in_cache(
            exp, AnalysisVBA(n_perm_fwer=16), [], parent_uid=uid)

    def test_distinct_parent_is_a_distinct_key(self):
        # exp is ignored by the key, so parent_uid is what keeps two clean
        # experiments' fits apart -- the whole contract of ignoring exp
        exp0, uid0 = _exp()
        exp1, uid1 = _exp()
        ana = AnalysisVBA(n_perm_fwer=15)
        assert uid0 != uid1
        run_ana(exp0, ana, [], parent_uid=uid0)
        assert not run_ana.check_call_in_cache(exp1, ana, [], parent_uid=uid1)

    def test_cached_result_matches_compute(self):
        exp, uid = _exp()
        ana = AnalysisVBA(n_perm_fwer=15)
        s0 = run_ana(exp, ana, [], parent_uid=uid)   # computed
        s1 = run_ana(exp, ana, [], parent_uid=uid)   # served from cache
        assert s0 == s1


# ---------------------------------------------------------------------------
# fit_params: how the fit runs, so filtered out of the identity entirely
# ---------------------------------------------------------------------------

class TestFitParams:
    """fit_params reaches Analysis.fit and nothing else.

    It must not key a leaf: a cell fit on 32 workers or a GPU has to be the
    same artifact as one fit serially on the CPU, or every machine forks the
    benchmark's cache and records (see run.FIT_IGNORE).
    """

    def test_forwarded_to_fit(self):
        exp, uid = _exp()
        run_ana(exp, _SpyVBA(n_perm_fwer=15), [], parent_uid=uid,
                fit_params=dict(n_jobs=2, gpu='auto'))
        assert _SpyVBA.seen == dict(n_jobs=2, gpu='auto')

    def test_does_not_key_the_cache(self):
        exp, uid = _exp()
        ana = AnalysisVBA(n_perm_fwer=15)
        run_ana(exp, ana, [], parent_uid=uid, fit_params=dict(n_jobs=1))
        # a different fit_params (and none at all) hits the same entry
        assert run_ana.check_call_in_cache(exp, ana, [], parent_uid=uid)
        assert run_ana.check_call_in_cache(exp, ana, [], parent_uid=uid,
                                           fit_params=dict(n_jobs=2))

    def test_recorded_but_outside_the_identity(self):
        # the record keeps it as provenance (how this leaf ran) while the uid
        # -- what says two leaves are the same artifact -- stays blind to it,
        # so a cell run on the GPU here files under the uid a CPU run elsewhere
        # would claim
        from glow._extra.benchmark.recipe import recipe_for_call

        exp, uid = _exp()
        ana = AnalysisVBA(n_perm_fwer=15)
        data.RECORDER.records.clear()
        run_ana(exp, ana, [], parent_uid=uid,
                fit_params=dict(n_jobs=1, gpu=False))

        rec, = [r for r in data.RECORDER.records.values()
                if r['function'] == 'run_ana']
        assert rec['inputs']['fit_params'] == {'n_jobs': 1, 'gpu': False}
        assert rec['uid'] == recipe_for_call(run_ana, dict(ana=ana),
                                             parents=(uid,)).uid

    def test_leaf_uid_is_unchanged_by_it(self):
        # the uid names what a cell will produce; an execution knob must not
        # rename it, or skip_recorded stops recognising finished cells
        from glow._extra.benchmark.recipe import recipe_for_call

        ana = AnalysisVBA(n_perm_fwer=15)
        bare = recipe_for_call(run_ana, dict(ana=ana), parents=('u',))
        with_fp = recipe_for_call(
            run_ana, dict(ana=ana, fit_params=dict(n_jobs=32, gpu='auto')),
            parents=('u',))
        assert bare.uid == with_fp.uid

    def test_none_means_fit_defaults(self):
        exp, uid = _exp()
        score = run_ana(exp, AnalysisVBA(n_perm_fwer=15), [], parent_uid=uid,
                        fit_params=None)
        assert _SCORE_KEYS <= set(score)


# ---------------------------------------------------------------------------
# provenance DAG: run_ana joins data.py's recorder graph via its exp input
# ---------------------------------------------------------------------------

class TestProvenanceDAG:
    def test_run_ana_leaf_carries_its_build_ancestor(self):
        # a fresh seed so the build is a cache miss and therefore records (a hit
        # would not, leaving run_ana an ancestor-less leaf)
        seed = _fresh_seed()
        data.RECORDER.records.clear()
        exp, uid = _exp(seed=seed)
        run_ana(exp, AnalysisVBA(n_perm_fwer=15), [], parent_uid=uid)

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
        exp, uid = _exp(seed=seed)
        run_ana(exp, ana, [], parent_uid=uid)
        row = data.RECORDER.flatten_to_df().iloc[0]
        assert row['run_ana.in.ana'] == repr(ana)


# ---------------------------------------------------------------------------
# run_segment: oracle best-Dice region of one Ward tree (segmentation quality)
# ---------------------------------------------------------------------------

class TestIdentityNeverHashesArrays:
    """Naming a leaf's cache entry touches no array (the recipe invariant)."""

    def test_cache_key_ignores_exp_and_masks(self, no_array_hashing):
        # exp and mask_target_list are ignored, so the key is built from
        # parent_uid + the recipe alone -- stand-ins prove neither is hashed
        run_ana.check_call_in_cache(object(), AnalysisVBA(n_perm_fwer=6),
                                    [np.ones((8, 8, 8), dtype=bool)],
                                    parent_uid='u0')

    def test_distinct_parents_give_distinct_keys(self):
        # ...and the key still separates two parents, which is what makes
        # ignoring exp safe
        ana = AnalysisVBA(n_perm_fwer=6)
        args_a = run_ana._get_args_id(object(), ana, [], parent_uid='u0')
        args_b = run_ana._get_args_id(object(), ana, [], parent_uid='u1')
        assert args_a != args_b
        # while the ignored arguments cannot change it
        assert args_a == run_ana._get_args_id(
            object(), ana, [np.ones(4)], parent_uid='u0')


class TestRunSegment:
    def _planted(self):
        """A clean WGN exp with one planted effect (see _planted_cell)."""
        return _planted_cell(
            dict(source='wgn', shape=(7, 7, 7), b=2, num_img=20, a=1,
                 seed=_fresh_seed()),
            dict(effect_llr=0.1, extenter_cls=ExtenterMinVar, n_vox_frac=0.1,
                 seed=0))

    def test_returns_oracle_confusion_counts(self):
        exp, mask, uid = self._planted()
        score = run_segment(exp, [mask], ClusterMode.FOCUS, parent_uid=uid)
        assert set(score) == {'tp', 'fp', 'tn', 'fn'}
        # the best region's counts vs the planted support: tp + fn is exactly
        # the support size (every target voxel is hit or missed)
        assert score['tp'] + score['fn'] == int(mask.sum())

    def test_cluster_mode_is_a_cache_axis(self):
        exp, mask, uid = self._planted()
        run_segment(exp, [mask], ClusterMode.FOCUS, parent_uid=uid)
        # a different Ward mode is its own segmentation -> distinct cache entry
        assert not run_segment.check_call_in_cache(
            exp, [mask], ClusterMode.NAIVE, parent_uid=uid)


# ---------------------------------------------------------------------------
# run_inner_edge: capture max-z vs num_inner_perm from one inner sampling
# ---------------------------------------------------------------------------

class TestRunInnerEdge:
    def _planted(self):
        return _planted_cell(
            dict(source='wgn', shape=(6, 6, 6), b=2, num_img=20, a=1,
                 seed=_fresh_seed()),
            dict(effect_llr=0.1, extenter_cls=ExtenterMinVar, n_vox_frac=0.1,
                 seed=0))

    def test_grid_and_matrix_shape(self):
        exp, mask, uid = self._planted()
        curve = json.loads(run_inner_edge(
            exp, [mask], cluster_mode=ClusterMode.FOCUS, max_inner_perm=30,
            n_perm_fwer=4, parent_uid=uid))
        grid = curve['num_inner_perm']
        # ascending, ends at max_inner_perm; one max-z row per outer perm
        assert grid == sorted(grid)
        assert grid[-1] == 30
        assert np.array(curve['max_z_null']).shape == (5, len(grid))

    def test_prefix_snapshot_equals_real_fit(self):
        # the whole point: the num_inner_perm=max snapshot reproduces a real
        # AnalysisGLOW fit at n_perm_inner=max (nested seeds -> shared draws), so
        # one sampling stands in for a fit at every num_inner_perm <= it. The
        # edge curve is built on the exact cpu_perm draw prefixes, which is
        # the same exact inner null AnalysisGLOW draws.
        exp, mask, uid = self._planted()
        curve = json.loads(run_inner_edge(
            exp, [mask], cluster_mode=ClusterMode.FOCUS, max_inner_perm=30,
            n_perm_fwer=4, parent_uid=uid))
        mz_max = np.array(curve['max_z_null'])[:, -1]
        ana = AnalysisGLOW(n_perm_fwer=4, n_perm_inner=30,
                           cluster_mode=ClusterMode.FOCUS).fit(exp)
        np.testing.assert_allclose(mz_max, ana.max_z_null, rtol=1e-6, atol=1e-9)

    def test_cluster_mode_is_a_cache_axis(self):
        exp, mask, uid = self._planted()
        run_inner_edge(exp, [mask], cluster_mode=ClusterMode.FOCUS,
                       max_inner_perm=30, n_perm_fwer=4, parent_uid=uid)
        # a different Ward projection is its own edge -> distinct cache entry
        assert not run_inner_edge.check_call_in_cache(
            exp, [mask], cluster_mode=ClusterMode.GLM_ERROR, max_inner_perm=30,
            n_perm_fwer=4, parent_uid=uid)

    def test_max_inner_perm_is_a_cache_axis(self):
        exp, mask, uid = self._planted()
        run_inner_edge(exp, [mask], cluster_mode=ClusterMode.FOCUS,
                       max_inner_perm=30, n_perm_fwer=4, parent_uid=uid)
        # a deeper sampling is a distinct capture (though its prefix agrees)
        assert not run_inner_edge.check_call_in_cache(
            exp, [mask], cluster_mode=ClusterMode.FOCUS, max_inner_perm=40,
            n_perm_fwer=4, parent_uid=uid)


# ---------------------------------------------------------------------------
# run_stat: one VBA / CET MANCOVA-stat variant reading the shared voxel-walk
# ---------------------------------------------------------------------------

class TestRunStat:
    def _planted(self, seed=None):
        # b=2 so the multivariate stats differ
        return _planted_cell(
            dict(source='wgn', shape=(6, 6, 6), b=2, num_img=24, a=1,
                 seed=_fresh_seed() if seed is None else seed),
            dict(effect_llr=0.15, extenter_cls=ExtenterMinVar, n_vox_frac=0.1,
                 seed=0))

    def test_matches_standalone_fit(self):
        # the shared walk is just a precompute of the same stat matrix, so a
        # variant reading it equals the standalone run_ana fit of that recipe
        exp, mask, uid = self._planted()
        ana = AnalysisVBA(get_stat=get_wilks, n_perm_fwer=15, z_flag=True)
        assert run_stat(exp, [mask], ana, stat_dict_inv[get_wilks],
                        parent_uid=uid) == run_ana(exp, ana, [mask],
                                                   parent_uid=uid)

    def test_variants_share_one_walk(self):
        # the first variant computes the walk, the rest read the same object
        # back from the memo (nothing re-walks the permutations)
        exp, mask, uid = self._planted()
        walk = voxel_stat_walk(exp, 15, parent_uid=uid)
        run_stat(exp, [mask], AnalysisVBA(get_stat=get_wilks, n_perm_fwer=15),
                 stat_dict_inv[get_wilks], parent_uid=uid)
        assert voxel_stat_walk(exp, 15, parent_uid=uid) is walk

    def test_walk_is_never_persisted(self):
        # the walk is ~250 MB a cell, so it stays in memory: no cache dir of
        # its own, and the memo holds one cell (the previous one is dropped)
        exp_a, _, uid_a = self._planted()
        exp_b, _, uid_b = self._planted()
        assert not hasattr(voxel_stat_walk, 'check_call_in_cache')
        walk_a = voxel_stat_walk(exp_a, 15, parent_uid=uid_a)
        voxel_stat_walk(exp_b, 15, parent_uid=uid_b)
        assert len(run._WALK_MEMO) == 1
        assert voxel_stat_walk(exp_a, 15, parent_uid=uid_a) is not walk_a

    def test_recipe_is_a_cache_axis(self):
        exp, mask, uid = self._planted()
        ana = AnalysisVBA(get_stat=get_wilks, n_perm_fwer=15)
        run_stat(exp, [mask], ana, stat_dict_inv[get_wilks], parent_uid=uid)
        # a different stat is a different variant -> distinct cache entry
        other = AnalysisVBA(get_stat=get_hotel_tr, n_perm_fwer=15)
        assert not run_stat.check_call_in_cache(
            exp, [mask], other, stat_dict_inv[get_hotel_tr], parent_uid=uid)


# ---------------------------------------------------------------------------
# run_prune: one pruning rule scored on a shared GLOW fit
# ---------------------------------------------------------------------------

class TestRunPrune:
    _GLOW = dict(n_perm_fwer=4, n_perm_inner=8, alpha_fwer=0.05)

    def _planted(self):
        return _planted_cell(
            dict(source='wgn', shape=(6, 6, 6), b=2, num_img=24, a=1,
                 seed=_fresh_seed()),
            dict(effect_llr=0.2, extenter_cls=ExtenterMinVar, n_vox_frac=0.1,
                 seed=0))

    def test_returns_prune_score(self):
        exp, mask, uid = self._planted()
        score = run_prune(exp, [mask], 'maxllr', parent_uid=uid, **self._GLOW)
        # per-effect confusion counts plus the output-region count
        assert {'n_selected', 'tp', 'fp', 'tn', 'fn'} <= set(score)

    def test_rules_share_one_fit(self):
        # the first rule fits GLOW; the other rules are glow_fit_for_prune hits
        exp, mask, uid = self._planted()
        assert not glow_fit_for_prune.check_call_in_cache(
            exp, parent_uid=uid, cluster_mode=ClusterMode.FOCUS, **self._GLOW)
        run_prune(exp, [mask], 'greedy', parent_uid=uid, **self._GLOW)
        assert glow_fit_for_prune.check_call_in_cache(
            exp, parent_uid=uid, cluster_mode=ClusterMode.FOCUS, **self._GLOW)

    def test_rule_is_a_cache_axis(self):
        exp, mask, uid = self._planted()
        run_prune(exp, [mask], 'greedy', parent_uid=uid, **self._GLOW)
        # a different rule is its own selection -> distinct cache entry
        assert not run_prune.check_call_in_cache(
            exp, [mask], 'dp', parent_uid=uid, **self._GLOW)

    def test_bad_rule_raises(self):
        exp, mask, uid = self._planted()
        with pytest.raises(ValueError):
            run_prune(exp, [mask], 'nope', parent_uid=uid, **self._GLOW)


# ---------------------------------------------------------------------------
# runtime leaves: time one piece of GLOW, return the analyzed voxel count
# ---------------------------------------------------------------------------

class TestRuntimeLeaves:
    """run_ana_time_1perm: the permutation counts it overrides, and its copy.

    The wall time itself is not asserted on -- a timing is not reproducible.
    What is testable is the contract around it: the counts reach the recipe,
    they key the cache (so two points on a sweep are two measurements), and
    the caller's recipe survives untouched.
    """

    def _planted(self):
        return _planted_cell(
            dict(source='wgn', shape=(6, 6, 6), b=2, num_img=20, a=1,
                 seed=_fresh_seed()),
            dict(effect_llr=0.1, extenter_cls=ExtenterMinVar, n_vox_frac=0.1,
                 seed=0))

    def _glow(self):
        return AnalysisGLOW(n_perm_fwer=500, n_perm_inner=8)

    def test_returns_num_vox(self):
        # the recorded measurement is time_sec; the return is the analyzed
        # voxel count, the sweep's size context
        exp, mask, uid = self._planted()
        num_vox = run_ana_time_1perm(exp, [mask], self._glow(),
                                     parent_uid=uid)
        assert num_vox == int((exp.mask_idx > -1).sum())

    def test_n_perm_fwer_is_a_cache_axis(self):
        exp, mask, uid = self._planted()
        ana = self._glow()
        run_ana_time_1perm(exp, [mask], ana, parent_uid=uid)
        # a different count is its own timing -> its own cache entry
        assert not run_ana_time_1perm.check_call_in_cache(
            exp, [mask], ana, n_perm_fwer=2, parent_uid=uid)

    def test_n_perm_inner_is_a_cache_axis(self):
        exp, mask, uid = self._planted()
        ana = self._glow()
        run_ana_time_1perm(exp, [mask], ana, n_perm_inner=4, parent_uid=uid)
        assert not run_ana_time_1perm.check_call_in_cache(
            exp, [mask], ana, n_perm_inner=8, parent_uid=uid)

    def test_the_callers_recipe_is_untouched(self):
        # the counts are overridden on a private copy: the caller's recipe is
        # what identifies the method everywhere else, cache key included
        exp, mask, uid = self._planted()
        ana = self._glow()
        run_ana_time_1perm(exp, [mask], ana, n_perm_inner=4, parent_uid=uid)
        assert ana.n_perm_fwer == 500
        assert ana.n_perm_inner == 8
        assert ana.effect_list is None

    def test_one_perm_is_the_default(self):
        exp, mask, uid = self._planted()
        ana = self._glow()
        # explicit 1 and the default are one measurement, not two
        run_ana_time_1perm(exp, [mask], ana, parent_uid=uid)
        assert run_ana_time_1perm.check_call_in_cache(
            exp, [mask], ana, n_perm_fwer=1, parent_uid=uid)

    def test_num_img_cuts_the_cohort(self):
        exp, mask, uid = self._planted()
        run_ana_time_1perm(exp, [mask], self._glow(), num_img=8,
                           parent_uid=uid)
        # the cut is the leaf's own, so the caller's exp keeps its subjects
        assert exp.y.shape[1] == 20

    def test_num_img_is_a_cache_axis(self):
        exp, mask, uid = self._planted()
        ana = self._glow()
        run_ana_time_1perm(exp, [mask], ana, num_img=8, parent_uid=uid)
        assert not run_ana_time_1perm.check_call_in_cache(
            exp, [mask], ana, num_img=12, parent_uid=uid)

    def test_num_img_past_the_cohort_raises(self):
        # a silently short curve is worse than a failure: the sweep would
        # plot a flat tail wherever it ran past the sample
        exp, mask, uid = self._planted()
        with pytest.raises(ValueError, match='cohort'):
            run_ana_time_1perm(exp, [mask], self._glow(), num_img=21,
                               parent_uid=uid)

    def test_the_cut_keeps_the_analysed_layout(self):
        # a strided view would make the timing measure the stride, not the
        # size -- the analysis path is written for F-contiguous y
        exp, _, _ = self._planted()
        cut = run._take_img(exp, 8)
        assert cut.y.shape == (exp.y.shape[0], 8, exp.y.shape[2])
        assert cut.x.shape == (exp.x.shape[0], 8)
        assert cut.y.flags['F_CONTIGUOUS'] and cut.y.flags['OWNDATA']

    def test_inner_perms_rejected_for_a_voxelwise_recipe(self):
        # no voxel-wise method has an inner null, so the sweep would otherwise
        # record a flat curve against a knob nothing reads
        exp, mask, uid = self._planted()
        with pytest.raises(ValueError, match='n_perm_inner'):
            run_ana_time_1perm(exp, [mask], AnalysisVBA(n_perm_fwer=4),
                               n_perm_inner=8, parent_uid=uid)
