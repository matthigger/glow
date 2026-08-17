"""Tests for glow._extra.benchmark.grid: the catalogue's grid builders.

Every axis is passed in explicitly, so nothing here depends on the paper's
current values -- retuning a config constant cannot turn these red, and a
builder's contract is checked on grids the catalogue never declares.
Building cells makes Extenters and draws HCP feature subsets, but runs no
experiment and reads no imaging data.

That config declares well-formed cells (and that each binds to its stage) is
test_config's job; the stages themselves are test_data / test_run.
"""
import pickle

import numpy as np
import pytest

from glow._extra.benchmark import grid, hcp
from glow.analysis import AnalysisCET, AnalysisGLOW, AnalysisVBA
from glow.analysis.mancova import stat_dict
from glow.effect import ExtenterMinVar, ExtenterSphere


def _recipes():
    """Return a small label -> recipe catalogue, standing in for config's."""
    return {'A-glow': AnalysisGLOW(n_perm_fwer=4),
            'B-vba': AnalysisVBA(n_perm_fwer=4),
            'C-cet': AnalysisCET(n_perm_fwer=4)}


def _ana_grid(ana_kwargs_dict, glow_fit_params=None):
    """Return the leaf grid config builds from a recipe catalogue."""
    return [dict(ana=ana,
                 fit_params=grid.fit_params_for(ana, glow_fit_params))
            for ana in ana_kwargs_dict.values()]


# ---------------------------------------------------------------------------
# data grid
# ---------------------------------------------------------------------------

class TestDataList:
    def test_product_over_source_b_seed(self):
        cells = grid.get_kwargs_data_list(sources=['wgn', 'hcp'],
                                          seeds=range(4), b_list=[1, 2],
                                          crop_n_vox=64)
        assert len(cells) == 2 * 4 * 2
        assert {c['source'] for c in cells} == {'wgn', 'hcp'}

    def test_wgn_sweeps_num_img_hcp_does_not(self):
        # HCP's N is its cohort, so a num_img axis would duplicate its cells
        # rather than vary them
        cells = grid.get_kwargs_data_list(sources=['wgn', 'hcp'], seeds=[0],
                                          num_img_list=[10, 20, 30],
                                          crop_n_vox=64)
        wgn = [c for c in cells if c['source'] == 'wgn']
        hcp_cells = [c for c in cells if c['source'] == 'hcp']
        assert [c['num_img'] for c in wgn] == [10, 20, 30]
        assert len(hcp_cells) == 1
        assert all('num_img' not in c for c in hcp_cells)

    def test_wgn_box_sized_from_crop(self):
        # the box is the smallest cube holding crop_n_vox voxels
        cell, = grid.get_kwargs_data_list(sources=['wgn'], seeds=[0],
                                          crop_n_vox=1000)
        assert cell['shape'] == (10, 10, 10)

    def test_crop_built_once_per_seed(self):
        # one extenter per (source, b, seed), shared across a cell's num_img
        cells = grid.get_kwargs_data_list(sources=['wgn'], seeds=[0],
                                          num_img_list=[10, 20, 30],
                                          crop_n_vox=64)
        assert len({id(c['extenter']) for c in cells}) == 1

    def test_crop_is_a_connected_sphere_of_crop_n_vox(self):
        cell, = grid.get_kwargs_data_list(sources=['hcp'], seeds=[0],
                                          crop_n_vox=512)
        extenter = cell['extenter']
        assert isinstance(extenter, ExtenterSphere)
        assert extenter.n_vox == 512
        assert extenter.connected

    def test_hcp_feats_distinct_sorted_in_pool(self):
        cell, = grid.get_kwargs_data_list(sources=['hcp'], seeds=[0],
                                          b_list=[3], crop_n_vox=64)
        feats = cell['hcp_feats']
        assert len(feats) == len(set(feats)) == 3
        assert list(feats) == sorted(feats)
        assert set(feats) <= set(hcp.HCP_FEATS)

    def test_hcp_feats_seed_deterministic(self):
        kwargs = dict(sources=['hcp'], seeds=[7], b_list=[2], crop_n_vox=64)
        (a,) = grid.get_kwargs_data_list(**kwargs)
        (b,) = grid.get_kwargs_data_list(**kwargs)
        assert a['hcp_feats'] == b['hcp_feats']

    def test_distinct_seeds_give_distinct_crops(self):
        cells = grid.get_kwargs_data_list(sources=['wgn'], seeds=[0, 1],
                                          crop_n_vox=64)
        assert len({c['extenter'].seed for c in cells}) == 2

    def test_empty_axis_gives_an_empty_grid(self):
        assert grid.get_kwargs_data_list(sources=[], seeds=[0],
                                         crop_n_vox=64) == []


class TestRuntimeDataList:
    def test_spans_the_crop_sweep_at_n_seed_each(self):
        cells = grid.get_kwargs_data_runtime(
            seed_offset=1000, crop_n_vox_list=[64, 512], n_seed=3)
        assert len(cells) == 2 * 3
        assert {c['extenter'].n_vox for c in cells} == {64, 512}

    def test_seeds_start_at_the_offset(self):
        cells = grid.get_kwargs_data_runtime(
            seed_offset=1000, crop_n_vox_list=[64], n_seed=3)
        assert sorted(c['seed'] for c in cells) == [1000, 1001, 1002]

    def test_offsets_keep_two_caches_disjoint(self):
        # each runtime cache takes its own seed block so no two share a data
        # cell (hence a cached leaf timing)
        one = grid.get_kwargs_data_runtime(
            seed_offset=0, crop_n_vox_list=[64], n_seed=3)
        two = grid.get_kwargs_data_runtime(
            seed_offset=100, crop_n_vox_list=[64], n_seed=3)
        assert not ({c['seed'] for c in one} & {c['seed'] for c in two})

    def test_defaults_to_hcp(self):
        cells = grid.get_kwargs_data_runtime(
            seed_offset=0, crop_n_vox_list=[64], n_seed=1)
        assert {c['source'] for c in cells} == {'hcp'}


# ---------------------------------------------------------------------------
# effect grid
# ---------------------------------------------------------------------------

class TestEffectList:
    def test_llr_and_extent_are_a_product(self):
        cells = grid.get_kwargs_effect_list(llr_list=[0.01, 0.1],
                                            n_vox_frac_list=[0.05, 0.1, 0.2])
        assert len(cells) == 2 * 3
        assert [c['effect_llr'] for c in cells] == [0.01] * 3 + [0.1] * 3
        assert [c['n_vox_frac'] for c in cells] == [0.05, 0.1, 0.2] * 2

    def test_cells_carry_the_support_ingredients(self):
        # effect_factory_single builds the support from these, not from a
        # pre-made mask; seed_from_exp places it per realization
        cell, = grid.get_kwargs_effect_list(llr_list=[0.03],
                                            n_vox_frac_list=[0.1])
        assert cell['kind'] == 'single'
        assert cell['extenter_cls'] is ExtenterMinVar
        assert cell['seed_from_exp'] is True

    def test_null_returns_single_none(self):
        assert grid.get_kwargs_effect_list(
            llr_list=None, n_vox_frac_list=[0.1]) == [None]

    def test_values_are_plain_floats(self):
        # a numpy scalar would hash differently from effect_factory's
        # float(effect_llr) cast, forking the cache entry
        cell, = grid.get_kwargs_effect_list(
            llr_list=np.array([0.03]), n_vox_frac_list=np.array([0.1]))
        assert type(cell['effect_llr']) is float
        assert type(cell['n_vox_frac']) is float


class TestTwoEffectList:
    def test_llr_and_angle_are_a_product(self):
        cells = grid.get_kwargs_two_effect_list(
            llr_list=[0.03, 0.1], angle_list=[0, 45, 90], n_vox_frac=0.1)
        assert len(cells) == 2 * 3
        assert all(c['kind'] == 'split' for c in cells)
        assert [c['angle'] for c in cells] == [0.0, 45.0, 90.0] * 2

    def test_split_base_is_overridable(self):
        cells = grid.get_kwargs_two_effect_list(
            llr_list=[0.03], angle_list=[0], n_vox_frac=0.1,
            extenter_cls=ExtenterSphere)
        assert cells[0]['extenter_cls'] is ExtenterSphere

    def test_values_are_plain_floats(self):
        cell, = grid.get_kwargs_two_effect_list(
            llr_list=[0.03], angle_list=[45], n_vox_frac=0.1)
        assert type(cell['effect_llr']) is float
        assert type(cell['angle']) is float
        assert type(cell['n_vox_frac']) is float


# ---------------------------------------------------------------------------
# leaf-fnc (recipe) grids
# ---------------------------------------------------------------------------

class TestFitParamsFor:
    def test_glow_gets_the_params_others_none(self):
        params = dict(n_jobs=3, gpu='auto')
        recipes = _recipes()
        assert grid.fit_params_for(recipes['A-glow'], params) is params
        assert grid.fit_params_for(recipes['B-vba'], params) is None
        assert grid.fit_params_for(recipes['C-cet'], params) is None


class TestStripGpu:
    def test_drops_only_the_device(self):
        gridlist = [dict(ana=None, fit_params=dict(n_jobs=8, gpu='auto'))]
        assert grid.strip_gpu(gridlist) == [
            dict(ana=None, fit_params=dict(n_jobs=8))]

    def test_fit_params_holding_only_gpu_becomes_none(self):
        gridlist = [dict(ana=None, fit_params=dict(gpu=True))]
        assert grid.strip_gpu(gridlist) == [dict(ana=None, fit_params=None)]

    def test_passes_plain_cells_through(self):
        gridlist = [dict(ana=None), dict(ana=None, fit_params=dict(n_jobs=2))]
        assert grid.strip_gpu(gridlist) == gridlist

    def test_does_not_mutate_the_input(self):
        # the grids are module-level singletons shared by every cache
        params = dict(n_jobs=8, gpu='auto')
        grid.strip_gpu([dict(ana=None, fit_params=params)])
        assert params == dict(n_jobs=8, gpu='auto')


class TestFilterAnaList:
    def test_keeps_named_recipes_in_grid_order(self):
        recipes = _recipes()
        kept = grid.filter_ana_list(_ana_grid(recipes), ['C-cet', 'B-vba'],
                                    recipes)
        assert [c['ana'] for c in kept] == [recipes['B-vba'],
                                            recipes['C-cet']]

    def test_matches_a_rebuilt_equal_recipe(self):
        # matched on the recipe repr, not identity, so an equal recipe
        # rebuilt elsewhere still selects -- the round trip a pickled recipe
        # makes (a pickle this test wrote itself)
        recipes = _recipes()
        rebuilt = [dict(ana=pickle.loads(pickle.dumps(ana)))
                   for ana in recipes.values()]
        kept = grid.filter_ana_list(rebuilt, ['B-vba'], recipes)
        assert [repr(c['ana']) for c in kept] == [repr(recipes['B-vba'])]

    def test_keeps_the_cell_whole(self):
        # a kept cell carries its fit_params through unchanged
        recipes = _recipes()
        params = dict(n_jobs=3)
        kept = grid.filter_ana_list(_ana_grid(recipes, params), ['A-glow'],
                                    recipes)
        assert kept == [dict(ana=recipes['A-glow'], fit_params=params)]

    def test_unknown_label_raises(self):
        recipes = _recipes()
        with pytest.raises(ValueError, match='unknown method label'):
            grid.filter_ana_list(_ana_grid(recipes), ['B-vba', 'nope'],
                                 recipes)

    def test_grid_without_a_recipe_axis_is_empty(self):
        # a non-run_ana leaf grid (segment / prune / ...) has no ana to select
        assert grid.filter_ana_list([dict(cluster_mode='Focus')], ['B-vba'],
                                    _recipes()) == []

    def test_no_labels_keeps_nothing(self):
        recipes = _recipes()
        assert grid.filter_ana_list(_ana_grid(recipes), [], recipes) == []


class TestRunStatList:
    def test_three_methods_over_stats_and_z(self):
        cells = grid.get_run_stat_list(n_perm_fwer=4, alpha_fwer=0.05,
                                       cft_pval=0.001)
        # VBA / VBA-TFCE / CET x every stat x {raw, z}
        assert len(cells) == 3 * len(stat_dict) * 2
        assert {c['stat_name'] for c in cells} == set(stat_dict)

    def test_every_variant_is_distinct(self):
        cells = grid.get_run_stat_list(n_perm_fwer=4, alpha_fwer=0.05,
                                       cft_pval=0.001)
        keys = {(repr(c['ana']), c['stat_name']) for c in cells}
        assert len(keys) == len(cells)

    def test_glow_is_excluded(self):
        # the bake-off is over the voxel-wise methods; GLOW uses the LLR
        cells = grid.get_run_stat_list(n_perm_fwer=4, alpha_fwer=0.05,
                                       cft_pval=0.001)
        assert not any(isinstance(c['ana'], AnalysisGLOW) for c in cells)

    def test_knobs_reach_every_recipe(self):
        cells = grid.get_run_stat_list(n_perm_fwer=7, alpha_fwer=0.02,
                                       cft_pval=0.005)
        assert all(c['ana'].n_perm_fwer == 7 for c in cells)
        assert all(c['ana'].alpha_fwer == 0.02 for c in cells)
        assert all(c['ana'].cft_pval == 0.005 for c in cells
                   if isinstance(c['ana'], AnalysisCET))
