"""Tests for glow._extra.benchmark.config: the paper-benchmark catalogue.

config declares each cache as the four-tuple driver.drive consumes
(kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc); the two upstream
grids are lists built by get_kwargs_data_list / get_kwargs_effect_list. Building
the cells makes Extenters / HCP feature subsets but runs no experiments and
downloads no data, so these check the catalogue is well-formed and that every
cell *binds* to its stage signature -- a typo'd kwarg is caught here rather than
deep in a multi-day sweep -- without executing any stage. The stages
(data_factory / effect_factory / run_ana) and the sweep are covered by test_data
/ test_run / test_driver.
"""
import inspect
import math
import pickle

import pytest

from glow._extra.benchmark import config, data, hcp
from glow._extra.benchmark.run import run_ana
from glow.analysis import Analysis
from glow.effect import ExtenterMinVar


# every paper-figure cache, and the subset whose leaf is run_ana (the rest
# carry their own fnc + kwargs grid -- segment its Ward-mode oracle, etc.)
LABELS = ['null', 'sweep_llr', 'sweep_extent', 'sweep_nimg', 'segment',
          'min_size', 'stat', 'prune', 'two-effect']
RUN_ANA_LABELS = ['null', 'sweep_llr', 'sweep_extent', 'sweep_nimg',
                  'two-effect']
# non-paper helper caches in the catalogue (the tiny end-to-end smoke cache, the
# runtime family, and the n_perm_inner convergence sweep); excluded from the
# paper-cardinality checks below
NON_PAPER_LABELS = ['smoke', 'runtime', 'runtime_segment',
                    'runtime_n_perm_fwer', 'runtime_n_perm_inner', 'runtime_b',
                    'sweep_n_perm_inner']
# every catalogue entry (paper + helper); the shape / bind checks cover all
ALL_LABELS = LABELS + NON_PAPER_LABELS


class TestCatalogueShape:
    def test_expected_labels(self):
        assert set(config.CONFIG) == set(LABELS) | set(NON_PAPER_LABELS)

    @pytest.mark.parametrize('label', ALL_LABELS)
    def test_entry_is_drive_four_tuple(self, label):
        # every cache is the (data, effect, fnc-kwargs, fnc) tuple drive
        # consumes, with non-empty grids and a callable leaf
        data_list, effect_list, fnc_kwargs, fnc = config.CONFIG[label]
        assert data_list and effect_list and fnc_kwargs
        assert callable(fnc)

    @pytest.mark.parametrize('label', RUN_ANA_LABELS)
    def test_run_ana_caches_share_the_recipe_grid(self, label):
        # the detection sweeps all fit/score via run_ana over the one shared
        # recipe grid
        _, _, fnc_kwargs, fnc = config.CONFIG[label]
        assert fnc is run_ana
        assert fnc_kwargs is config.RUN_ANA_LIST

    @pytest.mark.parametrize('label', ALL_LABELS)
    def test_grids_are_materialized_lists(self, label):
        # the data / effect grids are concrete lists (re-iterable; the driver
        # re-walks the effect grid per data cell, so a one-shot iterator breaks)
        data_grid, effect_grid, _, _ = config.CONFIG[label]
        assert isinstance(data_grid, list)
        assert isinstance(effect_grid, list)

    def test_five_analysis_recipes(self):
        assert len(config.RUN_ANA_LIST) == 5
        assert all(isinstance(c['ana'], Analysis) for c in config.RUN_ANA_LIST)

    def test_run_ana_cells_carry_only_the_recipe(self):
        # each leaf cell carries just its ana; the method name is not passed
        # (recovered from ana_kwargs_dict at read time -- see run / plot)
        assert all(set(c) == {'ana'} for c in config.RUN_ANA_LIST)
        assert ([c['ana'] for c in config.RUN_ANA_LIST]
                == list(config.ana_kwargs_dict.values()))


class TestFilterAnaList:
    def test_keeps_named_recipes_in_grid_order(self):
        kept = config.filter_ana_list(config.RUN_ANA_LIST, ['CET', 'VBA'])
        assert [c['ana'] for c in kept] == [config.ana_kwargs_dict['VBA'],
                                            config.ana_kwargs_dict['CET']]

    def test_matches_a_rebuilt_equal_recipe(self):
        # matched on the recipe repr, not identity, so an equal recipe rebuilt
        # elsewhere still selects -- the round trip the AWS run bundle does to
        # a recipe (a pickle this test wrote itself, see glow._extra.aws.bundle)
        grid = [dict(ana=pickle.loads(pickle.dumps(ana)))
                for ana in config.ana_kwargs_dict.values()]
        kept = config.filter_ana_list(grid, ['VBA-TFCE'])
        assert [repr(c['ana']) for c in kept] == [
            repr(config.ana_kwargs_dict['VBA-TFCE'])]

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError):
            config.filter_ana_list(config.RUN_ANA_LIST, ['VBA', 'TFCE'])

    def test_grid_without_a_recipe_axis_is_empty(self):
        # a non-run_ana leaf grid (segment / prune / ...) has no ana to select
        _, _, fnc_kwargs, _ = config.CONFIG['segment']
        assert config.filter_ana_list(fnc_kwargs, ['VBA']) == []


class TestIterKwargsData:
    def test_returns_a_list(self):
        assert isinstance(config.get_kwargs_data_list(), list)

    def test_defaults_both_sources_n_seed(self):
        cells = config.get_kwargs_data_list()
        assert len(cells) == 2 * config.N_SEED
        assert {c['source'] for c in cells} == {'wgn', 'hcp'}

    def test_wgn_box_sized_from_crop(self):
        side = math.ceil(config.CROP_N_VOX ** (1 / 3))
        cell = config.get_kwargs_data_list(sources=['wgn'], seeds=[0])[0]
        assert cell['shape'] == (side, side, side)

    def test_crop_built_once_per_seed(self):
        # one extenter per (source, b, seed), shared across a WGN cell's num_img
        cells = config.get_kwargs_data_list(sources=['wgn'], seeds=[0],
                                            num_img_list=[10, 20, 30])
        assert len(cells) == 3
        assert len({id(c['extenter']) for c in cells}) == 1

    def test_hcp_feats_distinct_sorted_in_pool(self):
        cell, = config.get_kwargs_data_list(sources=['hcp'], seeds=[0], b_list=[3])
        feats = cell['hcp_feats']
        assert len(feats) == len(set(feats)) == 3
        assert list(feats) == sorted(feats)
        assert set(feats) <= set(hcp.HCP_FEATS)

    def test_hcp_feats_seed_deterministic(self):
        (a,) = config.get_kwargs_data_list(sources=['hcp'], seeds=[7], b_list=[2])
        (b,) = config.get_kwargs_data_list(sources=['hcp'], seeds=[7], b_list=[2])
        assert a['hcp_feats'] == b['hcp_feats']


class TestIterKwargsEffect:
    def test_returns_a_list(self):
        assert isinstance(config.get_kwargs_effect_list(), list)

    def test_default_is_moderate_llr_at_fixed_extent(self):
        cells = config.get_kwargs_effect_list()
        assert len(cells) == 1
        assert cells[0]['effect_llr'] == config.MODERATE_EFFECT_LLR
        assert cells[0]['n_vox_frac'] == config.EFFECT_N_VOX_FRAC
        # cells carry the ingredients effect_factory builds the support from
        assert cells[0]['extenter_cls'] is ExtenterMinVar
        assert cells[0]['seed_from_exp'] is True

    def test_null_returns_single_none(self):
        assert config.get_kwargs_effect_list(llr_list=None) == [None]

    def test_extent_sweeps_support_at_fixed_per_voxel_llr(self):
        # no whole-region-LLR knob: effect_llr is held fixed, the support varies
        cells = config.get_kwargs_effect_list(n_vox_frac_list=[0.05, 0.2])
        assert [c['effect_llr'] for c in cells] == [config.MODERATE_EFFECT_LLR] * 2
        assert [c['n_vox_frac'] for c in cells] == [0.05, 0.2]

    def test_llr_and_extent_are_a_product(self):
        cells = config.get_kwargs_effect_list(llr_list=[0.01, 0.1],
                                              n_vox_frac_list=[0.05, 0.1, 0.2])
        assert len(cells) == 2 * 3


class TestCellsBindToStages:
    """Every cell is valid kwargs for the stage the driver feeds it to."""

    _SIG_DATA = {'wgn': inspect.signature(data.data_factory_wgn),
                 'hcp': inspect.signature(data.data_factory_hcp)}

    @pytest.mark.parametrize('label', ALL_LABELS)
    def test_data_cells_bind(self, label):
        # source selects the builder; the rest are its kwargs
        for cell in config.CONFIG[label][0]:
            sig = self._SIG_DATA[cell['source']]
            sig.bind(**{k: v for k, v in cell.items() if k != 'source'})

    _SIG_EFFECT = {'single': inspect.signature(data.effect_factory_single),
                   'split': inspect.signature(data.effect_factory_split)}

    @pytest.mark.parametrize('label', ALL_LABELS)
    def test_effect_cells_bind(self, label):
        # kind selects the builder (effect_factory dispatches on it); exp is
        # supplied by the driver; a None cell is the no-plant null path
        for cell in config.CONFIG[label][1]:
            if cell is None:
                continue
            sig = self._SIG_EFFECT[cell['kind']]
            sig.bind(exp=None,
                     **{k: v for k, v in cell.items() if k != 'kind'})

    @pytest.mark.parametrize('label', ALL_LABELS)
    def test_fnc_cells_bind(self, label):
        # exp / mask_target_list are supplied by the driver
        _, _, fnc_kwargs, fnc = config.CONFIG[label]
        sig = inspect.signature(fnc)
        for cell in fnc_kwargs:
            sig.bind(exp=None, mask_target_list=[], **cell)


class TestGridCardinality:
    def test_products_match_old_catalogue(self):
        # cells x methods, identical to the published paper's trial counts
        # (paper caches only; non-paper helpers like 'smoke' are excluded)
        n = {label: len(d) * len(e) * len(fk)
             for label, (d, e, fk, _) in config.CONFIG.items()
             if label in LABELS}
        assert n == {
            'null': 2 * config.N_SEED_NULL * 5,
            'sweep_llr': 2 * len(config.B_LLR_SWEEP) * config.N_SEED
            * len(config.EFFECT_LLR_GRID) * 5,
            'sweep_extent': 2 * config.N_SEED * len(config.EXTENT_FRAC_GRID) * 5,
            'sweep_nimg': config.N_SEED * len(config.NIMG_GRID) * 5,
            'segment': 2 * config.N_SEED * len(config.EFFECT_LLR_GRID)
            * len(config.SEGMENT_MODES),
            'min_size': config.N_SEED * len(config.EFFECT_LLR_GRID),
            'stat': 2 * config.N_SEED * len(config.EFFECT_LLR_GRID)
            * len(config.RUN_STAT_LIST),
            'prune': 2 * config.N_SEED * len(config.EFFECT_LLR_GRID)
            * len(config.RUN_PRUNE_LIST),
            'two-effect': 2 * config.N_SEED * len(config.TWO_EFFECT_LLR_GRID)
            * len(config.ANGLE_GRID) * 5,
        }

    def test_null_plants_nothing(self):
        assert config.CONFIG['null'][1] == [None]

    def test_hcp_cells_carry_no_num_img_axis(self):
        # HCP's N is its cohort, so an HCP cell never carries num_img (it would
        # be a dead axis silently duplicating cells)
        for label in LABELS:
            for cell in config.CONFIG[label][0]:
                if cell['source'] == 'hcp':
                    assert 'num_img' not in cell
