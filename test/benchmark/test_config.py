"""Tests for glow._extra.benchmark.config: the paper-benchmark catalogue.

config declares each cache as the four-tuple driver.drive consumes
(kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc). These check the
catalogue is well-formed and that every cell binds to its stage signature -- a
typo'd kwarg is caught here rather than deep in a multi-day sweep -- without
executing any stage. Building the cells makes Extenters / HCP feature subsets
but runs no experiments and downloads no data.

What is deliberately NOT checked is which values the catalogue holds: grid
sizes, seed counts, permutation counts, how many recipes there are, which
sources a cache draws from. Those are config's to tune, and a test pinning them
would have to be edited on every tuning -- which trains one to edit it unread.
The invariants live with the builders instead (test_grid, driven by explicit
axes), so they hold for whatever the catalogue declares. Every case here is
parametrized off config.CONFIG itself, so a cache added tomorrow is covered
without touching this file.

The stages (data_factory / effect_factory / run_ana) and the sweep are covered
by test_data / test_run / test_driver.
"""
import inspect

import pytest

from glow._extra.benchmark import config, data
from glow.analysis import Analysis, AnalysisGLOW


# every catalogue entry; the shape / bind checks cover all of them
LABELS = sorted(config.CONFIG)

# the caches whose leaf is run_ana, read off the catalogue rather than named
RUN_ANA_LABELS = [label for label in LABELS
                  if config.CONFIG[label][3].__name__ == 'run_ana']


class TestCatalogueShape:
    @pytest.mark.parametrize('label', LABELS)
    def test_entry_is_drive_four_tuple(self, label):
        # every cache is the (data, effect, fnc-kwargs, fnc) tuple drive
        # consumes, with non-empty grids and a callable leaf
        data_list, effect_list, fnc_kwargs, fnc = config.CONFIG[label]
        assert data_list and effect_list and fnc_kwargs
        assert callable(fnc)

    @pytest.mark.parametrize('label', LABELS)
    def test_grids_are_materialized_lists(self, label):
        # the data / effect grids are concrete lists (re-iterable; the driver
        # re-walks the effect grid per data cell, so a one-shot iterator
        # breaks)
        data_grid, effect_grid, _, _ = config.CONFIG[label]
        assert isinstance(data_grid, list)
        assert isinstance(effect_grid, list)

    @pytest.mark.parametrize('label', RUN_ANA_LABELS)
    def test_run_ana_caches_share_the_recipe_grid(self, label):
        # the detection sweeps all fit/score over the one shared recipe grid,
        # by identity: strip_gpu / filter_ana_list rebuild cells rather than
        # mutating them precisely because it is a singleton
        assert config.CONFIG[label][2] is config.RUN_ANA_LIST

    def test_recipes_are_analyses(self):
        assert config.RUN_ANA_LIST
        assert all(isinstance(c['ana'], Analysis)
                   for c in config.RUN_ANA_LIST)

    def test_run_ana_cells_carry_the_recipe_and_how_to_run_it(self):
        # each leaf cell carries its ana and its fit_params, nothing else: the
        # method name is not passed (recovered from ana_kwargs_dict at read
        # time -- see run / plot), and fit_params is filtered back out of the
        # identity by the leaf (run.FIT_IGNORE)
        assert all(set(c) == {'ana', 'fit_params'}
                   for c in config.RUN_ANA_LIST)
        assert ([c['ana'] for c in config.RUN_ANA_LIST]
                == list(config.ana_kwargs_dict.values()))


class TestPaperAxes:
    """The wrappers that hand grid the paper's values (config's only logic)."""

    def test_data_grid_applies_the_paper_axes(self):
        cells = config.data_grid()
        assert {c['source'] for c in cells} == set(config.DATA_AXES['sources'])
        assert len({c['seed'] for c in cells}) == len(
            list(config.DATA_AXES['seeds']))

    def test_data_grid_overrides_one_axis_and_keeps_the_rest(self):
        cells = config.data_grid(seeds=[0])
        assert {c['seed'] for c in cells} == {0}
        assert {c['source'] for c in cells} == set(config.DATA_AXES['sources'])

    def test_effect_grid_applies_the_paper_axes(self):
        cell, = config.effect_grid()
        assert cell['effect_llr'] == config.MODERATE_EFFECT_LLR
        assert cell['n_vox_frac'] == config.EFFECT_N_VOX_FRAC

    def test_effect_grid_overrides_one_axis_and_keeps_the_rest(self):
        cells = config.effect_grid(n_vox_frac_list=[0.05, 0.2])
        assert [c['n_vox_frac'] for c in cells] == [0.05, 0.2]
        assert {c['effect_llr'] for c in cells} == {
            config.MODERATE_EFFECT_LLR}

    def test_runtime_data_grid_uses_the_runtime_seed_count(self):
        cells = config.runtime_data_grid(seed_offset=0, crop_n_vox_list=[64])
        assert len({c['seed'] for c in cells}) == config.RUNTIME_N_SEED

    @pytest.mark.parametrize('wrapper, kwargs',
                             [('data_grid', dict(seed=[0])),
                              ('effect_grid', dict(llr=[0.1]))])
    def test_a_misspelled_axis_raises(self, wrapper, kwargs):
        # the overrides are **kwargs, so a typo'd axis name would otherwise
        # sit alongside the real one and silently leave the default in place
        with pytest.raises(TypeError, match='unexpected keyword argument'):
            getattr(config, wrapper)(**kwargs)


class TestFitParams:
    """Which leaves ask for what, and how a sweep opts out of the device."""

    def test_only_glow_configures_its_fit(self):
        for cell in config.RUN_ANA_LIST:
            expected = (config.GLOW_FIT_PARAMS
                        if isinstance(cell['ana'], AnalysisGLOW) else None)
            assert cell['fit_params'] == expected

    @pytest.mark.parametrize(
        'label', [label for label in LABELS if label.startswith('runtime')])
    def test_timing_leaves_carry_none(self, label):
        # the runtime family times every method the same way (the leaf's own
        # default: all cores, CPU), or the figure compares hardware. fit_params
        # does not key a leaf either, so a device timing would be served from
        # the CPU timing's entry rather than measured.
        for cell in config.CONFIG[label][2]:
            assert cell.get('fit_params') is None


class TestCellsBindToStages:
    """Every cell is valid kwargs for the stage the driver feeds it to."""

    _SIG_DATA = {'wgn': inspect.signature(data.data_factory_wgn),
                 'hcp': inspect.signature(data.data_factory_hcp)}

    @pytest.mark.parametrize('label', LABELS)
    def test_data_cells_bind(self, label):
        # source selects the builder; the rest are its kwargs
        for cell in config.CONFIG[label][0]:
            sig = self._SIG_DATA[cell['source']]
            sig.bind(**{k: v for k, v in cell.items() if k != 'source'})

    _SIG_EFFECT = {'single': inspect.signature(data.effect_factory_single),
                   'split': inspect.signature(data.effect_factory_split)}

    @pytest.mark.parametrize('label', LABELS)
    def test_effect_cells_bind(self, label):
        # kind selects the builder (effect_factory dispatches on it); exp and
        # its parent_uid are supplied by the driver; a None cell is the
        # no-plant null path
        for cell in config.CONFIG[label][1]:
            if cell is None:
                continue
            sig = self._SIG_EFFECT[cell['kind']]
            sig.bind(exp=None, parent_uid='',
                     **{k: v for k, v in cell.items() if k != 'kind'})

    @pytest.mark.parametrize('label', LABELS)
    def test_fnc_cells_bind(self, label):
        # exp / mask_target_list / parent_uid are supplied by the driver
        _, _, fnc_kwargs, fnc = config.CONFIG[label]
        sig = inspect.signature(fnc)
        for cell in fnc_kwargs:
            sig.bind(exp=None, mask_target_list=[], parent_uid='', **cell)
