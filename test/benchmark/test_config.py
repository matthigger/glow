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

The stages (get_exp_effect / run_ana) and the sweep are covered by
test_cell / test_run / test_driver.
"""
import inspect

import pytest

from glow._extra.benchmark import config, data
from glow.analysis import (Analysis, AnalysisCET, AnalysisGLOWBase,
                           AnalysisOracleSegment, AnalysisVBA)
from glow.analysis.mancova import stat_dict


# every catalogue entry; the shape / bind checks cover all of them
LABELS = sorted(config.CONFIG)

# the run_ana caches, each sharing one of the two recipe-grid singletons
RUN_ANA_LABELS = [label for label in LABELS
                  if config.CONFIG[label][3].__name__ == 'run_ana']

# the run_ana caches taking the GLOW-only grid: the null path, whose
# calibration figure reads no voxel-wise arm
GLOW_ONLY_LABELS = ['null']

# the run_ana caches carrying their own recipe grid rather than a shared
# singleton: the smoothing sweep, which is the voxel-wise arms crossed with a
# kernel width and so cannot be shared with anything
OWN_GRID_LABELS = ['sweep_fwhm']


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

    @pytest.mark.parametrize('label', [x for x in RUN_ANA_LABELS
                                       if x not in OWN_GRID_LABELS])
    def test_run_ana_caches_share_the_recipe_grid(self, label):
        # the detection sweeps all fit/score over the one shared recipe grid,
        # by identity: strip_gpu / filter_ana_list rebuild cells rather than
        # mutating them precisely because it is a singleton. The null path
        # shares the GLOW-only grid on the same terms.
        expect = (config.RUN_ANA_GLOW_LIST if label in GLOW_ONLY_LABELS
                  else config.RUN_ANA_LIST)
        assert config.CONFIG[label][2] is expect

    def test_smooth_grid_is_the_voxel_arms_crossed_with_the_kernel(self):
        # one leaf per (voxel-wise arm, width), GLOW absent: it takes no
        # kernel, so it enters the figure as its own caches' flat reference.
        # Oracle-RBA absent too: it is handed the segmentation a kernel only
        # approximates, so a width would not mean anything on it.
        grid = config.RUN_ANA_SMOOTH_LIST
        assert len(grid) == (len(config.SMOOTH_LABEL_LIST)
                             * len(config.SMOOTH_FWHM_GRID))
        assert not any(isinstance(c['ana'], AnalysisGLOWBase) for c in grid)
        assert not any(isinstance(c['ana'], AnalysisOracleSegment)
                       for c in grid)
        assert {c['ana'].fwhm for c in grid} == {
            f or None for f in config.SMOOTH_FWHM_GRID}

    def test_smooth_grid_leaves_the_unsmoothed_recipes_alone(self):
        # fwhm 0 must reach fit as None, so the baseline arm's repr -- and so
        # every artifact keyed on it -- is the paper's own, which is what
        # makes the unsmoothed leaf the very leaf the detection caches run
        unsmoothed = {repr(c['ana']) for c in config.RUN_ANA_SMOOTH_LIST
                      if c['ana'].fwhm is None}
        assert unsmoothed == {repr(config.ana_kwargs_dict[m])
                              for m in config.SMOOTH_LABEL_LIST}

    def test_tuning_strengths_are_cells_the_reported_caches_plant(self):
        # taken off EFFECT_LLR_GRID rather than typed, so a tuning cell is a
        # reported cell and a leaf the two share is computed once. Float
        # identity is the whole point here.
        assert len(config.TUNE_LLR_GRID) == 5
        assert set(config.TUNE_LLR_GRID) <= set(config.EFFECT_LLR_GRID)

    @pytest.mark.parametrize('label', ['sweep_fwhm', 'vba_stat'])
    def test_tuning_caches_span_the_tuning_axis(self, label):
        # a tuning cache reports a mean over cells, so it spans five
        # strengths where a reported cache spans eleven
        llr = {c['effect_llr'] for c in config.CONFIG[label][1]}
        assert llr == {float(v) for v in config.TUNE_LLR_GRID}

    def test_stat_pool_is_the_full_cross(self):
        # every arm carries every stat, which is what makes the appendix's
        # table a balanced panel: VBA's and CET's ties across the pool are
        # the derivation's own prediction (rank(H) = 1), so they are printed
        # rather than assumed. The width is free -- one voxel_stat_walk per
        # cell computes every stat (see grid.get_run_stat_list).
        assert len(config.RUN_STAT_LIST) == len(stat_dict) * 2 * 3
        per_stat = {}
        for spec in config.RUN_STAT_LIST:
            per_stat.setdefault(spec['stat_name'], []).append(spec)
        assert set(per_stat) == set(stat_dict)
        assert {len(v) for v in per_stat.values()} == {6}

    def test_recipes_are_analyses(self):
        assert config.RUN_ANA_LIST
        assert all(isinstance(c['ana'], Analysis)
                   for c in config.RUN_ANA_LIST)

    def test_glow_only_grid_holds_glow_alone(self):
        # a voxel-wise arm reaching this grid is a fit whose only reader
        # (plot._plot_calibration_faceted) drops it
        assert config.RUN_ANA_GLOW_LIST
        assert all(isinstance(c['ana'], AnalysisGLOWBase)
                   for c in config.RUN_ANA_GLOW_LIST)

    def test_spelled_out_arms_match_the_recipe_defaults(self):
        """The catalogue's spelled-out stats equal the arm defaults.

        config writes get_stat / z_flag out for VBA, VBA-TFCE, CET and
        Oracle-RBA rather than leaning on the recipe defaults, so the
        paper's arms read off the catalogue -- and states in a comment
        that the two agree. Nothing else checks that claim, so an edit to
        either side alone would silently split them, changing published
        numbers or invalidating records depending on which moved.

        This is the one place the catalogue's VALUES are pinned (cf. the
        module docstring): the stats are not a tuning knob but a finding,
        and the agreement is an invariant rather than a chosen grid size.
        """
        for label, cls, tfce_flag in [('VBA', AnalysisVBA, False),
                                      ('VBA-TFCE', AnalysisVBA, True),
                                      ('CET', AnalysisCET, False),
                                      ('Oracle-RBA', AnalysisOracleSegment,
                                       False)]:
            ana = config.ana_kwargs_dict[label]
            kwargs = dict(n_perm_fwer=ana.n_perm_fwer)
            if cls is AnalysisVBA:
                kwargs['tfce_flag'] = tfce_flag
            default = cls(**kwargs)

            assert ana.get_stat is default.get_stat, (
                f'{label}: catalogue stat {ana.get_stat.__name__} != '
                f'recipe default {default.get_stat.__name__}')
            assert ana.z_flag == default.z_flag, (
                f'{label}: catalogue z_flag {ana.z_flag} != '
                f'recipe default {default.z_flag}')

    def test_run_ana_cells_carry_the_recipe_and_how_to_run_it(self):
        # each leaf cell carries its ana and its fit_params, nothing else: the
        # method name is not passed (recovered from ana_kwargs_dict at read
        # time -- see run / plot), and fit_params is filtered back out of the
        # identity by the leaf (run.FIT_IGNORE)
        assert all(set(c) == {'ana', 'fit_params'}
                   for c in config.RUN_ANA_LIST)
        # the shared grid is the reported GLOW variants plus every non-GLOW
        # arm: an extra GLOW entry here would cost a per-perm fit in each of
        # the five caches that share it
        assert ([c['ana'] for c in config.RUN_ANA_LIST]
                == [ana for label, ana in config.ana_kwargs_dict.items()
                    if label in config.REPORTED_GLOW_LABEL_LIST
                    or label not in config.GLOW_LABEL_LIST])
        # the reported arms prune greedily, one per Ward projection,
        # headline first
        assert config.REPORTED_GLOW_LABEL_LIST[0] == \
            config.REPORTED_GLOW_LABEL
        arms = [config.ana_kwargs_dict[label]
                for label in config.REPORTED_GLOW_LABEL_LIST]
        assert {a.prune_rule for a in arms} == {'greedy'}
        assert len({str(a.cluster_mode) for a in arms}) == len(arms)
        assert config.REPORTED_GLOW_LABEL in config.GLOW_LABEL_LIST
        # the selection rule is not a recipe axis: prune_rule applies
        # downstream of the permutation test, so the prune cache settles it by
        # re-selecting one shared fit (config.PRUNE_RULES), and a dp recipe
        # here would buy a second full fit for a selection already recorded
        assert {a.prune_rule for a in config.ana_kwargs_dict.values()
                if isinstance(a, AnalysisGLOWBase)} == {'greedy'}
        assert 'dp' in config.PRUNE_RULES


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
                        if isinstance(cell['ana'], AnalysisGLOWBase) else None)
            assert cell['fit_params'] == expected

    def test_wall_clock_cache_gives_glow_the_device(self):
        # runtime_num_vox asks what a user waits on this machine, so GLOW is
        # given the card and the RAM-capped worker count; the voxel-wise arms
        # have no backend and take the leaf's default (all cores, CPU)
        for cell in config.CONFIG['runtime_num_vox'][2]:
            expected = (config.GLOW_FIT_PARAMS
                        if isinstance(cell['ana'], AnalysisGLOWBase) else None)
            assert cell['fit_params'] == expected

    @pytest.mark.parametrize(
        'label', [lbl for lbl in LABELS if lbl.startswith('runtime_1perm')])
    def test_growth_rate_leaves_take_no_fit_params(self, label):
        # run_ana_time_1perm pins its own core, device and BLAS thread -- the
        # serial contract is the measurement, not a cell's knob, so the leaf
        # has no fit_params parameter to bind
        for cell in config.CONFIG[label][2]:
            assert 'fit_params' not in cell


class TestCellsBindToStages:
    """Every cell is valid kwargs for the stage the driver feeds it to."""

    _SIG_DATA = {'wgn': inspect.signature(data.build_clean_wgn),
                 'hcp': inspect.signature(data.build_clean_hcp)}

    @pytest.mark.parametrize('label', LABELS)
    def test_data_cells_bind(self, label):
        # source selects the builder; the rest are its kwargs
        for cell in config.CONFIG[label][0]:
            sig = self._SIG_DATA[cell['source']]
            sig.bind(**{k: v for k, v in cell.items() if k != 'source'})

    @pytest.mark.parametrize('label', LABELS)
    def test_effect_cells_bind(self, label):
        # plant_effect dispatches on kind; exp and the placement seed are
        # supplied by the cell; a None cell is the no-plant null path
        sig = inspect.signature(data.plant_effect)
        for cell in config.CONFIG[label][1]:
            if cell is None:
                continue
            sig.bind(None, seed=0, **cell)

    @pytest.mark.parametrize('label', LABELS)
    def test_fnc_cells_bind(self, label):
        # the cell and its parent_uid are supplied by the driver
        _, _, fnc_kwargs, fnc = config.CONFIG[label]
        sig = inspect.signature(fnc)
        for cell in fnc_kwargs:
            sig.bind(None, parent_uid='', **cell)
