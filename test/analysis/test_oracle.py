"""Tests for AnalysisOracleSegment: the perfect-segmentation bound.

Three claims carry the arm. Its regions are the planted supports plus the
one region holding everything they left, read off the experiment rather
than declared (so no computed array can reach a benchmark identity). It
reads them once, before any permutation, since a permuted draw carries no
plants at all. And each region is scored by pooling its voxels into one
(e, h) at its own voxel count, which is the statistic GLOW walks its tree
with -- so what the arm measures is segmentation and multiplicity alone.
"""
import random

import numpy as np
import pytest

from glow._extra.benchmark import store
from glow._extra.benchmark.cell import build_cell, get_exp_effect
from glow._extra.benchmark.data import build_clean_wgn, plant_effect
from glow._extra.benchmark.recipe import canon, recipe_id
from glow.analysis import AnalysisOracleSegment
from glow.analysis.mancova import (decompose, get_hotel_tr, get_llr,
                                   get_mancova, get_wilks)
from glow.effect import ExtenterMinVar
from glow.experiment.exper import ExperimentScaled

# small enough that a fit is a fraction of a second, strong enough that the
# plant is recovered without a wide permutation null
_SHAPE = (6, 6, 6)
_NUM_IMG = 20
_SEED = 1
_N_PERM_FWER = 100
_ALPHA_FWER = 0.05
_EFFECT_LLR = 0.5


@pytest.fixture(autouse=True)
def _records_to_tmp(monkeypatch, tmp_path):
    """Mirror the recorder's per-hash files to a tmp dir, not the real one."""
    monkeypatch.setattr(store.RECORDER, 'folder', tmp_path)


def _clean(*, b: int = 2, seed: int = _SEED):
    """Build a small effect-free WGN experiment.

    build_clean_wgn / plant_effect are the uncached halves of a cell's
    realization, so a test calling them writes no cache entry and no
    record (cf. get_exp_effect above them).
    """
    return build_clean_wgn(shape=_SHAPE, b=b, num_img=_NUM_IMG, seed=seed)


def _planted(*, kind: str = 'single', n_vox_frac: float = 0.1,
             effect_llr: float = _EFFECT_LLR, seed: int = _SEED):
    """Build a small WGN experiment with one plant, or a split pair.

    Returns:
        exp (Experiment): the planted experiment.
        mask_target_list (list): the realized (X, Y, Z) bool supports, one
            per plant, in plant order.
    """
    return plant_effect(_clean(seed=seed), seed=seed, kind=kind,
                        effect_llr=effect_llr, extenter_cls=ExtenterMinVar,
                        n_vox_frac=n_vox_frac,
                        angle=90.0 if kind == 'split' else None)


def _num_reg(label_map) -> int:
    """Count the regions a label map holds (labels 0..num_reg-1)."""
    return int(label_map.max()) + 1


def _ana(**kwargs):
    """An oracle recipe at the test's permutation count."""
    return AnalysisOracleSegment(n_perm_fwer=_N_PERM_FWER,
                                 alpha_fwer=_ALPHA_FWER, **kwargs)


def _regions(exp):
    """The oracle segmentation of a scaled exp, as (label_map, vox_list)."""
    exp = ExperimentScaled.from_exp(exp)
    label_map = AnalysisOracleSegment.get_label_map(exp)
    return label_map, AnalysisOracleSegment.get_vox_list(exp, label_map)


class TestRegions:
    """What the oracle segmentation is, on each path a cache walks."""

    def test_no_plant_is_one_region_over_the_whole_volume(self):
        exp = _clean()
        label_map = AnalysisOracleSegment.get_label_map(exp)

        assert _num_reg(label_map) == 1
        np.testing.assert_array_equal(label_map > -1, exp.mask_idx > -1)

    def test_one_plant_is_its_support_plus_the_complement(self):
        exp, (mask,) = _planted()
        label_map = AnalysisOracleSegment.get_label_map(exp)

        assert _num_reg(label_map) == 2
        np.testing.assert_array_equal(label_map == 0, mask)
        np.testing.assert_array_equal(label_map == 1,
                                      (exp.mask_idx > -1) & ~mask)

    def test_two_plants_are_both_supports_plus_the_complement(self):
        exp, mask_list = _planted(kind='split')
        label_map = AnalysisOracleSegment.get_label_map(exp)

        assert len(mask_list) == 2
        assert _num_reg(label_map) == 3
        for reg_idx, mask in enumerate(mask_list):
            np.testing.assert_array_equal(label_map == reg_idx, mask)

    def test_a_plant_over_the_whole_volume_leaves_one_region(self):
        # the extent sweep's top point plants over the whole analysis
        # volume, so the complement comes out empty and is dropped rather
        # than tested as a region no voxel is in
        exp, (mask,) = _planted(n_vox_frac=1.0)
        label_map = AnalysisOracleSegment.get_label_map(exp)

        assert mask.sum() == (exp.mask_idx > -1).sum()
        assert _num_reg(label_map) == 1

    def test_overlapping_supports_go_to_the_later_plant(self):
        # they should not overlap, but the labels are assigned in plant
        # order, so the regions stay a partition when they do
        exp = _clean()
        mask_active = exp.mask_idx > -1
        offset = np.zeros((exp.y.shape[0], _NUM_IMG), dtype=exp.y.dtype)

        mask_a, mask_b = mask_active.copy(), mask_active.copy()
        mask_a[exp.mask_idx > 20] = False
        mask_b[(exp.mask_idx > 30) | (exp.mask_idx < 10)] = False
        exp = exp.add_offset(offset, mask=mask_a).add_offset(offset,
                                                             mask=mask_b)

        label_map = AnalysisOracleSegment.get_label_map(exp)
        assert _num_reg(label_map) == 3
        np.testing.assert_array_equal(label_map == 0, mask_a & ~mask_b)
        np.testing.assert_array_equal(label_map == 1, mask_b)

    def test_a_swallowed_support_is_not_tested_empty(self):
        # a plant entirely inside a later one keeps no voxel of its own
        exp = _clean()
        mask_active = exp.mask_idx > -1
        offset = np.zeros((exp.y.shape[0], _NUM_IMG), dtype=exp.y.dtype)

        mask_inner = mask_active & (exp.mask_idx < 10)
        exp = exp.add_offset(offset, mask=mask_inner)
        exp = exp.add_offset(offset, mask=mask_active & (exp.mask_idx < 20))

        # the swallowed plant, the swallower, no complement label wasted
        assert _num_reg(AnalysisOracleSegment.get_label_map(exp)) == 2

    def test_the_regions_partition_the_analysed_voxels(self):
        exp, _ = _planted(kind='split')
        label_map, vox_list = _regions(exp)

        num_vox = int((exp.mask_idx > -1).sum())
        assert sorted(np.concatenate(vox_list)) == list(range(num_vox))
        assert [len(v) for v in vox_list] == [
            int((label_map == r).sum()) for r in range(len(vox_list))]


class TestStat:
    """One GLM per region, over all of that region's voxels."""

    def test_defaults_take_the_arms_stat_and_standardize(self):
        # the arm is read beside VBA / CET, so it takes their stat (the
        # raw Hotelling-Lawley trace); z-scoring is the one knob it does
        # not share with them (see fit)
        ana = AnalysisOracleSegment(n_perm_fwer=2)
        assert ana.get_stat is get_hotel_tr
        assert ana.z_flag is True

    def test_the_stat_pools_the_regions_voxels(self):
        exp, _ = _planted()
        exp_scaled = ExperimentScaled.from_exp(exp)
        _, vox_list = _regions(exp)
        q_tup = decompose(exp_scaled.x, exp_scaled.contrast)

        ana = _ana(z_flag=False).fit(exp)
        for reg_idx, vox_idx in enumerate(vox_list):
            e, h, _ = get_mancova(y=exp_scaled.y[:, :, vox_idx], q_tup=q_tup)
            assert np.isclose(ana.stat[0, reg_idx], get_hotel_tr(e, h),
                              rtol=1e-4)

    def test_n_is_the_regions_voxel_count(self):
        # the convention the GLOW arms walk their tree with: a region of n
        # voxels contributes n observations (get_stat(e, h, n=size)), not
        # the one a region mean would
        exp, _ = _planted()
        exp_scaled = ExperimentScaled.from_exp(exp)
        _, vox_list = _regions(exp)
        q_tup = decompose(exp_scaled.x, exp_scaled.contrast)

        ana = _ana(z_flag=False, get_stat=get_llr).fit(exp)
        for reg_idx, vox_idx in enumerate(vox_list):
            e, h, _ = get_mancova(y=exp_scaled.y[:, :, vox_idx], q_tup=q_tup)
            size = len(vox_idx)
            assert size > 1
            assert np.isclose(ana.stat[0, reg_idx], get_llr(e, h, n=size),
                              rtol=1e-4)
            assert not np.isclose(ana.stat[0, reg_idx], get_llr(e, h, n=1),
                                  rtol=1e-4)

    def test_the_planted_region_re_measures_its_per_voxel_llr(self):
        # the whole-region LLR of a plant is its per-voxel target times
        # its size, which is what the pooled statistic at n=size reports
        exp, (mask,) = _planted()
        ana = _ana(z_flag=False, get_stat=get_llr).fit(exp)

        assert np.isclose(ana.stat[0, 0], _EFFECT_LLR * int(mask.sum()),
                          rtol=1e-2)

    def test_the_raw_null_scale_falls_with_region_size(self):
        # why z_flag defaults on: e accumulates over a region's voxels
        # where h does not, so the small region's raw null sits an order
        # of magnitude above the large one's and a raw max-stat family
        # would compare them on scale rather than on evidence
        exp, _ = _planted()
        _, vox_list = _regions(exp)
        ana = _ana(z_flag=False).fit(exp)

        assert len(vox_list[0]) < len(vox_list[1])
        assert ana.stat[1:, 0].mean() > 10 * ana.stat[1:, 1].mean()

    def test_get_stat_is_pluggable(self):
        exp, _ = _planted()
        ana = _ana(z_flag=False, get_stat=get_wilks).fit(exp)

        # 1 - Wilks' lambda lives in [0, 1); the trace does not
        assert ana.get_stat is get_wilks
        assert (ana.stat[np.isfinite(ana.stat)] < 1).all()


class TestDiscovery:
    """What a fit hands the benchmark's scorer."""

    def test_a_strong_plant_is_recovered_exactly(self):
        exp, (mask,) = _planted()
        ana = _ana().fit(exp)

        eff, = ana.effect_list
        np.testing.assert_array_equal(eff.mask, mask)
        # the regions are the hypotheses, so a discovery keeps the region
        # index and the p-value it was rejected at (score.score_effects
        # records both)
        assert eff.reg_idx == 0
        assert eff.pval_fwer <= _ALPHA_FWER
        assert ana.fwer.pval[0] == eff.pval_fwer

    def test_no_plant_discovers_nothing(self):
        # one hypothesis over effect-free images, on a fixed seed
        ana = _ana().fit(_clean())

        assert ana.stat.shape == (_N_PERM_FWER + 1, 1)
        assert ana.effect_list == []
        assert ana.fwer.pval[0] > _ALPHA_FWER

    def test_a_significant_null_region_is_reported_as_one(self):
        # the effect-free remainder is a hypothesis like any other: it
        # carries its own p-value in the family, and rejecting it hands
        # the scorer a region of that size -- the false positive it is --
        # rather than nothing
        exp, (mask,) = _planted()
        ana = _ana().fit(exp)
        assert ana.fwer.reg_active.all()
        assert np.isfinite(ana.fwer.pval).all()

        ana.fwer.reg_sig[:] = True
        eff_list = ana._discover(ExperimentScaled.from_exp(exp))
        assert [eff.reg_idx for eff in eff_list] == [0, 1]
        np.testing.assert_array_equal(eff_list[1].mask,
                                      (exp.mask_idx > -1) & ~mask)

    def test_fit_returns_self_and_scales_idempotently(self):
        exp, _ = _planted()
        ana = _ana()
        assert ana.fit(exp) is ana

        # a pre-scaled exp passes through ExperimentScaled.from_exp, so
        # the arm may be handed either
        again = _ana().fit(ExperimentScaled.from_exp(exp))
        np.testing.assert_allclose(again.stat, ana.stat)

    def test_gpu_raises(self):
        with pytest.raises(ValueError, match='no GPU backend'):
            _ana().fit(_planted()[0], gpu=True)


class TestPermutationWalk:
    """The regions are captured once, not re-read per draw."""

    def test_a_permuted_draw_carries_no_plants(self):
        # the trap the capture exists for: Experiment.permute builds its
        # draw with no patch_list, so the oracle segmentation of a
        # permuted experiment is the whole volume as one region
        exp, _ = _planted()
        assert exp.permute(1).patch_list == []
        assert _num_reg(AnalysisOracleSegment.get_label_map(
            exp.permute(1))) == 1

    def test_the_null_draws_keep_every_region(self):
        # under a per-draw read each null row would come back with one
        # region and broadcast across the columns, leaving them identical
        exp, _ = _planted()
        ana = _ana(z_flag=False).fit(exp)

        assert ana.stat.shape == (_N_PERM_FWER + 1, 2)
        assert np.isfinite(ana.stat).all()
        assert not np.allclose(ana.stat[1:, 0], ana.stat[1:, 1])

    def test_a_rebuilt_cell_fits_end_to_end(self):
        # the benchmark path: a cell's Experiment is rebuilt from masks
        # and offsets, so its patch_list is the one add_offset recorded on
        # the rebuild rather than one carried over from the realization
        kwargs_data = dict(source='wgn', shape=_SHAPE, b=2,
                           num_img=_NUM_IMG, a=1,
                           seed=random.randrange(2 ** 31))
        kwargs_effect = dict(effect_llr=_EFFECT_LLR,
                             extenter_cls=ExtenterMinVar, n_vox_frac=0.1)
        cell = get_exp_effect(kwargs_data, kwargs_effect)
        ana = _ana().fit(build_cell(cell))

        assert ana.stat.shape == (_N_PERM_FWER + 1, 2)
        mask, = cell.mask_target_list
        assert any(np.array_equal(eff.mask, mask) for eff in ana.effect_list)

    @pytest.mark.parametrize('n_jobs', [1, 2])
    def test_the_walk_is_independent_of_n_jobs(self, n_jobs):
        # each row is seeded by its permutation index, which is why n_jobs
        # stays out of every benchmark cache key
        exp, _ = _planted()
        np.testing.assert_allclose(_ana().fit(exp, n_jobs=n_jobs).stat,
                                   _ana().fit(exp, n_jobs=1).stat)


class TestRecipeIsDeclarative:
    """No computed array may reach the recipe, before or after a fit."""

    def test_repr_is_the_declared_knobs_and_survives_a_fit(self):
        ana = _ana()
        expect = (f'AnalysisOracleSegment(get_stat=get_hotel_tr, '
                  f'n_perm_fwer={_N_PERM_FWER}, alpha_fwer={_ALPHA_FWER}, '
                  f'z_flag=True)')
        assert repr(ana) == expect

        # the label map and the stat matrix are fit outputs, so a fitted
        # recipe still hashes to the artifact it named before it ran
        ana.fit(_planted()[0])
        assert repr(ana) == expect
        assert repr(_ana()) == repr(ana)

    def test_record_fields_hold_no_arrays(self):
        ana = _ana().fit(_planted()[0])
        assert ana.label_map is not None
        for name in AnalysisOracleSegment.RECORD_FIELDS:
            assert not isinstance(getattr(ana, name), np.ndarray)

    def test_canon_and_recipe_id_accept_a_fitted_recipe(self):
        ana = _ana().fit(_planted()[0])

        assert canon({'ana': ana}) == {'ana': repr(ana)}
        uid = recipe_id('run_ana', {'ana': ana})
        assert uid == recipe_id('run_ana', {'ana': _ana()})
        assert uid != recipe_id('run_ana', {'ana': _ana(z_flag=False)})
