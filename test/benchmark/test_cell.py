"""Tests for glow._extra.benchmark.cell: the payload and its rebuild.

Two claims carry the design. The payload is small -- it holds masks, a
design and a constant offset, never the images. And a rebuild from it is the
experiment the cell realized, value for value, which is what lets a leaf
rebuild instead of unpickling one.
"""
import io
import random

import joblib
import numpy as np
import pytest

from glow._extra.benchmark import cell as cell_mod
from glow._extra.benchmark import data
from glow._extra.benchmark.cell import (ExpEffect, build_cell,
                                        exp_effect_recipe, get_exp_effect)
from glow.effect import ExtenterMinVar
from glow.experiment.exper import ExperimentScaled


def _fresh_seed() -> int:
    """A seed unlikely to already be in the on-disk cache, so a call misses."""
    return random.randrange(2 ** 31)


def _kwargs_data(**kwargs) -> dict:
    """One small WGN data cell, on a fresh seed unless given one."""
    return {**dict(source='wgn', shape=(7, 7, 7), b=2, num_img=20, a=1,
                   seed=_fresh_seed()), **kwargs}


def _kwargs_effect(**kwargs) -> dict:
    """One MinVar effect cell."""
    return {**dict(effect_llr=0.1, extenter_cls=ExtenterMinVar,
                   n_vox_frac=0.1), **kwargs}


@pytest.fixture(autouse=True)
def _records_to_tmp(monkeypatch, tmp_path):
    """Mirror the recorder's per-hash files to a tmp dir, not the real one."""
    monkeypatch.setattr(data.RECORDER, 'folder', tmp_path)


@pytest.fixture(autouse=True)
def _empty_build_memo():
    """Leave no built experiment behind for the next test to read."""
    cell_mod._BUILD_MEMO.clear()
    yield
    cell_mod._BUILD_MEMO.clear()


class TestPayload:
    """What get_exp_effect stores, and what it refuses to store."""

    def test_holds_masks_and_a_design_but_no_images(self):
        cell = get_exp_effect(_kwargs_data(), _kwargs_effect())
        num_vox = int(cell.mask_ana.sum())

        assert cell.mask_ana.shape == (7, 7, 7)
        assert cell.mask_ana.dtype == bool
        assert cell.mask_dead.shape == (7, 7, 7)
        assert cell.x.shape == (2, 20)                 # the a=1 row + bias
        assert cell.contrast.tolist() == [False, True]
        assert isinstance(cell.y_hash, str)
        # nothing of image size: the offset is one constant per feature and
        # image, shared by every voxel of the support
        assert [e['offset'].shape for e in cell.effect_list] == [(2, 20)]
        assert not hasattr(cell, 'y')
        assert num_vox > 0

    def test_is_smaller_than_the_images_it_replaces(self):
        cell = get_exp_effect(_kwargs_data(), _kwargs_effect())
        exp = cell.build()

        buffer = io.BytesIO()
        joblib.dump(cell, buffer, compress=3)
        assert len(buffer.getvalue()) < exp.y.nbytes

    def test_the_support_is_a_fraction_of_the_analysed_volume(self):
        # the sizing rule: n_vox_frac of what is analysed, not of what the
        # source holds (which is larger: the screen has already run)
        cell = get_exp_effect(_kwargs_data(), _kwargs_effect(n_vox_frac=0.2))
        mask, = cell.mask_target_list
        assert int(mask.sum()) == round(0.2 * int(cell.mask_ana.sum()))

    def test_the_null_path_plants_nothing(self):
        cell = get_exp_effect(_kwargs_data())
        assert cell.kwargs_effect is None
        assert cell.effect_list == []
        assert cell.mask_target_list == []

    def test_the_screen_is_recorded_not_rerun(self):
        # mask_dead names what drop_constant_vox cut, so the rebuild neither
        # re-screens nor has to agree with another box on a borderline voxel
        cell = get_exp_effect(_kwargs_data())
        assert cell.mask_dead is not None
        # the analysed voxels and the dead ones do not overlap
        assert not (cell.mask_ana & cell.mask_dead).any()


class TestRebuild:
    """build() reproduces the experiment the cell was realized from."""

    def test_reproduces_the_realized_experiment(self):
        kwargs_data, kwargs_effect = _kwargs_data(), _kwargs_effect()
        cell = get_exp_effect(kwargs_data, kwargs_effect)

        # what the realization itself produced, built the long way
        exp = data.build_clean(kwargs_data)
        exp, _ = data.plant_effect(
            exp, seed=cell_mod.seed_from_uid(cell_mod.data_uid(kwargs_data)),
            **kwargs_effect)
        want = ExperimentScaled.from_exp(exp)

        got = cell.build()
        np.testing.assert_array_equal(got.mask_idx, want.mask_idx)
        np.testing.assert_array_equal(got.y, want.y)
        np.testing.assert_array_equal(got.x, want.x)

    def test_returns_a_scaled_experiment_carrying_its_source(self):
        exp = get_exp_effect(_kwargs_data(), _kwargs_effect()).build()
        assert isinstance(exp, ExperimentScaled)
        # the source survives, so a kernel can still borrow context
        assert exp.source is not None
        assert exp.meta['features'] == ['feat_0', 'feat_1']

    def test_the_plant_is_replayable(self):
        # the offsets go on through add_offset, so they land in patch_list
        # and an inflate can replay them onto voxels loaded later
        cell = get_exp_effect(_kwargs_data(), _kwargs_effect())
        exp = cell.build()
        assert len(exp.patch_list) == 1
        np.testing.assert_array_equal(exp.patch_list[0]['mask'],
                                      cell.mask_target_list[0])

    def test_builds_the_null_path_too(self):
        exp = get_exp_effect(_kwargs_data()).build()
        assert exp.patch_list == []

    def test_build_cell_reuses_the_last_experiment(self):
        # a cell's leaves each need the same experiment; the first builds it
        cell = get_exp_effect(_kwargs_data(), _kwargs_effect())
        assert build_cell(cell) is build_cell(cell)

    def test_build_cell_drops_the_previous_cell(self):
        # one entry, so a whole grid's experiments cannot accumulate
        first = get_exp_effect(_kwargs_data(), _kwargs_effect())
        second = get_exp_effect(_kwargs_data(), _kwargs_effect())
        build_cell(first)
        build_cell(second)
        assert list(cell_mod._BUILD_MEMO) == [second.uid]


class TestValidation:
    """A rebuild is checked, not trusted."""

    def test_different_images_raise(self):
        cell = get_exp_effect(_kwargs_data(), _kwargs_effect())
        # the source now reads something else: a different draw
        cell.source.seed = cell.source.seed + 1
        with pytest.raises(ValueError, match='no longer reads the images'):
            cell.build()

    def test_a_wrong_offset_raises(self):
        cell = get_exp_effect(_kwargs_data(), _kwargs_effect())
        cell.effect_list[0]['offset'] = cell.effect_list[0]['offset'] * 0.5
        with pytest.raises(ValueError, match='per-voxel LLR'):
            cell.build()

    def test_a_wrong_support_raises(self):
        # the same offset over the wrong voxels: the region's LLR moves,
        # which is what makes this an end-to-end check
        cell = get_exp_effect(_kwargs_data(), _kwargs_effect())
        mask = cell.effect_list[0]['mask']
        rolled = np.roll(mask, shift=3, axis=0) & (cell.mask_ana)
        cell.effect_list[0]['mask'] = rolled
        with pytest.raises(ValueError, match='per-voxel LLR'):
            cell.build()

    def test_a_voxel_the_source_cannot_read_raises(self):
        cell = get_exp_effect(_kwargs_data(), _kwargs_effect())
        bigger = np.ones((8, 8, 8), dtype=bool)
        cell.mask_ana = bigger
        with pytest.raises(ValueError, match='does not match'):
            cell.build()

    def test_a_faithful_rebuild_passes(self):
        # the tolerance is not so tight that an honest rebuild trips it
        cell = get_exp_effect(_kwargs_data(), _kwargs_effect())
        cell.build()


class TestIdentity:
    """A cell is named by what was asked for, never by what came back."""

    def test_the_recipe_names_the_cell_before_it_is_built(self):
        kwargs_data, kwargs_effect = _kwargs_data(), _kwargs_effect()
        uid = exp_effect_recipe(kwargs_data, kwargs_effect).uid
        cell = get_exp_effect(kwargs_data, kwargs_effect)
        assert cell.uid == uid

    def test_the_record_is_filed_under_that_uid(self):
        data.RECORDER.records.clear()
        kwargs_data, kwargs_effect = _kwargs_data(), _kwargs_effect()
        get_exp_effect(kwargs_data, kwargs_effect)

        record, = data.RECORDER.records.values()
        assert record['uid'] == exp_effect_recipe(kwargs_data,
                                                  kwargs_effect).uid
        assert record['op'].endswith('get_exp_effect')
        assert record['parents'] == []

    def test_the_declared_knobs_flatten_to_one_column_each(self):
        # what the figures group by: recurse_list expands each cell spec
        data.RECORDER.records.clear()
        kwargs_data = _kwargs_data(seed=7)
        get_exp_effect(kwargs_data, _kwargs_effect())

        row = data.RECORDER.flatten_to_df().iloc[0]
        assert row['get_exp_effect.in.kwargs_data.source'] == 'wgn'
        assert row['get_exp_effect.in.kwargs_data.seed'] == 7
        assert row['get_exp_effect.in.kwargs_effect.effect_llr'] == 0.1

    def test_the_null_cell_is_a_different_cell(self):
        kwargs_data = _kwargs_data()
        assert (exp_effect_recipe(kwargs_data).uid
                != exp_effect_recipe(kwargs_data, _kwargs_effect()).uid)

    def test_the_placement_follows_the_uid(self):
        # each data realization plants somewhere of its own...
        effect = _kwargs_effect()
        one = get_exp_effect(_kwargs_data(), effect)
        two = get_exp_effect(_kwargs_data(), effect)
        assert not np.array_equal(one.mask_target_list[0],
                                  two.mask_target_list[0])

    def test_the_placement_holds_across_the_strength_grid(self):
        # ...but a cell's support does not move when only the strength does,
        # since the seed comes from the data half of the uid
        kwargs_data = _kwargs_data()
        weak = get_exp_effect(kwargs_data, _kwargs_effect(effect_llr=0.05))
        strong = get_exp_effect(kwargs_data, _kwargs_effect(effect_llr=0.3))
        np.testing.assert_array_equal(weak.mask_target_list[0],
                                      strong.mask_target_list[0])


class TestSplitCell:
    """The two-effect (cleaving) cell, through the same payload."""

    def test_plants_two_disjoint_supports(self):
        cell = get_exp_effect(
            _kwargs_data(b=3),
            _kwargs_effect(kind='split', angle=45.0, effect_llr=0.2))
        mask0, mask1 = cell.mask_target_list
        assert not (mask0 & mask1).any()
        assert len(cell.effect_list) == 2
        # different directions, so the two offsets differ
        assert not np.allclose(cell.effect_list[0]['offset'],
                               cell.effect_list[1]['offset'])

    def test_both_halves_are_validated(self):
        cell = get_exp_effect(
            _kwargs_data(b=3),
            _kwargs_effect(kind='split', angle=45.0, effect_llr=0.2))
        cell.build()
        cell.effect_list[1]['offset'] = cell.effect_list[1]['offset'] * 2
        with pytest.raises(ValueError, match='effect 1'):
            cell.build()


def test_the_payload_is_an_exp_effect():
    assert isinstance(get_exp_effect(_kwargs_data()), ExpEffect)
