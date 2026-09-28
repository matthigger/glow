"""Tests for glow._extra.benchmark.data: realizing one cell's Experiment.

build_clean and plant_effect are the two uncached halves of a cell's
realization. Neither is memoised or recorded -- the small payload that
replaces their Experiment is what gets stored (see test_cell) -- so what is
covered here is what they compute: the shapes and design a build produces,
the crop and the screen it applies, and where a plant lands and how big it
is.
"""
import random

import numpy as np
import pytest

from glow._extra.benchmark import data, hcp
from glow.effect import ExtenterMinVar, ExtenterSphere
from glow.experiment import ExperimentImageOnly
from glow.experiment.exper import NoBiasTermWarning
from glow.experiment.smooth import get_sigma_vox


def _fresh_seed() -> int:
    """A seed no other test shares, so no build can be confused with one."""
    return random.randrange(2 ** 31)


# ---------------------------------------------------------------------------
# the clean (effect-free) build
# ---------------------------------------------------------------------------

class TestBuildCleanWGN:
    def test_shapes_and_bias(self):
        exp = data.build_clean_wgn(shape=(6, 6, 6), b=3, num_img=40, a=2,
                                   has_bias=True, seed=_fresh_seed())
        # b channels, num_img images, 6*6*6 voxels; x is the a rows + 1 bias
        assert exp.y.shape == (3, 40, 216)
        assert exp.x.shape == (3, 40)
        # bias row prepended -> a leading False then the a interest columns
        assert exp.contrast.tolist() == [False, True, True]

    def test_contrast_arg_sets_a(self):
        # has_bias=False -> no all-ones row, which warns (regression at origin)
        with pytest.warns(NoBiasTermWarning):
            exp = data.build_clean_wgn(contrast=np.array([False, True]),
                                       has_bias=False, b=2, num_img=20,
                                       seed=_fresh_seed())
        assert exp.x.shape == (2, 20)
        assert exp.contrast.tolist() == [False, True]

    def test_extenter_crops_to_support(self):
        ext = ExtenterSphere(n_vox=20, connected=True, seed=0, contiguous=True)
        exp = data.build_clean_wgn(shape=(8, 8, 8), b=2, num_img=30,
                                   extenter=ext, seed=_fresh_seed())
        assert (exp.mask_idx > -1).sum() == 20
        assert exp.y.shape == (2, 30, 20)

    def test_sits_on_the_hcp_grid(self):
        # one fwhm in mm is one kernel in voxels on both sources
        exp = data.build_clean_wgn(shape=(5, 5, 5), b=1, num_img=20,
                                   seed=_fresh_seed())
        hcp_affine = np.diag([-2.0, -2.0, 2.0, 1.0])
        np.testing.assert_allclose(get_sigma_vox(2.0, exp.meta['affine']),
                                   get_sigma_vox(2.0, hcp_affine))
        assert data.WGN_VOX_MM == 2.0

    def test_deterministic_for_a_seed(self):
        kw = dict(shape=(5, 5, 5), b=2, num_img=20, a=1, seed=_fresh_seed())
        e0 = data.build_clean_wgn(**kw)
        e1 = data.build_clean_wgn(**kw)
        assert np.array_equal(e0.y, e1.y)
        assert np.array_equal(e0.x, e1.x)

    def test_carries_a_source_so_it_can_grow_back(self):
        # what lets a kernel read past the crop (glow.experiment.source)
        exp = data.build_clean_wgn(shape=(6, 6, 6), b=2, num_img=20,
                                   extenter=ExtenterSphere(radius=2, seed=0),
                                   seed=_fresh_seed())
        assert exp.source is not None
        assert exp.source.mask.shape == (6, 6, 6)


class TestBuildCleanDispatch:
    def test_wgn_routes_to_its_builder(self):
        kw = dict(shape=(4, 4, 4), b=2, num_img=10, a=1, seed=_fresh_seed())
        assert np.array_equal(
            data.build_clean({'source': 'wgn', **kw}).y,
            data.build_clean_wgn(**kw).y)

    def test_bad_source_raises(self):
        with pytest.raises(ValueError, match="'wgn' or 'hcp'"):
            data.build_clean({'source': 'nope'})


class TestBuildCleanHCP:
    def test_builds_through_the_bundle_loader(self, monkeypatch):
        feats = ('fa', 'md')
        # the image-only experiment the mocked bundle loader stands in for
        img = ExperimentImageOnly.from_gauss(shape=(4, 4, 4), b=len(feats),
                                             num_img=10, seed=0)
        seen = {}

        def fake_build(hcp_feats):
            seen['feats'] = tuple(hcp_feats)
            return img

        # mock the single HCP loader (the bundle glue)
        monkeypatch.setattr(hcp, 'build_exp_img_from_bundle', fake_build)
        exp = data.build_clean_hcp(hcp_feats=feats, a=1, seed=_fresh_seed())

        # the requested features reached the loader; x was sampled (a=1 + bias)
        assert seen['feats'] == feats
        assert exp.y.shape[0] == len(feats)
        assert exp.x.shape == (2, 10)


# ---------------------------------------------------------------------------
# the plant
# ---------------------------------------------------------------------------

class TestPlantEffect:
    def _clean(self, shape=(6, 6, 6), b=2):
        """A fresh clean experiment to plant on."""
        return data.build_clean_wgn(shape=shape, b=b, num_img=20, a=1,
                                    seed=_fresh_seed())

    def test_plants_effect_on_support(self):
        exp = self._clean()
        # the plant returns the supports as a list (one for 'single')
        exp_eff, (mask,) = data.plant_effect(
            exp, seed=0, effect_llr=0.05, extenter_cls=ExtenterMinVar,
            n_vox_frac=0.1)
        # effect added in place: same shapes, mask over the spatial grid, y
        # changed, and the extenter grew n_vox_frac of the analysis volume
        support = int((exp.mask_idx > -1).sum())
        assert exp_eff.y.shape == exp.y.shape
        assert mask.shape == exp.mask_idx.shape
        assert int(mask.sum()) == round(0.1 * support)
        assert not np.array_equal(exp.y, exp_eff.y)

    def test_the_offset_is_recorded_as_a_patch(self):
        # what lets an inflate replay the plant onto voxels loaded later
        exp = self._clean()
        exp_eff, (mask,) = data.plant_effect(
            exp, seed=0, effect_llr=0.05, extenter_cls=ExtenterMinVar,
            n_vox_frac=0.1)
        patch, = exp_eff.patch_list
        np.testing.assert_array_equal(patch['mask'], mask)
        assert patch['offset'].shape == exp.y.shape[:2]

    def test_the_seed_places_the_support(self):
        kw = dict(effect_llr=0.05, extenter_cls=ExtenterMinVar,
                  n_vox_frac=0.1)
        exp = self._clean()
        _, (one,) = data.plant_effect(exp, seed=1, **kw)
        _, (two,) = data.plant_effect(exp, seed=2, **kw)
        assert not np.array_equal(one, two)
        # and one seed is one place, whatever the strength asked for
        _, (weak,) = data.plant_effect(exp, seed=1, **kw)
        _, (strong,) = data.plant_effect(exp, seed=1,
                                         **{**kw, 'effect_llr': 0.3})
        np.testing.assert_array_equal(one, weak)
        np.testing.assert_array_equal(weak, strong)

    def test_extenter_kwargs_reach_the_extenter(self):
        # a centred sphere lands at one place whatever the seed
        kw = dict(effect_llr=0.05, extenter_cls=ExtenterSphere,
                  n_vox_frac=0.1, extenter_kwargs={'vox_init': 'center'})
        exp = self._clean()
        _, (one,) = data.plant_effect(exp, seed=1, **kw)
        _, (two,) = data.plant_effect(exp, seed=2, **kw)
        assert int(one.sum()) == round(0.1 * int((exp.mask_idx > -1).sum()))
        np.testing.assert_array_equal(one, two)

    def test_bad_kind_raises(self):
        with pytest.raises(ValueError, match="'single' or 'split'"):
            data.plant_effect(self._clean(), seed=0, kind='nope',
                              effect_llr=0.05, extenter_cls=ExtenterMinVar,
                              n_vox_frac=0.1)

    def test_split_plants_two_disjoint_halves(self):
        # the cleaving base: a data-driven ExtenterMinVar extent grown from
        # its own seeded start, bisected into two disjoint halves
        exp = self._clean(shape=(8, 8, 8), b=3)
        exp_eff, mask_target_list = data.plant_effect(
            exp, seed=0, kind='split', effect_llr=0.1,
            extenter_cls=ExtenterMinVar, n_vox_frac=0.1, angle=45.0)
        support = int((exp.mask_idx > -1).sum())
        assert len(mask_target_list) == 2
        mask0, mask1 = mask_target_list
        assert not (mask0 & mask1).any()
        assert int((mask0 | mask1).sum()) == round(0.1 * support)
        assert not np.array_equal(exp.y, exp_eff.y)

    def test_split_records_a_patch_per_half(self):
        exp = self._clean(shape=(8, 8, 8), b=3)
        exp_eff, masks = data.plant_effect(
            exp, seed=0, kind='split', effect_llr=0.1,
            extenter_cls=ExtenterMinVar, n_vox_frac=0.1, angle=30.0)
        assert len(exp_eff.patch_list) == 2
        for patch, mask in zip(exp_eff.patch_list, masks):
            np.testing.assert_array_equal(patch['mask'], mask)


# ---------------------------------------------------------------------------
# the constant-voxel screen
# ---------------------------------------------------------------------------

class TestDropsConstantVox:
    """Every build screens its experiment before handing it over.

    The screen runs in _sample_x_and_crop, shared by both builders, so
    one place decides which voxels exist and every recipe in the sweep
    then tests the same ones -- GLOW and the voxel-wise arms control
    FWER over one family rather than each pruning its own.
    """

    @staticmethod
    def build(**kwargs):
        """Build one clean WGN experiment."""
        return data.build_clean_wgn(**kwargs)

    def test_wgn_experiment_is_screened(self):
        """The builder's output carries mask_dead, not None."""
        exp = self.build(shape=(4, 4, 4), b=2, num_img=20, seed=0)
        assert exp.mask_dead is not None
        assert exp.mask_dead.shape == exp.mask_idx.shape

    def test_gaussian_noise_loses_nothing(self):
        """WGN varies everywhere, so the screen is a no-op on it."""
        exp = self.build(shape=(4, 4, 4), b=2, num_img=20, seed=0)
        assert exp.num_vox_dropped == 0
        assert exp.y.shape[2] == 64

    def test_crop_runs_first(self):
        """The count is over the cropped volume, not the whole image."""
        exp = self.build(shape=(6, 6, 6), b=2, num_img=20, seed=0,
                         extenter=ExtenterSphere(radius=2, seed=0))
        assert exp.mask_dead.sum() == exp.num_vox_dropped
        assert exp.y.shape[2] == int((exp.mask_idx > -1).sum())
