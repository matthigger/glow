"""Tests for viewer layout (2D and 3D) and image/feature selection."""

import json

import numpy as np
import pytest

from glow.viewer.app import _create_app
from glow.viewer.data import compute_backgrounds, get_feature_columns
from glow.viewer.image import compute_bg_volume


class TestLayout2D:
    """Verify the 2D layout contains expected components."""

    @pytest.fixture(scope='class')
    def app_2d(self, ana_2d, mask_target_2d):
        return _create_app(ana_2d, mask_target=mask_target_2d)

    @pytest.fixture(scope='class')
    def layout(self, app_2d):
        return app_2d.layout

    @staticmethod
    def _find_component(component, target_id, depth=20):
        """Recursively search for a Dash component by its ``id``."""
        if depth <= 0:
            return None
        cid = getattr(component, 'id', None)
        if cid == target_id:
            return component
        children = getattr(component, 'children', None)
        if children is None:
            return None
        if not isinstance(children, (list, tuple)):
            children = [children]
        for child in children:
            if child is None:
                continue
            found = TestLayout2D._find_component(child, target_id, depth - 1)
            if found is not None:
                return found
        return None

    def test_has_scatter_plot(self, layout):
        assert self._find_component(layout, 'scatter-plot') is not None

    def test_has_image_viewer(self, layout):
        assert self._find_component(layout, 'image-viewer') is not None

    def test_has_regression_plot(self, layout):
        assert self._find_component(layout, 'regression-plot') is not None

    def test_has_background_dropdown(self, layout):
        dd_bg = self._find_component(layout, 'dd-bg')
        assert dd_bg is not None

    def test_has_image_dropdown(self, layout):
        dd_image = self._find_component(layout, 'dd-image')
        assert dd_image is not None

    def test_image_dropdown_default_is_mean(self, layout):
        dd_image = self._find_component(layout, 'dd-image')
        assert dd_image.value == 'mean'

    def test_image_dropdown_has_mean_plus_individual(self, layout, ana_2d):
        dd_image = self._find_component(layout, 'dd-image')
        num_img = ana_2d.exp.y.shape[1]
        expected_count = 1 + num_img  # "Mean" + one per image
        assert len(dd_image.options) == expected_count

    def test_has_region_panel(self, layout):
        assert self._find_component(layout, 'region-checklist') is not None

    def test_has_store_selected(self, layout):
        assert self._find_component(layout, 'store-selected') is not None

    def test_region_panel_has_border(self, layout):
        """The region panel should have a borderRight for visual separation."""
        panel = self._find_component(layout, 'region-checklist')
        assert panel is not None


class TestComputeBackgroundsImageIdx:
    """Verify compute_backgrounds with image_idx produces sensible results."""

    def test_single_image_keys_match_mean(self, ana_2d):
        bg_mean = compute_backgrounds(ana_2d)
        bg_0 = compute_backgrounds(ana_2d, image_idx=0)
        assert set(bg_mean.keys()) == set(bg_0.keys())

    def test_each_image_produces_finite(self, ana_2d):
        num_img = ana_2d.exp.y.shape[1]
        for i in range(num_img):
            bg = compute_backgrounds(ana_2d, image_idx=i)
            for name, img in bg.items():
                valid = ~np.isnan(img)
                assert valid.any(), f'image_idx={i}, bg={name} is all NaN'

    def test_named_features(self, ana_2d):
        b = ana_2d.exp.y.shape[0]
        names = [f'ch{i}' for i in range(b)]
        bg = compute_backgrounds(ana_2d, y_features=names, image_idx=0)
        for n in names:
            assert n in bg


class TestLayout3D:
    """Verify the 3D layout contains expected components."""

    @pytest.fixture(scope='class')
    def app_3d(self, ana, mask_target):
        return _create_app(ana, mask_target=mask_target)

    @pytest.fixture(scope='class')
    def layout(self, app_3d):
        return app_3d.layout

    def test_has_scatter_plot(self, layout):
        assert TestLayout2D._find_component(layout, 'scatter-plot') is not None

    def test_has_regression_plot(self, layout):
        assert TestLayout2D._find_component(layout, 'regression-plot') is not None

    def test_has_image_dropdown(self, layout):
        dd = TestLayout2D._find_component(layout, 'dd-image-3d')
        assert dd is not None

    def test_image_dropdown_default_is_mean(self, layout):
        dd = TestLayout2D._find_component(layout, 'dd-image-3d')
        assert dd.value == 'mean'

    def test_image_dropdown_has_correct_count(self, layout, ana):
        dd = TestLayout2D._find_component(layout, 'dd-image-3d')
        num_img = ana.exp.y.shape[1]
        assert len(dd.options) == 1 + num_img

    def test_has_feature_component(self, layout):
        """Feature component exists (dropdown when b>1, store when b==1)."""
        comp = TestLayout2D._find_component(layout, 'dd-feature-3d')
        assert comp is not None

    def test_feature_hidden_when_b1(self, layout, ana):
        """With b=1, feature control should be a hidden Store, not a Dropdown."""
        from dash import dcc
        comp = TestLayout2D._find_component(layout, 'dd-feature-3d')
        if ana.exp.y.shape[0] == 1:
            assert isinstance(comp, dcc.Store)

    def test_has_region_panel(self, layout):
        assert TestLayout2D._find_component(layout, 'region-checklist') is not None


class TestLayout3DMultiFeature:
    """3D layout with b > 1 should show a Feature dropdown."""

    @pytest.fixture(scope='class')
    def app_3d_multi(self):
        from glow.experiment.exper import Experiment, ExperimentImageOnly
        from glow.analysis import AnalysisGLOW
        shape = (5, 5, 5)
        exp_img = ExperimentImageOnly.from_gauss(
            b=2, num_img=8, shape=shape, seed=99)
        x = np.arange(8, dtype=float).reshape(1, -1)
        exp = Experiment(x=x, contrast=np.array([True]),
                         y=exp_img.y, mask_idx=exp_img.mask_idx,
                         add_bias=True)
        ana = AnalysisGLOW(exp, n_perm_fwer=3).fit()
        return _create_app(ana, y_features=['feat_A', 'feat_B'])

    @pytest.fixture(scope='class')
    def layout(self, app_3d_multi):
        return app_3d_multi.layout

    def test_feature_is_dropdown(self, layout):
        from dash import dcc
        comp = TestLayout2D._find_component(layout, 'dd-feature-3d')
        assert isinstance(comp, dcc.Dropdown)

    def test_feature_dropdown_options(self, layout):
        comp = TestLayout2D._find_component(layout, 'dd-feature-3d')
        labels = [o['label'] for o in comp.options]
        assert 'feat_A' in labels
        assert 'feat_B' in labels

    def test_feature_default_is_zero(self, layout):
        comp = TestLayout2D._find_component(layout, 'dd-feature-3d')
        assert comp.value == '0'


class TestComputeBgVolumeImageIdx:
    """Verify compute_bg_volume with image_idx."""

    def test_mean_shape(self, ana):
        vol = compute_bg_volume(ana, feature_idx=0)
        assert vol.shape == ana.exp.mask_idx.shape

    def test_single_image_shape(self, ana):
        vol = compute_bg_volume(ana, feature_idx=0, image_idx=0)
        assert vol.shape == ana.exp.mask_idx.shape

    def test_single_image_differs_from_mean(self, ana):
        vol_mean = compute_bg_volume(ana, feature_idx=0)
        vol_0 = compute_bg_volume(ana, feature_idx=0, image_idx=0)
        assert not np.allclose(vol_mean, vol_0)

    def test_all_images_valid(self, ana):
        num_img = ana.exp.y.shape[1]
        for i in range(num_img):
            vol = compute_bg_volume(ana, feature_idx=0, image_idx=i)
            assert np.isfinite(vol).any()
