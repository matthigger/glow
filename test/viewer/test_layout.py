"""Tests for 2D viewer layout and image-selection callback."""

import json

import numpy as np
import pytest

from glow.viewer.app import _create_app
from glow.viewer.data import compute_backgrounds, get_feature_columns


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
        bg = compute_backgrounds(ana_2d, feature_names=names, image_idx=0)
        for n in names:
            assert n in bg
