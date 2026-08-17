"""Tests for viewer layout (2D and 3D) and image/feature selection."""

import json

import numpy as np
import pytest

from glow._extra.viewer.app import _create_app
from glow._extra.viewer.data import compute_backgrounds, get_feature_columns
from glow._extra.viewer.image import compute_bg_volume

# the page's three rows, in the order they are laid out
ROW_IDS = ('segmentation-panel', 'detail-panel', 'image-panel')


def _page_rows(layout):
    """Return the (id, div) of each page row, in layout order.

    Reads the top level of the page only, so a row that ended up nested
    inside another would not be counted as one.
    """
    return [(c.id, c) for c in layout.children
            if getattr(c, 'id', None) in ROW_IDS]


class TestLayout2D:
    """Verify the 2D layout contains expected components."""

    @pytest.fixture(scope='class')
    def app_2d(self, ana_2d, exp_2d, mask_target_2d):
        return _create_app(ana_2d, exp_2d, mask_target=mask_target_2d)

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

    def test_image_dropdown_has_mean_plus_individual(self, layout, exp_2d):
        dd_image = self._find_component(layout, 'dd-image')
        num_img = exp_2d.y.shape[1]
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

    def test_rows_are_in_page_order(self, layout):
        assert [rid for rid, _ in _page_rows(layout)] == list(ROW_IDS)

    def test_image_has_its_own_row(self, layout):
        row = dict(_page_rows(layout))['image-panel']
        assert len(row.children) == 1
        assert self._find_component(row, 'image-viewer') is not None
        assert self._find_component(row, 'regression-plot') is None

    def test_detail_row_holds_the_region_plots(self, layout):
        row = dict(_page_rows(layout))['detail-panel']
        assert self._find_component(row, 'region-checklist') is not None
        assert self._find_component(row, 'regression-plot') is not None
        assert self._find_component(row, 'image-viewer') is None


class TestComputeBackgroundsImageIdx:
    """Verify compute_backgrounds with image_idx produces sensible results."""

    def test_single_image_keys_match_mean(self, exp_2d):
        bg_mean = compute_backgrounds(exp_2d)
        bg_0 = compute_backgrounds(exp_2d, image_idx=0)
        assert set(bg_mean.keys()) == set(bg_0.keys())

    def test_each_image_produces_finite(self, exp_2d):
        num_img = exp_2d.y.shape[1]
        for i in range(num_img):
            bg = compute_backgrounds(exp_2d, image_idx=i)
            for name, img in bg.items():
                valid = ~np.isnan(img)
                assert valid.any(), f'image_idx={i}, bg={name} is all NaN'

    def test_named_features(self, exp_2d):
        b = exp_2d.y.shape[0]
        names = [f'ch{i}' for i in range(b)]
        bg = compute_backgrounds(exp_2d, y_features=names, image_idx=0)
        for n in names:
            assert n in bg


class TestLayout3D:
    """Verify the 3D layout contains expected components."""

    @pytest.fixture(scope='class')
    def app_3d(self, ana, exp, mask_target):
        return _create_app(ana, exp, mask_target=mask_target)

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

    def test_image_dropdown_has_correct_count(self, layout, exp):
        dd = TestLayout2D._find_component(layout, 'dd-image-3d')
        num_img = exp.y.shape[1]
        assert len(dd.options) == 1 + num_img

    def test_has_feature_component(self, layout):
        """Feature component exists (dropdown when b>1, store when b==1)."""
        comp = TestLayout2D._find_component(layout, 'dd-feature-3d')
        assert comp is not None

    def test_feature_hidden_when_b1(self, layout, exp):
        """With b=1, feature control should be a hidden Store, not a Dropdown."""
        from dash import dcc
        comp = TestLayout2D._find_component(layout, 'dd-feature-3d')
        if exp.y.shape[0] == 1:
            assert isinstance(comp, dcc.Store)

    def test_has_region_panel(self, layout):
        assert TestLayout2D._find_component(layout, 'region-checklist') is not None

    def test_rows_are_in_page_order(self, layout):
        assert [rid for rid, _ in _page_rows(layout)] == list(ROW_IDS)

    def test_image_has_its_own_row(self, layout):
        """The three ortho slicers get the row to themselves."""
        row = dict(_page_rows(layout))['image-panel']
        assert len(row.children) == 1
        assert TestLayout2D._find_component(row, 'dd-image-3d') is not None
        assert TestLayout2D._find_component(row, 'regression-plot') is None

    def test_detail_row_holds_the_region_plots(self, layout):
        row = dict(_page_rows(layout))['detail-panel']
        assert TestLayout2D._find_component(row,
                                            'region-checklist') is not None
        assert TestLayout2D._find_component(row,
                                            'regression-plot') is not None
        assert TestLayout2D._find_component(row, 'dd-image-3d') is None


class TestLayout3DMultiFeature:
    """3D layout with b > 1 should show a Feature dropdown."""

    @pytest.fixture(scope='class')
    def app_3d_multi(self):
        from glow.experiment.exper import Experiment, ExperimentImageOnly
        from glow.analysis import AnalysisGLOWSplit
        shape = (5, 5, 5)
        exp_img = ExperimentImageOnly.from_gauss(
            b=2, num_img=8, shape=shape, seed=99)
        x = np.arange(8, dtype=float).reshape(1, -1)
        exp = Experiment(x=x, contrast=np.array([True]),
                         y=exp_img.y, mask_idx=exp_img.mask_idx,
                         add_bias=True)
        ana = AnalysisGLOWSplit(n_perm_fwer=3).fit(exp)
        return _create_app(ana, exp, y_features=['feat_A', 'feat_B'])

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

    def test_mean_shape(self, exp):
        vol = compute_bg_volume(exp, feature_idx=0)
        assert vol.shape == exp.mask_idx.shape

    def test_single_image_shape(self, exp):
        vol = compute_bg_volume(exp, feature_idx=0, image_idx=0)
        assert vol.shape == exp.mask_idx.shape

    def test_single_image_differs_from_mean(self, exp):
        vol_mean = compute_bg_volume(exp, feature_idx=0)
        vol_0 = compute_bg_volume(exp, feature_idx=0, image_idx=0)
        assert not np.allclose(vol_mean, vol_0)

    def test_all_images_valid(self, exp):
        num_img = exp.y.shape[1]
        for i in range(num_img):
            vol = compute_bg_volume(exp, feature_idx=0, image_idx=i)
            assert np.isfinite(vol).any()
