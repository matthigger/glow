"""Tests for viewer callback logic.

These test the callback functions directly (without running a Dash server)
by calling them with mock clickData / hoverData dicts.
"""

import copy
import json

import numpy as np
import pytest

from glow._extra.viewer.app import _create_app, _point_region
from glow._extra.viewer.data import prep_df, compute_target_stats
from glow._extra.viewer.scatter import build_scatter
from glow._extra.viewer.regression import build_regression_figure


class TestToggleRegionLogic:
    """Test the selection toggle logic used by _register_selection_callback.

    We replicate the callback body here since the inner function is not
    importable, but the logic is simple enough to test directly.
    """

    @staticmethod
    def _toggle(click_data, selected_json, is_target_click=False):
        """Replicate the core of toggle_region's click branch."""
        if click_data is None:
            return None, None

        point = click_data['points'][0]
        reg_idx = _point_region(point)
        if reg_idx is None:
            return None, None

        if reg_idx != 'target':
            reg_idx = int(reg_idx)

        selected = json.loads(selected_json)
        if reg_idx in selected:
            selected.remove(reg_idx)
        else:
            selected.append(reg_idx)
        return json.dumps(selected), reg_idx

    def test_add_region(self):
        click = {'points': [{'text': '42'}]}
        result, reg = self._toggle(click, '[]')
        assert json.loads(result) == [42]
        assert reg == 42

    def test_remove_region(self):
        click = {'points': [{'text': '42'}]}
        result, reg = self._toggle(click, '[42]')
        assert json.loads(result) == []
        assert reg == 42

    def test_add_target(self):
        click = {'points': [{'text': 'target'}]}
        result, reg = self._toggle(click, '[]')
        assert json.loads(result) == ['target']
        assert reg == 'target'

    def test_remove_target(self):
        click = {'points': [{'text': 'target'}]}
        result, reg = self._toggle(click, '["target"]')
        assert json.loads(result) == []

    def test_add_multiple(self):
        click1 = {'points': [{'text': '10'}]}
        click2 = {'points': [{'text': '20'}]}
        result, _ = self._toggle(click1, '[]')
        result, _ = self._toggle(click2, result)
        assert json.loads(result) == [10, 20]

    def test_target_with_existing_regions(self):
        click = {'points': [{'text': 'target'}]}
        result, _ = self._toggle(click, '[42, 100]')
        assert json.loads(result) == [42, 100, 'target']

    def test_none_text_ignored(self):
        click = {'points': [{}]}
        result, reg = self._toggle(click, '[42]')
        assert result is None

    def test_none_click_ignored(self):
        result, reg = self._toggle(None, '[42]')
        assert result is None


class TestTargetStarClickable:
    """Verify the target star names itself the way clicks expect."""

    def test_target_text_is_the_target_string(self, df_with_target, ana,
                                              exp, target_stats):
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr',
                            '__none__', target_stats=target_stats)
        star = [t for t in fig.data
                if getattr(t, 'text', None) is not None
                and 'target' in list(t.text)]
        assert len(star) == 1
        assert star[0].text[0] == 'target'
        assert _point_region({'text': star[0].text[0]}) == 'target'


class TestScatterCallbackIntegration:
    """End-to-end: build figure, verify it can be rebuilt with different axes.

    This catches the IndexError regression and y-feature responsiveness.
    """

    def test_rebuild_all_y_features(self, df_with_target, ana, exp,
                                    feature_cols, target_stats):
        """Changing Y feature must not crash (IndexError regression)."""
        generic, sig, prune, mask = feature_cols
        all_y = generic + sig + prune + mask
        for y_feat in all_y:
            fig = build_scatter(df_with_target, ana, exp, 'n_voxel', y_feat,
                                '__none__', target_stats=target_stats)
            assert fig.layout.yaxis.title.text == y_feat, \
                f'y-axis title should be {y_feat}'

    def test_rebuild_all_x_features(self, df_with_target, ana, exp,
                                    feature_cols, target_stats):
        for x_feat in sum(feature_cols, []):
            fig = build_scatter(df_with_target, ana, exp, x_feat, 'llr_z',
                                '__none__', target_stats=target_stats)
            assert fig.layout.xaxis.title.text == x_feat

    def test_selected_round_trip(self, df_with_target, ana, exp):
        """Select a region, rebuild figure, verify it's highlighted."""
        reg = int(df_with_target['region_idx'].iloc[5])
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr_z',
                            '__none__', selected_reg={reg})
        sizes, picked = [], None
        for t in fig.data:
            cd = getattr(t, 'customdata', None)
            if cd is None or np.ndim(cd) != 2:
                continue
            for i, s_reg in enumerate(t.text):
                sizes.append(t.marker.size[i])
                if int(s_reg) == reg:
                    picked = t.marker.size[i]
        assert picked == max(sizes), \
            'selected region should have the largest marker'


class TestRegressionClickToImage:
    """Clicking a regression data point should yield an image index."""

    def _build_reg_fig(self, ana, exp, df):
        num_vox = exp.y.shape[2]
        reg_idx = num_vox  # first internal node
        return build_regression_figure(
            ana_glow=ana, exp=exp, region_list=[reg_idx],
            x_feat_idx=0, y_feat_idx=0, df=df)

    def test_marker_traces_have_customdata(self, ana, exp, df_with_target):
        fig = self._build_reg_fig(ana, exp, df_with_target)
        marker_traces = [t for t in fig.data if t.mode == 'markers']
        assert len(marker_traces) >= 1
        for t in marker_traces:
            assert t.customdata is not None, 'regression markers need customdata'
            assert len(t.customdata) == len(t.x)

    def test_customdata_are_image_indices(self, ana, exp, df_with_target):
        """One region's traces partition the images, fold by fold."""
        fig = self._build_reg_fig(ana, exp, df_with_target)
        marker_traces = [t for t in fig.data if t.mode == 'markers']
        num_img = exp.y.shape[1]

        img_idx = [int(i) for t in marker_traces for i in t.customdata]
        assert sorted(img_idx) == list(range(num_img))

    @staticmethod
    def _simulate_click(img_idx):
        """Build a Plotly clickData dict for a regression marker."""
        return {'points': [{'customdata': img_idx, 'pointIndex': img_idx}]}

    def test_click_extracts_image_idx(self):
        """The callback logic: extract customdata and return str(img_idx)."""
        click = self._simulate_click(3)
        point = click['points'][0]
        img_idx = point.get('customdata')
        assert img_idx is not None
        assert str(int(img_idx)) == '3'

    def test_click_line_trace_no_customdata(self, ana, exp, df_with_target):
        """OLS fit-line traces should NOT have customdata (ignored on click)."""
        fig = self._build_reg_fig(ana, exp, df_with_target)
        line_traces = [t for t in fig.data if t.mode == 'lines']
        for t in line_traces:
            assert t.customdata is None

    def test_multiple_regions(self, ana, exp, df_with_target):
        """Every region's markers carry image-index customdata, per fold."""
        num_vox = exp.y.shape[2]
        regs = [num_vox, num_vox + 1]
        fig = build_regression_figure(
            ana_glow=ana, exp=exp, region_list=regs,
            x_feat_idx=0, y_feat_idx=0, df=df_with_target)
        marker_traces = [t for t in fig.data if t.mode == 'markers']
        assert len(marker_traces) == 2 * len(regs)

        for reg_idx in regs:
            img_idx = [int(i) for t in marker_traces
                       if t.legendgroup == f'reg-{reg_idx}'
                       for i in t.customdata]
            assert sorted(img_idx) == list(range(exp.y.shape[1]))


class TestRegressionFoldMarkers:
    """Marker shape names the fold: square segmentation, circle test."""

    @staticmethod
    def _marker_traces(ana, exp, df, x_feat_idx=0):
        """Build the panel on one region and return its marker traces."""
        fig = build_regression_figure(
            ana_glow=ana, exp=exp, region_list=[exp.y.shape[2]],
            x_feat_idx=x_feat_idx, y_feat_idx=0, df=df)
        return [t for t in fig.data if t.mode == 'markers']

    def test_one_trace_per_fold(self, ana, exp, df_with_target):
        traces = self._marker_traces(ana, exp, df_with_target)
        assert [t.marker.symbol for t in traces] == ['square', 'circle']

    def test_square_holds_the_tree_fold(self, ana, exp, df_with_target):
        """The square trace draws exactly the images the tree was built on."""
        square, circle = self._marker_traces(ana, exp, df_with_target)

        in_segment = np.asarray(ana.img_segment, dtype=bool)
        assert list(square.customdata) == list(np.flatnonzero(in_segment))
        assert list(circle.customdata) == list(np.flatnonzero(~in_segment))

    def test_fold_named_in_legend_and_hover(self, ana, exp, df_with_target):
        square, circle = self._marker_traces(ana, exp, df_with_target)

        assert square.name.endswith('segmentation fold')
        assert circle.name.endswith('test fold')
        assert all('segmentation fold' in t for t in square.text)
        assert all('test fold' in t for t in circle.text)

    def test_markers_follow_their_own_images(self, ana, exp, df_with_target):
        """Each point keeps the x of the image its customdata names."""
        for trace in self._marker_traces(ana, exp, df_with_target,
                                         x_feat_idx=1):
            img_idx = [int(i) for i in trace.customdata]
            assert np.array_equal(np.asarray(trace.x), exp.x[1][img_idx])

    def test_a_fit_without_the_fold_still_plots(self, ana, exp,
                                               df_with_target):
        """An older fit (no img_segment) falls back to one circle trace."""
        ana_old = copy.copy(ana)
        ana_old.img_segment = None

        traces = self._marker_traces(ana_old, exp, df_with_target)
        assert len(traces) == 1
        assert traces[0].marker.symbol == 'circle'
        assert list(traces[0].customdata) == list(range(exp.y.shape[1]))


class TestHoverIsAnsweredInTheBrowser:
    """Hover must not cost a server round trip.

    Every preview a hover triggers waits on store-hover and store-center,
    so a server callback in front of them puts the overlays, the
    regression and the slices a whole round trip behind the mouse.
    """

    @staticmethod
    def _callback_for(app, input_id, input_prop):
        """Return the callback spec fed by one input property."""
        for spec in app._callback_list:
            for inp in spec.get('inputs', []):
                if (inp.get('id') == input_id
                        and inp.get('property') == input_prop):
                    return spec
        return None

    def test_hover_callback_is_clientside(self, ana, exp, mask_target):
        app = _create_app(ana, exp, mask_target=mask_target)
        spec = self._callback_for(app, 'scatter-plot', 'hoverData')
        assert spec is not None, 'nothing listens to hoverData'
        assert spec['clientside_function'] is not None, \
            'hover went back to the server; previews now cost two hops'

    def test_the_preview_waits_for_the_mouse_to_settle(self, ana, exp,
                                                       mask_target):
        """hoverData must not reach store-hover directly.

        A reader crossing the cloud passes over hundreds of regions. If
        each one published, each would queue an overlay render, and the
        previews would arrive seconds after the mouse stopped.
        """
        app = _create_app(ana, exp, mask_target=mask_target)
        spec = self._callback_for(app, 'scatter-plot', 'hoverData')
        outs = str(spec['output'])
        assert 'store-hover' not in outs, \
            'every point the mouse crosses now queues a preview'

        timed = self._callback_for(app, 'hover-timer', 'n_intervals')
        assert timed is not None, 'nothing publishes the settled hover'
        assert 'store-hover' in str(timed['output'])
        assert timed['clientside_function'] is not None

    def test_centering_the_slicers_is_clientside(self, ana, exp,
                                                 mask_target):
        """store-center -> setpos must not add a hop before the slices."""
        app = _create_app(ana, exp, mask_target=mask_target)
        spec = self._callback_for(app, 'store-center', 'data')
        assert spec is not None, 'nothing listens to store-center'
        assert spec['clientside_function'] is not None, \
            'centring went back to the server; slices now cost three hops'


class TestHoverCenters:
    """The centre table the clientside hover reads."""

    def test_matches_compute_region_center(self, ana, exp):
        from glow._extra.viewer.image import (compute_region_center,
                                              region_centers)
        centers = region_centers(exp, ana)
        num_reg = exp.y.shape[2] + ana.children.shape[0]
        for reg in (0, num_reg // 3, num_reg - 1):
            ref = compute_region_center(reg, exp, ana)
            assert np.allclose(ref, centers[reg]), reg

    def test_every_drawn_region_has_a_centre(self, df_with_target, ana, exp,
                                             target_stats, mask_target):
        """A drawn region with no centre hovers without moving the view."""
        from glow._extra.viewer.app import _hover_centers

        for min_vox in (0, 2, 4):
            fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr',
                                '__none__', target_stats=target_stats,
                                min_vox=min_vox)
            centers = _hover_centers(df_with_target, ana, exp, min_vox,
                                     mask_target)
            drawn = {s for t in fig.data
                     if getattr(t, 'text', None) is not None for s in t.text}
            assert drawn <= set(centers), (min_vox, drawn - set(centers))
