"""Tests for viewer callback logic.

These test the callback functions directly (without running a Dash server)
by calling them with mock clickData / hoverData dicts.
"""

import json

import numpy as np
import pytest

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
        reg_idx = point.get('customdata')
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
        click = {'points': [{'customdata': 42}]}
        result, reg = self._toggle(click, '[]')
        assert json.loads(result) == [42]
        assert reg == 42

    def test_remove_region(self):
        click = {'points': [{'customdata': 42}]}
        result, reg = self._toggle(click, '[42]')
        assert json.loads(result) == []
        assert reg == 42

    def test_add_target(self):
        click = {'points': [{'customdata': 'target'}]}
        result, reg = self._toggle(click, '[]')
        assert json.loads(result) == ['target']
        assert reg == 'target'

    def test_remove_target(self):
        click = {'points': [{'customdata': 'target'}]}
        result, reg = self._toggle(click, '["target"]')
        assert json.loads(result) == []

    def test_add_multiple(self):
        click1 = {'points': [{'customdata': 10}]}
        click2 = {'points': [{'customdata': 20}]}
        result, _ = self._toggle(click1, '[]')
        result, _ = self._toggle(click2, result)
        assert json.loads(result) == [10, 20]

    def test_target_with_existing_regions(self):
        click = {'points': [{'customdata': 'target'}]}
        result, _ = self._toggle(click, '[42, 100]')
        assert json.loads(result) == [42, 100, 'target']

    def test_none_customdata_ignored(self):
        click = {'points': [{}]}
        result, reg = self._toggle(click, '[42]')
        assert result is None

    def test_none_click_ignored(self):
        result, reg = self._toggle(None, '[42]')
        assert result is None


class TestTargetStarClickable:
    """Verify the target star trace has the right customdata for clicks."""

    def test_target_customdata_is_string(self, df_with_target, ana, exp,
                                         target_stats):
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr',
                            '__none__', target_stats=target_stats)
        star = [t for t in fig.data
                if getattr(t, 'customdata', None) is not None
                and 'target' in list(t.customdata)]
        assert len(star) == 1
        assert star[0].customdata[0] == 'target'


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
        main = [t for t in fig.data
                if t.mode == 'markers' and t.showlegend is False
                and getattr(t, 'customdata', None) is not None
                and 'target' not in list(t.customdata)][0]
        idx = list(main.customdata).index(reg)
        sizes = np.array(main.marker.size)
        assert sizes[idx] == sizes.max(), \
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
            assert len(t.customdata) == exp.y.shape[1]

    def test_customdata_are_image_indices(self, ana, exp, df_with_target):
        fig = self._build_reg_fig(ana, exp, df_with_target)
        marker_traces = [t for t in fig.data if t.mode == 'markers']
        num_img = exp.y.shape[1]
        for t in marker_traces:
            assert list(t.customdata) == list(range(num_img))

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
        """Each region's marker trace should carry image-index customdata."""
        num_vox = exp.y.shape[2]
        regs = [num_vox, num_vox + 1]
        fig = build_regression_figure(
            ana_glow=ana, exp=exp, region_list=regs,
            x_feat_idx=0, y_feat_idx=0, df=df_with_target)
        marker_traces = [t for t in fig.data if t.mode == 'markers']
        assert len(marker_traces) == 2
        for t in marker_traces:
            assert list(t.customdata) == list(range(exp.y.shape[1]))
