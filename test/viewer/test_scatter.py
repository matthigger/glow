"""Tests for glow.viewer.scatter (figure building)."""

import numpy as np
import plotly.graph_objects as go

from glow.viewer.scatter import build_scatter
from glow.viewer.data import get_feature_columns


class TestBuildScatterBasic:
    """build_scatter returns a valid figure for all axis combinations."""

    def test_returns_figure(self, df_with_target, ana):
        fig = build_scatter(df_with_target, ana, 'n_voxel', 'llr', '__none__')
        assert isinstance(fig, go.Figure)

    def test_all_axis_combinations_no_crash(self, df_with_target, ana,
                                            feature_cols):
        generic, sig, prune, mask = feature_cols
        all_cols = generic + sig + prune + mask
        for x in all_cols:
            for y in all_cols:
                fig = build_scatter(df_with_target, ana, x, y, '__none__')
                assert isinstance(fig, go.Figure), f'crash on x={x}, y={y}'

    def test_with_color_feature(self, df_with_target, ana):
        fig = build_scatter(df_with_target, ana, 'n_voxel', 'llr_z',
                            'pval_fwer')
        assert isinstance(fig, go.Figure)

    def test_log_y(self, df_with_target, ana):
        fig = build_scatter(df_with_target, ana, 'n_voxel', 'llr_z',
                            '__none__', log_y=True)
        assert isinstance(fig, go.Figure)

    def test_log_y_axis_range_reasonable(self, df_with_target, ana,
                                         target_stats):
        """Log-y must set an explicit range derived from visible data,
        not Plotly's unbounded autorange (regression: axis went to 10^270)."""
        fig = build_scatter(df_with_target, ana, 'n_voxel', 'llr_z',
                            '__none__', log_y=True, target_stats=target_stats)
        yaxis = fig.layout.yaxis
        assert yaxis.type == 'log'
        assert yaxis.range is not None, 'log-y should set an explicit range'

        log_lo, log_hi = yaxis.range
        pos_vals = df_with_target['llr_z'].values
        pos_vals = pos_vals[np.isfinite(pos_vals) & (pos_vals > 0)]
        data_lo = np.log10(pos_vals.min())
        data_hi = np.log10(pos_vals.max())

        assert log_lo <= data_lo, 'range should include the smallest point'
        assert log_hi >= data_hi, 'range should include the largest point'
        assert log_hi - log_lo < 1000, \
            f'range span {log_hi - log_lo} is unreasonably large'


class TestBuildScatterTraces:
    """Verify trace structure and content."""

    def _build(self, df, ana, target_stats=None, **kw):
        return build_scatter(df, ana, 'n_voxel', 'llr', '__none__',
                             target_stats=target_stats, **kw)

    def test_has_scatter_trace(self, df_with_target, ana):
        fig = self._build(df_with_target, ana)
        marker_traces = [t for t in fig.data if t.mode == 'markers']
        assert len(marker_traces) >= 1

    def test_target_star_present(self, df_with_target, ana, target_stats):
        fig = self._build(df_with_target, ana, target_stats=target_stats)
        star_traces = [t for t in fig.data
                       if getattr(t, 'customdata', None) is not None
                       and 'target' in list(t.customdata)]
        assert len(star_traces) == 1, 'target star trace missing'

    def test_target_star_on_top_of_scatter(self, df_with_target, ana,
                                           target_stats):
        """Target star must render after the main scatter for clickability."""
        fig = self._build(df_with_target, ana, target_stats=target_stats)
        main_idx = None
        star_idx = None
        for i, t in enumerate(fig.data):
            cd = getattr(t, 'customdata', None)
            if cd is not None and t.mode == 'markers' and 'target' not in list(cd):
                main_idx = i
            if cd is not None and 'target' in list(cd):
                star_idx = i
        assert main_idx is not None, 'main scatter trace not found'
        assert star_idx is not None, 'target star trace not found'
        assert star_idx > main_idx, \
            f'target star (trace {star_idx}) must be after main scatter (trace {main_idx})'

    def test_target_star_absent_without_stats(self, df_with_target, ana):
        fig = self._build(df_with_target, ana, target_stats=None)
        star_traces = [t for t in fig.data
                       if getattr(t, 'customdata', None) is not None
                       and 'target' in list(t.customdata)]
        assert len(star_traces) == 0

    def test_tree_edges_present(self, df_with_target, ana):
        fig = self._build(df_with_target, ana, plot_tree=True)
        line_traces = [t for t in fig.data if t.mode == 'lines']
        assert len(line_traces) >= 1

    def test_tree_edges_absent(self, df_with_target, ana):
        fig = self._build(df_with_target, ana, plot_tree=False)
        line_traces = [t for t in fig.data
                       if t.mode == 'lines' and t.showlegend is False
                       and getattr(t.line, 'color', None) == 'lightgrey']
        assert len(line_traces) == 0

    def test_customdata_ints(self, df_with_target, ana):
        fig = self._build(df_with_target, ana)
        main = [t for t in fig.data
                if t.mode == 'markers' and t.showlegend is False
                and getattr(t, 'customdata', None) is not None
                and 'target' not in list(t.customdata)]
        assert len(main) == 1
        for val in main[0].customdata:
            assert isinstance(val, (int, np.integer)), \
                f'customdata should be int, got {type(val)}'


class TestBuildScatterSelection:
    def test_selected_markers_larger(self, df_with_target, ana):
        reg0 = int(df_with_target['region_idx'].iloc[0])
        fig = build_scatter(df_with_target, ana, 'n_voxel', 'llr_z',
                            '__none__', selected_reg={reg0})
        main = [t for t in fig.data
                if t.mode == 'markers' and t.showlegend is False
                and getattr(t, 'customdata', None) is not None
                and 'target' not in list(t.customdata)][0]
        sizes = np.array(main.marker.size)
        idx_in_trace = list(main.customdata).index(reg0)
        assert sizes[idx_in_trace] > sizes.min()


class TestHoverText:
    """Verify hover text contains parent/children info."""

    def test_hover_has_parent_children(self, df_with_target, ana):
        fig = build_scatter(df_with_target, ana, 'n_voxel', 'llr_z',
                            '__none__')
        main = [t for t in fig.data
                if t.mode == 'markers' and t.showlegend is False
                and getattr(t, 'customdata', None) is not None
                and 'target' not in list(t.customdata)][0]
        for text in main.text:
            assert 'parent:' in text, f'missing parent in hover: {text}'
            assert 'children:' in text, f'missing children in hover: {text}'

    def test_leaf_hover_says_none(self, df_with_target, ana):
        """Leaf nodes should show 'children: none (leaf)'."""
        num_vox = ana.exp.y.shape[2]
        fig = build_scatter(df_with_target, ana, 'n_voxel', 'llr_z',
                            '__none__')
        main = [t for t in fig.data
                if t.mode == 'markers' and t.showlegend is False
                and getattr(t, 'customdata', None) is not None
                and 'target' not in list(t.customdata)][0]
        for i, reg_idx in enumerate(main.customdata):
            if reg_idx < num_vox:
                assert 'none (leaf)' in main.text[i]
                break
        else:
            pytest.skip('no leaf nodes visible')

    def test_root_hover_says_none(self, df_with_target, ana):
        """Root node should show 'parent: none (root)'."""
        from glow.graph import get_parent
        num_vox = ana.exp.y.shape[2]
        parent = get_parent(ana.children, num_vox)
        roots = np.where(parent == -1)[0]

        fig = build_scatter(df_with_target, ana, 'n_voxel', 'llr_z',
                            '__none__')
        main = [t for t in fig.data
                if t.mode == 'markers' and t.showlegend is False
                and getattr(t, 'customdata', None) is not None
                and 'target' not in list(t.customdata)][0]
        for i, reg_idx in enumerate(main.customdata):
            if reg_idx in roots:
                assert 'none (root)' in main.text[i]
                break
        else:
            pytest.skip('root node not visible')


class TestModelOverlay:
    """Verify the model overlay doesn't crash (regression for IndexError)."""

    def test_llr_vs_n_voxel(self, df_with_target, ana):
        fig = build_scatter(df_with_target, ana, 'n_voxel', 'llr', '__none__')
        assert isinstance(fig, go.Figure)
