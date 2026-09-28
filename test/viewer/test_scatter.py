"""Tests for glow._extra.viewer.scatter (figure building)."""

import numpy as np
import plotly.graph_objects as go

from glow._extra.viewer.scatter import build_scatter
from glow._extra.viewer.data import get_feature_columns


class TestBuildScatterBasic:
    """build_scatter returns a valid figure for all axis combinations."""

    def test_returns_figure(self, df_with_target, ana, exp):
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr',
                            '__none__')
        assert isinstance(fig, go.Figure)

    def test_all_axis_combinations_no_crash(self, df_with_target, ana, exp,
                                            feature_cols):
        generic, sig, prune, mask = feature_cols
        all_cols = generic + sig + prune + mask
        for x in all_cols:
            for y in all_cols:
                fig = build_scatter(df_with_target, ana, exp, x, y, '__none__')
                assert isinstance(fig, go.Figure), f'crash on x={x}, y={y}'

    def test_with_color_feature(self, df_with_target, ana, exp):
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr_z',
                            'pval_fwer')
        assert isinstance(fig, go.Figure)

    def test_log_y(self, df_with_target, ana, exp):
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr_z',
                            '__none__', log_y=True)
        assert isinstance(fig, go.Figure)

    def test_log_y_axis_range_reasonable(self, df_with_target, ana, exp,
                                         target_stats):
        """Log-y must set an explicit range derived from visible data,
        not Plotly's unbounded autorange (regression: axis went to 10^270)."""
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr_z',
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


def _region_traces(fig):
    """Marker traces carrying a customdata row per region."""
    return [t for t in fig.data
            if getattr(t, 'customdata', None) is not None
            and np.ndim(t.customdata) == 2]


def _star_traces(fig):
    """The target-mask star, which names itself 'target' in text."""
    return [t for t in fig.data
            if getattr(t, 'text', None) is not None
            and 'target' in list(t.text)]


def _marker_regions(fig):
    """Region indices the scatter draws, over all its marker traces."""
    return [int(s) for t in _region_traces(fig) for s in t.text]


class TestBuildScatterTraces:
    """Verify trace structure and content."""

    def _build(self, df, ana, exp, target_stats=None, **kw):
        return build_scatter(df, ana, exp, 'n_voxel', 'llr', '__none__',
                             target_stats=target_stats, **kw)

    def test_has_scatter_trace(self, df_with_target, ana, exp):
        fig = self._build(df_with_target, ana, exp)
        marker_traces = [t for t in fig.data if t.mode == 'markers']
        assert len(marker_traces) >= 1

    def test_target_star_present(self, df_with_target, ana, exp, target_stats):
        fig = self._build(df_with_target, ana, exp, target_stats=target_stats)
        assert len(_star_traces(fig)) == 1, 'target star missing'

    def test_target_star_on_top_of_scatter(self, df_with_target, ana, exp,
                                           target_stats):
        """Target star must render after the main scatter for clickability."""
        fig = self._build(df_with_target, ana, exp, target_stats=target_stats)
        region = _region_traces(fig)
        star = _star_traces(fig)
        assert region, 'no region marker trace found'
        assert len(star) == 1, 'target star trace not found'
        main_idx = max(list(fig.data).index(t) for t in region)
        star_idx = list(fig.data).index(star[0])
        assert star_idx > main_idx, \
            f'the star (trace {star_idx}) must come after the markers ' \
            f'(trace {main_idx}) to stay clickable'

    def test_target_star_absent_without_stats(self, df_with_target, ana, exp):
        fig = self._build(df_with_target, ana, exp, target_stats=None)
        assert len(_star_traces(fig)) == 0

    def test_tree_edges_present(self, df_with_target, ana, exp):
        fig = self._build(df_with_target, ana, exp, plot_tree=True)
        line_traces = [t for t in fig.data if t.mode == 'lines']
        assert len(line_traces) >= 1

    def test_tree_edges_absent(self, df_with_target, ana, exp):
        fig = self._build(df_with_target, ana, exp, plot_tree=False)
        line_traces = [t for t in fig.data
                       if t.mode == 'lines' and t.showlegend is False
                       and getattr(t.line, 'color', None) == 'lightgrey']
        assert len(line_traces) == 0

    def test_every_region_names_itself_in_text(self, df_with_target, ana,
                                                exp):
        """The click callback reads text, so every marker must carry it.

        customdata ships as float32 and never reaches the callback, so
        a region missing from text is a region that cannot be selected.
        """
        fig = self._build(df_with_target, ana, exp)
        for t in _region_traces(fig):
            assert len(t.text) == len(t.x)
        assert (sorted(_marker_regions(fig))
                == sorted(df_with_target['region_idx']))


class TestMinVoxLine:
    """Vertical line at ana.min_vox is drawn only on the H1 z-vs-size view."""

    @staticmethod
    def _vlines_at(fig, x):
        """Return shape entries that look like a vertical line at *x*."""
        return [s for s in fig.layout.shapes
                if s.type == 'line'
                and s.x0 == x and s.x1 == x]

    def test_drawn_when_y_is_llr_z_x_is_n_voxel(self, df_with_target, ana, exp):
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr_z',
                            '__none__')
        vlines = self._vlines_at(fig, ana.min_vox)
        assert len(vlines) == 1, (
            f'expected exactly one vertical line at x={ana.min_vox}, '
            f'got {len(vlines)}')

    def test_annotation_labels_min_vox(self, df_with_target, ana, exp):
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr_z',
                            '__none__')
        labels = [a.text for a in fig.layout.annotations]
        assert f'min_vox={int(ana.min_vox)}' in labels, (
            f'min_vox annotation missing; got {labels}')

    def test_absent_when_y_is_llr(self, df_with_target, ana, exp):
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr',
                            '__none__')
        assert len(self._vlines_at(fig, ana.min_vox)) == 0

    def test_absent_when_y_is_n_voxel(self, df_with_target, ana, exp):
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'n_voxel',
                            '__none__')
        assert len(self._vlines_at(fig, ana.min_vox)) == 0

    def test_absent_when_x_is_not_n_voxel(self, df_with_target, ana, exp):
        fig = build_scatter(df_with_target, ana, exp, 'llr', 'llr_z',
                            '__none__')
        assert len(self._vlines_at(fig, ana.min_vox)) == 0


class TestBuildScatterSelection:
    def test_selected_markers_larger(self, df_with_target, ana, exp):
        reg0 = int(df_with_target['region_idx'].iloc[0])
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr_z',
                            '__none__', selected_reg={reg0})
        sizes, picked = [], None
        for t in _region_traces(fig):
            for i, s_reg in enumerate(t.text):
                sizes.append(t.marker.size[i])
                if int(s_reg) == reg0:
                    picked = t.marker.size[i]
        assert picked is not None, 'the selected region was not drawn'
        assert picked > min(sizes)


class TestHoverText:
    """Verify hover text contains parent/children info."""

    def test_hover_has_parent_children(self, df_with_target, ana, exp):
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr_z',
                            '__none__')
        for t in _region_traces(fig):
            tpl = t.hovertemplate
            assert 'parent:' in tpl, f'missing parent in hover: {tpl}'
            assert 'children:' in tpl, f'missing children in hover: {tpl}'

    def test_leaf_hover_says_none(self, df_with_target, ana, exp):
        """Leaf nodes should show 'children: none (leaf)'."""
        num_vox = exp.y.shape[2]
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr_z',
                            '__none__')
        for t in _region_traces(fig):
            if any(int(s) < num_vox for s in t.text):
                assert 'none (leaf)' in t.hovertemplate
                break
        else:
            pytest.skip('no leaf nodes visible')

    def test_root_hover_says_none(self, df_with_target, ana, exp):
        """Root node should show 'parent: none (root)'."""
        from glow.graph import get_parent
        num_vox = exp.y.shape[2]
        parent = get_parent(ana.children, num_vox)
        roots = np.where(parent == -1)[0]

        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr_z',
                            '__none__')
        for t in _region_traces(fig):
            if any(int(s) in roots for s in t.text):
                assert 'none (root)' in t.hovertemplate
                break
        else:
            pytest.skip('root node not visible')


class TestModelOverlay:
    """Verify the model overlay doesn't crash (regression for IndexError)."""

    def test_llr_vs_n_voxel(self, df_with_target, ana, exp):
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr',
                            '__none__')
        assert isinstance(fig, go.Figure)


class TestPatchableSelection:
    """The restyle path must address the same points the figure drew."""

    def test_visible_regions_matches_the_built_traces(self, df_with_target,
                                                      ana, exp):
        """A click patches traces by index, never rebuilding the figure.

        If the split it assumes drifts from the split build_scatter made,
        a click silently outlines the wrong regions, so the two are
        pinned together here.
        """
        from glow._extra.viewer.scatter import visible_regions

        for log_y, min_vox in ((False, 0), (True, 0), (False, 3)):
            fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr',
                                '__none__', log_y=log_y, min_vox=min_vox)
            groups = visible_regions(df_with_target, ana, exp, 'llr',
                                     log_y=log_y, min_vox=min_vox)
            assert len(groups) == len(_region_traces(fig))
            for trace_idx, state, reg in groups:
                drawn = [int(s) for s in fig.data[trace_idx].text]
                assert drawn == list(reg), (log_y, min_vox, state)

    def test_no_per_point_hover_strings(self, df_with_target, ana, exp):
        """Hover text is a template per trace, not a string per region.

        text still carries one entry per point, but it is the bare
        region index the click callback reads -- not a rendered label.
        """
        fig = build_scatter(df_with_target, ana, exp, 'n_voxel', 'llr',
                            '__none__')
        traces = _region_traces(fig)
        assert traces
        for t in traces:
            assert t.hovertemplate
            assert t.hovertext is None
            assert all(s.isdigit() for s in t.text)

    def test_selection_style_marks_only_the_selected(self, df_with_target,
                                                     ana, exp):
        """selection_style is what both the build and the patch call."""
        from glow._extra.viewer.scatter import selection_style

        reg = np.array([3, 4, 5])
        size, width, color = selection_style(reg, {4}, 'no_effect')
        assert size[1] > size[0] and size[1] > size[2]
        assert width[1] > width[0]
        assert color[1] == 'black'
