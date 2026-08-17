"""Tests for the per-region draw histogram and the mode switch it shares.

The figure builder is called directly (no Dash server), as the other
viewer tests do; the wiring is checked through the app's callback map and
its layout.

Run:
    ~/venv_glow/bin/pytest test/viewer/test_hist.py -v
"""

import json

import numpy as np
import pytest

from glow._extra.viewer.app import _create_app, _has_stat
from glow._extra.viewer.hist import (build_hist, build_empty_hist,
                                     region_stat, _common_bins, UNIT_LABEL)
from glow._extra.viewer.image import get_region_color
from glow._extra.viewer.data import prep_df

from .test_layout import TestLayout2D


_find = TestLayout2D._find_component


def _big_regions(ana, n=2):
    """Return the n largest regions -- the ones sure to carry draws."""
    return [int(i) for i in np.argsort(ana.size)[::-1][:n]]


def _hist_traces(fig):
    """Return the histogram traces of a figure."""
    return [t for t in fig.data if t.type == 'histogram']


class TestRegionDraws:
    """region_stat pulls one column, in the requested unit."""

    def test_llr_column_is_the_matrix_column(self, ana_stat):
        reg = _big_regions(ana_stat, 1)[0]
        draw, obs = region_stat(ana_stat, reg)
        col = ana_stat.stat[:, reg]
        np.testing.assert_allclose(draw, col[np.isfinite(col)])
        assert obs == col[0]

    def test_observed_is_row_zero_and_is_llr(self, ana_stat):
        reg = _big_regions(ana_stat, 1)[0]
        _, obs = region_stat(ana_stat, reg)
        assert obs == pytest.approx(ana_stat.llr[reg])

    def test_z_unit_uses_the_analysis_moments(self, ana_stat):
        """The z the viewer plots is the z the FWER test compared."""
        reg = _big_regions(ana_stat, 1)[0]
        _, obs_z = region_stat(ana_stat, reg, unit='z')
        assert obs_z == pytest.approx(ana_stat.fwer.stat_obs[reg])

    def test_z_unit_is_centred(self, ana_stat):
        """Standardizing by the column's own moments zeroes its mean."""
        reg = _big_regions(ana_stat, 1)[0]
        draw_z, _ = region_stat(ana_stat, reg, unit='z')
        assert draw_z.mean() == pytest.approx(0.0, abs=1e-9)

    def test_no_nan_survives(self, ana_stat):
        num_reg = len(ana_stat.size)
        for reg in range(num_reg):
            draw, _ = region_stat(ana_stat, reg)
            assert np.isfinite(draw).all()


class TestBuildHist:
    """The overlaid figure itself."""

    def test_one_trace_per_region(self, ana_stat):
        regs = _big_regions(ana_stat, 3)
        fig = build_hist(ana_stat, regs)
        assert len(_hist_traces(fig)) == 3

    def test_overlay_mode(self, ana_stat):
        """Overlaid, not stacked -- the regions have to sit on each other."""
        fig = build_hist(ana_stat, _big_regions(ana_stat, 2))
        assert fig.layout.barmode == 'overlay'

    def test_bins_are_shared(self, ana_stat):
        """One set of edges, or two regions' bar heights don't compare."""
        fig = build_hist(ana_stat, _big_regions(ana_stat, 3))
        xbins = {(t.xbins.start, t.xbins.end, t.xbins.size)
                 for t in _hist_traces(fig)}
        assert len(xbins) == 1

    def test_bin_count_is_honoured(self, ana_stat):
        regs = _big_regions(ana_stat, 2)
        wide = build_hist(ana_stat, regs, n_bins=20)
        fine = build_hist(ana_stat, regs, n_bins=200)
        assert _hist_traces(fine)[0].xbins.size < \
            _hist_traces(wide)[0].xbins.size

    def test_trace_data_is_the_column(self, ana_stat):
        reg = _big_regions(ana_stat, 1)[0]
        fig = build_hist(ana_stat, [reg])
        col = ana_stat.stat[:, reg]
        np.testing.assert_allclose(_hist_traces(fig)[0].x,
                                   col[np.isfinite(col)])

    def test_colors_match_the_shared_palette(self, ana_stat):
        """A region is the same colour here as in the image overlay."""
        regs = _big_regions(ana_stat, 2)
        color_map = {r: i for i, r in enumerate(regs)}
        fig = build_hist(ana_stat, regs, color_map=color_map)
        for trace, reg in zip(_hist_traces(fig), regs):
            r, g, b = get_region_color(color_map[reg])
            assert trace.marker.line.color == f'rgb({r},{g},{b})'

    def test_observed_line_per_region(self, ana_stat):
        """Each region marks where its observed draw fell in its null."""
        regs = _big_regions(ana_stat, 2)
        fig = build_hist(ana_stat, regs)
        vlines = [s for s in fig.layout.shapes if s.type == 'line']
        assert len(vlines) == 2
        want = sorted(float(ana_stat.llr[r]) for r in regs)
        assert sorted(float(s.x0) for s in vlines) == pytest.approx(want)

    def test_observed_line_follows_the_unit(self, ana_stat):
        reg = _big_regions(ana_stat, 1)[0]
        fig = build_hist(ana_stat, [reg], unit='z')
        line = [s for s in fig.layout.shapes if s.type == 'line'][0]
        assert float(line.x0) == pytest.approx(
            ana_stat.fwer.stat_obs[reg])

    def test_log_y_sets_the_count_axis(self, ana_stat):
        fig = build_hist(ana_stat, _big_regions(ana_stat, 1), log_y=True)
        assert fig.layout.yaxis.type == 'log'

    def test_hover_reg_is_translucent(self, ana_stat):
        """A hovered-not-selected region is drawn faded, as elsewhere."""
        regs = _big_regions(ana_stat, 2)
        fig = build_hist(ana_stat, regs, color_map={regs[0]: 0},
                         hover_reg=regs[1], n_selected=1)
        opacity = [float(t.marker.color.split(',')[-1].rstrip(')'))
                   for t in _hist_traces(fig)]
        assert opacity[1] < opacity[0]

    def test_empty_selection_prompts(self, ana_stat):
        fig = build_hist(ana_stat, [])
        assert len(_hist_traces(fig)) == 0
        assert len(fig.layout.annotations) == 1

    def test_target_is_skipped(self, ana_stat):
        """The target mask is no tree region, so it has no column."""
        reg = _big_regions(ana_stat, 1)[0]
        fig = build_hist(ana_stat, ['target', reg])
        assert len(_hist_traces(fig)) == 1

    def test_target_alone_prompts(self, ana_stat):
        fig = build_hist(ana_stat, ['target'])
        assert len(_hist_traces(fig)) == 0

    def test_legend_carries_the_observed_value(self, ana_stat):
        """It rides in the legend, not beside the line -- narrow panel."""
        reg = _big_regions(ana_stat, 1)[0]
        fig = build_hist(ana_stat, [reg])
        assert 'obs' in _hist_traces(fig)[0].name

    def test_df_puts_the_pval_in_the_legend(self, ana_stat, exp):
        reg = _big_regions(ana_stat, 1)[0]
        df = prep_df(ana_stat, exp)
        fig = build_hist(ana_stat, [reg], df=df)
        assert 'p=' in _hist_traces(fig)[0].name

    def test_no_annotations_to_collide(self, ana_stat):
        fig = build_hist(ana_stat, _big_regions(ana_stat, 3))
        assert len(fig.layout.annotations) == 0

    def test_empty_hist_carries_a_message(self):
        fig = build_empty_hist()
        assert len(fig.data) == 0
        assert fig.layout.annotations[0].text


class TestCommonBins:
    """One set of edges over the pooled range, or none at all."""

    def test_edges_span_every_region(self):
        """The pooled range, so no region's draws fall off the axis."""
        low = np.array([0.0, 1.0, 2.0])
        high = np.array([8.0, 9.0, 10.0])
        xbins = _common_bins([low, high], 4)
        assert xbins['start'] == 0.0 and xbins['end'] == 10.0
        assert xbins['size'] == pytest.approx(2.5)

    def test_empty_regions_are_skipped(self):
        """A gated region contributes no values to the pooled range."""
        xbins = _common_bins([np.array([]), np.array([1.0, 3.0])], 2)
        assert xbins['start'] == 1.0 and xbins['end'] == 3.0

    def test_a_constant_pool_is_degenerate(self):
        """hi == lo would be a zero-width bin; plotly picks its own."""
        assert _common_bins([np.full(5, 2.0)], 10) is None

    def test_zero_bins_does_not_divide_by_zero(self):
        assert _common_bins([np.array([0.0, 1.0])], 0)['size'] > 0

    def test_a_constant_column_still_draws(self, ana_stat):
        """The degenerate branch reaches plotly, it does not raise."""
        reg = _big_regions(ana_stat, 1)[0]
        flat = _Shim(ana_stat, reg)
        fig = build_hist(flat, [reg])
        assert len(_hist_traces(fig)) == 1
        assert _hist_traces(fig)[0].xbins.size is None


class _Shim:
    """An analysis whose one region drew the same LLR every time.

    Cheaper than fitting for a case the data will not produce: a column
    of identical draws, which is what sends _common_bins down its
    degenerate branch.
    """

    def __init__(self, ana, reg_idx):
        self.stat = np.array(ana.stat, dtype=float)
        self.stat[:, reg_idx] = 2.0
        self.mu, self.std, self.size = ana.mu, ana.std, ana.size
        self.min_vox = ana.min_vox


class TestMinVoxRegions:
    """A region below min_vox is NaN in every row -- it has no null."""

    @pytest.fixture(scope='class')
    def ana_gated(self, exp):
        from glow.analysis import AnalysisGLOW
        return AnalysisGLOW(n_perm_fwer=5, min_vox=4,
                            keep_stat=True).fit(exp)

    def test_gated_region_has_no_draws(self, ana_gated):
        small = int(np.flatnonzero(ana_gated.size < 4)[0])
        draw, _ = region_stat(ana_gated, small)
        assert len(draw) == 0

    def test_gated_region_alone_explains_itself(self, ana_gated):
        small = int(np.flatnonzero(ana_gated.size < 4)[0])
        fig = build_hist(ana_gated, [small])
        assert len(_hist_traces(fig)) == 0
        assert 'min_vox' in fig.layout.annotations[0].text

    def test_gated_region_does_not_sink_the_others(self, ana_gated):
        """One empty region among drawable ones is noted, not fatal."""
        small = int(np.flatnonzero(ana_gated.size < 4)[0])
        big = _big_regions(ana_gated, 1)[0]
        fig = build_hist(ana_gated, [small, big])
        assert len(_hist_traces(fig)) == 1
        assert 'no draws' in fig.layout.annotations[-1].text


class TestHasDraws:
    """_has_stat decides whether the panel exists at all."""

    def test_true_when_kept(self, ana_stat):
        assert _has_stat(ana_stat)

    def test_false_by_default(self, ana):
        assert not _has_stat(ana)

    def test_false_for_a_pre_keep_stat_bundle(self, ana):
        """A bundle pickled before keep_stat existed has no attribute."""
        class _Old:
            pass
        assert not _has_stat(_Old())


class TestLayoutWithDraws:
    """The histogram panel appears only for a draw-carrying analysis."""

    @pytest.fixture(scope='class')
    def layout(self, ana_stat_2d, exp_2d, mask_target_2d):
        app = _create_app(ana_stat_2d, exp_2d, mask_target=mask_target_2d)
        return app.layout

    @pytest.fixture(scope='class')
    def layout_plain(self, ana_2d, exp_2d, mask_target_2d):
        app = _create_app(ana_2d, exp_2d, mask_target=mask_target_2d)
        return app.layout

    @pytest.mark.parametrize('comp_id', ['hist-plot', 'hist-unit',
                                         'hist-bins', 'hist-log-y'])
    def test_component_present(self, layout, comp_id):
        assert _find(layout, comp_id) is not None

    @pytest.mark.parametrize('comp_id', ['hist-plot', 'hist-unit',
                                         'hist-bins', 'hist-log-y'])
    def test_absent_without_draws(self, layout_plain, comp_id):
        assert _find(layout_plain, comp_id) is None

    def test_scatter_survives_either_way(self, layout, layout_plain):
        """The panel sits beside the scatter; it does not replace it."""
        assert _find(layout, 'scatter-plot') is not None
        assert _find(layout_plain, 'scatter-plot') is not None

    def test_panel_ends_the_detail_row(self, layout):
        """Last column of the detail row -- right of REGRESSION."""
        row = _find(layout, 'detail-panel').children
        assert len(row) == 3
        assert TestLayout2D._find_component(row[1],
                                            'regression-plot') is not None
        assert TestLayout2D._find_component(row[-1], 'hist-plot') is not None

    def test_detail_row_loses_the_column_without_draws(self, layout_plain):
        """REGRESSION takes the width back when there are no draws."""
        assert len(_find(layout_plain, 'detail-panel').children) == 2

    def test_panel_shares_the_row_with_regression(self, layout):
        """Side by side in one row, so the two split its width evenly."""
        row = _find(layout, 'detail-panel').children
        assert row[-1].style['flex'] == '1'
        assert row[1].style['flex'] == '1'

    def test_unit_defaults_to_llr(self, layout):
        assert _find(layout, 'hist-unit').value == 'llr'


class TestCallbacksRegistered:
    """The wiring is registered with a kept stat and skipped without."""

    @staticmethod
    def _outputs(app):
        return ' '.join(app.callback_map.keys())

    def test_hist_callbacks_registered(self, ana_stat_2d, exp_2d):
        app = _create_app(ana_stat_2d, exp_2d)
        assert 'hist-plot.figure' in self._outputs(app)

    def test_no_hist_callbacks_without_stat(self, ana_2d, exp_2d):
        app = _create_app(ana_2d, exp_2d)
        assert 'hist-plot' not in self._outputs(app)


class TestHistCallback:
    """The callback body, driven the way the browser drives it.

    The classes above call build_hist directly; this one goes through
    app.callback_map, so Dash's own argument handling and JSON response
    are in the path. What is under test is the wiring build_hist never
    sees: the hover merge, the stale-index guard, and the three controls.
    """

    @pytest.fixture(scope='class')
    def app(self, ana_stat_2d, exp_2d, mask_target_2d):
        return _create_app(ana_stat_2d, exp_2d, mask_target=mask_target_2d)

    @staticmethod
    def _fire(app, visible, hover=None, unit='llr', n_bins=50, log_y=(),
              selected=None):
        """Dispatch the callback and return the figure, as a dict."""
        selected = list(visible) if selected is None else list(selected)
        raw = app.callback_map['hist-plot.figure']['callback']
        resp = raw(list(visible),
                   'null' if hover is None else json.dumps(hover),
                   unit, n_bins, list(log_y), json.dumps(selected),
                   outputs_list={'id': 'hist-plot', 'property': 'figure'})
        return json.loads(resp)['response']['hist-plot']['figure']

    @staticmethod
    def _traces(fig):
        return [t for t in fig['data'] if t['type'] == 'histogram']

    def test_checked_regions_become_traces(self, app, ana_stat_2d):
        regs = _big_regions(ana_stat_2d, 2)
        assert len(self._traces(self._fire(app, regs))) == 2

    def test_hover_adds_an_unchecked_region(self, app, ana_stat_2d):
        """Hovering previews a region without clicking it in."""
        first, second = _big_regions(ana_stat_2d, 2)
        fig = self._fire(app, [first], hover=second, selected=[first])
        assert len(self._traces(fig)) == 2

    def test_hover_on_a_checked_region_does_not_double_it(self, app,
                                                          ana_stat_2d):
        reg = _big_regions(ana_stat_2d, 1)[0]
        fig = self._fire(app, [reg], hover=reg)
        assert len(self._traces(fig)) == 1

    def test_unit_dropdown_reaches_the_axis(self, app, ana_stat_2d):
        regs = _big_regions(ana_stat_2d, 1)
        assert self._fire(app, regs, unit='llr')['layout'][
            'xaxis']['title']['text'] == UNIT_LABEL['llr']
        assert self._fire(app, regs, unit='z')['layout'][
            'xaxis']['title']['text'] == UNIT_LABEL['z']

    def test_unit_dropdown_moves_the_observed_line(self, app, ana_stat_2d):
        """z is not a relabelling of llr: the marked draw moves with it."""
        reg = _big_regions(ana_stat_2d, 1)[0]
        x_llr = self._fire(app, [reg])['layout']['shapes'][0]['x0']
        x_z = self._fire(app, [reg], unit='z')['layout']['shapes'][0]['x0']
        assert x_llr == pytest.approx(ana_stat_2d.llr[reg])
        assert x_z == pytest.approx(ana_stat_2d.fwer.stat_obs[reg])

    def test_bins_dropdown_reaches_the_edges(self, app, ana_stat_2d):
        regs = _big_regions(ana_stat_2d, 2)
        wide = self._fire(app, regs, n_bins=20)
        fine = self._fire(app, regs, n_bins=200)
        assert (self._traces(fine)[0]['xbins']['size'] <
                self._traces(wide)[0]['xbins']['size'])

    def test_log_checkbox_reaches_the_count_axis(self, app, ana_stat_2d):
        """The checklist hands over a list, not a bool."""
        regs = _big_regions(ana_stat_2d, 1)
        assert 'type' not in self._fire(app, regs)['layout']['yaxis']
        assert self._fire(app, regs, log_y=('on',))[
            'layout']['yaxis']['type'] == 'log'

    def test_nothing_checked_is_the_prompt(self, app):
        fig = self._fire(app, [])
        assert len(self._traces(fig)) == 0
        assert fig['layout']['annotations'][0]['text']

    def test_a_stale_index_is_dropped_not_raised(self, app):
        """A browser session can hand back an index from another fit."""
        fig = self._fire(app, [10 ** 9])
        assert len(self._traces(fig)) == 0

    def test_target_alone_is_the_prompt(self, app):
        """The target mask is no tree region, so it has no column."""
        fig = self._fire(app, ['target'])
        assert len(self._traces(fig)) == 0

    def test_target_beside_a_region_leaves_the_region(self, app,
                                                      ana_stat_2d):
        reg = _big_regions(ana_stat_2d, 1)[0]
        fig = self._fire(app, ['target', reg])
        assert len(self._traces(fig)) == 1
