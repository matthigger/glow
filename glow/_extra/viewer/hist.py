"""Per-region histogram of the FWER draws, for the glow viewer.

Where the scatter shows each region as one point -- its observed LLR and the
two moments that turned it into a z -- this shows the column those moments
came out of: whether a region's null is the bell the pair stands in for.

Only available when the analysis kept its draw matrix (a GLOW fit with
keep_stat=True); the viewer hides the panel otherwise, so nothing here
handles a missing .stat.

Regions overlay in the shared palette, so one region is the same colour
everywhere in the dashboard. Raw LLR is legible only across regions of a
similar size, LLR carrying a 0.5 * size prefactor, which is what the z unit
is for: it puts every region's draws on the scale the FWER comparison
happens on, where the nulls land on top of each other.

The observed draw (row 0) is in the histogram like any other, being
exchangeable with the permuted rows and contributing to mu and std with them
(Analysis.z_score_stat). It also gets its own dashed line, since where it
falls in its own null is the point of looking.
"""

import numpy as np
import plotly.graph_objects as go

from glow.analysis._base import Z_STD_FLOOR
from .image import get_region_color


# x-axis unit -> (axis title, legend/hover label)
UNIT_LABEL = {
    'llr': 'LLR',
    'z': 'z  (LLR - mu) / std',
}

# bin counts offered in the viewer's Bins dropdown
BIN_CHOICES = (20, 50, 100, 200)

DEFAULT_BINS = 50


def region_stat(ana_glow, reg_idx, unit='llr'):
    """Return one region's draws, NaNs dropped, in the requested unit.

    Args:
        ana_glow (AnalysisGLOWBase): analysis fit with keep_stat=True
        reg_idx (int): region index (a column of ana_glow.stat)
        unit (str): 'llr' for the raw draws, 'z' to standardize them by
            the region's own mu and std -- the same pair, and the same
            Z_STD_FLOOR guard on a degenerate column, that produced
            fwer.stat_obs.

    Returns:
        draw (np.array): (n_valid,) finite draws, empty when the region
            carries none (size < min_vox leaves the whole column NaN)
        obs (float): the observed draw (row 0) in the same unit, NaN when
            the region has none
    """
    col = np.asarray(ana_glow.stat[:, reg_idx], dtype=float)

    if unit == 'z':
        mu = float(ana_glow.mu[reg_idx])
        std = float(ana_glow.std[reg_idx])
        denom = std if std > Z_STD_FLOOR else 1.0
        col = (col - mu) / denom

    obs = col[0]
    return col[np.isfinite(col)], obs


def build_empty_hist(message=None):
    """Return an empty histogram figure carrying a prompt message."""
    if message is None:
        message = ('Select or hover over a region to see the draws its '
                   'z was measured against')
    fig = go.Figure()
    fig.update_layout(
        height=340,
        margin=dict(l=45, r=10, t=15, b=35),
        plot_bgcolor='white',
        annotations=[dict(
            text=message,
            xref='paper', yref='paper', x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=13, color='#999'),
        )],
    )
    return fig


def _common_bins(value_list, n_bins):
    """Return plotly xbins shared by every trace, or None if degenerate.

    Plotly bins each histogram trace on its own data by default, so two
    overlaid regions would be drawn on different bin edges and their bar
    heights would not be comparable. One set of edges over the pooled
    range fixes that.
    """
    pooled = np.concatenate([v for v in value_list if len(v)])
    lo, hi = float(pooled.min()), float(pooled.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return dict(start=lo, end=hi, size=(hi - lo) / max(int(n_bins), 1))


def build_hist(ana_glow, region_list, unit='llr', n_bins=DEFAULT_BINS,
               log_y=False, color_map=None, hover_reg=None, n_selected=0,
               df=None):
    """Build the overlaid per-region draw histogram.

    Args:
        ana_glow (AnalysisGLOWBase): analysis fit with keep_stat=True
        region_list (list[int]): region indices to overlay (visible +
            hover). 'target' entries are skipped: the target mask is not a
            tree region, so it has no column in the draw matrix -- the
            same reason data.compute_target_stats leaves its llr_z NaN.
        unit (str): 'llr' or 'z'; see region_stat
        n_bins (int): bins over the pooled range of every shown region
        log_y (bool): log the count axis, which is where the tail that
            sets the FWER threshold lives
        color_map (dict): reg_idx -> palette index, matching the image
            overlay
        hover_reg (int | None): region being hovered, drawn translucent
        n_selected (int): number of selected (non-hover) regions, so the
            hovered one takes the colour it would keep if clicked
        df (pd.DataFrame | None): region stats, for the hover box

    Returns:
        fig (go.Figure)
    """
    color_map = color_map or {}

    tree_regs = [r for r in region_list if r != 'target']
    if not tree_regs:
        return build_empty_hist()

    # gather first: the bin edges have to be shared, so nothing can be
    # drawn until every region's draws are in hand
    entry_list, empty_list = [], []
    for reg_idx in tree_regs:
        draw, obs = region_stat(ana_glow, reg_idx, unit=unit)
        if len(draw) == 0:
            empty_list.append(reg_idx)
            continue
        entry_list.append((reg_idx, draw, obs))

    if not entry_list:
        return build_empty_hist(
            f'No draws for region(s) {_fmt_reg_list(empty_list)}: below '
            f'min_vox={getattr(ana_glow, "min_vox", "?")}, so the whole '
            f'column is NaN and the region is out of the FWER family.')

    xbins = _common_bins([d for _, d, _ in entry_list], n_bins)

    fig = go.Figure()
    for reg_idx, draw, obs in entry_list:
        is_hover = (reg_idx == hover_reg and reg_idx not in color_map)
        cidx = color_map.get(reg_idx, n_selected if is_hover else 0)
        r, g, b = get_region_color(cidx)
        opacity = 0.35 if is_hover else 0.55

        size = int(ana_glow.size[reg_idx])
        name = f'Region {reg_idx} ({size} vox)'

        # the observed value rides in the legend rather than beside its
        # line: this panel is narrow, and regions whose observed draws sit
        # close together -- which in z is all of them, that being the
        # point of z -- wrote their labels on top of each other
        legend_name = name
        if np.isfinite(obs):
            legend_name = f'{name}<br>{_obs_label(reg_idx, obs, df)}'

        fig.add_trace(go.Histogram(
            x=draw,
            xbins=xbins,
            marker=dict(color=f'rgba({r},{g},{b},{opacity})',
                        line=dict(width=1, color=f'rgb({r},{g},{b})')),
            name=legend_name,
            legendgroup=f'reg-{reg_idx}',
            showlegend=True,
            hovertemplate=(f'<b>{name}</b><br>{UNIT_LABEL[unit]}: '
                           '%{x}<br>draws: %{y}<extra></extra>'),
        ))

        # the observed draw, marked where it falls in its own null
        if np.isfinite(obs):
            fig.add_vline(
                x=obs,
                line=dict(color=f'rgb({r},{g},{b})', dash='dash', width=2),
            )

    n_draw = ana_glow.stat.shape[0]
    fig.update_layout(
        barmode='overlay',
        xaxis_title=UNIT_LABEL[unit],
        yaxis_title=f'draws  (of {n_draw})',
        # the regression panel's geometry: the two sit in one column, and
        # an inside legend is what fits at this width
        height=340,
        margin=dict(l=45, r=10, t=15, b=35),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top',
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ddd', borderwidth=1,
                    tracegroupgap=2, font=dict(size=10)),
        hoverlabel=dict(bgcolor='white'),
        plot_bgcolor='white',
    )
    if log_y:
        fig.update_yaxes(type='log')
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')

    if empty_list:
        fig.add_annotation(
            text=f'no draws (below min_vox): {_fmt_reg_list(empty_list)}',
            xref='paper', yref='paper', x=0.5, y=1.02,
            showarrow=False, font=dict(size=10, color='#999'))

    return fig


def _fmt_reg_list(reg_list, max_show=4):
    """Render a region-index list for a message, truncated when long."""
    head = ', '.join(str(r) for r in reg_list[:max_show])
    if len(reg_list) > max_show:
        return f'{head}, ... (+{len(reg_list) - max_show})'
    return head


def _obs_label(reg_idx, obs, df):
    """Label the observed draw: the value, and its p when available."""
    label = f'obs {obs:.4g}'
    if df is None:
        return label
    row = df.loc[df['region_idx'] == reg_idx]
    if not len(row):
        return label
    pval = row.iloc[0].get('pval_fwer')
    if pval is None or not np.isfinite(pval):
        return label
    return f'{label}  p={pval:.3g}'
