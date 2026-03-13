"""Scatter plot of hierarchical segmentation regions with tree overlay.

Builds on the existing scatter_plotly() in glow/plot.py but designed for
interactive use in the Dash viewer: clickable points, swappable axes,
and threshold reference lines.
"""

import numpy as np
import plotly.graph_objects as go

from glow.graph import get_parent


def _ensure_1d(arr):
    """Return a 1-D view: if 2-D (b, num_reg), take first row."""
    return arr if arr.ndim == 1 else arr[0]


# columns where a log scale is the sensible default
_LOG_COLS = {'n_voxel', 'vox_in_target', 'vox_out_target'}
_ADJ_COL = 'llr_adjusted'

# estimate_state -> (plotly symbol, default color, legend label)
_STATE_STYLE = {
    'no_effect':   ('circle',  'steelblue', 'no effect'),
    'has_effect':  ('diamond', 'green',     'contains effect(s)'),
}
_STATE_ORDER = ['no_effect', 'has_effect']

# threshold lines drawn on pval axes:
#   column -> (analysis attribute name, line style)
_PVAL_THRESHOLD_MAP = {
    'pval_fwer': ('alpha_fwer', dict(color='red', dash='dot', width=1.5)),
}


def _compute_adj_thresh(ana_glow):
    """Compute the llr_adjusted value at the alpha_fwer significance boundary.

    Returns the minimum llr_adjusted among regions with pval <= alpha_fwer,
    i.e. the effective decision boundary on the adjusted-statistic axis.

    Returns None if alpha_fwer or the adjusted stat is not available,
    or if no regions are significant.
    """
    alpha = getattr(ana_glow, 'alpha_fwer', None)
    if alpha is None:
        return None

    pval = getattr(ana_glow, 'pval', None)
    adj = getattr(ana_glow, 'llr_adjusted_0', None)
    if pval is None or adj is None:
        return None
    if adj.ndim > 1:
        adj = adj[0]

    sig = ~np.isnan(pval) & (pval <= alpha)
    if not sig.any():
        return None

    return float(np.nanmin(adj[sig]))


def build_scatter(df, ana_glow, x_feat, y_feat, color_feat,
                  selected_reg=None, plot_tree=True,
                  log_y=False, target_stats=None):
    """Build an interactive Plotly scatter figure.

    Args:
        df (pd.DataFrame): region DataFrame (from viewer.data.prep_df)
        ana_glow (AnalysisGLOW): completed analysis (for tree + thresholds)
        x_feat (str): column name for x axis
        y_feat (str): column name for y axis
        color_feat (str): column name for color
        selected_reg (set): currently selected region indices (highlighted)
        plot_tree (bool): whether to draw hierarchy edges
        log_y (bool): apply log scale to y axis
        target_stats (dict|None): stats for the full target mask (from
            ``compute_target_stats``).  When both axes have finite values,
            a star marker is drawn at the target's position.

    Returns:
        fig (go.Figure): Plotly figure with clickable scatter
    """
    if selected_reg is None:
        selected_reg = set()

    num_vox = ana_glow.exp.y.shape[2]
    children = ana_glow.children
    parent = get_parent(children, num_vox)

    # sort by region_idx for consistent indexing
    _df = df.sort_values('region_idx').copy()
    x = _df[x_feat].values
    y = _df[y_feat].values
    no_color = (color_feat == '__none__')
    color = None if no_color else _df[color_feat].values
    states = _df['estimate_state'].values

    # in log mode, hide regions with non-positive y values
    if log_y:
        vis = np.isfinite(y) & (y > 0)
    else:
        vis = np.ones(len(y), dtype=bool)

    fig = go.Figure()

    # --- tree edges (only between visible endpoints) ---
    if plot_tree:
        tree_x, tree_y = [], []
        for idx, child in enumerate(children):
            par = idx + num_vox
            for c in child:
                if vis[c] and vis[par]:
                    tree_x += [x[c], x[par], None]
                    tree_y += [y[c], y[par], None]
        fig.add_trace(go.Scatter(
            x=tree_x, y=tree_y,
            mode='lines',
            line=dict(color='lightgrey', width=0.5),
            hoverinfo='skip',
            showlegend=False,
        ))

    # --- apply visibility mask to per-point arrays ---
    x_v = x[vis]
    y_v = y[vis]
    color_v = None if color is None else color[vis]
    states_v = states[vis]

    # --- per-point symbols from estimate_state ---
    symbols = np.array([_STATE_STYLE[s][0] for s in states_v])

    # --- build hover text ---
    hover_cols = ['region_idx', 'n_voxel', 'llr', 'llr_adjusted',
                  'pval_fwer']
    for c in ('pval_homo',
              'f1', 'sens', 'spec', 'vox_in_target',
              'vox_out_target', 'llr_mu_h0', 'llr_std_h0'):
        if c in _df.columns and not _df[c].isna().all():
            hover_cols.append(c)

    hover_text = []
    for _, row in _df[vis].iterrows():
        parts = []
        for c in hover_cols:
            v = row[c]
            if c == 'region_idx':
                parts.append(f'Region {int(v)}')
            elif isinstance(v, (int, np.integer)):
                parts.append(f'{c}: {v}')
            elif np.isnan(v):
                continue
            else:
                parts.append(f'{c}: {v:.4g}')
        state = row['estimate_state']
        if state != 'no_effect':
            label = _STATE_STYLE[state][2]
            parts.append(f'<b>{label}</b>')
        reg_idx = int(row['region_idx'])
        p = parent[reg_idx]
        parts.append(f'parent: {p}' if p != -1 else 'parent: none (root)')
        if reg_idx >= num_vox:
            c0, c1 = children[reg_idx - num_vox]
            parts.append(f'children: {c0}, {c1}')
        else:
            parts.append('children: none (leaf)')
        hover_text.append('<br>'.join(parts))

    # --- marker sizing: larger when selected; diamonds always get a border ---
    reg_indices = _df['region_idx'].values[vis]
    is_selected = np.isin(reg_indices, list(selected_reg))
    is_diamond = np.array([s == 'has_effect' for s in states_v])
    marker_size = np.where(is_selected, 14,
                           np.where(is_diamond, 10, 7))
    marker_line_width = np.where(is_selected, 2,
                                 np.where(is_diamond, 1.5, 0))
    marker_line_color = np.where(is_selected, 'black',
                                 np.where(is_diamond, 'black',
                                          'rgba(0,0,0,0)'))

    # --- main scatter ---
    if no_color:
        pt_colors = np.array([_STATE_STYLE[s][1] for s in states_v])
        marker_kwargs = dict(
            size=marker_size,
            symbol=symbols,
            color=pt_colors,
            showscale=False,
            line=dict(width=marker_line_width, color=marker_line_color),
        )
    else:
        marker_kwargs = dict(
            size=marker_size,
            symbol=symbols,
            color=color_v,
            colorscale='Viridis',
            colorbar=dict(title=color_feat, x=1.02, len=0.5, y=0.15,
                         yanchor='bottom'),
            showscale=True,
            line=dict(width=marker_line_width, color=marker_line_color),
        )

    fig.add_trace(go.Scatter(
        x=x_v, y=y_v,
        mode='markers',
        marker=marker_kwargs,
        customdata=reg_indices.tolist(),
        text=hover_text,
        hoverinfo='text',
        showlegend=False,
    ))

    # --- target mask star marker (on top of scatter for clickability) ---
    _add_target_star(fig, target_stats, x_feat, y_feat, log_y=log_y)

    # --- legend-only traces for each estimate state ---
    for state_key in _STATE_ORDER:
        symbol, default_color, label = _STATE_STYLE[state_key]
        legend_line = (dict(width=1.5, color='black')
                       if state_key == 'has_effect' else dict(width=0))
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, symbol=symbol, color=default_color,
                        line=legend_line),
            showlegend=True,
            name=label,
            legendgroup='estimate',
            legendgrouptitle_text='region estimations:',
        ))

    # --- axis scales ---
    # x: auto-log for size/count columns
    if x_feat in _LOG_COLS:
        fig.update_xaxes(type='log')
    # y: user-controlled toggle
    if log_y:
        fig.update_yaxes(type='log')

    # --- model overlay (llr on y vs n_voxel on x) ---
    _add_model_overlay(fig, ana_glow, x_feat, y_feat)

    # --- threshold reference lines ---
    _add_threshold_lines(fig, ana_glow, x_feat, y_feat)

    # --- layout ---
    fig.update_layout(
        xaxis_title=x_feat,
        yaxis_title=y_feat,
        height=500,
        margin=dict(l=60, r=80, t=30, b=50),
        legend=dict(x=1.02, y=1.0, xanchor='left', yanchor='top',
                    tracegroupgap=5),
        hoverlabel=dict(bgcolor='white'),
        plot_bgcolor='white',
    )
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')

    return fig


def _add_target_star(fig, target_stats, x_feat, y_feat, log_y=False):
    """Add an open-star outline at the full target mask's position.

    Clickable (customdata='target') so it behaves like any other region.
    Added before the main scatter so it renders behind region markers.
    Only drawn when both axis features have finite values.
    """
    if target_stats is None:
        return

    x_val = target_stats.get(x_feat)
    y_val = target_stats.get(y_feat)
    if x_val is None or y_val is None:
        return
    if not np.isfinite(x_val) or not np.isfinite(y_val):
        return
    if log_y and y_val <= 0:
        return

    hover_parts = ['<b>Target mask</b>']
    for k, v in target_stats.items():
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        if isinstance(v, (int, np.integer)):
            hover_parts.append(f'{k}: {v}')
        else:
            hover_parts.append(f'{k}: {v:.4g}')
    hover_text = '<br>'.join(hover_parts)

    fig.add_trace(go.Scatter(
        x=[x_val], y=[y_val],
        mode='markers',
        marker=dict(
            size=16,
            symbol='star-open',
            color='black',
            line=dict(width=2, color='black'),
        ),
        customdata=['target'],
        text=[hover_text],
        hoverinfo='text',
        showlegend=True,
        name='target mask',
        legendgroup='target',
    ))


def _add_model_overlay(fig, ana_glow, x_feat, y_feat):
    """Add size-regression model line when llr is on y-axis vs n_voxel.

    Uses the fitted model (sqrt or power_law) stored on the analysis object.
    """
    from glow.experiment.analysis import AnalysisGLOW

    adj_model = getattr(ana_glow, 'adj_model', None)
    adj_beta = getattr(ana_glow, 'adj_beta', None)
    if adj_model is None or adj_beta is None:
        return

    if y_feat != 'llr' or x_feat != 'n_voxel':
        return

    sizes = _ensure_1d(ana_glow.size).astype(float)
    sizes = sizes[sizes > 0]
    if len(sizes) == 0:
        return
    sz = np.linspace(max(sizes.min(), 1), sizes.max(), 200)
    mean_line = AnalysisGLOW.predict_null_mean(sz, adj_model, adj_beta)

    fig.add_trace(go.Scatter(
        x=sz, y=mean_line,
        mode='lines',
        line=dict(color='rgba(200,0,0,0.6)', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='skip',
    ))

    b = adj_beta
    if adj_model == 'sqrt':
        eq_text = f'E[stat|H0] = {b[0]:.4f} + {b[1]:.4f}·√size'
    else:
        eq_text = f'E[stat|H0] = exp({b[0]:.4f} + {b[1]:.4f}·ln(size))'
    fig.add_annotation(
        text=eq_text,
        xref='paper', yref='paper',
        x=0.02, y=0.02,
        showarrow=False,
        font=dict(size=11, color='rgba(200,0,0,0.8)',
                  family='monospace'),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='rgba(200,0,0,0.3)',
        borderwidth=1,
        borderpad=4,
    )


def _add_threshold_lines(fig, ana_glow, x_feat, y_feat):
    """Add dotted reference lines for thresholds when relevant."""
    alpha_fwer = getattr(ana_glow, 'alpha_fwer', None)

    # --- p-value axes ---
    for feat, (attr, style) in _PVAL_THRESHOLD_MAP.items():
        val = getattr(ana_glow, attr, None)
        if val is None:
            continue

        label = f'alpha_fwer={alpha_fwer}' if feat == 'pval_fwer' else f'{attr}={val}'

        if x_feat == feat:
            fig.add_vline(x=val, line=style,
                          annotation_text=label,
                          annotation_position='top')
        if y_feat == feat:
            fig.add_hline(y=val, line=style,
                          annotation_text=label,
                          annotation_position='right')

    # --- llr_adjusted axis: draw alpha_fwer line ---
    if alpha_fwer is not None and (
            'llr_adjusted' in (x_feat, y_feat)):
        adj_thresh = _compute_adj_thresh(ana_glow)
        if adj_thresh is not None:
            style = dict(color='red', dash='dot', width=1.5)
            label = f'alpha_fwer={alpha_fwer}'
            if x_feat == 'llr_adjusted':
                fig.add_vline(x=adj_thresh, line=style,
                              annotation_text=label,
                              annotation_position='top')
            if y_feat == 'llr_adjusted':
                fig.add_hline(y=adj_thresh, line=style,
                              annotation_text=label,
                              annotation_position='right')



