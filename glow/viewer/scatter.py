"""Scatter plot of hierarchical segmentation regions with tree overlay.

Builds on the existing scatter_plotly() in glow/plot.py but designed for
interactive use in the Dash viewer: clickable points, swappable axes,
and threshold reference lines.
"""

import numpy as np
import plotly.graph_objects as go


# columns where a log scale is the sensible default
_LOG_COLS = {'n_voxel', 'hotel_tr', 'vox_in_target', 'vox_out_target'}

# estimate_state -> (plotly symbol, default color, legend label)
_STATE_STYLE = {
    'no_effect':    ('circle',        'steelblue', 'no effect'),
    'partial':      ('triangle-down', 'orange',    'partial effect (pruned)'),
    'full_effect':  ('diamond',       'green',     'full effect'),
    'multi_effect': ('triangle-up',   'red',       '> 1 effect (pruned)'),
}
# display order for legend entries
_STATE_ORDER = ['no_effect', 'partial', 'full_effect', 'multi_effect']

# threshold lines drawn on pval axes:
#   column -> (analysis attribute name, line style)
_PVAL_THRESHOLD_MAP = {
    'pval_fwer': ('alpha_fwer', dict(color='red', dash='dot', width=1.5)),
    'pval_homo': ('alpha_prune', dict(color='orange', dash='dot', width=1.5)),
}


def _compute_z_thresh(ana_glow):
    """Compute the z_stat value corresponding to alpha_fwer.

    This is the (1 - alpha_fwer) quantile of the per-permutation
    max-statistic distribution used by the Westfall-Young procedure.

    Returns None if alpha_fwer is not available.
    """
    alpha = getattr(ana_glow, 'alpha_fwer', None)
    if alpha is None:
        return None

    active = ana_glow.size[0, :] >= 1
    if not active.any():
        return None

    stat_max = np.sort(np.nanmax(ana_glow.z_stat[:, active], axis=1))
    idx = int((1 - alpha) * len(stat_max))
    idx = min(idx, len(stat_max) - 1)
    return float(stat_max[idx])


def build_scatter(df, ana_glow, x_feat, y_feat, color_feat,
                  selected_reg=None, plot_tree=True):
    """Build an interactive Plotly scatter figure.

    Args:
        df (pd.DataFrame): region DataFrame (from viewer.data.prep_df)
        ana_glow (AnalysisGLOW): completed analysis (for tree + thresholds)
        x_feat (str): column name for x axis
        y_feat (str): column name for y axis
        color_feat (str): column name for color
        selected_reg (set): currently selected region indices (highlighted)
        plot_tree (bool): whether to draw hierarchy edges

    Returns:
        fig (go.Figure): Plotly figure with clickable scatter
    """
    if selected_reg is None:
        selected_reg = set()

    num_vox = ana_glow.exp.y.shape[2]
    children = ana_glow.child_dict[0]

    # sort by region_idx for consistent indexing
    _df = df.sort_values('region_idx').copy()
    x = _df[x_feat].values
    y = _df[y_feat].values
    no_color = (color_feat == '__none__')
    color = None if no_color else _df[color_feat].values
    states = _df['estimate_state'].values

    fig = go.Figure()

    # --- tree edges ---
    if plot_tree:
        tree_x, tree_y = [], []
        for idx, child in enumerate(children):
            par = idx + num_vox
            for c in child:
                tree_x += [x[c], x[par], None]
                tree_y += [y[c], y[par], None]
        fig.add_trace(go.Scatter(
            x=tree_x, y=tree_y,
            mode='lines',
            line=dict(color='lightgrey', width=0.5),
            hoverinfo='skip',
            showlegend=False,
        ))

    # --- per-point symbols from estimate_state ---
    symbols = np.array([_STATE_STYLE[s][0] for s in states])

    # --- build hover text ---
    hover_cols = ['region_idx', 'n_voxel', 'z_stat', 'hotel_tr', 'pval_fwer']
    for c in ('pval_homo', 'f1', 'sens', 'spec', 'vox_in_target',
              'vox_out_target', 'hotel_tr_mu_h0', 'hotel_tr_std_h0'):
        if c in _df.columns and not _df[c].isna().all():
            hover_cols.append(c)

    hover_text = []
    for _, row in _df.iterrows():
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
        hover_text.append('<br>'.join(parts))

    # --- marker sizing: larger when selected ---
    reg_indices = _df['region_idx'].values
    is_selected = np.isin(reg_indices, list(selected_reg))
    marker_size = np.where(is_selected, 14, 7)
    marker_line_width = np.where(is_selected, 2, 0)
    marker_line_color = np.where(is_selected, 'black', 'rgba(0,0,0,0)')

    # --- main scatter ---
    if no_color:
        # each state gets its own colour
        pt_colors = np.array([_STATE_STYLE[s][1] for s in states])
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
            color=color,
            colorscale='Viridis',
            colorbar=dict(title=color_feat, x=1.02, len=0.5, y=0.15,
                         yanchor='bottom'),
            showscale=True,
            line=dict(width=marker_line_width, color=marker_line_color),
        )

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=marker_kwargs,
        customdata=reg_indices,
        text=hover_text,
        hoverinfo='text',
        showlegend=False,
    ))

    # --- legend-only traces for each estimate state ---
    for state_key in _STATE_ORDER:
        symbol, default_color, label = _STATE_STYLE[state_key]
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, symbol=symbol, color=default_color),
            showlegend=True,
            name=label,
            legendgroup='estimate',
            legendgrouptitle_text='region estimations:',
        ))

    # --- axis scales ---
    if x_feat in _LOG_COLS:
        fig.update_xaxes(type='log')
    if y_feat in _LOG_COLS:
        fig.update_yaxes(type='log')

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

    # --- z_stat axis: draw alpha_fwer line in z-space ---
    if alpha_fwer is not None and ('z_stat' in (x_feat, y_feat)):
        z_thresh = _compute_z_thresh(ana_glow)
        if z_thresh is not None:
            style = dict(color='red', dash='dot', width=1.5)
            label = f'alpha_fwer={alpha_fwer}'
            if x_feat == 'z_stat':
                fig.add_vline(x=z_thresh, line=style,
                              annotation_text=label,
                              annotation_position='top')
            if y_feat == 'z_stat':
                fig.add_hline(y=z_thresh, line=style,
                              annotation_text=label,
                              annotation_position='right')
