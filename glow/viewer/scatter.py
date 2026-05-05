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

    When significant regions exist, returns the minimum llr_adjusted among
    them (the empirical decision boundary).  Otherwise falls back to
    ``adj_crit`` — the exact critical value from the permutation null
    distribution (stored during analysis).

    Returns None only when neither source is available.
    """
    alpha = getattr(ana_glow, 'alpha_fwer', None)
    if alpha is None:
        return None

    pval = getattr(ana_glow, 'pval', None)
    adj = getattr(ana_glow, 'llr_adjusted_0', None)
    if pval is None or adj is None:
        return getattr(ana_glow, 'adj_crit', None)
    if adj.ndim > 1:
        adj = adj[0]

    sig = ~np.isnan(pval) & (pval <= alpha)
    if sig.any():
        return float(np.nanmin(adj[sig]))

    return getattr(ana_glow, 'adj_crit', None)


def build_scatter_h0(df_h0, ana_glow, x_feat, y_feat, color_feat,
                     log_y=False):
    """Plotly scatter for the H0 (Permuted Samples) view.

    Sister to ``build_scatter`` for the case where points come from
    held-out fit permutations under the null (via ``prep_df_h0``).  No
    tree edges, no per-region selection, no target stars: just the
    cloud, the GAM mu / sigma overlay, and the FWER threshold reference.

    Args:
        df_h0 (pd.DataFrame): from glow.viewer.data.prep_df_h0
        ana_glow (AnalysisGLOW): for mu_gam / sigma_gam / adj_crit
        x_feat (str): one of H0_FEATURES (typically 'n_voxel')
        y_feat (str): one of H0_FEATURES (typically 'llr',
            'llr_adjusted', or 'z_score')
        color_feat (str): one of H0_FEATURES + '__none__'
        log_y (bool): apply log scale to y (only sensible for raw 'llr')
    """
    if df_h0 is None or len(df_h0) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text='no fit-permutation cloud retained on this analysis<br>'
                 '(rerun with keep_fit_data=True)',
            xref='paper', yref='paper', x=0.5, y=0.5,
            showarrow=False, font=dict(size=12, color='#666'))
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                          height=420)
        return fig

    x = df_h0[x_feat].values
    y = df_h0[y_feat].values
    no_color = (color_feat in (None, '__none__'))
    color = None if no_color else df_h0[color_feat].values

    if log_y:
        vis = np.isfinite(y) & (y > 0)
    else:
        vis = np.isfinite(y)
    x_v, y_v = x[vis], y[vis]
    color_v = None if color is None else color[vis]

    fig = go.Figure()
    # Use WebGL (Scattergl) for the H0 scatter -- the cloud can be up
    # to a few hundred thousand points when keep_fit_data='all' is
    # used for diagnostic deep-dives.  Plain go.Scatter is slow above
    # ~10k points; Scattergl handles 1M+.
    marker_kw = dict(size=4, opacity=0.55,
                     line=dict(width=0))
    is_categorical = color_feat == 'perm_idx'
    if color_v is not None and is_categorical:
        # qualitative palette so adjacent perm indices look distinct.
        # Plotly's Alphabet palette has 26 colors; we cycle for >26
        # perms (typical demo has 25-50 fit perms, so each colour gets
        # used by ~2 perms -- still visually clear).
        import plotly.colors as pc
        palette = pc.qualitative.Alphabet
        n_palette = len(palette)
        marker_kw['color'] = [palette[int(p) % n_palette] for p in color_v]
        # no colorbar for qualitative -- the legend would have 50 rows.
    elif color_v is not None:
        marker_kw['color'] = color_v
        marker_kw['colorscale'] = 'Viridis'
        marker_kw['showscale'] = True
        marker_kw['colorbar'] = dict(title=color_feat, thickness=12,
                                     len=0.7)
    else:
        marker_kw['color'] = 'rgba(80,80,80,0.55)'
    fig.add_trace(go.Scattergl(
        x=x_v, y=y_v,
        mode='markers',
        marker=marker_kw,
        hovertemplate=(f'{x_feat}=%{{x}}<br>'
                       f'{y_feat}=%{{y:.3f}}<extra>H0</extra>'),
        showlegend=False,
    ))

    # GAM overlay (always relevant in H0): use raw mu, or zero/one for
    # the adjusted axes since by construction the mean is 0 and SD 1.
    _add_h0_gam_overlay(fig, ana_glow, x_feat, y_feat)

    # FWER threshold reference -- only meaningful when y matches the
    # active score_method's natural adjusted axis.
    _add_h0_threshold_line(fig, ana_glow, x_feat, y_feat)

    fig.update_layout(
        xaxis=dict(title=x_feat),
        yaxis=dict(title=y_feat,
                   type='log' if log_y else 'linear'),
        margin=dict(l=10, r=10, t=10, b=40),
        height=420,
        plot_bgcolor='white',
        annotations=[
            dict(text='Permuted Samples (H₀)',
                 xref='paper', yref='paper', x=0.99, y=0.99,
                 xanchor='right', yanchor='top',
                 showarrow=False,
                 font=dict(size=11, color='#666',
                           family='monospace'),
                 bgcolor='rgba(255,255,255,0.85)',
                 bordercolor='#ccc', borderwidth=1, borderpad=3),
        ],
    )
    if x_feat == 'n_voxel':
        fig.update_xaxes(type='log')
    return fig


def _add_h0_gam_overlay(fig, ana_glow, x_feat, y_feat):
    """Mu / sigma GAM curves on the H0 scatter, axis-aware."""
    from glow.analysis import AnalysisGLOW
    if x_feat != 'n_voxel':
        return
    mu_gam = getattr(ana_glow, 'mu_gam', None) or getattr(ana_glow, 'adj_gam',
                                                          None)
    if mu_gam is None:
        return

    sizes = np.asarray(_ensure_1d(ana_glow.size), dtype=float)
    sizes = sizes[sizes > 0]
    if len(sizes) == 0:
        return
    sz = np.geomspace(max(sizes.min(), 1), sizes.max(), 200)

    mu_fn = AnalysisGLOW.mu_fn_from_gam(mu_gam)
    sigma_gam = getattr(ana_glow, 'sigma_gam', None)
    sigma_fn = AnalysisGLOW.sigma_fn_from_gam(sigma_gam)
    BIAS_LOG_VAR = 1.27  # digamma correction for visual match
    sigma = sigma_fn(sz) * np.exp(0.5 * BIAS_LOG_VAR)
    mu = mu_fn(sz)

    if y_feat == 'llr':
        mean_line = mu
        upper, lower = mu + sigma, mu - sigma
    elif y_feat == 'llr_adjusted':
        mean_line = np.zeros_like(sz)
        upper, lower = sigma, -sigma
    elif y_feat == 'z_score':
        if sigma_gam is None:
            return
        mean_line = np.zeros_like(sz)
        # in z space sigma is 1 by construction (after bias correction);
        # use it as a unit reference band
        upper, lower = np.ones_like(sz), -np.ones_like(sz)
    else:
        return

    fig.add_trace(go.Scatter(
        x=sz, y=mean_line, mode='lines',
        line=dict(color='rgba(200,0,0,0.7)', width=2, dash='dash'),
        name='GAM μ̂', showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([sz, sz[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill='toself', fillcolor='rgba(200,0,0,0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False, hoverinfo='skip',
    ))


def _add_h0_threshold_line(fig, ana_glow, x_feat, y_feat):
    """Horizontal FWER threshold reference, axis-aware."""
    crit = getattr(ana_glow, 'adj_crit', None)
    if crit is None or not np.isfinite(crit):
        return
    score_method = getattr(ana_glow, 'score_method', 'mean_adj')

    # crit lives in mean_adj units when score_method='mean_adj', and in
    # z units when score_method='z_score'.  Draw the line only on the
    # axis where it is meaningful.
    if score_method == 'mean_adj' and y_feat == 'llr_adjusted':
        line_y = float(crit)
        label = f'FWER α={ana_glow.alpha_fwer:.2f} (LLR−μ̂)'
    elif score_method == 'z_score' and y_feat == 'z_score':
        line_y = float(crit)
        label = f'FWER α={ana_glow.alpha_fwer:.2f} (z)'
    else:
        return

    fig.add_hline(
        y=line_y,
        line=dict(color='rgba(220,80,80,0.6)', width=1.5, dash='dot'),
        annotation_text=label,
        annotation_position='top right',
        annotation_font_size=10,
        annotation_font_color='rgba(180,40,40,0.9)',
    )


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
              'dice', 'sens', 'spec', 'vox_in_target',
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
            elif isinstance(v, str):
                parts.append(f'{c}: {v}')
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
        fig.update_yaxes(type='log',
                         range=_log_y_range(y_v, ana_glow, y_feat,
                                            target_stats))

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


def _log_y_range(y_v, ana_glow, y_feat, target_stats):
    """Compute an explicit [log10_min, log10_max] range for log-y mode.

    Includes visible scatter data, any threshold hline, and the target star
    so that Plotly doesn't auto-range to absurd extremes from outliers.
    Returns *None* when there are no positive values at all (let Plotly
    fall back to defaults).
    """
    pos = y_v[np.isfinite(y_v) & (y_v > 0)]
    if len(pos) == 0:
        return None

    lo = np.log10(pos.min())
    hi = np.log10(pos.max())

    if y_feat == _ADJ_COL:
        thresh = _compute_adj_thresh(ana_glow)
        if thresh is not None and thresh > 0:
            t = np.log10(thresh)
            lo, hi = min(lo, t), max(hi, t)
    for feat, (attr, _) in _PVAL_THRESHOLD_MAP.items():
        if y_feat == feat:
            val = getattr(ana_glow, attr, None)
            if val is not None and val > 0:
                t = np.log10(val)
                lo, hi = min(lo, t), max(hi, t)

    # include target star
    if target_stats and y_feat in target_stats:
        ty = target_stats[y_feat]
        if np.isfinite(ty) and ty > 0:
            t = np.log10(ty)
            lo, hi = min(lo, t), max(hi, t)

    pad = max((hi - lo) * 0.05, 0.5)
    return [lo - pad, hi + pad]


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
    """Add GAM size-adjustment curve and ±sigma envelope.

    Active when x is ``n_voxel`` and y is either ``llr`` (raw LLR; mean
    line tracks mu_fn) or ``llr_adjusted`` (mean line at zero by
    construction; the sigma envelope is the informative piece).
    """
    from glow.analysis import AnalysisGLOW

    if x_feat != 'n_voxel' or y_feat not in ('llr', 'llr_adjusted'):
        return

    adj_gam = getattr(ana_glow, 'adj_gam', None)
    if adj_gam is None:
        return

    sizes = _ensure_1d(ana_glow.size).astype(float)
    sizes = sizes[sizes > 0]
    if len(sizes) == 0:
        return
    sz = np.linspace(max(sizes.min(), 1), sizes.max(), 200)

    r2 = getattr(ana_glow, '_primary_r2', None)
    if y_feat == 'llr':
        mu_fn = AnalysisGLOW.mu_fn_from_gam(adj_gam)
        mean_line = mu_fn(sz)
    else:  # llr_adjusted -- the GAM mean is zero by construction
        mean_line = np.zeros_like(sz)

    fig.add_trace(go.Scatter(
        x=sz, y=mean_line, mode='lines',
        line=dict(color='rgba(200,0,0,0.6)', width=2, dash='dash'),
        showlegend=False, hoverinfo='skip',
    ))

    # GAM-estimated ±1 sigma envelope (when sigma_gam is fitted; legacy
    # mean-only analyses have sigma_gam=None and skip this).  The
    # log-transform digamma bias (~0.53x) is corrected so the envelope
    # visually matches the empirical scatter spread.
    sigma_gam = getattr(ana_glow, 'sigma_gam', None)
    if sigma_gam is not None:
        sigma_fn = AnalysisGLOW.sigma_fn_from_gam(sigma_gam)
        BIAS_LOG_VAR = 1.27
        sigma_line = sigma_fn(sz) * np.exp(0.5 * BIAS_LOG_VAR)
        fig.add_trace(go.Scatter(
            x=np.concatenate([sz, sz[::-1]]),
            y=np.concatenate([mean_line + sigma_line,
                              (mean_line - sigma_line)[::-1]]),
            fill='toself',
            fillcolor='rgba(200,0,0,0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False, hoverinfo='skip',
        ))

    if y_feat == 'llr':
        eq_text = 'E[stat|H0] = GAM(log10(size))'
    else:
        eq_text = 'llr - GAM(log10(size))  (mean=0 by construction)'
    if r2 is not None and np.isfinite(r2):
        eq_text += f'  (R²={r2:.3f})'
    if sigma_gam is not None:
        eq_text += '  ±σ̂ shaded'
    fig.add_annotation(
        text=eq_text,
        xref='paper', yref='paper',
        x=0.02, y=0.02,
        showarrow=False,
        font=dict(size=11, color='rgba(200,0,0,0.8)', family='monospace'),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='rgba(200,0,0,0.3)',
        borderwidth=1, borderpad=4,
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

    # --- adjusted-stat axis: draw alpha_fwer line ---
    if alpha_fwer is not None and _ADJ_COL in (x_feat, y_feat):
        adj_thresh = _compute_adj_thresh(ana_glow)
        if adj_thresh is not None:
            style = dict(color='red', dash='dot', width=1.5)
            label = f'alpha_fwer={alpha_fwer}'
            if x_feat == _ADJ_COL:
                fig.add_vline(x=adj_thresh, line=style,
                              annotation_text=label,
                              annotation_position='top')
            if y_feat == _ADJ_COL:
                fig.add_hline(y=adj_thresh, line=style,
                              annotation_text=label,
                              annotation_position='right')



