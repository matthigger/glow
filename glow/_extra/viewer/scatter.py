"""Scatter plot of hierarchical segmentation regions with tree overlay.

Builds on the existing scatter_plotly() in glow/plot.py but designed for
interactive use in the Dash viewer: clickable points, swappable axes,
and threshold reference lines.

Every clickable trace names its region in text -- the region index as a
string, or 'target' for the star. Dash forwards only a point's scalar
properties, and it recovers customdata by indexing the trace as the
browser holds it; plotly ships a float array there in binary, so that
index yields nothing and a numeric customdata row never reaches the
callback. customdata therefore carries the hover numbers alone.
"""

import numpy as np
import plotly.graph_objects as go

from glow.graph import get_parent

from .data import fwer_crit_llr_z


def _ensure_1d(arr):
    """Return a 1-D view: if 2-D (b, num_reg), take first row."""
    return arr if arr.ndim == 1 else arr[0]


# columns where a log scale is the sensible default
LOG_COLS = {'n_voxel', 'vox_in_target', 'vox_out_target'}

_ADJ_COL = 'llr_z'

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
    """Compute the llr_z value at the alpha_fwer significance boundary.

    When significant regions exist, returns the minimum llr_z among them
    (the empirical decision boundary). Otherwise falls back to the FWER
    critical llr_z derived from the max-z null (fwer_crit_llr_z). Returns
    None only when neither source is available.
    """
    alpha = getattr(ana_glow, 'alpha_fwer', None)
    if alpha is None:
        return None

    pval = getattr(ana_glow, 'pval', None)
    adj = getattr(ana_glow, 'z', None)
    if pval is None or adj is None:
        return fwer_crit_llr_z(ana_glow)
    if adj.ndim > 1:
        adj = adj[0]

    sig = ~np.isnan(pval) & (pval <= alpha)
    if sig.any():
        return float(np.nanmin(adj[sig]))

    return fwer_crit_llr_z(ana_glow)



def build_scatter(df, ana_glow, exp, x_feat, y_feat, color_feat,
                  selected_reg=None, plot_tree=True,
                  log_y=False, target_stats=None, min_vox=0):
    """Build an interactive Plotly scatter figure.

    Args:
        df (pd.DataFrame): region DataFrame (from viewer.data.prep_df)
        ana_glow (AnalysisGLOWBase): completed analysis (for tree + thresholds)
        exp (Experiment): the experiment the analysis was fit on (num_vox)
        x_feat (str): column name for x axis
        y_feat (str): column name for y axis
        color_feat (str): column name for color
        selected_reg (set): currently selected region indices (highlighted)
        plot_tree (bool): whether to draw hierarchy edges
        log_y (bool): apply log scale to y axis
        target_stats (dict | None): stats for the full target mask (from
            compute_target_stats). When both axes have finite values, a
            star marker is drawn at the target's position.
        min_vox (int): scatter only regions with at least this many voxels
            (n_voxel >= min_vox); 0 (default) scatters every region.  Large
            trees have one point per region (num_vox leaves + internal nodes),
            so a size cut keeps the figure responsive.  Hidden regions also
            drop their tree edges, since an edge is drawn only between two
            visible endpoints.

    Returns:
        fig (go.Figure): Plotly figure with clickable scatter
    """
    if selected_reg is None:
        selected_reg = set()

    num_vox = exp.y.shape[2]
    children = ana_glow.children
    parent = get_parent(children, num_vox)

    # sort by region_idx for consistent indexing
    _df = df.sort_values('region_idx').copy()
    x = _df[x_feat].values
    y = _df[y_feat].values
    no_color = (color_feat == '__none__')
    color = None if no_color else _df[color_feat].values
    states = _df['estimate_state'].values

    vis = _visible(_df, y, log_y, min_vox)

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

    # --- one trace per hover-wording group ---------------------------
    # Everything that used to vary per point in the hover string --
    # the estimate label, whether a region has a parent, whether it has
    # children -- is constant inside a group, so the figure carries one
    # template per group instead of one rendered string per region.
    hover_cols = ['n_voxel', 'llr', 'llr_z', 'pval_fwer']
    for c in ('dice', 'sens', 'ppv', 'spec', 'vox_in_target',
              'vox_out_target', 'llr_mu_h0', 'llr_std_h0'):
        if c in _df.columns and not _df[c].isna().all():
            hover_cols.append(c)

    int_cols = {c for c in hover_cols
                if np.issubdtype(_df[c].dtype, np.integer)}
    col_v = {c: _df[c].values[vis] for c in hover_cols}
    reg_v = _df['region_idx'].values[vis].astype(np.int64)
    parent_v = parent[reg_v]

    cmin = cmax = None
    if color_v is not None and np.isfinite(color_v).any():
        cmin = float(np.nanmin(color_v))
        cmax = float(np.nanmax(color_v))

    shown_scale = False
    for grp in _marker_groups(states_v, reg_v, parent_v, num_vox):
        pos = grp['pos']
        cols = list(hover_cols)
        stack = [col_v[c][pos] for c in hover_cols]
        if grp['has_parent']:
            cols.append('parent')
            stack.append(parent_v[pos])
        if grp['has_children']:
            cols += ['child0', 'child1']
            kids = children[reg_v[pos] - num_vox]
            stack += [kids[:, 0], kids[:, 1]]

        # float32 so plotly ships the block as binary rather than as
        # json numbers; every field is shown to 4 significant digits and
        # the indices are far below float32's exact-integer ceiling
        customdata = np.column_stack(stack).astype(np.float32)

        size, line_width, line_color = selection_style(
            reg_v[pos], selected_reg, grp['state'])
        marker = dict(size=size, symbol=_STATE_STYLE[grp['state']][0],
                      line=dict(width=line_width, color=line_color))
        if color_v is None:
            marker['color'] = _STATE_STYLE[grp['state']][1]
            marker['showscale'] = False
        else:
            marker.update(color=color_v[pos], colorscale='Viridis',
                          cmin=cmin, cmax=cmax, showscale=not shown_scale)
            if not shown_scale:
                marker['colorbar'] = dict(title=color_feat, x=1.02, len=0.5,
                                          y=0.15, yanchor='bottom')
                shown_scale = True

        fig.add_trace(go.Scatter(
            x=x_v[pos], y=y_v[pos],
            mode='markers',
            marker=marker,
            customdata=customdata,
            text=[str(r) for r in reg_v[pos]],
            hovertemplate=_hover_template(cols, int_cols, grp),
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
    if x_feat in LOG_COLS:
        fig.update_xaxes(type='log')
    # y: user-controlled toggle
    if log_y:
        fig.update_yaxes(type='log',
                         range=_log_y_range(y_v, ana_glow, y_feat,
                                            target_stats))

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


# Selected regions are drawn larger and outlined; a diamond keeps a thin
# outline so its shape reads against the colour scale.
_SIZE_SELECTED, _SIZE_DIAMOND, _SIZE_PLAIN = 14, 10, 7


def selection_style(reg_idx, selected_reg, state):
    """Return the marker size and outline for one trace's points.

    Shared by the figure builder and the callback that restyles a
    selection, so a click cannot drift from a rebuild.

    Args:
        reg_idx (np.array): (n,) int region index per point.
        selected_reg (set): region indices currently selected.
        state (str): the group's estimate_state, which fixes the symbol.

    Returns:
        size (list): (n,) marker size.
        line_width (list): (n,) outline width.
        line_color (list): (n,) outline colour.
    """
    sel = np.isin(reg_idx, list(selected_reg))
    diamond = state == 'has_effect'
    size = np.where(sel, _SIZE_SELECTED,
                    _SIZE_DIAMOND if diamond else _SIZE_PLAIN)
    width = np.where(sel, 2, 1.5 if diamond else 0)
    color = np.where(sel, 'black', 'black' if diamond else 'rgba(0,0,0,0)')
    return size.tolist(), width.tolist(), color.tolist()


def _marker_groups(states_v, reg_v, parent_v, num_vox):
    """Split visible regions into groups that share a hover wording.

    Args:
        states_v (np.array): (n,) estimate_state per visible region.
        reg_v (np.array): (n,) int region index per visible region.
        parent_v (np.array): (n,) parent index, -1 at the root.
        num_vox (int): leaves below this index have no children.

    Yields:
        dict: {state, has_parent, has_children, pos}, pos being the
            positions into the visible arrays, in trace order.
    """
    has_par = parent_v != -1
    has_kid = reg_v >= num_vox
    for state in _STATE_ORDER:
        for hp in (True, False):
            for hk in (True, False):
                pos = np.flatnonzero((states_v == state)
                                     & (has_par == hp) & (has_kid == hk))
                if len(pos):
                    yield {'state': state, 'has_parent': hp,
                           'has_children': hk, 'pos': pos}


def _hover_template(cols, int_cols, grp):
    """Build one hovertemplate for a group of regions.

    The region index comes from the trace's text array, not customdata;
    see the module docstring.

    Args:
        cols (list[str]): the customdata columns, in order.
        int_cols (set): columns to print without a float format.
        grp (dict): a _marker_groups entry.

    Returns:
        str: the plotly hovertemplate, its trailing box suppressed.
    """
    parts = ['Region %{text}']
    for i, c in enumerate(cols):
        if c in ('parent', 'child0', 'child1'):
            continue
        if c in int_cols:
            parts.append(f'{c}: %{{customdata[{i}]}}')
        else:
            parts.append(f'{c}: %{{customdata[{i}]:.4g}}')

    if grp['state'] != 'no_effect':
        parts.append(f'<b>{_STATE_STYLE[grp["state"]][2]}</b>')

    if grp['has_parent']:
        parts.append(f'parent: %{{customdata[{cols.index("parent")}]}}')
    else:
        parts.append('parent: none (root)')
    if grp['has_children']:
        i0, i1 = cols.index('child0'), cols.index('child1')
        parts.append(f'children: %{{customdata[{i0}]}}, '
                     f'%{{customdata[{i1}]}}')
    else:
        parts.append('children: none (leaf)')

    return '<br>'.join(parts) + '<extra></extra>'


def visible_regions(df, ana_glow, exp, y_feat, log_y=False, min_vox=0,
                    plot_tree=True):
    """Return the region indices each marker trace of build_scatter holds.

    The selection restyle patches traces by index, so it has to reproduce
    the split build_scatter made without rebuilding the figure. Both go
    through _marker_groups, so the two cannot disagree.

    Args:
        df (pd.DataFrame): region DataFrame, as build_scatter takes it.
        ana_glow (AnalysisGLOWBase): completed analysis (tree shape).
        exp (Experiment): the experiment it was fit on (num_vox).
        y_feat (str): y column, which decides what log_y hides.
        log_y (bool): y on a log scale, hiding non-positive values.
        min_vox (int): size cut, as build_scatter applies it.
        plot_tree (bool): whether the edge trace occupies index 0.

    Returns:
        list[tuple]: (trace_idx, state, reg_idx) per marker trace.
    """
    num_vox = exp.y.shape[2]
    parent = get_parent(ana_glow.children, num_vox)
    _df = df.sort_values('region_idx')
    vis = _visible(_df, _df[y_feat].values, log_y, min_vox)

    reg_v = _df['region_idx'].values[vis].astype(np.int64)
    states_v = _df['estimate_state'].values[vis]
    first = 1 if plot_tree else 0
    return [(first + i, g['state'], reg_v[g['pos']])
            for i, g in enumerate(
                _marker_groups(states_v, reg_v, parent[reg_v], num_vox))]


def _visible(_df, y, log_y, min_vox):
    """Return the boolean mask of regions the scatter draws.

    Args:
        _df (pd.DataFrame): region frame sorted by region_idx.
        y (np.array): (num_reg,) the y column's values.
        log_y (bool): drop non-positive y, which a log axis cannot show.
        min_vox (int): drop regions below this size; 0 keeps every one.

    Returns:
        np.array: (num_reg,) bool.
    """
    if log_y:
        vis = np.isfinite(y) & (y > 0)
    else:
        vis = np.ones(len(y), dtype=bool)
    if min_vox and min_vox > 1:
        vis &= _df['n_voxel'].values >= min_vox
    return vis

def _log_y_range(y_v, ana_glow, y_feat, target_stats):
    """Compute an explicit [log10_min, log10_max] range for log-y mode.

    Includes visible scatter data, any threshold hline, and the target star
    so Plotly does not auto-range to absurd extremes from outliers. Returns
    None when there are no positive values at all (Plotly falls back to its
    defaults).
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

    Clickable (text='target') so it behaves like any other region.
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
        text=['target'],
        hovertemplate=hover_text + '<extra></extra>',
        showlegend=True,
        name='target mask',
        legendgroup='target',
    ))



def _add_threshold_lines(fig, ana_glow, x_feat, y_feat):
    """Add dotted reference lines for thresholds when relevant."""
    alpha_fwer = getattr(ana_glow, 'alpha_fwer', None)

    # --- p-value axes ---
    for feat, (attr, style) in _PVAL_THRESHOLD_MAP.items():
        val = getattr(ana_glow, attr, None)
        if val is None:
            continue

        label = (f'alpha_fwer={alpha_fwer}' if feat == 'pval_fwer'
                 else f'{attr}={val}')

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

    # --- min_vox gate: vertical line on the H1 z-vs-size scatter ---
    _add_min_vox_line(fig, ana_glow, x_feat, y_feat)


def _add_min_vox_line(fig, ana_glow, x_feat, y_feat):
    """Draw a vertical line at x = ana_glow.min_vox on the z-vs-size view.

    The horizontal alpha_fwer threshold on the llr_z axis only applies to
    regions with size >= min_vox (smaller regions are excluded from the
    max-z null and assigned NaN p-values). Showing where that cutoff sits
    along the size axis makes the gating visible.

    Only drawn when x is n_voxel and y is llr_z; anywhere else the cutoff
    is not a meaningful reference. Silently skipped when min_vox is missing
    (older / partially-constructed analyses).
    """
    if x_feat != 'n_voxel' or y_feat != _ADJ_COL:
        return
    min_vox = getattr(ana_glow, 'min_vox', None)
    if min_vox is None:
        return
    style = dict(color='black', dash='solid', width=1.5)
    fig.add_vline(x=min_vox, line=style,
                  annotation_text=f'min_vox={int(min_vox)}',
                  annotation_position='top')



