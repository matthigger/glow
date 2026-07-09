"""Per-region regression scatter for the glow viewer.

For each selected region, plots individual images as markers:
    x = design-matrix feature value for that image
    y = mean imaging feature across the region's voxels for that image

Includes an OLS best-fit line per region and relevant statistics.
Colours are coordinated with the image slicer overlay.
"""

import numpy as np
import plotly.graph_objects as go

import glow.graph
from .image import get_region_color


def _get_voxel_indices(reg_idx, children, num_vox):
    """Return the leaf (voxel) indices belonging to reg_idx."""
    return list(glow.graph.iter_postorder(
        children=children, num_leaf=num_vox,
        node_start=reg_idx, only_leaf=True))


def get_x_labels(exp):
    """Return design matrix rows with labels using original indices.

    The bias row is included (labelled as such) so users see the same
    indexing as the design matrix.  The default_idx points to the first
    non-bias row so the dropdown starts on something useful.

    Args:
        exp (Experiment): may be an ExperimentScaled.

    Returns:
        x (np.array): (a, num_img) full design matrix
        x_names (list[str]): human-readable name per row
        default_idx (int): index of first non-bias row
    """
    # x is (a, num_img); contrast is (a,)
    x = exp.x
    contrast = exp.contrast

    is_bias = np.all(x == 1.0, axis=1)

    x_names = []
    default_idx = 0
    found_default = False
    for i in range(x.shape[0]):
        if is_bias[i]:
            tag = 'interest' if contrast[i] else 'nuisance'
            x_names.append(f'x{i} (bias, {tag})')
        elif contrast[i]:
            x_names.append(f'x{i} (interest)')
            if not found_default:
                default_idx = i
                found_default = True
        else:
            x_names.append(f'x{i} (nuisance)')
            if not found_default:
                default_idx = i
                found_default = True

    return x, x_names, default_idx


def get_y_labels(exp, y_features=None):
    """Return human-readable names for each imaging feature.

    Falls back to exp.meta['features'] when y_features is not supplied,
    then to y0, y1, ...
    """
    b = exp.y.shape[0]
    if y_features is None:
        y_features = getattr(exp, 'meta', {}).get('features')
    if y_features is not None and len(y_features) == b:
        return list(y_features)
    return [f'y{i}' for i in range(b)]


def _region_means(y, vox_indices):
    """Compute mean y across voxels in a region, per image, per feature.

    Args:
        y (np.array): (b, num_img, num_vox) imaging data
        vox_indices (list[int]): voxel indices for the region

    Returns:
        y_mean (np.array): (b, num_img) mean imaging value per image
    """
    return y[:, :, vox_indices].mean(axis=2)


def _region_stds(y, vox_indices):
    """Compute std of y across voxels in a region, per image, per feature.

    Args:
        y (np.array): (b, num_img, num_vox) imaging data
        vox_indices (list[int]): voxel indices for the region

    Returns:
        y_std (np.array): (b, num_img) spatial std per image
    """
    return y[:, :, vox_indices].std(axis=2)


def _ols_fit(x_vec, y_vec):
    """Simple OLS: y = a + b*x.  Returns (a, b, y_hat)."""
    X = np.column_stack([np.ones_like(x_vec), x_vec])
    beta, *_ = np.linalg.lstsq(X, y_vec, rcond=None)
    y_hat = X @ beta
    return beta[0], beta[1], y_hat


def _r_squared(y_vec, y_hat):
    """Coefficient of determination."""
    ss_res = np.sum((y_vec - y_hat) ** 2)
    ss_tot = np.sum((y_vec - y_vec.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def build_empty_regression():
    """Return an empty regression figure with a prompt message."""
    fig = go.Figure()
    fig.update_layout(
        height=340,
        margin=dict(l=45, r=10, t=15, b=35),
        plot_bgcolor='white',
        annotations=[dict(
            text='Select or hover over a region to see its '
                 'per-image regression',
            xref='paper', yref='paper', x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=13, color='#999'),
        )],
    )
    return fig


def build_regression_figure(ana_glow, exp, region_list, x_feat_idx, y_feat_idx,
                            df=None, color_map=None, hover_reg=None,
                            n_selected=0, y_features=None,
                            subject_names=None, target_vox=None):
    """Build a regression scatter for one or more regions.

    Args:
        ana_glow: AnalysisGLOW (provides the Ward tree)
        exp (Experiment): the experiment the analysis was fit on
        region_list (list[int]): region indices to show (visible + hover)
        x_feat_idx (int): which design-matrix row (original index)
        y_feat_idx (int): which imaging feature index
        df (pd.DataFrame): region-level stats (for hover annotations)
        color_map (dict): reg_idx -> colour palette index
        hover_reg (int|None): region being hovered (drawn translucent)
        n_selected (int): number of selected (non-hover) regions
        y_features (list[str]|None): human-readable imaging feature names
        subject_names (list[str]|None): per-image subject names for hover
        target_vox (np.array|None): voxel indices for the full target mask.
            When provided, an additional trace with star markers shows
            the target mask's per-image regression.

    Returns:
        fig (go.Figure)
    """
    from .data import get_original_y

    children = ana_glow.children
    y_orig = get_original_y(exp)
    num_vox = y_orig.shape[2]
    num_img = y_orig.shape[1]

    x_full, x_names, _ = get_x_labels(exp)
    y_names = get_y_labels(exp, y_features=y_features)

    x_feat_idx = min(x_feat_idx, len(x_names) - 1)
    y_feat_idx = min(y_feat_idx, len(y_names) - 1)

    x_label = x_names[x_feat_idx]
    y_label = y_names[y_feat_idx]

    # x_design is (num_img,)
    x_design = x_full[x_feat_idx]

    fig = go.Figure()

    for reg_idx in region_list:
        is_target = (reg_idx == 'target')

        # determine colour
        if reg_idx in (color_map or {}):
            cidx = color_map[reg_idx]
        elif reg_idx == hover_reg:
            cidx = n_selected
        else:
            cidx = 0
        r, g, b = get_region_color(cidx)
        is_hover = (reg_idx == hover_reg and reg_idx not in (color_map or {}))
        opacity = 0.4 if is_hover else 1.0

        # get voxel data
        if is_target and target_vox is not None:
            vox_idx = target_vox
        else:
            vox_idx = _get_voxel_indices(reg_idx, children, num_vox)
        # y_mean and y_std_all are (b, num_img); y_vals and y_std (num_img,)
        y_mean = _region_means(y_orig, vox_idx)
        y_std_all = _region_stds(y_orig, vox_idx)
        y_vals = y_mean[y_feat_idx]
        y_std = y_std_all[y_feat_idx]

        # OLS fit
        intercept, slope, y_hat = _ols_fit(x_design, y_vals)
        r2 = _r_squared(y_vals, y_hat)

        # region stats from df
        stat_parts = []
        if df is not None and not is_target:
            row = df.loc[df['region_idx'] == reg_idx]
            if len(row):
                row = row.iloc[0]
                for c in ('n_voxel', 'llr', 'llr_z', 'pval_fwer'):
                    v = row.get(c)
                    if v is not None and not (isinstance(v, float) and
                                              np.isnan(v)):
                        if isinstance(v, (int, np.integer)):
                            stat_parts.append(f'{c}: {v}')
                        else:
                            stat_parts.append(f'{c}: {v:.4g}')
                est = row.get('estimate_state', '')
                if est == 'has_effect':
                    stat_parts.append(f'<b>contains effect(s)</b>')

        # hover text per image
        region_name = 'Target mask' if is_target else f'Region {reg_idx}'
        _has_subj = (subject_names and len(subject_names) == num_img)
        hover_texts = []
        for img_i in range(num_img):
            img_label = subject_names[img_i] if _has_subj else f'image {img_i}'
            parts = [f'<b>{region_name}</b>',
                     img_label,
                     f'{x_label} = {x_design[img_i]:.4g}',
                     f'{y_label} (mean) = {y_vals[img_i]:.4g}',
                     f'sd = {y_std[img_i]:.4g}']
            hover_texts.append('<br>'.join(parts))

        n_vox = len(vox_idx)
        legend_label = (f'Target mask ({n_vox} vox)' if is_target
                        else f'Region {reg_idx} ({n_vox} vox)')

        # scatter points with +/-1 SD error bars
        err_opacity = max(opacity * 0.5, 0.15)
        fig.add_trace(go.Scatter(
            x=x_design, y=y_vals,
            mode='markers',
            marker=dict(
                size=8,
                color=f'rgba({r},{g},{b},{opacity})',
                line=dict(width=1,
                          color=f'rgba({r},{g},{b},{min(opacity+0.2, 1.0)})'),
            ),
            error_y=dict(
                type='data',
                array=y_std,
                visible=True,
                color=f'rgba({r},{g},{b},{err_opacity})',
                thickness=1,
                width=3,
            ),
            customdata=list(range(num_img)),
            text=hover_texts,
            hoverinfo='text',
            name=legend_label,
            legendgroup=f'reg-{reg_idx}',
            showlegend=True,
        ))

        # OLS fit line
        x_sorted = np.sort(x_design)
        y_line = intercept + slope * x_sorted

        # build annotation for fit line
        fit_parts = [f'slope={slope:.4g}', f'R\u00b2={r2:.3f}']
        fit_parts += stat_parts
        fit_text = '<br>'.join(fit_parts)

        fig.add_trace(go.Scatter(
            x=x_sorted, y=y_line,
            mode='lines',
            line=dict(color=f'rgba({r},{g},{b},{opacity})', width=2,
                      dash='dash' if is_hover else 'solid'),
            hoverinfo='text',
            text=fit_text,
            name=f'OLS (R\u00b2={r2:.3f})',
            legendgroup=f'reg-{reg_idx}',
            showlegend=False,
        ))

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=f'mean {y_label} over region',
        height=340,
        margin=dict(l=45, r=10, t=15, b=35),
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top',
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#ddd', borderwidth=1,
                    tracegroupgap=2, font=dict(size=10)),
        hoverlabel=dict(bgcolor='white'),
        plot_bgcolor='white',
    )
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')

    return fig
