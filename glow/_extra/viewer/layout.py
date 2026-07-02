"""Dash layout builders for the glow viewer.

Pure component-tree constructors with no callback wiring: the two
page layouts (_make_layout_2d, _make_layout_3d), the collapsible
experiment/analysis detail panels (_detail_panels), and the smaller
panel and key/value helpers they compose. app.py imports the page
builders and registers callbacks separately.
"""

import numpy as np
from dash import html, dcc

from .data import fwer_crit_llr_z
from .scatter import LOG_COLS


def _controls_column(generic_cols, sig_cols, prune_cols, mask_cols,
                     default_x, default_y, log_y_default, default_color):
    """Build dropdowns + Log Y toggle as a narrow vertical panel."""
    _divider_style = {'color': '#999', 'fontStyle': 'italic',
                      'fontSize': '10px'}

    def _add_group(options, cols, heading):
        """Append a disabled divider heading then the columns."""
        if not cols:
            return
        options.append({
            'label': html.Span(f'── {heading} ──', style=_divider_style),
            'value': f'__div_{heading}__',
            'disabled': True,
        })
        options += [{'label': c, 'value': c} for c in cols]

    def _dd(id_, value, label, none_option=False):
        """Build a labelled feature dropdown grouped by column category."""
        options = []
        if none_option:
            options.append({'label': 'None', 'value': '__none__'})
        options += [{'label': c, 'value': c} for c in generic_cols]
        options += [{'label': c, 'value': c} for c in sig_cols]
        _add_group(options, prune_cols, 'pruning')
        _add_group(options, mask_cols, 'target mask')
        return html.Div([
            html.Label(label, style={'fontWeight': 'bold',
                                     'fontSize': '12px',
                                     'marginBottom': '2px'}),
            dcc.Dropdown(id=id_, options=options, value=value,
                         clearable=False, style={'width': '100%'}),
        ], style={'marginBottom': '6px'})

    return html.Div([
        _dd('dd-x', default_x, 'X feature'),
        _dd('dd-y', default_y, 'Y feature'),
        # Log Y toggle sits right below Y feature
        dcc.Checklist(
            id='log-y-switch',
            options=[{'label': ' Log Y', 'value': 'on'}],
            value=['on'] if log_y_default else [],
            style={'fontSize': '12px', 'marginBottom': '6px'},
        ),
        _dd('dd-color', default_color, 'Color', none_option=True),
    ], style={'width': '180px', 'padding': '10px',
              'borderRight': '1px solid #ddd', 'flexShrink': '0'})


def _region_panel(region_ids):
    """Build the left-hand region selection panel (shared by 2D and 3D).

    Args:
        region_ids (np.array): (n_display,) int region indices to offer in
            the 'Add by index' lookup dropdown.  Restricted to the displayed
            (size >= min_vox) regions so the dropdown stays responsive on
            large trees -- it matches the set scattered above.
    """
    region_options = [{'label': f'Region {int(i)}', 'value': int(i)}
                      for i in region_ids]
    return html.Div([
        html.Label('Selected regions',
                   style={'fontWeight': 'bold', 'fontSize': '13px'}),

        # lookup dropdown
        html.Div([
            html.Label('Add by index',
                       style={'fontSize': '11px', 'color': '#777',
                              'marginBottom': '2px'}),
            dcc.Dropdown(
                id='dd-region-lookup',
                options=region_options,
                value=None,
                placeholder='e.g. Region 884',
                clearable=True,
                searchable=True,
                style={'fontSize': '12px'},
            ),
        ], style={'marginTop': '4px', 'marginBottom': '4px',
                  'borderBottom': '1px solid #eee', 'paddingBottom': '6px'}),

        # hover preview toggle (always visible)
        html.Div([
            dcc.Checklist(
                id='toggle-hover-preview',
                options=[{'label': ' Preview on hover', 'value': 'on'}],
                value=['on'],
                style={'fontSize': '12px'},
            ),
        ], style={'marginBottom': '4px',
                  'borderBottom': '1px solid #eee', 'paddingBottom': '4px'}),

        # per-region checklist
        html.Div(
            id='region-list-wrapper',
            children=[
                dcc.Checklist(
                    id='region-checklist',
                    options=[],
                    value=[],
                    style={'fontSize': '12px'},
                ),
            ],
            style={'maxHeight': '280px', 'overflowY': 'auto'},
        ),
        html.Div(id='region-placeholder',
                 children='Click a point in the scatter plot to add '
                          'regions.',
                 style={'color': '#999', 'fontSize': '12px',
                        'fontStyle': 'italic', 'marginTop': '6px'}),
        html.Button('Clear all', id='btn-clear',
                    style={'marginTop': '8px', 'fontSize': '12px'}),
    ], style={'width': '180px', 'padding': '10px',
              'borderRight': '1px solid #ddd', 'flexShrink': '0'})


def _section_header(title):
    """Return a styled section header with a top border."""
    return html.Div([
        html.H4(title,
                 style={'margin': '0', 'fontSize': '14px',
                        'letterSpacing': '1px', 'color': '#555',
                        'textTransform': 'uppercase'}),
    ], style={'padding': '10px 20px 4px 20px',
              'borderTop': '2px solid #ccc', 'marginTop': '6px'})


def _kv_row(key, val):
    """Render a single key/value row as a flex line."""
    return html.Div([
        html.Span(f'{key}:',
                  style={'fontWeight': 'bold', 'minWidth': '160px',
                         'display': 'inline-block', 'color': '#555'}),
        html.Span(str(val), style={'fontFamily': 'monospace'}),
    ], style={'fontSize': '12px', 'marginBottom': '2px'})


def _array_preview(arr, max_chars=600):
    """Truncated text preview of a numpy array, suitable for nested display."""
    try:
        s = np.array2string(arr, threshold=20, edgeitems=2, precision=4,
                            max_line_width=80)
    except Exception:
        return '<unrenderable array>'
    if len(s) > max_chars:
        s = s[:max_chars] + '...'
    return s


def _render_value(key, val, depth=0):
    """Recursively render a value as Dash html.

    Scalars/short reprs: single key/val row.
    numpy arrays: collapsed <details> with shape/dtype summary; expand
        for a truncated text preview of the values.
    dicts/lists/tuples: collapsed <details> with size summary; expand
        to recursively render each entry.
    """
    if val is None or isinstance(val, (bool, int, float, str)):
        return _kv_row(key, repr(val))

    indent = {'paddingLeft': f'{16 * (depth + 1)}px'}

    if isinstance(val, np.ndarray):
        return html.Details([
            html.Summary(
                f'{key}: ndarray shape={tuple(val.shape)} dtype={val.dtype}',
                style={'fontSize': '12px', 'cursor': 'pointer',
                       'fontFamily': 'monospace'}),
            html.Pre(_array_preview(val),
                     style={'fontFamily': 'monospace', 'fontSize': '11px',
                            'whiteSpace': 'pre-wrap',
                            'background': '#f7f7f7',
                            'padding': '6px', 'border': '1px solid #ddd',
                            'maxHeight': '240px', 'overflowY': 'auto',
                            'marginTop': '2px', **indent}),
        ], style={'marginBottom': '2px'})

    if isinstance(val, dict):
        children = [_render_value(repr(k), v, depth + 1)
                    for k, v in val.items()]
        return html.Details([
            html.Summary(f'{key}: dict ({len(val)} entries)',
                         style={'fontSize': '12px', 'cursor': 'pointer',
                                'fontFamily': 'monospace'}),
            html.Div(children, style=indent),
        ], style={'marginBottom': '2px'})

    if isinstance(val, (list, tuple)):
        # short flat lists of scalars: render inline
        if all(isinstance(x, (bool, int, float, str)) or x is None
               for x in val):
            s = repr(val)
            if len(s) <= 120:
                return _kv_row(key, s)
        children = [_render_value(f'[{i}]', v, depth + 1)
                    for i, v in enumerate(val)]
        return html.Details([
            html.Summary(f'{key}: {type(val).__name__} ({len(val)} items)',
                         style={'fontSize': '12px', 'cursor': 'pointer',
                                'fontFamily': 'monospace'}),
            html.Div(children, style=indent),
        ], style={'marginBottom': '2px'})

    # fallback: repr (truncated)
    try:
        s = repr(val)
    except Exception:
        s = '<unrenderable>'
    if len(s) > 200:
        s = s[:200] + '...'
    return _kv_row(key, s)


def _detail_panels(ana_glow, exp):
    """Build the experiment + analysis detail <details> panels.

    Both are collapsed by default. Values are pulled directly from ana_glow
    and exp at layout time, with no callbacks.
    """
    meta = getattr(exp, 'meta', {}) or {}

    y_shape = getattr(exp.y, 'shape', None) if exp.y is not None else '(unset)'
    x = getattr(exp, 'x', None)
    x_shape = getattr(x, 'shape', None)
    contrast = getattr(exp, 'contrast', None)

    mask_idx = exp.mask_idx
    n_active = int((mask_idx >= 0).sum())
    n_total = int(mask_idx.size)

    subjects = meta.get('subjects', [])
    features = meta.get('features', [])

    exp_rows = [
        _kv_row('y.shape', y_shape),
        _kv_row('image shape', tuple(mask_idx.shape)),
        _kv_row('x.shape', x_shape),
        _kv_row('contrast', np.asarray(contrast).tolist()
                if contrast is not None else None),
        _kv_row('mask active voxels', f'{n_active} / {n_total}'),
        _kv_row('subjects', f'{len(subjects)} '
                + (f'(first: {subjects[0]})' if subjects else '')),
        _kv_row('features', features),
    ]
    if 'affine' in meta and meta['affine'] is not None:
        exp_rows.append(_render_value('affine', meta['affine']))

    # --- analysis detail ---
    cluster_mode = getattr(ana_glow, 'cluster_mode', None)
    mode_label = str(cluster_mode) if cluster_mode is not None else '?'
    pval = getattr(ana_glow, 'pval', None)
    if pval is not None and len(pval):
        pval_min = float(np.nanmin(pval))
    else:
        pval_min = None

    ana_rows = [
        _kv_row('class', type(ana_glow).__name__),
        _kv_row('cluster_mode',
                f'{cluster_mode!r}  ({mode_label})'),
        _kv_row('alpha_fwer', getattr(ana_glow, 'alpha_fwer', None)),
        _kv_row('n_perm_fwer',
                getattr(ana_glow, 'n_perm_fwer', '<not stored>')),
        _kv_row('n_perm_inner',
                getattr(ana_glow, 'n_perm_inner', '<not stored>')),
        _kv_row('min_vox', getattr(ana_glow, 'min_vox', None)),
        _kv_row('adj_crit (llr_z)', fwer_crit_llr_z(ana_glow)),
        _kv_row('# significant regions',
                int(np.sum(~np.isnan(pval)
                           & (pval <= getattr(ana_glow, 'alpha_fwer', 0.05))))
                if pval is not None else 0),
        _kv_row('# discovered effects',
                len(getattr(ana_glow, 'effect_list', []) or [])),
        _kv_row('min p-value', pval_min),
    ]

    panel_style = {'padding': '10px 20px', 'borderTop': '1px solid #ddd'}
    summary_style = {'fontSize': '13px', 'fontWeight': 'bold',
                     'cursor': 'pointer', 'color': '#555',
                     'textTransform': 'uppercase', 'letterSpacing': '1px'}
    return html.Div([
        html.Details([
            html.Summary('Experiment detail', style=summary_style),
            html.Div(exp_rows, style={'marginTop': '8px'}),
        ], style=panel_style),
        html.Details([
            html.Summary('Analysis detail', style=summary_style),
            html.Div(ana_rows, style={'marginTop': '8px'}),
        ], style=panel_style),
    ])


def _regression_panel(x_names, y_names, default_x=0):
    """Build the right-hand regression scatter panel."""
    x_opts = [{'label': n, 'value': i} for i, n in enumerate(x_names)]
    y_opts = [{'label': n, 'value': i} for i, n in enumerate(y_names)]
    return html.Div([
        html.H4('REGRESSION', style={
            'margin': '0', 'fontSize': '14px',
            'letterSpacing': '1px', 'color': '#555',
            'marginBottom': '4px'}),
        html.Div([
            html.Div([
                html.Label('X', style={'fontSize': '11px',
                                       'fontWeight': 'bold',
                                       'marginRight': '4px'}),
                dcc.Dropdown(id='dd-reg-x', options=x_opts,
                             value=default_x, clearable=False,
                             style={'width': '100%', 'fontSize': '12px'}),
            ], style={'flex': '1', 'marginRight': '6px'}),
            html.Div([
                html.Label('Y', style={'fontSize': '11px',
                                       'fontWeight': 'bold',
                                       'marginRight': '4px'}),
                dcc.Dropdown(id='dd-reg-y', options=y_opts,
                             value=0, clearable=False,
                             style={'width': '100%', 'fontSize': '12px'}),
            ], style={'flex': '1'}),
        ], style={'display': 'flex', 'marginBottom': '4px'}),
        dcc.Graph(id='regression-plot',
                  config={'scrollZoom': True},
                  style={'width': '100%'}),
    ], style={'width': '380px', 'flexShrink': '0', 'padding': '10px',
              'borderLeft': '1px solid #ddd'})


def _defaults(generic_cols, sig_cols, prune_cols, mask_cols):
    """Compute default dropdown values and log-toggle state."""
    all_cols = generic_cols + sig_cols + prune_cols + mask_cols
    default_x = 'n_voxel' if 'n_voxel' in all_cols else all_cols[0]
    default_y = ('llr' if 'llr' in all_cols
                 else all_cols[min(1, len(all_cols) - 1)])
    log_y_default = default_y in LOG_COLS
    default_color = 'dice' if 'dice' in mask_cols else '__none__'
    return all_cols, default_x, default_y, log_y_default, default_color


def _make_layout_3d(generic_cols, sig_cols, prune_cols, mask_cols,
                    slicer0, slicer1, slicer2,
                    x_names=None, y_names=None, region_ids=None,
                    default_reg_x=0, num_img=0, feat_names=None,
                    subject_names=None):
    """Build layout for 3D data (with dash-slicer ortho views)."""
    all_cols, default_x, default_y, log_val, default_color = _defaults(
        generic_cols, sig_cols, prune_cols, mask_cols)

    feat_names = feat_names or []
    image_options = [{'label': 'Mean', 'value': 'mean'}]
    if subject_names and len(subject_names) == num_img:
        image_options += [{'label': subject_names[i], 'value': str(i)}
                          for i in range(num_img)]
    else:
        image_options += [{'label': f'Image {i}', 'value': str(i)}
                          for i in range(num_img)]
    feat_options = [{'label': n, 'value': str(i)}
                    for i, n in enumerate(feat_names)]

    _dd_label = {'fontSize': '11px', 'fontWeight': 'bold',
                 'marginBottom': '2px'}

    # dropdowns below IMAGE title: Feature (only when b > 1) + Image
    image_dd_children = []
    if len(feat_names) > 1:
        image_dd_children.append(html.Div([
            html.Label('Feature', style=_dd_label),
            dcc.Dropdown(
                id='dd-feature-3d',
                options=feat_options,
                value='0',
                clearable=False,
                style={'width': '100%', 'fontSize': '12px'}),
        ], style={'flex': '1', 'marginRight': '6px'}))
    else:
        image_dd_children.append(
            dcc.Store(id='dd-feature-3d', data='0'))
    image_dd_children.append(html.Div([
        html.Label('Image', style=_dd_label),
        dcc.Dropdown(
            id='dd-image-3d',
            options=image_options,
            value='mean',
            clearable=False,
            style={'width': '100%', 'fontSize': '12px'}),
    ], style={'flex': '1'}))

    return html.Div([
        # --- APP HEADER ---
        html.Div([
            html.H2('GLOW: Analysis Viewer',
                     style={'margin': '0', 'letterSpacing': '2px'}),
        ], style={'padding': '12px 20px', 'borderBottom': '2px solid #333',
                  'background': '#fafafa'}),

        # --- HIERARCHICAL SEGMENTATION ---
        _section_header('Hierarchical Segmentation'),
        html.Div([
            _controls_column(generic_cols, sig_cols, prune_cols, mask_cols,
                             default_x, default_y, log_val, default_color),
            html.Div([
                dcc.Graph(id='scatter-plot',
                          config={'scrollZoom': True},
                          clear_on_unhover=True,
                          style={'width': '100%'}),
            ], style={'flex': '1', 'padding': '0'}),
        ], style={'display': 'flex', 'padding': '0 20px'}),

        # --- IMAGE + REGRESSION (side by side) ---
        html.Div(id='lower-panel', children=[
            _region_panel(region_ids),

            # center: three linked ortho slicers
            html.Div([
                html.H4('IMAGE', style={
                    'margin': '0', 'fontSize': '14px',
                    'letterSpacing': '1px', 'color': '#555',
                    'marginBottom': '4px'}),
                html.Div(image_dd_children,
                         style={'display': 'flex',
                                'marginBottom': '4px'}),
                html.Div(style={
                    'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr 1fr',
                    'gap': '4px',
                }, children=[
                    html.Div([
                        slicer0.graph,
                        html.Div([slicer0.slider],
                                 style={'marginTop': '2px'}),
                        *slicer0.stores,
                    ]),
                    html.Div([
                        slicer1.graph,
                        html.Div([slicer1.slider],
                                 style={'marginTop': '2px'}),
                        *slicer1.stores,
                    ]),
                    html.Div([
                        slicer2.graph,
                        html.Div([slicer2.slider],
                                 style={'marginTop': '2px'}),
                        *slicer2.stores,
                    ]),
                ]),
            ], style={'flex': '1', 'padding': '10px'}),

            # right: regression scatter
            _regression_panel(x_names or [], y_names or [],
                              default_x=default_reg_x),
        ], style={'display': 'flex', 'padding': '0 20px 20px 20px',
                  'borderTop': '2px solid #ccc', 'marginTop': '6px'}),

        # --- HIDDEN STORES ---
        dcc.Store(id='store-selected', data='[]'),
        dcc.Store(id='store-center', data='null'),
        dcc.Store(id='store-hover', data='null'),
    ], style={'fontFamily': 'Helvetica, Arial, sans-serif',
              'maxWidth': '1400px', 'margin': '0 auto'})


def _make_layout_2d(generic_cols, sig_cols, prune_cols, mask_cols, bg_names,
                    x_names=None, y_names=None, region_ids=None,
                    default_reg_x=0, num_img=0, subject_names=None):
    """Build layout for 2D data (single go.Image view)."""
    all_cols, default_x, default_y, log_val, default_color = _defaults(
        generic_cols, sig_cols, prune_cols, mask_cols)

    bg_default = ('RGB' if 'RGB' in bg_names
                  else bg_names[0] if bg_names else '__none__')
    image_options = [{'label': 'Mean', 'value': 'mean'}]
    if subject_names and len(subject_names) == num_img:
        image_options += [{'label': subject_names[i], 'value': str(i)}
                          for i in range(num_img)]
    else:
        image_options += [{'label': f'Image {i}', 'value': str(i)}
                          for i in range(num_img)]

    _dd_label = {'fontSize': '11px', 'fontWeight': 'bold',
                 'marginBottom': '2px'}

    return html.Div([
        # --- APP HEADER ---
        html.Div([
            html.H2('GLOW: Analysis Viewer',
                     style={'margin': '0', 'letterSpacing': '2px'}),
        ], style={'padding': '12px 20px', 'borderBottom': '2px solid #333',
                  'background': '#fafafa'}),

        # --- HIERARCHICAL SEGMENTATION ---
        _section_header('Hierarchical Segmentation'),
        html.Div([
            _controls_column(generic_cols, sig_cols, prune_cols, mask_cols,
                             default_x, default_y, log_val, default_color),
            html.Div([
                dcc.Graph(id='scatter-plot',
                          config={'scrollZoom': True},
                          clear_on_unhover=True,
                          style={'width': '100%'}),
            ], style={'flex': '1', 'padding': '0'}),
        ], style={'display': 'flex', 'padding': '0 20px'}),

        # --- IMAGE + REGRESSION (side by side) ---
        html.Div(id='lower-panel', children=[
            # left panel: region selection (aligned with controls column)
            _region_panel(region_ids),

            # center: IMAGE with dropdowns below title
            html.Div([
                html.H4('IMAGE', style={
                    'margin': '0', 'fontSize': '14px',
                    'letterSpacing': '1px', 'color': '#555',
                    'marginBottom': '4px'}),
                html.Div([
                    html.Div([
                        html.Label('Background', style=_dd_label),
                        dcc.Dropdown(
                            id='dd-bg',
                            options=[{'label': n, 'value': n}
                                     for n in bg_names],
                            value=bg_default,
                            clearable=False,
                            style={'width': '100%', 'fontSize': '12px'}),
                    ], style={'flex': '1', 'marginRight': '6px'}),
                    html.Div([
                        html.Label('Image', style=_dd_label),
                        dcc.Dropdown(
                            id='dd-image',
                            options=image_options,
                            value='mean',
                            clearable=False,
                            style={'width': '100%', 'fontSize': '12px'}),
                    ], style={'flex': '1'}),
                ], style={'display': 'flex', 'marginBottom': '4px'}),
                dcc.Graph(id='image-viewer',
                          config={'scrollZoom': True},
                          style={'width': '100%', 'height': '340px'}),
            ], style={'flex': '1', 'padding': '10px'}),

            # right: regression scatter
            _regression_panel(x_names or [], y_names or [],
                              default_x=default_reg_x),
        ], style={'display': 'flex', 'padding': '0 20px 20px 20px',
                  'borderTop': '2px solid #ccc', 'marginTop': '6px'}),

        # --- HIDDEN STORES ---
        dcc.Store(id='store-selected', data='[]'),
        dcc.Store(id='store-center', data='null'),
        dcc.Store(id='store-hover', data='null'),
    ], style={'fontFamily': 'Helvetica, Arial, sans-serif',
              'maxWidth': '1400px', 'margin': '0 auto'})
