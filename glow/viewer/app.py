"""Dash application for the glow viewer.

Defines the layout and all callbacks.  The ``launch()`` function is the
public entry point.

Stores:
    store-selected: JSON list of region indices [123, 456, ...]
        Updated by scatter clicks and the clear button.

    region-checklist: dcc.Checklist whose options mirror store-selected
        and whose value is the subset currently visible in the image viewer.
"""

import json
import os

import numpy as np
import dash_daq as daq
from dash import Dash, html, dcc, callback_context, no_update
from dash.dependencies import Input, Output, State

from .data import prep_df, get_feature_columns, compute_backgrounds
from .scatter import build_scatter
from .image import (build_label_map, build_2d_figure, build_region_overlay,
                    compute_bg_volume, get_region_color,
                    compute_region_center)
from .regression import (build_regression_figure, build_empty_regression,
                         _get_x_labels, _get_y_labels)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _controls_column(core_cols, mask_cols, default_x, default_y,
                     log_y_default):
    """Build dropdowns + Log Y toggle as a narrow vertical panel."""
    def _dd(id_, value, label, none_option=False):
        options = []
        if none_option:
            options.append({'label': 'None', 'value': '__none__'})
        options += [{'label': c, 'value': c} for c in core_cols]
        if mask_cols:
            options.append({
                'label': html.Span('── mask ──',
                                   style={'color': '#999',
                                          'fontStyle': 'italic',
                                          'fontSize': '10px'}),
                'value': '__divider__',
                'disabled': True,
            })
            options += [{'label': c, 'value': c} for c in mask_cols]
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
        html.Div([
            html.Label('Log Y', style={'fontSize': '12px',
                                       'marginRight': '6px',
                                       'color': '#555'}),
            daq.BooleanSwitch(id='log-y-switch', on=log_y_default,
                              color='#119DFF'),
        ], style={'display': 'flex', 'alignItems': 'center',
                  'marginBottom': '6px'}),
        _dd('dd-color', '__none__', 'Color', none_option=True),
    ], style={'width': '180px', 'padding': '10px',
              'borderRight': '1px solid #ddd', 'flexShrink': '0'})


def _region_panel():
    """Build the left-hand region selection panel (shared by 2D and 3D)."""
    return html.Div([
        html.Label('Selected regions',
                   style={'fontWeight': 'bold', 'fontSize': '13px'}),

        # hover preview toggle (always visible)
        html.Div([
            dcc.Checklist(
                id='toggle-hover-preview',
                options=[{'label': ' Preview on hover', 'value': 'on'}],
                value=['on'],
                style={'fontSize': '12px'},
            ),
        ], style={'marginTop': '4px', 'marginBottom': '4px',
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


def _regression_panel(x_names, y_names):
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
                             value=0, clearable=False,
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


def _defaults(core_cols, mask_cols):
    """Compute default dropdown values and log-toggle state."""
    from .scatter import _LOG_COLS
    all_cols = core_cols + mask_cols
    default_x = 'n_voxel' if 'n_voxel' in all_cols else all_cols[0]
    default_y = ('hotel_tr_adjusted' if 'hotel_tr_adjusted' in all_cols
                 else all_cols[min(1, len(all_cols) - 1)])
    log_y_default = default_y in _LOG_COLS
    return all_cols, default_x, default_y, log_y_default


def _make_layout_3d(core_cols, mask_cols, slicer0, slicer1, slicer2,
                    x_names=None, y_names=None):
    """Build layout for 3D data (with dash-slicer ortho views)."""
    all_cols, default_x, default_y, log_val = _defaults(core_cols, mask_cols)

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
            _controls_column(core_cols, mask_cols,
                             default_x, default_y, log_val),
            html.Div([
                dcc.Graph(id='scatter-plot',
                          config={'scrollZoom': True},
                          clear_on_unhover=True,
                          style={'width': '100%'}),
            ], style={'flex': '1', 'padding': '0'}),
        ], style={'display': 'flex', 'padding': '0 20px'}),

        # --- IMAGE + REGRESSION (side by side) ---
        html.Div([
            _region_panel(),

            # center: three linked ortho slicers
            html.Div([
                html.H4('IMAGE', style={
                    'margin': '0', 'fontSize': '14px',
                    'letterSpacing': '1px', 'color': '#555',
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
            _regression_panel(x_names or [], y_names or []),
        ], style={'display': 'flex', 'padding': '0 20px 20px 20px',
                  'borderTop': '2px solid #ccc', 'marginTop': '6px'}),

        # --- HIDDEN STORES ---
        dcc.Store(id='store-selected', data='[]'),
        dcc.Store(id='store-center', data='null'),
        dcc.Store(id='store-hover', data='null'),
    ], style={'fontFamily': 'Helvetica, Arial, sans-serif',
              'maxWidth': '1400px', 'margin': '0 auto'})


def _make_layout_2d(core_cols, mask_cols, bg_names,
                    x_names=None, y_names=None):
    """Build layout for 2D data (single go.Image view)."""
    all_cols, default_x, default_y, log_val = _defaults(core_cols, mask_cols)

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
            _controls_column(core_cols, mask_cols,
                             default_x, default_y, log_val),
            html.Div([
                dcc.Graph(id='scatter-plot',
                          config={'scrollZoom': True},
                          clear_on_unhover=True,
                          style={'width': '100%'}),
            ], style={'flex': '1', 'padding': '0'}),
        ], style={'display': 'flex', 'padding': '0 20px'}),

        # --- IMAGE + REGRESSION (side by side) ---
        html.Div([
            # left panel: background + region list
            html.Div([
                html.Label('Background', style={'fontWeight': 'bold',
                                                'fontSize': '13px'}),
                dcc.Dropdown(
                    id='dd-bg',
                    options=[{'label': 'None', 'value': '__none__'}] +
                            [{'label': n, 'value': n} for n in bg_names],
                    value=bg_names[0] if bg_names else '__none__',
                    clearable=False,
                    style={'marginBottom': '12px'},
                ),
                _region_panel(),
            ], style={'width': '180px', 'flexShrink': '0'}),

            # center: image viewer
            html.Div([
                html.H4('IMAGE', style={
                    'margin': '0', 'fontSize': '14px',
                    'letterSpacing': '1px', 'color': '#555',
                    'marginBottom': '4px'}),
                dcc.Graph(id='image-viewer',
                          config={'scrollZoom': True},
                          style={'width': '100%', 'height': '340px'}),
            ], style={'flex': '1', 'padding': '10px'}),

            # right: regression scatter
            _regression_panel(x_names or [], y_names or []),
        ], style={'display': 'flex', 'padding': '0 20px 20px 20px',
                  'borderTop': '2px solid #ccc', 'marginTop': '6px'}),

        # --- HIDDEN STORES ---
        dcc.Store(id='store-selected', data='[]'),
        dcc.Store(id='store-center', data='null'),
        dcc.Store(id='store-hover', data='null'),
    ], style={'fontFamily': 'Helvetica, Arial, sans-serif',
              'maxWidth': '1400px', 'margin': '0 auto'})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def _create_app(ana_glow, mask_target=None, feature_names=None):
    """Create and wire up the Dash app.

    Args:
        ana_glow (AnalysisGLOW): completed analysis
        mask_target: optional target mask
        feature_names (list[str] | None): optional human-readable names for
            each imaging feature (used in the background dropdown).

    Returns:
        app (Dash): configured Dash application
    """
    df = prep_df(ana_glow, mask_target=mask_target)
    core_cols, mask_cols = get_feature_columns(df)
    mask_idx = ana_glow.exp.mask_idx
    ndim = mask_idx.ndim
    is_3d = ndim == 3

    app = Dash(__name__, update_title=None)

    if is_3d:
        _setup_3d(app, ana_glow, df, core_cols, mask_cols,
                  feature_names=feature_names)
    else:
        _setup_2d(app, ana_glow, df, core_cols, mask_cols,
                  feature_names=feature_names)

    return app


def _setup_3d(app, ana_glow, df, core_cols, mask_cols, feature_names=None):
    """Set up the app for 3D data using dash-slicer."""
    from dash_slicer import VolumeSlicer

    bg_vol = compute_bg_volume(ana_glow, feature_idx=0)

    scene_id = 'glow-viewer'
    slicer0 = VolumeSlicer(app, bg_vol, axis=0, scene_id=scene_id)
    slicer1 = VolumeSlicer(app, bg_vol, axis=1, scene_id=scene_id)
    slicer2 = VolumeSlicer(app, bg_vol, axis=2, scene_id=scene_id)

    for s in (slicer0, slicer1, slicer2):
        s.graph.config['scrollZoom'] = False
        s.graph.style = {'height': '280px'}

    _, x_names, _ = _get_x_labels(ana_glow.exp)
    y_names = _get_y_labels(ana_glow.exp, feature_names=feature_names)

    app.layout = _make_layout_3d(core_cols, mask_cols,
                                 slicer0, slicer1, slicer2,
                                 x_names=x_names, y_names=y_names)

    # --- shared callbacks ---
    _register_scatter_callback(app, df, ana_glow)
    _register_selection_callback(app, ana_glow)
    _register_checklist_sync_callback(app, df)
    _register_hover_callback(app, ana_glow)
    _register_placeholder_callback(app)
    _register_regression_callback(app, ana_glow, df,
                                  feature_names=feature_names)

    # --- setpos store: dash-slicer picks this up automatically ---
    setpos_store = dcc.Store(
        id={'context': 'viewer-center', 'scene': scene_id, 'name': 'setpos'},
        data=None,
    )
    app.layout.children.append(setpos_store)

    @app.callback(
        Output({'context': 'viewer-center', 'scene': scene_id,
                'name': 'setpos'}, 'data'),
        [Input('store-center', 'data')],
        prevent_initial_call=True,
    )
    def center_slicers(center_json):
        if not center_json or center_json == 'null':
            return no_update
        center = json.loads(center_json)  # [i, j, k] in numpy order
        # dash-slicer setpos expects (x, y, z) = reversed numpy order
        return [center[2], center[1], center[0]]

    # --- overlay callback: visible regions + hover -> slicer overlay ---
    @app.callback(
        [Output(slicer0.overlay_data.id, 'data'),
         Output(slicer1.overlay_data.id, 'data'),
         Output(slicer2.overlay_data.id, 'data')],
        [Input('region-checklist', 'value'),
         Input('store-hover', 'data')],
        [State('store-selected', 'data')],
    )
    def update_overlays(visible, hover_json, selected_json):
        selected = json.loads(selected_json)
        visible = visible or []

        # append hover region if not already visible
        hover_reg = json.loads(hover_json) if hover_json and hover_json != 'null' else None
        show_list = list(visible)
        if hover_reg is not None and hover_reg not in show_list:
            show_list.append(hover_reg)

        # build overlay for all shown regions
        label_map = build_label_map(show_list, ana_glow)

        # color index must match position in selected list (for consistency)
        color_map = {r: i for i, r in enumerate(selected)}

        n_sel = len(selected)
        return (
            _build_overlay(slicer0, label_map, show_list, color_map,
                           hover_reg=hover_reg, n_selected=n_sel),
            _build_overlay(slicer1, label_map, show_list, color_map,
                           hover_reg=hover_reg, n_selected=n_sel),
            _build_overlay(slicer2, label_map, show_list, color_map,
                           hover_reg=hover_reg, n_selected=n_sel),
        )


def _build_overlay(slicer, label_map, visible_list, color_map,
                   hover_reg=None, n_selected=0):
    """Build overlay with colours matching the selected-list order.

    The hover region (if not already selected) uses the next colour in
    the palette so it keeps the same colour if the user clicks to add it.
    """
    from .image import get_region_color

    mask = np.zeros(label_map.shape, dtype=np.uint8)
    colors = []

    for label_val, reg_idx in enumerate(visible_list, start=1):
        region_voxels = label_map == reg_idx
        if region_voxels.any():
            mask[region_voxels] = label_val
        if reg_idx == hover_reg and reg_idx not in color_map:
            # hover-only: next colour in the palette
            r, g, b = get_region_color(n_selected)
            colors.append((r, g, b, 160))
        else:
            cidx = color_map.get(reg_idx, label_val - 1)
            r, g, b = get_region_color(cidx)
            colors.append((r, g, b, 160))

    if not colors:
        return slicer.create_overlay_data(mask, (0, 0, 0, 0))
    return slicer.create_overlay_data(mask, colors)


def _setup_2d(app, ana_glow, df, core_cols, mask_cols, feature_names=None):
    """Set up the app for 2D data using Plotly go.Image."""
    mask_idx = ana_glow.exp.mask_idx
    bg_dict = compute_backgrounds(ana_glow, feature_names=feature_names)
    bg_names = list(bg_dict.keys())

    _, x_names, _ = _get_x_labels(ana_glow.exp)
    y_names = _get_y_labels(ana_glow.exp, feature_names=feature_names)

    app.layout = _make_layout_2d(core_cols, mask_cols, bg_names,
                                 x_names=x_names, y_names=y_names)

    # --- shared callbacks ---
    _register_scatter_callback(app, df, ana_glow)
    _register_selection_callback(app, ana_glow)
    _register_checklist_sync_callback(app, df)
    _register_hover_callback(app, ana_glow)
    _register_placeholder_callback(app)
    _register_regression_callback(app, ana_glow, df,
                                  feature_names=feature_names)

    # --- image callback: visible regions + hover + background -> figure ---
    @app.callback(
        Output('image-viewer', 'figure'),
        [Input('region-checklist', 'value'),
         Input('store-hover', 'data'),
         Input('dd-bg', 'value')],
    )
    def update_image(visible, hover_json, bg_name):
        visible = visible or []

        # append hover region if not already visible
        hover_reg = json.loads(hover_json) if hover_json and hover_json != 'null' else None
        show_list = list(visible)
        if hover_reg is not None and hover_reg not in show_list:
            show_list.append(hover_reg)

        if bg_name and bg_name != '__none__' and bg_name in bg_dict:
            bg_img = bg_dict[bg_name]
        else:
            bg_img = np.full(mask_idx.shape, np.nan)

        label_map = build_label_map(show_list, ana_glow)
        return build_2d_figure(bg_img, label_map, show_list)


# ---------------------------------------------------------------------------
# Shared callbacks
# ---------------------------------------------------------------------------

def _register_scatter_callback(app, df, ana_glow):
    """Scatter plot updates when axes change or selection changes."""
    @app.callback(
        Output('scatter-plot', 'figure'),
        [Input('dd-x', 'value'),
         Input('dd-y', 'value'),
         Input('dd-color', 'value'),
         Input('store-selected', 'data'),
         Input('log-y-switch', 'on')],
    )
    def update_scatter(x_feat, y_feat, color_feat, selected_json,
                       log_y_on):
        selected = set(json.loads(selected_json))
        return build_scatter(df, ana_glow, x_feat, y_feat, color_feat,
                             selected_reg=selected,
                             log_y=bool(log_y_on))


def _register_selection_callback(app, ana_glow):
    """Scatter click or clear button -> update store-selected + store-center."""
    @app.callback(
        [Output('store-selected', 'data'),
         Output('store-center', 'data')],
        [Input('scatter-plot', 'clickData'),
         Input('btn-clear', 'n_clicks')],
        [State('store-selected', 'data')],
    )
    def toggle_region(click_data, clear_clicks, selected_json):
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'btn-clear':
            return '[]', 'null'

        if click_data is None:
            return no_update, no_update

        point = click_data['points'][0]
        reg_idx = point.get('customdata')
        if reg_idx is None:
            return no_update, no_update
        reg_idx = int(reg_idx)

        selected = json.loads(selected_json)
        if reg_idx in selected:
            selected.remove(reg_idx)
            return json.dumps(selected), no_update  # removal — don't recenter
        else:
            selected.append(reg_idx)
            center = compute_region_center(reg_idx, ana_glow)
            center_json = json.dumps(center) if center else 'null'
            return json.dumps(selected), center_json


def _register_checklist_sync_callback(app, df):
    """Sync the checklist options/value when store-selected changes.

    New regions default to visible (checked).  Previously unchecked
    regions stay unchecked.
    """
    @app.callback(
        [Output('region-checklist', 'options'),
         Output('region-checklist', 'value')],
        [Input('store-selected', 'data')],
        [State('region-checklist', 'options'),
         State('region-checklist', 'value')],
    )
    def sync_checklist(selected_json, prev_options, prev_value):
        selected = json.loads(selected_json)
        prev_options = prev_options or []
        prev_value = prev_value or []

        # which regions existed before?
        prev_set = {o['value'] for o in prev_options}
        prev_visible = set(prev_value)

        # build new options with coloured labels
        new_options = []
        for idx, reg_idx in enumerate(selected):
            r, g, b = get_region_color(idx)
            rows = df.loc[df['region_idx'] == reg_idx, 'n_voxel']
            size = int(rows.values[0]) if len(rows) else 0
            label = html.Span([
                html.Span('\u25A0 ',
                          style={'color': f'rgb({r},{g},{b})',
                                 'fontSize': '16px'}),
                f'Region {reg_idx}  ({size} vox)',
            ])
            new_options.append({'label': label, 'value': reg_idx})

        # new value: keep previously visible that still exist,
        # plus any newly added regions (default visible)
        new_regs = set(selected) - prev_set
        new_value = [r for r in selected
                     if r in new_regs or r in prev_visible]

        return new_options, new_value


def _register_hover_callback(app, ana_glow):
    """Hover over scatter -> update store-hover (+ center slicers).

    Also updates the hover toggle label to show the hovered region index.
    """
    @app.callback(
        [Output('store-hover', 'data'),
         Output('store-center', 'data', allow_duplicate=True),
         Output('toggle-hover-preview', 'options')],
        [Input('scatter-plot', 'hoverData')],
        [State('toggle-hover-preview', 'value')],
        prevent_initial_call=True,
    )
    def update_hover(hover_data, toggle):
        # always update the label to reflect the hovered region
        if hover_data is None:
            label = ' Preview on hover'
            if 'on' not in (toggle or []):
                return 'null', no_update, [{'label': label, 'value': 'on'}]
            return 'null', no_update, [{'label': label, 'value': 'on'}]

        point = hover_data['points'][0]
        reg_idx = point.get('customdata')
        if reg_idx is None:
            label = ' Preview on hover'
            return 'null', no_update, [{'label': label, 'value': 'on'}]

        reg_idx = int(reg_idx)
        label = f' Region {reg_idx}'
        new_options = [{'label': label, 'value': 'on'}]

        if 'on' not in (toggle or []):
            return 'null', no_update, new_options

        center = compute_region_center(reg_idx, ana_glow)
        center_json = json.dumps(center) if center else 'null'
        return json.dumps(reg_idx), center_json, new_options


def _register_placeholder_callback(app):
    """The placeholder hint is static -- always visible below the checklist."""
    # No dynamic callback needed; the text is set in _region_panel().
    pass


def _register_regression_callback(app, ana_glow, df, feature_names=None):
    """Regression scatter: visible regions + hover + axis dropdowns -> figure.

    Colours match the slicer overlay (same palette index per region).
    """
    @app.callback(
        Output('regression-plot', 'figure'),
        [Input('region-checklist', 'value'),
         Input('store-hover', 'data'),
         Input('dd-reg-x', 'value'),
         Input('dd-reg-y', 'value')],
        [State('store-selected', 'data')],
    )
    def update_regression(visible, hover_json, x_feat_idx, y_feat_idx,
                          selected_json):
        selected = json.loads(selected_json)
        visible = visible or []

        hover_reg = (json.loads(hover_json)
                     if hover_json and hover_json != 'null' else None)
        show_list = list(visible)
        if hover_reg is not None and hover_reg not in show_list:
            show_list.append(hover_reg)

        n_selected = len(selected)
        color_map = {r: i for i, r in enumerate(selected)}

        if not show_list:
            return build_empty_regression()

        return build_regression_figure(
            ana_glow=ana_glow,
            region_list=show_list,
            x_feat_idx=int(x_feat_idx),
            y_feat_idx=int(y_feat_idx),
            df=df,
            color_map=color_map,
            hover_reg=hover_reg,
            n_selected=n_selected,
            feature_names=feature_names,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _check_port(port):
    """Check whether *port* is available.  If not, offer to free it.

    Uses a plain socket bind test (cross-platform).  If the port is
    occupied, prompts the user for confirmation before attempting to
    kill the blocking process.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('127.0.0.1', port))
        sock.close()
        return  # port is free
    except OSError:
        pass

    print(f'\n  Port {port} is already in use.')
    answer = input('  Kill the process using it? [y/N] ').strip().lower()
    if answer not in ('y', 'yes'):
        print('  Aborted.  Use --port to choose a different port.')
        raise SystemExit(1)

    import subprocess
    import sys
    import time

    pids = _find_pids_on_port(port)
    if not pids:
        print(f'  Could not identify process on port {port}.  '
              f'Use --port to choose a different port.')
        raise SystemExit(1)

    # try SIGTERM first, escalate to SIGKILL if needed
    if sys.platform == 'win32':
        for pid in pids:
            subprocess.call(
                ['taskkill', '/F', '/PID', str(pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    else:
        import signal as _sig
        for pid in pids:
            try:
                os.kill(pid, _sig.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

        # give SIGTERM 2 seconds to work
        if not _wait_for_port(port, socket, timeout=2.0):
            # escalate to SIGKILL
            for pid in pids:
                try:
                    os.kill(pid, _sig.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    if _wait_for_port(port, socket, timeout=5.0):
        print(f'  Freed port {port}.')
    else:
        print(f'  Port {port} still in use.  Use --port to choose '
              f'a different port.')
        raise SystemExit(1)


def _find_pids_on_port(port):
    """Return list of PIDs listening on *port* (best-effort, cross-platform)."""
    import subprocess
    import sys

    pids = []
    if sys.platform == 'win32':
        try:
            out = subprocess.check_output(
                ['netstat', '-ano'], stderr=subprocess.DEVNULL,
            ).decode()
            for line in out.splitlines():
                if f':{port}' in line and 'LISTENING' in line:
                    try:
                        pids.append(int(line.strip().split()[-1]))
                    except ValueError:
                        pass
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    else:
        try:
            out = subprocess.check_output(
                ['lsof', '-t', '-i', f':{port}'],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            for pid_str in out.splitlines():
                try:
                    pids.append(int(pid_str))
                except ValueError:
                    pass
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return pids


def _wait_for_port(port, socket_mod, timeout=5.0):
    """Poll until *port* is free.  Returns True if freed within *timeout*."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        s = socket_mod.socket(socket_mod.AF_INET, socket_mod.SOCK_STREAM)
        s.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_REUSEADDR, 1)
        try:
            s.bind(('127.0.0.1', port))
            s.close()
            return True
        except OSError:
            s.close()
        time.sleep(0.1)
    return False


def launch(ana_glow, mask_target=None, port=8050, debug=False,
           feature_names=None):
    """Launch the glow viewer dashboard.

    Args:
        ana_glow (AnalysisGLOW): completed analysis
        mask_target (np.array): optional target mask (boolean, same shape
            as ana_glow.exp.mask_idx).  When provided, per-region f1/sens/
            spec/vox_in_target/vox_out_target columns become available.
        port (int): server port
        debug (bool): enable Dash debug mode (hot-reload).  If True,
            consider setting dev_tools_props_check=False for performance.
        feature_names (list[str] | None): optional human-readable names for
            each imaging feature (used in the background dropdown).
    """
    import signal
    import sys

    _check_port(port)

    app = _create_app(ana_glow, mask_target=mask_target,
                      feature_names=feature_names)

    # clean shutdown on Ctrl+C (and SIGTERM on Unix)
    def _shutdown(signum, frame):
        print('\n  shutting down glow:viewer ...')
        os._exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if sys.platform != 'win32':
        signal.signal(signal.SIGTERM, _shutdown)

    print(f'\n  glow:viewer running at http://localhost:{port}')
    print('  press Ctrl+C to stop\n')
    app.run(port=port, debug=debug,
            dev_tools_props_check=False if debug else None)
