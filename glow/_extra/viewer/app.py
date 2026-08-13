"""Dash application for the glow viewer.

Defines the layout and all callbacks. The launch() function is the public
entry point.

Stores:
    store-selected: JSON list of region indices [123, 456, ...], updated by
        scatter clicks and the clear button.
    region-checklist: dcc.Checklist whose options mirror store-selected and
        whose value is the subset currently visible in the image viewer.
"""

import json
import os

import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback_context, no_update
from dash.dependencies import Input, Output, State

from .data import (prep_df, get_feature_columns, compute_backgrounds,
                    compute_bg_ranges, compute_target_stats)
from .scatter import build_scatter
from .image import (build_label_map, compute_bg_volume, get_region_color,
                    compute_region_center)
from .regression import (build_regression_figure, build_empty_regression,
                         get_x_labels, get_y_labels)
from ._port import _check_port
from .layout import (_make_layout_2d, _make_layout_3d,
                     _detail_panels)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def _create_app(ana_glow, exp, mask_target=None, y_features=None,
                subject_names=None, extra_df=None, min_vox=None,
                url_base_pathname=None, server=None,
                routes_pathname_prefix=None, requests_pathname_prefix=None):
    """Create and wire up the Dash app.

    Args:
        ana_glow (AnalysisGLOW): completed analysis
        exp (Experiment): the experiment the analysis was fit on
        mask_target: optional target mask
        y_features (list[str] | None): imaging feature names (auto-extracted
            from exp.meta['features'] when None).
        subject_names (list[str] | None): per-image subject names
            (auto-extracted from exp.meta['subjects'] when None).
        extra_df (pd.DataFrame | None): optional extra per-region data
            (keyed on region_idx) merged into the scatter DataFrame.
        min_vox (int | None): scatter (and offer in the lookup dropdown)
            only regions with at least this many voxels. None or 0 shows
            every region -- callers wanting the gentle large-tree default
            should resolve it via launch(). No prompting happens here, so
            this stays safe for the headless multi-demo web server.
        url_base_pathname (str | None): when serving under a path prefix on
            a shared Flask server (e.g. "/wgn2d/"). Default None serves at
            the root.
        server (flask.Flask | None): existing Flask server to mount onto.
            When None, Dash creates its own. Used by the multi-demo web
            entry point to host several apps under one server.
        routes_pathname_prefix (str | None): Dash routes_pathname_prefix,
            for mounting behind a path-stripping WSGI dispatcher (the
            Zenodo viewer mounts each app on its own server with routes at
            "/" while requests_pathname_prefix carries the external path).
            When given, used instead of url_base_pathname.
        requests_pathname_prefix (str | None): Dash requests_pathname_prefix
            paired with routes_pathname_prefix; see above.

    Returns:
        app (Dash): configured Dash application
    """
    min_vox = int(min_vox or 0)
    meta = getattr(exp, 'meta', {})
    if y_features is None:
        y_features = meta.get('features')
    if subject_names is None:
        subject_names = meta.get('subjects')

    df = prep_df(ana_glow, exp, mask_target=mask_target, extra_df=extra_df)
    generic_cols, sig_cols, prune_cols, mask_cols = get_feature_columns(df)
    mask_idx = exp.mask_idx
    ndim = mask_idx.ndim
    is_3d = ndim == 3

    # pre-compute target mask stats and voxel indices (if target provided)
    target_stats = None
    target_vox = None
    if mask_target is not None:
        target_stats = compute_target_stats(exp, mask_target)
        target_vox = mask_idx[mask_target & (mask_idx >= 0)]

    dash_kw = {'update_title': None}
    if (routes_pathname_prefix is not None
            or requests_pathname_prefix is not None):
        dash_kw['routes_pathname_prefix'] = routes_pathname_prefix
        dash_kw['requests_pathname_prefix'] = requests_pathname_prefix
    elif url_base_pathname is not None:
        dash_kw['url_base_pathname'] = url_base_pathname
    if server is not None:
        dash_kw['server'] = server
    app = Dash(__name__, **dash_kw)

    if is_3d:
        _setup_3d(app, ana_glow, exp, df,
                  generic_cols, sig_cols, prune_cols, mask_cols,
                  y_features=y_features, subject_names=subject_names,
                  target_stats=target_stats, target_vox=target_vox,
                  min_vox=min_vox)
    else:
        _setup_2d(app, ana_glow, exp, df,
                  generic_cols, sig_cols, prune_cols, mask_cols,
                  y_features=y_features, subject_names=subject_names,
                  target_stats=target_stats, target_vox=target_vox,
                  min_vox=min_vox)

    return app


def _display_region_ids(ana_glow, exp, min_vox):
    """Return the region indices to display (scatter + lookup dropdown).

    Keeps regions with size >= min_vox.  When min_vox is 0/1 (no cut) every
    region is returned, so the lookup dropdown matches the scattered set.

    Args:
        ana_glow (AnalysisGLOW): completed analysis (for size + tree shape).
        exp (Experiment): the experiment the analysis was fit on (num_vox).
        min_vox (int): minimum region size in voxels; 0/1 means no cut.

    Returns:
        region_ids (np.array): (n_display,) int region indices, ascending.
    """
    num_reg = exp.y.shape[2] + ana_glow.children.shape[0]
    if not min_vox or min_vox <= 1:
        return np.arange(num_reg)
    return np.flatnonzero(ana_glow.size >= min_vox)


def _setup_3d(app, ana_glow, exp, df,
              generic_cols, sig_cols, prune_cols, mask_cols,
              y_features=None, subject_names=None,
              target_stats=None, target_vox=None, min_vox=0):
    """Set up the app for 3D data using dash-slicer."""
    from dash_slicer import VolumeSlicer

    bg_vol = compute_bg_volume(exp, feature_idx=0)

    # Per-feature clim across all images so colour scale is stable when
    # switching images but adapts when switching features.
    from .data import get_original_y
    _y_all = get_original_y(exp)
    _mask = exp.mask_idx
    _valid_idx = _mask[_mask >= 0].ravel()
    _per_feat_clim = {}
    for _fi in range(_y_all.shape[0]):
        _vals = _y_all[_fi, :, _valid_idx]
        _per_feat_clim[_fi] = [float(np.nanmin(_vals)),
                               float(np.nanmax(_vals))]
    del _y_all, _valid_idx

    scene_id = 'glow-viewer'
    slicer0 = VolumeSlicer(app, bg_vol, axis=0, scene_id=scene_id)
    slicer1 = VolumeSlicer(app, bg_vol, axis=1, scene_id=scene_id)
    slicer2 = VolumeSlicer(app, bg_vol, axis=2, scene_id=scene_id)

    for s in (slicer0, slicer1, slicer2):
        s.graph.config['scrollZoom'] = False
        s.graph.style = {'height': '280px'}

    _, x_names, default_reg_x = get_x_labels(exp)
    y_names = get_y_labels(exp, y_features=y_features)

    b = exp.y.shape[0]
    num_img = exp.y.shape[1]
    region_ids = _display_region_ids(ana_glow, exp, min_vox)
    if y_features is None:
        feat_names = [f'feature {i}' for i in range(b)]
    else:
        feat_names = list(y_features)
    app.layout = _make_layout_3d(generic_cols, sig_cols, prune_cols, mask_cols,
                                 slicer0, slicer1, slicer2,
                                 x_names=x_names, y_names=y_names,
                                 region_ids=region_ids,
                                 default_reg_x=default_reg_x,
                                 num_img=num_img, feat_names=feat_names,
                                 subject_names=subject_names)
    app.layout.children.append(_detail_panels(ana_glow, exp))

    # pre-compute target mask in image space for overlays
    mask_target_img = None
    if target_vox is not None:
        mask_idx = exp.mask_idx
        mask_target_img = np.zeros(mask_idx.shape, dtype=bool)
        mask_target_img[mask_idx >= 0] = np.isin(
            mask_idx[mask_idx >= 0], target_vox)

    # --- shared callbacks ---
    _register_scatter_callback(app, df, ana_glow, exp,
                               target_stats=target_stats, min_vox=min_vox)
    _register_selection_callback(app, ana_glow, exp,
                                 mask_target_img=mask_target_img)
    _register_checklist_sync_callback(app, df,
                                      target_stats=target_stats)
    _register_hover_callback(app, ana_glow, exp,
                             mask_target_img=mask_target_img)
    _register_regression_callback(app, ana_glow, exp, df,
                                  y_features=y_features,
                                  subject_names=subject_names,
                                  target_vox=target_vox)
    _register_regression_click_callback(app, 'dd-image-3d')

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
        """Convert a stored centre to dash-slicer setpos coordinates."""
        if not center_json or center_json == 'null':
            return no_update
        # center is [i, j, k] in numpy order; setpos wants reversed (x, y, z)
        center = json.loads(center_json)
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
        """Rebuild the three slicer overlays from visible + hover regions."""
        selected = json.loads(selected_json)
        visible = visible or []

        # append hover region if not already visible
        hover_reg = (json.loads(hover_json)
                     if hover_json and hover_json != 'null' else None)
        show_list = list(visible)
        if hover_reg is not None and hover_reg not in show_list:
            show_list.append(hover_reg)
        show_list = [r for r in show_list if _valid_reg(r, ana_glow, exp)]

        # build overlay for tree regions only ('target' handled separately)
        tree_regs = [r for r in show_list if r != 'target']
        label_map = build_label_map(tree_regs, exp, ana_glow)

        # color index must match position in selected list (for consistency)
        color_map = {r: i for i, r in enumerate(selected)}

        n_sel = len(selected)
        return (
            _build_overlay(slicer0, label_map, show_list, color_map,
                           hover_reg=hover_reg, n_selected=n_sel,
                           mask_target_img=mask_target_img),
            _build_overlay(slicer1, label_map, show_list, color_map,
                           hover_reg=hover_reg, n_selected=n_sel,
                           mask_target_img=mask_target_img),
            _build_overlay(slicer2, label_map, show_list, color_map,
                           hover_reg=hover_reg, n_selected=n_sel,
                           mask_target_img=mask_target_img),
        )

    # --- background volume callback: feature / image dropdown -> volume ---
    _prop = 'data' if b == 1 else 'value'

    @app.callback(
        [Output(slicer0.state.id, 'data', allow_duplicate=True),
         Output(slicer1.state.id, 'data', allow_duplicate=True),
         Output(slicer2.state.id, 'data', allow_duplicate=True),
         Output(slicer0.clim.id, 'data', allow_duplicate=True),
         Output(slicer1.clim.id, 'data', allow_duplicate=True),
         Output(slicer2.clim.id, 'data', allow_duplicate=True)],
        [Input('dd-feature-3d', _prop),
         Input('dd-image-3d', 'value')],
        [State(slicer0.state.id, 'data'),
         State(slicer1.state.id, 'data'),
         State(slicer2.state.id, 'data')],
        prevent_initial_call=True,
    )
    def update_bg_volume(feat_val, img_val, st0, st1, st2):
        """Swap the slicer background volume on feature/image change."""
        feat_idx = int(feat_val) if feat_val is not None else 0
        img_idx = None if img_val in (None, 'mean') else int(img_val)
        new_vol = compute_bg_volume(exp,feature_idx=feat_idx,
                                    image_idx=img_idx)
        for s in (slicer0, slicer1, slicer2):
            s._volume = new_vol
        clim = _per_feat_clim.get(feat_idx, _per_feat_clim[0])
        # dash-slicer's upload_requested_slice only re-renders when
        # index_changed is True; force it so the new volume is displayed.
        import time
        t = time.time()
        return (
            {**st0, 'index_changed': True, '_vt': t},
            {**st1, 'index_changed': True, '_vt': t},
            {**st2, 'index_changed': True, '_vt': t},
            clim, clim, clim,
        )


def _build_overlay(slicer, label_map, visible_list, color_map,
                   hover_reg=None, n_selected=0, mask_target_img=None):
    """Build overlay with colours matching the selected-list order.

    The hover region (if not already selected) uses the next colour in the
    palette so it keeps the same colour if the user clicks to add it.
    'target' entries use mask_target_img for their voxels.
    """
    from .image import get_region_color

    mask = np.zeros(label_map.shape, dtype=np.uint8)
    colors = []

    for label_val, reg_idx in enumerate(visible_list, start=1):
        if reg_idx == 'target' and mask_target_img is not None:
            region_voxels = mask_target_img
        else:
            region_voxels = label_map == reg_idx
        if region_voxels.any():
            mask[region_voxels] = label_val

        if reg_idx == hover_reg and reg_idx not in color_map:
            r, g, b = get_region_color(n_selected)
            colors.append((r, g, b, 160))
        else:
            cidx = color_map.get(reg_idx, label_val - 1)
            r, g, b = get_region_color(cidx)
            colors.append((r, g, b, 160))

    if not colors:
        return slicer.create_overlay_data(mask, (0, 0, 0, 0))
    return slicer.create_overlay_data(mask, colors)


def _setup_2d(app, ana_glow, exp, df,
              generic_cols, sig_cols, prune_cols, mask_cols,
              y_features=None, subject_names=None,
              target_stats=None, target_vox=None, min_vox=0):
    """Set up the app for 2D data using Plotly go.Image."""
    mask_idx = exp.mask_idx
    bg_dict = compute_backgrounds(exp, y_features=y_features)
    bg_ranges = compute_bg_ranges(exp, y_features=y_features)
    bg_names = list(bg_dict.keys())

    _, x_names, default_reg_x = get_x_labels(exp)
    y_names = get_y_labels(exp, y_features=y_features)

    region_ids = _display_region_ids(ana_glow, exp, min_vox)
    num_img = exp.y.shape[1]
    app.layout = _make_layout_2d(generic_cols, sig_cols, prune_cols, mask_cols,
                                 bg_names,
                                 x_names=x_names, y_names=y_names,
                                 region_ids=region_ids,
                                 default_reg_x=default_reg_x,
                                 num_img=num_img,
                                 subject_names=subject_names)
    app.layout.children.append(_detail_panels(ana_glow, exp))

    # pre-compute target mask in image space for overlays
    mask_target_img = None
    if target_vox is not None:
        mask_target_img = np.zeros(mask_idx.shape, dtype=bool)
        mask_target_img[mask_idx >= 0] = np.isin(
            mask_idx[mask_idx >= 0], target_vox)

    # --- shared callbacks ---
    _register_scatter_callback(app, df, ana_glow, exp,
                               target_stats=target_stats, min_vox=min_vox)
    _register_selection_callback(app, ana_glow, exp,
                                 mask_target_img=mask_target_img)
    _register_checklist_sync_callback(app, df,
                                      target_stats=target_stats)
    _register_hover_callback(app, ana_glow, exp,
                             mask_target_img=mask_target_img)
    _register_regression_callback(app, ana_glow, exp, df,
                                  y_features=y_features,
                                  subject_names=subject_names,
                                  target_vox=target_vox)
    _register_regression_click_callback(app, 'dd-image')

    # --- image callback: regions + hover + background + image -> figure ---
    @app.callback(
        Output('image-viewer', 'figure'),
        [Input('region-checklist', 'value'),
         Input('store-hover', 'data'),
         Input('dd-bg', 'value'),
         Input('dd-image', 'value')],
    )
    def update_image(visible, hover_json, bg_name, image_sel):
        """Render the 2D background + region overlays as a go.Image figure."""
        visible = visible or []

        # append hover region if not already visible
        hover_reg = (json.loads(hover_json)
                     if hover_json and hover_json != 'null' else None)
        show_list = list(visible)
        if hover_reg is not None and hover_reg not in show_list:
            show_list.append(hover_reg)
        show_list = [r for r in show_list if _valid_reg(r, ana_glow, exp)]

        # resolve background: precomputed mean or single-image on-the-fly
        if image_sel is not None and image_sel != 'mean':
            active_bg = compute_backgrounds(
                exp, y_features=y_features,
                image_idx=int(image_sel))
        else:
            active_bg = bg_dict

        if bg_name and bg_name != '__none__' and bg_name in active_bg:
            bg_img = active_bg[bg_name]
        else:
            bg_img = np.full(mask_idx.shape, np.nan)

        # build label map for tree regions only
        tree_regs = [r for r in show_list if r != 'target']
        label_map = build_label_map(tree_regs, exp, ana_glow)

        from .image import _bg_to_rgba, _overlay_mask
        gmin, gmax = bg_ranges.get(bg_name, (None, None))
        rgba = _bg_to_rgba(bg_img, channel=bg_name, vmin=gmin, vmax=gmax)
        # overlay each entry in order, using palette color from position
        for color_idx, reg_idx in enumerate(show_list):
            if reg_idx == 'target' and mask_target_img is not None:
                r, g, b = get_region_color(color_idx)
                _overlay_mask(rgba, mask_target_img, (r, g, b))
            else:
                region_mask = label_map == reg_idx
                if region_mask.any():
                    r, g, b = get_region_color(color_idx)
                    _overlay_mask(rgba, region_mask, (r, g, b))

        fig = go.Figure()
        fig.add_trace(go.Image(z=rgba))
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        return fig


# ---------------------------------------------------------------------------
# Shared callbacks
# ---------------------------------------------------------------------------

def _valid_reg(reg_idx, ana_glow, exp):
    """Return True if reg_idx is in range for this analysis tree."""
    if reg_idx == 'target':
        return True
    num_reg = exp.y.shape[2] + ana_glow.children.shape[0]
    return isinstance(reg_idx, (int, np.integer)) and 0 <= reg_idx < num_reg


def _register_scatter_callback(app, df, ana_glow, exp, target_stats=None,
                               min_vox=0):
    """Scatter plot updates when axes change or selection changes."""
    @app.callback(
        Output('scatter-plot', 'figure'),
        [Input('dd-x', 'value'),
         Input('dd-y', 'value'),
         Input('dd-color', 'value'),
         Input('store-selected', 'data'),
         Input('log-y-switch', 'value')],
    )
    def update_scatter(x_feat, y_feat, color_feat, selected_json, log_y_val):
        """Rebuild the scatter on axis, colour, selection, or log-y change."""
        log_y = 'on' in (log_y_val or [])
        selected = set(json.loads(selected_json))
        return build_scatter(df, ana_glow, exp, x_feat, y_feat, color_feat,
                             selected_reg=selected,
                             log_y=log_y,
                             target_stats=target_stats,
                             min_vox=min_vox)


def _register_selection_callback(app, ana_glow, exp, mask_target_img=None):
    """Update store-selected on scatter click, clear, or lookup pick."""
    @app.callback(
        [Output('store-selected', 'data'),
         Output('store-center', 'data'),
         Output('dd-region-lookup', 'value')],
        [Input('scatter-plot', 'clickData'),
         Input('btn-clear', 'n_clicks'),
         Input('dd-region-lookup', 'value')],
        [State('store-selected', 'data')],
    )
    def toggle_region(click_data, clear_clicks, lookup_val, selected_json):
        """Add/remove the triggering region and centre the slicers on it."""
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'btn-clear':
            return '[]', 'null', None

        if trigger_id == 'dd-region-lookup':
            if lookup_val is None:
                return no_update, no_update, no_update
            reg_idx = int(lookup_val)
            if not _valid_reg(reg_idx, ana_glow, exp):
                return no_update, no_update, no_update
            selected = json.loads(selected_json)
            if reg_idx not in selected:
                selected.append(reg_idx)
            center = compute_region_center(reg_idx, exp, ana_glow)
            center_json = json.dumps(center) if center else 'null'
            return json.dumps(selected), center_json, None

        if click_data is None:
            return no_update, no_update, no_update

        point = click_data['points'][0]
        reg_idx = point.get('customdata')
        if reg_idx is None:
            return no_update, no_update, no_update

        # keep 'target' as a string; everything else becomes int
        if reg_idx != 'target':
            reg_idx = int(reg_idx)

        if not _valid_reg(reg_idx, ana_glow, exp):
            return no_update, no_update, no_update

        selected = json.loads(selected_json)
        if reg_idx in selected:
            selected.remove(reg_idx)
            return json.dumps(selected), no_update, no_update
        else:
            selected.append(reg_idx)
            if reg_idx == 'target' and mask_target_img is not None:
                coords = np.argwhere(mask_target_img)
                center = coords.mean(axis=0).tolist() if len(coords) else None
            else:
                center = compute_region_center(reg_idx, exp, ana_glow)
            center_json = json.dumps(center) if center else 'null'
            return json.dumps(selected), center_json, no_update


def _register_checklist_sync_callback(app, df, target_stats=None):
    """Sync the checklist options/value when store-selected changes.

    New regions default to visible (checked).  Previously unchecked
    regions stay unchecked.
    """
    n_target_vox = (int(target_stats['n_voxel'])
                    if target_stats is not None else 0)

    @app.callback(
        [Output('region-checklist', 'options'),
         Output('region-checklist', 'value')],
        [Input('store-selected', 'data')],
        [State('region-checklist', 'options'),
         State('region-checklist', 'value')],
    )
    def sync_checklist(selected_json, prev_options, prev_value):
        """Mirror store-selected into the checklist; new regions visible."""
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
            if reg_idx == 'target':
                display = f'Target mask  ({n_target_vox} vox)'
            else:
                rows = df.loc[df['region_idx'] == reg_idx, 'n_voxel']
                size = int(rows.values[0]) if len(rows) else 0
                display = f'Region {reg_idx}  ({size} vox)'
            label = html.Span([
                html.Span('\u25A0 ',
                          style={'color': f'rgb({r},{g},{b})',
                                 'fontSize': '16px'}),
                display,
            ])
            new_options.append({'label': label, 'value': reg_idx})

        # new value: keep previously visible that still exist,
        # plus any newly added regions (default visible)
        new_regs = set(selected) - prev_set
        new_value = [r for r in selected
                     if r in new_regs or r in prev_visible]

        return new_options, new_value


def _register_hover_callback(app, ana_glow, exp, mask_target_img=None):
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
        """Track the hovered region: store it, centre slicers, label toggle."""
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

        if reg_idx != 'target':
            reg_idx = int(reg_idx)
        if not _valid_reg(reg_idx, ana_glow, exp):
            return 'null', no_update, no_update

        if reg_idx == 'target':
            label = ' Target mask'
        else:
            label = f' Region {reg_idx}'
        new_options = [{'label': label, 'value': 'on'}]

        if 'on' not in (toggle or []):
            return 'null', no_update, new_options

        if reg_idx == 'target' and mask_target_img is not None:
            coords = np.argwhere(mask_target_img)
            center = coords.mean(axis=0).tolist() if len(coords) else None
        else:
            center = compute_region_center(reg_idx, exp, ana_glow)
        center_json = json.dumps(center) if center else 'null'
        return json.dumps(reg_idx), center_json, new_options


def _register_regression_callback(app, ana_glow, exp, df, y_features=None,
                                   subject_names=None, target_vox=None):
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
        """Rebuild the per-region regression on selection or axis change."""
        selected = json.loads(selected_json)
        visible = visible or []

        hover_reg = (json.loads(hover_json)
                     if hover_json and hover_json != 'null' else None)
        show_list = list(visible)
        if hover_reg is not None and hover_reg not in show_list:
            show_list.append(hover_reg)

        # guard against stale indices from a previous browser session
        show_list = [r for r in show_list if _valid_reg(r, ana_glow, exp)]

        n_selected = len(selected)
        color_map = {r: i for i, r in enumerate(selected)}

        if not show_list:
            return build_empty_regression()

        return build_regression_figure(
            ana_glow=ana_glow,
            exp=exp,
            region_list=show_list,
            x_feat_idx=int(x_feat_idx),
            y_feat_idx=int(y_feat_idx),
            df=df,
            color_map=color_map,
            hover_reg=hover_reg,
            n_selected=n_selected,
            y_features=y_features,
            subject_names=subject_names,
            target_vox=target_vox,
        )


def _register_regression_click_callback(app, image_dd_id):
    """Switch the Image dropdown on a regression data-point click.

    Each marker trace in the regression figure carries customdata with
    0-based image indices so the IMAGE view can show that observation.
    """
    @app.callback(
        Output(image_dd_id, 'value'),
        [Input('regression-plot', 'clickData')],
        prevent_initial_call=True,
    )
    def on_regression_click(click_data):
        """Return the clicked point's image index for the Image dropdown."""
        if not click_data:
            return no_update
        point = click_data['points'][0]
        img_idx = point.get('customdata')
        if img_idx is None:
            return no_update
        return str(int(img_idx))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _suggest_min_vox(size, max_regions):
    """Return the smallest size cutoff that keeps at most max_regions regions.

    Among all integer cutoffs T, returns the smallest for which
    (size >= T).sum() <= max_regions: one more than the (max_regions+1)-th
    largest region size.  Assumes len(size) > max_regions.

    The cutoff is a size, never a rank, so one region size is never split
    across the boundary: every region of a given voxel count is shown, or
    none of it is.  A tree whose sizes tie heavily at the boundary
    therefore keeps well under max_regions rather than breaking the tie
    arbitrarily -- the Ward trees this viewer draws tie hardest at the
    smallest sizes, where the leaves are, so that is the safe direction.

    Args:
        size (np.array): (num_reg,) int voxel count per region.
        max_regions (int): target ceiling on the number of displayed regions.

    Returns:
        min_vox (int): the suggested cutoff (>= 2).
    """
    # the (max_regions+1)-th largest size; any cutoff strictly above it keeps
    # at most max_regions regions (ties at that size are excluded)
    kth = np.partition(size, -(max_regions + 1))[-(max_regions + 1)]
    return int(kth) + 1


def _resolve_min_vox(ana_glow, min_vox, max_regions):
    """Resolve the effective scatter size cutoff.

    The viewer draws one point per Ward-tree region (num_vox leaves plus
    internal nodes), so a full-brain experiment is hundreds of thousands
    of points: enough to exhaust memory and to make every callback lag.
    A tree over max_regions is therefore capped by default, at the size
    cutoff _suggest_min_vox picks -- the caller opts out, rather than
    opting in, because the sluggish case is the common one.

      - min_vox is an int  -> use it verbatim (0 disables the cut).
      - min_vox is None and num_reg <= max_regions -> no cut (return 0).
      - min_vox is None and num_reg  > max_regions -> the suggested cut.

    Args:
        ana_glow (AnalysisGLOW): completed analysis (for size + tree shape).
        min_vox (int | None): caller-supplied cutoff, or None to auto-resolve.
        max_regions (int): target ceiling on the number of displayed regions.

    Returns:
        min_vox (int): the effective cutoff (0 = scatter everything).
    """
    if min_vox is not None:
        return int(min_vox)

    size = getattr(ana_glow, 'size', None)
    if size is None:
        return 0
    num_reg = len(size)
    if num_reg <= max_regions:
        return 0

    suggested = _suggest_min_vox(size, max_regions)
    kept = int((size >= suggested).sum())
    print(f'  glow:viewer: {num_reg:,} regions is more than the '
          f'{max_regions:,} this dashboard draws smoothly; showing the '
          f'{kept:,} regions of >= {suggested} voxels.  Pass min_vox=0 '
          f'(--min-vox 0) to show them all.')
    return suggested


def launch(ana_glow, exp, mask_target=None, port=8050, debug=False,
           y_features=None, subject_names=None,
           extra_df=None, quiet=True, min_vox=None, max_regions=10_000):
    """Launch the glow viewer dashboard.

    Args:
        ana_glow (AnalysisGLOW): completed analysis
        exp (Experiment): the experiment the analysis was fit on (the
            analysis does not store it; pass the one given to fit)
        mask_target (np.array): optional target mask (boolean, same shape
            as exp.mask_idx). When provided, per-region dice/sens/spec/
            vox_in_target/vox_out_target columns become available.
        port (int): server port
        debug (bool): enable Dash debug mode (hot-reload). If True, consider
            setting dev_tools_props_check=False for performance.
        y_features (list[str] | None): human-readable names for each
            imaging feature. When None, extracted from exp.meta['features']
            if available.
        subject_names (list[str] | None): human-readable names for each
            image / subject. When None, extracted from exp.meta['subjects']
            if available.
        extra_df (pd.DataFrame | None): optional extra per-region data
            (keyed on region_idx) merged into the scatter DataFrame.
        quiet (bool): suppress Dash/Werkzeug request logs.
        min_vox (int | None): scatter (and offer in the lookup dropdown) only
            regions with at least this many voxels.  None (default) auto-
            resolves: keep every region when there are <= max_regions of
            them, otherwise prompt on an interactive terminal or warn and
            keep all for non-interactive Python callers.  0 forces every
            region (and silences the warning); a positive int is used as-is.
        max_regions (int): ceiling used by the None default to pick a cutoff
            and to decide whether to prompt/warn at all.

    Note:
        Regions below min_vox are dropped everywhere in the dashboard (the
        scatter, its tree edges, and the 'Add by index' lookup), so set
        min_vox=0 if you need to inspect a small region by hand.
    """
    import logging
    import signal
    import sys

    _check_port(port)

    min_vox = _resolve_min_vox(ana_glow, min_vox, max_regions)

    if quiet:
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

    app = _create_app(ana_glow, exp, mask_target=mask_target,
                      y_features=y_features, subject_names=subject_names,
                      extra_df=extra_df, min_vox=min_vox)

    # clean shutdown on Ctrl+C (and SIGTERM on Unix)
    def _shutdown(signum, frame):
        """Print a notice and exit immediately on SIGINT/SIGTERM."""
        print('\n  shutting down glow:viewer ...')
        os._exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if sys.platform != 'win32':
        signal.signal(signal.SIGTERM, _shutdown)

    print(f'\n  glow:viewer running at http://localhost:{port}')
    print('  press Ctrl+C to stop\n')
    app.run(port=port, debug=debug,
            dev_tools_props_check=False if debug else None)
