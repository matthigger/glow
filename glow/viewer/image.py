"""Image viewer utilities for the glow viewer.

3D images: uses dash-slicer VolumeSlicer (three linked orthogonal views
with click-to-navigate crosshairs and region overlay).

2D images: Plotly go.Image with background + colored region overlay.
"""

import numpy as np
import plotly.graph_objects as go

import glow.graph

# qualitative palette for selected regions (up to 10, then cycles)
REGION_COLORS = [
    (31, 119, 180),    # blue
    (255, 127, 14),    # orange
    (44, 160, 44),     # green
    (214, 39, 40),     # red
    (148, 103, 189),   # purple
    (140, 86, 75),     # brown
    (227, 119, 194),   # pink
    (127, 127, 127),   # grey
    (188, 189, 34),    # olive
    (23, 190, 207),    # cyan
]


def get_region_color(idx):
    """Return an RGB tuple for the i-th selected region."""
    return REGION_COLORS[idx % len(REGION_COLORS)]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def compute_region_center(reg_idx, ana_glow):
    """Compute the centre-of-mass (voxel coordinates) for a single region.

    Args:
        reg_idx (int): region index
        ana_glow (AnalysisGLOW): completed analysis

    Returns:
        list[float] or None: [i, j, k] voxel coordinates, or None if the
            region has no voxels.
    """
    label_map = build_label_map([reg_idx], ana_glow)
    coords = np.argwhere(label_map == reg_idx)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0).tolist()


def build_label_map(reg_idx_list, ana_glow):
    """Build a spatial label map for a list of region indices.

    Args:
        reg_idx_list (list[int]): region indices to show
        ana_glow (AnalysisGLOW): provides mask_idx and children

    Returns:
        label_map (np.array): same shape as mask_idx, -1 outside regions
    """
    if not reg_idx_list:
        return np.full(ana_glow.exp.mask_idx.shape, -1, dtype=int)

    return glow.graph.get_label_map(
        reg_idx_list=reg_idx_list,
        mask_idx=ana_glow.exp.mask_idx,
        children=ana_glow.child_dict[0])


def compute_bg_volume(ana_glow, feature_idx=0):
    """Compute a background volume from mean image data.

    Averages across images for the given feature, maps back into image
    space.  Voxels outside the analysis mask are set to zero.

    Args:
        ana_glow (AnalysisGLOW): completed analysis
        feature_idx (int): which imaging feature to use (default 0)

    Returns:
        vol (np.array): same shape as mask_idx, float32
    """
    exp = ana_glow.exp
    mask_idx = exp.mask_idx
    y = exp.y  # (b, num_img, num_vox)

    # mean across images for one feature
    feat_idx = min(feature_idx, y.shape[0] - 1)
    y_mean = y[feat_idx].mean(axis=0)  # (num_vox,)

    vol = np.zeros(mask_idx.shape, dtype=np.float32)
    vol[mask_idx >= 0] = y_mean[mask_idx[mask_idx >= 0]]
    return vol


# ---------------------------------------------------------------------------
# 3D: dash-slicer overlay
# ---------------------------------------------------------------------------

def build_region_overlay(slicer, label_map, reg_idx_list, alpha=160):
    """Build overlay data for one VolumeSlicer showing selected regions.

    Regions are rendered as semi-transparent coloured masks over the
    slicer's base volume (which shows the anatomical background).

    Args:
        slicer (VolumeSlicer): the slicer instance (used to encode the
            overlay for the correct axis)
        label_map (np.array): spatial label map (-1 outside regions)
        reg_idx_list (list[int]): ordered list of selected region indices
        alpha (int): overlay opacity 0-255

    Returns:
        overlay data suitable for the slicer's overlay_data Store
    """
    mask = np.zeros(label_map.shape, dtype=np.uint8)

    # build color list: label 1 -> first color, label 2 -> second, ...
    # (label 0 is auto-inserted as transparent by dash-slicer)
    colors = []
    for color_idx, reg_idx in enumerate(reg_idx_list):
        label = color_idx + 1  # 1-based (0 = no overlay)
        region_voxels = label_map == reg_idx
        if region_voxels.any():
            mask[region_voxels] = label
        r, g, b = get_region_color(color_idx)
        colors.append((r, g, b, alpha))

    if not colors:
        # no regions selected — return empty overlay
        return slicer.create_overlay_data(mask, (0, 0, 0, 0))

    return slicer.create_overlay_data(mask, colors)


# ---------------------------------------------------------------------------
# 2D: Plotly go.Image fallback
# ---------------------------------------------------------------------------

def _bg_to_rgba(bg_slice):
    """Convert a 2D background array to an RGBA uint8 image.

    Maps the non-NaN range to a grey scale [20, 235].
    """
    h, w = bg_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid = ~np.isnan(bg_slice)
    if not valid.any():
        return rgba

    vmin = np.nanmin(bg_slice)
    vmax = np.nanmax(bg_slice)
    rng = vmax - vmin if vmax != vmin else 1.0

    grey = ((bg_slice[valid] - vmin) / rng * 215 + 20).astype(np.uint8)
    rgba[valid, 0] = grey
    rgba[valid, 1] = grey
    rgba[valid, 2] = grey
    rgba[valid, 3] = 255

    return rgba


def _overlay_regions(rgba, label_slice, reg_idx_list, alpha=0.55):
    """Overlay coloured regions onto an RGBA image (mutates in-place)."""
    for color_idx, reg_idx in enumerate(reg_idx_list):
        mask = label_slice == reg_idx
        if not mask.any():
            continue
        r, g, b = get_region_color(color_idx)
        rgba[mask, 0] = np.clip(
            rgba[mask, 0] * (1 - alpha) + r * alpha, 0, 255).astype(np.uint8)
        rgba[mask, 1] = np.clip(
            rgba[mask, 1] * (1 - alpha) + g * alpha, 0, 255).astype(np.uint8)
        rgba[mask, 2] = np.clip(
            rgba[mask, 2] * (1 - alpha) + b * alpha, 0, 255).astype(np.uint8)


def _overlay_mask(rgba, bool_mask, color, alpha=0.55):
    """Overlay a boolean mask with a given RGB colour (mutates in-place)."""
    if bool_mask is None or not bool_mask.any():
        return
    r, g, b = color
    rgba[bool_mask, 0] = np.clip(
        rgba[bool_mask, 0] * (1 - alpha) + r * alpha, 0, 255).astype(np.uint8)
    rgba[bool_mask, 1] = np.clip(
        rgba[bool_mask, 1] * (1 - alpha) + g * alpha, 0, 255).astype(np.uint8)
    rgba[bool_mask, 2] = np.clip(
        rgba[bool_mask, 2] * (1 - alpha) + b * alpha, 0, 255).astype(np.uint8)


def build_2d_figure(bg_img, label_map, reg_idx_list):
    """Build a single-view Plotly figure for 2D images.

    Args:
        bg_img (np.array): 2D background (NaN outside mask)
        label_map (np.array): 2D label map (-1 outside regions)
        reg_idx_list (list[int]): selected region indices

    Returns:
        fig (go.Figure)
    """
    rgba = _bg_to_rgba(bg_img)
    _overlay_regions(rgba, label_map, reg_idx_list)

    fig = go.Figure()
    fig.add_trace(go.Image(z=rgba))
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig
