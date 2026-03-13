"""Image viewer utilities for the glow viewer.

3D images: uses dash-slicer VolumeSlicer (three linked orthogonal views
with click-to-navigate crosshairs and region overlay).

2D images: Plotly go.Image with background + colored region overlay.
"""

import numpy as np

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
        children=ana_glow.children)


def compute_bg_volume(ana_glow, feature_idx=0, image_idx=None):
    """Compute a background volume from image data.

    Uses original (pre-scaled) intensities so that the background matches
    the user's input images.  Voxels outside the analysis mask are set to
    zero.

    Args:
        ana_glow (AnalysisGLOW): completed analysis
        feature_idx (int): which imaging feature to use (default 0)
        image_idx (int | None): if provided, use a single image (0-indexed)
            instead of the mean across all images.

    Returns:
        vol (np.array): same shape as mask_idx, float32
    """
    from .data import get_original_y

    exp = ana_glow.exp
    mask_idx = exp.mask_idx
    y = get_original_y(exp)  # (b, num_img, num_vox)

    feat_idx = min(feature_idx, y.shape[0] - 1)
    if image_idx is not None:
        y_agg = y[feat_idx, image_idx, :]  # (num_vox,) single image
    else:
        y_agg = y[feat_idx].mean(axis=0)   # (num_vox,) mean across images

    vol = np.zeros(mask_idx.shape, dtype=np.float32)
    vol[mask_idx >= 0] = y_agg[mask_idx[mask_idx >= 0]]
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

_CHANNEL_SCALES = {'red': 0, 'green': 1, 'blue': 2}


def _bg_to_rgba(bg_slice, channel=None):
    """Convert a background array to an RGBA uint8 image.

    * 3-D input ``(H, W, 3)`` is composited as RGB.
    * 2-D input with *channel* ``'red'``, ``'green'``, or ``'blue'`` uses
      the matching single-colour ramp (black -> colour).
    * Otherwise falls back to greyscale ``[20, 235]``.
    """
    if bg_slice.ndim == 3:
        return _rgb_to_rgba(bg_slice)

    h, w = bg_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid = ~np.isnan(bg_slice)
    if not valid.any():
        return rgba

    vmin = np.nanmin(bg_slice)
    vmax = np.nanmax(bg_slice)
    rng = vmax - vmin if vmax != vmin else 1.0

    intensity = ((bg_slice[valid] - vmin) / rng * 215 + 20).astype(np.uint8)

    ch_idx = _CHANNEL_SCALES.get(channel.lower() if channel else '')
    if ch_idx is not None:
        rgba[valid, ch_idx] = intensity
    else:
        rgba[valid, 0] = intensity
        rgba[valid, 1] = intensity
        rgba[valid, 2] = intensity

    rgba[valid, 3] = 255
    return rgba


def _rgb_to_rgba(rgb_slice):
    """Convert a ``(H, W, 3)`` RGB background to an RGBA uint8 image.

    All three channels share a single global normalisation to ``[0, 255]``
    so that relative colour balance is preserved.
    """
    h, w, _ = rgb_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    vmin = np.nanmin(rgb_slice)
    vmax = np.nanmax(rgb_slice)
    rng = vmax - vmin if vmax != vmin else 1.0

    for c in range(3):
        ch = rgb_slice[:, :, c]
        valid = ~np.isnan(ch)
        if not valid.any():
            continue
        rgba[valid, c] = np.clip(
            (ch[valid] - vmin) / rng * 255, 0, 255).astype(np.uint8)

    any_valid = ~np.isnan(rgb_slice).all(axis=2)
    rgba[any_valid, 3] = 255
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


