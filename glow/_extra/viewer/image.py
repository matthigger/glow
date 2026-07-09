"""Image viewer utilities for the glow viewer.

3D images: uses dash-slicer VolumeSlicer (three linked orthogonal views
with click-to-navigate crosshairs and region overlay).

2D images: Plotly go.Image with background + colored region overlay.
"""

import numpy as np

import glow.graph

# qualitative palette for selected regions (up to 10, then cycles):
# blue, orange, green, red, purple, brown, pink, grey, olive, cyan
REGION_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def get_region_color(idx):
    """Return an RGB tuple for the i-th selected region."""
    return REGION_COLORS[idx % len(REGION_COLORS)]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def compute_region_center(reg_idx, exp, ana_glow):
    """Compute the centre-of-mass (voxel coordinates) for a single region.

    Args:
        reg_idx (int): region index
        exp (Experiment): the experiment the analysis was fit on (mask_idx)
        ana_glow (AnalysisGLOW): completed analysis (tree)

    Returns:
        list[float] or None: [i, j, k] voxel coordinates, or None if the
            region has no voxels.
    """
    label_map = build_label_map([reg_idx], exp, ana_glow)
    coords = np.argwhere(label_map == reg_idx)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0).tolist()


def build_label_map(reg_idx_list, exp, ana_glow):
    """Build a spatial label map for a list of region indices.

    Args:
        reg_idx_list (list[int]): region indices to show
        exp (Experiment): the experiment the analysis was fit on (mask_idx)
        ana_glow (AnalysisGLOW): provides the Ward tree (children)

    Returns:
        label_map (np.array): same shape as mask_idx, -1 outside regions
    """
    if not reg_idx_list:
        return np.full(exp.mask_idx.shape, -1, dtype=int)

    return glow.graph.get_label_map(
        reg_idx_list=reg_idx_list,
        mask_idx=exp.mask_idx,
        children=ana_glow.children)


def compute_bg_volume(exp, feature_idx=0, image_idx=None):
    """Compute a background volume from image data.

    Uses original (pre-scaled) intensities so that the background matches
    the user's input images.  Voxels outside the analysis mask are set to
    zero.

    Args:
        exp (Experiment): the experiment the analysis was fit on
        feature_idx (int): which imaging feature to use (default 0)
        image_idx (int | None): if provided, use a single image (0-indexed)
            instead of the mean across all images.

    Returns:
        vol (np.array): same shape as mask_idx, float32
    """
    from .data import get_original_y

    mask_idx = exp.mask_idx
    # y is (b, num_img, num_vox)
    y = get_original_y(exp)

    feat_idx = min(feature_idx, y.shape[0] - 1)
    # y_agg is (num_vox,): a single image or the mean across images
    if image_idx is not None:
        y_agg = y[feat_idx, image_idx, :]
    else:
        y_agg = y[feat_idx].mean(axis=0)

    vol = np.zeros(mask_idx.shape, dtype=np.float32)
    vol[mask_idx >= 0] = y_agg[mask_idx[mask_idx >= 0]]
    return vol


# ---------------------------------------------------------------------------
# 3D: dash-slicer overlay
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2D: Plotly go.Image fallback
# ---------------------------------------------------------------------------

_CHANNEL_SCALES = {'red': 0, 'green': 1, 'blue': 2}


def _bg_to_rgba(bg_slice, channel=None, vmin=None, vmax=None):
    """Convert a background array to an RGBA uint8 image.

    A 3-D input (H, W, 3) is composited as RGB. A 2-D input with channel
    'red', 'green', or 'blue' uses the matching single-colour ramp (black
    to colour); otherwise it falls back to greyscale [20, 235].

    When vmin / vmax are supplied the colour scale is pinned to that range
    (keeping it consistent across different image selections).
    """
    if bg_slice.ndim == 3:
        return _rgb_to_rgba(bg_slice, vmin=vmin, vmax=vmax)

    h, w = bg_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    valid = ~np.isnan(bg_slice)
    if not valid.any():
        return rgba

    if vmin is None:
        vmin = np.nanmin(bg_slice)
    if vmax is None:
        vmax = np.nanmax(bg_slice)
    rng = vmax - vmin if vmax != vmin else 1.0

    intensity = np.clip(
        (bg_slice[valid] - vmin) / rng * 215 + 20, 20, 235
    ).astype(np.uint8)

    ch_idx = _CHANNEL_SCALES.get(channel.lower() if channel else '')
    if ch_idx is not None:
        rgba[valid, ch_idx] = intensity
    else:
        rgba[valid, 0] = intensity
        rgba[valid, 1] = intensity
        rgba[valid, 2] = intensity

    rgba[valid, 3] = 255
    return rgba


def _rgb_to_rgba(rgb_slice, vmin=None, vmax=None):
    """Convert a (H, W, 3) RGB background to an RGBA uint8 image.

    All three channels share a single global normalisation to [0, 255] so
    relative colour balance is preserved.

    When vmin / vmax are supplied the colour scale is pinned to that range
    (keeping it consistent across different image selections).
    """
    h, w, _ = rgb_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if vmin is None:
        vmin = np.nanmin(rgb_slice)
    if vmax is None:
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


