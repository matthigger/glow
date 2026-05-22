"""Threshold-Free Cluster Enhancement (TFCE).

Pure-Python implementation (Smith & Nichols, 2009) matching FSL's
fslmaths -tfce output exactly.
"""

import numpy as np
from scipy.ndimage import label, generate_binary_structure

from glow.mask import bbox_crop


def apply_tfce_img(x, H=2.0, E=0.5, connectivity=None, n_steps=100):
    """apply TFCE to a 2d or 3d statistical image.

    computes TFCE(p) = sum_h e(h)^E * h^H where e(h) is the cluster
    extent at threshold h.

    Args:
        x (np.array): 2d or 3d array of statistical values
        H (float): height exponent
        E (float): extent exponent
        connectivity (int or None): neighbourhood size.
            3d: 6 (faces, default), 18 (faces+edges), or 26 (full).
            2d: 4 (edges, default) or 8 (edges+corners).
            None selects face/edge connectivity for the input ndim.
        n_steps (int): number of threshold steps (100 matches FSL)

    Returns:
        tfce (np.array): TFCE-enhanced image, same shape as x
    """
    x = np.asarray(x)
    ndim = x.ndim
    assert ndim in (2, 3), f'expected 2d or 3d input, got {ndim}d'

    # connectivity structure for scipy.ndimage.label
    if connectivity is None:
        struct = generate_binary_structure(ndim, 1)
    elif ndim == 3:
        if connectivity == 6:
            struct = generate_binary_structure(3, 1)
        elif connectivity == 18:
            struct = generate_binary_structure(3, 2)
        else:
            struct = generate_binary_structure(3, 3)
    else:
        if connectivity == 4:
            struct = generate_binary_structure(2, 1)
        else:
            struct = generate_binary_structure(2, 2)

    # handle negative values and find max
    img_max = x.max()
    if img_max <= 0:
        return np.zeros_like(x)

    # FSL uses fixed number of steps with dh = max / n_steps
    dh = img_max / n_steps
    thresholds = np.linspace(dh, img_max, n_steps)

    tfce = np.zeros_like(x)

    for h in thresholds:
        binary = x >= h
        if not binary.any():
            continue

        labeled, n_clusters = label(binary, structure=struct)
        cluster_sizes = np.bincount(labeled.ravel())
        extent_map = cluster_sizes[labeled]
        extent_map[labeled == 0] = 0

        # TFCE contribution: e^E * h^H (no dh factor, matching FSL)
        tfce += (extent_map ** E) * (h ** H)

    return tfce


def apply_tfce_x(x, mask_idx):
    """apply TFCE to a vector of stats given a 2d or 3d mask index.

    Crops to the bounding box of active voxels before running TFCE,
    then pads with a single-voxel border of zeros (TFCE requires the
    image boundary to be zero).

    Args:
        x (np.array): 1d array of statistical values (one per active voxel)
        mask_idx (np.array): 2d or 3d index array, -1 for inactive voxels,
            otherwise the voxel index into x

    Returns:
        tfce (np.array): TFCE stats for each active voxel
    """
    if mask_idx.ndim == 2:
        pass  # keep 2d
    else:
        mask_idx = np.atleast_3d(mask_idx)

    # crop to bounding box of active voxels, then pad (TFCE needs zero border)
    mask_idx_bb, _ = bbox_crop(mask_idx, mask=(mask_idx > -1))
    mask_idx_bb = np.pad(mask_idx_bb, pad_width=1, constant_values=-1)
    mask_bool_bb_pad = mask_idx_bb > -1

    img = np.zeros(mask_idx_bb.shape)
    img[mask_bool_bb_pad] = x

    img_tfce = apply_tfce_img(img)
    return img_tfce[mask_bool_bb_pad]
