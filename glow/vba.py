"""Threshold-Free Cluster Enhancement (TFCE).

Pure-Python implementation (Smith & Nichols, 2009) matching FSL's
fslmaths -tfce output exactly.
"""

import numpy as np
from scipy.ndimage import label, generate_binary_structure


def apply_tfce_img(x, H=2.0, E=0.5, connectivity=6, n_steps=100):
    """apply TFCE to a 3d statistical image.

    computes TFCE(p) = sum_h e(h)^E * h^H where e(h) is the cluster
    extent at threshold h.

    Args:
        x (np.array): 3d array of statistical values
        H (float): height exponent
        E (float): extent exponent
        connectivity (int): 6 (faces), 18 (faces+edges), or 26 (full)
        n_steps (int): number of threshold steps (100 matches FSL)

    Returns:
        tfce (np.array): TFCE-enhanced image, same shape as x
    """
    x = np.asarray(x)

    # connectivity structure for scipy.ndimage.label
    if connectivity == 6:
        struct = generate_binary_structure(3, 1)
    elif connectivity == 18:
        struct = generate_binary_structure(3, 2)
    else:
        struct = generate_binary_structure(3, 3)

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
    """apply TFCE to a vector of stats given a 3d mask index.

    pads the mask with a single-voxel border of zeros (TFCE requires
    the image boundary to be zero).

    Args:
        x (np.array): 1d array of statistical values (one per active voxel)
        mask_idx (np.array): 3d index array, -1 for inactive voxels,
            otherwise the voxel index into x

    Returns:
        tfce (np.array): TFCE stats for each active voxel
    """
    # zero pad (TFCE requires border of zeros)
    mask_idx = np.pad(np.atleast_3d(mask_idx), pad_width=1,
                      constant_values=-1)

    mask_bool = mask_idx > -1
    img = np.zeros(mask_idx.shape)
    img[mask_bool] = x

    img_tfce = apply_tfce_img(img)
    return img_tfce[mask_bool]
