import numpy as np
from scipy.ndimage import label, generate_binary_structure


def apply_tfce_img(x, H=2.0, E=0.5, connectivity=6, n_steps=100):
    """applies tfce to an array (must be 3d)

    Pure Python implementation of Threshold-Free Cluster Enhancement
    (Smith & Nichols, 2009). Computes:
        TFCE(p) = Σ e(h)^E × h^H

    This matches FSL's fslmaths -tfce output exactly.

    Args:
        x: 3d array of statistical values (higher = more significant)
        H: height exponent (default 2.0)
        E: extent exponent (default 0.5)
        connectivity: 6 (faces), 18 (faces+edges), or 26 (full)
        n_steps: number of threshold steps (default 100, matches FSL)

    Returns:
        tfce enhanced image (same shape as input)
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

    # fsl uses fixed number of steps with dh = max / n_steps
    dh = img_max / n_steps
    thresholds = np.linspace(dh, img_max, n_steps)

    tfce = np.zeros_like(x)

    for h in thresholds:
        # threshold image
        binary = x >= h

        if not binary.any():
            continue

        # find connected components
        labeled, n_clusters = label(binary, structure=struct)

        # compute cluster sizes efficiently
        cluster_sizes = np.bincount(labeled.ravel())

        # map cluster size to each voxel (vectorized)
        extent_map = cluster_sizes[labeled]

        # zero out background
        extent_map[labeled == 0] = 0

        # tfce contribution: e^E * h^H (no dh factor, matching FSL)
        contribution = (extent_map ** E) * (h ** H)
        tfce += contribution

    return tfce


def apply_tfce_x(x, mask_idx):
    """applies tfce to a vector of stats given a 3d mask index

    Args:
        x: 1d array of statistical values
        mask_idx: 3d array where values > -1 indicate valid voxels

    Returns:
        tfce stats for valid voxels
    """
    # zero pad (tfce requires border of zeros)
    mask_idx = np.pad(np.atleast_3d(mask_idx), pad_width=1, constant_values=-1)

    # reshape into original image dimensions
    mask_bool = mask_idx > -1
    img = np.zeros(mask_idx.shape)
    img[mask_bool] = x

    # apply tfce
    img_tfce = apply_tfce_img(img)

    # extract tfce stats
    return img_tfce[mask_bool]
