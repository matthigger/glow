import os
import pathlib
import subprocess
import tempfile
import warnings
from shutil import which

import nibabel as nib
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

        if n_clusters == 0:
            continue

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


# --- FSL-based implementation for validation ---

fsl_path = pathlib.Path('/usr/local/fsl')
fsl_conf_sh = fsl_path / 'etc/fslconf/fsl.sh'
fslmaths_path = fsl_path / 'bin/fslmaths'

_fsl_available = which(str(fslmaths_path)) is not None


def is_fsl_available():
    """check if fsl is installed and accessible"""
    return _fsl_available


def apply_tfce_img_fsl(x):
    """applies tfce using fsl's fslmaths (for validation only)

    Args:
        x: 3d array to apply tfce to

    Returns:
        tfce enhanced image

    Raises:
        RuntimeError: if fsl is not available
    """
    if not _fsl_available:
        raise RuntimeError(f'fslmaths not found at: {fslmaths_path}')

    # get input / output files
    f = tempfile.NamedTemporaryFile(suffix='.nii.gz').name
    f_x_in = f.replace('.nii', '_x.nii')
    f_out = f.replace('.nii', '_x_tfce.nii')

    # write input to disk
    img = nib.Nifti1Image(x, affine=np.eye(4))
    img.to_filename(f_x_in)

    cmd = f'source {fsl_conf_sh} && {fslmaths_path} {f_x_in} -tfce 2 .5 6 {f_out}'
    proc = subprocess.run(cmd, shell=True, capture_output=True,
                          executable='/bin/bash')
    assert not proc.returncode, proc.stderr

    x_tfce = nib.load(f_out).get_fdata()

    # cleanup
    os.remove(f_out)
    os.remove(f_x_in)

    return x_tfce
