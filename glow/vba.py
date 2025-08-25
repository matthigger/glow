import os
import pathlib
import subprocess
import tempfile
import warnings
from collections import Counter
from shutil import which

import nibabel as nib
import numpy as np
from sklearn.metrics import f1_score

# to be modified per installation
fsl_path = pathlib.Path('/usr/local/fsl')

fsl_conf_sh = fsl_path / 'etc/fslconf/fsl.sh'
fslmaths_path = fsl_path / 'bin/fslmaths'

if not which(str(fslmaths_path)):
    warnings.warn(f'fslmaths not found at: {fslmaths_path}')


def apply_tfce_x(x, mask_idx):
    # zero pad (TFCE requires border of zeros)
    mask_idx = np.pad(np.atleast_3d(mask_idx), pad_width=1, constant_values=-1)

    # reshape into original image dimensions
    mask_bool = mask_idx > -1
    img = np.zeros(mask_idx.shape)
    img[mask_bool] = x

    # apply tfce
    img_tfce = apply_tfce_img(img)

    # extract tfce stats
    return img_tfce[mask_bool]


def apply_tfce_img(x):
    """ applies tfce to an array (must be 3d)

    Args:
        x (np.array): a 3d array to apply tfce to
    """

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


def optimize_cluster_thresh(mask_est, mask_true, mask_active=None):
    """ optimizes a cluster volume threshold to maximize F1 score

    note: this shouldn't ever be used in practice (no access to ground
    truth!) but its a useful upper bound of cluster extent thresholding for
    analysis

    Args:
        mask_est (np.array): output of scipy.ndimage.label of estimated regions
        mask_true (np.array): ground-truth binary mask (0 = no effect, 1 = true
            effect region)
        mask_active (np.array): voxels outside of mask_active excluded from
            analysis

    Returns:
        thresh_size (int): cluster volume threshold (in voxels) that maximizes
            the F1 score.
        mask_est_prune (np.array): binary mask where regions smaller
            than the best threshold are zeroed out
        f1 (float): f1 score achieved
    """
    # cast type / copy (mask_est may be written on)
    _mask_est = mask_est.copy()
    mask_true = mask_true.astype(bool)
    assert mask_est.shape == mask_true.shape

    if mask_active is not None:
        # apply active mask
        assert mask_active.shape == mask_est.shape
        _mask_est = mask_est[mask_active]
        mask_true = mask_true[mask_active]
    else:
        _mask_est = mask_est.flatten()
        mask_true = mask_true.flatten()

    # count size of each region
    reg_size = Counter(_mask_est)
    if 0 in reg_size.keys():
        # background
        del reg_size[0]

    # if regions have same size, they should be processed together
    reg_size_inv = dict()
    for reg, size in list(reg_size.items()):
        if size not in reg_size_inv:
            # unique size: just record it
            reg_size_inv[size] = reg
        else:
            # size already seen, process reg with reg_rep
            reg_rep = reg_size_inv[size]
            _mask_est[_mask_est == reg] = reg_rep
            del reg_size[reg]

            # modify mask_true too (needed to produce final output)
            mask_true[_mask_est == reg] = reg_rep

    # add regions (from smallest to largest)
    thresh_size = np.inf
    f1 = 0
    mask_est_prune = np.zeros(_mask_est.shape, dtype=bool)
    for reg_idx in sorted(reg_size, key=reg_size.get, reverse=True):
        mask_est_prune[_mask_est == reg_idx] = True
        _f1 = f1_score(y_true=mask_true, y_pred=mask_est_prune)

        if _f1 > f1:
            f1 = _f1
            thresh_size = reg_size[reg_idx]

    # build mask_est_prune
    # (we build a fresh copy of reg_size here as the one above was modified
    # for duplicates)
    mask_est_prune = np.zeros(mask_est.shape, dtype=bool)
    for reg, size in Counter(mask_est.flatten()).items():
        if reg == 0:
            continue
        if size >= thresh_size:
            mask_est_prune[mask_est == reg] = True
    if mask_active is not None:
        mask_est_prune[~mask_active] = False

    return thresh_size, mask_est_prune, f1
