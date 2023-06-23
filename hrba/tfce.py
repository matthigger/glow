import os
import pathlib
import subprocess
import tempfile
import warnings
from shutil import which

import nibabel as nib
import numpy as np

fsl_path = pathlib.Path('/home/matt/fsl')
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
