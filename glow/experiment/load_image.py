"""Image loaders: NIfTI and color (PNG/JPG) into masked y arrays."""

from collections import defaultdict

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

import glow.mask
from glow.mask import get_mask_idx


def load_image_nii(df: pd.DataFrame, dtype=np.float32, mask=None):
    """Load NIfTI images from a subject x feature dataframe.

    Streams images in two passes so peak memory is one image rather than
    the full (b, num_sbj) stack.  Pass 1 walks every file to validate the
    shared affine (and, with no explicit mask, to accumulate a
    nonzero-voxel count); pass 2 masks each image into y and discards it.

    The analysis support is either supplied (a brain-mask NIfTI) or
    inferred.  When inferred, a voxel is kept only where every image is
    nonzero -- a crude proxy for "in brain" that fails for maps that are
    legitimately zero inside the brain (e.g. NODDI isovf in dense tissue),
    so callers with a real brain mask should pass it.

    Args:
        df (pd.DataFrame): index=subject, columns=feature, values=file paths
        dtype: numpy dtype for the output y array.  Default np.float32
            halves memory versus float64 and keeps the LLR hot loop in
            float32; nibabel preserves precision when the on-disk type is
            itself float32.
        mask (path): optional path to a brain-mask NIfTI on the images'
            grid.  When given, its nonzero voxels are the analysis support
            (its affine must match the images'); when None the
            every-image-nonzero rule above is used instead.

    Returns:
        y (np.array): (b, num_sbj, num_vox) masked image intensities.
            Subject axis is ordered by sorted(df.index); feature axis
            follows df.columns.  Dtype matches the dtype argument.
        y_names (list): feature names, in df.columns order
        mask_idx (np.array): voxel index array (-1 outside the support)
        affine (np.array): (4, 4) NIfTI affine (consistent across all images)
    """
    y_names = list(df.columns)
    subjects = sorted(df.index)

    # ---- Pass 1: scan all files, check affine; count nonzeros only when
    # the support must be inferred (no explicit mask)
    affine = None
    img_shape = None
    vox_count = None
    for feat in df.columns:
        for sbj in df.index:
            file = df.loc[sbj, feat]

            img = nib.load(file)
            if affine is None:
                affine = img.affine
                img_shape = img.shape
            assert np.array_equal(img.affine,
                                  affine), 'affine mismatch'

            if mask is None:
                arr = img.get_fdata(dtype=dtype)
                if vox_count is None:
                    vox_count = np.zeros(arr.shape, dtype=np.int32)
                vox_count += arr != 0
                del arr
            del img

    if mask is None:
        # inferred support: drop any voxel zero in some subject / feature
        mask = vox_count == df.size
        del vox_count
    else:
        # explicit brain-mask NIfTI: must sit on the images' grid
        mask_img = nib.load(mask)
        assert np.array_equal(mask_img.affine, affine), 'mask affine mismatch'
        mask = mask_img.get_fdata() > 0
        assert mask.shape == img_shape, \
            f'mask shape {mask.shape} != image shape {img_shape}'
    mask_idx = glow.mask.get_mask_idx(mask)

    # ---- Pass 2: reload each file, mask into y, discard
    num_vox = int(mask.sum())
    y = np.empty((len(y_names), len(subjects), num_vox), dtype=dtype)
    sbj_to_idx = {sbj: idx for idx, sbj in enumerate(subjects)}
    for feat_idx, feat in enumerate(y_names):
        for sbj in df.index:
            file = df.loc[sbj, feat]

            img = nib.load(file)
            arr = img.get_fdata(dtype=dtype)
            y[feat_idx, sbj_to_idx[sbj], :] = arr[mask]
            del arr, img

    return y, y_names, mask_idx, affine


def load_image_color(df: pd.DataFrame, channel_names: dict = None):
    """Load non-NIfTI (e.g. PNG) images from a subject x feature dataframe.

    Args:
        df (pd.DataFrame): index=subject, columns=feature, values=file paths
        channel_names (dict): optional {feature: [name0, name1, ...]}
            overriding the default feat0/feat1/... naming for
            multi-channel images (e.g. {'rgb': ['red', 'green', 'blue']}).
            Lengths shorter than the channel count fall back to default
            naming for the remaining channels.

    Returns:
        feat_sbj_img (dict): feat -> sbj -> np.array
        mask_idx (np.array): voxel index array (all active)
    """
    channel_names = channel_names or {}
    shape = None
    dtype = None

    def check_shape_type(x, shape, dtype):
        """Validate x against the running shape/dtype, return its own."""
        if shape is not None and x.shape != shape:
            raise RuntimeError('images dont have consistent shapes')
        if dtype is not None and x.dtype != dtype:
            raise RuntimeError('images dont have same data type')
        return x.shape, x.dtype

    feat_sbj_img = defaultdict(dict)
    for feat in df.columns:
        for sbj in df.index:
            file = df.loc[sbj, feat]

            x = np.array(Image.open(file))
            if x.ndim == 2:
                # grayscale: single feature
                feat_sbj_img[feat][sbj] = x
                shape, dtype = check_shape_type(x, shape, dtype)
            elif x.ndim == 3:
                # multi-channel (e.g. RGB or RGBA): one feature per channel
                names_for_feat = channel_names.get(feat, [])
                for idx in range(x.shape[2]):
                    if idx < len(names_for_feat):
                        _feat = names_for_feat[idx]
                    else:
                        _feat = feat + str(idx)
                    _x = x[:, :, idx]
                    feat_sbj_img[_feat][sbj] = _x
                    shape, dtype = check_shape_type(_x, shape, dtype)
            else:
                msg = 'non nifti must be 2d / 3d (3rd is rgb color)'
                raise RuntimeError(msg)

    mask_idx = get_mask_idx(np.ones(shape))

    return feat_sbj_img, mask_idx
