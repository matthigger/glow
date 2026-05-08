from collections import defaultdict

import nibabel as nib
import numpy as np
from PIL import Image

import glow.mask
from glow.mask import get_mask_idx


def load_image_nii(df):
    """load NIfTI images from a subject x feature dataframe.

    Args:
        df (pd.DataFrame): index=subject, columns=feature, values=file paths

    Returns:
        feat_sbj_img (dict): feat -> sbj -> np.array
        mask_idx (np.array): voxel index array (-1 where any image is zero)
        affine (np.array): (4, 4) NIfTI affine (consistent across all images)
    """
    affine = None
    feat_sbj_img = defaultdict(dict)
    for feat in df.columns:
        for sbj in df.index:
            file = df.loc[sbj, feat]

            img = nib.load(file)
            if affine is None:
                affine = img.affine
            assert np.array_equal(img.affine,
                                  affine), 'affine mismatch'
            feat_sbj_img[feat][sbj] = img.get_fdata()

    # count nonzero voxels per position (also check images have same shape)
    vox_count = None
    for feat, sbj_img in feat_sbj_img.items():
        for sbj, img in sbj_img.items():
            if vox_count is None:
                vox_count = np.zeros(img.shape)
            vox_count += img != 0

    # build mask_idx (exclude any voxel which any subject is missing)
    mask = vox_count == df.size
    mask_idx = glow.mask.get_mask_idx(mask)

    return feat_sbj_img, mask_idx, affine


def load_image_color(df, channel_names=None):
    """load non-NIfTI (e.g. PNG) images from a subject x feature dataframe.

    Args:
        df (pd.DataFrame): index=subject, columns=feature, values=file paths
        channel_names (dict, optional): ``{feature: [name0, name1, ...]}``
            overriding the default ``feat0``/``feat1``/... naming for
            multi-channel images (e.g. ``{'rgb': ['red', 'green',
            'blue']}``).  Lengths shorter than the channel count fall
            back to default naming for the remaining channels.

    Returns:
        feat_sbj_img (dict): feat -> sbj -> np.array
        mask_idx (np.array): voxel index array (all active)
    """
    channel_names = channel_names or {}
    shape = None
    dtype = None

    def check_shape_type(x, shape, dtype):
        if shape is not None and x.shape != shape:
            raise RuntimeError('images dont have consistent shapes')
        if dtype is not None and x.dtype != dtype:
            raise RuntimeError('images dont have same data type')
        return x.shape, x.dtype

    feat_sbj_img = defaultdict(dict)
    for feat in df.columns:
        for sbj in df.index:
            file = df.loc[sbj, feat]

            # load non nifti image
            x = np.array(Image.open(file))
            if x.ndim == 2:
                # image has 1 feature (grayscale)
                feat_sbj_img[feat][sbj] = x

                # ensure consistent shape
                shape, dtype = check_shape_type(x, shape, dtype)
            elif x.ndim == 3:
                # image has multiple features (e.g. RGB or RGBA)
                names_for_feat = channel_names.get(feat, [])
                for idx in range(x.shape[2]):
                    if idx < len(names_for_feat):
                        _feat = names_for_feat[idx]
                    else:
                        _feat = feat + str(idx)
                    _x = x[:, :, idx]
                    feat_sbj_img[_feat][sbj] = _x

                    # ensure consistent shape
                    shape, dtype = check_shape_type(_x, shape, dtype)
            else:
                msg = 'non nifti must be 2d / 3d (3rd is rgb color)'
                raise RuntimeError(msg)

    # build mask_idx
    mask_idx = get_mask_idx(np.ones(shape))

    return feat_sbj_img, mask_idx
