"""test image loading functions"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from PIL import Image

from glow.experiment.load_image import load_image_color, load_image_nii


class TestLoadImageColor:
    """test loading color images (jpg, png)"""

    @pytest.mark.parametrize('rgb_png', ['mandrill_small.png',
                                         'squares_test.png'])
    def test_rgb_splits_into_per_channel_features(self, rgb_png):
        """RGB images split into per-channel (R, G, B) features for all sbj"""
        test_data_dir = Path(__file__).parent.parent / 'data'

        # same RGB PNG for two subjects
        path = test_data_dir / rgb_png
        df = pd.DataFrame({'img': [path, path]}, index=['sbj0', 'sbj1'])

        feat_sbj_img, mask_idx = load_image_color(df)

        # the 'img' column is a 3-channel RGB PNG, so it is split into one
        # feature per channel: img0 (R), img1 (G), img2 (B)
        assert list(feat_sbj_img.keys()) == ['img0', 'img1', 'img2']

        # every channel feature carries both subjects, as 2D arrays of the
        # image's spatial shape
        with Image.open(path) as im:
            spatial_shape = (im.height, im.width)
        for feat in ['img0', 'img1', 'img2']:
            assert set(feat_sbj_img[feat]) == {'sbj0', 'sbj1'}
            for sbj in ['sbj0', 'sbj1']:
                ch = feat_sbj_img[feat][sbj]
                assert ch.ndim == 2
                assert ch.shape == spatial_shape

        # mask_idx covers the full image (color loader masks nothing)
        assert mask_idx.shape == spatial_shape
        assert (mask_idx >= -1).all()

        # known top-left pixel: the per-channel split must preserve the
        # source channel intensities at that location
        rgb = np.array(Image.open(path))
        for ch_idx, feat in enumerate(['img0', 'img1', 'img2']):
            assert feat_sbj_img[feat]['sbj0'][0, 0] == rgb[0, 0, ch_idx]

    def test_shape_consistency_error(self):
        """test that inconsistent shapes raise error"""
        test_data_dir = Path(__file__).parent.parent / 'data'

        # try to load images with different shapes
        df = pd.DataFrame({
            'img': [
                test_data_dir / 'mandrill.png',  # larger
                test_data_dir / 'mandrill_small.png'  # smaller
            ]
        }, index=['sbj0', 'sbj1'])

        # should raise RuntimeError about inconsistent shapes
        with pytest.raises(RuntimeError, match='consistent shapes'):
            load_image_color(df)


class TestLoadImageNii:
    """test loading NIfTI images"""

    def test_load_nifti(self):
        """test loading nifti images"""
        test_data_dir = Path(__file__).parent.parent / 'data'

        # create dataframe with nifti test images
        df = pd.DataFrame({
            'feat0': [
                test_data_dir / 'img0_feat0.nii.gz',
                test_data_dir / 'img1_feat0.nii.gz',
                test_data_dir / 'img2_feat0.nii.gz'
            ],
            'feat1': [
                test_data_dir / 'img0_feat1.nii.gz',
                test_data_dir / 'img1_feat1.nii.gz',
                test_data_dir / 'img2_feat1.nii.gz'
            ]
        }, index=['img0', 'img1', 'img2'])

        # new interface: y, y_names, mask_idx, affine
        y, y_names, mask_idx, affine = load_image_nii(df)

        # y shape: (num_feat, num_sbj, num_vox_in_mask)
        num_vox = int((mask_idx > -1).sum())
        assert y.shape == (2, 3, num_vox)

        # feature ordering preserved from df.columns
        assert y_names == ['feat0', 'feat1']

        # mask_idx is the index map (-1 outside)
        assert (mask_idx >= -1).all()
        assert (mask_idx > -1).any()  # at least some voxels included

        # affine is a 4x4 array
        assert affine.shape == (4, 4)

    def test_nifti_mask_excludes_missing_voxels(self, tmp_path):
        """mask drops any voxel that is zero in at least one subject"""
        import nibabel as nib

        affine = np.eye(4)
        # two subjects on an 8-voxel grid.  Zero a single voxel in only
        # subject 0; it should be excluded from the mask even though
        # subject 1 has data there (mask = vox_count == df.size).
        arr0 = np.ones((2, 2, 2))
        arr0[0, 0, 0] = 0
        arr1 = np.full((2, 2, 2), 2.0)

        df = pd.DataFrame({'feat0': []}, dtype=object)
        for sbj, arr in [('img0', arr0), ('img1', arr1)]:
            file = tmp_path / f'{sbj}_feat0.nii.gz'
            nib.Nifti2Image(arr, affine=affine).to_filename(file)
            df.loc[sbj, 'feat0'] = file

        y, y_names, mask_idx, _affine = load_image_nii(df)

        # exactly the one deliberately-zeroed voxel is dropped (7 of 8)
        assert mask_idx[0, 0, 0] == -1
        assert int((mask_idx > -1).sum()) == 7
        assert y.shape == (1, 2, 7)
        # surviving voxels are non-zero for every subject
        assert (y != 0).all()
