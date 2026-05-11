"""test image loading functions"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from glow.experiment.load_image import load_image_color, load_image_nii


class TestLoadImageColor:
    """test loading color images (jpg, png)"""

    def test_load_grayscale(self):
        """test loading color images (PNG with RGB channels)"""
        # use test data that exists
        test_data_dir = Path(__file__).parent.parent / 'data'

        # create dataframe with test images
        # Note: PNG images typically have RGB channels, so they'll be split
        df = pd.DataFrame({
            'img': [
                test_data_dir / 'mandrill_small.png',
                test_data_dir / 'mandrill_small.png'
            ]
        }, index=['sbj0', 'sbj1'])

        # load images
        feat_sbj_img, mask_idx = load_image_color(df)

        # PNG images with RGB are split into img0, img1, img2 (R, G, B)
        # check structure - should have at least one feature
        assert len(feat_sbj_img) > 0

        # get first feature name
        first_feat = list(feat_sbj_img.keys())[0]
        assert 'sbj0' in feat_sbj_img[first_feat]
        assert 'sbj1' in feat_sbj_img[first_feat]

        # check shapes match
        img0 = feat_sbj_img[first_feat]['sbj0']
        img1 = feat_sbj_img[first_feat]['sbj1']
        assert img0.shape == img1.shape
        assert img0.ndim == 2  # each channel is 2D

        # check mask_idx
        assert mask_idx.shape == img0.shape
        assert (mask_idx >= -1).all()

    def test_load_rgb(self):
        """test loading RGB images"""
        test_data_dir = Path(__file__).parent.parent / 'data'

        # use jpg images which are RGB
        df = pd.DataFrame({
            'img': [
                test_data_dir / 'img0_feat0.jpg',
                test_data_dir / 'img0_feat1.jpg'
            ]
        }, index=['sbj0', 'sbj1'])

        # load images
        feat_sbj_img, mask_idx = load_image_color(df)

        # RGB images should be split into separate features
        # should have img0, img1, img2 (R, G, B channels)
        assert 'img0' in feat_sbj_img or 'img' in feat_sbj_img

        # check that we got images for both subjects
        first_feat = list(feat_sbj_img.keys())[0]
        assert 'sbj0' in feat_sbj_img[first_feat]
        assert 'sbj1' in feat_sbj_img[first_feat]

        # check all images have same shape
        shapes = set()
        for feat, sbj_img in feat_sbj_img.items():
            for sbj, img in sbj_img.items():
                shapes.add(img.shape)
        assert len(shapes) == 1  # all same shape

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

    def test_nifti_mask_excludes_missing_voxels(self):
        """test that mask excludes voxels where any subject has zero"""
        test_data_dir = Path(__file__).parent.parent / 'data'

        # load nifti images
        df = pd.DataFrame({
            'feat0': [
                test_data_dir / 'img0_feat0.nii.gz',
                test_data_dir / 'img1_feat0.nii.gz'
            ]
        }, index=['img0', 'img1'])

        y, y_names, mask_idx, _affine = load_image_nii(df)

        # all values in y are within the mask, so all are non-zero by
        # construction (mask = vox_count == df.size).
        assert (y != 0).all()
        assert (mask_idx > -1).sum() == y.shape[2]
