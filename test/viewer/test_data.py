"""Tests for glow.viewer.data (DataFrame preparation and feature columns)."""

import numpy as np
import pandas as pd

from glow.viewer.data import (prep_df, get_feature_columns,
                               compute_target_stats, compute_backgrounds)


class TestPrepDf:
    def test_shape(self, ana, df_no_target):
        num_vox = ana.exp.y.shape[2]
        num_reg = num_vox + ana.children.shape[0]
        assert len(df_no_target) == num_reg

    def test_required_columns(self, df_no_target):
        for col in ('region_idx', 'n_voxel', 'llr', 'llr_z',
                     'pval_fwer', 'significant', 'discovered',
                     'estimate_state'):
            assert col in df_no_target.columns, f'missing column: {col}'

    def test_region_idx_contiguous(self, ana, df_no_target):
        num_vox = ana.exp.y.shape[2]
        num_reg = num_vox + ana.children.shape[0]
        assert set(df_no_target['region_idx']) == set(range(num_reg))

    def test_n_voxel_positive(self, df_no_target):
        assert (df_no_target['n_voxel'] > 0).all()

    def test_leaves_have_size_one(self, ana, df_no_target):
        num_vox = ana.exp.y.shape[2]
        leaves = df_no_target[df_no_target['region_idx'] < num_vox]
        assert (leaves['n_voxel'] == 1).all()

    def test_estimate_state_values(self, df_no_target):
        assert set(df_no_target['estimate_state'].unique()) <= {
            'no_effect', 'has_effect'}

    def test_mask_target_columns(self, df_with_target):
        for col in ('dice', 'sens', 'ppv', 'spec', 'vox_in_target',
                    'vox_out_target', 'vox_target_missed', 'vox_outside_both'):
            assert col in df_with_target.columns, f'missing column: {col}'

    def test_dice_range(self, df_with_target):
        dice = df_with_target['dice']
        assert (dice >= 0).all() and (dice <= 1).all()

    def test_extra_df_merge(self, ana):
        num_reg = ana.exp.y.shape[2] + ana.children.shape[0]
        extra = pd.DataFrame({
            'region_idx': [0, 1, 2],
            'custom_metric': [10.0, 20.0, 30.0],
        })
        df = prep_df(ana, extra_df=extra)
        assert 'custom_metric' in df.columns
        assert len(df) == num_reg
        assert df.loc[df['region_idx'] == 0, 'custom_metric'].values[0] == 10.0


class TestGetFeatureColumns:
    def test_returns_four_lists(self, df_no_target):
        result = get_feature_columns(df_no_target)
        assert len(result) == 4
        generic, sig, prune, mask = result
        assert isinstance(generic, list)
        assert isinstance(sig, list)

    def test_n_voxel_in_generic(self, df_no_target):
        generic, _, _, _ = get_feature_columns(df_no_target)
        assert 'n_voxel' in generic

    def test_llr_in_significance(self, df_no_target):
        _, sig, _, _ = get_feature_columns(df_no_target)
        assert 'llr' in sig
        assert 'llr_z' in sig

    def test_mask_columns_present_with_target(self, df_with_target):
        _, _, _, mask_cols = get_feature_columns(df_with_target)
        assert 'dice' in mask_cols

    def test_excludes_boolean_and_index(self, df_no_target):
        generic, sig, prune, mask = get_feature_columns(df_no_target)
        all_cols = generic + sig + prune + mask
        for excluded in ('region_idx', 'discovered', 'significant',
                         'estimate_state'):
            assert excluded not in all_cols


class TestComputeTargetStats:
    def test_returns_dict(self, target_stats):
        assert isinstance(target_stats, dict)

    def test_required_keys(self, target_stats):
        for key in ('n_voxel', 'llr', 'llr_z'):
            assert key in target_stats

    def test_n_voxel_matches_mask(self, ana, mask_target, target_stats):
        mask_idx = ana.exp.mask_idx
        expected = (mask_idx[mask_target & (mask_idx >= 0)]).size
        assert target_stats['n_voxel'] == expected

    def test_self_metrics(self, target_stats):
        assert target_stats['dice'] == 1.0
        assert target_stats['sens'] == 1.0
        assert target_stats['ppv'] == 1.0
        assert target_stats['spec'] == 1.0

    def test_empty_mask_returns_none(self, ana):
        empty_mask = np.zeros_like(ana.exp.mask_idx, dtype=bool)
        assert compute_target_stats(ana, empty_mask) is None


class TestComputeBackgrounds:
    """Tests for compute_backgrounds including per-image selection."""

    def test_mean_returns_dict(self, ana):
        bg = compute_backgrounds(ana)
        assert isinstance(bg, dict)
        assert len(bg) >= 1

    def test_mean_shapes_match_mask(self, ana):
        bg = compute_backgrounds(ana)
        mask_shape = ana.exp.mask_idx.shape
        for name, img in bg.items():
            assert img.shape[:len(mask_shape)] == mask_shape

    def test_single_image_returns_dict(self, ana):
        bg = compute_backgrounds(ana, image_idx=0)
        assert isinstance(bg, dict)
        assert len(bg) >= 1

    def test_single_image_differs_from_mean(self, ana):
        bg_mean = compute_backgrounds(ana)
        bg_0 = compute_backgrounds(ana, image_idx=0)
        key = list(bg_mean.keys())[0]
        if key == 'RGB':
            key = list(bg_mean.keys())[1] if len(bg_mean) > 1 else key
        valid = ~np.isnan(bg_mean[key])
        if valid.any():
            assert not np.allclose(bg_mean[key][valid], bg_0[key][valid]), \
                'single image should generally differ from the mean'

    def test_all_image_indices_valid(self, ana):
        num_img = ana.exp.y.shape[1]
        for i in range(num_img):
            bg = compute_backgrounds(ana, image_idx=i)
            assert isinstance(bg, dict)

    def test_y_features_used_as_keys(self, ana):
        names = [f'feat_{i}' for i in range(ana.exp.y.shape[0])]
        bg = compute_backgrounds(ana, y_features=names)
        for n in names:
            assert n in bg
