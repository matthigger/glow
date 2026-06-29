"""test effect class"""

import numpy as np
import pytest

import glow
from glow.analysis.mancova import get_mancova


class TestEffect:
    """test EffectEstimate class methods"""

    def test_from_x_y_contrast(self):
        """from_x_y_contrast stores the direct MANCOVA decomposition."""
        # create simple data
        x = np.random.randn(2, 10)  # 2 features, 10 images
        y = np.random.randn(3, 10, 5)  # 3 imaging features, 10 images, 5 voxels
        contrast = np.array([True, False])
        mask = np.ones((10, 10), dtype=bool)
        mask[:5, :5] = False  # exclude some voxels

        eff = glow.effect.EffectEstimate.from_x_y_contrast(
            x=x,
            y=y,
            contrast=contrast,
            mask=mask
        )

        # e/h must match a direct decomposition; y_mean the per-voxel mean
        e, h, _ = get_mancova(x=x, y=y, contrast=contrast)
        assert np.array_equal(eff.mask, mask)
        np.testing.assert_array_equal(eff.y_mean, y.mean(axis=2))
        np.testing.assert_array_equal(eff.e, e)
        np.testing.assert_array_equal(eff.h, h)

    def test_from_exp_mask(self):
        """from_exp_mask matches from_x_y_contrast on the masked voxels."""
        # create experiment
        exp = glow.experiment.Experiment.from_gauss(
            seed=0,
            shape=(10, 10),
            a=2,
            b=2,
            num_img=10
        )

        # create mask
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[3:7, 3:7] = True
        mask = np.logical_and(mask, exp.mask_idx > -1)

        eff = glow.effect.EffectEstimate.from_exp_mask(exp=exp, mask=mask)

        # building directly from the same masked voxels must agree
        y = exp.y[..., tuple(exp.mask_idx[mask])]
        eff_direct = glow.effect.EffectEstimate.from_x_y_contrast(
            x=exp.x, y=y, contrast=exp.contrast, mask=mask)
        np.testing.assert_array_equal(eff.mask, eff_direct.mask)
        np.testing.assert_array_equal(eff.y_mean, eff_direct.y_mean)
        np.testing.assert_array_equal(eff.e, eff_direct.e)
        np.testing.assert_array_equal(eff.h, eff_direct.h)


    def test_is_close_identical(self):
        """test is_close with identical effects"""
        # create experiment and effect
        exp = glow.experiment.Experiment.from_gauss(
            seed=0,
            shape=(10, 10),
            a=2,
            b=2,
            num_img=10
        )
        
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[3:7, 3:7] = True
        mask = np.logical_and(mask, exp.mask_idx > -1)
        
        eff1 = glow.effect.EffectEstimate.from_exp_mask(exp=exp, mask=mask)
        eff2 = glow.effect.EffectEstimate.from_exp_mask(exp=exp, mask=mask)
        
        # should be identical
        assert eff1.is_close(eff2)
    
    @pytest.mark.parametrize('eff1, eff2', [
        # different mask shape
        (glow.effect.EffectEstimate(mask=np.ones((10, 10), dtype=bool),
                            y_mean=np.zeros((2, 10))),
         glow.effect.EffectEstimate(mask=np.ones((8, 8), dtype=bool),
                            y_mean=np.zeros((2, 10)))),
        # same shape, different mask values
        (glow.effect.EffectEstimate(mask=np.ones((10, 10), dtype=bool),
                            y_mean=np.zeros((2, 10))),
         glow.effect.EffectEstimate(mask=np.zeros((10, 10), dtype=bool),
                            y_mean=np.zeros((2, 10)))),
        # different y_mean shape (different number of features)
        (glow.effect.EffectEstimate(mask=np.ones((10, 10), dtype=bool),
                            y_mean=np.zeros((2, 10))),
         glow.effect.EffectEstimate(mask=np.ones((10, 10), dtype=bool),
                            y_mean=np.zeros((3, 10)))),
        # same y_mean shape, very different values
        (glow.effect.EffectEstimate(mask=np.ones((10, 10), dtype=bool),
                            y_mean=np.zeros((2, 10))),
         glow.effect.EffectEstimate(mask=np.ones((10, 10), dtype=bool),
                            y_mean=np.full((2, 10), 10.0))),
    ])
    def test_is_close_non_close_cases(self, eff1, eff2):
        """is_close is False when mask or y_mean shape/values differ."""
        assert not eff1.is_close(eff2)

    def test_is_close_tolerance(self):
        """test is_close with tolerance parameters"""
        mask = np.ones((10, 10), dtype=bool)
        rng = np.random.default_rng(seed=0)
        y_mean1 = rng.standard_normal((2, 10))
        y_mean2 = y_mean1 + 1e-7  # very small difference
        
        eff1 = glow.effect.EffectEstimate(mask=mask, y_mean=y_mean1)
        eff2 = glow.effect.EffectEstimate(mask=mask, y_mean=y_mean2)
        
        # should be close with default tolerance
        assert eff1.is_close(eff2)
        
        # should not be close with very strict tolerance
        assert not eff1.is_close(eff2, rtol=1e-10, atol=1e-10)
