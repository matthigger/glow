"""test effect class"""

import numpy as np
import pytest

import glow


class TestEffect:
    """test Effect class methods"""
    
    def test_from_x_y_contrast(self):
        """test creating effect from x, y, contrast"""
        # create simple data
        x = np.random.randn(2, 10)  # 2 features, 10 images
        y = np.random.randn(3, 10, 5)  # 3 imaging features, 10 images, 5 voxels
        contrast = np.array([True, False])
        mask = np.ones((10, 10), dtype=bool)
        mask[:5, :5] = False  # exclude some voxels
        
        # create effect
        eff = glow.effect.Effect.from_x_y_contrast(
            x=x,
            y=y,
            contrast=contrast,
            mask=mask
        )
        
        # check attributes
        assert eff.y_mean.shape == (3, 10)
        assert np.array_equal(eff.mask, mask)
        assert hasattr(eff, 'e')
        assert hasattr(eff, 'h')
    
    def test_from_exp_mask(self):
        """test creating effect from experiment and mask"""
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
        
        # create effect
        eff = glow.effect.Effect.from_exp_mask(exp=exp, mask=mask)
        
        # check attributes
        assert np.array_equal(eff.mask, mask)
        assert eff.y_mean.shape[0] == exp.y.shape[0]
        assert eff.y_mean.shape[1] == exp.y.shape[1]
        assert hasattr(eff, 'e')
        assert hasattr(eff, 'h')
    
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
        
        eff1 = glow.effect.Effect.from_exp_mask(exp=exp, mask=mask)
        eff2 = glow.effect.Effect.from_exp_mask(exp=exp, mask=mask)
        
        # should be identical
        assert eff1.is_close(eff2)
    
    def test_is_close_different_mask_shape(self):
        """test is_close with different mask shapes"""
        exp = glow.experiment.Experiment.from_gauss(
            seed=0,
            shape=(10, 10),
            a=2,
            b=2,
            num_img=10
        )
        
        mask1 = np.zeros((10, 10), dtype=bool)
        mask1[3:7, 3:7] = True
        mask1 = np.logical_and(mask1, exp.mask_idx > -1)
        
        mask2 = np.zeros((8, 8), dtype=bool)
        mask2[2:6, 2:6] = True
        
        eff1 = glow.effect.Effect.from_exp_mask(exp=exp, mask=mask1)
        
        # manually create effect with different shape
        y_mean = np.random.randn(2, 10)
        eff2 = glow.effect.Effect(mask=mask2, y_mean=y_mean)
        
        # should not be close (different shapes)
        assert not eff1.is_close(eff2)
    
    def test_is_close_different_mask_values(self):
        """test is_close with different mask values"""
        exp = glow.experiment.Experiment.from_gauss(
            seed=0,
            shape=(10, 10),
            a=2,
            b=2,
            num_img=10
        )
        
        mask1 = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask1[3:7, 3:7] = True
        mask1 = np.logical_and(mask1, exp.mask_idx > -1)
        
        mask2 = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask2[4:8, 4:8] = True  # different location
        mask2 = np.logical_and(mask2, exp.mask_idx > -1)
        
        eff1 = glow.effect.Effect.from_exp_mask(exp=exp, mask=mask1)
        eff2 = glow.effect.Effect.from_exp_mask(exp=exp, mask=mask2)
        
        # should not be close (different masks)
        assert not eff1.is_close(eff2)
    
    def test_is_close_different_y_mean_shape(self):
        """test is_close with different y_mean shapes"""
        mask = np.ones((10, 10), dtype=bool)
        y_mean1 = np.random.randn(2, 10)
        y_mean2 = np.random.randn(3, 10)  # different number of features
        
        eff1 = glow.effect.Effect(mask=mask, y_mean=y_mean1)
        eff2 = glow.effect.Effect(mask=mask, y_mean=y_mean2)
        
        # should not be close (different y_mean shapes)
        assert not eff1.is_close(eff2)
    
    def test_is_close_different_y_mean_values(self):
        """test is_close with different y_mean values"""
        mask = np.ones((10, 10), dtype=bool)
        y_mean1 = np.random.randn(2, 10)
        y_mean2 = y_mean1 + 10  # very different values
        
        eff1 = glow.effect.Effect(mask=mask, y_mean=y_mean1)
        eff2 = glow.effect.Effect(mask=mask, y_mean=y_mean2)
        
        # should not be close (different y_mean values)
        assert not eff1.is_close(eff2)
    
    def test_is_close_tolerance(self):
        """test is_close with tolerance parameters"""
        mask = np.ones((10, 10), dtype=bool)
        y_mean1 = np.random.randn(2, 10)
        y_mean2 = y_mean1 + 1e-7  # very small difference
        
        eff1 = glow.effect.Effect(mask=mask, y_mean=y_mean1)
        eff2 = glow.effect.Effect(mask=mask, y_mean=y_mean2)
        
        # should be close with default tolerance
        assert eff1.is_close(eff2)
        
        # should not be close with very strict tolerance
        assert not eff1.is_close(eff2, rtol=1e-10, atol=1e-10)
    
    def test_extra_kwargs_storage(self):
        """test that extra kwargs are stored as attributes"""
        mask = np.ones((10, 10), dtype=bool)
        y_mean = np.random.randn(2, 10)
        
        eff = glow.effect.Effect(
            mask=mask,
            y_mean=y_mean,
            seed=42,
            hotel_tr=10.5,
            custom_attr='test'
        )
        
        # check that extra kwargs are stored
        assert eff.seed == 42
        assert eff.hotel_tr == 10.5
        assert eff.custom_attr == 'test'
