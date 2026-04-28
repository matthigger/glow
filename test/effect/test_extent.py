import pytest

from glow.effect.extent import *
from glow.mask import get_mask_idx


def test_iter_vox_neighbor():
    mask_idx = np.array([[-1, -1, -1],
                         [3, 4, 5],
                         [6, 7, 8]])
    mask = np.zeros_like(mask_idx)
    mask[2, 0] = 1
    assert [3, 7] == sorted(iter_vox_neighbor(mask, mask_idx))

    mask[1, 1] = 1
    mask[2, 1] = 1
    assert [3, 5, 8] == sorted(iter_vox_neighbor(mask, mask_idx))

    mask_idx = np.arange(27).reshape((3, 3, 3))
    mask = np.zeros_like(mask_idx)
    mask[1, 1, 1] = 1
    assert [4, 10, 12, 14, 16, 22] == sorted(iter_vox_neighbor(mask, mask_idx))


class TestExtenterSphere:
    def test_call(self):
        mask_idx = np.arange(64).reshape((8, 8))
        mask_idx[:2, :] = -1

        mask_expect_tup = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 0, 0, 0, 0]]), \
            np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1, 1, 0, 0],
                      [1, 1, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1]])

        for radius, mask_expect in zip((3, 10), mask_expect_tup):
            extenter = ExtenterSphere(radius=radius)
            mask = extenter(mask_idx=mask_idx, seed=0)

            assert np.allclose(mask, mask_expect)

    def test_contiguous(self):
        mask = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [1, 1, 1]])
        mask_idx = get_mask_idx(mask)

        extenter = ExtenterSphere(radius=4)

        with pytest.raises(ContiguousRegionNotFound):
            # any voxel selected (from 6 below), dilated 4 times, and then masked
            # again will not be contiguous
            extenter(mask_idx=mask_idx, seed=0, contiguous=True)

        # any region produced (there is only one really) would be contiguous
        mask_idx = np.arange(9).reshape(3, 3)
        mask = extenter(mask_idx=mask_idx, seed=0, contiguous=True)
        assert np.allclose(np.ones((3, 3)), mask)

    def test_n_vox(self):
        mask_idx = np.arange(25).reshape((5, 5))
        extenter = ExtenterSphere(n_vox=7)
        mask = extenter(mask_idx=mask_idx, vox_init=12)
        assert mask.sum() == 7
        assert mask.shape == mask_idx.shape

    def test_connected_component_restrict(self):
        mask = np.array([[1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1]])
        mask_idx = get_mask_idx(mask)
        extenter = ExtenterSphere(n_vox=5, connected=True)
        vox_init = mask_idx[0, 0]
        mask_obs = extenter(mask_idx=mask_idx, vox_init=vox_init)

        comp_mask = np.zeros_like(mask, dtype=bool)
        comp_mask[0, :] = True
        assert mask_obs.sum() == 5
        assert np.all(mask_obs[~comp_mask] == 0)

    def test_connected_component_size_error(self):
        mask = np.array([[1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 1, 1]])
        mask_idx = get_mask_idx(mask)
        extenter = ExtenterSphere(n_vox=5, connected=True)
        with pytest.raises(ValueError):
            extenter(mask_idx=mask_idx, seed=0)


class TestExtenterMinVar:
    def test_call(self):
        mask = np.array([[0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 1., 1., 1., 0., 0.],
                         [0., 0., 1., 1., 1., 0., 0.],
                         [0., 0., 1., 1., 1., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0.]])
        mask_idx = np.arange(mask.size).reshape(mask.shape)

        extenter_min_var = ExtenterMinVar(n_vox=mask.sum())

        # test case 1: no noise, single image
        mask_obs = extenter_min_var(y=mask.reshape((1, 1, mask.size)),
                                    mask_idx=mask_idx, vox_init=17)
        np.testing.assert_equal(mask_obs, mask)

        # test case 2: noise, multi-image
        n_img = 10
        y = np.broadcast_to(mask.reshape((1, 1, mask.size)),
                            (1, n_img, mask.size))
        rng = np.random.default_rng(seed=0)
        y = y + rng.standard_normal(y.shape) / 10
        mask_obs = extenter_min_var(y=y, mask_idx=mask_idx, vox_init=17)
        np.testing.assert_equal(mask_obs, mask)


class TestExtenterMinVarRandomInit:
    """test ExtenterMinVar with random initial voxel selection"""
    
    def test_minvar_without_vox_init(self):
        """test that ExtenterMinVar works without specifying vox_init"""
        mask_idx = np.arange(64).reshape((8, 8))
        
        # create extenter that grows to size 10
        extenter = ExtenterMinVar(n_vox=10)
        
        # create simple y data (1 feature, 1 image, 64 voxels)
        y = np.random.standard_normal((1, 1, 64))
        
        # call without vox_init (should select random voxel internally, lines 119-121)
        mask = extenter(mask_idx=mask_idx, y=y, seed=42)
        
        # verify mask has correct size
        assert mask.sum() == 10
        
        # verify mask is within bounds
        assert mask.shape == mask_idx.shape
    
    def test_minvar_with_different_seeds(self):
        """test that different seeds produce different results"""
        mask_idx = np.arange(64).reshape((8, 8))
        extenter = ExtenterMinVar(n_vox=10)

        rng = np.random.default_rng(42)
        y = rng.standard_normal((1, 1, 64))
        
        mask1 = extenter(mask_idx=mask_idx, y=y, seed=0)
        mask2 = extenter(mask_idx=mask_idx, y=y, seed=1)
        
        # different seeds should produce different masks
        assert not np.array_equal(mask1, mask2)
        
        # but both should have correct size
        assert mask1.sum() == 10
        assert mask2.sum() == 10
