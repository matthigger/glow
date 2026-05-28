import numpy as np
import pytest

from glow.effect.extent import (
    ContiguousRegionNotFound,
    ExtenterMinVar,
    ExtenterSphere,
    iter_vox_neighbor,
)
from glow.mask import get_mask_idx


def test_iter_vox_neighbor_2d_single_seed():
    # 3x3 grid; row 0 is outside the analysis mask (-1):
    #   [[-1, -1, -1],
    #    [ 3,  4,  5],
    #    [ 6,  7,  8]]
    # seed the single voxel at (2, 0) (index 6). Its in-mask face-neighbours
    # are (1, 0)=3 and (2, 1)=7; the (3, 0) neighbour is off-grid.
    mask_idx = np.array([[-1, -1, -1],
                         [3, 4, 5],
                         [6, 7, 8]])
    mask = np.zeros_like(mask_idx)
    mask[2, 0] = 1
    assert [3, 7] == sorted(iter_vox_neighbor(mask, mask_idx))


def test_iter_vox_neighbor_2d_multi_seed():
    # same grid; seed voxels at (2,0)=6, (1,1)=4 and (2,1)=7.
    # face-neighbours outside the seeded region (and in-mask) are
    # (1,0)=3, (1,2)=5 and (2,2)=8.
    mask_idx = np.array([[-1, -1, -1],
                         [3, 4, 5],
                         [6, 7, 8]])
    mask = np.zeros_like(mask_idx)
    mask[2, 0] = 1
    mask[1, 1] = 1
    mask[2, 1] = 1
    assert [3, 5, 8] == sorted(iter_vox_neighbor(mask, mask_idx))


def test_iter_vox_neighbor_3d_single_seed():
    # 3x3x3 grid indexed 0..26; seed the centre voxel (1,1,1)=13.
    # its six face-neighbours are 4, 10, 12, 14, 16, 22 (no diagonals,
    # matching 6-connectivity).
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

    def test_contiguous_unreachable_raises(self):
        # two disconnected rows: any voxel selected, dilated 4 times, then
        # re-masked spans both rows and so is never contiguous.
        mask = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [1, 1, 1]])
        mask_idx = get_mask_idx(mask)

        extenter = ExtenterSphere(radius=4)

        with pytest.raises(ContiguousRegionNotFound):
            extenter(mask_idx=mask_idx, seed=0, contiguous=True)

    def test_contiguous_reachable_succeeds(self):
        # fully-connected grid: the produced region is contiguous, so the
        # contiguous=True request succeeds and fills the grid.
        extenter = ExtenterSphere(radius=4)
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

    def test_random_init_reaches_n_vox(self):
        # without vox_init the seed voxel is chosen randomly; the grown
        # region must still reach the requested size and grid shape.
        mask_idx = np.arange(64).reshape((8, 8))
        extenter = ExtenterMinVar(n_vox=10)
        rng = np.random.default_rng(42)
        y = rng.standard_normal((1, 1, 64))
        mask = extenter(mask_idx=mask_idx, y=y, seed=42)
        assert mask.sum() == 10
        assert mask.shape == mask_idx.shape
