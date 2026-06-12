import numpy as np
import pytest
from scipy.ndimage import generate_binary_structure, label

from glow.effect.extent import (
    ContiguousRegionNotFound,
    ExtenterMinVar,
    ExtenterSphere,
    get_diameter,
    iter_bfs,
    iter_vox_neighbor,
    split_mask,
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


def _is_contiguous(mask):
    # single connected component under the same face-connectivity the
    # extent functions use (4-conn in 2D, 6-conn in 3D)
    structure = (generate_binary_structure(3, 1) if mask.ndim == 3
                 else generate_binary_structure(2, 1))
    _, n_components = label(mask, structure=structure)
    return n_components == 1


def _assert_valid_split(mask, mask0, mask1):
    # the two pieces partition mask exactly, are disjoint, and each is a
    # single connected component (an empty piece is allowed, e.g. one voxel)
    assert np.array_equal(mask0 | mask1, mask)
    assert not (mask0 & mask1).any()
    assert _is_contiguous(mask0)
    if mask1.any():
        assert _is_contiguous(mask1)


class TestIterBfs:
    def test_distances_along_a_line(self):
        # a 1x5 line; BFS from the (0, 0) end visits voxels in order with
        # graph distances 0, 1, 2, 3, 4.
        mask = np.ones((1, 5), dtype=bool)
        visited = list(iter_bfs(mask, (0, 0)))
        assert [ijk for ijk, _ in visited] == [(0, 0), (0, 1), (0, 2),
                                               (0, 3), (0, 4)]
        assert [d for _, d in visited] == [0, 1, 2, 3, 4]

    def test_blocked_halts_the_front(self):
        # blocking the (0, 2) voxel walls off everything beyond it: the front
        # neither yields nor expands through a blocked voxel, so only
        # (0, 0) and (0, 1) are reached.
        mask = np.ones((1, 5), dtype=bool)
        blocked = np.zeros_like(mask)
        blocked[0, 2] = True
        reached = [ijk for ijk, _ in iter_bfs(mask, (0, 0), blocked=blocked)]
        assert reached == [(0, 0), (0, 1)]

    def test_seed_outside_mask_raises(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        with pytest.raises(AssertionError):
            list(iter_bfs(mask, (0, 0)))


class TestGetDiameter:
    def test_line_endpoints(self):
        # the diameter of a 1x6 line is its two ends.
        mask = np.ones((1, 6), dtype=bool)
        ijk0, ijk1 = get_diameter(mask)
        assert {ijk0, ijk1} == {(0, 0), (0, 5)}

    def test_disconnected_raises(self):
        # two separated rows are not a single connected component.
        mask = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [1, 1, 1]], dtype=bool)
        with pytest.raises(ValueError):
            get_diameter(mask)


class TestSplitMask:
    def test_even_line_splits_in_half(self):
        # a 1x6 line splits into two contiguous runs of three voxels.
        mask = np.ones((1, 6), dtype=bool)
        mask0, mask1 = split_mask(mask)
        _assert_valid_split(mask, mask0, mask1)
        assert mask0.sum() == 3
        assert mask1.sum() == 3

    def test_odd_line_differs_by_one(self):
        # a 1x7 line cannot split evenly; the pieces differ by one voxel.
        mask = np.ones((1, 7), dtype=bool)
        mask0, mask1 = split_mask(mask)
        _assert_valid_split(mask, mask0, mask1)
        assert abs(int(mask0.sum()) - int(mask1.sum())) == 1

    def test_square_block_2d(self):
        # a solid 4x4 block splits into two contiguous halves of eight.
        mask = np.ones((4, 4), dtype=bool)
        mask0, mask1 = split_mask(mask)
        _assert_valid_split(mask, mask0, mask1)
        assert mask0.sum() == 8
        assert mask1.sum() == 8

    def test_cube_block_3d(self):
        # a solid 2x2x2 cube splits into two contiguous halves of four.
        mask = np.ones((2, 2, 2), dtype=bool)
        mask0, mask1 = split_mask(mask)
        _assert_valid_split(mask, mask0, mask1)
        assert mask0.sum() == 4
        assert mask1.sum() == 4

    def test_l_shape_stays_contiguous(self):
        # an L-shaped region: a straight cut would sever a piece, so this
        # exercises the curved seam the BFS fronts grow.
        mask = np.array([[1, 0, 0, 0],
                         [1, 0, 0, 0],
                         [1, 1, 1, 1]], dtype=bool)
        mask0, mask1 = split_mask(mask)
        _assert_valid_split(mask, mask0, mask1)
        assert abs(int(mask0.sum()) - int(mask1.sum())) <= 1

    def test_single_voxel(self):
        # one voxel cannot be split; it lands wholly in one piece.
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        mask0, mask1 = split_mask(mask)
        _assert_valid_split(mask, mask0, mask1)
        assert mask0.sum() == 1
        assert mask1.sum() == 0

    def test_disconnected_raises(self):
        mask = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [1, 1, 1]], dtype=bool)
        with pytest.raises(ValueError):
            split_mask(mask)
