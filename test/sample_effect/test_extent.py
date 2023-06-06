from hrba.sample_effect.extent import *


def test_get_mask_idx():
    mask = np.ones((4, 4))
    mask[0, :] = 0
    mask_idx = get_mask_idx(mask)

    mask_idx_expect = np.array([[-1, -1, -1, -1],
                                [0, 1, 2, 3],
                                [4, 5, 6, 7],
                                [8, 9, 10, 11]])

    assert np.array_equal(mask_idx, mask_idx_expect)


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

        extenter_min_var = ExtenterMinVar(n=mask.sum())

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
