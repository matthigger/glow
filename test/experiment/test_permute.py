from hrba.experiment.permute import *
from hrba.graph import iter_topo


def test_permute_eps():
    # prep test case
    a, b, num_img, num_vox, n_permute = 3, 4, 10, 6, 5
    rng = np.random.default_rng(seed=0)
    x = rng.standard_normal((a, num_img))
    x[0, :] = 1
    contrast = np.ones(a, dtype=bool)
    contrast[0] = False
    y_mean = rng.standard_normal((b, num_img, num_vox))
    children = np.arange((num_vox - 1) * 2).reshape(num_vox - 1, 2)

    # compute eps quick way
    eps, myo_obs, last_term_obs = permute_eps(x=x, contrast=contrast,
                                              y_mean=y_mean, children=children,
                                              n_permute=n_permute)

    x_both = x[~contrast, :], x
    h_list = [np.linalg.pinv(_x) @ _x for _x in x_both]

    for reg_idx in range(num_vox * 2 - 1):
        # compute y_mean for region
        vox_list = list(iter_topo(children=children, node_start=reg_idx,
                                  only_leaf=True))
        _y_mean = np.zeros((b, num_img))
        for vox in vox_list:
            _y_mean += y_mean[:, :, vox]
        _y_mean /= len(vox_list)

        for perm_idx in range(n_permute):
            # compute freed lane permutation p
            p = get_perm_matrix(seed=perm_idx, num_img=num_img)
            p = (np.eye(num_img) - h_list[0]) @ p + h_list[0]

            # compute eps the slow way
            myo_exp = np.zeros((b, b))
            for vox in vox_list:
                yv = y_mean[:, :, vox]
                myo_exp += yv @ p @ p.T @ yv.T
            myo_exp /= len(vox_list)

            for model_idx, h in enumerate(h_list):
                last_term_exp = _y_mean @ p @ h @ p.T @ _y_mean.T
                eps_exp = (myo_exp - last_term_exp) / num_img

                eps_obs = eps[perm_idx, reg_idx, model_idx, ...]

                # grab relevant data to debug
                myo_obs0 = myo_obs[perm_idx, reg_idx, ...]
                _reg_perm_model = reg_idx, perm_idx, model_idx
                last_term_obs0 = \
                    last_term_obs[perm_idx, reg_idx, model_idx, :, :]

                assert np.allclose(myo_obs0, myo_exp)
                assert np.allclose(last_term_obs0, last_term_exp)

                assert np.allclose(eps_exp, eps_obs)


def test_get_perm_matrix():
    perm = get_perm_matrix(num_img=5, seed=0)
    np.testing.assert_array_almost_equal(perm, np.eye(5))

    perm = get_perm_matrix(num_img=5, seed=1)
    perm_expect = np.array([[0., 0., 0., 0., 1.],
                            [1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 1., 0.]])

    np.testing.assert_array_almost_equal(perm, perm_expect)
