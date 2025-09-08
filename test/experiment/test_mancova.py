from glow.experiment import ExperimentImageOnly
from glow.experiment.mancova import *
from glow.graph import iter_stat, iter_topo


def test_get_mancova():
    seed = 0
    a = 2
    b = 3
    num_vox = 5
    exp = ExperimentImageOnly.from_gauss(seed=seed, b=b, num_img=4, shape=(num_vox,))
    children = np.arange(2 * num_vox - 2).reshape((-1, 2), order='C')

    for add_bias in range(2):
        exp = exp.sample_x(a=a, seed=seed, add_bias=add_bias)

        for reg_idx in iter_topo(children=children, num_leaf=num_vox):
            vox = np.array(list(iter_topo(children=children,
                                          num_leaf=num_vox,
                                          node_start=reg_idx,
                                          only_leaf=True)))
            _y = exp.y[:, :, vox]

            e_obs, h_obs, sigma = get_mancova(x=exp.x, y=_y, contrast=exp.contrast)

            # slow and steady compute of e and h
            yr = np.concatenate([exp.y[:, :, _vox] for _vox in vox], axis=1)
            xr = np.concatenate([exp.x for _vox in vox], axis=1)

            if not exp.contrast.all():
                # project yr into nullspace of covariates
                _xr = xr[~exp.contrast, :]
                p = np.eye(yr.shape[1]) - np.linalg.pinv(_xr) @ _xr
                yr = yr @ p
                xr = xr[exp.contrast, :] @ p

            hat = yr @ np.linalg.pinv(xr) @ xr
            err = yr - hat

            h_exp = hat @ hat.T
            e_exp = err @ err.T

            # test mancova stats
            assert np.allclose(h_obs, h_exp)
            assert np.allclose(e_obs, e_exp)


def test_get_hotel_tr():
    def get_hotel_tr_trusted(e, h):
        inv_e = np.linalg.inv(e)
        mat = inv_e @ h
        return np.trace(mat)

    rng = np.random.default_rng(0)
    b = 3
    for _ in range(10):
        e = rng.standard_normal((b, b))
        h = rng.standard_normal((b, b))

        exp = get_hotel_tr_trusted(e, h)
        obs = get_hotel_tr(e, h)
        assert np.isclose(exp, obs)
