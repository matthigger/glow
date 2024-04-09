from hglm.experiment.regress import *
from hglm.graph import iter_size_yout_ybar, iter_topo
from test.helper import generate_dummy_data


def test_eps_sigma():
    """ test that regress stats computed properly for random data

    it is convenient here to test all region in an arbitrary hierarchy (
    allows us to use the iter_size_yout_ybar() generator), though these
    outputs generated another way would be just as valid
    """
    b = 3
    num_vox = 100
    x, y, contrast = generate_dummy_data(b=b, num_vox=100)
    children = np.arange(2 * num_vox - 2).reshape((-1, 2))

    get_eps = prep_get_eps(x)
    for reg_idx, size, yout, ybar in iter_size_yout_ybar(y, children):
        # build y corresponding to region
        vox_idx = list(iter_topo(children=children,
                                 num_leaf=num_vox,
                                 node_start=reg_idx,
                                 only_leaf=True))
        _y = np.atleast_3d(y[:, :, vox_idx])

        # compute eps (slow & reliable)
        xr = np.tile(x, (1, len(vox_idx)))
        yr = _y.reshape((b, -1), order='F')
        yhat = yr @ np.linalg.pinv(xr) @ xr
        error = yr - yhat
        eps_exp = error @ error.T / error.shape[1]

        # compute & validate eps
        eps = get_eps(size, yout, ybar)
        assert np.allclose(eps_exp, eps)

        # compute sigma (slow & reliable)
        _y_center = _y - _y.mean(axis=2)[:, :, np.newaxis]
        sigma_exp = np.cov(_y_center.reshape((b, -1), order='F'), bias=True)

        sigma = get_sigma(size, yout, ybar)
        assert np.allclose(sigma_exp, sigma)
