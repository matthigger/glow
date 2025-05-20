from hglm.experiment import Experiment
from hglm.experiment.regress import *
from hglm.graph import iter_size_yout_ybar, iter_topo


def test_all():
    """ test that regress stats computed properly for random data

    it is convenient here to test all region in an arbitrary hierarchy (
    allows us to use the iter_size_yout_ybar() generator), though these
    outputs generated another way would be just as valid
    """
    exp = Experiment.from_gauss(b=3, shape=(10, 10), seed=0)
    b, num_img, num_vox = exp.y.shape
    children = np.arange(2 * num_vox - 2).reshape((-1, 2))

    for reg_idx, size, yout, ybar in iter_size_yout_ybar(exp.y, children):
        yout = yout[:, :, 0]
        ybar = ybar[:, :, 0]

        # build y corresponding to region
        vox_idx = list(iter_topo(children=children,
                                 num_leaf=num_vox,
                                 node_start=reg_idx,
                                 only_leaf=True))
        _y = np.atleast_3d(exp.y[:, :, vox_idx])

        # compute sigma (slow & reliable)
        _y_center = _y - _y.mean(axis=2)[:, :, np.newaxis]
        sigma_exp = np.cov(_y_center.reshape((b, -1), order='F'), bias=True)

        sigma = get_sigma(size, yout, ybar)
        assert np.allclose(sigma_exp, sigma)


def test_scale_sigma():
    b, num_img, num_vox = 3, 10, 100
    rng = np.random.default_rng(seed=0)
    y = rng.standard_normal((b, num_img, num_vox))

    sigma_before = get_sigma_from_y(y)

    # test 1: gain=10
    gain_exp = 10
    y1 = scale_sigma(y, gain=gain_exp)
    sigma_after = get_sigma_from_y(y1)
    gain_obs = np.trace(sigma_after) / np.trace(sigma_before)
    assert np.isclose(gain_exp, gain_obs)

    # output has same mean (per image) as input
    assert np.allclose(y.mean(axis=2), y1.mean(axis=2))

    # test 2: to desired tr_sigma
    tr_sigma_exp = 10
    y2 = scale_sigma(y, tr_sigma=tr_sigma_exp)
    sigma_after = get_sigma_from_y(y2)
    assert np.isclose(tr_sigma_exp, np.trace(sigma_after))

    # output has same mean (per image) as input
    assert np.allclose(y.mean(axis=2), y2.mean(axis=2))


def test_get_size_yout_ybar():
    b, num_img, num_vox = 3, 10, 100
    rng = np.random.default_rng(seed=0)
    y = rng.standard_normal((b, num_img, num_vox))
    size_obs, yout_obs, ybar_obs = get_size_yout_ybar(y)

    yout_exp = 0
    for vox_idx in range(num_vox):
        _y = y[:, :, vox_idx]
        yout_exp += _y @ _y.T
    ybar_exp = y.mean(axis=2)

    assert size_obs == num_vox
    assert np.allclose(yout_obs, yout_exp)
    assert np.allclose(ybar_obs, ybar_exp)
