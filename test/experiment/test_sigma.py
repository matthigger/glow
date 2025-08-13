import numpy as np

from hglm.experiment import stretch_sigma
from hglm.experiment.sigma import get_sigma_from_y, get_size_yout_ymean


def test_scale_sigma():
    b, num_img, num_vox = 3, 10, 100
    rng = np.random.default_rng(seed=0)
    y = rng.standard_normal((b, num_img, num_vox))

    sigma_before = get_sigma_from_y(y)

    # test 1: gain=10
    gain_exp = 10
    y1 = stretch_sigma(y, scale=gain_exp ** .5)
    sigma_after = get_sigma_from_y(y1)
    gain_obs = np.trace(sigma_after) / np.trace(sigma_before)
    assert np.isclose(gain_exp, gain_obs)

    # output has same mean (per image) as input
    assert np.allclose(y.mean(axis=2), y1.mean(axis=2))


def test_get_size_yout_ymean():
    b, num_img, num_vox = 3, 10, 100
    rng = np.random.default_rng(seed=0)
    y = rng.standard_normal((b, num_img, num_vox))
    size_obs, yout_obs, ymean_obs = get_size_yout_ymean(y)

    yout_exp = 0
    for vox_idx in range(num_vox):
        _y = y[:, :, vox_idx]
        yout_exp += _y @ _y.T
    ymean_exp = y.mean(axis=2)

    assert size_obs == num_vox
    assert np.allclose(yout_obs, yout_exp)
    assert np.allclose(ymean_obs, ymean_exp)
