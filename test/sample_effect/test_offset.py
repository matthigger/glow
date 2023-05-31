from pytest import raises

from helper import generate_dummy_data
from hrba.sample_effect.offset import *


def test_compute_offset():
    # generate dummy data
    x, y, contrast = generate_dummy_data(seed=0)
    b, num_img, reg_size = y.shape

    # prep
    _x = x[~contrast, :]
    a = (~contrast).sum(), contrast.size
    h = np.linalg.pinv(_x) @ _x, np.linalg.pinv(x) @ x
    y_mean = y.mean(axis=2)

    ssy = (y ** 2).sum()

    for f_stat_exp in (0, 1, 10):
        beta_expected, offset = compute_offset(x=x, y=y, contrast=contrast,
                                               f_stat=f_stat_exp)

        # validate beta is the mmse estimator
        beta_observed = (y_mean + offset) @ np.linalg.pinv(x)
        assert np.allclose(beta_observed, beta_expected)

        # validate that the offset yields proper f stat
        y_mean_off = y_mean + offset
        err0 = ssy - reg_size * np.trace(y_mean_off @ h[0] @ y_mean_off.T)
        err1 = ssy - reg_size * np.trace(y_mean_off @ h[0] @ y_mean_off.T)
        const = b * (a[0] - a[1]) / (reg_size * num_img - b * a[1])
        f_stat_obs = (err0 - err1) / err1 * const
        assert np.isclose(f_stat_obs, f_stat_exp)


def test_get_offset_to_tr_eps():
    # generate dummy data
    x, y, contrast = generate_dummy_data(seed=0)
    b, num_img, num_vox = y.shape

    # the sbj-pooled spatial covariance (within sbj) is lower bound on trace
    # ofe error
    n = num_img * num_vox
    min_error = ((y - y.mean(axis=2)[..., np.newaxis]) ** 2).sum() / n

    for scale in (1, 3, 100):
        tr_eps = scale * min_error
        offset = get_offset_to_tr_eps(x, y, tr_eps=tr_eps)

        # apply offset
        _y = y + offset[..., np.newaxis]

        # compute error
        y_hat = _y.mean(axis=2) @ np.linalg.pinv(x) @ x
        error = _y - y_hat[..., np.newaxis]
        tr_eps_obs = (error ** 2).sum() / (num_img * num_vox)

        assert np.isclose(tr_eps, tr_eps_obs)

    with raises(RuntimeError):
        # check runtime error is thrown if user requests too low a tr_eps value
        tr_eps = min_error / 2
        get_offset_to_tr_eps(x, y, tr_eps=tr_eps)
