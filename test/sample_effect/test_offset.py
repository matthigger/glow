from pytest import raises

from helper import generate_dummy_data
from hrba.f_stat import *
from hrba.sample_effect.offset import *


def test_compute_offset():
    # range(100) takes ~35 seconds to run
    for seed in range(1):
        # generate dummy data
        x, y, contrast = generate_dummy_data(seed=seed)

        for f_stat_exp in (0, 1, 10, 100, 1e6):
            # validate that f stat is achieved
            offset = compute_offset(x=x, y=y, contrast=contrast,
                                    f_stat=f_stat_exp)
            f_stat_obs = get_f_stat(x=x, y=y + offset[..., np.newaxis],
                                    contrast=contrast)
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
