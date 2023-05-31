from hrba.sample_effect.offset import *


def test_compute_offset():
    # generate dummy data
    rng = np.random.default_rng(seed=0)
    a, b, num_img, reg_size = 2, 3, 5, 10
    x = rng.standard_normal((a, num_img))
    y = rng.standard_normal((b, num_img, reg_size))
    x[0, :] = 1  # bias
    contrast = np.array([False, True])

    # prep
    _x = x[~contrast, :]
    a = (~contrast).sum(), a
    h = np.linalg.pinv(_x) @ _x, np.linalg.pinv(x) @ x
    y_mean = y.mean(axis=2)

    ssy = (y ** 2).sum()

    for f_stat_exp in (0, 1, 10):
        beta_expected, offset = compute_offset(x=x, y=y, contrast=contrast,
                                               f_stat=f_stat_exp)

        # validate beta is the mmse estimator
        beta_observed = (y_mean + offset) @  np.linalg.pinv(x)
        assert np.allclose(beta_observed, beta_expected)

        # validate that the offset yields proper f stat
        y_mean_off = y_mean + offset
        err0 = ssy - reg_size * np.trace(y_mean_off @ h[0] @ y_mean_off.T)
        err1 = ssy - reg_size * np.trace(y_mean_off @ h[0] @ y_mean_off.T)
        const = b * (a[0] - a[1]) / (reg_size * num_img - b * a[1])
        f_stat_obs = (err0 - err1) / err1 * const
        assert np.isclose(f_stat_obs, f_stat_exp)
