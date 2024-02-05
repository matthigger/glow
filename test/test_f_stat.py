from hrba.f_stat import *
from test.helper import generate_dummy_data


def test_get_f_stat_qr():
    for seed in range(10):
        x, y, contrast = generate_dummy_data(seed=0)

        f_stat_exp = get_f_stat(x, y, contrast)
        f_stat_obs = get_f_stat_qr(x, y, contrast)
        assert np.isclose(f_stat_exp, f_stat_obs)


def test_stat_computer():
    x, y, contrast = generate_dummy_data(seed=0)
    stat_computer = RegStatComputer(x=x, contrast=contrast, extra_flag=True)

    b, num_img, num_vox = y.shape
    myo = np.einsum('ijk,ajk->ia', y, y) / (num_img * num_vox)
    y_mean = y.mean(axis=2)
    rs = stat_computer(reg_size=num_vox, y_mean=y_mean, myo=myo)

    x = x[~contrast, :], x

    for _x, _eps, _eps_r, _eps_s in zip(x, rs['eps'], rs['eps_r'],
                                        rs['eps_s']):
        # compute eps the trustworthy / slow way
        h = np.linalg.pinv(_x) @ _x
        y_est = y_mean @ h
        y_err = y - y_est[..., np.newaxis]
        eps_exp = np.einsum('ijk,ajk->ia', y_err, y_err) / (num_img * num_vox)

        # validate eps compute
        assert np.allclose(eps_exp, _eps)

        # validate eps_s compute (doesnt vary full to reduced model)
        eps_s_exp = 0
        for img_idx in range(num_img):
            _y = y[:, img_idx, :]
            eps_s_exp += np.cov(_y, ddof=0)
        eps_s_exp /= num_img
        assert np.allclose(_eps_s, eps_s_exp)

        # validate eps_r compute
        y_err_mean = y_mean - y_est
        eps_r_exp = np.cov(y_err_mean, ddof=0)
        assert np.allclose(_eps_r, eps_r_exp)
