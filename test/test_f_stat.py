from hglm.f_stat import *
from test.helper import generate_dummy_data


def get_f_stat_reliable(x, y, contrast):
    """ computes f statistic

    Args:
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities
        contrast (np.array): (a) True for each corresponding feature in x which
            is "of interest" (other x features form the reduced model in
            computing f statistic)

    Returns:
        f_stat (float): f statistic of MMSE regression from x to y
    """
    # compute tr_eps
    _x = x[~contrast, :]
    tr_eps = get_tr_eps(_x, y), get_tr_eps(x, y)

    return (tr_eps[0] - tr_eps[1]) / tr_eps[1] * get_f_const(y, contrast)


def test_get_f_stat_qr():
    for seed in range(10):
        x, y, contrast = generate_dummy_data(seed=0)

        f_stat_exp = get_f_stat_reliable(x, y, contrast)
        f_stat_obs = get_f_stat(x, y, contrast)
        assert np.isclose(f_stat_exp, f_stat_obs)
        assert np.isclose(f_stat_exp, f_stat_obs)
