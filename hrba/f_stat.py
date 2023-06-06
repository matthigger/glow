import numpy as np


def get_f_stat(x, y, contrast):
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

    # prep
    b, num_img, reg_size = y.shape
    _x = x[~contrast, :]

    # compute tr_eps
    tr_eps = get_tr_eps(_x, y), get_tr_eps(x, y)

    a = (~contrast).sum(), contrast.size
    const = (reg_size * num_img - b * a[1]) / b * (a[1] - a[0])

    return (tr_eps[0] - tr_eps[1]) / tr_eps[1] * const


def get_tr_eps(x, y):
    """ compute mean trace of residual covariance

    Attributes:
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities

    Returns:
        tr_eps (float): trace of residual covariance (error).  mean per
            observation
    """
    b, num_img, reg_size = y.shape

    h = np.linalg.pinv(x) @ x

    y_mean = y.mean(axis=2)
    ssy = (y ** 2).sum()

    tr_eps = ssy - reg_size * np.trace(y_mean @ h @ y_mean.T)

    return tr_eps / (num_img * reg_size)
