import numpy as np
import hglm.experiment


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
    b, num_img, reg_size = y.shape
    q = hglm.experiment.decompose(x, contrast)

    # compute norm of spatial cov square of t
    # |r| \Sigma_r & = \sum_{v \in r} (Y_v - \bar{Y}_r)(Y_v - \bar{Y}_r)^T
    #              & = Y_r Y_r^T - |r| \bar{Y}_r \bar{Y}_r^T
    y_mean = y.mean(axis=2)
    yr = y.reshape((y.shape[0], -1), order='F')
    space_cov = yr @ yr.T / reg_size - y_mean @ y_mean.T

    f_const = get_f_const(y, contrast)

    return (f_const * np.linalg.norm(q[1] @ y_mean.T) ** 2 /
            (np.trace(space_cov) + np.linalg.norm(q[2] @ y_mean.T) ** 2))


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


def get_f_degrees(num_img, reg_size, a):
    dfn = num_img * reg_size - a[1]
    dfd = a[1] - a[0]

    return dfn, dfd


def get_f_const(y, contrast):
    # prep
    b, num_img, reg_size = y.shape
    a = (~contrast).sum(), contrast.size

    const = (reg_size * num_img - b * a[1]) / b * (a[1] - a[0])

    assert const >= 0, 'negative f stat constant: insufficient samples'

    return const
