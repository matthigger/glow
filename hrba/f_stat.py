from collections import namedtuple

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
    # compute tr_eps
    _x = x[~contrast, :]
    tr_eps = get_tr_eps(_x, y), get_tr_eps(x, y)

    return (tr_eps[0] - tr_eps[1]) / tr_eps[1] * get_f_const(y, contrast)


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


"""
size (int): size of region
y_mean (np.array): (b, num_img) mean imaging features (across voxel)
myo (np.array): (b, b) mean y outer product of all observations (across 
    voxel and image)           
f_stat (float): f statistic
llr (float): log likelihood ratio of full over reduced model. (note model is 
    distinct from traditional f stat assumptions, handles space and image 
    variance seperately, see paper)

The following variables are all tuples of (b, b) arrays corresponding to (
    reduced, full) models (note: eps = eps_s + eps_r)

eps (tuple): total error covariance
eps_s (tuple): spatial covariance, image pooled covariance of y across voxels
eps_r (tuple): average error covariance, error from voxel averaged y
"""
RegStat = namedtuple('RegStat', ['size', 'y_mean', 'myo', 'f_stat', 'llr',
                                 'eps', 'eps_s', 'eps_r'])


class RegStatComputer:
    """ provides a call method which computes region stats

    this method is useful for computing stats iteratively in a graph as it
    requires reg_size, y_mean and myo all of which may be computed recursively

    All attributes below are tuples corresponding to (reduced, full) models

    Attributes:
        h (tuple): (b, b) estimate forming matrix
        i_minus_h (tuple): (b, b)residual forming matrix
        a (tuple): (int) number of x features in model
    """

    def __init__(self, x, contrast):
        # pre-compute
        num_img = x.shape[1]
        x = x[~contrast, :], x
        self.h = tuple(np.linalg.pinv(_x) @ _x for _x in x)
        self.i_minus_h = tuple(np.eye(num_img) - _h for _h in self.h)
        self.a = (~contrast).sum(), contrast.size

    def __call__(self, reg_size, y_mean, myo):
        """ computes statistics

        Args:
            reg_size (int): size of region
            y_mean (np.array): (b, num_img) mean imaging features (across
                voxel)
            myo (np.array): (b, b) mean y outer product of all observations
                (across voxel and image)

        Returns:
            reg_stat (RegStat): region stat
        """

        # compute error covariance (eps)
        num_img = self.h[0].shape[0]
        eps = tuple(myo - y_mean @ _h @ y_mean.T / num_img
                    for _h in self.h)

        # compute f stat
        tr_eps = tuple(np.trace(_eps) for _eps in eps)
        b = myo.shape[0]
        const = (reg_size * num_img - b * self.a[1])
        const /= b * (self.a[1] - self.a[0])
        f_stat = (tr_eps[0] - tr_eps[1]) / tr_eps[1] * const

        # break up error covariance by mean regression error & spatial cov
        eps_r = tuple(y_mean @ _imh @ y_mean.T / num_img
                      for _imh in self.i_minus_h)
        eps_s = tuple(_eps - _eps_r
                      for _eps, _eps_r in zip(eps, eps_r))

        # compute log-likelihood-ratio (note: log_p missing additive
        # constants, which cancel out in computing llr)
        log_p = np.array([np.log10(np.linalg.det(_eps_s)) * reg_size +
                          np.log10(np.linalg.det(_eps_r))
                          for _eps_s, _eps_r in zip(eps_s, eps_r)])
        log_p *= - num_img / 2
        llr = log_p[1] - log_p[0]

        return RegStat(size=reg_size, y_mean=y_mean, myo=myo, f_stat=f_stat,
                       llr=llr, eps=eps, eps_r=eps_r, eps_s=eps_s)
