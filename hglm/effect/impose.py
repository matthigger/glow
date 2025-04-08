import numpy as np
import scipy.stats
from scipy.optimize import minimize

import hglm.experiment
import hglm.f_stat
from hglm.experiment import wilks_to_chi2, get_wilks


def compute_offset(x, y, contrast, pval=None):
    """ get offset to y, constant across voxels, which imposes an f-stat

    Args:
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities
        contrast (np.array): (a) True for each corresponding feature in x which
            is "of interest" (other x features form the reduced model in
            computing f statistic)
        pval (float): severity of desired effect

    Returns:
        offset (np.array): (b, num_img) offset to apply to all images to
            produce desired pval
    """
    # prep constants
    a0 = (~contrast).sum()
    a1 = contrast.size - a0
    b, num_img, reg_size = y.shape

    # prep matrices
    y_mean = y.mean(axis=2)

    # get target chi2 to impose given pvalue
    df = a1 * b
    chi2_target = scipy.stats.chi2.ppf(1 - pval, df=df)

    # qr decomposition of x
    q = hglm.experiment.decompose(x, contrast)
    yq1q1y = y_mean @ q[1].T @ q[1] @ y_mean.T
    yq2q2y = y_mean @ q[2].T @ q[2] @ y_mean.T
    yq1_norm2 = np.linalg.norm(y_mean @ q[1].T) ** 2
    yq2_norm2 = np.linalg.norm(y_mean @ q[2].T) ** 2

    # compute norm of spatial cov square of t
    # |r| \Sigma_r & = \sum_{v \in r} (Y_v - \bar{Y}_r)(Y_v - \bar{Y}_r)^T
    #              & = Y_r Y_r^T - |r| \bar{Y}_r \bar{Y}_r^T
    yr = y.reshape((y.shape[0], -1), order='F')
    sigma = yr @ yr.T / reg_size - y_mean @ y_mean.T

    def constraint(alpha):
        """ when this function output is zero, chi2_target achieved """
        alpha1, alpha2 = alpha
        h = (1 + alpha1) ** 2 / num_img * yq1q1y
        e = sigma + (1 + alpha2) ** 2 / num_img * yq2q2y
        wilks = get_wilks(e, h)
        chi2, _ = wilks_to_chi2(wilks, a=a1, b=b, n=num_img)
        return (chi2 - chi2_target) ** 2

    def obj(alpha):
        r"""  ||\Delta||^2 = \sum_{i=1}^3 \alpha_i^2 ||Q_i \bar{Y}_r^T||^2

        Args:
             alpha (tuple): a1, a2 (per equations)
        """
        a1, a2 = alpha
        return a1 ** 2 * yq1_norm2 + a2 ** 2 * yq2_norm2

    # find scale of each offset which achieves chi2 while being as close as
    # possible to original problem
    res = minimize(fun=obj, x0=np.array([0, 0]),
                   constraints=[{'type': 'eq',
                                 'fun': constraint}],
                   options={'maxiter': 1000})
    assert res.success, 'optimization failed'

    # compute offset
    alpha1, alpha2 = res.x
    offset = alpha1 * y_mean @ q[1].T @ q[1] + \
             alpha2 * y_mean @ q[2].T @ q[2]

    return offset
