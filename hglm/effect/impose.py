import numpy as np
import scipy.stats
from scipy.optimize import minimize

import hglm.experiment
import hglm.f_stat


def wilks_to_chi2(wilks, a, b, n):
    df = a * b
    chi2 = -(n - 0.5 * (a + b + 1)) * np.log(wilks)
    return chi2, df


def chi2_to_wilks(chi2, a, b, n):
    scale = -(n - 0.5 * (a + b + 1))
    return np.exp(chi2 / scale)


def get_wilks(e, h):
    return np.linalg.det(e) / np.linalg.det(e + h)


def compute_offset(x, y, contrast, pval=None, rough=None):
    """ get offset to y, constant across voxels, which imposes an f-stat

    Args:
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities
        contrast (np.array): (a) True for each corresponding feature in x which
            is "of interest" (other x features form the reduced model in
            computing f statistic)
        pval (float): p-value may be passed in place of f stat
        rough (float): roughness coefficient desired.  a ratio of the spatial
            covariance over the power of error in the reduced model

            rough = sigma_tr * num_img / q2_norm2

            q2_norm2 is average power of the residuals in the reduced model,
            applied to the spatially averaged data

    Returns:
        offset (np.array): (b, num_img) offset to apply to all images to
            produce desired f stat
        sigma_gain (float): spatial covariance scaling needed to achieve
            roughness coefficient, None if "rough" is not input.
        rough (float): roughness coefficient achieved
    """
    # prep constants
    a0 = (~contrast).sum()
    a1 = contrast.size - a0
    b, num_img, reg_size = y.shape

    # prep matrices
    y_mean = y.mean(axis=2)

    # get target chi2 to impose given p_value
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

    if rough is not None:
        raise NotImplementedError

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
