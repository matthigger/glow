import numpy as np
import scipy.stats
from scipy.optimize import minimize

from hglm.experiment import wilks_to_chi2, get_wilks, decompose


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
            covariance over total error (bounded between 0 and 1, inclusive)

    Returns:
        offset (np.array): (b, num_img) offset to apply to all images to
            produce desired f stat
        sigma_gain (float): spatial covariance scaling needed to achieve
            roughness coefficient, None if "rough" is not input.
        rough (float): roughness coefficient achieved
    """
    if rough is not None:
        assert 0 <= rough <= 1, 'invalid rough given'

    # prep constants
    a1 = contrast.sum()
    b, num_img, num_vox = y.shape
    y_mean = y.mean(axis=2)

    # get target chi2 to impose given pvalue
    chi2_target = scipy.stats.chi2.ppf(1 - pval, df=a1 * b)

    # qr decomposition of x
    q = decompose(x, contrast)
    yq1_norm2 = ((y_mean @ q[1].T) ** 2).sum()
    yq2_norm2 = ((y_mean @ q[2].T) ** 2).sum()
    yq1q1y = y_mean @ q[1].T @ q[1] @ y_mean.T
    yq2q2y = y_mean @ q[2].T @ q[2] @ y_mean.T

    # compute sigma_sum (non-normalized spatial covariance)
    yr = y.reshape((b, -1), order='F')
    sigma_sum_orig = yr @ yr.T - y_mean @ y_mean.T * num_vox

    def get_e_h_sigma(alpha):
        """" computes e, h, sigma_sum under a given alpha
        """
        if len(alpha) == 2:
            # don't optimize roughness
            alpha1, alpha2 = alpha
            sigma_sum = sigma_sum_orig
        else:
            # optimize roughness too
            alpha1, alpha2, sigma_gain_sqrt = alpha
            sigma_sum = (sigma_gain_sqrt ** 2) * sigma_sum_orig

        h = (1 + alpha1) ** 2 * yq1q1y * num_vox
        e = (1 + alpha2) ** 2 * yq2q2y * num_vox + sigma_sum
        return e, h, sigma_sum

    def constraint_chi(alpha):
        """ when this function output is zero, chi2_target achieved """
        e, h, _ = get_e_h_sigma(alpha)
        wilks = get_wilks(e, h)
        chi2, _ = wilks_to_chi2(wilks, a=a1, b=b, n=num_img)
        return chi2 - chi2_target

    def constraint_rough(alpha):
        e, h, sigma_sum = get_e_h_sigma(alpha)
        _rough = np.trace(sigma_sum) / np.trace(e)

        return _rough - rough

    def obj(alpha):
        r"""  ||\Delta||^2 = \sum_{i=1}^3 \alpha_i^2 ||Q_i \bar{Y}_r^T||^2

        Args:
             alpha (tuple): a1, a2 (per equations)
        """

        _obj = alpha[0] ** 2 * yq1_norm2 + alpha[1] ** 2 * yq2_norm2

        if len(alpha) == 3:
            # scaling spatial covariance too (to achieve given roughness)
            sigma_gain_sqrt = alpha[2]
            _obj += (1 - sigma_gain_sqrt) ** 2 * np.trace(sigma_sum_orig)

        return _obj

    # setup starting point & constraints (assuming no rough constraint)
    x0 = np.zeros(2)
    constraints = [dict(type='eq', fun=constraint_chi)]
    if rough is not None:
        # add rough constraint
        constraints.append(dict(type='eq', fun=constraint_rough))
        x0 = np.array([0, 0, 1])

    # optimize
    res = minimize(fun=obj, x0=x0, constraints=constraints,
                   method='trust-constr',
                   options=dict(maxiter=10000))
    assert res.success, 'optimization failed'

    # compute offset
    alpha1, alpha2 = res.x[0], res.x[1]
    offset = alpha1 * y_mean @ q[1].T @ q[1] + \
             alpha2 * y_mean @ q[2].T @ q[2]

    # get sigma_gain
    if rough is None:
        # no roughness constraint given
        sigma_gain = None
    else:
        sigma_gain = res.x[2] ** 2

    # compute roughness achieved
    e, h, sigma_sum = get_e_h_sigma(res.x)
    rough = np.trace(sigma_sum) / np.trace(e)

    return offset, sigma_gain, rough
