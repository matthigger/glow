import warnings

import numpy as np
from scipy.optimize import minimize

from glow.experiment import decompose, get_hotel_tr


def compute_offset(x, y, contrast, hotel_tr):
    """find the smallest offset to y that imposes a given Hotelling's trace.

    Args:
        x (np.array): (a, num_img) design matrix
        y (np.array): (b, num_img, num_vox) image intensities
        contrast (np.array): (a,) boolean, True for features of interest
        hotel_tr (float): target Hotelling's trace

    Returns:
        offset (np.array): (b, num_img) constant offset across voxels
    """
    # prep constants
    b, num_img, num_vox = y.shape

    # prep matrices
    y_mean = y.mean(axis=2)

    # qr decomposition of x
    q = decompose(x, contrast)
    yq1_norm2 = ((y_mean @ q[1].T) ** 2).sum()
    yq2_norm2 = ((y_mean @ q[2].T) ** 2).sum()
    yq1q1y = y_mean @ q[1].T @ q[1] @ y_mean.T
    yq2q2y = y_mean @ q[2].T @ q[2] @ y_mean.T

    # compute sigma_sum (non-normalized spatial covariance)
    yr = y.reshape((b, -1), order='F')
    sigma_sum = yr @ yr.T - y_mean @ y_mean.T * num_vox

    def get_e_h_sigma(alpha):
        """compute e, h, sigma_sum under a given alpha."""
        alpha1, alpha2 = alpha
        h = (1 + alpha1) ** 2 * yq1q1y * num_vox
        e = (1 + alpha2) ** 2 * yq2q2y * num_vox + sigma_sum
        return e, h, sigma_sum

    def constraint(alpha):
        """zero when target Hotelling's trace is achieved."""
        e, h, _ = get_e_h_sigma(alpha)
        _hotel_tr = get_hotel_tr(e, h)
        return _hotel_tr - hotel_tr

    def obj(alpha):
        r"""squared offset norm: \sum_i alpha_i^2 ||Q_i Y_bar^T||^2."""
        a1, a2 = alpha
        return a1 ** 2 * yq1_norm2 + a2 ** 2 * yq2_norm2

    # setup starting point & constraints
    x0 = np.zeros(2)
    constraints = [dict(type='eq', fun=constraint)]

    # optimize
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='delta_grad == 0.0')
        res = minimize(fun=obj, x0=x0, constraints=constraints,
                       method='trust-constr',
                       options=dict(maxiter=10000))
    assert res.success, 'optimization failed'

    # flip signs on alpha to ensure that 1 + alpha_i is not negative
    x_opt = res.x
    for idx, alpha_i in enumerate(x_opt[:2]):
        # post-hoc fix so 1+alpha_i >= 0 for i=0,1
        if 1 + alpha_i < 0:
            x_opt[idx] = -2 - alpha_i
    assert np.isclose(obj(res.x), obj(x_opt)), 'alpha sign flip failure'


    # compute offset
    alpha1, alpha2 = x_opt
    offset = alpha1 * y_mean @ q[1].T @ q[1] + \
             alpha2 * y_mean @ q[2].T @ q[2]

    return offset
