import numpy as np
from scipy.optimize import minimize
from scipy.stats import f

from hglm.f_stat import get_f_degrees, get_f_const


def compute_offset(x, y, contrast, f_stat=None, p_val=None, rough=None):
    """ get offset to y, constant across voxels, which imposes an f-stat

    Args:
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities
        contrast (np.array): (a) True for each corresponding feature in x which
            is "of interest" (other x features form the reduced model in
            computing f statistic)
        f_stat (float): target f statistic
        p_val (float): p-value may be passed in place of f stat
        rough (float): roughness coefficient desired.  a ratio of the spatial
            covariance over the power of error in the reduced model

            rough = sigma_tr / q2_norm2

            q2_norm2 is average power of the residuals in the reduced model,
            applied to the spatially averaged data

    Returns:
        offset (np.array): (b, num_img) offset to apply to all images to
            produce desired f stat
        rough (float): roughness coefficient achieved
        sigma_gain (float): spatial covariance scaling needed to achieve
            roughness coefficient, None if "rough" is not input.
    """
    assert (f_stat is None) != (p_val is None), 'f_stat xor p_val required'

    # prep constants
    a = (~contrast).sum(), contrast.size
    b, num_img, reg_size = y.shape

    # prep matrices
    y_mean = y.mean(axis=2)

    if f_stat is None:
        # get target f_stat to impose given p_value
        dfn, dfd = get_f_degrees(num_img, reg_size, a)
        f_stat = f.ppf(1 - p_val, dfn=dfn, dfd=dfd)

    # collect constants into k
    k = f_stat / get_f_const(y, contrast)

    # qr decomposition of x
    to_sorted = np.eye(a[1])[np.argsort(contrast), :]
    q, r = np.linalg.qr((to_sorted @ x).T, mode='complete')
    q = q.T
    q1 = q[a[0]: a[1], :]
    q2 = q[a[1]:, :]
    q1_norm2 = np.linalg.norm(q1 @ y_mean.T) ** 2
    q2_norm2 = np.linalg.norm(q2 @ y_mean.T) ** 2

    # compute norm of spatial cov square of t
    # |r| \Sigma_r & = \sum_{v \in r} (Y_v - \bar{Y}_r)(Y_v - \bar{Y}_r)^T
    #              & = Y_r Y_r^T - |r| \bar{Y}_r \bar{Y}_r^T
    yr = y.reshape((y.shape[0], -1), order='F')
    sigma = yr @ yr.T / reg_size - y_mean @ y_mean.T
    sigma_tr = np.trace(sigma)

    if rough is None:
        def constraint(alpha):
            """ when this function output is zero, F-stat is achieved """
            a1, a2 = alpha
            return k * sigma_tr + \
                k * (1 + a2) ** 2 * q2_norm2 - \
                (1 + a1) ** 2 * q1_norm2

    else:
        assert rough >= 0

        def constraint(alpha):
            """ when this function output is zero, F-stat is achieved """
            a1, a2 = alpha
            return k * (1 + rough) * (1 + a2) ** 2 * q2_norm2 - \
                (1 + a1) ** 2 * q1_norm2

    def obj(alpha):
        """  ||\Delta||^2 = \sum_{i=1}^3 \alpha_i^2 ||Q_i \bar{Y}_r^T||^2

        Args:
             alpha (tuple): a1, a2 (per equations)
        """
        a1, a2 = alpha
        return a1 ** 2 * q1_norm2 + a2 ** 2 * q2_norm2

    # find scale of each offset which achieves F while being as close as
    # possible to original problem
    res = minimize(fun=obj, x0=np.array([0, 0]),
                   constraints=[{'type': 'eq',
                                 'fun': constraint}],
                   options={'maxiter': 1000})
    assert res.success, 'optimization failed'

    a1, a2 = res.x
    offset = a1 * y_mean @ q1.T @ q1 + a2 * y_mean @ q2.T @ q2

    if rough is None:
        # compute roughness achieved (None was passed)
        rough = sigma_tr / (1 + a2) ** 2 * q2_norm2

        # no spatial covariance scaling needed
        sigma_gain = None

    else:
        # compute spatial covariance scaling assumed in computation above (to
        # achieve given roughness)
        sigma_gain = rough * (1 + a2) ** 2 * q2_norm2 / sigma_tr

    return offset, sigma_gain, rough
