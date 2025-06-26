import warnings

import numpy as np
import scipy.stats
from scipy.optimize import minimize, root_scalar
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from hglm.experiment import decompose, get_manova, Permuter, \
    get_f_ratio


def estimate_f_kde(x, y, contrast, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed=seed)
    perm = Permuter(x=x[~contrast, :])
    perm_idx_min = rng.integers(1,
                                np.iinfo(np.uint32).max,
                                dtype=np.uint32)
    y_perm = perm(y, n_perm=n_perm, perm_idx_min=perm_idx_min, keep_orig=True)

    stat = list()
    for perm_idx in range(n_perm):
        # compute / record wilks
        e, h = get_manova(x=x, y=y_perm[..., perm_idx], contrast=contrast)
        stat.append(get_f_ratio(e, h))

    # sklearn: estimate distribution (kde of gaussians, grid search bandwidth)
    stat = np.array(stat).reshape(-1, 1)
    scott_bw = np.std(stat, ddof=1) * n_perm ** (-1 / 5)
    bw = np.logspace(-0.7, 0.7, 21) * scott_bw
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bw}, cv=5)
    grid.fit(stat)
    kde = grid.best_estimator_

    # scipy: find wilks value which gives proper cdf
    bw_factor = kde.bandwidth / np.std(stat)
    return scipy.stats.gaussian_kde(stat.flatten(), bw_method=bw_factor)


def pval_to_f_ratio(f_kde, pval):
    assert 0 <= pval < 1, 'invalid p-value given'

    # clip cdf: re-normalize so cdf starts at 0
    cdf = lambda z: f_kde.integrate_box_1d(-np.inf, z)
    cdf0 = cdf(0)
    cdf_clip = lambda z: max((cdf(z) - cdf0) / (1 - cdf0), 0)

    # find root
    obj = lambda z: cdf_clip(z) - (1 - pval)
    res = root_scalar(obj, xtol=1e-6, method='brentq',
                      bracket=[0, 10 * f_kde.dataset.max()])
    assert res.converged, 'optimization failed: target_wilks'

    return res.root


def compute_offset(x, y, contrast, f_ratio):
    """ get offset to y, constant across voxels, which imposes an f-stat

    Args:
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities
        contrast (np.array): (a) True for each corresponding feature in x which
            is "of interest" (other x features form the reduced model in
            computing f statistic)
        f_ratio (float): f_ratio to be achieved by offset

    Returns:
        offset (np.array): (b, num_img) offset to apply to all images to
            produce desired pval
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
        """" computes e, h, sigma_sum under a given alpha
        """
        alpha1, alpha2 = alpha
        h = (1 + alpha1) ** 2 * yq1q1y * num_vox
        e = (1 + alpha2) ** 2 * yq2q2y * num_vox + sigma_sum
        return e, h, sigma_sum

    def constraint(alpha):
        """ when this function output is zero, chi2_target achieved """
        e, h, _ = get_e_h_sigma(alpha)
        _f_ratio = get_f_ratio(e, h)
        return _f_ratio - f_ratio

    def obj(alpha):
        r"""  ||\Delta||^2 = \sum_{i=1}^3 \alpha_i^2 ||Q_i \bar{Y}_r^T||^2

        Args:
             alpha (tuple): a1, a2 (per equations)
        """
        a1, a2 = alpha
        return a1 ** 2 * yq1_norm2 + a2 ** 2 * yq2_norm2

    # setup starting point & constraints (assuming no rough constraint)
    x0 = np.zeros(2)
    constraints = [dict(type='eq', fun=constraint)]

    # optimize
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='delta_grad == 0.0')
        res = minimize(fun=obj, x0=x0, constraints=constraints,
                       method='trust-constr',
                       options=dict(maxiter=10000))
    assert res.success, 'optimization failed'

    # compute offset
    alpha1, alpha2 = res.x
    offset = alpha1 * y_mean @ q[1].T @ q[1] + \
             alpha2 * y_mean @ q[2].T @ q[2]

    return offset
