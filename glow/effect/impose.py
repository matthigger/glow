"""Solve for the offset that imposes a target effect size on imaging data."""

import warnings

import numpy as np
from scipy.optimize import minimize

from glow.analysis.mancova import decompose, get_llr


def compute_offset(x, y, contrast, effect_llr: float):
    """Find the smallest offset to y that imposes a given effect strength.

    The target is expressed as size-normalized LLR:
    (1/2) * ln|det(I + E^{-1}H)|.

    Note that effect_llr here is per-voxel-equivalent: the LLR you
    will observe for the planted region under H1 is approximately
    effect_llr * |region| (because the un-normalized LLR carries an
    n-prefactor that this routine divides out by calling
    get_llr(e, h, n=1)). So asking for effect_llr=0.5 on a
    614-voxel region plants a region whose downstream observed LLR is
    ~307, not 0.5.

    Args:
        x (np.array): (a, num_img) design matrix
        y (np.array): (b, num_img, num_vox) image intensities
        contrast (np.array): (a,) boolean, True for features of interest
        effect_llr (float): target size-normalized log-likelihood ratio
            (i.e. per-voxel LLR contribution; multiply by region size to
            get the LLR you will observe for the region)

    Returns:
        offset (np.array): (b, num_img) constant offset across voxels
        sigma_scale (float | None): always None (kept for API compatibility)
    """
    b, num_img, num_vox = y.shape

    q = decompose(x, contrast)
    y_mean = y.mean(axis=2)
    yq1_norm2 = ((y_mean @ q[1].T) ** 2).sum()
    yq2_norm2 = ((y_mean @ q[2].T) ** 2).sum()
    yq1q1y = y_mean @ q[1].T @ q[1] @ y_mean.T
    yq2q2y = y_mean @ q[2].T @ q[2] @ y_mean.T

    yr = y.reshape((b, -1), order='F')
    sigma_orig = yr @ yr.T - y_mean @ y_mean.T * num_vox

    return _solve_offset_only(
        y_mean, q, yq1_norm2, yq2_norm2, yq1q1y, yq2q2y,
        sigma_orig, num_vox, effect_llr,
    )


def _solve_offset_only(y_mean, q, yq1_norm2, yq2_norm2, yq1q1y, yq2q2y,
                        sigma_orig, num_vox: int, effect_llr: float):
    """Solve the 2-variable offset optimisation, holding sigma fixed.

    Minimises the offset norm over the interest/nuisance scalings
    (alpha1, alpha2) subject to the LLR matching effect_llr; sigma_scale
    is always 1 (returned as None).

    Args:
        y_mean (np.array): (b, num_img) mean image over voxels
        q: QR bases (q1 interest, q2 nuisance) from decompose
        yq1_norm2 (float): squared norm of y_mean projected onto q1
        yq2_norm2 (float): squared norm of y_mean projected onto q2
        yq1q1y (np.array): (b, b) interest outer product
        yq2q2y (np.array): (b, b) nuisance outer product
        sigma_orig (np.array): (b, b) within-region scatter
        num_vox (int): number of voxels in the region
        effect_llr (float): target size-normalized log-likelihood ratio

    Returns:
        offset (np.array): (b, num_img) constant offset across voxels
        sigma_scale (float | None): always None (kept for API compatibility)
    """

    def get_e_h(alpha):
        alpha1, alpha2 = alpha
        h = (1 + alpha1) ** 2 * yq1q1y * num_vox
        e = (1 + alpha2) ** 2 * yq2q2y * num_vox + sigma_orig
        return e, h

    def constraint(alpha):
        e, h = get_e_h(alpha)
        return get_llr(e, h, n=1) - effect_llr

    def obj(alpha):
        a1, a2 = alpha
        return a1 ** 2 * yq1_norm2 + a2 ** 2 * yq2_norm2

    x0 = np.zeros(2)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='delta_grad == 0.0')
        warnings.filterwarnings('ignore', message='Singular Jacobian')
        res = minimize(fun=obj, x0=x0,
                       constraints=[dict(type='eq', fun=constraint)],
                       method='trust-constr',
                       options=dict(maxiter=10000))
    assert res.success, 'optimization failed'

    x_opt = _fix_alpha_signs(res.x, obj)
    alpha1, alpha2 = x_opt
    offset = alpha1 * y_mean @ q[1].T @ q[1] + \
             alpha2 * y_mean @ q[2].T @ q[2]
    return offset, None


def _fix_alpha_signs(alpha, obj_fn):
    """Ensure 1 + alpha_i >= 0 by flipping signs (preserves objective).

    Args:
        alpha (np.array): (2,) the (alpha1, alpha2) scalings to correct
        obj_fn (Callable): objective used to assert the flip is value-preserving

    Returns:
        x_opt (np.array): (2,) sign-corrected scalings
    """
    x_opt = alpha.copy()
    for idx in range(min(2, len(x_opt))):
        if 1 + x_opt[idx] < 0:
            x_opt[idx] = -2 - x_opt[idx]
    assert np.isclose(obj_fn(alpha), obj_fn(x_opt)), 'alpha sign flip failure'
    return x_opt
