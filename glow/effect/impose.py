"""Solve for the offset that imposes a target effect size on imaging data."""

import warnings

import numpy as np
from scipy.optimize import brentq, minimize

from glow.analysis.mancova import decompose, get_llr, get_mancova


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

    # Warm-start onto the constraint manifold: at alpha = 0 the constraint
    # gradient is the baseline interest-to-error ratio, which underflows
    # trust-constr's gtol when the region mean is near-orthogonal to the
    # interest contrast, stalling the solve on a flat start.
    e0 = yq2q2y * num_vox + sigma_orig
    alpha1_ws = _warm_start_alpha1(y_mean @ q[1].T, e0, num_vox, effect_llr)
    x0 = np.array([alpha1_ws, 0.0])
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


def _warm_start_alpha1(m, e0, num_vox: int, effect_llr: float) -> float:
    """Interest scaling that hits effect_llr with the nuisance held fixed.

    Warm-starts _solve_offset_only. Holding alpha2 = 0 (the error matrix
    then staying e0), the LLR reduces to a monotone equation in
    t = (1 + alpha1)**2,
        (1/2) sum_i ln(1 + t mu_i) = effect_llr,
    with mu_i the eigenvalues of num_vox m.T E0^-1 m. Its root is the
    feasible interest scaling on that slice. Same LLR-in-t inversion as
    _solve_alpha, with the region mean's own un-normalised interest
    projection m as the direction, so the returned scale is 1 + alpha1.

    Args:
        m (np.array): (b, a1) interest projection of the region mean
            (y_mean q1.T)
        e0 (np.array): (b, b) error matrix at alpha2 = 0 (nuisance unscaled)
        num_vox (int): voxel count of the region
        effect_llr (float): target size-normalized LLR

    Returns:
        alpha1 (float): interest scaling; 0.0 when the region mean carries no
            interest signal (mu = 0, so no positive target is reachable)
    """
    if effect_llr <= 0:
        return 0.0
    mu = np.linalg.eigvalsh(num_vox * (m.T @ np.linalg.solve(e0, m)))
    mu = mu[mu > 0]
    if mu.size == 0:
        return 0.0
    target = 2.0 * effect_llr
    if mu.size == 1:
        t = (np.exp(target) - 1.0) / mu[0]
    else:
        t_hi = (np.exp(target) - 1.0) / mu.max()
        t = brentq(lambda t: np.sum(np.log1p(t * mu)) - target, 0.0, t_hi)
    return float(np.sqrt(t) - 1.0)


def _fix_alpha_signs(alpha, obj_fn):
    """Ensure 1 + alpha_i >= 0 by flipping signs (preserves objective).

    Args:
        alpha (np.array): (2,) the (alpha1, alpha2) scalings to correct
        obj_fn (Callable): objective used to assert the flip preserves value

    Returns:
        x_opt (np.array): (2,) sign-corrected scalings
    """
    x_opt = alpha.copy()
    for idx in range(min(2, len(x_opt))):
        if 1 + x_opt[idx] < 0:
            x_opt[idx] = -2 - x_opt[idx]
    assert np.isclose(obj_fn(alpha), obj_fn(x_opt)), 'alpha sign flip failure'
    return x_opt


def sample_beta_direction(a1: int, b: int, angle: float, seed: int):
    """Sample a unit (a1, b) coefficient direction at a seeded rotation angle.

    Draws a fixed orthonormal pair (u_base, w) spanning a random 2-plane in
    the a1*b-dimensional coefficient space from seed -- crucially
    independent of angle -- then returns cos(angle) u_base + sin(angle) w,
    reshaped to (a1, b) and Frobenius-normalized. Because the plane depends
    only on (a1, b, seed), two draws sharing a seed are separated by exactly
    their angle difference: sample(a1, b, 0, s) and sample(a1, b, 60, s)
    span 60 degrees; sample(a1, b, -30, s) and sample(a1, b, 30, s) likewise.

    Args:
        a1 (int): interest-contrast count (coefficient rows)
        b (int): imaging-feature count (coefficient columns)
        angle (float): rotation in degrees from the base direction
        seed (int): seeds the (u_base, w) plane

    Returns:
        beta_direction (np.array): (a1, b) unit (Frobenius) coefficient
            direction

    Raises:
        ValueError: if a1*b < 2 (a one-dimensional coefficient space has a
            single direction, so no angle can be imposed)
    """
    dim = a1 * b
    if dim < 2:
        raise ValueError(
            f'a1*b={dim} < 2: a one-dimensional coefficient space has a '
            'single direction, so no angle can be imposed')

    theta = np.radians(angle)
    rng = np.random.default_rng(seed)

    # (u, w): an orthonormal basis for a random 2-plane. Both are drawn from
    # seed independent of angle, so draws sharing a seed sit exactly their
    # angle difference apart.
    u = rng.standard_normal(dim)
    u /= np.linalg.norm(u)
    w = rng.standard_normal(dim)
    w -= (w @ u) * u
    w /= np.linalg.norm(w)

    beta = np.cos(theta) * u + np.sin(theta) * w
    return (beta / np.linalg.norm(beta)).reshape(a1, b)


def _solve_alpha(e, d, num_vox: int, effect_llr: float) -> float:
    """Scale d so that imposing alpha*d hits the target per-voxel LLR.

    The size-normalized LLR of a region whose interest coefficient is
    alpha*d (d a unit (b, a1) direction) is, by Sylvester's identity,
        (1/2) sum_i ln(1 + alpha^2 mu_i),
    where mu_i are the eigenvalues of num_vox * d.T E^-1 d (a1 x a1, PD).
    Strictly increasing in alpha^2, so a unique alpha hits any positive
    target: closed-form for one interest contrast (a1 == 1), a bracketed
    1-D solve otherwise.

    Args:
        e (np.array): (b, b) region error matrix
        d (np.array): (b, a1) unit (Frobenius) coefficient direction
        num_vox (int): voxel count of the region
        effect_llr (float): target size-normalized LLR

    Returns:
        alpha (float): nonnegative scale for d
    """
    if effect_llr <= 0:
        return 0.0

    mu = np.linalg.eigvalsh(num_vox * (d.T @ np.linalg.solve(e, d)))
    target = 2.0 * effect_llr
    if mu.size == 1:
        return float(np.sqrt((np.exp(target) - 1.0) / mu[0]))

    # monotone in t = alpha^2; the largest eigenvalue's log term alone
    # reaches the target, so [0, t_hi] brackets the root
    t_hi = (np.exp(target) - 1.0) / mu.max()
    t = brentq(lambda t: np.sum(np.log1p(t * mu)) - target, 0.0, t_hi)
    return float(np.sqrt(t))


def impose_effect(x, y, contrast, *, beta_direction, effect_llr: float,
                  purge_interest: bool = True):
    """Scale a chosen coefficient direction to a target effect size.

    Plants a pure mean-shift along beta_direction in the interest subspace:
    the offset is beta_direction.T @ q1, which lies entirely in the row space
    of q1, so it grows the hypothesis matrix H without touching the error
    matrix E. A single scale alpha then hits the target size-normalized LLR
    (1/2) ln det(I + E^-1 H): with d the unit (Frobenius) coefficient
    direction (beta_direction.T normalized) and mu the eigenvalues of
    num_vox d.T E^-1 d, the LLR is (1/2) sum_i ln(1 + alpha^2 mu_i), strictly
    increasing in alpha, so alpha is unique (closed-form for one interest
    contrast, a bracketed 1-D solve for several -- see _solve_alpha).
    purge_interest subtracts the region's existing interest coefficient so
    the least-squares-recovered effect equals alpha * beta_direction.T
    exactly; otherwise the planted effect adds on top of the baseline.

    Args:
        x (np.array): (a, num_img) design matrix
        y (np.array): (b, num_img, num_vox) region image intensities
        contrast (np.array): (a,) boolean, True for the interest contrasts
        beta_direction (np.array): (a1, b) coefficient direction; magnitude
            is ignored. A (b,) vector is accepted as the a1 == 1 case.
        effect_llr (float): target size-normalized (per-voxel) LLR
        purge_interest (bool): if True, cancel the region's existing interest
            coefficient so the recovered effect is exactly along
            beta_direction; if False, add the effect on top of the baseline

    Returns:
        offset (np.array): (b, num_img) constant offset across voxels
    """
    b, num_img, num_vox = y.shape
    contrast = np.asarray(contrast)
    a1 = int(contrast.sum())

    beta_direction = np.asarray(beta_direction, dtype=float)
    if beta_direction.ndim == 1:
        beta_direction = beta_direction.reshape(1, b)
    assert beta_direction.shape == (a1, b), (
        f'beta_direction must be (a1, b)=({a1}, {b}), '
        f'got {beta_direction.shape}')

    q1 = decompose(x, contrast)[1]
    y_mean = y.mean(axis=2)
    e, _, _ = get_mancova(x=x, y=y, contrast=contrast)

    # only the direction of beta_direction matters; its magnitude is set by
    # alpha to hit effect_llr, so normalize (Frobenius) before scaling
    d = beta_direction.T / np.linalg.norm(beta_direction)
    alpha = _solve_alpha(e, d, num_vox, effect_llr)

    # The offset lives purely in span(q1); offset @ q1.T == coef_add, so the
    # least-squares interest coefficient becomes y_mean @ q1.T + coef_add.
    coef_add = alpha * d
    if purge_interest:
        coef_add = coef_add - y_mean @ q1.T
    return coef_add @ q1
