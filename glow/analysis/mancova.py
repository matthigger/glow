"""MANCOVA statistics for permutation testing."""

import numpy as np


def get_mancova(*, x=None, y, contrast=None, q_tup=None):
    """compute MANCOVA error (E), hypothesis (H) and spatial covariance.

    exactly one of x or q_tup must be provided.

    Args:
        x (np.array): (a, num_img) design matrix
        y (np.array): (b, num_img, num_vox) image data
        contrast (np.array): (a,) boolean, True for features of interest.
            required when x is given.
        q_tup: pre-computed QR decomposition from decompose()

    Returns:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        sigma (np.array): (b, b) spatial covariance (un-normalised)
    """
    assert (x is None) != (q_tup is None), 'x xor q required'
    if q_tup is None:
        assert contrast is not None
        q_tup = decompose(x, contrast)

    b, num_img, num_vox = y.shape

    # compute sigma_sum (non-normalised spatial covariance).  Pass
    # dtype=y.dtype so numpy keeps the accumulator in y.dtype rather
    # than silently promoting float32 -> float64 (the default for
    # reduction-on-float32).
    y_mean = y.mean(axis=2, dtype=y.dtype)
    yr = y.reshape((b, -1), order='F')
    sigma = yr @ yr.T - y_mean @ y_mean.T * num_vox

    # compute observed e and h
    yq1 = y_mean @ q_tup[1].T
    yq2 = y_mean @ q_tup[2].T
    h = num_vox * yq1 @ yq1.T
    e = sigma + num_vox * yq2 @ yq2.T

    return e, h, sigma


def get_roughness(e, sigma):
    """Roughness coefficient: Tr(sigma) / Tr(E).

    Measures the fraction of the error matrix attributable to spatial
    (voxel-to-voxel) covariance vs the residual-mean projection.
    Values near 1 indicate spatially noisy ("rough") data; values near 0
    indicate error dominated by the between-image mean structure.

    Args:
        e (np.array): (b, b) error matrix
        sigma (np.array): (b, b) spatial covariance (un-normalised)

    Returns:
        float: roughness in [0, 1]
    """
    tr_e = np.trace(e)
    if tr_e == 0:
        return np.nan
    return float(np.trace(sigma) / tr_e)


def decompose(x, contrast):
    """decompose design matrix into orthonormal basis via QR.

    partitions x into nuisance, interest and residual spaces.

    Args:
        x (np.array): (a, num_img) design matrix
        contrast (np.array): (a,) boolean, True for features of interest

    Returns:
        q0 (np.array): nuisance subspace
        q1 (np.array): interest subspace
        q2 (np.array): residual subspace

    Output dtype matches ``x.dtype`` — important for AnalysisGLOW's hot
    loop, where the returned q matrices feed every-permutation einsums
    against y and a silent float32 -> float64 promotion would erase the
    bandwidth win from float32 y.
    """
    a = (~contrast).sum(), contrast.size
    # np.eye defaults to float64; match x.dtype so to_sorted @ x doesn't
    # promote a float32 design matrix.
    to_sorted = np.eye(a[1], dtype=x.dtype)[np.argsort(contrast), :]
    q, r = np.linalg.qr((to_sorted @ x).T, mode='complete')
    q = q.T
    return q[:a[0], :], q[a[0]: a[1], :], q[a[1]:, :]


# ---------------------------------------------------------------------------
# shared log-likelihood primitive
# ---------------------------------------------------------------------------

def loglik_from_cov(cov, n):
    """Gaussian profile log-likelihood from a covariance matrix.

    Returns -(n/2) * log|det(cov/n)|.  Additive constants (that
    cancel in all ratios) are omitted.

    Args:
        cov (np.array): (b, b) un-normalised covariance (e.g. E or E+H)
        n (int): number of voxels

    Returns:
        float: profile log-likelihood (higher = better fit)
    """
    s, logdet = np.linalg.slogdet(cov / n)
    if s <= 0:
        return np.nan
    return -0.5 * logdet * n


# ---------------------------------------------------------------------------
# five MANCOVA test statistics
# ---------------------------------------------------------------------------

def get_llr(e, h, n=None, *, size_normalize=False):
    """Log-likelihood ratio: (n/2) * ln|det(I + E^{-1}H)|.

    Equivalent to LL_full - LL_null where both likelihoods are
    Gaussian profile log-likelihoods on the same region.  The (n/2)
    prefactor makes LLR scale linearly with region size under H0.

    When size_normalize=True, returns (1/2) * ln|det(I + E^{-1}H)|
    instead (size-independent).

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        n (int): number of voxels in the region (unused when size_normalize)
        size_normalize (bool): if True, return size-independent LLR

    Returns:
        float
    """
    _n = 1 if size_normalize else n
    ll_alt = loglik_from_cov(e, _n)
    ll_null = loglik_from_cov(e + h, _n)
    if np.isnan(ll_alt) or np.isnan(ll_null):
        return np.nan
    return ll_alt - ll_null


def get_wilks(e, h, n=None):
    """1 - Wilks' Lambda: 1 - det(E) / det(E + H).

    Values in [0, 1); larger = more evidence against H0.

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        n: unused (accepted for uniform stat-function interface)

    Returns:
        float: 1 - Wilks' Lambda
    """
    sign_e, logdet_e = np.linalg.slogdet(e)
    sign_t, logdet_t = np.linalg.slogdet(e + h)
    return 1.0 - np.exp(logdet_e - logdet_t)


def get_pillai(e, h, n=None):
    """Pillai's trace: tr((H + E)^-1 H).

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        n: unused (accepted for uniform stat-function interface)

    Returns:
        float

    Raises:
        np.linalg.LinAlgError: if H + E is singular
    """
    try:
        return float(np.trace(np.linalg.solve(h + e, h)))
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError(
            f'singular (H + E) matrix (shape {e.shape}). '
            f'This typically means num_img <= b (too few images for the '
            f'number of features). Consider reducing b or adding more images.'
        )


def get_hotel_tr(e, h, n=None):
    """Hotelling-Lawley trace: tr(E^-1 H).

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        n: unused (accepted for uniform stat-function interface)

    Returns:
        float

    Raises:
        np.linalg.LinAlgError: if E is singular
    """
    try:
        return float(np.trace(np.linalg.solve(e, h)))
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError(
            f'singular error matrix E (shape {e.shape}). '
            f'This typically means num_img <= b (too few images for the '
            f'number of features). Consider reducing b or adding more images.'
        )


def get_roys_root(e, h, n=None):
    """Roy's largest root: max eigenvalue of E^-1 H.

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        n: unused (accepted for uniform stat-function interface)

    Returns:
        float

    Raises:
        np.linalg.LinAlgError: if E is singular
    """
    try:
        x = np.linalg.solve(e, h)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError(
            f'singular error matrix E (shape {e.shape}). '
            f'This typically means num_img <= b (too few images for the '
            f'number of features). Consider reducing b or adding more images.'
        )
    eigvals = np.linalg.eigvals(x)
    return float(np.max(np.real(eigvals)))


def llr_from_ysum_yout(ysum, yout, size, q_tup):
    """Compute LLR from pre-aggregated sufficient statistics.

    Avoids a full tree walk when only a single region's LLR is needed
    (e.g. for random re-partitions in the pruning permutation test).

    Args:
        ysum (np.array): (b, num_img) sum of y across voxels in the region
        yout (np.array): (b, b) sum of y @ y.T across voxels
        size (int): number of voxels in the region
        q_tup: (q0, q1, q2) from decompose()

    Returns:
        float: log-likelihood ratio
    """
    a0 = ysum @ q_tup[0].T
    t = yout - a0 @ a0.T / size
    a1 = ysum @ q_tup[1].T
    h = a1 @ a1.T / size
    e = t - h
    return get_llr(e, h, n=size)


stat_dict = {
    'llr': get_llr,
    'wilks': get_wilks,
    'pillai': get_pillai,
    'hotel_tr': get_hotel_tr,
    'roys_root': get_roys_root,
}

stat_dict_inv = {fn: name for name, fn in stat_dict.items()}
