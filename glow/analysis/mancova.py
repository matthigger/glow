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


def is_intercept_only_nuisance(x, contrast):
    """True when the nuisance projection Q0 commutes with every permutation.

    Q0 commutes with all permutation matrices iff its column space is a
    permutation-invariant subspace.  Over the full symmetric group on
    num_img coordinates, the only such subspaces are spanned by the
    all-ones vector — i.e., the nuisance must be "intercept-only":
    every nuisance column is constant across images.

    When this holds, several quantities computed inside the FL inner
    loop become deterministically invariant under permutation
    (notably ``yout`` and ``t = yout - a0 a0.T / size``), enabling a
    Phase-1-precompute fast path in ``AnalysisGLOW``.  For non-constant
    nuisance columns Q0 does not commute and the fast path is invalid.

    Args:
        x (np.array): (a, num_img) design matrix
        contrast (np.array): (a,) boolean, True for features of interest

    Returns:
        bool — True iff every nuisance column (x rows where contrast is
        False) is constant across its num_img entries.
    """
    x_nuis = x[~contrast]
    if x_nuis.shape[0] == 0:
        return True
    # Use a relative tolerance keyed to the largest entry in each column,
    # so this works for x in either float32 or float64.
    return all(np.allclose(row, row[0]) for row in x_nuis)


# ---------------------------------------------------------------------------
# five MANCOVA test statistics
# ---------------------------------------------------------------------------

def get_llr(e, h, n=None):
    """Log-likelihood ratio: (n/2) * ln|det(I + E^{-1}H)|.

    Equivalent to LL_full - LL_null where both likelihoods are
    Gaussian profile log-likelihoods on the same region.  The (n/2)
    prefactor makes LLR scale linearly with region size under H0.
    Pass ``n=1`` for a size-independent (per-voxel) LLR.

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        n (int): number of voxels in the region; use ``n=1`` for a
            size-independent result

    Returns:
        float
    """
    sign_e, logdet_e = np.linalg.slogdet(e)
    sign_t, logdet_t = np.linalg.slogdet(e + h)
    if sign_e <= 0 or sign_t <= 0:
        return np.nan
    return 0.5 * n * (logdet_t - logdet_e)


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


stat_dict = {
    'llr': get_llr,
    'wilks': get_wilks,
    'pillai': get_pillai,
    'hotel_tr': get_hotel_tr,
    'roys_root': get_roys_root,
}

stat_dict_inv = {fn: name for name, fn in stat_dict.items()}
