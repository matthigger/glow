"""MANCOVA statistics for permutation testing."""

import numpy as np


def get_mancova(*, x=None, y, contrast=None, q_tup=None):
    """Compute MANCOVA error (E), hypothesis (H) and spatial covariance.

    Exactly one of x or q_tup must be provided.

    Args:
        x (np.array): (a, num_img) design matrix
        y (np.array): (b, num_img, num_vox) image data
        contrast (np.array): (a,) boolean, True for features of interest.
            Required when x is given.
        q_tup (tuple): pre-computed (q0, q1, q2) QR basis from decompose()

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
    """Decompose the design matrix into an orthonormal basis via QR.

    Partitions x into nuisance, interest and residual spaces.

    Args:
        x (np.array): (a, num_img) design matrix
        contrast (np.array): (a,) boolean, True for features of interest

    Returns:
        q0 (np.array): (a0, num_img) nuisance subspace
        q1 (np.array): (a1, num_img) interest subspace
        q2 (np.array): (num_img - a, num_img) residual subspace

    Output dtype matches x.dtype -- important for AnalysisGLOW's hot
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


def is_intercept_only_nuisance(x, contrast) -> bool:
    """Return True when nuisance projection Q0 commutes with all permutations.

    Q0 commutes with all permutation matrices iff its column space is a
    permutation-invariant subspace. Over the full symmetric group on
    num_img coordinates, the only such subspaces are spanned by the
    all-ones vector -- i.e., the nuisance must be "intercept-only":
    every nuisance column is constant across images.

    When this holds, several quantities computed inside the FL inner
    loop become deterministically invariant under permutation
    (notably yout and t = yout - a0 a0.T / size), enabling a
    Phase-1-precompute fast path in AnalysisGLOW. For non-constant
    nuisance columns Q0 does not commute and the fast path is invalid.

    Args:
        x (np.array): (a, num_img) design matrix
        contrast (np.array): (a,) boolean, True for features of interest

    Returns:
        bool: True iff every nuisance column (x rows where contrast is
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

def get_llr(e, h, n: int = 1):
    """Compute the log-likelihood ratio: (n/2) * ln|det(I + E^{-1}H)|.

    Equivalent to LL_full - LL_null where both likelihoods are
    Gaussian profile log-likelihoods on the same region. The (n/2)
    prefactor makes LLR scale linearly with region size under H0.
    The default n=1 gives a size-independent (per-voxel) LLR.

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        n (int): number of voxels in the region; default n=1 gives a
            size-independent (per-voxel) result

    Returns:
        float: the log-likelihood ratio, or np.nan if E or E + H is
            not positive-definite
    """
    sign_e, logdet_e = np.linalg.slogdet(e)
    sign_t, logdet_t = np.linalg.slogdet(e + h)
    if sign_e <= 0 or sign_t <= 0:
        return np.nan
    return 0.5 * n * (logdet_t - logdet_e)


def get_wilks(e, h, n=None) -> float:
    """Compute 1 - Wilks' Lambda: 1 - det(E) / det(E + H).

    Values in [0, 1); larger = more evidence against H0.

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        n: unused (accepted for uniform stat-function interface)

    Returns:
        float: 1 - Wilks' Lambda, or np.nan when det(E) or det(E + H) is
            not positive
    """
    sign_e, logdet_e = np.linalg.slogdet(e)
    sign_t, logdet_t = np.linalg.slogdet(e + h)
    # Without this guard an indefinite E (float cancellation on a voxel
    # with almost no variance across images) sends |det E| far above
    # det(E + H), the difference of logs past 709, and the exp to +inf,
    # returning -inf where the statistic is simply undefined.
    if sign_e <= 0 or sign_t <= 0:
        return np.nan
    # expm1 rather than 1 - exp: E + H >= E for positive-semidefinite H,
    # so the ratio sits at or just below 1 and the log-difference at or
    # just below 0, exactly where 1 - exp(d) loses its leading digits.
    return float(-np.expm1(logdet_e - logdet_t))


def _finite_or_nan(value: float) -> float:
    """Fold a non-finite statistic into np.nan, the not-analysed sentinel.

    np.linalg.solve raises only on an exactly singular matrix. A merely
    near-singular E returns without complaint, and the trace of the
    solution can overflow to +-inf (or reach 1e200 on the way there). NaN
    is what the walk already uses for a region carrying no usable
    statistic (AnalysisVoxel._walk_stat), and what every reduction over
    the stat matrix skips, so non-finite output joins it here rather than
    travelling on to own a max-stat null.
    """
    return value if np.isfinite(value) else np.nan


def _safe_solve(m, h, label: str):
    """Solve m @ x = h; re-raise a descriptive error if m is singular.

    label names the singular matrix in the message, e.g. '(H + E) matrix'
    or 'error matrix E'.
    """
    try:
        return np.linalg.solve(m, h)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError(
            f'singular {label} (shape {m.shape}). '
            f'This typically means num_img <= b (too few images for the '
            f'number of features). Consider reducing b or adding more images.'
        )


def get_pillai(e, h, n=None) -> float:
    """Compute Pillai's trace: tr((H + E)^-1 H).

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        n: unused (accepted for uniform stat-function interface)

    Returns:
        float, or np.nan when H + E is near-singular enough to overflow

    Raises:
        np.linalg.LinAlgError: if H + E is singular
    """
    return _finite_or_nan(
        float(np.trace(_safe_solve(h + e, h, '(H + E) matrix'))))


def get_hotel_tr(e, h, n=None) -> float:
    """Compute the Hotelling-Lawley trace: tr(E^-1 H).

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        n: unused (accepted for uniform stat-function interface)

    Returns:
        float, or np.nan when E is near-singular enough to overflow

    Raises:
        np.linalg.LinAlgError: if E is singular
    """
    return _finite_or_nan(
        float(np.trace(_safe_solve(e, h, 'error matrix E'))))


def get_roys_root(e, h, n=None) -> float:
    """Compute Roy's largest root: the max eigenvalue of E^-1 H.

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
        n: unused (accepted for uniform stat-function interface)

    Returns:
        float, or np.nan when E is near-singular enough to overflow

    Raises:
        np.linalg.LinAlgError: if E is singular
    """
    eigvals = np.linalg.eigvals(_safe_solve(e, h, 'error matrix E'))
    return _finite_or_nan(float(np.max(np.real(eigvals))))


stat_dict = {
    'llr': get_llr,
    'wilks': get_wilks,
    'pillai': get_pillai,
    'hotel_tr': get_hotel_tr,
    'roys_root': get_roys_root,
}

stat_dict_inv = {fn: name for name, fn in stat_dict.items()}
