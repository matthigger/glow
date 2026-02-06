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

    # compute sigma_sum (non-normalised spatial covariance)
    y_mean = y.mean(axis=2)
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
    """
    a = (~contrast).sum(), contrast.size
    to_sorted = np.eye(a[1])[np.argsort(contrast), :]
    q, r = np.linalg.qr((to_sorted @ x).T, mode='complete')
    q = q.T
    return q[:a[0], :], q[a[0]: a[1], :], q[a[1]:, :]


# ---------------------------------------------------------------------------
# four MANCOVA test statistics
# ---------------------------------------------------------------------------

def get_neg_wilks(e, h):
    """negative Wilks' lambda (larger = more evidence against H0).

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix

    Returns:
        float: -det(E) / det(E + H)
    """
    sign_e, logdet_e = np.linalg.slogdet(e)
    sign_t, logdet_t = np.linalg.slogdet(e + h)
    return -np.exp(logdet_e - logdet_t)


def get_pillai(e, h):
    """Pillai's trace: tr((H + E)^-1 H).

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix

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


def get_hotel_tr(e, h):
    """Hotelling-Lawley trace: tr(E^-1 H).

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix

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


def get_roys_root(e, h):
    """Roy's largest root: max eigenvalue of E^-1 H.

    Args:
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix

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


stat_dict = {'Wilks': get_neg_wilks,
             'Pillai': get_pillai,
             'Hotelling Tr': get_hotel_tr,
             'Roy Root': get_roys_root}
