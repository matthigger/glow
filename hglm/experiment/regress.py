import numpy as np


def get_manova(*, x=None, y, contrast=None, q_tup=None):
    assert (x is None) != (q_tup is None), 'x xor q required'
    if q_tup is None:
        assert contrast is not None
        q_tup = decompose(x, contrast)

    b, num_img, num_vox = y.shape

    # compute sigma_sum (non-normalized spatial covariance)
    y_mean = y.mean(axis=2)
    yr = y.reshape((b, -1), order='F')
    sigma_sum = yr @ yr.T - y_mean @ y_mean.T * num_vox

    # compute observed e and h
    yq1 = y_mean @ q_tup[1].T
    yq2 = y_mean @ q_tup[2].T
    h = num_vox * yq1 @ yq1.T
    e = sigma_sum + num_vox * yq2 @ yq2.T

    return e, h


def decompose(x, contrast):
    """ decomposes x into orthonormal basis

    Args:
        x (np.array): (a, num_img) explanatory variables
        contrast (np.array): (a) True for x features of interest
    """
    a = (~contrast).sum(), contrast.size
    to_sorted = np.eye(a[1])[np.argsort(contrast), :]
    q, r = np.linalg.qr((to_sorted @ x).T, mode='complete')
    q = q.T
    return q[:a[0], :], q[a[0]: a[1], :], q[a[1]:, :]


def get_wilks(e, h):
    sign_e, logdet_e = np.linalg.slogdet(e)
    sign_t, logdet_t = np.linalg.slogdet(e + h)

    return np.exp(logdet_e - logdet_t)


def get_f_ratio(e, h):
    sign_e, logdet_e = np.linalg.slogdet(e)
    sign_h, logdet_h = np.linalg.slogdet(h)

    return np.exp(logdet_h - logdet_e)
