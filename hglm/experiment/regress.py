import numpy as np


def get_manova(x, y, contrast):
    b, num_img, num_vox = y.shape

    # compute sigma_sum (non-normalized spatial covariance)
    y_mean = y.mean(axis=2)
    yr = y.reshape((b, -1), order='F')
    sigma_sum = yr @ yr.T - y_mean @ y_mean.T * num_vox

    # compute observed e and h
    q = decompose(x, contrast)
    yq1 = y_mean @ q[1].T
    yq2 = y_mean @ q[2].T
    h = num_vox * yq1 @ yq1.T
    e = sigma_sum +  num_vox * yq2 @ yq2.T

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


def get_size_yout_ymean(y):
    """ compute size yout ymean directly from imaging features

    Args:
        y (np.array): (b, num_img, num_vox) imaging features

    Returns:
        size (int): size, in voxels, of region
        yout (np.array): (b, b) sum of yv @ yv.T across all voxels of region
        ymean (np.array): (b, num_img) average, across voxels, of features
    """
    size = y.shape[2]
    yout = np.einsum('bnr,anr->ba', y, y, optimize=True)
    ymean = y.mean(axis=2)
    return size, yout, ymean


def get_sigma(size, yout, ymean):
    """ sigma is spatial covariance across voxels, pooled across images

    inputs efficiently computed for hierarchical regions, see
    hglm.graph.iter_size_yout_ymean()

    Args:
        size (int): size, in voxels, of region
        yout (np.array): (b, b) sum of yv @ yv.T across all voxels of region
        ymean (np.array): (b, num_img) average, across voxels, of features

    Returns:
        sigma (np.array): (b, b) spatial covariance, pooled across images
    """
    num_img = ymean.shape[1]
    return (yout / size - ymean @ ymean.T) / num_img


def get_sigma_from_y(y):
    size, yout, ymean = get_size_yout_ymean(y)
    return get_sigma(size=size, yout=yout, ymean=ymean)


def get_rough(x, y):
    """ computes roughness coefficient

    rough = num_img * tr_sigma / ||q2 y_mean ||^2
    """
    a, _ = x.shape
    b, num_img, num_vox = y.shape
    tr_sigma = np.trace(get_sigma_from_y(y))

    q, r = np.linalg.qr(x.T, mode='complete')
    q = q.T
    q2 = q[a:, :]
    y_mean = y.mean(axis=2)
    yq2 = np.linalg.norm(y_mean @ q2.T) ** 2

    return tr_sigma * num_img / yq2


def scale_sigma(y, gain=None, tr_sigma=None):
    """ change space cov in y: multiply by gain or impose given tr_cov

    Args:
        y (np.array): (b, num_img, num_vox) image intensities
        gain (float): non-negative scaling factor
            space_cov_tr_out / space_cov_tr_in
        tr_sigma (float): desired spatial covariance trace

    Returns:
        y (np.array): (b, num_img, num_vox) image intensity, with scaling
            applied
    """
    assert (gain is None) != (tr_sigma is None), 'gain xor tr_cov required'

    b, num_img, num_vox = y.shape

    # de-mean
    mean = y.mean(axis=2)
    y_demean = y - mean[:, :, np.newaxis]

    if gain is None:
        # compute gain (if needed)
        _y = y_demean.reshape((b, -1))
        tr_sigma_in = np.trace(_y @ _y.T) / (num_img * num_vox)
        gain = tr_sigma / tr_sigma_in

    # apply gain
    y_demean *= np.sqrt(gain)

    # re-mean
    return y_demean + mean[:, :, np.newaxis]


def wilks_to_chi2(wilks, a, b, n):
    df = a * b
    chi2 = -(n - 0.5 * (a + b + 1)) * np.log(wilks)
    return chi2, df


def chi2_to_wilks(chi2, a, b, n):
    scale = -(n - 0.5 * (a + b + 1))
    return np.exp(chi2 / scale)


def get_wilks(e, h):
    return np.linalg.det(e) / np.linalg.det(e + h)
