import numpy as np
from scipy.ndimage import binary_dilation


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


def get_rough(y, mask, mask_idx, n_shell=None):
    """ computes roughness coefficient: tr_sigma / tr_sigma_other

    where tr_sigma is the trace of sigma within the region and tr_sigma_other
    is the trace of sigma for "other", see n_shell below

    Args:
        y (np.array): (b, num_img, num_vox) imaging features
        mask (np.array): boolean mask, defines region
        mask_idx (np.array): same shape as image.  -1 where voxel not
            included in analysis, otherwise contains voxel index
        n_shell (int): defines comparison "other" region.  if None then all
            non-regions voxels are used, else a positive integer expected,
            defines the width of the outer shell used in comparison volume
    Returns:
        rough (float): tr_sigma / tr_sigma_other
    """
    num_vox = y.shape[2]

    def mask_to_bool(mask):
        # produces boolean index corresponding to given mask
        idx = mask_idx[mask]
        bool_reg = np.zeros(num_vox, dtype=bool)
        bool_reg[idx] = True
        return bool_reg

    # build bool_reg, boolean indexing for region
    bool_reg = mask_to_bool(mask)

    # build bool_other, boolean indexing for comparison volume
    if n_shell is None:
        bool_other = ~bool_reg
    else:
        mask_other = binary_dilation(mask, iterations=n_shell)
        mask_other = mask_other & ~mask & (mask_idx > -1)
        bool_other = mask_to_bool(mask_other)

    assert bool_reg.sum() > 1, 'region contains 1 or fewer voxels'
    assert bool_other.sum() > 1, 'other region contains 1 or fewer voxels'

    sigma_reg = get_sigma_from_y(y[:, :, bool_reg])
    sigma_other = get_sigma_from_y(y[:, :, bool_other])

    return np.trace(sigma_reg) / np.trace(sigma_other)


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


def wilks_to_chi2(wilks, b, n, contrast=None, a=None):
    assert (contrast is None) != (a is None), 'contrast xor a required'

    if a is None:
        a = contrast.sum()

    df = a * b
    scale = n - 1 - 0.5 * (b - a + 1)
    chi2 = - scale * np.log(wilks)
    return chi2, df


def get_wilks(e, h):
    sign_e, logdet_e = np.linalg.slogdet(e)
    sign_t, logdet_t = np.linalg.slogdet(e + h)

    return np.exp(logdet_e - logdet_t)


def get_f_ratio(e, h):
    sign_e, logdet_e = np.linalg.slogdet(e)
    sign_h, logdet_h = np.linalg.slogdet(h)

    return np.exp(logdet_h - logdet_e)
