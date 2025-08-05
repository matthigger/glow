import numpy as np
from scipy.ndimage import binary_dilation


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
