import numpy as np


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
