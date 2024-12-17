import numpy as np


class ComputeRegress:
    """ regression computations eps, llr (hierarchical & flat model)

    see hglm.graph.iter_size_yout_ybar(), which provides inputs efficiently

    epsilon is the error covariance matrix (mean outer product of residuals)

        num_img * num_vox * eps = sigma + eps_mean

    where sigma is the image pooled spatial covariance and

        eps_mean = ybar @ (I - q.T @ q) @ ybar.T / num_img

    Attributes:
        h (np.array): (num_img, num_img) hat matrix (multiply ybar to get
            the estimate: "hat"
    """

    def __init__(self, x):
        # compute hat matrix
        num_img = x.shape[1]
        q, r = np.linalg.qr(x.T, mode='reduced')
        self.h = q @ q.T
        self.i_minus_h = np.eye(num_img) - self.h

    def get_eps(self, size, yout, ybar):
        """ computes epsilon, the b x b error covariance matrix

        Args:
            size (int): size, in voxels, of region
            yout (np.array): (b, b) sum of yv @ yv.T across all voxels of region
            ybar (np.array): (b, num_img) average, across voxels, of features

        Returns:
            eps (np.array): (b, b) error covariance matrix, per sample:
                (Y - \beta X) @ (Y - \beta X).T
        """
        num_img = ybar.shape[1]
        return (yout / size - ybar @ self.h @ ybar.T) / num_img

    def get_eps_mean(self, size, yout, ybar):
        return ybar @ self.i_minus_h @ ybar.T


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


def get_size_yout_ybar(y):
    """ compute size yout ybar directly from imaging features

    Args:
        y (np.array): (b, num_img, num_vox) imaging features

    Returns:
        size (int): size, in voxels, of region
        yout (np.array): (b, b) sum of yv @ yv.T across all voxels of region
        ybar (np.array): (b, num_img) average, across voxels, of features
    """
    size = y.shape[2]
    yout = np.einsum('bnr,anr->ba', y, y, optimize=True)
    ybar = y.mean(axis=2)
    return size, yout, ybar


def get_sigma(size, yout, ybar):
    """ sigma is spatial covariance across voxels, pooled across images

    inputs efficiently computed for hierarchical regions, see
    hglm.graph.iter_size_yout_ybar()

    Args:
        size (int): size, in voxels, of region
        yout (np.array): (b, b) sum of yv @ yv.T across all voxels of region
        ybar (np.array): (b, num_img) average, across voxels, of features

    Returns:
        sigma (np.array): (b, b) spatial covariance, pooled across images
    """
    num_img = ybar.shape[1]
    return (yout / size - ybar @ ybar.T) / num_img


def get_sigma_from_y(y):
    size, yout, ybar = get_size_yout_ybar(y)
    return get_sigma(size=size, yout=yout, ybar=ybar)


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


def get_llr(eps0, eps1, size=None):
    """ computes log likelihood ratio between two models

    assumes that estimated error covariance has no error (covariance of samples
    cancels with covariance of normal distribution)

    Args:
        size (int): size of region
        eps0 (np.array): (b, b) error covariance matrix, reduced model
        eps1 (np.array): (b, b) error covariance matrix, full model

    Returns:
        llr (float): log likelihood ratio
    """
    return (log_det(eps0) - log_det(eps1)) * size / 2


def get_llr_hier(size, sigma, eps_mean0, eps_mean1):
    """ computes log likelihood ratio between two models

    Args:
        size (int): size of region
        sigma (np.array): (b, b) spatial  covariance matrix
        eps_mean0 (np.array): (b, b) error covariance matrix, reduced model
        eps_mean1 (np.array): (b, b) error covariance matrix, full model

    Returns:
        llr_hier (float): log likelihood ratio (hierarchical model)
    """
    eps_hier0 = eps_mean0 / size + sigma
    eps_hier1 = eps_mean1 / size + sigma
    return (log_det(eps_hier0) - log_det(eps_hier1)) / 2


def log_det(x):
    return np.log(np.linalg.det(np.atleast_2d(x)))
