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
        q, r = np.linalg.qr(x.T, mode='reduced')
        self.h = q @ q.T

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

    def get_eps_mean(self, ybar):
        """ computes epsilon, the b x b error covariance matrix

        Args:
            ybar (np.array): (b, num_img) average, across voxels, of features

        Returns:
            eps (np.array): (b, b) error covariance matrix, per sample:
                (Y - \beta X) @ (Y - \beta X).T
        """
        num_img = ybar.shape[1]
        return (ybar @ (np.eye(num_img) - self.h) @ ybar.T) / num_img


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
