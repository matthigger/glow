import numpy as np


def prep_get_eps(x):
    # compute hat matrix
    q, r = np.linalg.qr(x.T, mode='reduced')
    h = q @ q.T

    def get_eps(size, yout, ybar):
        """ computes epsilon, the b x b error covariance matrix

        inputs efficiently computed for hierarchical regions, see
        hglm.graph.iter_size_yout_ybar()

        Args:
            size (int): size, in voxels, of region
            yout (np.array): (b, b) sum of yv @ yv.T across all voxels of region
            ybar (np.array): (b, num_img) average, across voxels, of features

        Returns:
            eps (np.array): (b, b) error covariance matrix, per sample:
                (Y - \beta X) @ (Y - \beta X).T
        """
        num_img = ybar.shape[1]
        return (yout / size - ybar @ h @ ybar.T) / num_img

    return get_eps


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


def get_llr(size, eps0, eps1):
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
    return (np.log(np.linalg.det(eps0)) -
            np.log(np.linalg.det(eps1))) * size / 2
