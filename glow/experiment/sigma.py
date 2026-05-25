"""Spatial covariance helpers."""

import numpy as np


def get_sigma(size, yout, ymean):
    """compute pooled spatial covariance from pre-aggregated statistics.

    inputs are efficiently computed for hierarchical regions; see
    glow.graph.iter_size_ysum_yout.

    Args:
        size (int): number of voxels in the region
        yout (np.array): (b, b) sum of yv @ yv.T across all voxels
        ymean (np.array): (b, num_img) mean across voxels per feature

    Returns:
        sigma (np.array): (b, b) spatial covariance, pooled across images
    """
    num_img = ymean.shape[1]
    return (yout / size - ymean @ ymean.T) / num_img


def stretch_sigma(y, scale):
    """scale spatial covariance of y by a constant factor.

    de-means across voxels, multiplies by scale, then re-means.

    Args:
        y (np.array): (b, num_img, num_vox) image intensities
        scale (float): multiplicative factor applied to de-meaned signal

    Returns:
        y (np.array): (b, num_img, num_vox) with scaling applied
    """
    mean = y.mean(axis=2)
    y_demean = y - mean[:, :, np.newaxis]
    y_demean *= scale
    return y_demean + mean[:, :, np.newaxis]
