"""Spatial Gaussian smoothing of imaging data.

Smoothing is the preprocessing a mass-univariate analysis leans on: it
raises the signal-to-noise ratio of a blob wider than a voxel, and makes
the residual field smooth enough for the random-field results (Worsley et
al. 1996) that voxel-wise inference grew out of. GLOW needs none of it
(the Ward tree pools over regions directly), so a kernel is a knob of the
voxel-wise arms alone.

fwhm is given in mm and converted to a per-axis voxel sigma against the
affine, so one fwhm means one physical kernel whatever the voxel size.

Smoothing is normalized within the mask: the smoothed data is divided by
the identically smoothed mask indicator, so a voxel near the edge averages
its in-mask neighbours instead of being pulled toward zero by the outside.
That correction is one positive factor per voxel, shared by every image,
which the MANCOVA statistics divide out. What it buys is that a constant
stays constant, and so a pre-scaling commutes with the kernel exactly (see
ExperimentScaled).

Everything here smooths exactly the voxels it is handed, diluting the edge
of the mask against whatever sits outside it. Experiment.smooth is the
entry point that borrows the context to make that edge honest.
"""

import numpy as np
import scipy.ndimage

# Each 1d pass is truncated at int(TRUNCATE * sigma + 0.5) voxels. Shared by
# the filter and halo_vox_needed, so a halo cannot disagree with the kernel
# it was cut for. 4.0 is scipy's own default.
TRUNCATE = 4.0

# fwhm = sigma * sqrt(8 ln 2), the full width at half maximum of a Gaussian
_FWHM_PER_SIGMA = np.sqrt(8 * np.log(2))


def get_vox_mm(affine=None):
    """Return the (3,) voxel edge lengths in mm of a voxel-to-mm affine.

    Args:
        affine (np.array): (4, 4) voxel-to-mm affine, or None for 1 mm
            isotropic

    Returns:
        vox_mm (np.array): (3,) edge length per axis
    """
    if affine is None:
        return np.ones(3)
    return np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))


def get_sigma_vox(fwhm: float, affine=None):
    """Return the (3,) per-axis kernel sigma in voxels.

    Args:
        fwhm (float): kernel full width at half maximum, in mm
        affine (np.array): (4, 4) voxel-to-mm affine; None is 1 mm
            isotropic

    Returns:
        sigma_vox (np.array): (3,) standard deviation per axis, in voxels
    """
    return fwhm / _FWHM_PER_SIGMA / get_vox_mm(affine)


def halo_vox_needed(fwhm: float, affine=None) -> int:
    """Return the halo depth a kernel needs to smooth an edge exactly.

    The filter is separable, three 1d passes of radius
    int(TRUNCATE * sigma + 0.5), so what it reads is the cube of that
    radius and not the ball: the depth returned here is a Chebyshev radius,
    and a halo grown under any other metric leaves the cube's corners
    outside the data.

    Args:
        fwhm (float): kernel full width at half maximum, in mm
        affine (np.array): (4, 4) voxel-to-mm affine; None is 1 mm
            isotropic

    Returns:
        halo_vox (int): voxels of context needed around the voxels under
            study, the widest axis deciding
    """
    sigma_vox = get_sigma_vox(fwhm, affine)
    return int((TRUNCATE * sigma_vox + 0.5).astype(int).max())


def smooth_volume(vol, fwhm: float, affine=None):
    """Gaussian-smooth every 3d volume in a stack.

    Args:
        vol (np.array): (..., X, Y, Z) volumes; the leading axes are
            looped over, each (X, Y, Z) smoothed independently
        fwhm (float): kernel full width at half maximum, in mm
        affine (np.array): (4, 4) voxel-to-mm affine; None is 1 mm
            isotropic

    Returns:
        out (np.array): (..., X, Y, Z), same dtype as vol
    """
    sigma = (0.0,) * (vol.ndim - 3) + tuple(get_sigma_vox(fwhm, affine))
    out = scipy.ndimage.gaussian_filter(vol, sigma=sigma, truncate=TRUNCATE,
                                        mode='constant', cval=0.0)
    return out.astype(vol.dtype, copy=False)


def _bbox(mask):
    """Return the slice tuple tightly containing mask.

    What the kernel runs over. Filtering the bounding box rather than the
    whole grid is exact, not an approximation: outside the box every voxel
    is out of mask, so the zeros a constant-mode filter reads past the
    box's edge are the zeros it would have read there anyway.

    Args:
        mask (np.array): (X, Y, Z) boolean, at least one voxel True

    Returns:
        box (tuple[slice]): one slice per axis
    """
    return tuple(slice(int(w.min()), int(w.max()) + 1)
                 for w in np.where(mask))


def _to_volume(y, mask_idx):
    """Scatter (b, num_img, num_vox) data into (b, num_img, X, Y, Z).

    Args:
        y (np.array): (b, num_img, num_vox) image intensities
        mask_idx (np.array): (X, Y, Z) int, -1 outside the mask

    Returns:
        vol (np.array): (b, num_img, X, Y, Z), zero outside the mask
    """
    vol = np.zeros((*y.shape[:2], *mask_idx.shape), dtype=y.dtype)
    sel = mask_idx > -1
    vol[:, :, sel] = y[:, :, mask_idx[sel]]
    return vol


def _from_volume(vol, mask_idx):
    """Gather (b, num_img, X, Y, Z) data back into (b, num_img, num_vox).

    Args:
        vol (np.array): (b, num_img, X, Y, Z) image intensities
        mask_idx (np.array): (X, Y, Z) int, -1 outside the mask

    Returns:
        y (np.array): (b, num_img, num_vox), ordered by mask_idx
    """
    sel = mask_idx > -1
    y = np.empty((*vol.shape[:2], int(sel.sum())), dtype=vol.dtype,
                 order='F')
    y[:, :, mask_idx[sel]] = vol[:, :, sel]
    return y


def smooth_y(y, mask_idx, fwhm: float, affine=None):
    """Gaussian-smooth each image within the mask.

    Args:
        y (np.array): (b, num_img, num_vox) image intensities
        mask_idx (np.array): (X, Y, Z) int, -1 outside the mask
        fwhm (float): kernel full width at half maximum, in mm. 0 or None
            returns y unchanged.
        affine (np.array): (4, 4) voxel-to-mm affine; None is 1 mm
            isotropic

    Returns:
        y_smooth (np.array): (b, num_img, num_vox), same dtype as y
    """
    if not fwhm:
        return y

    sel = mask_idx > -1
    if not sel.any():
        return y
    # the values still index y's columns, so slicing the grid down to what
    # is masked changes what the kernel runs over and nothing else
    mask_idx = mask_idx[_bbox(sel)]

    vol = _to_volume(y, mask_idx)
    vol = smooth_volume(vol, fwhm=fwhm, affine=affine)

    # normalize within the mask, so its edge averages in-mask neighbours
    # rather than the zeros _to_volume left outside (see the module
    # docstring)
    sel = mask_idx > -1
    weight = smooth_volume(sel.astype(vol.dtype), fwhm=fwhm, affine=affine)
    vol[:, :, sel] /= weight[sel]

    return _from_volume(vol, mask_idx)
