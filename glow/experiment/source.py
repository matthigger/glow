"""Reloadable handles on the images an experiment was cut from.

An Experiment keeps only the voxels it analyses: apply_mask crops it,
drop_constant_vox screens it, and the benchmark cuts a 25k-voxel sphere out
of a 225k-voxel brain. A source is what lets one grow back, so a spatial
filter can read real neighbours past the analysis support.

A source holds no image data. It carries what a loader needs to run again
(which files, which features, in what order, or which seed), so it pickles
small and resolves its own paths on whatever machine reads it. Growth stops
at the source's own support, which for a brain mask is the brain.

Concrete sources mirror the loaders: SourceGauss for
ExperimentImageOnly.from_gauss, SourceNifti for .from_paths, and
SourceBundle for the npy bundle, beside the bundle it reads
(glow._extra.benchmark.hcp).
"""

from abc import ABC, abstractmethod

import nibabel as nib
import numpy as np
import scipy.linalg


class ImageSource(ABC):
    """Metadata naming an experiment's images, able to read them again.

    Concrete sources implement load; the three attributes below are the
    rest of the contract.

    Attributes:
        mask (np.array): (X, Y, Z) boolean, the source's whole support.
            An inflate can reach no further than this.
        features (tuple): feature names, in the order load stacks them
        affine (np.array): (4, 4) voxel-to-mm transform, or None where the
            images carry none
    """

    affine = None

    def __deepcopy__(self, memo):
        """Return self: a source is immutable metadata, so a copy is it.

        _copy_with deepcopies an experiment's whole __dict__, and a source
        that held an array (SourceNifti's support is one) would be copied
        with every crop. Nothing mutates a source, so sharing one is both
        cheaper and truer: two experiments cut from the same images do
        read the same images.
        """
        return self

    @abstractmethod
    def load(self, mask=None):
        """Read the images over mask into a y array.

        Args:
            mask (np.array): (X, Y, Z) boolean subset of self.mask, or
                None for the whole support

        Returns:
            y (np.array): (b, num_img, num_vox) intensities, voxels
                ordered as glow.mask.get_mask_idx numbers them over mask
        """

    def _as_subset(self, mask):
        """Return mask as a boolean array, checked to sit inside the support.

        Args:
            mask (np.array): (X, Y, Z) boolean, or None for the whole
                support

        Returns:
            mask (np.array): (X, Y, Z) boolean

        Raises:
            ValueError: mask is the wrong shape, or asks for a voxel the
                source does not have
        """
        support = self.mask
        if mask is None:
            return support
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != support.shape:
            raise ValueError(f'mask shape {mask.shape} does not match the '
                             f'source support {support.shape}')
        if (mask & ~support).any():
            n_out = int((mask & ~support).sum())
            raise ValueError(f'{n_out} of the requested voxels lie outside '
                             f'the source support')
        return mask


def sample_gauss(*, b: int, num_img: int, shape: tuple, seed: int = None,
                 mu=None, cov=None, dtype=np.float32):
    """Draw Gaussian images with prescribed sample mean and covariance.

    The draw is centred, and covariance-projected, over every voxel at
    once, so it is only reproducible whole: a subset is taken by drawing
    the box and gathering (see SourceGauss).

    Args:
        b (int): imaging features
        num_img (int): images to sample
        shape (tuple): spatial shape of each image
        seed (int): random seed
        mu (np.array): (b,) target sample mean, or None for zeros
        cov (np.array): (b, b) target sample covariance, or None for the
            identity
        dtype: numpy dtype of the returned array

    Returns:
        y (np.array): (b, num_img, num_vox) intensities, num_vox the
            voxels of shape in C order
    """
    num_vox = np.prod(shape)
    rng = np.random.default_rng(seed=seed)
    y = rng.multivariate_normal(np.zeros(b), np.eye(b), num_img * num_vox)

    y = y.reshape((b, num_img, num_vox))
    y = y - y.mean(axis=(1, 2))[:, np.newaxis, np.newaxis]
    y = y.reshape((b, -1))

    if cov is not None:
        # project to proper covariance
        _cov = y @ y.T / (num_img * num_vox - 1)
        p = np.linalg.inv(scipy.linalg.sqrtm(_cov))
        p = scipy.linalg.sqrtm(cov) @ p
        y = p @ y

    if mu is not None:
        # add to proper mean
        y = y + mu[:, np.newaxis]

    return y.reshape((b, num_img, num_vox)).astype(dtype, copy=False)


class SourceGauss(ImageSource):
    """Gaussian images, redrawn from the seed that first sampled them.

    Nothing is stored and nothing is read from disk: load redraws the whole
    box and gathers the voxels asked for (see sample_gauss for why the
    whole box).

    Attributes:
        shape (tuple): spatial shape of each image
        b (int): imaging features
        num_img (int): images
        seed (int): the draw's seed; None draws fresh, and such a source
            reloads different images, so from_gauss leaves it set
        mu (np.array): (b,) target sample mean, or None
        cov (np.array): (b, b) target sample covariance, or None
        dtype: numpy dtype of the drawn y
    """

    def __init__(self, *, shape: tuple, b: int, num_img: int,
                 seed: int = None, mu=None, cov=None, dtype=np.float32):
        """Store the draw parameters (see the class Attributes)."""
        self.shape = tuple(int(n) for n in shape)
        self.b = int(b)
        self.num_img = int(num_img)
        self.seed = seed
        self.mu = mu
        self.cov = cov
        self.dtype = dtype

    def __repr__(self):
        """A compact identity string: the draw parameters."""
        return (f'{type(self).__name__}(shape={self.shape}, b={self.b}, '
                f'num_img={self.num_img}, seed={self.seed})')

    @property
    def mask(self):
        """Return the (X, Y, Z) boolean support: every voxel of the box."""
        return np.ones(self.shape, dtype=bool)

    @property
    def features(self) -> tuple:
        """Return the feature names from_gauss labels the draw with."""
        return tuple(f'feat_{i}' for i in range(self.b))

    def load(self, mask=None):
        """Redraw the box and gather mask's voxels (see ImageSource.load)."""
        mask = self._as_subset(mask)
        y = sample_gauss(b=self.b, num_img=self.num_img, shape=self.shape,
                         seed=self.seed, mu=self.mu, cov=self.cov,
                         dtype=self.dtype)
        if mask.all():
            return y
        # mask.ravel() is C order and so is the drawn voxel axis, which is
        # the order get_mask_idx numbers a mask in
        return y[:, :, mask.ravel()]


class SourceNifti(ImageSource):
    """NIfTI images on disk, read again one file at a time.

    Mirrors the NIfTI branch of ExperimentImageOnly.from_paths: subjects
    sorted, features in the path table's column order, each file read with
    get_fdata(dtype) so the on-disk scaling is applied as the loader
    applied it.

    A voxel subset costs a whole file. nibabel's ArrayProxy refuses fancy
    indexing, and a gzipped NIfTI inflates from the start whatever slice is
    asked for, so load reads each volume and gathers from it.

    Attributes:
        paths (pd.DataFrame): index=subject, columns=feature, values=paths
        mask (np.array): (X, Y, Z) boolean support, as the loader realized
            it (an explicit brain mask, or the every-image-nonzero rule
            load_image_nii falls back to)
        dtype: numpy dtype the images are read as
        affine (np.array): (4, 4) voxel-to-mm transform
    """

    def __init__(self, paths, *, mask, dtype=np.float32, affine=None):
        """Store the path table, the realized support, and the read dtype.

        Args:
            paths (pd.DataFrame): index=subject, columns=feature, values
                are file paths
            mask (np.array): (X, Y, Z) boolean support the loader realized
            dtype: numpy dtype to read the images as
            affine (np.array): (4, 4) voxel-to-mm transform, or None
        """
        self.paths = paths
        self.mask = np.asarray(mask, dtype=bool)
        self.dtype = dtype
        self.affine = affine

    def __repr__(self):
        """A compact identity string: the table's dimensions."""
        num_img, b = self.paths.shape
        return f'{type(self).__name__}(num_img={num_img}, b={b})'

    @property
    def features(self) -> tuple:
        """Return the feature names, in the path table's column order."""
        return tuple(self.paths.columns)

    def load(self, mask=None):
        """Read each volume and gather mask (see ImageSource.load)."""
        mask = self._as_subset(mask)
        subjects = sorted(self.paths.index)
        num_vox = int(mask.sum())
        y = np.empty((len(self.features), len(subjects), num_vox),
                     dtype=self.dtype)
        for feat_idx, feat in enumerate(self.features):
            for sbj_idx, sbj in enumerate(subjects):
                img = nib.load(self.paths.loc[sbj, feat])
                arr = img.get_fdata(dtype=self.dtype)
                y[feat_idx, sbj_idx, :] = arr[mask]
                del arr, img
        return y
