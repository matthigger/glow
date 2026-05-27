from functools import wraps
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.ndimage import label
from scipy.ndimage import binary_dilation, generate_binary_structure
from tqdm import tqdm

from ..util import HashBySlots

# Connectivity: 6-connectivity (face neighbors only) for 3D, matching Ward clustering
# This ensures effects grow and clustering merges using the same neighbor definition
CONNECTIVITY_3D = generate_binary_structure(3, 1)  # 6-connectivity (faces only)


class ContiguousRegionNotFound(RuntimeError):
    pass


@runtime_checkable
class Extenter(Protocol):
    """Sample a contiguous voxel mask defining an effect's spatial extent.

    All concrete extenters share the same call signature. ``y`` is required
    by data-driven extenters (e.g. ``ExtenterMinVar``) and ignored by
    geometric ones (e.g. ``ExtenterSphere``); pass it whenever it's
    available.
    """

    def __call__(self, *, mask_idx, y=None, seed=None, contiguous=False,
                 max_iter=100, **kwargs) -> np.ndarray:  # pragma: no cover
        ...


def resample_to_contiguous(fnc):
    """decorator: re-sample until the mask yields a contiguous region."""

    @wraps(fnc)
    def wrapped(self, *, mask_idx, seed=None, contiguous=False,
                max_iter=100, **kwargs):
        _seed = seed
        for _ in range(max_iter):
            mask = fnc(self, mask_idx=mask_idx, **kwargs, seed=_seed)

            if not contiguous:
                # user didn't insist on contiguous, no need to check
                return mask

            # ensure mask, when applied, yields contiguous region
            # Use 6-connectivity (face neighbors) to match clustering and effect growth
            structure = CONNECTIVITY_3D if (mask_idx.ndim == 3) else generate_binary_structure(2, 1)
            _, n_components = label((mask_idx >= 0) & mask, structure=structure)
            if n_components == 1:
                return mask

            # mask not contiguous, get a new seed (from previous) and try again
            rng = np.random.default_rng(seed=_seed)
            _seed = rng.integers(low=0, high=2 ** 32, size=1)[0]

        raise ContiguousRegionNotFound()

    return wrapped


class ExtenterSphere(HashBySlots):
    """build effect extent as a randomly placed sphere."""

    __slots__ = ('radius', 'n_vox', 'connected')

    def __init__(self, radius=None, n_vox=None, connected=False):
        if radius is None and n_vox is None:
            raise ValueError('radius or n_vox required')
        if radius is not None and n_vox is not None:
            raise ValueError('specify radius or n_vox, not both')
        self.radius = None if radius is None else int(radius)
        self.n_vox = None if n_vox is None else int(n_vox)
        self.connected = bool(connected)

    @resample_to_contiguous
    def __call__(self, mask_idx, y=None, seed=None, vox_init=None):
        """return a boolean mask defining the sphere extent.

        Args:
            mask_idx (np.array): voxel index array (-1 outside analysis)
            y (np.array): (b, num_img, num_vox) image intensities (unused)
            seed: random seed for reproducibility
            vox_init (int): seed voxel (random if not passed)

        Returns:
            mask (np.array): boolean, True within extent
        """
        # choose a random initial voxel
        rng = np.random.default_rng(seed=seed)
        mask_bool = mask_idx > -1
        structure = CONNECTIVITY_3D if (mask_idx.ndim == 3) else generate_binary_structure(2, 1)
        if self.connected:
            comp_idx, num_comp = label(mask_bool, structure=structure)
            if num_comp == 0:
                raise ValueError('no voxels available in mask')
            comp_sizes = np.bincount(comp_idx.ravel())
            comp_sizes[0] = 0
            if self.n_vox is not None:
                eligible = np.flatnonzero(comp_sizes >= self.n_vox)
                if eligible.size == 0:
                    raise ValueError('no connected component has >= n_vox voxels')
            else:
                eligible = np.flatnonzero(comp_sizes > 0)

            if vox_init is None:
                comp_id = rng.choice(eligible)
                mask_bool = comp_idx == comp_id
                vox_init = rng.choice(mask_idx[mask_bool])
            else:
                comp_id = comp_idx[mask_idx == vox_init]
                if comp_id.size == 0 or comp_id[0] == 0:
                    raise ValueError('vox_init not in mask')
                comp_id = comp_id[0]
                if self.n_vox is not None and comp_sizes[comp_id] < self.n_vox:
                    raise ValueError('vox_init component has fewer voxels than n_vox')
                mask_bool = comp_idx == comp_id
        else:
            if vox_init is None:
                vox_init = rng.choice(mask_idx[mask_bool])

        # dilate to full extent using 6-connectivity (face neighbors) for 3D
        mask = mask_idx == vox_init
        mask = np.logical_and(mask, mask_bool)

        if self.n_vox is None:
            mask = binary_dilation(mask, structure=structure, iterations=self.radius)
            # ensure extent doesn't exceed original mask
            return np.logical_and(mask, mask_bool)

        # grow until reaching desired voxel count
        count = int(mask.sum())
        while count < self.n_vox:
            prev_mask = mask
            mask = binary_dilation(mask, structure=structure, iterations=1)
            mask = np.logical_and(mask, mask_bool)
            count = int(mask.sum())

        # trim excess from outer shell
        excess = count - self.n_vox
        if excess > 0:
            shell = np.logical_and(mask, np.logical_not(prev_mask))
            shell_idx = np.flatnonzero(shell)
            if shell_idx.size >= excess:
                mask_flat = mask.ravel()
                mask_flat[shell_idx[:excess]] = False
                mask = mask_flat.reshape(mask.shape)
        return mask


def iter_vox_neighbor(mask, mask_idx):
    """yield voxel indices of all face-neighbours of mask.

    uses 6-connectivity for 3d, matching Ward clustering.

    Args:
        mask (np.array): boolean, same shape as image
        mask_idx (np.array): voxel index array (-1 outside analysis)

    Yields:
        vox_idx (int): neighbour voxel index (non-reflexive)
    """
    # Use 6-connectivity (face neighbors) for 3D, matching clustering connectivity
    structure = CONNECTIVITY_3D if (mask_idx.ndim == 3) else generate_binary_structure(2, 1)
    mask_neighbor = binary_dilation(mask, structure=structure) & np.logical_not(mask)
    for vox_idx in mask_idx[mask_neighbor]:
        if vox_idx > -1:
            yield vox_idx


class ExtenterMinVar(HashBySlots):
    """grow effect extent from a seed voxel to greedily minimise variance."""

    __slots__ = ('n_vox',)

    def __init__(self, n_vox):
        self.n_vox = int(n_vox)

    @resample_to_contiguous
    def __call__(self, mask_idx, y=None, seed=None, vox_init=None,
                 verbose=False):
        """return a boolean mask of n_vox voxels with minimal pooled variance.

        Args:
            mask_idx (np.array): voxel index array (-1 outside analysis)
            y (np.array): (b, num_img, num_vox) image intensities (required)
            seed: random seed for reproducibility
            vox_init (int): seed voxel (random if not passed)

        Returns:
            mask (np.array): boolean, True within extent
        """
        assert y is not None, 'ExtenterMinVar requires y'

        # choose a random initial voxel
        if vox_init is None:
            rng = np.random.default_rng(seed=seed)
            mask_bool = mask_idx > -1
            vox_init = rng.choice(mask_idx[mask_bool])
        mask = mask_idx == vox_init
        assert mask.sum(), 'vox_init not in mask_idx'

        mu = y[:, :, vox_init]
        y_norm_sq = (mu ** 2).sum()

        tqdm_dict = dict(total=self.n_vox - 1,
                         disable=not verbose,
                         desc='finding min var extent')
        for n in tqdm(range(1, self.n_vox), **tqdm_dict):
            # init
            min_var = np.inf
            vox_idx_best = None

            lam0 = n / (n + 1)
            lam1 = 1 / (n + 1)
            for vox_idx in iter_vox_neighbor(mask=mask, mask_idx=mask_idx):
                #
                y_new = y[:, :, vox_idx]
                _mu = lam0 * mu + lam1 * y_new
                var = ((y_norm_sq + (y_new ** 2).sum()) / (n + 1) -
                       (_mu ** 2).sum())

                # store it if new vox idx minimizes variance from among choices
                if var < min_var:
                    min_var = var
                    vox_idx_best = vox_idx

            assert vox_idx_best is not None

            # add vox_idx_best to mask & update mu & y_norm_sq
            mask[mask_idx == vox_idx_best] = True
            y_new = y[:, :, vox_idx_best]
            mu = lam0 * mu + lam1 * y_new
            y_norm_sq += (y_new ** 2).sum()

        return mask
