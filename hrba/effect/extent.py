import numpy as np
from scipy.ndimage.morphology import binary_dilation


class ExtenterSphere:
    """ builds effect extent as a randomly placed sphere """

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, mask_idx, y=None, seed=None):
        """ returns a mask of extent

        Args:
            mask_idx (np.array): same shape as image.  -1 where voxel not
                included in analysis, otherwise contains voxel index
            y (np.array): (b, num_img, num_vox) image intensities
            seed: initializes random number generator (given seed function is
                deterministic)

        Returns:
            mask (np.array): same shape as image.  boolean, True within extent
        """
        # choose a random initial voxel
        rng = np.random.default_rng(seed=seed)
        mask_bool = mask_idx > -1
        vox_init = rng.choice(mask_idx[mask_bool])

        # dilate to full extent
        mask = mask_idx == vox_init
        mask = binary_dilation(mask, iterations=self.radius)

        # ensure extent doesn't exceed original mask
        return np.logical_and(mask, mask_bool)


def iter_vox_neighbor(mask, mask_idx):
    """ iterates through voxel idx of all neighbors of mask

        Args:
            mask (np.array): same shape as image, boolean
            mask_idx (np.array): same shape as image.  -1 where voxel not
                included in analysis, otherwise contains voxel index

        Yields:
            vox_idx (int): voxel index (in mask_idx) corresponding which
                neighbors the mask (neighbor not reflexive)
    """
    mask_neighbor = binary_dilation(mask) & np.logical_not(mask)
    for vox_idx in mask_idx[mask_neighbor]:
        if vox_idx > -1:
            yield vox_idx


class ExtenterMinVar:
    """ grows effect extent from single voxel to greedily minimize var (trace)
    """

    def __init__(self, n):
        self.n = int(n)

    def __call__(self, y, mask_idx, seed=None, vox_init=None):
        """ returns a mask of extent

        Args:
            mask_idx (np.array): same shape as image.  -1 where voxel not
                included in analysis, otherwise contains voxel index
            y (np.array): (b, num_img, num_vox) image intensities
            seed: initializes random number generator (given seed function is
                deterministic)
            vox_init (int): seed voxel (if not passed, then randomly chosen)

        Returns:
            mask (np.array): same shape as image.  boolean, True within extent
        """

        # choose a random initial voxel
        if vox_init is None:
            rng = np.random.default_rng(seed=seed)
            mask_bool = mask_idx > -1
            vox_init = rng.choice(mask_idx[mask_bool])
        mask = mask_idx == vox_init
        assert mask.sum(), 'vox_init not in mask_idx'

        mu = y[:, :, vox_init]
        y_norm_sq = (mu ** 2).sum()

        for n in range(1, self.n):
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


            # add vox_idx_best to mask & update vec_stat
            mask[mask_idx == vox_idx_best] = True
            y_new = y[:, :, vox_idx_best]
            mu = lam0 * mu + lam1 * y_new
            y_norm_sq += (y_new ** 2).sum()

        return mask
