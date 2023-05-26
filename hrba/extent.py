import numpy as np
from scipy.ndimage.morphology import binary_dilation


class ExtenterSphere:
    """ builds effect extent as a randomly placed sphere """

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, mask_idx, seed=None):
        """ returns a mask of extent """
        # choose a random initial voxel
        rng = np.random.default_rng(seed=seed)
        mask_bool = mask_idx > -1
        vox_init = rng.choice(mask_idx[mask_bool])

        # dilate to full extent
        mask = mask_idx == vox_init
        mask = binary_dilation(mask, iterations=self.radius)

        # ensure extent doesn't exceed original mask
        return np.logical_and(mask, mask_bool)
