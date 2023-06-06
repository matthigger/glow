import numpy as np

def get_mask_idx(mask):
    """ builds mask_idx from mask

    Args:
        mask (np.array): boolean, False where voxels not to be analyzed

    Returns:
        mask_idx (np.array): -1 where voxels not to be analyzed, all other
            voxels get a unique integer (voxel index)
    """
    mask = mask.astype(bool)
    mask_idx = np.full(mask.shape, fill_value=-1)
    num_vox = mask.sum()
    mask_idx[mask] = np.arange(num_vox)
    return mask_idx