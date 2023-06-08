import numpy as np

from hrba.graph import iter_topo


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


def mask_from_tree(mask_idx, dendro, reg_idx):
    """ builds mask of single region from dendrogram (slow)

    Args:
        mask_idx (np.array): -1 where voxels not to be analyzed, all other
            voxels get a unique integer (voxel index)
        dendro (np.array): (num_leaf - 1, 2) dendrogram arrays (equiv to
            sklearn.cluster.Ward.children_)
        reg_idx (int): region index

    Returns:
        mask (np.array): False where region doesn't contain voxel,
            True otherwise
    """
    mask = np.zeros(mask_idx.shape, dtype=bool)
    num_leaf = dendro.shape[0] + 1

    for _reg_idx in iter_topo(dendro, node_start=reg_idx):
        if _reg_idx < num_leaf:
            mask[mask_idx == _reg_idx] = True

    return mask
