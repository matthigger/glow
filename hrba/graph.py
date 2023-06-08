import numpy as np


def node_sum(dendro, val_dict):
    """ given item per leaf in tree, sums leaf values and adds to dictionary

    Args:
        dendro (np.array): (2, num_leaf) dendrogram arrays (equiv to
            sklearn.cluster.Ward.children_)
        val_dict (dict): keys are leaf indices (0 to num_leaf), values
            are items to be summed

    Returns:
        val_dict (dict): same structure as input, but now includes all
            nodes, not just leafs
    """
    num_leaf = len(val_dict)
    for node_idx, (c0, c1) in enumerate(dendro.T):
        node_idx += num_leaf
        new_val = val_dict[c0] + val_dict[c1]
        val_dict[node_idx] = new_val

    return val_dict


def get_hit_miss(mask, mask_idx, dendro):
    """ for each node, count how many voxels are in / out of effect

    Args:
        mask (np.array): target mask (boolean, same shape as mask_idx)
        mask_idx (np.array):
        dendro (np.array): dendrogram

    Returns:
        miss_hit_dict (dict): keys are node indices, values are (2) arrays
            containing number of voxels misses (effect voxels not in
            region) and hits (effect voxels in region)
    """
    # build miss_hit_dict for leaf nodes
    num_vox = (mask_idx >= 0).sum()
    hit = np.zeros(num_vox)
    hit[mask_idx[mask.astype(bool)]] = 1
    miss = np.ones(num_vox) - hit
    miss_hit_dict = dict(enumerate(np.vstack((miss, hit)).T))

    return node_sum(dendro=dendro, val_dict=miss_hit_dict)
