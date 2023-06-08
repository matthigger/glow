from collections import defaultdict

import numpy as np


def node_sum(dendro, val_dict):
    """ given item per leaf in tree, sums leaf values and adds to dictionary

    Args:
        dendro (np.array): (num_leaf - 1, 2) dendrogram arrays (equiv to
            sklearn.cluster.Ward.children_)
        val_dict (dict): keys are leaf indices (0 to num_leaf), values
            are items to be summed

    Returns:
        val_dict (dict): same structure as input, but now includes all
            nodes, not just leafs
    """
    num_leaf = len(val_dict)
    for node_idx, (c0, c1) in enumerate(dendro):
        node_idx += num_leaf
        new_val = val_dict[c0] + val_dict[c1]
        val_dict[node_idx] = new_val

    return val_dict


def get_hit_miss(mask, mask_idx, dendro):
    """ for each node, count how many voxels are in / out of effect

    Args:
        mask (np.array): target mask (boolean, same shape as mask_idx)
        mask_idx (np.array):
        dendro (np.array): (num_leaf - 1, 2) dendrogram arrays (equiv to
            sklearn.cluster.Ward.children_)

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


def iter_topo(dendro, num_leaf=None, node_start=None):
    """ topological sort, leafs to root

    Args:
        dendro (np.array): (num_leaf - 1, 2) dendrogram arrays (equiv to
            sklearn.cluster.Ward.children_)
        num_leaf (int): number of leafs in tree
        node_start (int): starting node, iterates over all nodes below

    Yields:
        node_idx (int): node idx
    """
    if num_leaf is None:
        # assumes that dendrogram is complete
        num_leaf = dendro.shape[0] + 1

    if node_start is None:
        # will search largest node (whole thing if dendrogram connected)
        node_start = num_leaf + dendro.shape[0] - 1

    if node_start >= num_leaf:
        for child in dendro[node_start - num_leaf, :]:
            yield from iter_topo(dendro, num_leaf=num_leaf, node_start=child)

    yield node_start


def iter_reg_stat_exp(dendro, exp, **kwargs):
    yield from iter_reg_stat(dendro, x=exp.x, y=exp.y, contrast=exp.contrast,
                             **kwargs)


def iter_reg_stat(dendro, x, y, contrast, include_leaf=True):
    """ iterates through region statistics of graph

    Args:
        dendro (np.array): (num_leaf - 1, 2) dendrogram arrays (equiv to
            sklearn.cluster.Ward.children_)
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities
        contrast (np.array): (a) True for each corresponding feature in x which
            is "of interest" (other x features form the reduced model in
            computing f statistic)
        include_leaf (bool): toggles inclusion of leafs

    Yields:
        reg_idx (int): region index
        size (int): number of voxels in region
        f_stat (tuple): f statistic
    """
    # prep
    num_leaf = dendro.shape[0] + 1
    b, num_img, reg_size = y.shape
    x = x[~contrast, :], x
    h = tuple(np.linalg.pinv(_x) @ _x for _x in x)
    a = (~contrast).sum(), contrast.size

    # init mean of squared y
    msy = (y ** 2).sum(axis=(0, 1)) / num_img
    msy_dict = defaultdict(lambda: 0)
    msy_dict.update(enumerate(msy))

    # init size_dict & y_mean_dict
    size_dict = dict()
    y_mean_dict = defaultdict(lambda: 0)

    def _compute_f_stat(msy, y_mean, reg_size):
        const = (reg_size * num_img - b * a[1]) / b * (a[1] - a[0])
        tr_eps = msy - np.trace(y_mean @ h[0] @ y_mean.T) / num_img, \
                 msy - np.trace(y_mean @ h[1] @ y_mean.T) / num_img
        return (tr_eps[0] - tr_eps[1]) / tr_eps[1] * const

    for reg_idx in iter_topo(dendro):
        if include_leaf and reg_idx < num_leaf:
            # single voxel region (don't delete on lookup)
            f_stat = _compute_f_stat(msy=msy_dict[reg_idx],
                                     y_mean=y[:, :, reg_idx],
                                     reg_size=1)
            yield reg_idx, 1, f_stat
        else:
            # multiple voxel region

            # compute reg_size & lam (% region from each child)
            child = dendro[reg_idx - num_leaf, :]
            size = tuple(size_dict.pop(c, 1) for c in child)
            reg_size = sum(size)
            size_dict[reg_idx] = reg_size
            lam = tuple(s / reg_size for s in size)

            # compute & store msy, y_mean
            for c, l in zip(child, lam):
                msy_dict[reg_idx] += msy_dict.pop(c) * l
                if c < num_leaf:
                    # child has 1 voxel, direct lookup
                    y_mean_dict[reg_idx] += y[:, :, c] * l
                else:
                    # child has many voxels, pop
                    y_mean_dict[reg_idx] += y_mean_dict.pop(c) * l

            # compute f_stat
            f_stat = _compute_f_stat(msy=msy_dict[reg_idx],
                                     y_mean=y_mean_dict[reg_idx],
                                     reg_size=reg_size)

            yield reg_idx, reg_size, f_stat
