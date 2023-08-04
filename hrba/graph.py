from collections import defaultdict

import numpy as np

from hrba.f_stat import RegStatComputer


def iter_edge(children, num_vox):
    """ iterates through all edges

    Args:
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_)
        num_vox (int): number of single voxel regions

    Yields:
        child_idx (int): child index
        parent_idx (int): parent index
    """
    for idx, child_vec in enumerate(children):
        parent_idx = num_vox + idx
        for c in child_vec:
            yield c, parent_idx


def node_sum(x, children):
    """ given item per leaf in graph, sums leaf values and adds to dictionary

    Args:
        x (np.array): value associated with each leaf (single voxel region)
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_)

    Returns:
        summed (np.array): same structure as input, but now includes all
            nodes, not just leafs
    """
    # prep output array
    num_leaf = x.size
    num_reg = num_leaf + children.shape[0]
    summed = np.empty(num_reg, dtype=x.dtype)
    summed[:num_leaf] = x

    # sum
    for node_idx, (c0, c1) in enumerate(children):
        node_idx += num_leaf
        summed[node_idx] = summed[c0] + summed[c1]

    return summed


def get_f1(mask, mask_idx, children):
    """ computes f1 score per region

    Args:
        mask (np.array): target mask (boolean, same shape as mask_idx)
        mask_idx (np.array):
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_)

    Returns:
        f1 (np.array): f1 score per region
    """
    # compute misses & hits per region
    # true positive: target voxels in estimated region
    # false positive: in estimated region but not in target mask
    fp, tp = get_miss_hits(mask, mask_idx, children)

    # false negative: targets outside of estimated region
    fn = mask.sum() - tp

    return 2 * tp / (2 * tp + fp + fn)


def get_miss_hits(mask, mask_idx, children):
    """ for each node, count how many voxels are in / out of effect

    Args:
        mask (np.array): target mask (boolean, same shape as mask_idx)
        mask_idx (np.array):
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_)

    Returns:
        miss (np.array): non-target voxels contained in node of segmentation
        hit (np.array): target voxels contained in node of segmentation
    """
    # build miss and hit for leaf nodes
    num_vox = (mask_idx >= 0).sum()
    hit = np.zeros(num_vox)
    hit[mask_idx[mask.astype(bool)]] = 1
    miss = np.ones(num_vox) - hit

    # sum to all other regions
    miss = node_sum(miss, children=children)
    hit = node_sum(hit, children=children)

    return miss, hit


def iter_topo(children, num_leaf=None, node_start=None, only_leaf=False):
    """ topological sort, leafs to root

    Args:
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_)
        num_leaf (int): number of leafs in graph
        node_start (int): starting node, iterates over all nodes below
        only_leaf (bool): if True, only leaves are yielded

    Yields:
        node_idx (int): node idx
    """
    if num_leaf is None:
        # assumes that graph is complete
        num_leaf = children.shape[0] + 1

    if node_start is None:
        # will search largest node (whole thing if graph connected)
        node_start = num_leaf + children.shape[0] - 1

    if node_start >= num_leaf:
        for child in children[int(node_start - num_leaf), :]:
            yield from iter_topo(children, num_leaf=num_leaf, node_start=child,
                                 only_leaf=only_leaf)

    if not only_leaf or node_start < num_leaf:
        yield node_start


def iter_reg_stat_exp(*, exp, **kwargs):
    yield from iter_reg_stat(x=exp.x, y=exp.y, contrast=exp.contrast, **kwargs)


def iter_reg_stat(x, y, contrast, children=None, include_leaf=True, **kwargs):
    """ iterates through region statistics of graph

    Args:
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities
        contrast (np.array): (a) True for each corresponding feature in x which
            is "of interest" (other x features form the reduced model in
            computing f statistic)
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_).  defaults to None, where reg stats
            are computed per voxel
        include_leaf (bool): toggles inclusion of leafs

    Yields:
        reg_idx (int): region index
        reg_stat (dict): keys are str, values are stats.  see RegStatComputer

    as well as all outputs of StatComputer (e.g. f_stat & log like ratio)
    """
    # prep
    b, num_img, num_vox = y.shape
    rs_computer = RegStatComputer(x=x, contrast=contrast, **kwargs)

    # init mean of y outer products
    myo_dict = defaultdict(lambda: 0)
    myo = np.einsum('ijk,ajk->iak', y, y) / num_img
    myo_dict.update(enumerate(np.rollaxis(myo, 2, 0)))

    # init size_dict & y_mean_dict
    size_dict = dict()
    y_mean_dict = defaultdict(lambda: 0)

    if children is None:
        # no graph passed, iterate through each voxel
        iter_reg_idx = range(num_vox)
    else:
        # graph passed, iterate through all regions
        iter_reg_idx = iter_topo(children=children)

    for reg_idx in iter_reg_idx:
        if include_leaf and reg_idx < num_vox:
            # single voxel region (don't delete on lookup)
            y_mean = y[:, :, reg_idx]
            reg_size = 1

        else:
            # multiple voxel region

            # compute reg_size & lam (% region from each child)
            child = children[reg_idx - num_vox, :]
            size = tuple(size_dict.pop(c, 1) for c in child)
            reg_size = sum(size)
            size_dict[reg_idx] = reg_size
            lam = tuple(s / reg_size for s in size)

            # compute & store myo, y_mean
            for c, l in zip(child, lam):
                if c < num_vox:
                    # child has 1 voxel, direct lookup
                    _y_mean = y[:, :, c]
                else:
                    # child has many voxels, pop
                    _y_mean = y_mean_dict.pop(c)

                # update y_mean & myo
                y_mean_dict[reg_idx] += _y_mean * l
                myo_dict[reg_idx] += myo_dict.pop(c) * l

            y_mean = y_mean_dict[reg_idx]

        # compute & yield stats
        reg_stat = rs_computer(reg_size=reg_size,
                               y_mean=y_mean,
                               myo=myo_dict[reg_idx])
        yield reg_idx, reg_stat


def child_to_parent(children, num_leaf=None):
    """ children points parents to child, parent point child to parent

    Args:
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_)

    Returns:
        parent (np.array): (num_reg) parent of every region
    """
    if num_leaf is None:
        # assumes complete binary tree
        num_leaf = children.shape[0] + 1

    # init parent
    num_reg = num_leaf + children.shape[0]
    parent = np.ones(num_reg, dtype=int) * np.nan

    # store parents
    for idx, children in enumerate(children):
        reg_idx = idx + num_leaf
        parent[children] = reg_idx

    return parent


def iter_ancestor(parent, node, include_self=True):
    if include_self:
        yield node

    while True:
        node = parent[int(node)]
        if np.isnan(node):
            break
        yield node
