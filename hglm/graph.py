import numpy as np


def iter_size_yout_ybar(y, children=None, perm=None,
                        block_exchange=False, **kwargs):
    """ iterates through region stats, less-redundant compute via graph

    Args:
        y (np.array): (b, num_img, num_vox) imaging features
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_).  If None is passed,
            then iterates through stats per individual voxel region only
        perm (Permuter): if passed, operates on y
        block_exchange (bool): toggles block exchange permuting. when True,
            every voxel of a multi-voxel region utilizes same permutation
            matrix.  when False, each voxel gets its own permutation matrix

    Yields:
        reg_idx (int): region index
        size (int): size, in voxels, of region
        yout (np.array): (b, b) sum of yv @ yv.T across all voxels of region
        ybar (np.array): (b, num_img) average, across voxels, of features
    """
    b, num_img, num_vox = y.shape

    size_yout_ybar = dict()
    for reg_idx in iter_topo(children=children, num_leaf=num_vox):
        if reg_idx < num_vox:
            # single voxel region
            size = 1
            yv = y[:, :, reg_idx]
            ybar = yv

            if perm is None:
                # compute yout (no permutation needed)
                yout = yv @ yv.T
            else:
                # permute & compute yout
                perm_idx_min = 0 if block_exchange else reg_idx
                ybar = perm(ybar, perm_idx_min=reg_idx, **kwargs)
                yout = np.einsum('bnp,cnp->bcp', ybar, ybar, optimize=True)

            size_yout_ybar[reg_idx] = size, yout, ybar
        else:
            # multi voxel region
            # look up stats of constituent regions
            c0, c1 = children[int(reg_idx - num_vox), :]
            size0, yout0, ybar0 = size_yout_ybar.pop(c0)
            size1, yout1, ybar1 = size_yout_ybar.pop(c1)

            # compute & store stats of their union
            size = size0 + size1
            yout = yout0 + yout1
            lam = size0 / size, size1 / size
            ybar = ybar0 * lam[0] + ybar1 * lam[1]
            size_yout_ybar[reg_idx] = size, yout, ybar

        yield reg_idx, size, yout, ybar


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


def iter_topo(*, children=None, num_leaf, node_start=None, only_leaf=False):
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
    if children is None:
        # no graph passed, iterate through leaves
        yield from range(num_leaf)
        return

    if node_start is None:
        # will search largest node (whole thing if graph connected)
        node_start = num_leaf + children.shape[0] - 1

    if node_start >= num_leaf:
        for child in children[int(node_start - num_leaf), :]:
            yield from iter_topo(children=children, num_leaf=num_leaf,
                                 node_start=child, only_leaf=only_leaf)

    if not only_leaf or node_start < num_leaf:
        yield node_start


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
