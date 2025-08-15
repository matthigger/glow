from collections import defaultdict

import numpy as np

from hglm.experiment.mancova import decompose


def iter_size_e_h(*, x, contrast, **kwargs):
    """  wraps iter_size_yout_ymean, provides e and h matrices of mancova

    Args:
        see iter_size_yout_ymean

    Yields:
        reg_idx (int): region index
        size (int): size, in voxels, of region
        e (np.array): (b, b, num_perm) e of mancova, for every permutation
        h (np.array): (b, b, num_perm) h of mancova, for every permutation
    """
    # get projection matrices
    q = decompose(x=x, contrast=contrast)
    p1 = q[1].T @ q[1]
    p01 = p1 + q[0].T @ q[0]

    for reg_idx, size, yout, ymean in iter_size_yout_ymean(**kwargs):
        # move N up front to batch over it
        Y = np.transpose(ymean, (2, 0, 1))  # (N, X, B)

        # h = size * (Y @ p1) @ Y^T  for each n
        Yp1 = np.matmul(Y, p1)  # (N, X, B)
        H = size * np.matmul(Yp1, np.transpose(Y, (0, 2, 1)))  # (N, X, X)
        h = np.transpose(H, (1, 2, 0))  # (X, X, N)

        # proj = size * (Y @ p01) @ Y^T
        Yp01 = np.matmul(Y, p01)  # (N, X, B)
        PROJ = size * np.matmul(Yp01, np.transpose(Y, (0, 2, 1)))  # (N, X, X)
        proj = np.transpose(PROJ, (1, 2, 0))  # (X, X, N)

        e = yout - proj

        yield reg_idx, size, e, h


def iter_size_yout_ymean(y, children=None, perm=None,
                         block_exchange=True, **kwargs):
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
        yout (np.array): (b, b, num_perm) sum of yv @ yv.T across all voxels of
            region
        ymean (np.array): (b, num_img, num_perm) average, across voxels,
            of features
    """
    b, num_img, num_vox = y.shape

    size_yout_ymean = dict()
    max_reg = num_vox
    max_reg += children.shape[0] if children is not None else 0
    for reg_idx in range(max_reg):
        if reg_idx < num_vox:
            # single voxel region
            size = 1
            yv = y[:, :, reg_idx]
            ymean = yv

            if perm is None:
                # compute yout (no permutation needed)
                yout = yv @ yv.T

                # add new axis (consistent with permutations)
                yout = yout[:, :, np.newaxis]
                ymean = ymean[:, :, np.newaxis]
            else:
                # permute & compute yout
                perm_idx_min = 1 if block_exchange else reg_idx + 2
                ymean = perm(ymean, perm_idx_min=perm_idx_min, **kwargs)
                yout = np.einsum('bnp,cnp->bcp', ymean, ymean, optimize=True)

            size_yout_ymean[reg_idx] = size, yout, ymean
        else:
            # multi voxel region
            # look up stats of constituent regions
            c0, c1 = children[int(reg_idx - num_vox), :]
            size0, yout0, ymean0 = size_yout_ymean.get(c0)
            size1, yout1, ymean1 = size_yout_ymean.get(c1)

            # compute & store stats of their union
            size = size0 + size1
            yout = yout0 + yout1
            lam = size0 / size, size1 / size
            ymean = ymean0 * lam[0] + ymean1 * lam[1]
            size_yout_ymean[reg_idx] = size, yout, ymean

        yield reg_idx, size, yout, ymean


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


def get_f1_sens_spec(mask, mask_idx, children):
    """Computes F1, sensitivity (recall/TPR), and specificity (TNR) per region.

    Args:
        mask (np.array): target mask (boolean, same shape as mask_idx)
        mask_idx (np.array):
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_)

    Returns:
        f1 (np.array): f1 score per region
        sens (np.array):  TP / (TP + FN) per region
        spec (np.array): TN / (TN + FP) per region
    """
    # compute misses & hits per region
    # true positive: target voxels in estimated region
    # false positive: in estimated region but not in target mask
    fp, tp = get_miss_hits(mask, mask_idx, children)

    # false negative: targets outside of estimated region
    fn = mask.sum() - tp

    # true negative: everything else
    total = float(mask.size)
    tn = total - tp - fp - fn

    # metrics with safe division (0 where undefined)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 * tp / (2 * tp + fp + fn)
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)

    f1 = np.nan_to_num(f1, nan=0)
    sens = np.nan_to_num(sens, nan=0)
    spec = np.nan_to_num(spec, nan=1)

    return f1, sens, spec


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


def get_parent(children, num_leaf):
    """ gets parent lookup array for a binary tree

    Args:
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_)
        num_leaf (int): number of leafs in graph

    Returns:
        parent (np.array): parent[idx] gives the parent of node idx.
    """
    num_nodes = num_leaf + children.shape[0]
    parent = np.full(num_nodes, -1, dtype=int)
    for i, (c0, c1) in enumerate(children):
        parent[c0] = parent[c1] = num_leaf + i

    return parent


class RegIntersectError(Exception):
    pass


def get_label_map(reg_idx_list, mask_idx, children, check_disjoint=False):
    """Builds a mask_idx array from a list of region indices.

    Args:
        reg_idx_list (list[int]): list of region indices to include
        mask_idx (np.array): -1 outside of label_map, otherwise contains smallest
            reg_idx which contains this voxel in reg_idx_list
        children (num_node, 2): array whose i-th row represents
            node-n_common+i's children.  this representation contains a node
            for any node in all input graphs
        check_disjoint (bool): if True, ensure no regions intersect

    Returns:
        label_map (np.array): same shape as mask_idx, -1 outside regions,
            reg_idx where voxel belongs to that region (smallest reg_idx if
            intersections)
    """
    reg_idx_list = sorted(reg_idx_list, reverse=True)

    num_vox = (mask_idx > -1).sum()
    label_map = np.full(mask_idx.shape, -1, dtype=int)

    for reg_idx in reg_idx_list:
        for vox in iter_topo(children=children,
                             num_leaf=num_vox,
                             node_start=reg_idx,
                             only_leaf=True):
            target_voxels = (mask_idx == vox)
            if check_disjoint and np.any(label_map[target_voxels] != -1):
                reg_idx_list = np.unique(label_map[target_voxels])
                raise RegIntersectError(f'{reg_idx} intersects {reg_idx_list}')

            label_map[target_voxels] = reg_idx

    return label_map


def graph_merge(n_common, children_list):
    """ combines many binary trees into a graph with common indexing

    this assumes each children matrix in children_list is in topo order (comes
    out of sklearn's Ward's this way)

    Args:
        n_common (int): we assume the first n_idx_common nodes represent
            identical objects
        children_list (list): (n) each is a (2, num_node_n) array
            whose i-th col represents node n_idx_common + i's children

    Returns:
        map_to_new (list): (n) each is a (num_node_n) array whose
            i-th item represents the new index (in children below)
            of this input graph's n_idx_common + i-th node
        children (num_node, 2): array whose i-th row represents
            node-n_common+i's children.  this representation
            contains a node for any node in all input graphs
        size (num_node): number of items represented in each output node
    """
    # our first new node is n_common (smaller idx are common to all)
    node_idx = n_common

    # init outputs
    map_to_new = list()
    children = list()
    size = [1] * n_common

    # bijection from fset to node index
    fset_to_node = dict()
    node_to_fset = dict()

    for _children in children_list:
        # init new map_to_new vector
        _map_to_new = np.full(_children.shape[0], -1, dtype=int)
        map_to_new.append(_map_to_new)

        for idx, (c0, c1) in enumerate(_children):
            # convert c0, c1 to common index
            # note: lookups assume topo ordering of inputs in _children,
            # that c0 and c1 already have a common index
            if c0 >= n_common:
                c0 = _map_to_new[c0 - n_common]
            if c1 >= n_common:
                c1 = _map_to_new[c1 - n_common]

            # build frozen set for current node
            fset0 = node_to_fset.get(c0, frozenset((c0,)))
            fset1 = node_to_fset.get(c1, frozenset((c1,)))
            fset = fset0 | fset1

            if fset in fset_to_node:
                # repeated node: some previous graph has the same node
                # store its name in map_to_new
                _map_to_new[idx] = fset_to_node[fset]

            else:
                # record node's children & size
                size.append(size[c0] + size[c1])
                children.append(sorted((c0, c1)))

                # store common index in map_to_new
                _map_to_new[idx] = node_idx

                # store fset (so we can lookup later if this node is
                # another's child)
                fset_to_node[fset] = node_idx
                node_to_fset[node_idx] = fset

                # increment
                node_idx += 1

    # cast lists to arrays (size & children)
    size = np.array(size)
    children = np.array(children)

    return map_to_new, children, size


def get_node_desc_dict(reg_idx_list, children, num_leaf):
    """ builds dict of first descendant of each region given

    we say that a descendant is "first" if there is no other closer
    descendant along the path from descendant to original region

    Args:
        reg_idx_list (list): list of region idx
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_)
        num_leaf (int): number of leafs in graph

    Returns:
        reg_desc_dict (dict): keys are reg_idx given, values are lists of
            "first" descendant of the reg_idx which are in reg_idx_list
    """
    # build parent representation of graph
    parent_all = get_parent(children, num_leaf=num_leaf)

    reg_idx_list = set(reg_idx_list)
    def _get_first_ancestor(reg_idx):
        while True:
            reg_idx = parent_all[reg_idx]
            if reg_idx == -1:
                return None
            elif reg_idx in reg_idx_list:
                return reg_idx

    # build reg_desc_dict
    reg_desc_dict = {reg_idx: list() for reg_idx in reg_idx_list}
    for reg_idx in reg_idx_list:
        first_ancestor = _get_first_ancestor(reg_idx)
        if first_ancestor is not None:
            reg_desc_dict[first_ancestor].append(reg_idx)

    return {k: sorted(v) for k, v in reg_desc_dict.items()}
