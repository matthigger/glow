from collections import Counter

import numpy as np

from glow.experiment.mancova import decompose


def iter_size_ysum_yout(y, children=None):
    """ iterates through region stats, less redundant compute via graph

    Args:
        y (np.array): (b, num_img, num_vox) imaging features
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_).  If None is passed,
            then iterates through stats per individual voxel region only

    Yields:
        reg_idx (int): region index
        size (int): size, in voxels, of region
        ysum (np.array): (b, num_img) sum of features across voxels
        yout (np.array): (b, b) sum of yv @ yv.T across all voxels
    """
    b, num_img, num_vox = y.shape

    if children is None:
        iter_reg = range(num_vox)
    else:
        iter_reg = range(num_vox + children.shape[0])
        ref_count = Counter(children.flatten())

    out_dict = dict()
    for reg_idx in iter_reg:
        if reg_idx < num_vox:
            # single voxel region
            ysum = y[:, :, reg_idx]
            yout = ysum @ ysum.T
            size = 1
        else:
            # multi voxel region (sum of constituent regions)
            c0, c1 = children[int(reg_idx - num_vox), :]
            size0, ysum0, yout0 = out_dict.get(c0)
            size1, ysum1, yout1 = out_dict.get(c1)

            # clean up intermediates (if no longer needed)
            for c in (c0, c1):
                ref_count[c] -= 1
                if not ref_count[c]:
                    del out_dict[c]

            # compute stats of union
            size = size0 + size1
            ysum = ysum0 + ysum1
            yout = yout0 + yout1

        # store and yield
        out_dict[reg_idx] = size, ysum, yout
        yield reg_idx, size, ysum, yout


def iter_stat(exp, n_perm=None, **kwargs):
    """ iterates through region stats, append mancova stats

    Args:
        exp (Experiment):
        n_perm (int): number of permutations to compute (not including
            unpermuted data)

    Yields:
        reg_idx (int): region index
        e (np.array): (b, b, num_perm) e of mancova
        h (np.array): (b, b, num_perm) h of mancova
    """

    # get projection matrices
    q = decompose(x=exp.x, contrast=exp.contrast)

    b = exp.y.shape[0]
    if n_perm is None:
        h = np.empty((b, b, 1))
    else:
        # pre compute permutations
        h = np.empty((b, b, 1 + n_perm))
        num_img = exp.y.shape[1]
        img_idx_dict = {idx: np.random.default_rng(idx).permutation(num_img)
                        for idx in range(1, n_perm + 1)}
    for reg_idx, size, ysum, yout in iter_size_ysum_yout(exp.y, **kwargs):
        # compute t (constant under permutations)
        a = ysum @ q[0].T
        t = yout - a @ a.T / size

        # compute h (unpermuted)
        a = ysum @ q[1].T
        h[:, :, 0] = a @ a.T / size

        if n_perm is not None:
            # compute h per permutation
            for perm_idx in range(1, n_perm + 1):
                a = ysum[:, img_idx_dict[perm_idx]] @ q[1].T
                h[:, :, perm_idx] = a @ a.T / size

        # compute e (remainder)
        e = t[:, :, np.newaxis] - h

        yield reg_idx, e, h


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


GRAPH_EXCLUDE = -1


class SCGraph:
    """ a graph which "short-circuits" parent and child relations

    short-circuit behavior: suppose A is a child of B which is a child of C in
    the full graph.  if only A and C are in the "subgraph" (not B) then: A is
    a "child" of C, and C is a "parent" of A

    subgraph is in scare quotes as our SCGraph isn't a proper subgraph: it
    contains edges not in the original

    Attributes:
        _parent_full (np.array): (num_vox) parent of the full graph
        included (np.array): (num_vox) boolean, true if node in "subgraph"
        parent (np.array): (num_vox) parent[idx] is "parent" of node idx
            (or SUBGRAPH_EXCLUDE if idx is not in the graph)
        children (dict): children[idx] is a sorted list of the children of
            node idx (nodes without children are empty lists, nodes excluded
            are not keys of this dictionary)
    """

    @classmethod
    def from_children(cls, children, num_leaf, **kwargs):
        parent = get_parent(children, num_leaf)
        return cls(parent, **kwargs)

    def __init__(self, parent, subset=None):
        if subset is None:
            included = np.ones(len(parent), dtype=bool)
        else:
            included = np.zeros(len(parent), dtype=bool)
            included[subset] = True

        self._parent_full = parent.copy()
        self.included = included.copy()
        self._rebuild()

    def modify(self, nodes_add=tuple(), nodes_rm=tuple()):
        for node in nodes_add:
            self.included[node] = True
        for node in nodes_rm:
            self.included[node] = False
        self._rebuild()

    def _rebuild(self):
        """ rebuild children dict with short-circuited relationships """

        def _get_ss_parent(node):
            # short-circuit parent
            while True:
                node = self._parent_full[node]
                if node == GRAPH_EXCLUDE:
                    return None
                elif self.included[node]:
                    return node

        node_list = np.where(self.included)[0]
        self.children = {node: [] for node in node_list}
        self.parent = np.full_like(self._parent_full,
                                   fill_value=GRAPH_EXCLUDE)
        for kid in node_list:
            par = _get_ss_parent(kid)
            if par is not None:
                self.children[par].append(kid)
                self.parent[kid] = par

        # sort kids
        self.children = {k: sorted(v) for k, v in self.children.items()}

    def iter_desc(self, node, incl_self=False):
        assert self.included[node]

        # gets all descendants of a given region
        if incl_self:
            yield node

        for _node in self.children[node]:
            yield from self.iter_desc(_node, incl_self=True)

    def iter_ancest(self, node, incl_self=False):
        assert self.included[node]

        # gets all ancestors of given region
        if incl_self:
            yield node

        while self.parent[node] != GRAPH_EXCLUDE:
            node = self.parent[node]
            yield node
