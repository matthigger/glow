from collections import Counter

import numpy as np

from glow.analysis.mancova import decompose


def iter_size_ysum_yout(y, children=None):
    """iterate region statistics, re-using partial sums via the graph.

    Args:
        y (np.array): (b, num_img, num_vox) imaging features
        children (np.array): (num_leaf - 1, 2) child index pairs. if None,
            iterates individual voxels only.

    Yields:
        reg_idx (int): region index
        size (int): number of voxels in region
        ysum (np.array): (b, num_img) sum across voxels
        yout (np.array): (b, b) sum of yv @ yv.T across voxels
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


def iter_stat(exp, **kwargs):
    """iterate region-level MANCOVA statistics (E, H).

    Computes one (E, H) pair per region for the given experiment.  To
    obtain a permutation null distribution, callers should loop
    externally over Freedman-Lane permutations of the experiment::

        for k in range(n_perm + 1):
            _exp = exp.permute(k) if k else exp
            for reg_idx, size, e, h in iter_stat(_exp, children=children):
                ...

    Args:
        exp (Experiment): experiment data
        **kwargs: forwarded to ``iter_size_ysum_yout`` (notably
            ``children`` for hierarchical regions)

    Yields:
        reg_idx (int): region index
        size (int): number of voxels in the region
        e (np.array): (b, b) error matrix
        h (np.array): (b, b) hypothesis matrix
    """
    q = decompose(x=exp.x, contrast=exp.contrast)
    for reg_idx, size, ysum, yout in iter_size_ysum_yout(exp.y, **kwargs):
        a0 = ysum @ q[0].T
        t = yout - a0 @ a0.T / size

        a1 = ysum @ q[1].T
        h = a1 @ a1.T / size

        e = t - h

        yield reg_idx, size, e, h


def compute_llr_batched(exp, children, q0, q1):
    """Vectorised LLR per region for a single (already-permuted) experiment.

    Computes the same per-region LLR statistic as the per-region loop::

        for reg_idx, size, e, h in iter_stat(exp, children=children):
            llr[reg_idx] = get_llr(e, h, n=size)

    but in a single batched pass over numpy: ysum/yout are built
    bottom-up (sequentially across tree layers, vectorised across all
    regions at each step), then E and H are formed via einsum and the
    LLR is computed via batched ``np.linalg.slogdet``.  At mandrill
    scale (~450k merged regions) this trades ~5 s of Python per-region
    overhead for ~0.3 s of pure numpy.

    Args:
        exp (Experiment): experiment data (already FL-permuted).
        children (np.array): (num_internal, 2) child index pairs in
            topological (bottom-up) order.
        q0 (np.array): nuisance subspace (from ``decompose``).
        q1 (np.array): interest subspace (from ``decompose``).

    Returns:
        llr (np.array): (num_reg,) LLR per region, NaN where E or
            E + H were not positive-definite.
        size (np.array): (num_reg,) voxel count per region.
    """
    y = exp.y
    b, num_img, num_vox = y.shape
    num_internal = children.shape[0]
    num_reg = num_vox + num_internal

    # use float32 only when both source arrays are float32; else float64
    dtype = y.dtype if y.dtype == np.float32 else np.float64

    # leaf ysum / yout: vectorised init
    ysum = np.empty((num_reg, b, num_img), dtype=dtype)
    ysum[:num_vox] = y.transpose(2, 0, 1)
    yout = np.empty((num_reg, b, b), dtype=dtype)
    # einsum 'vbn,vcn->vbc' = per-voxel outer product summed over images
    yout[:num_vox] = np.einsum('vbn,vcn->vbc',
                               ysum[:num_vox], ysum[:num_vox],
                               optimize=True)
    size = np.empty(num_reg, dtype=int)
    size[:num_vox] = 1

    # bottom-up build over internal nodes.  children is in topological
    # order so a single pass suffices.
    for i in range(num_internal):
        c0, c1 = children[i, 0], children[i, 1]
        idx = num_vox + i
        ysum[idx] = ysum[c0] + ysum[c1]
        yout[idx] = yout[c0] + yout[c1]
        size[idx] = size[c0] + size[c1]

    # E, H per region via einsum (matches iter_stat math exactly)
    sz = size.astype(dtype)[:, None, None]
    a0 = np.einsum('rbn,an->rba', ysum, q0, optimize=True)
    t = yout - np.einsum('rba,rca->rbc', a0, a0, optimize=True) / sz

    a1 = np.einsum('rbn,vn->rbv', ysum, q1, optimize=True)
    h = np.einsum('rbv,rcv->rbc', a1, a1, optimize=True) / sz

    e = t - h

    # LLR via batched slogdet:
    #   get_llr(e, h, n=size) = (size/2) * (ln|E+H| - ln|E|)
    # NaN where either determinant is non-positive (matches the
    # ``np.isnan`` short-circuit in get_llr / loglik_from_cov).
    sign_t, logdet_t = np.linalg.slogdet(e + h)
    sign_e, logdet_e = np.linalg.slogdet(e)
    valid = (sign_t > 0) & (sign_e > 0)
    llr = np.full(num_reg, np.nan)
    llr[valid] = (size[valid] / 2.0) * (logdet_t[valid] - logdet_e[valid])

    return llr, size


def node_sum(x, children):
    """sum leaf values up through the graph.

    Args:
        x (np.array): one value per leaf (single-voxel region)
        children (np.array): (num_leaf - 1, 2) child index pairs

    Returns:
        summed (np.array): values for all nodes (leaves + internal)
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


def get_dice_sens_spec(mask, mask_idx, children):
    """compute Dice, sensitivity (recall/TPR), and specificity (TNR) per region.

    Args:
        mask (np.array): target mask (boolean, same shape as mask_idx)
        mask_idx (np.array): voxel index array (-1 outside analysis)
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_)

    Returns:
        dice (np.array): Dice score per region
        sens (np.array): TP / (TP + FN) per region
        spec (np.array): TN / (TN + FP) per region
    """
    # compute misses & hits per region
    # true positive: target voxels in estimated region
    # false positive: in estimated region but not in target mask
    fp, tp = get_miss_hits(mask, mask_idx, children)

    # false negative: targets outside of estimated region
    fn = mask.sum() - tp

    # true negative: analysis voxels not in target and not in region
    total = float((mask_idx >= 0).sum())
    tn = total - tp - fp - fn

    # metrics with safe division (0 where undefined)
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = 2 * tp / (2 * tp + fp + fn)
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)

    dice = np.nan_to_num(dice, nan=0)
    sens = np.nan_to_num(sens, nan=0)
    spec = np.nan_to_num(spec, nan=1)

    return dice, sens, spec


def get_miss_hits(mask, mask_idx, children):
    """count target (hit) and non-target (miss) voxels per node.

    Args:
        mask (np.array): target mask (boolean, same shape as mask_idx)
        mask_idx (np.array): voxel index array (-1 outside analysis)
        children (np.array): (num_leaf - 1, 2) child index pairs

    Returns:
        miss (np.array): non-target voxels per node
        hit (np.array): target voxels per node
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
    """topological sort, leaves to root.

    Supports forests: when ``node_start`` is None, iterates from every
    root (nodes with no parent).

    Args:
        children (np.array): (num_internal, 2) child index pairs
        num_leaf (int): number of leaves
        node_start (int): subtree root (defaults to all roots)
        only_leaf (bool): if True, yield only leaves

    Yields:
        node_idx (int): node index
    """
    if children is None:
        yield from range(num_leaf)
        return

    if node_start is None:
        parent = get_parent(children, num_leaf)
        roots = np.where(parent == -1)[0]
        for root in roots:
            yield from iter_topo(children=children, num_leaf=num_leaf,
                                 node_start=root, only_leaf=only_leaf)
        return

    if node_start >= num_leaf:
        for child in children[int(node_start - num_leaf), :]:
            yield from iter_topo(children=children, num_leaf=num_leaf,
                                 node_start=child, only_leaf=only_leaf)

    if not only_leaf or node_start < num_leaf:
        yield node_start


def get_parent(children, num_leaf):
    """build parent lookup array for a binary tree.

    Args:
        children (np.array): (num_leaf - 1, 2) child index pairs
        num_leaf (int): number of leaves

    Returns:
        parent (np.array): parent[idx] gives the parent of node idx
    """
    num_nodes = num_leaf + children.shape[0]
    parent = np.full(num_nodes, -1, dtype=int)
    for i, (c0, c1) in enumerate(children):
        parent[c0] = parent[c1] = num_leaf + i

    return parent


class RegIntersectError(Exception):
    pass


def get_label_map(reg_idx_list, mask_idx, children, check_disjoint=False):
    """build a mask_idx array from a list of region indices.

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
    """merge many binary trees into a single graph with shared indexing.

    Regions are identified by a 128-bit XOR hash of their leaf labels,
    so identical leaf sets produce identical nodes regardless of how
    different trees decomposed them.  When two trees both contain the
    same region but built it from different child pairs, the merged
    graph stores one (c0, c1) pair (the first encountered) — that's
    enough for ``iter_size_ysum_yout`` to compute (size, ysum, yout)
    correctly, since those quantities are functions of the leaf set
    only and are path-independent.

    Args:
        n_common (int): number of shared leaf nodes
        children_list (list): each element is a (num_node, 2) child
            index array (assumes topological ordering within each tree)

    Returns:
        map_to_new (list): per-tree arrays mapping each tree's
            non-leaf node indices (offset by n_common) to the merged
            index space.  Leaves keep their original indices in [0, n_common).
        children (np.array): (num_merged_nodes, 2) merged child pairs
        size (np.array): voxel count per merged node (leaves first,
            then internal nodes in encounter order)
    """
    rng = np.random.default_rng(seed=0)
    leaf_hash = {}
    for i in range(n_common):
        hi = int(rng.integers(0, 2**63)) << 64
        lo = int(rng.integers(0, 2**63))
        leaf_hash[i] = hi | lo

    node_idx = n_common
    map_to_new = list()
    children = list()
    size = [1] * n_common
    hash_to_node = dict()
    node_to_hash = leaf_hash.copy()

    for _children in children_list:
        _map_to_new = np.full(_children.shape[0], -1, dtype=int)
        map_to_new.append(_map_to_new)

        for idx, (c0, c1) in enumerate(_children):
            if c0 >= n_common:
                c0 = _map_to_new[c0 - n_common]
            if c1 >= n_common:
                c1 = _map_to_new[c1 - n_common]

            h = node_to_hash[c0] ^ node_to_hash[c1]

            if h in hash_to_node:
                _map_to_new[idx] = hash_to_node[h]
            else:
                size.append(size[c0] + size[c1])
                children.append(sorted((c0, c1)))
                _map_to_new[idx] = node_idx
                hash_to_node[h] = node_idx
                node_to_hash[node_idx] = h
                node_idx += 1

    size = np.array(size)
    children = np.array(children)
    return map_to_new, children, size


GRAPH_EXCLUDE = -1


def dp_antichain(nodes, children_map, gain, lam=0.0):
    """Bottom-up DP finding the antichain that maximises total gain.

    Maximises ``sum_{i in E} [gain(i) - lam(i)]`` over antichains E
    of the tree defined by *children_map*.

    Args:
        nodes: node indices in ascending (bottom-up) order
        children_map (dict): node -> list of child nodes in *nodes*.
            Nodes absent from the dict (or with empty list) are leaves.
        gain: array-like or dict mapping node -> gain value
        lam (float or dict): penalty per region — scalar or per-node dict.

    Returns:
        selected (list[int]): sorted indices of antichain regions
        info (dict): diagnostic keys ``best``, ``chose``
    """
    _lam_is_dict = isinstance(lam, dict)
    best = {}
    chose = {}

    for node in nodes:
        kids = children_map.get(node, [])
        lam_node = lam[node] if _lam_is_dict else lam
        g_net = gain[node] - lam_node

        if not kids:
            best[node] = max(g_net, 0.0)
            chose[node] = g_net > 0
        else:
            split_val = sum(best[k] for k in kids)
            best[node] = max(g_net, split_val)
            chose[node] = g_net >= split_val

    all_children = set()
    for kids in children_map.values():
        all_children.update(kids)
    roots = sorted(n for n in nodes if n not in all_children)

    selected = []

    def _bt(node):
        if chose[node]:
            selected.append(node)
        else:
            for kid in children_map.get(node, []):
                _bt(kid)

    for root in roots:
        _bt(root)

    return sorted(selected), dict(best=best, chose=chose)


class SCGraph:
    """graph with short-circuited parent/child relations.

    if A -> B -> C in the full tree but B is excluded, SCGraph treats
    A as a direct child of C.

    Attributes:
        included (np.array): boolean, True if node is in the subgraph
        parent (np.array): short-circuit parent (-1 if excluded/root)
        children (dict): node -> sorted list of short-circuit children
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
        """add or remove nodes and rebuild short-circuit relationships."""
        for node in nodes_add:
            self.included[node] = True
        for node in nodes_rm:
            self.included[node] = False
        self._rebuild()

    def _rebuild(self):
        """rebuild children dict with short-circuited relationships."""

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
        """yield all descendants of node in the short-circuit subgraph."""
        assert self.included[node]
        if incl_self:
            yield node

        for _node in self.children[node]:
            yield from self.iter_desc(_node, incl_self=True)

    def iter_ancest(self, node, incl_self=False):
        """yield all ancestors of node in the short-circuit subgraph."""
        assert self.included[node]
        if incl_self:
            yield node

        while self.parent[node] != GRAPH_EXCLUDE:
            node = self.parent[node]
            yield node
