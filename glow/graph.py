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


def iter_mancova(exp, **kwargs):
    """iterate region-level MANCOVA statistics (E, H).

    Computes one (E, H) pair per region for the given experiment.  To
    obtain a permutation null distribution, callers should loop
    externally over Freedman-Lane permutations of the experiment::

        for k in range(n_perm + 1):
            _exp = exp.permute(k) if k else exp
            for reg_idx, size, e, h in iter_mancova(_exp, children=children):
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


def compute_tree_layers(children, num_vox):
    """compute per-node depth from leaves for a Ward tree.

    Leaves are 0; internal nodes are ``1 + max(child depths)``.  Depends
    only on ``children``, so callers running many ``compute_llr_batched``
    passes against the same tree (e.g. the inner FL loop in
    ``AnalysisGLOW``) should compute this once and pass via the
    ``layer`` kwarg to avoid the per-pass Python loop.

    Args:
        children (np.array): (num_internal, 2) child index pairs in
            topological (bottom-up) order.
        num_vox (int): number of leaf voxels.

    Returns:
        layer (np.array): (num_reg,) int32 depth per region.
    """
    num_internal = children.shape[0]
    layer = np.zeros(num_vox + num_internal, dtype=np.int32)
    c0_all = children[:, 0]
    c1_all = children[:, 1]
    for i in range(num_internal):
        layer[num_vox + i] = 1 + max(layer[c0_all[i]], layer[c1_all[i]])
    return layer


def compute_phase1(y, children, layer=None):
    """Bottom-up build of ``(ysum, yout, size)`` for every region.

    Phase 1 of ``compute_llr_batched`` extracted as a public helper so
    callers running many Phase 2 passes against the same data can hoist
    Phase 1 out of their inner loop (used by AnalysisGLOW's
    intercept-only fast path).  Depends only on ``y`` and ``children``.

    Args:
        y (np.array): (b, num_img, num_vox) imaging features.
        children (np.array): (num_internal, 2) child index pairs in
            topological (bottom-up) order.
        layer (np.array | None): (num_reg,) per-node depths from
            ``compute_tree_layers``.  Computed internally if None.

    Returns:
        ysum (np.array): (num_reg, b, num_img) — per-region image sums.
        yout (np.array): (num_reg, b, b) — per-region ysum of y[v] @ y[v].T.
        size (np.array): (num_reg,) — voxel count per region.
    """
    b, num_img, num_vox = y.shape
    num_internal = children.shape[0]
    num_reg = num_vox + num_internal

    # use float32 only when y is float32; else float64
    dtype = y.dtype if y.dtype == np.float32 else np.float64

    # Process by layer so each layer's nodes are a single batched numpy
    # add instead of a Python-loop iteration per node.  ~2.3x faster
    # than the per-node loop on mandrill (5.4 ms -> 2.3 ms).
    c0_all = children[:, 0]
    c1_all = children[:, 1]

    if layer is None:
        layer = compute_tree_layers(children, num_vox)

    ysum = np.empty((num_reg, b, num_img), dtype=dtype)
    ysum[:num_vox] = y.transpose(2, 0, 1)
    yout = np.empty((num_reg, b, b), dtype=dtype)
    yout[:num_vox] = np.einsum('vbn,vcn->vbc',
                               ysum[:num_vox], ysum[:num_vox],
                               optimize=True)
    size = np.empty(num_reg, dtype=int)
    size[:num_vox] = 1

    internal_layer = layer[num_vox:]
    max_L = int(internal_layer.max()) if num_internal else 0
    for L in range(1, max_L + 1):
        nodes = np.where(internal_layer == L)[0]
        if len(nodes) == 0:
            continue
        c0_L = c0_all[nodes]
        c1_L = c1_all[nodes]
        tgt = num_vox + nodes
        ysum[tgt] = ysum[c0_L] + ysum[c1_L]
        yout[tgt] = yout[c0_L] + yout[c1_L]
        size[tgt] = size[c0_L] + size[c1_L]

    return ysum, yout, size


def compute_llr_batched(exp, children, q0, q1, min_size=1, layer=None):
    """Vectorised LLR per region for a single (already-permuted) experiment.

    Computes the same per-region LLR statistic as the per-region loop::

        for reg_idx, size, e, h in iter_mancova(exp, children=children):
            llr[reg_idx] = get_llr(e, h, n=size)

    but in a single batched pass over numpy.  Two phases:

    1. Bottom-up build of ``(size, ysum, yout)`` for ALL regions.
       Cannot be skipped for small regions because every internal node
       needs its children's ``ysum``/``yout``.  This is the cheap part
       (a few numpy adds per region).

    2. Per-region E/H/LLR via einsum + batched ``np.linalg.slogdet``.
       This is where the bulk of the FLOPs live.  When ``min_size > 1``
       we skip Phase 2 for regions with ``size < min_size`` and leave
       their LLR as NaN.  At ``min_size=4`` on typical neuroimaging
       trees, ~70% of regions drop out, cutting Phase 2's cost roughly
       proportionally.

    Args:
        exp (Experiment): experiment data (already FL-permuted).
        children (np.array): (num_internal, 2) child index pairs in
            topological (bottom-up) order.
        q0 (np.array): nuisance subspace (from ``decompose``).
        q1 (np.array): interest subspace (from ``decompose``).
        min_size (int): regions with size < min_size get NaN LLR (and
            their E/H matrices are never computed).  Default 1 keeps
            every region.
        layer (np.array | None): (num_reg,) per-node depths from
            ``compute_tree_layers``.  Depends only on ``children``;
            pass precomputed to skip the per-call Python loop in hot
            inner FL loops (~34% of this function's runtime on a 5k
            vox tree).  Computed internally if None.

    Returns:
        llr (np.array): (num_reg,) LLR per region.  NaN where size <
            min_size, or where E or E + H were not positive-definite.
        size (np.array): (num_reg,) voxel count per region.
    """
    y = exp.y
    dtype = y.dtype if y.dtype == np.float32 else np.float64
    num_reg = y.shape[2] + children.shape[0]

    # --- Phase 1: bottom-up build of ysum / yout / size for ALL regions ---
    ysum, yout, size = compute_phase1(y, children, layer=layer)

    llr = np.full(num_reg, np.nan)

    # --- Phase 2: E, H, LLR — only for regions with size >= min_size ---
    active = size >= min_size
    if not active.any():
        return llr, size

    sz_a = size[active].astype(dtype)[:, None, None]
    ysum_a = ysum[active]
    yout_a = yout[active]

    a0 = np.einsum('rbn,an->rba', ysum_a, q0, optimize=True)
    t = yout_a - np.einsum('rba,rca->rbc', a0, a0, optimize=True) / sz_a

    a1 = np.einsum('rbn,vn->rbv', ysum_a, q1, optimize=True)
    h = np.einsum('rbv,rcv->rbc', a1, a1, optimize=True) / sz_a

    e = t - h

    # LLR = (size/2) * (ln|E+H| - ln|E|).  NaN where either determinant
    # is non-positive (matches the sign-check short-circuit in get_llr).
    sign_t, logdet_t = np.linalg.slogdet(e + h)
    sign_e, logdet_e = np.linalg.slogdet(e)
    valid_a = (sign_t > 0) & (sign_e > 0)

    sz_a_1d = size[active]
    llr_a = np.where(valid_a,
                     (sz_a_1d / 2.0) * (logdet_t - logdet_e),
                     np.nan)
    llr[active] = llr_a

    return llr, size


def compute_llr_inner_fast(t, ysum, size, q1_T_perm, min_size=1):
    """Per-region LLR using precomputed (t, ysum) and a perm-applied q1.T.

    Phase-2-only variant of ``compute_llr_batched`` for callers driving
    the FL inner loop under intercept-only nuisance (where ``t`` and
    ``ysum`` are FL-invariant — see
    ``glow.analysis.mancova.is_intercept_only_nuisance``).  Skipping
    Phase 1 across all 200 inner perms saves ~50–70% of inner-loop wall
    time at paper-config scale.

    Correctness precondition: the caller must have established that Q0
    commutes with permutations, otherwise ``t`` and ``ysum`` are NOT
    deterministically invariant under FL and the returned LLR will not
    match ``compute_llr_batched(_exp_inner, ...)``.

    Args:
        t (np.array): (num_reg, b, b) — precomputed
            ``yout - a0 @ a0.T / size`` on the unpermuted data.
        ysum (np.array): (num_reg, b, num_img) — precomputed Phase 1 ysum
            on the unpermuted data.
        size (np.array): (num_reg,) — voxel count per region.
        q1_T_perm (np.array): (num_img, n_intrst) — ``q1.T`` with rows
            permuted by the FL permutation for this inner perm.
            Equivalent to ``freed_lane @ q1.T`` under intercept-only.
        min_size (int): regions with size < min_size return NaN LLR.

    Returns:
        llr (np.array): (num_reg,) LLR per region.  NaN where size <
            min_size, or where E or E + H were not positive-definite.
        size (np.array): same array passed in.
    """
    num_reg = ysum.shape[0]
    dtype = ysum.dtype if ysum.dtype == np.float32 else np.float64

    llr = np.full(num_reg, np.nan)
    active = size >= min_size
    if not active.any():
        return llr, size

    sz_a = size[active].astype(dtype)[:, None, None]
    ysum_a = ysum[active]
    t_a = t[active]

    a1 = np.einsum('rbn,nv->rbv', ysum_a, q1_T_perm, optimize=True)
    h = np.einsum('rbv,rcv->rbc', a1, a1, optimize=True) / sz_a
    e = t_a - h

    sign_t, logdet_t = np.linalg.slogdet(e + h)
    sign_e, logdet_e = np.linalg.slogdet(e)
    valid_a = (sign_t > 0) & (sign_e > 0)

    sz_a_1d = size[active]
    llr_a = np.where(valid_a,
                     (sz_a_1d / 2.0) * (logdet_t - logdet_e),
                     np.nan)
    llr[active] = llr_a
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



def iter_postorder(*, children=None, num_leaf, node_start=None, only_leaf=False):
    """DFS post-order traversal; yields nodes in topological order (leaves to root).

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
            yield from iter_postorder(children=children, num_leaf=num_leaf,
                                      node_start=root, only_leaf=only_leaf)
        return

    if node_start >= num_leaf:
        for child in children[int(node_start - num_leaf), :]:
            yield from iter_postorder(children=children, num_leaf=num_leaf,
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
        for vox in iter_postorder(children=children,
                                  num_leaf=num_vox,
                                  node_start=reg_idx,
                                  only_leaf=True):
            target_voxels = (mask_idx == vox)
            if check_disjoint and np.any(label_map[target_voxels] != -1):
                reg_idx_list = np.unique(label_map[target_voxels])
                raise RegIntersectError(f'{reg_idx} intersects {reg_idx_list}')

            label_map[target_voxels] = reg_idx

    return label_map


GRAPH_EXCLUDE = -1


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
