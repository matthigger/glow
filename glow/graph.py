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


def compute_llr_batched(exp, children, q0, q1, min_size=1):
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

    Returns:
        llr (np.array): (num_reg,) LLR per region.  NaN where size <
            min_size, or where E or E + H were not positive-definite.
        size (np.array): (num_reg,) voxel count per region.
    """
    y = exp.y
    b, num_img, num_vox = y.shape
    dtype = y.dtype if y.dtype == np.float32 else np.float64
    num_reg = num_vox + children.shape[0]

    # --- Phase 1: bottom-up build of ysum / yout / size for ALL regions ---
    ysum = np.empty((num_reg, b, num_img), dtype=dtype)
    yout = np.empty((num_reg, b, b), dtype=dtype)
    size = np.empty(num_reg, dtype=int)
    for reg_idx, sz, ys, yo in iter_size_ysum_yout(y, children=children):
        ysum[reg_idx] = ys
        yout[reg_idx] = yo
        size[reg_idx] = sz

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


def _slogdet_batched(M):
    """Batched ``log|det(M)|`` for the small symmetric matrices in LLR.

    Closed-form for ``b in {1, 2}`` -- significantly cheaper than
    ``np.linalg.slogdet``'s LU dispatch, which dominates the per-perm
    cost at ``b=2`` in glow's inner loop.  Falls back to numpy for
    larger ``b``.  Returns ``(sign, log|det|)`` matching numpy's API.
    """
    b = M.shape[-1]
    if b == 1:
        d = M[..., 0, 0]
        with np.errstate(divide='ignore'):
            return np.sign(d), np.log(np.abs(d))
    if b == 2:
        det = (M[..., 0, 0] * M[..., 1, 1]
               - M[..., 0, 1] * M[..., 1, 0])
        with np.errstate(divide='ignore'):
            return np.sign(det), np.log(np.abs(det))
    return np.linalg.slogdet(M)


def build_dfs_preorder(children, num_vox):
    """DFS pre-order leaf permutation and per-region leaf ranges.

    Given a (forest of) binary tree(s) on ``num_vox`` leaves in
    topological (bottom-up) order, this returns a permutation
    ``leaf_ord`` of the original voxel indices such that every region's
    leaves occupy a contiguous range ``[region_l[r], region_h[r])`` on
    the permuted leaf axis.  That is the prerequisite for cumsum-and-
    diff region aggregation (see ``_reg_sum_cumsum`` and
    ``compute_optimize/perm_llr_compute.tex``).

    Roots are laid out end-to-end -- the first root takes positions
    ``[0, size_root_0)``, the next takes ``[size_root_0, ...)``, etc.

    Args:
        children (np.array): (num_internal, 2) child index pairs in
            topological order -- each row references indices ``< num_vox
            + row_idx``.
        num_vox (int): number of leaves.

    Returns:
        leaf_ord (np.array): (num_vox,) original voxel index visited at
            each DFS position -- i.e. ``y_dfs[..., k] = y[..., leaf_ord[k]]``.
        region_l (np.array): (num_reg,) leaf range start per region.
        region_h (np.array): (num_reg,) leaf range end per region.
    """
    num_internal = int(children.shape[0])
    num_reg = num_vox + num_internal

    # bottom-up region sizes
    size = node_sum(np.ones(num_vox, dtype=np.int64), children=children)

    # find roots (no parent)
    parent = get_parent(children, num_vox)
    roots = np.where(parent == -1)[0]

    # top-down range assignment.  Lay roots end-to-end, then propagate
    # to each internal node's two children: left child gets the front
    # slice, right child gets the back slice.  Processing internal
    # nodes in reverse topological order guarantees the parent's range
    # is filled before its children's.
    region_l = np.empty(num_reg, dtype=np.int64)
    region_h = np.empty(num_reg, dtype=np.int64)
    offset = 0
    for root in roots:
        region_l[root] = offset
        region_h[root] = offset + size[root]
        offset += int(size[root])

    for i in range(num_internal - 1, -1, -1):
        node = num_vox + i
        c0, c1 = children[i]
        l = region_l[node]
        sz0 = size[c0]
        region_l[c0] = l
        region_h[c0] = l + sz0
        region_l[c1] = l + sz0
        region_h[c1] = region_h[node]

    # The leaf at original index v lives at DFS position region_l[v];
    # inverting gives leaf_ord[position] = v.
    leaf_ord = np.empty(num_vox, dtype=np.int64)
    leaf_ord[region_l[:num_vox]] = np.arange(num_vox)
    return leaf_ord, region_l, region_h


def _reg_sum_cumsum(x_dfs, axis, region_l, region_h):
    """Per-region sums via cumsum-and-diff along ``axis``.

    ``x_dfs`` is laid out in DFS pre-order along ``axis`` (length V),
    so every region's voxels form a contiguous range.  We prepend a
    zero slice along ``axis`` and cumsum into the rest, then index at
    ``region_h`` and ``region_l`` to read out half-open range sums.
    The ``l = 0`` case is handled correctly because we left a zero in
    the prepended slot.

    Output length along ``axis`` is ``len(region_l) == num_reg``.
    """
    out_shape = list(x_dfs.shape)
    out_shape[axis] = x_dfs.shape[axis] + 1
    c = np.empty(out_shape, dtype=x_dfs.dtype)
    head = [slice(None)] * x_dfs.ndim
    head[axis] = 0
    c[tuple(head)] = 0
    tail = [slice(None)] * x_dfs.ndim
    tail[axis] = slice(1, None)
    np.cumsum(x_dfs, axis=axis, out=c[tuple(tail)])
    return (np.take(c, region_h, axis=axis)
            - np.take(c, region_l, axis=axis))


def compute_llr_perm_full(*, y, q0, q1, perms, leaf_ord, region_l, region_h,
                          min_size=1, perm_chunk=8):
    """Per-region LLR for many Freedman-Lane permutations in one sweep.

    Algorithm (see ``compute_optimize/perm_llr_compute.tex``):

      1.  *Phase 1 (once)* -- per-voxel sufficient statistics:

              T_v   = Y_v Y_v^T               (b, b)
              S0_v  = Q0 Y_v                  (a0, b)
              r_v   = (I - Q0 Q0^T) Y_v       (N, b)   FL residuals

          Aggregated over each region via cumsum-and-diff on the DFS
          pre-order axis -- no tree walk in the inner loop.

      2.  *Phase 2 (per perm)* -- one ``(a, N) @ (N, V*b)`` GEMM
          assembles ``gamma_v = Q^T P r_v`` for every voxel and feature.
          From ``gamma`` we read the FL-shifted ``rho`` (Q0 part) and
          ``beta`` (Q1 part) needed to assemble ``E*``, ``H*`` and
          finally the per-region LLR.

    Both intercept-only and general-Q0 nuisance ride this same code
    path: under intercept-only Q0 commutes with P, so ``rho`` is
    numerically zero (and the ``X_v`` cross-correction below vanishes
    up to roundoff).  The waste is ``O(V a0 b^2)`` FLOPs, dwarfed by
    the dominant ``O(V N a b)`` gamma GEMM.

    Args:
        y (np.array): (b, num_img, num_vox) imaging features.  The
            caller's original (unpermuted) data -- permutations are
            applied to ``q0/q1`` instead.
        q0 (np.array): (a0, num_img) nuisance subspace from
            ``decompose``.
        q1 (np.array): (a1, num_img) interest subspace.
        perms (np.array): (n_perm, num_img) int -- ``perms[p, k]``
            gives the original-image index that the FL-permuted data
            puts at position ``k``.  Matches glow's
            ``get_freed_lane`` convention so callers can build this as
            ``np.argsort(rng.permutation(num_img))`` per draw.
        leaf_ord, region_l, region_h: from ``build_dfs_preorder``.
        min_size (int): regions with ``size < min_size`` return NaN
            draws.
        perm_chunk (int): number of perms to batch through one gamma
            GEMM.  Trade-off: larger chunks reduce Python / BLAS call
            overhead but multiply the (Pc, V, ...) temporary memory.
            Default 8 is the sweet spot empirically at V in [25k,
            55k] for both intercept-only and general-Q0.

    Returns:
        draws (np.array): (n_perm, num_reg) LLR per region per perm.
            NaN for ``size < min_size`` or non-positive-definite
            ``E``/``E + H``.
    """
    b, num_img, num_vox = y.shape
    n_perm = int(perms.shape[0])
    num_reg = int(region_l.shape[0])
    a0 = int(q0.shape[0])
    a1 = int(q1.shape[0])
    a = a0 + a1

    # match glow's existing dtype policy: float32 stays float32, else
    # float64.  Cumsums over ~10^6 entries are stable enough in fp32
    # for our purposes (see perm_llr_compute.tex, "Numerical care").
    dtype = y.dtype if y.dtype == np.float32 else np.float64

    # -------------------- Phase 1: per-voxel state --------------------
    # Reorder y so its voxel axis is DFS pre-order; downstream cumsums
    # along that axis then deliver region sums via two index reads.
    y_dfs = np.ascontiguousarray(y[:, :, leaf_ord]).astype(dtype, copy=False)

    # S0_v[v, a, j] = Q0 Y_v in math = sum_n q0[a, n] * y_dfs[j, n, v]
    S0_v = np.einsum('an,jnv->vaj', q0, y_dfs, optimize=True)

    # T_v[v, i, j] = Y_v^T Y_v in math = sum_n y_dfs[i, n, v] * y_dfs[j, n, v]
    T_v = np.einsum('inv,jnv->vij', y_dfs, y_dfs, optimize=True)

    # Build the FL residuals r_v = (I - Q0 Q0^T) Y_v directly in the
    # (N, V, b) layout the dominant GEMM needs, then reshape to
    # (N, V*b) for free.  Holding r_v in (V, N, b) instead would force
    # a non-contiguous transpose + copy at reshape time, doubling peak
    # memory at the 600k-voxel scale.
    r_v_nvb = np.ascontiguousarray(y_dfs.transpose(1, 2, 0))           # (N, V, b)
    del y_dfs
    r_v_nvb -= np.einsum('an,vaj->nvj', q0, S0_v, optimize=True)       # in-place
    r_v_flat = r_v_nvb.reshape(num_img, num_vox * b)                   # (N, V*b)

    # one-time region statistics (perm-invariant)
    S0_r = _reg_sum_cumsum(S0_v, axis=0,
                           region_l=region_l, region_h=region_h)   # (R, a0, b)
    T_r = _reg_sum_cumsum(T_v, axis=0,
                          region_l=region_l, region_h=region_h)    # (R, b, b)

    sz_1d = (region_h - region_l).astype(dtype)
    inv_sz = np.empty_like(sz_1d)
    np.divide(1.0, sz_1d, out=inv_sz, where=sz_1d > 0)
    inv_sz_3d = inv_sz[:, None, None]
    active = (region_h - region_l) >= min_size

    Q = np.vstack([q0, q1]).astype(dtype, copy=False)              # (a, N)
    draws = np.full((n_perm, num_reg), np.nan, dtype=np.float64)

    # -------------------- Phase 2: per-perm hot loop -----------------
    for s in range(0, n_perm, perm_chunk):
        chunk = perms[s:s + perm_chunk]
        Pc = int(chunk.shape[0])

        # Match glow's FL convention: a permutation acts on the image
        # axis as ``y_perm[..., k] = y[..., perm[k]]``.  Then
        # ``(Q^T P r_v)[a, j] = sum_n Q[n, a] * r_v[perm[n], j]``,
        # which after substitution m = perm[n] reads off rows of Q at
        # ``perm^{-1}``.  ``argsort`` inverts the perm.
        pi_inv = np.argsort(chunk, axis=1)
        tQ = Q[:, pi_inv].transpose(1, 0, 2)                       # (Pc, a, N)

        # Dominant compute: (Pc*a, N) @ (N, V*b) -> (Pc*a, V*b).
        # Reshape lands gamma in (Pc, V, a, b).
        gamma = (tQ.reshape(Pc * a, num_img) @ r_v_flat
                 ).reshape(Pc, a, num_vox, b).transpose(0, 2, 1, 3)

        rho = gamma[..., :a0, :]                                   # (Pc, V, a0, b)
        beta = gamma[..., a0:, :]                                  # (Pc, V, a1, b)

        # T_v's permutation-dependent correction: X_v = rho^T S0_v
        # in math; in our (V, a, b) indexing that's an einsum over a.
        X_v = np.einsum('pvai,vaj->pvij', rho, S0_v, optimize=True)  # (Pc, V, b, b)

        # Region aggregation via cumsum-and-diff on the V axis.
        rho_r = _reg_sum_cumsum(rho, axis=1,
                                region_l=region_l, region_h=region_h)  # (Pc, R, a0, b)
        beta_r = _reg_sum_cumsum(beta, axis=1,
                                 region_l=region_l, region_h=region_h)  # (Pc, R, a1, b)
        X_r = _reg_sum_cumsum(X_v, axis=1,
                              region_l=region_l, region_h=region_h)    # (Pc, R, b, b)

        # FL-shifted sufficient statistics per region.
        S0_star = rho_r + S0_r                                     # (Pc, R, a0, b)
        S1_star = beta_r                                           # (Pc, R, a1, b)
        T_star = T_r + X_r + X_r.swapaxes(-1, -2)                  # (Pc, R, b, b)

        # E + H = T* - (1/|r|) S0*^T S0*;  H = (1/|r|) S1*^T S1*.
        EH = T_star - np.einsum('prai,praj->prij',
                                S0_star, S0_star,
                                optimize=True) * inv_sz_3d
        E = EH - np.einsum('prai,praj->prij',
                           S1_star, S1_star,
                           optimize=True) * inv_sz_3d

        sign_EH, ld_EH = _slogdet_batched(EH)
        sign_E, ld_E = _slogdet_batched(E)
        valid = (sign_EH > 0) & (sign_E > 0) & active[None]
        llr_chunk = np.where(valid,
                             0.5 * sz_1d[None] * (ld_EH - ld_E),
                             np.nan)
        draws[s:s + Pc] = llr_chunk

    return draws


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
    fp, tp = get_fp_tp(mask, mask_idx, children)

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


def get_fp_tp(mask, mask_idx, children):
    """count false-positive and true-positive voxels per node.

    Treats each region as a predictor of the target mask.

    Args:
        mask (np.array): target mask (boolean, same shape as mask_idx)
        mask_idx (np.array): voxel index array (-1 outside analysis)
        children (np.array): (num_leaf - 1, 2) child index pairs

    Returns:
        fp (np.array): non-target voxels per node (in region, not in target)
        tp (np.array): target voxels per node (in region and in target)
    """
    num_vox = (mask_idx >= 0).sum()
    tp = np.zeros(num_vox)
    tp[mask_idx[mask.astype(bool)]] = 1
    fp = np.ones(num_vox) - tp

    fp = node_sum(fp, children=children)
    tp = node_sum(tp, children=children)

    return fp, tp



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
