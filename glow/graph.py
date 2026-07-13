"""Ward-tree region statistics and MANCOVA / LLR permutation kernels.

Regions are nodes of a binary Ward tree over voxels: the num_vox leaves
are single-voxel regions and each internal node is the union of its two
children, in topological (bottom-up) order. The tree is given as a
children array (num_internal, 2) of child index pairs, the
sklearn.cluster.Ward.children_ convention.

Two families of routines live here:

  - region aggregation: walk the tree once, re-using each region's
    sufficient statistics (size, ysum, yout) to build its parent's.
    iter_size_ysum_yout, iter_mancova and node_sum are the per-region
    streaming primitives; build_dfs_preorder / _reg_sum_cumsum trade the
    tree walk for a DFS pre-order layout plus cumsum-and-diff so an
    entire tree's region sums fall out of two index reads.

  - permutation LLR: compute_llr_batched scores one already-permuted
    experiment; iter_llr_perm streams the per-region LLR across many
    Freedman-Lane (Freedman & Lane 1983) permutation draws. Both feed
    the MANCOVA log-likelihood-ratio statistic in glow.analysis.mancova.

The remaining helpers (confusion_counts_tree, get_fp_tp, get_label_map,
get_parent, iter_postorder, SCGraph) are graph bookkeeping over the same
children representation.
"""
from collections import Counter

import numpy as np

from glow.analysis.mancova import decompose
from glow.mask import counts_from_tp_fp


def iter_size_ysum_yout(y, children=None):
    """Yield per-region sufficient statistics, reusing tree partial sums.

    Walks regions in topological order so each internal node's stats are
    the sum of its two children's; children are dropped from the cache
    once their last parent has consumed them.

    Args:
        y (np.array): (b, num_img, num_vox) imaging features
        children (np.array): (num_leaf - 1, 2) child index pairs. If None,
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

            # Drop each child once its last parent has consumed it, so the
            # cache holds only the active frontier rather than the whole tree.
            for c in (c0, c1):
                ref_count[c] -= 1
                if not ref_count[c]:
                    del out_dict[c]

            # stats of the union are the sums of the children's stats
            size = size0 + size1
            ysum = ysum0 + ysum1
            yout = yout0 + yout1

        out_dict[reg_idx] = size, ysum, yout
        yield reg_idx, size, ysum, yout


def iter_mancova(exp, **kwargs):
    """Yield region-level MANCOVA statistics (E, H), one pair per region.

    To obtain a permutation null distribution, callers loop externally
    over Freedman-Lane (Freedman & Lane 1983) permutations of the
    experiment:

        for k in range(n_perm + 1):
            _exp = exp.permute(k) if k else exp
            for reg_idx, size, e, h in iter_mancova(_exp, children=children):
                ...

    Args:
        exp (Experiment): experiment data
        **kwargs: forwarded to iter_size_ysum_yout (notably children for
            hierarchical regions)

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


def compute_llr_batched(exp, children, q0, q1, min_size: int = 1,
                        acc_dtype=np.float64):
    """Compute vectorised per-region LLR for one (already-permuted) exp.

    Computes the same per-region LLR statistic as the per-region loop:

        for reg_idx, size, e, h in iter_mancova(exp, children=children):
            llr[reg_idx] = get_llr(e, h, n=size)

    but in a single batched pass over numpy. Two phases:

    1. Bottom-up build of (size, ysum, yout) for ALL regions. Cannot be
       skipped for small regions because every internal node needs its
       children's ysum / yout. This is the cheap part (a few numpy adds
       per region).

    2. Per-region E / H / LLR via einsum + batched np.linalg.slogdet.
       This is where the bulk of the FLOPs live. When min_size > 1 we
       skip Phase 2 for regions with size < min_size and leave their LLR
       as NaN. At min_size=4 on typical neuroimaging trees, ~70% of
       regions drop out, cutting Phase 2's cost roughly proportionally.

    Args:
        exp (Experiment): experiment data (already FL-permuted)
        children (np.array): (num_internal, 2) child index pairs in
            topological (bottom-up) order
        q0 (np.array): (a0, num_img) nuisance subspace (from decompose)
        q1 (np.array): (a1, num_img) interest subspace (from decompose)
        min_size (int): regions with size < min_size get NaN LLR (and
            their E / H matrices are never computed). Default 1 keeps
            every region.
        acc_dtype: accumulation dtype for the sufficient statistics and the
            E / H assembly. Default np.float64 keeps E accurate for
            low-variance regions on a large DC offset, whose E cancels two
            terms of magnitude trace(yout) down to a far smaller residual;
            float32 there collapses E to rounding noise (see the Note).
            Exposed as float32 only to reproduce that collapse in tests.

    Returns:
        llr (np.array): (num_reg,) LLR per region. NaN where size <
            min_size, or where E (or E + H) is not positive definite (its
            slogdet sign <= 0) -- a genuinely degenerate region.
        size (np.array): (num_reg,) int voxel count per region.

    Note:
        E is formed by cancelling two terms of magnitude trace(yout) (the
        region's un-centred energy Sum y^2) down to the residual scatter,
        which for a near-constant region on a large offset is orders of
        magnitude smaller. In float32 that subtraction loses every digit: E
        becomes rounding noise, its inner-null std collapses, and the
        standardized z explodes, poisoning the max-z null. Accumulating in
        float64 keeps E accurate, so a plain positive-definite (sign) check
        suffices and a healthy large region -- which aggregates more voxels
        and so is better conditioned, not worse -- keeps its LLR. This
        mirrors iter_llr_perm, which computes the inner null the same way.
    """
    dtype = np.dtype(acc_dtype)
    y = np.asarray(exp.y, dtype=dtype)
    b, num_img, num_vox = y.shape
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

    # LLR = (size/2) * (ln|E+H| - ln|E|).  NaN where E or E + H is not
    # positive definite.  With float64 accumulation (acc_dtype) E is exact
    # enough that its slogdet sign is a reliable positive-definite test (see
    # the Note), so no scale floor is needed: a genuinely degenerate region
    # is dropped while a healthy large region -- better conditioned, not
    # worse -- keeps its LLR.  Matches iter_llr_perm's inner-null check.
    sign_t, logdet_t = np.linalg.slogdet(e + h)
    sign_e, logdet_e = np.linalg.slogdet(e)

    sz_a_1d = size[active]
    valid_a = (sign_t > 0) & (sign_e > 0)

    llr_a = np.where(valid_a,
                     (sz_a_1d / 2.0) * (logdet_t - logdet_e),
                     np.nan)
    llr[active] = llr_a

    return llr, size


def _slogdet_batched(M):
    """Compute batched log|det(M)| for the small symmetric matrices in LLR.

    Closed-form for b in {1, 2} -- significantly cheaper than
    np.linalg.slogdet's LU dispatch, which dominates the per-perm cost at
    b=2 in glow's inner loop. Falls back to numpy for larger b.

    Args:
        M (np.array): (..., b, b) stack of square matrices

    Returns:
        sign (np.array): (...,) sign of each determinant, numpy's slogdet API
        logabsdet (np.array): (...,) log|det(M)| per matrix
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


def build_dfs_preorder(children, num_vox: int):
    """Build a DFS pre-order leaf permutation and per-region leaf ranges.

    Given a (forest of) binary tree(s) on num_vox leaves in topological
    (bottom-up) order, this returns a permutation leaf_ord of the
    original voxel indices such that every region's leaves occupy a
    contiguous range [region_l[r], region_h[r]) on the permuted leaf
    axis. That is the prerequisite for cumsum-and-diff region aggregation
    (see _reg_sum_cumsum).

    Roots are laid out end-to-end -- the first root takes positions
    [0, size_root_0), the next takes [size_root_0, ...), etc.

    Args:
        children (np.array): (num_internal, 2) child index pairs in
            topological order -- each row references indices
            < num_vox + row_idx
        num_vox (int): number of leaves

    Returns:
        leaf_ord (np.array): (num_vox,) original voxel index visited at
            each DFS position -- i.e. y_dfs[..., k] = y[..., leaf_ord[k]]
        region_l (np.array): (num_reg,) leaf range start per region
        region_h (np.array): (num_reg,) leaf range end per region
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


def _reg_sum_cumsum(x_dfs, axis: int, region_l, region_h):
    """Compute per-region sums via cumsum-and-diff along axis.

    x_dfs is laid out in DFS pre-order along axis (length num_vox), so
    every region's voxels form a contiguous range. We prepend a zero
    slice along axis and cumsum into the rest, then index at region_h and
    region_l to read out half-open range sums. The l = 0 case is handled
    correctly because we left a zero in the prepended slot.

    Args:
        x_dfs (np.array): per-voxel values, DFS pre-order along axis
        axis (int): axis carrying the num_vox DFS-ordered voxels
        region_l (np.array): (num_reg,) leaf range start per region
        region_h (np.array): (num_reg,) leaf range end per region

    Returns:
        out (np.array): x_dfs with its axis (length num_vox) replaced by
            a per-region sum axis of length num_reg == len(region_l)
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

def iter_llr_perm(*, y, q0, q1, perms, leaf_ord, region_l, region_h,
                  min_size: int = 1, perm_chunk: int = 8,
                  acc_dtype=np.float64):
    """Stream per-region LLR across many Freedman-Lane permutations.

    Generator over chunks of size perm_chunk. Phase 1 (per-voxel
    sufficient statistics) is built once on entry; each iteration runs
    Phase 2 for the next Pc perms and yields one (Pc, num_reg) fp64
    chunk. Callers that need raw draws stack via
    np.vstack(list(iter_llr_perm(...))); callers that only need moments
    (e.g. inner_perm.cpu_perm) fold each chunk into a Welford accumulator
    and never materialize the full draws array. Closing the generator
    releases all Phase-1 state.

    Freedman-Lane permutation (Freedman & Lane 1983) permutes the
    nuisance residuals; here the permutation is carried on q0 / q1 rather
    than re-permuting y.

      1. Phase 1 (once) -- per-voxel sufficient statistics:

             T_v   = Y_v Y_v^T               (b, b)
             S0_v  = Q0 Y_v                  (a0, b)
             r_v   = (I - Q0 Q0^T) Y_v       (num_img, b)   FL residuals

         Aggregated over each region via cumsum-and-diff on the DFS
         pre-order axis -- no tree walk in the inner loop.

      2. Phase 2 (per perm) -- one (a, num_img) @ (num_img, num_vox*b)
         GEMM assembles gamma_v = Q^T P r_v for every voxel and feature.
         From gamma we read the FL-shifted rho (Q0 part) and beta (Q1
         part) needed to assemble E*, H* and finally the per-region LLR.

    Both intercept-only and general-Q0 nuisance ride this same code path:
    under intercept-only Q0 commutes with P, so rho is numerically zero
    (and the X_v cross-correction below vanishes up to roundoff). The
    waste is O(num_vox a0 b^2) FLOPs, dwarfed by the dominant
    O(num_vox num_img a b) gamma GEMM.

    Args:
        y (np.array): (b, num_img, num_vox) imaging features. The
            caller's original (unpermuted) data -- permutations are
            applied to q0 / q1 instead.
        q0 (np.array): (a0, num_img) nuisance subspace from decompose
        q1 (np.array): (a1, num_img) interest subspace
        perms (np.array): (n_perm, num_img) int -- perms[p, k] gives the
            original-image index that the FL-permuted data puts at
            position k. Matches glow's get_freed_lane convention so
            callers can build this as np.argsort(rng.permutation(num_img))
            per draw.
        leaf_ord (np.array): (num_vox,) from build_dfs_preorder
        region_l (np.array): (num_reg,) leaf range start, from
            build_dfs_preorder
        region_h (np.array): (num_reg,) leaf range end, from
            build_dfs_preorder
        min_size (int): regions with size < min_size return NaN in every
            chunk
        perm_chunk (int): number of perms to batch through one gamma
            GEMM. Trade-off: larger chunks reduce Python / BLAS call
            overhead but multiply the (Pc, num_vox, ...) temporary
            memory. Default 8 is the sweet spot empirically at num_vox in
            [25k, 55k] for both intercept-only and general-Q0.
        acc_dtype: accumulation dtype for the sufficient statistics and the
            E / H assembly. Default np.float64 keeps the per-region error
            matrix accurate for low-variance voxels on a large DC offset
            (float32 there collapses E to rounding noise; see the dtype note
            below). float32 is for tests reproducing that collapse only.

    Yields:
        llr_chunk (np.array): (Pc, num_reg) fp64 LLR draws for the next Pc
            perms (Pc <= perm_chunk; the final yield may be short). NaN
            for size < min_size or non-positive-definite E / E + H.
    """
    b, num_img, num_vox = y.shape
    n_perm = int(perms.shape[0])
    a0 = int(q0.shape[0])
    a1 = int(q1.shape[0])
    a = a0 + a1

    # Accumulate in float64 (acc_dtype default) even for float32 y.  Each
    # region's error matrix E is formed by cancelling two terms of magnitude
    # ~num_img * size * mean(y)^2 -- the raw second moment T_r and the nuisance
    # projection S0*^T S0* / size -- down to the residual
    # ~num_img * size * var(y).  For low-variance voxels on a large DC offset
    # (e.g. HCP background at mean -0.76, std 5e-4) that subtraction loses
    # every significant digit in float32: E collapses to rounding noise or
    # goes negative, so the per-region inner-null std degenerates (~1e-6
    # instead of ~5e-3) and the standardized z explodes, poisoning the
    # Westfall-Young max-z null.  float64
    # keeps E accurate; float32 only ever bought bandwidth (the dominant GEMM
    # could be re-narrowed in isolation if large-num_vox memory matters).
    # acc_dtype=float32 is exposed only to reproduce that float32 collapse in
    # tests (see test/analysis/test_inner_perm_hcp.py).
    dtype = np.dtype(acc_dtype)

    # -------------------- Phase 1: per-voxel state --------------------
    # Reorder y so its voxel axis is DFS pre-order; downstream cumsums
    # along that axis then deliver region sums via two index reads.
    y_dfs = np.ascontiguousarray(y[:, :, leaf_ord]).astype(dtype, copy=False)

    # S0_v[v, a, j] = Q0 Y_v in math = sum_n q0[a, n] * y_dfs[j, n, v]
    S0_v = np.einsum('an,jnv->vaj', q0, y_dfs, optimize=True)

    # T_v[v, i, j] = Y_v^T Y_v in math = sum_n y_dfs[i, n, v] * y_dfs[j, n, v]
    T_v = np.einsum('inv,jnv->vij', y_dfs, y_dfs, optimize=True)

    # Build the FL residuals r_v = (I - Q0 Q0^T) Y_v directly in the
    # (num_img, num_vox, b) layout the dominant GEMM needs, then reshape
    # to (num_img, num_vox*b) for free.  Holding r_v in (num_vox,
    # num_img, b) instead would force a non-contiguous transpose + copy
    # at reshape time, doubling peak memory at the 600k-voxel scale.
    r_v_nvb = np.ascontiguousarray(y_dfs.transpose(1, 2, 0))
    del y_dfs
    # subtract the Q0 projection in place to form the FL residuals
    r_v_nvb -= np.einsum('an,vaj->nvj', q0, S0_v, optimize=True)
    r_v_flat = r_v_nvb.reshape(num_img, num_vox * b)

    # one-time, permutation-invariant region statistics
    # S0_r: (num_reg, a0, b)
    S0_r = _reg_sum_cumsum(S0_v, axis=0,
                           region_l=region_l, region_h=region_h)
    # T_r: (num_reg, b, b)
    T_r = _reg_sum_cumsum(T_v, axis=0,
                          region_l=region_l, region_h=region_h)

    sz_1d = (region_h - region_l).astype(dtype)
    inv_sz = np.empty_like(sz_1d)
    np.divide(1.0, sz_1d, out=inv_sz, where=sz_1d > 0)
    inv_sz_3d = inv_sz[:, None, None]
    active = (region_h - region_l) >= min_size

    # Q: (a, num_img)
    Q = np.vstack([q0, q1]).astype(dtype, copy=False)

    # -------------------- Phase 2: per-perm hot loop -----------------
    for s in range(0, n_perm, perm_chunk):
        chunk = perms[s:s + perm_chunk]
        Pc = int(chunk.shape[0])

        # Match glow's FL convention: a permutation acts on the image
        # axis as y_perm[..., k] = y[..., perm[k]].  Then
        # (Q^T P r_v)[a, j] = sum_n Q[n, a] * r_v[perm[n], j], which
        # after substitution m = perm[n] reads off rows of Q at
        # perm^{-1}.  argsort inverts the perm.
        pi_inv = np.argsort(chunk, axis=1)
        # tQ: (Pc, a, num_img)
        tQ = Q[:, pi_inv].transpose(1, 0, 2)

        # Dominant compute: (Pc*a, num_img) @ (num_img, num_vox*b) ->
        # (Pc*a, num_vox*b).  Reshape lands gamma in (Pc, num_vox, a, b).
        gamma = (tQ.reshape(Pc * a, num_img) @ r_v_flat
                 ).reshape(Pc, a, num_vox, b).transpose(0, 2, 1, 3)

        # rho: (Pc, num_vox, a0, b)
        rho = gamma[..., :a0, :]
        # beta: (Pc, num_vox, a1, b)
        beta = gamma[..., a0:, :]

        # T_v's permutation-dependent correction: X_v = rho^T S0_v in
        # math; in our (num_vox, a, b) indexing that's an einsum over a.
        # X_v: (Pc, num_vox, b, b)
        X_v = np.einsum('pvai,vaj->pvij', rho, S0_v, optimize=True)

        # Region aggregation via cumsum-and-diff on the num_vox axis.
        # rho_r: (Pc, num_reg, a0, b)
        rho_r = _reg_sum_cumsum(rho, axis=1,
                                region_l=region_l, region_h=region_h)
        # beta_r: (Pc, num_reg, a1, b)
        beta_r = _reg_sum_cumsum(beta, axis=1,
                                 region_l=region_l, region_h=region_h)
        # X_r: (Pc, num_reg, b, b)
        X_r = _reg_sum_cumsum(X_v, axis=1,
                              region_l=region_l, region_h=region_h)

        # FL-shifted sufficient statistics per region.
        # S0_star: (Pc, num_reg, a0, b)
        S0_star = rho_r + S0_r
        # S1_star: (Pc, num_reg, a1, b)
        S1_star = beta_r
        # T_star: (Pc, num_reg, b, b)
        T_star = T_r + X_r + X_r.swapaxes(-1, -2)

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
                             np.nan).astype(np.float64, copy=False)
        yield llr_chunk


def build_survivor_kernels(y, children, survivor_idx, q0,
                           acc_dtype=np.float64):
    """Build per-survivor Freedman-Lane par/perp kernels (general Q0, low-rank).

    The survivor-only precompute for the race tail (inner_perm.cpu_perm_race).
    Under Freedman-Lane (Y_v* = P (I - Q0Q0^T) Y_v + Q0Q0^T Y_v) the
    sum-over-voxels b x b second moment of a region r decomposes as

        yout_perm[r] = yout_u[r] + C[r] + C[r]^T,
        C[r][i, j]   = sum_k y_par[i, k, .] . y_perp[j, perm[k], .]  (over r),

    with y_par = Q0Q0^T y, y_perp = y - y_par. The naive cross kernel
    M_r[i,j,k,l] = sum_v y_par[i,k,v] y_perp[j,l,v] is (b, b, N, N); but y_par
    lives in the a0-dim column space of Q0 (y_par[i,k,v] = sum_a Q0[a,k] s0[i,a,v],
    s0 = Q0 y), so M factors as M_r[i,j,k,l] = sum_a Q0[a,k] K_r[i,j,a,l]. We
    therefore store the LOW-RANK kernel

        K_r[i, j, a, l] = sum_{v in r} s0[i, a, v] y_perp[j, l, v]

    of shape (b, b, a0, N) -- a num_img/a0 reduction in both storage and build
    cost vs M. The per-draw cross term is then a contraction with a
    column-permuted Q0 (see compute_llr_inner_kernel), no gather of a dense
    tensor. Valid for ANY nuisance; survivor leaves are read off
    build_dfs_preorder's contiguous ranges. See
    publications/submissions/2026_glow/notes/inner_perm_race.tex and
    docs/notes/general_q0_race_kernel.md.

    Memory: num_surv * b^2 * a0 * num_img floats for K (vs the dense M's
    num_surv * b^2 * num_img^2).

    Args:
        y (np.array): (b, num_img, num_vox) raw voxel data, unpermuted
        children (np.array): (num_reg - num_vox, 2) Ward tree
        survivor_idx (np.array): (num_surv,) region indices to build kernels for
        q0 (np.array): (a0, num_img) nuisance subspace (orthonormal rows)
        acc_dtype: dtype the kernel arrays are stored in, and so the dtype
            compute_llr_inner_kernel assembles E in. Default np.float64 keeps E
            accurate for near-constant regions on a large DC offset, whose E
            cancels two terms of magnitude trace(yout) down to a far smaller
            residual; float32 there collapses E to rounding noise, its
            inner-null std collapses, and the standardized z explodes, poisoning
            the max-z null (see compute_llr_batched). Mirrors iter_llr_perm.
            Exposed as float32 only to reproduce that collapse in tests. The
            region sums are always built in float64 (the cumsum-and-diff below
            has its own small/deep-region cancellation, orthogonal to E's).

    Returns:
        kernels (dict): with keys
            K            (num_surv, b, b, a0, num_img) low-rank par/perp kernel
            ysum_u_S     (num_surv, b, num_img)  unpermuted per-region image sum
            yout_u_S     (num_surv, b, b)         unpermuted per-region Y Y^T
            size_S       (num_surv,)              voxel count per survivor
            survivor_idx (num_surv,)              pass-through region indices
        (K, ysum_u_S, yout_u_S are acc_dtype.)
    """
    num_vox = y.shape[2]
    out_dtype = np.dtype(acc_dtype)
    survivor_idx = np.asarray(survivor_idx, dtype=np.int64)
    leaf_ord, region_l, region_h = build_dfs_preorder(
        children=children, num_vox=num_vox)
    rl = region_l[survivor_idx]
    rh = region_h[survivor_idx] - 1

    def reg_sum(xvox):
        # Survivor region sums of a per-voxel array (DFS-ordered on the last
        # axis) by cumsum-and-diff, reading only the survivor [l, h) boundaries
        # -- every voxel touched once, no re-gathering of leaves shared by
        # nested survivors (the per-survivor loop's bottleneck). Accumulated in
        # float64: a small/deep region is the difference of two large partial
        # sums, so a float32 cumsum loses it to catastrophic cancellation
        # (a 4-voxel region at leaf ~6e5 carries ~1e-2 relative error in fp32).
        np.cumsum(xvox, axis=-1, out=xvox)
        hi = xvox[..., rh]
        lo = xvox[..., np.maximum(rl - 1, 0)]
        lo[..., rl == 0] = 0.0
        return np.ascontiguousarray(np.moveaxis(hi - lo, -1, 0))

    # per-voxel sufficient statistics in DFS leaf order (float64 for reg_sum)
    yf = y.astype(np.float64, copy=False)
    q0f = q0.astype(np.float64, copy=False)
    s0 = np.einsum('an,inv->iav', q0f, yf, optimize=True)      # (b, a0, num_vox)
    y_perp = yf - np.einsum('ak,iav->ikv', q0f, s0, optimize=True)
    yf_dfs = yf[:, :, leaf_ord]
    s0_dfs = s0[:, :, leaf_ord]
    y_perp_dfs = y_perp[:, :, leaf_ord]
    del yf, s0, y_perp

    # yout_u (b,b) reads yf_dfs; ysum_u (b,N) consumes it (in-place cumsum).
    yout_u = reg_sum(np.einsum('inv,jnv->ijv', yf_dfs, yf_dfs, optimize=True))
    ysum_u = reg_sum(yf_dfs)
    # K (b,b,a0,N): per-voxel kvox[i,j,a,l,v] = s0[i,a,v] y_perp[j,l,v]; the
    # full (b,b,a0,num_vox) prefix is transient, only survivor K is kept.
    kvox = s0_dfs[:, None, :, None, :] * y_perp_dfs[None, :, None, :, :]
    K = reg_sum(kvox)
    size = (region_h[survivor_idx] - region_l[survivor_idx]).astype(np.int64)
    return dict(K=K.astype(out_dtype, copy=False),
                ysum_u_S=ysum_u.astype(out_dtype, copy=False),
                yout_u_S=yout_u.astype(out_dtype, copy=False),
                size_S=size, survivor_idx=survivor_idx)


def compute_llr_inner_kernel(kernels, q0, q1, freed_lane, perm, num_reg,
                             min_size: int = 1):
    """Per-perm LLR for survivors only via the low-rank par/perp kernel (general Q0).

    The general-nuisance per-draw kernel: given build_survivor_kernels' output
    and one Freedman-Lane draw (freed_lane plus its index array perm), forms
    each survivor's permuted (ysum, yout), then the usual E / H / LLR. The
    cross term collapses to a contraction with a column-permuted Q0,

        C[s, i, j] = sum_{a, l} K[s, i, j, a, l] Q0[a, perm_inv[l]],

    so no dense tensor is gathered per draw; the permuted image sum folds the
    FL matrix into the (small) q0 / q1 bases first. Per-draw cost is
    O(num_surv * b^2 * a0 * num_img). Returns the same LLR as
    compute_llr_batched on the FL-permuted experiment (verified to fp).

    Args:
        kernels (dict): output of build_survivor_kernels
        q0 (np.array): (a0, num_img) nuisance basis (rows) -- the SAME passed to
            build_survivor_kernels
        q1 (np.array): (a1, num_img) interest basis (rows)
        freed_lane (np.array): (num_img, num_img) FL matrix from get_freed_lane
            (applied along the image axis), for the SAME draw as perm
        perm (np.array): (num_img,) the index array behind freed_lane
            (= permute._perm_indices(seed, num_img))
        num_reg (int): total region count (output array size)
        min_size (int): survivors with size < min_size get NaN LLR

    Returns:
        llr (np.array): (num_reg,) LLR -- NaN off-survivor and below min_size
    """
    K = kernels['K']
    ysum_u_S = kernels['ysum_u_S']
    yout_u_S = kernels['yout_u_S']
    size_S = kernels['size_S']
    survivor_idx = kernels['survivor_idx']
    dtype = K.dtype

    llr = np.full(num_reg, np.nan, dtype=np.float64)
    if survivor_idx.size == 0:
        return llr
    active_S = size_S >= min_size
    if not active_S.any():
        return llr

    # fold the FL matrix into the bases once (shared across survivors): the
    # permuted image sum only ever enters via its q0 / q1 projections.
    fl = freed_lane.astype(dtype, copy=False)
    fl_q0 = fl @ q0.T.astype(dtype, copy=False)            # (num_img, a0)
    fl_q1 = fl @ q1.T.astype(dtype, copy=False)            # (num_img, a1)

    # C[s, i, j] = sum_{a, l} K[s, i, j, a, l] Q0[a, perm_inv[l]]: one
    # contraction of K against the column-permuted Q0 (perm is a bijection, so
    # the gather k -> perm[k] becomes a reindex of Q0's columns by perm_inv).
    perm_inv = np.argsort(perm)
    q0pi = q0[:, perm_inv].astype(dtype, copy=False)       # (a0, num_img)
    C = np.einsum('sijal,al->sij', K, q0pi, optimize=True)  # (num_surv, b, b)
    yout_perm_S = yout_u_S + C + C.transpose(0, 2, 1)

    sz_a = size_S[active_S].astype(dtype)[:, None, None]
    ysum_a = ysum_u_S[active_S]
    yout_a = yout_perm_S[active_S]
    a0_S = np.einsum('sbn,na->sba', ysum_a, fl_q0, optimize=True)
    t_S = yout_a - np.einsum('sba,sca->sbc', a0_S, a0_S, optimize=True) / sz_a
    a1_S = np.einsum('sbn,na->sba', ysum_a, fl_q1, optimize=True)
    h_S = np.einsum('sba,sca->sbc', a1_S, a1_S, optimize=True) / sz_a
    e_S = t_S - h_S

    sign_t, ld_t = np.linalg.slogdet(e_S + h_S)
    sign_e, ld_e = np.linalg.slogdet(e_S)
    valid = (sign_t > 0) & (sign_e > 0)
    sz_1d = size_S[active_S]
    llr[survivor_idx[active_S]] = np.where(
        valid, (sz_1d / 2.0) * (ld_t - ld_e), np.nan)
    return llr


def node_sum(x, children):
    """Sum leaf values up through the tree to every node.

    Args:
        x (np.array): (num_leaf,) one value per leaf (single-voxel region)
        children (np.array): (num_leaf - 1, 2) child index pairs

    Returns:
        summed (np.array): (num_reg,) value for all nodes (leaves +
            internal), in topological order
    """
    num_leaf = x.size
    num_reg = num_leaf + children.shape[0]
    summed = np.empty(num_reg, dtype=x.dtype)
    summed[:num_leaf] = x

    for node_idx, (c0, c1) in enumerate(children):
        node_idx += num_leaf
        summed[node_idx] = summed[c0] + summed[c1]

    return summed


def confusion_counts_tree(mask, mask_idx, children) -> dict:
    """Per-region confusion counts, scoring every region as a target predictor.

    The tree analogue of glow.mask.confusion_counts: one pass over the Ward
    tree (get_fp_tp) yields tp / fp for all regions at once, then the shared
    glow.mask.counts_from_tp_fp fills in fn / tn. The detection metrics
    (Dice, sensitivity, PPV, specificity) follow via stats_from_counts; we
    return the counts so the caller may derive whichever it needs.

    Args:
        mask (np.array): target mask, boolean, same shape as mask_idx
        mask_idx (np.array): voxel index array (-1 outside analysis)
        children (np.array): (num_leaf - 1, 2) child index pairs (equiv to
            sklearn.cluster.Ward.children_)

    Returns:
        a dict {'tp', 'fp', 'tn', 'fn'} of (num_reg,) count arrays
    """
    fp, tp = get_fp_tp(mask, mask_idx, children)
    return counts_from_tp_fp(tp, fp, n_pos=mask.sum(),
                             n_total=float((mask_idx >= 0).sum()))


def get_fp_tp(mask, mask_idx, children):
    """Count false-positive and true-positive voxels per node.

    Treats each region as a predictor of the target mask.

    Args:
        mask (np.array): target mask, boolean, same shape as mask_idx
        mask_idx (np.array): voxel index array (-1 outside analysis)
        children (np.array): (num_leaf - 1, 2) child index pairs

    Returns:
        fp (np.array): (num_reg,) non-target voxels per node (in region,
            not in target)
        tp (np.array): (num_reg,) target voxels per node (in region and
            in target)
    """
    num_vox = (mask_idx >= 0).sum()
    tp = np.zeros(num_vox)
    tp[mask_idx[mask.astype(bool)]] = 1
    fp = np.ones(num_vox) - tp

    fp = node_sum(fp, children=children)
    tp = node_sum(tp, children=children)

    return fp, tp



def iter_postorder(*, children=None, num_leaf: int,
                   node_start: int | None = None, only_leaf: bool = False):
    """Traverse the tree in DFS post-order, yielding nodes leaves-to-root.

    Yields nodes in topological order. Supports forests: when node_start
    is None, iterates from every root (nodes with no parent).

    Args:
        children (np.array): (num_internal, 2) child index pairs
        num_leaf (int): number of leaves
        node_start (int | None): subtree root (defaults to all roots)
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


def get_parent(children, num_leaf: int):
    """Build a parent-lookup array for a binary tree.

    Args:
        children (np.array): (num_leaf - 1, 2) child index pairs
        num_leaf (int): number of leaves

    Returns:
        parent (np.array): (num_reg,) parent[idx] gives the parent of
            node idx, -1 for a root
    """
    num_nodes = num_leaf + children.shape[0]
    parent = np.full(num_nodes, -1, dtype=int)
    for i, (c0, c1) in enumerate(children):
        parent[c0] = parent[c1] = num_leaf + i

    return parent


class RegIntersectError(Exception):
    """Raised when regions that should be disjoint share voxels."""
    pass


def get_label_map(reg_idx_list, mask_idx, children,
                  check_disjoint: bool = False):
    """Build a label_map array from a list of region indices.

    Args:
        reg_idx_list (list[int]): region indices to include
        mask_idx (np.array): voxel index array, -1 outside of analysis,
            otherwise the voxel's index
        children (np.array): (num_internal, 2) child index pairs whose
            i-th row gives the children of node num_leaf + i. This
            representation contains a node for any node in all input
            graphs.
        check_disjoint (bool): if True, ensure no regions intersect

    Returns:
        label_map (np.array): same shape as mask_idx, -1 outside regions,
            reg_idx where voxel belongs to that region (smallest reg_idx
            if intersections)

    Raises:
        RegIntersectError: if check_disjoint and two regions overlap
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
    """A tree with short-circuited parent/child relations over a node subset.

    If A is a child of B and B a child of C in the full tree but B is
    excluded, SCGraph treats A as a direct child of C.

    Attributes:
        included (np.array): (num_node,) boolean, True if node is in the
            subgraph
        parent (np.array): (num_node,) short-circuit parent (-1 if
            excluded/root)
        children (dict): node -> sorted list of short-circuit children
    """

    @classmethod
    def from_children(cls, children, num_leaf: int, **kwargs):
        """Construct from a children array (sklearn.cluster.Ward.children_)."""
        parent = get_parent(children, num_leaf)
        return cls(parent, **kwargs)

    def __init__(self, parent, subset=None):
        """Build from a full-tree parent array, restricted to subset.

        Args:
            parent (np.array): (num_node,) full-tree parent index per
                node, -1 for a root
            subset: indices of nodes to include; None includes every node
        """
        if subset is None:
            included = np.ones(len(parent), dtype=bool)
        else:
            included = np.zeros(len(parent), dtype=bool)
            included[subset] = True

        self._parent_full = parent.copy()
        self.included = included.copy()
        self._rebuild()

    def modify(self, nodes_add=tuple(), nodes_rm=tuple()):
        """Add or remove nodes, then rebuild short-circuit relationships."""
        for node in nodes_add:
            self.included[node] = True
        for node in nodes_rm:
            self.included[node] = False
        self._rebuild()

    def _rebuild(self):
        """Rebuild the children dict with short-circuited relationships."""

        # walk up the full-tree parent chain to the nearest included node
        def _get_ss_parent(node):
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

        self.children = {k: sorted(v) for k, v in self.children.items()}

    def iter_desc(self, node, incl_self: bool = False):
        """Yield all descendants of node in the short-circuit subgraph."""
        assert self.included[node]
        if incl_self:
            yield node

        for _node in self.children[node]:
            yield from self.iter_desc(_node, incl_self=True)

    def iter_ancest(self, node, incl_self: bool = False):
        """Yield all ancestors of node in the short-circuit subgraph."""
        assert self.included[node]
        if incl_self:
            yield node

        while self.parent[node] != GRAPH_EXCLUDE:
            node = self.parent[node]
            yield node
