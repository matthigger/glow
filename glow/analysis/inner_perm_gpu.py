"""GPU inner Freedman-Lane permutation backend (hoisted precompute).

A device-side counterpart to inner_perm.cpu_perm with the same
keyword-only signature, so it drops into AnalysisGLOW.run_inner_perm.
Where cpu_perm rides glow.graph.iter_llr_perm (per-perm GEMM plus
cumsum-and-diff over the DFS pre-order voxel axis), this module hoists
every permutation-invariant quantity into a one-time precompute and
reduces each inner draw to a small row-permutation, one cumsum-and-diff,
and a closed-form determinant.

Two paths, dispatched on mancova.is_intercept_only_nuisance:

  - intercept-only. Q0 commutes with every permutation, so
    T_u = yout_u - a_0 a_0^T / size and log|T_u| are perm-invariant and
    hoist out of the loop entirely. Each draw builds only
    H = a_1 a_1^T / size from the row-permuted Q1 Y, then E = T_u - H.
  - general-Q0. Q0 Q0^T no longer commutes, so yout_perm and a_0 vary
    per draw. They split into a perm-invariant beta (built once) and a
    perm-dependent alpha (one wide matmul per chunk), with the cross
    term entering as yout_perm = yout_u + T_C + T_C^T.

Both reduce to the same per-region LLR as glow.graph.compute_llr_batched
on the Freedman-Lane permuted experiment (Freedman & Lane 1983).

Permutation convention. glow stores data in row layout, so
permute.get_freed_lane returns the column-layout transpose
(I - Q0 Q0^T)[:, perm] + Q0 Q0^T; glow's perm IS the textbook pi, not
its inverse. The two paths gather OPPOSITE directions: intercept-only
reads Q1[:, pi^-1] (the residual projector collapses against Q1 since
Q1 Q0^T = 0), general-Q0 gathers R[:, pi] (no cancellation, and the
alpha factorisation is clean only in this direction). _chunk_args_for_path
hands each path its own tensor.

Three device-specific choices that carry most of the speed. All three
were measured, not guessed; changing any of them costs an order of
magnitude:

  - the voxel axis stays LAST through the whole per-chunk loop. cumsum
    and index_select on a non-final axis are non-coalesced and run ~16x
    slower (14ms vs 0.9ms at num_vox=100k).
  - the (b, b) Gram products go through _gram's elementwise broadcast,
    never matmul or einsum. cuBLAS's batched-GEMM per-matrix overhead is
    catastrophic on the tiny (b, b) outputs here -- ~33ms vs 0.06ms at
    400k matrices of shape (2, 2). The crossover sits near b = 8.
  - region aggregation is cumsum-and-diff on the DFS pre-order axis
    (3 kernel launches), not a layer-by-layer tree sweep (O(log num_vox)
    launches per channel).

Accumulation dtype. acc_dtype defaults to float64 and is deliberately
NOT taken from y.dtype: each region's E cancels two terms of magnitude
trace(yout) down to a far smaller residual, and in float32 a
near-constant region on a large DC offset loses every digit -- E becomes
rounding noise, the inner-null std collapses, and the standardized z
explodes into the max-z null. This mirrors graph.iter_llr_perm and
graph.compute_llr_batched. float32 is exposed for the capacity /
throughput map and for tests that reproduce that collapse; note fp64
runs at 1/64 of fp32 throughput on consumer Ada parts, so the dtype is
also the dominant speed knob.
"""
import numpy as np

import glow.graph
from glow.experiment import permute
from .mancova import is_intercept_only_nuisance


# b range where the closed-form (cofactor) determinant covers CUDA graph
# capture; torch.linalg.slogdet dispatches to cuSOLVER's LU, which needs
# runtime workspace and is not reliably capture-safe.
_CAPTURE_B_MAX_CLOSED_FORM = 4


def is_available() -> bool:
    """Return True iff torch is importable and a CUDA device is visible.

    Cheap probe: no device work, no exception when torch is absent.
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _torch_dtype(acc_dtype):
    """Map a numpy accumulation dtype to (np.dtype, torch dtype)."""
    import torch
    np_dtype = np.dtype(acc_dtype)
    if np_dtype == np.float32:
        return np_dtype, torch.float32
    if np_dtype == np.float64:
        return np_dtype, torch.float64
    raise ValueError(f'acc_dtype must be float32 or float64, got {np_dtype}')


def _gram(a):
    """Compute the Gram matrix of a's matrix axis by elementwise reduce.

    Args:
        a (torch.Tensor): (..., b, k)

    Returns:
        h (torch.Tensor): (..., b, b) with
            h[..., i, j] = sum_k a[..., i, k] a[..., j, k]

    Implemented as unsqueeze-multiply-sum rather than a @ a.transpose:
    both einsum and matmul dispatch to batched cuBLAS GEMM, which is
    ~500x slower than the elementwise reduce at the batch counts and
    tiny (b, b) sizes here (see the module docstring).
    """
    return (a.unsqueeze(-2) * a.unsqueeze(-3)).sum(dim=-1)


def _slogdet_batched(m):
    """Compute batched (sign, log|det|) for the small LLR matrices.

    Closed-form cofactor expansion for b in {1, 2, 3, 4} -- cheaper than
    torch.linalg.slogdet's LU dispatch at these sizes (mirroring
    graph._slogdet_batched) and, unlike it, pure elementwise tensor ops
    and so CUDA-graph capturable. Falls back to torch.linalg.slogdet for
    b >= 5.

    Args:
        m (torch.Tensor): (..., b, b) stack of symmetric matrices

    Returns:
        sign (torch.Tensor): (...,) sign of each determinant
        logabsdet (torch.Tensor): (...,) log|det| per matrix
    """
    import torch
    b = m.shape[-1]
    if b == 1:
        d = m[..., 0, 0]
        return torch.sign(d), torch.log(torch.abs(d))
    if b == 2:
        det = m[..., 0, 0] * m[..., 1, 1] - m[..., 0, 1] * m[..., 1, 0]
        return torch.sign(det), torch.log(torch.abs(det))
    if b == 3:
        a00 = m[..., 0, 0]; a01 = m[..., 0, 1]; a02 = m[..., 0, 2]
        a10 = m[..., 1, 0]; a11 = m[..., 1, 1]; a12 = m[..., 1, 2]
        a20 = m[..., 2, 0]; a21 = m[..., 2, 1]; a22 = m[..., 2, 2]
        det = (a00 * (a11 * a22 - a12 * a21)
               - a01 * (a10 * a22 - a12 * a20)
               + a02 * (a10 * a21 - a11 * a20))
        return torch.sign(det), torch.log(torch.abs(det))
    if b == 4:
        a00 = m[..., 0, 0]; a01 = m[..., 0, 1]
        a02 = m[..., 0, 2]; a03 = m[..., 0, 3]
        a10 = m[..., 1, 0]; a11 = m[..., 1, 1]
        a12 = m[..., 1, 2]; a13 = m[..., 1, 3]
        a20 = m[..., 2, 0]; a21 = m[..., 2, 1]
        a22 = m[..., 2, 2]; a23 = m[..., 2, 3]
        a30 = m[..., 3, 0]; a31 = m[..., 3, 1]
        a32 = m[..., 3, 2]; a33 = m[..., 3, 3]
        # the six 2x2 minors of rows 2, 3 are shared by all four cofactors
        m22_33 = a22 * a33 - a23 * a32
        m21_33 = a21 * a33 - a23 * a31
        m21_32 = a21 * a32 - a22 * a31
        m20_33 = a20 * a33 - a23 * a30
        m20_32 = a20 * a32 - a22 * a30
        m20_31 = a20 * a31 - a21 * a30
        c0 = a11 * m22_33 - a12 * m21_33 + a13 * m21_32
        c1 = a10 * m22_33 - a12 * m20_33 + a13 * m20_32
        c2 = a10 * m21_33 - a11 * m20_33 + a13 * m20_31
        c3 = a10 * m21_32 - a11 * m20_32 + a12 * m20_31
        det = a00 * c0 - a01 * c1 + a02 * c2 - a03 * c3
        return torch.sign(det), torch.log(torch.abs(det))
    return torch.linalg.slogdet(m)


def _reg_sum_cumsum(x_dfs, dim, region_l, region_h):
    """Compute per-region sums via cumsum-and-diff along dim.

    Device port of glow.graph._reg_sum_cumsum. x_dfs is laid out in DFS
    pre-order along dim (length num_vox), so every region occupies a
    contiguous range; cumsum, prepend a zero slice, then read out
    half-open range sums at region_h and region_l.

    Allocates fresh tensors rather than writing into a view, keeping the
    op sequence CUDA-graph friendly. region_l / region_h must already be
    on x_dfs's device.

    Args:
        x_dfs (torch.Tensor): per-voxel values, DFS pre-order along dim
        dim (int): the axis carrying the num_vox DFS-ordered voxels
        region_l (torch.Tensor): (num_reg,) leaf range start per region
        region_h (torch.Tensor): (num_reg,) leaf range end per region

    Returns:
        out (torch.Tensor): x_dfs with its dim replaced by a num_reg axis
    """
    import torch
    cs = torch.cumsum(x_dfs, dim=dim)
    pad_shape = list(x_dfs.shape)
    pad_shape[dim] = 1
    pad = torch.zeros(pad_shape, dtype=x_dfs.dtype, device=x_dfs.device)
    c = torch.cat([pad, cs], dim=dim)
    return (torch.index_select(c, dim, region_h)
            - torch.index_select(c, dim, region_l))


# ---------------------------------------------------------------------------
# Phase 1 precompute + per-chunk hot loop: intercept-only nuisance

def _prep_intercept_state(*, y, q0, q1, children, min_vox, device,
                          acc_dtype, center: bool = True,
                          scan_dtype=np.float64):
    """Build the perm-invariant device state for intercept-only nuisance.

    Uploads y, reorders its voxel axis to DFS pre-order on device (a
    host-side y[:, :, leaf_ord] gather costs ~14ms at num_vox=25k and
    blocks the CPU that could be building the next tree), then forms the
    permutation-invariant region statistics: T_u = E + H and log|T_u|.

    Assumes the caller verified is_intercept_only_nuisance -- without it
    T_u is not permutation-invariant and hoisting it is wrong.

    Two ways to reach the same T_u:

      center=False forms it as written, T_u = yout_u - a_0 a_0^T / size.
        Both terms scale as num_img * size * mean(y)^2 while their
        difference is the residual scatter, so a near-constant voxel on a
        large DC offset (HCP background: mean -0.76, across-image std
        5e-4) cancels ~7e-8 of the leading term -- below float32 eps.
        That is the a55e7237 FWER collapse (test_inner_perm_hcp.py).

      center=True (default) splits the same quantity into two pieces that
        each carry no cancellation. With s = Q0 y the per-voxel nuisance
        coefficients and w = y - Q0^T s the per-voxel residual,

            T_u = W_r + S_r
            W_r = sum_{v in r} sum_n w w^T
            S_r = sum_a [sum_v s_a s_a^T
                         - (sum_v s_a)(sum_v s_a)^T / size]

        The cross terms vanish exactly because Q0 w = 0, so this is an
        identity, not an approximation. W_r is a sum of squares of
        already-centered data; S_r is the spatial scatter of the nuisance
        coefficients, identically zero for a leaf region and computed at
        scan_dtype. The DC offset never enters a subtraction, so the hot
        loop is safe in float32.

    Q1 w == Q1 y for intercept-only nuisance (Q1 Q0^T = 0 and a
    permutation fixes Q0's constant rows), so the per-draw path is
    unchanged by centering -- state['Y'] carries w and H is identical.

    Args:
        y (np.array): (b, num_img, num_vox) imaging features, unpermuted
        q0 (np.array): (a0, num_img) nuisance subspace from decompose
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are inactive
        device (str): torch device string
        acc_dtype: hot-loop accumulation dtype (see the module docstring)
        center (bool): build T_u by the cancellation-free split above.
            False reproduces the direct form, for tests that exhibit the
            float32 collapse.
        scan_dtype: dtype for the perm-invariant region scans that build
            T_u. Kept float64 independent of acc_dtype: a region sum is
            cum[h] - cum[l] of two prefixes that grow like num_vox, so a
            small deep region loses ~eps * num_vox / size relatively --
            ~1e-2 in float32 at full-brain scale. These scans run once
            per tree, not per draw, and fp64 costs 2x bandwidth (not the
            64x it costs on FLOPs), so accuracy here is nearly free.

    Returns:
        state (dict): device tensors and scalars consumed by
            _intercept_chunk_llr:
            {dev, torch_dtype, b, num_img, num_vox, a0, a1, num_reg,
             Y (b, num_img, num_vox), Q1 (a1, num_img),
             region_l_t, region_h_t, size_f, active (num_reg,),
             T_u (num_reg, b, b), log_T_u, sign_T (num_reg,)}
    """
    import torch

    b, num_img, num_vox = y.shape
    np_dtype, torch_dtype = _torch_dtype(acc_dtype)
    _, scan_torch = _torch_dtype(scan_dtype)
    dev = torch.device(device)

    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=num_vox)
    region_l_t = torch.from_numpy(region_l.astype(np.int64)).to(dev)
    region_h_t = torch.from_numpy(region_h.astype(np.int64)).to(dev)
    num_reg = int(region_l.shape[0])

    leaf_ord_t = torch.from_numpy(leaf_ord.astype(np.int64)).to(dev)
    q0_t = torch.from_numpy(
        np.ascontiguousarray(q0).astype(np_dtype, copy=False)).to(dev)
    q1_t = torch.from_numpy(
        np.ascontiguousarray(q1).astype(np_dtype, copy=False)).to(dev)

    size_d = region_h_t - region_l_t
    size_f = size_d.to(torch_dtype)
    active = size_d >= min_vox
    inv_size_scan = torch.where(
        size_d > 0, 1.0 / size_d.to(scan_torch),
        torch.zeros(num_reg, dtype=scan_torch, device=dev))

    if center:
        # Split y into nuisance coefficients and per-voxel residual on the
        # host in float64: this is the one place the DC offset is removed,
        # and doing it before the narrowing cast is what makes float32
        # viable downstream.
        y64 = np.ascontiguousarray(y).astype(np.float64, copy=False)
        q064 = np.ascontiguousarray(q0).astype(np.float64, copy=False)
        s_np = np.einsum('an,bnv->abv', q064, y64, optimize=True)
        w_np = y64 - np.einsum('an,abv->bnv', q064, s_np, optimize=True)

        y_dfs = torch.from_numpy(
            np.ascontiguousarray(w_np).astype(np_dtype, copy=False)
        ).to(dev).index_select(2, leaf_ord_t).contiguous()
        s_t = torch.from_numpy(
            np.ascontiguousarray(s_np).astype(np.float64, copy=False)
        ).to(dev).to(scan_torch).index_select(2, leaf_ord_t).contiguous()
        del y64, q064, s_np, w_np

        # W_r: per-voxel Gram of centered w (no cancellation, so the
        # narrow hot-loop dtype is fine here), summed at scan_dtype.
        w_gram = torch.einsum('bnv,cnv->bcv', y_dfs, y_dfs).to(scan_torch)
        w_r = _reg_sum_cumsum(w_gram, -1, region_l_t, region_h_t)
        del w_gram

        # S_r: spatial scatter of the nuisance coefficients, summed over
        # nuisance lanes.  Zero for a leaf, so a leaf's T_u is exactly W_r.
        s_gram = torch.einsum('abv,acv->bcv', s_t, s_t)
        s_sq_r = _reg_sum_cumsum(s_gram, -1, region_l_t, region_h_t)
        s_sum_r = _reg_sum_cumsum(s_t, -1, region_l_t, region_h_t)
        del s_gram
        s_r = s_sq_r - torch.einsum('abr,acr->bcr', s_sum_r, s_sum_r) \
                            * inv_size_scan[None, None, :]

        t_u_scan = (w_r + s_r).permute(2, 0, 1).contiguous()
        del w_r, s_r, s_sq_r, s_sum_r, s_t
    else:
        y_dfs = torch.from_numpy(
            np.ascontiguousarray(y).astype(np_dtype, copy=False)
        ).to(dev).index_select(2, leaf_ord_t).contiguous()

        yout_leaf = torch.einsum('bnv,cnv->bcv', y_dfs, y_dfs)
        yout_u = _reg_sum_cumsum(yout_leaf, -1, region_l_t, region_h_t)
        z0_leaf = torch.einsum('an,bnv->abv', q0_t, y_dfs)
        a_0 = _reg_sum_cumsum(z0_leaf, -1, region_l_t, region_h_t)
        t_u_scan = (yout_u
                    - torch.einsum('abr,acr->bcr', a_0, a_0)
                      * inv_size_scan.to(yout_u.dtype)[None, None, :]
                    ).permute(2, 0, 1).contiguous()
        del yout_leaf, yout_u, z0_leaf, a_0

    # The determinant of T_u is taken at scan_dtype (once per tree) and
    # only then narrowed: the per-draw loop differences log|T_u| against
    # log|E|, so both wanting the hot-loop dtype is what keeps that
    # subtraction consistent.
    sign_t, log_t_u = _slogdet_batched(t_u_scan)

    return dict(
        dev=dev, torch_dtype=torch_dtype,
        b=b, num_img=num_img, num_vox=num_vox,
        a0=int(q0.shape[0]), a1=int(q1.shape[0]), num_reg=num_reg,
        Y=y_dfs, Q1=q1_t,
        region_l_t=region_l_t, region_h_t=region_h_t,
        size_f=size_f, active=active,
        T_u=t_u_scan.to(torch_dtype),
        log_T_u=log_t_u.to(torch_dtype),
        sign_T=sign_t.to(torch_dtype))


def _intercept_chunk_llr(chunk_inv_t, state):
    """Compute one chunk of draws on the intercept-only path.

    Args:
        chunk_inv_t (torch.Tensor): (Pc, num_img) int64 inverse
            permutations pi^-1 -- the rows of Q1 to read for the
            FL-permuted Q1 Y (see the module docstring)
        state (dict): _prep_intercept_state output

    Returns:
        llr (torch.Tensor): (Pc, num_reg) per-region LLR
        valid (torch.Tensor): (Pc, num_reg) bool, False where the region
            is inactive or E / T_u is not positive definite
    """
    import torch

    y_dfs = state['Y']
    q1_t = state['Q1']
    t_u = state['T_u']
    log_t_u = state['log_T_u']
    sign_t = state['sign_T']
    size_f = state['size_f']
    active = state['active']

    q1_perm = q1_t[:, chunk_inv_t].permute(1, 0, 2).contiguous()
    z_1 = torch.einsum('pkn,bnv->pkbv', q1_perm, y_dfs)
    a_1 = _reg_sum_cumsum(z_1, -1, state['region_l_t'], state['region_h_t'])

    h = (_gram(a_1.permute(0, 3, 2, 1).contiguous())
         / size_f[None, :, None, None])
    e = t_u[None] - h

    sign_e, log_e = _slogdet_batched(e)
    valid = active[None] & (sign_t[None] > 0) & (sign_e > 0)
    llr = 0.5 * size_f[None] * (log_t_u[None] - log_e)
    return llr, valid


# ---------------------------------------------------------------------------
# Phase 1 precompute + per-chunk hot loop: general Q0
#
# Q0 Q0^T does not commute with a permutation, so yout_perm and a_0 vary
# per draw.  Both split into a perm-invariant beta (one-time) and a
# perm-dependent alpha (one wide matmul per chunk); the cross term is
# T_C = alpha beta, entering as yout_perm = yout_u + T_C + T_C^T.

def _prep_general_state(*, y, q0, q1, children, min_vox, device,
                        acc_dtype, matmul_dtype=None):
    """Build the perm-invariant device state for general-Q0 nuisance.

    Args match _prep_intercept_state, plus:
        matmul_dtype: dtype for the dominant per-chunk alpha matmul.
            Defaults to acc_dtype. Narrowing it (bfloat16 is ~30x faster
            than float32 on consumer Ada) feeds a reduced-mantissa
            product into E's cancellation, so it is opt-in only and
            wants the same validation as acc_dtype=float32.

    Returns:
        state (dict): as _prep_intercept_state, plus q01_T
            (num_img, a0 + a1), R_mat (num_img, num_img),
            y_2d_mm (b * num_vox, num_img), beta_a0 (num_reg, b, a0),
            beta_leaf (num_vox, b, a0), yout_u (num_reg, b, b)
    """
    import torch

    b, num_img, num_vox = y.shape
    a0 = int(q0.shape[0])
    a1 = int(q1.shape[0])
    np_dtype, torch_dtype = _torch_dtype(acc_dtype)
    dev = torch.device(device)
    if matmul_dtype is None:
        matmul_dtype = torch_dtype

    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=num_vox)
    region_l_t = torch.from_numpy(region_l.astype(np.int64)).to(dev)
    region_h_t = torch.from_numpy(region_h.astype(np.int64)).to(dev)
    num_reg = int(region_l.shape[0])

    leaf_ord_t = torch.from_numpy(leaf_ord.astype(np.int64)).to(dev)
    y_raw = torch.from_numpy(
        np.ascontiguousarray(y).astype(np_dtype, copy=False)).to(dev)
    y_dfs = y_raw.index_select(2, leaf_ord_t).contiguous()
    del y_raw
    q0_t = torch.from_numpy(
        np.ascontiguousarray(q0).astype(np_dtype, copy=False)).to(dev)
    q1_t = torch.from_numpy(
        np.ascontiguousarray(q1).astype(np_dtype, copy=False)).to(dev)

    # Q0's rows are the orthonormal nuisance directions, so Q0^T Q0 is
    # the (num_img, num_img) projector onto their span.
    r_mat = (torch.eye(num_img, dtype=torch_dtype, device=dev)
             - q0_t.T @ q0_t)

    size_d = region_h_t - region_l_t
    size_f = size_d.to(torch_dtype)
    active = size_d >= min_vox

    y_2d = y_dfs.permute(0, 2, 1).reshape(b * num_vox, num_img)

    beta_leaf_bav = (y_2d @ q0_t.T).reshape(b, num_vox, a0) \
                                  .permute(0, 2, 1).contiguous()
    beta_a0 = _reg_sum_cumsum(beta_leaf_bav, -1, region_l_t, region_h_t)
    beta_a0 = beta_a0.permute(2, 0, 1).contiguous()
    beta_leaf = beta_leaf_bav.permute(2, 0, 1).contiguous()

    yout_leaf = torch.einsum('bnv,cnv->bcv', y_dfs, y_dfs)
    yout_u = _reg_sum_cumsum(yout_leaf, -1, region_l_t, region_h_t)
    yout_u = yout_u.permute(2, 0, 1).contiguous()

    return dict(
        dev=dev, torch_dtype=torch_dtype, matmul_dtype=matmul_dtype,
        b=b, num_img=num_img, num_vox=num_vox,
        a0=a0, a1=a1, ak=a0 + a1, num_reg=num_reg,
        Y=y_dfs, Q0=q0_t, Q1=q1_t, R_mat=r_mat,
        q01_T=torch.cat([q0_t.T, q1_t.T], dim=1).contiguous(),
        y_2d_mm=y_2d.to(matmul_dtype),
        region_l_t=region_l_t, region_h_t=region_h_t,
        size_f=size_f, active=active,
        beta_a0=beta_a0, beta_leaf=beta_leaf, yout_u=yout_u)


def _general_chunk_llr(chunk_t, state):
    """Compute one chunk of draws on the general-Q0 path.

    Per draw (dropping the chunk index), with R = I - Q0^T Q0:

        U       = R[:, pi] [Q0^T | Q1^T]                (num_img, a0 + a1)
        alpha   = Y_2d U                                (b, num_vox, a0 + a1)
        T_C     = sum_a alpha[..., :a0] beta_a0         per region
        T       = (yout_u + T_C + T_C^T) - a_0 a_0^T / size
        H       = alpha_a1 alpha_a1^T / size
        LLR     = 0.5 size (log|T| - log|T - H|)

    Args:
        chunk_t (torch.Tensor): (Pc, num_img) int64 FORWARD permutations
            pi -- the opposite direction from the intercept-only path
            (see the module docstring)
        state (dict): _prep_general_state output

    Returns:
        llr (torch.Tensor): (Pc, num_reg) per-region LLR
        valid (torch.Tensor): (Pc, num_reg) bool positive-definite mask
    """
    import torch

    r_mat = state['R_mat']
    q01_t = state['q01_T']
    torch_dtype = state['torch_dtype']
    beta_a0 = state['beta_a0']
    yout_u = state['yout_u']
    size_f = state['size_f']
    active = state['active']
    region_l_t = state['region_l_t']
    region_h_t = state['region_h_t']
    num_vox = state['num_vox']
    num_img = state['num_img']
    b = state['b']
    a0 = state['a0']
    ak = state['ak']
    pc = int(chunk_t.shape[0])

    perm_exp = chunk_t.unsqueeze(1).expand(pc, num_img, num_img)
    r_perm = torch.gather(
        r_mat.unsqueeze(0).expand(pc, num_img, num_img), 2, perm_exp)
    u = torch.matmul(r_perm, q01_t)
    x = u.permute(1, 0, 2).reshape(num_img, pc * ak).contiguous() \
         .to(state['matmul_dtype'])

    alpha = torch.mm(state['y_2d_mm'], x).to(torch_dtype)
    alpha = alpha.reshape(b, num_vox, pc, ak) \
                 .permute(2, 0, 3, 1).contiguous()
    alpha_a0_leaf = alpha[:, :, :a0, :]
    alpha_a1_leaf = alpha[:, :, a0:, :]

    beta_leaf_bav = state['beta_leaf'].permute(1, 2, 0)
    t_c_leaf = (alpha_a0_leaf.unsqueeze(2)
                * beta_leaf_bav.unsqueeze(0).unsqueeze(1)).sum(dim=3)

    alpha_a0_r = _reg_sum_cumsum(alpha_a0_leaf, -1, region_l_t, region_h_t)
    alpha_a1_r = _reg_sum_cumsum(alpha_a1_leaf, -1, region_l_t, region_h_t)
    t_c_r = _reg_sum_cumsum(t_c_leaf, -1, region_l_t, region_h_t)

    alpha_a0_r = alpha_a0_r.permute(0, 3, 1, 2).contiguous()
    alpha_a1_r = alpha_a1_r.permute(0, 3, 1, 2).contiguous()
    t_c_r = t_c_r.permute(0, 3, 1, 2).contiguous()

    a0_full = alpha_a0_r + beta_a0[None]
    yout_perm = yout_u[None] + t_c_r + t_c_r.transpose(-1, -2)

    inv_sz = (1.0 / size_f)[None, :, None, None]
    t = yout_perm - _gram(a0_full) * inv_sz
    h = _gram(alpha_a1_r) * inv_sz
    e = t - h

    sign_t, log_t = _slogdet_batched(t)
    sign_e, log_e = _slogdet_batched(e)
    valid = active[None] & (sign_t > 0) & (sign_e > 0)
    llr = 0.5 * size_f[None] * (log_t - log_e)
    return llr, valid


# ---------------------------------------------------------------------------
# Dispatch and public API

def _dispatch_state(*, exp, q0, q1, children, min_vox, device, acc_dtype,
                    matmul_dtype=None, force_path=None, center: bool = True,
                    scan_dtype=np.float64):
    """Build the phase-1 state matching exp's nuisance regime.

    Args:
        force_path (str): 'intercept' or 'general' to skip the dispatch
            (tests and benchmarks); None dispatches on
            mancova.is_intercept_only_nuisance.
        center (bool): intercept-only path only -- build T_u by the
            cancellation-free split (see _prep_intercept_state).
        scan_dtype: intercept-only path only -- dtype of the
            perm-invariant region scans.

    Returns:
        state (dict): the phase-1 state
        chunk_llr (callable): the matching per-chunk function
    """
    assert force_path in (None, 'intercept', 'general'), force_path
    if force_path is None:
        intercept = is_intercept_only_nuisance(exp.x, exp.contrast)
    else:
        intercept = force_path == 'intercept'
    if intercept:
        state = _prep_intercept_state(
            y=exp.y, q0=q0, q1=q1, children=children, min_vox=min_vox,
            device=device, acc_dtype=acc_dtype, center=center,
            scan_dtype=scan_dtype)
        return state, _intercept_chunk_llr
    state = _prep_general_state(
        y=exp.y, q0=q0, q1=q1, children=children, min_vox=min_vox,
        device=device, acc_dtype=acc_dtype, matmul_dtype=matmul_dtype)
    return state, _general_chunk_llr


def _build_perms(base_seed: int, n_perm: int, num_img: int):
    """Build the (n_perm, num_img) FL index array; draw i uses base_seed + i.

    Matches inner_perm.cpu_perm's construction exactly, so a given
    (base_seed, n_perm) yields the same draws on both backends.
    """
    perms = np.empty((n_perm, num_img), dtype=np.int64)
    for i in range(n_perm):
        perms[i] = permute._perm_indices(base_seed + i, num_img)
    return perms


def _build_perm_tensors(perms, device):
    """Upload the forward perms pi and their inverses pi^-1 to device.

    Both directions are needed because the two paths gather opposite
    ones (see the module docstring). argsort runs once on the host so the
    hot loop never pays it.
    """
    import torch
    perms_t = torch.from_numpy(
        np.ascontiguousarray(perms).astype(np.int64, copy=False)).to(device)
    perms_inv_t = torch.from_numpy(
        np.argsort(perms, axis=1).astype(np.int64)).to(device)
    return perms_t, perms_inv_t


def _chunk_perms_for_path(chunk_llr, perms_t, perms_inv_t):
    """Return the perm tensor the chosen path gathers with."""
    if chunk_llr is _intercept_chunk_llr:
        return perms_inv_t
    return perms_t


def _chan_combine(llr, valid, n, mean, m2):
    """Fold one (Pc, num_reg) draw-chunk into running per-region moments.

    Device port of inner_perm._welford_combine: one step of Chan's
    parallel-combine rule (Chan, Golub & LeVeque 1979), with invalid
    cells excluded from the count. The CPU path uses this rather than a
    naive sum / sum-of-squares pass because var << mean^2 here (LLR
    carries a 0.5 * size prefactor), where the textbook one-pass formula
    loses precision to cancellation.

    Args:
        llr (torch.Tensor): (Pc, num_reg) draws, arbitrary where invalid
        valid (torch.Tensor): (Pc, num_reg) bool validity mask
        n (torch.Tensor): (num_reg,) running valid-sample count
        mean (torch.Tensor): (num_reg,) running mean
        m2 (torch.Tensor): (num_reg,) running sum of squared deviations

    Returns:
        n, mean, m2 (torch.Tensor): the updated (num_reg,) accumulators
    """
    import torch
    llr_safe = torch.where(valid, llr, torch.zeros_like(llr)).to(n.dtype)
    n_b = valid.sum(dim=0).to(n.dtype)
    sum_b = llr_safe.sum(dim=0)

    ones = torch.ones_like(n_b)
    safe_nb = torch.where(n_b > 0, n_b, ones)
    mean_b = sum_b / safe_nb
    dev = torch.where(valid, llr_safe - mean_b[None, :],
                      torch.zeros_like(llr_safe))
    m2_b = (dev * dev).sum(dim=0)

    new_n = n + n_b
    safe_new_n = torch.where(new_n > 0, new_n, torch.ones_like(new_n))
    delta = mean_b - mean
    mean = mean + delta * (n_b / safe_new_n)
    m2 = m2 + m2_b + delta * delta * (n * n_b / safe_new_n)
    return new_n, mean, m2


def _chan_finalize(n, mean, m2):
    """Reduce running (n, mean, m2) to (mu, std) as numpy arrays.

    Mirrors inner_perm._welford_finalize: NaN mu where no valid sample
    accumulated, NaN std where fewer than two did, ddof=1, and ULP-level
    negative variance clamped to zero before the sqrt.
    """
    import torch
    nan = torch.full_like(mean, float('nan'))
    ones = torch.ones_like(n)
    mu = torch.where(n > 0, mean, nan)
    var = torch.where(n > 1, m2 / torch.where(n > 1, n - 1, ones), nan)
    var = torch.clamp(var, min=0.0)
    std = torch.where(n > 1, torch.sqrt(var), nan)
    return mu.cpu().numpy(), std.cpu().numpy()


def gpu_perm_full(*, exp, base_seed: int, n_perm: int, q0, q1, children,
                  min_vox: int, perm_chunk: int = 8, device: str = 'cuda',
                  acc_dtype=np.float64, matmul_dtype=None,
                  force_path=None, center: bool = True,
                  scan_dtype=np.float64):
    """Compute the raw (n_perm, num_reg) inner-perm LLR draws on device.

    The materializing counterpart of gpu_perm, for tests and for callers
    that inspect individual draws. Peak memory carries the whole draws
    matrix, so prefer gpu_perm at production n_perm.

    Args match inner_perm.cpu_reliable_full plus perm_chunk, device,
    acc_dtype, matmul_dtype and force_path (see _dispatch_state).

    Returns:
        draws (np.array): (n_perm, num_reg) per-draw LLR, NaN where a
            region is below min_vox or not positive definite
    """
    import torch

    state, chunk_llr = _dispatch_state(
        exp=exp, q0=q0, q1=q1, children=children, min_vox=min_vox,
        device=device, acc_dtype=acc_dtype, matmul_dtype=matmul_dtype,
        force_path=force_path, center=center, scan_dtype=scan_dtype)
    perms = _build_perms(base_seed, n_perm, exp.y.shape[1])
    perms_t, perms_inv_t = _build_perm_tensors(perms, state['dev'])
    src = _chunk_perms_for_path(chunk_llr, perms_t, perms_inv_t)

    out = np.full((n_perm, state['num_reg']), np.nan, dtype=np.float64)
    for s in range(0, n_perm, perm_chunk):
        llr, valid = chunk_llr(src[s:s + perm_chunk], state)
        nan = torch.full_like(llr, float('nan'))
        out[s:s + perm_chunk] = torch.where(valid, llr, nan) \
                                     .to(torch.float64).cpu().numpy()
    return out


def gpu_perm(*, exp, base_seed: int, n_perm: int, q0, q1, children,
             min_vox: int, perm_chunk: int = 8, device: str = 'cuda',
             acc_dtype=np.float64, matmul_dtype=None, force_path=None,
             center: bool = True, scan_dtype=np.float64):
    """Compute inner-perm (mu, std) on device -- the production entry.

    A drop-in for inner_perm.cpu_perm: same keyword-only signature, same
    seed-to-draw mapping (draw i uses base_seed + i), same Chan-parallel
    moment reduction, so the two agree to float round-off. The
    (n_perm, num_reg) draws matrix is never materialized -- each chunk
    folds into device-side accumulators.

    Args:
        exp (Experiment): experiment to sample inner perms from
        base_seed (int): draw i uses RNG seed base_seed + i
        n_perm (int): number of inner FL draws
        q0 (np.array): (a0, num_img) nuisance subspace
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN
        perm_chunk (int): draws per device chunk. Sets the transient
            footprint; see plan_perm_chunk.
        device (str): torch device string
        acc_dtype: accumulation dtype, default float64 (module docstring)
        matmul_dtype: general-Q0 alpha-matmul dtype, default acc_dtype
        force_path (str): 'intercept' / 'general' to skip the dispatch

    Returns:
        mu (np.array): (num_reg,) inner-null mean per region
        std (np.array): (num_reg,) inner-null std per region
    """
    import torch

    state, chunk_llr = _dispatch_state(
        exp=exp, q0=q0, q1=q1, children=children, min_vox=min_vox,
        device=device, acc_dtype=acc_dtype, matmul_dtype=matmul_dtype,
        force_path=force_path, center=center, scan_dtype=scan_dtype)
    perms = _build_perms(base_seed, n_perm, exp.y.shape[1])
    perms_t, perms_inv_t = _build_perm_tensors(perms, state['dev'])
    src = _chunk_perms_for_path(chunk_llr, perms_t, perms_inv_t)

    num_reg = state['num_reg']
    dev = state['dev']
    n = torch.zeros(num_reg, dtype=torch.float64, device=dev)
    mean = torch.zeros(num_reg, dtype=torch.float64, device=dev)
    m2 = torch.zeros(num_reg, dtype=torch.float64, device=dev)

    for s in range(0, n_perm, perm_chunk):
        llr, valid = chunk_llr(src[s:s + perm_chunk], state)
        n, mean, m2 = _chan_combine(llr, valid, n, mean, m2)
    return _chan_finalize(n, mean, m2)
