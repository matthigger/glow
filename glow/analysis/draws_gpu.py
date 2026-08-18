"""GPU Freedman-Lane draw matrix (hoisted precompute).

The device counterpart of glow.analysis.draws.cpu_reliable: same
keyword-only signature, same seed-to-draw mapping, so the two can be
compared cell by cell (test_draws_gpu.py). Where that anchor walks every
region of every draw through iter_mancova, this module hoists every
permutation-invariant quantity into a one-time precompute and reduces
each draw to a small row-permutation, one cumsum-and-diff, and a
closed-form determinant.

ONE path serves any nuisance design. Everything is expressed in terms of
the per-voxel nuisance split, taken once in float64:

    s0 = Q0 y        (a0, b, num_vox)
    u  = y - Q0^T s0                       so Q0 u = 0
    T_v = sum_n u u^T                      (b, b, num_vox)

Because u already lies in the residual space, a Freedman-Lane draw
(Freedman & Lane 1983) collapses to a column gather on it: with
freed_lane = (I - Q0 Q0^T)[:, pi] + Q0 Q0^T applied in glow's row layout,
u @ freed_lane = u[:, pi] and (Q0^T s0) @ freed_lane = Q0^T s0. So

    y* = u[:, pi] + Q0^T s0,     alpha = Q01[:, pi^-1] u

with rho = alpha[:a0] and beta = alpha[a0:] -- no (num_img, num_img)
Freedman-Lane matrix, no per-draw gather of the data, and ONE permutation
direction for both blocks. For intercept-only nuisance rho is identically
zero, so the intercept-only fast path is what this reduces to rather than
a branch to maintain (the same treatment glow.graph.iter_llr_perm gives
its own rho / X_v terms). Output matches
glow.graph.compute_llr_batched on the permuted experiment.

Permutation convention. glow stores data in row layout, so
permute.get_freed_lane returns the column-layout transpose
(I - Q0 Q0^T)[:, perm] + Q0 Q0^T; glow's perm IS the textbook pi, not its
inverse. The hot loop gathers Q01 at pi^-1.

Three device-specific choices, each worth an order of magnitude:

  - the voxel axis stays LAST through the per-chunk loop; cumsum and
    index_select on a non-final axis are non-coalesced.
  - the (b, b) Gram products go through _gram's elementwise broadcast,
    never matmul or einsum -- cuBLAS's per-matrix overhead dominates on
    outputs this small. The crossover sits near b = 8.
  - region aggregation is cumsum-and-diff on the DFS pre-order axis (3
    kernel launches), not a layer-by-layer tree sweep.

Dtype policy. acc_dtype defaults to float32, which is safe only because
T = E + H is never formed as the difference of two DC-scale terms. Written
directly, T = yout - a_0 a_0^T / size cancels two terms of magnitude
num_img * size * mean(y)^2 down to the residual scatter, which for a
near-constant voxel on a DC offset falls below float32 eps: E becomes
rounding noise, the per-region std collapses, and z explodes into the
max-z null (test_draws_hcp.py). Instead T is assembled as

    T = W_r + S_r
    W_r = reg_sum(T_v - sum_a rho_a rho_a^T)
    S_r = reg_sum(sum_a s* s*^T) - gram(reg_sum s*) / size,  s* = rho + s0

an identity, not an approximation -- the cross terms vanish because
Q0 u = 0. Of the four terms S_r expands into over s* = rho + s0, the
nuisance-coefficient scatter Scat(s0, s0) is the only one still carrying
the DC offset and does not depend on the permutation, so it is hoisted
with reg_sum(T_v) into a per-tree T_inv at scan_dtype; the rho-rho terms
cancel against W_r. What is left per draw is the rho / s0 cross term,
which cancels only to the spatial spread of s0. So acc_dtype carries the
whole per-draw loop and scan_dtype touches only prep; scan_dtype=float32
reproduces the collapse, which is how the regression test pins it.
"""
import numpy as np

import glow.graph
from glow.experiment import permute
from ._base import Z_STD_FLOOR
from .draws import DrawSummary


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


def unavailable_reason() -> str:
    """Say why is_available() is False, in one clause.

    The three ways a device goes missing want different fixes: no torch,
    a CPU-only torch build (pip's default wheel -- torch.version.cuda is
    None however many cards are in the machine), or a CUDA build that
    cannot see one. The middle case is the quiet one, since nvidia-smi
    still lists the card, so the message names the wheel.

    Printed by fit(verbose=True) through _fit_gpu.describe_backend,
    because gpu='auto' falls back to the CPU without a word.

    Returns:
        reason (str): the clause, or 'a device is visible' when one is.
    """
    try:
        import torch
    except ImportError:
        return 'torch is not installed'
    if getattr(torch.version, 'cuda', None) is None:
        return (f'torch {torch.__version__} is a CPU-only build '
                f'(torch.version.cuda is None)')
    try:
        if torch.cuda.is_available():
            return 'a device is visible'
    except Exception as err:
        return f'torch.cuda.is_available() raised {err!r}'
    return (f'torch {torch.__version__} is a CUDA build, but '
            f'torch.cuda.is_available() is False')


def _torch_dtype(acc_dtype):
    """Map a numpy accumulation dtype to (np.dtype, torch dtype)."""
    import torch
    np_dtype = np.dtype(acc_dtype)
    if np_dtype == np.float32:
        return np_dtype, torch.float32
    if np_dtype == np.float64:
        return np_dtype, torch.float64
    raise ValueError(f'acc_dtype must be float32 or float64, got {np_dtype}')


def _gram_a(x):
    """Contract a (Pc, a, b, M) stack over its a axis into (Pc, b, b, M).

    Args:
        x (torch.Tensor): (Pc, a, b, M) -- M is the voxel or region axis,
            kept last

    Returns:
        g (torch.Tensor): (Pc, b, b, M) with
            g[p, i, j, m] = sum_a x[p, a, i, m] x[p, a, j, m]

    Implemented as unsqueeze-multiply-sum, never einsum or matmul: those
    dispatch to batched cuBLAS GEMM, whose per-matrix overhead dominates at
    these tiny (b, b) outputs (see the module docstring).
    """
    return (x.unsqueeze(3) * x.unsqueeze(2)).sum(dim=1)


def _gram_a_cross(x, y):
    """Contract (Pc, a, b, M) against (a, c, M) over a, giving (Pc, b, c, M).

    The asymmetric companion to _gram_a, for the one per-draw term that
    pairs a permutation-dependent factor with a permutation-invariant one.
    Same broadcast-reduce reason for avoiding einsum / matmul.

    Args:
        x (torch.Tensor): (Pc, a, b, M)
        y (torch.Tensor): (a, c, M)

    Returns:
        g (torch.Tensor): (Pc, b, c, M) with
            g[p, i, j, m] = sum_a x[p, a, i, m] y[a, j, m]
    """
    return (x.unsqueeze(3) * y[None, :, None, :, :]).sum(dim=1)


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
# Phase 1 precompute + per-chunk hot loop (one path, any nuisance)

def prep_shared(exp, *, q0, q1, device: str = 'cuda',
                acc_dtype=np.float32, scan_dtype=np.float64):
    """Build the per-FIT device state: independent of tree and outer perm.

    Splits y once, in float64, into the nuisance coefficients and the
    per-voxel residual:

        s0 = Q0 y                (a0, b, num_vox)
        u  = y - Q0^T s0         (b, num_img, num_vox),   Q0 u = 0
        T_v = sum_n u u^T        (b, b, num_vox)

    A Freedman-Lane draw gathers u along the image axis, leaves the
    nuisance-fitted part alone, and cannot change T_v at all (a sum over
    that same axis), so all three are computed once per fit -- the float64
    split, the transfer and the T_v contraction stay outside the loop.

    Args:
        exp (Experiment): the unpermuted experiment
        q0 (np.array): (a0, num_img) nuisance subspace from decompose
        q1 (np.array): (a1, num_img) interest subspace
        device (str): torch device string
        acc_dtype: hot-loop dtype (module docstring)
        scan_dtype: dtype for the DC-carrying prep scans

    Returns:
        shared (dict): {dev, torch_dtype, scan_torch, b, num_img, num_vox,
            a0, a1, ak, U0 (b, num_img, num_vox) residual in ORIGINAL voxel
            order, s0_0 (a0, b, num_vox), T_v (b, b, num_vox),
            Q0 (a0, num_img), Q01 (a0 + a1, num_img)}
    """
    import torch

    y = exp.y
    b, num_img, num_vox = y.shape
    np_dtype, torch_dtype = _torch_dtype(acc_dtype)
    _, scan_torch = _torch_dtype(scan_dtype)
    dev = torch.device(device)

    # The one place the DC offset is removed, done in float64 before the
    # narrowing cast: u for a near-constant voxel is ~1e-4 of y, so
    # centring in float32 would keep only ~3 digits of it.
    y64 = np.ascontiguousarray(y).astype(np.float64, copy=False)
    q064 = np.ascontiguousarray(q0).astype(np.float64, copy=False)
    s0_np = np.einsum('an,bnv->abv', q064, y64, optimize=True)
    u_np = y64 - np.einsum('an,abv->bnv', q064, s0_np, optimize=True)

    u0 = torch.from_numpy(
        np.ascontiguousarray(u_np).astype(np_dtype, copy=False)).to(dev)
    s0_0 = torch.from_numpy(np.ascontiguousarray(s0_np)).to(dev).to(scan_torch)
    del y64, q064, s0_np, u_np

    return dict(
        dev=dev, torch_dtype=torch_dtype, scan_torch=scan_torch,
        b=b, num_img=num_img, num_vox=num_vox,
        a0=int(q0.shape[0]), a1=int(q1.shape[0]),
        ak=int(q0.shape[0]) + int(q1.shape[0]),
        U0=u0, s0_0=s0_0,
        T_v=torch.einsum('bnv,cnv->bcv', u0, u0).to(scan_torch),
        Q0=torch.from_numpy(
            np.ascontiguousarray(q0).astype(np_dtype, copy=False)).to(dev),
        Q01=torch.from_numpy(
            np.ascontiguousarray(np.vstack([q0, q1])).astype(np_dtype,
                                                             copy=False)
        ).to(dev))


def prep_tree(shared, *, children, min_vox: int):
    """Derive the per-tree state from the shared per-fit state.

    Everything here is a gather of shared arrays or a region scan on the
    tree -- no host work and no transfer, so a new tree does not repeat
    prep_shared's float64 split or its device copy.

    Args:
        shared (dict): prep_shared output
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are inactive

    Returns:
        state (dict): consumed by _chunk_llr -- {dev, torch_dtype,
            scan_torch, b, num_img, num_vox, a0, a1, ak, num_reg,
            U (DFS-ordered residual), Q01, s0, s0_r, T_inv, region_l_t,
            region_h_t, size_f, inv_size, active}
    """
    import torch

    dev = shared['dev']
    scan_torch = shared['scan_torch']
    torch_dtype = shared['torch_dtype']
    num_vox = shared['num_vox']

    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=num_vox)
    region_l_t = torch.from_numpy(region_l.astype(np.int64)).to(dev)
    region_h_t = torch.from_numpy(region_h.astype(np.int64)).to(dev)
    leaf_ord_t = torch.from_numpy(leaf_ord.astype(np.int64)).to(dev)
    num_reg = int(region_l.shape[0])

    u_dfs = shared['U0'].index_select(2, leaf_ord_t).contiguous()
    s0 = shared['s0_0'].index_select(2, leaf_ord_t).contiguous()
    t_v = shared['T_v'].index_select(2, leaf_ord_t).contiguous()

    size_d = region_h_t - region_l_t
    inv_size_scan = torch.where(
        size_d > 0, 1.0 / size_d.to(scan_torch),
        torch.zeros(num_reg, dtype=scan_torch, device=dev))

    # T_inv = reg_sum(T_v) + Scat(s0, s0): the whole permutation-invariant
    # part of T = E + H, and the whole of its DC-carrying cancellation.
    # Both region scans run once per tree at scan_dtype, so the per-draw
    # loop never touches float64 (see the module docstring).
    t_v_r = _reg_sum_cumsum(t_v, -1, region_l_t, region_h_t)
    s0_sq_r = _reg_sum_cumsum(
        (s0.unsqueeze(2) * s0.unsqueeze(1)).sum(dim=0), -1,
        region_l_t, region_h_t)
    s0_r = _reg_sum_cumsum(s0, -1, region_l_t, region_h_t)
    t_inv = (t_v_r + s0_sq_r
             - (s0_r.unsqueeze(1) * s0_r.unsqueeze(2)).sum(dim=0)
               * inv_size_scan[None, None, :])
    del t_v, t_v_r, s0_sq_r

    return dict(
        dev=dev, torch_dtype=torch_dtype, scan_torch=scan_torch,
        b=shared['b'], num_img=shared['num_img'], num_vox=num_vox,
        a0=shared['a0'], a1=shared['a1'], ak=shared['ak'], num_reg=num_reg,
        U=u_dfs, Q01=shared['Q01'],
        s0=s0.to(torch_dtype), s0_r=s0_r.to(torch_dtype),
        T_inv=t_inv.to(torch_dtype),
        region_l_t=region_l_t, region_h_t=region_h_t,
        size_f=size_d.to(torch_dtype),
        inv_size=inv_size_scan.to(torch_dtype),
        active=size_d >= min_vox)


def _chunk_llr(chunk_inv_t, state):
    """Compute one chunk of draws.

    A Freedman-Lane draw acts on the residual as a plain column gather:
    with u already satisfying Q0 u = 0, y* = u[:, perm] + Q0^T s0, so both
    projections read one column-permuted basis,

        alpha = Q01[:, perm^-1] u        (Pc, a0 + a1, b, num_vox)

    with rho = alpha[:a0] and beta = alpha[a0:]. Hence no (num_img,
    num_img) Freedman-Lane matrix, no per-draw gather of the data, and one
    permutation direction for both blocks.

    T = E + H is then assembled so that the permutation-invariant part --
    which is also the whole of the DC-carrying cancellation -- is hoisted
    into T_inv at prep time (see the module docstring):

        T = T_inv + (P_rs + P_rs^T) - (G(R,S) + G(R,S)^T) / size
                  - G(R,R) / size

        P_rs = reg_sum(sum_a rho_a s0_a^T)      per draw, per voxel
        R    = reg_sum(rho),  S = reg_sum(s0)   S hoisted
        G(X,Y)[b,c] = sum_a X[a,b] Y[a,c]       region-space only

    The rho-rho terms cancel between the two brackets of the underlying
    W_r + S_r split, which is why only the rho / s0 cross term needs a
    per-draw per-voxel scan. At rho = 0 (intercept-only nuisance) every
    term but T_inv vanishes in value, giving the fast path's answer without
    a second code path -- though the terms are still computed.

    Args:
        chunk_inv_t (torch.Tensor): (Pc, num_img) int64 inverse
            permutations pi^-1
        state (dict): _prep_state output

    Returns:
        llr (torch.Tensor): (Pc, num_reg) per-region LLR
        valid (torch.Tensor): (Pc, num_reg) bool, False where the region is
            inactive or T / E is not positive definite
    """
    import torch

    u_dfs = state['U']
    a0 = state['a0']
    region_l_t = state['region_l_t']
    region_h_t = state['region_h_t']
    size_f = state['size_f']
    inv = state['inv_size'][None, None, None, :]
    active = state['active']

    # V stays the last axis throughout: cumsum / index_select on an inner
    # axis is non-coalesced (module docstring).
    q01_perm = state['Q01'][:, chunk_inv_t].permute(1, 0, 2).contiguous()
    alpha = torch.einsum('pkn,bnv->pkbv', q01_perm, u_dfs)
    rho = alpha[:, :a0]
    beta = alpha[:, a0:]

    # The only per-draw per-voxel scan that T needs: the rho / s0 cross
    # term. Its rho-rho counterpart cancelled against W_r, and everything
    # else is either hoisted (T_inv) or region-space.
    p_rs = _reg_sum_cumsum(_gram_a_cross(rho, state['s0']), -1,
                           region_l_t, region_h_t)
    r_r = _reg_sum_cumsum(rho, -1, region_l_t, region_h_t)
    g_rs = _gram_a_cross(r_r, state['s0_r'])
    t = (state['T_inv'][None]
         + (p_rs + p_rs.transpose(1, 2))
         - (g_rs + g_rs.transpose(1, 2)) * inv
         - _gram_a(r_r) * inv)
    del p_rs, g_rs, r_r

    beta_r = _reg_sum_cumsum(beta, -1, region_l_t, region_h_t)
    h = _gram_a(beta_r) * inv

    t = t.permute(0, 3, 1, 2)
    e = t - h.permute(0, 3, 1, 2)

    sign_t, log_t = _slogdet_batched(t)
    sign_e, log_e = _slogdet_batched(e)
    valid = active[None] & (sign_t > 0) & (sign_e > 0)
    llr = 0.5 * size_f[None] * (log_t - log_e)
    return llr, valid


# ---------------------------------------------------------------------------
# Dispatch and public API

def _build_perms(base_seed: int, n_perm: int, num_img: int):
    """Build the (n_perm, num_img) FL index array; draw i uses base_seed + i.

    Routes through permute._perm_indices, the sole source of truth for the
    mapping, so a given (base_seed, n_perm) yields the same draws as the
    CPU anchor -- including the identity at seed 0.
    """
    perms = np.empty((n_perm, num_img), dtype=np.int64)
    for i in range(n_perm):
        perms[i] = permute._perm_indices(base_seed + i, num_img)
    return perms


def _build_perm_inv_tensor(perms, device):
    """Upload the inverse permutations pi^-1 the hot loop gathers Q01 with.

    argsort runs once on the host so the per-chunk loop never pays it.
    """
    import torch
    return torch.from_numpy(
        np.argsort(perms, axis=1).astype(np.int64)).to(device)


def _nan_where_invalid(x, valid):
    """Return x in float64 with the invalid cells replaced by NaN."""
    import torch
    x64 = x.to(torch.float64)
    return torch.where(valid, x64, torch.full_like(x64, float('nan')))


def gpu_perm(*, exp, base_seed: int, n_perm: int, q0, q1, children,
             min_vox: int, perm_chunk: int = 16, device: str = 'cuda',
             acc_dtype=np.float32, scan_dtype=np.float64):
    """Compute the (n_perm, num_reg) Freedman-Lane LLR draws on device.

    A drop-in for glow.analysis.draws.cpu_reliable: same keyword-only
    signature, same seed-to-draw mapping (draw i uses base_seed + i, and
    seed 0 is the identity), same NaN convention, so the two agree cell by
    cell to float round-off.

    Peak host memory carries the whole matrix, (n_perm, num_reg) float64.
    Callers needing only the column moments, the observed row and the row
    maxima should stream instead (gpu_summarize).

    Args:
        exp (Experiment): experiment to sample permutations from
        base_seed (int): draw i uses RNG seed base_seed + i
        n_perm (int): number of FL draws, counting the observed
        q0 (np.array): (a0, num_img) nuisance subspace
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN
        perm_chunk (int): draws per device chunk. Both a throughput and a
            memory knob: the working set scales as
            perm_chunk * a0 * b^2 * num_vox and is fastest while it stays
            L2-resident. _fit_gpu.resolve_perm_chunk sizes it from free
            device memory, since the default overruns a small card at
            full-brain num_vox.
        device (str): torch device string
        acc_dtype: hot-loop dtype, default float32 (module docstring). A
            GLOW fit overrides it to float64, reproducing a CPU fit's
            p-values exactly (see _fit_gpu).
        scan_dtype: dtype for the DC-carrying prep scans, default float64 --
            the one group still carrying the DC offset

    Returns:
        draws (np.array): (n_perm, num_reg) per-draw LLR, NaN where a
            region is below min_vox or not positive definite
    """
    state = prep_tree(
        prep_shared(exp, q0=q0, q1=q1, device=device, acc_dtype=acc_dtype,
                    scan_dtype=scan_dtype),
        children=children, min_vox=min_vox)
    perms = _build_perms(base_seed, n_perm, exp.y.shape[1])
    src = _build_perm_inv_tensor(perms, state['dev'])

    out = np.full((n_perm, state['num_reg']), np.nan, dtype=np.float64)
    for s in range(0, n_perm, perm_chunk):
        llr, valid = _chunk_llr(src[s:s + perm_chunk], state)
        out[s:s + perm_chunk] = _nan_where_invalid(llr, valid).cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Streaming reduction: two passes, never the matrix

def _chan_combine(llr, valid, n, mean, m2):
    """Fold one (Pc, num_reg) draw-chunk into running per-region moments.

    One step of Chan's parallel-combine rule (Chan, Golub & LeVeque 1979),
    with invalid cells excluded from the count, giving nanmean /
    nanstd(ddof=1) semantics once finalized. Chan rather than a naive
    sum-of-squares pass because var << mean^2 here (LLR carries a
    0.5 * size prefactor), where the one-pass formula loses the variance to
    cancellation.

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


def _chan_moments(n, mean, m2):
    """Reduce running (n, mean, m2) to on-device (mu, std).

    Matches np.nanmean / np.nanstd(ddof=1) as z_score_stat calls them: NaN
    mu where no valid sample accumulated, NaN std where fewer than two did,
    negative variance clamped to zero before the sqrt. Left on the device
    for the second pass.

    Args:
        n (torch.Tensor): (num_reg,) valid-sample count
        mean (torch.Tensor): (num_reg,) running mean
        m2 (torch.Tensor): (num_reg,) running sum of squared deviations

    Returns:
        mu, std (torch.Tensor): (num_reg,) each, on n's device
    """
    import torch
    nan = torch.full_like(mean, float('nan'))
    ones = torch.ones_like(n)
    mu = torch.where(n > 0, mean, nan)
    var = torch.where(n > 1, m2 / torch.where(n > 1, n - 1, ones), nan)
    return mu, torch.where(n > 1, torch.sqrt(torch.clamp(var, min=0.0)), nan)


def _nanmax_rows(x):
    """Reduce (Pc, M) to (Pc,) row maxima, ignoring NaN.

    torch has no nanmax, so NaN is pushed to -inf for the reduction and the
    all-NaN rows are restored afterwards -- np.nanmax's convention, which
    fwer.from_max depends on: a NaN draw sits out of the comparison, where a
    -inf would join it and move every p-value.

    Args:
        x (torch.Tensor): (Pc, M) values over the comparison set

    Returns:
        row_max (torch.Tensor): (Pc,) max per row, NaN for an all-NaN row
    """
    import torch
    bad = torch.isnan(x)
    filled = torch.where(bad, torch.full_like(x, float('-inf')), x)
    row_max = filled.max(dim=1).values
    return torch.where(bad.all(dim=1),
                       torch.full_like(row_max, float('nan')), row_max)


def gpu_summarize(*, exp, base_seed: int, n_perm: int, q0, q1, children,
                  min_vox: int, reg_active=None, perm_chunk: int = 16,
                  device: str = 'cuda', acc_dtype=np.float32,
                  scan_dtype=np.float64):
    """Summarize the draw matrix on device without ever forming it.

    Equivalent to draws.summarize_draws(gpu_perm(...)) at O(num_reg +
    n_perm) host memory instead of O(n_perm * num_reg).

    Two passes, because standardizing needs moments the first pass has not
    finished. Pass one folds each chunk into Chan accumulators for
    (mu, std); pass two re-draws the same chunks, standardizes them and
    keeps the row maxima plus row 0. Re-drawing is exact -- a chunk is a
    deterministic function of its permutation indices and the prep state --
    and doubles the LLR work to buy the memory back. Prep is paid once.

    Args:
        exp (Experiment): experiment to sample permutations from
        base_seed (int): draw i uses RNG seed base_seed + i; 0 puts the
            observed draw in row 0
        n_perm (int): number of FL draws, counting the observed
        q0 (np.array): (a0, num_img) nuisance subspace
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN
        reg_active (np.array): (num_reg,) boolean comparison set. None
            takes size >= min_vox, the set prep_tree already derived.
        perm_chunk (int): draws per device chunk; see gpu_perm
        device (str): torch device string
        acc_dtype: hot-loop dtype; the reduction itself is float64 either
            way, matching the CPU path's numpy
        scan_dtype: dtype for the DC-carrying prep scans

    Returns:
        DrawSummary: see glow.analysis.draws.DrawSummary
    """
    import torch

    state = prep_tree(
        prep_shared(exp, q0=q0, q1=q1, device=device, acc_dtype=acc_dtype,
                    scan_dtype=scan_dtype),
        children=children, min_vox=min_vox)
    dev = state['dev']
    num_reg = state['num_reg']
    src = _build_perm_inv_tensor(
        _build_perms(base_seed, n_perm, exp.y.shape[1]), dev)

    active = (state['active'] if reg_active is None
              else torch.from_numpy(
                  np.ascontiguousarray(reg_active, dtype=bool)).to(dev))

    # Pass 1: (mu, std). Row 0's raw LLR is unstandardized, so it owes
    # nothing to the moments and is read off here.
    n = torch.zeros(num_reg, dtype=torch.float64, device=dev)
    mean = torch.zeros_like(n)
    m2 = torch.zeros_like(n)
    llr_obs = None
    for s in range(0, n_perm, perm_chunk):
        llr, valid = _chunk_llr(src[s:s + perm_chunk], state)
        if llr_obs is None:
            llr_obs = _nan_where_invalid(llr[:1], valid[:1])[0]
        n, mean, m2 = _chan_combine(llr, valid, n, mean, m2)
    mu, std = _chan_moments(n, mean, m2)

    # Pass 2: z against those moments, then the two reductions the fit
    # keeps. Z_STD_FLOOR is shared with z_score_stat, so a degenerate
    # column is divided by the same 1.0 on both paths.
    denom = torch.where(std > Z_STD_FLOOR, std, torch.ones_like(std))
    max_stat = np.full(n_perm, np.nan, dtype=np.float64)
    any_active = bool(active.any())
    z_obs = None
    for s in range(0, n_perm, perm_chunk):
        llr, valid = _chunk_llr(src[s:s + perm_chunk], state)
        z = _nan_where_invalid(
            (llr.to(torch.float64) - mu[None]) / denom[None], valid)
        if z_obs is None:
            z_obs = z[0].clone()
        if any_active:
            max_stat[s:s + perm_chunk] = \
                _nanmax_rows(z[:, active]).cpu().numpy()

    return DrawSummary(llr=llr_obs.cpu().numpy(), mu=mu.cpu().numpy(),
                       std=std.cpu().numpy(), z_obs=z_obs.cpu().numpy(),
                       max_stat=max_stat)

