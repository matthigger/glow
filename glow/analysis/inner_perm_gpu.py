"""GPU inner Freedman-Lane permutation backend (hoisted precompute).

A device-side counterpart to inner_perm.cpu_perm with the same
keyword-only signature, so it drops into AnalysisGLOW.run_inner_perm.
Where cpu_perm rides glow.graph.iter_llr_perm (per-perm GEMM plus
cumsum-and-diff over the DFS pre-order voxel axis), this module hoists
every permutation-invariant quantity into a one-time precompute and
reduces each inner draw to a small row-permutation, one cumsum-and-diff,
and a closed-form determinant.

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

Dtype policy. acc_dtype defaults to float32, which on consumer Ada parts
is worth 2.4-5.4x over float64 (fp64 runs at 1/64 the FLOP rate). That is
only safe because T = E + H is never formed as the difference of two
DC-scale terms. Written directly, T = yout - a_0 a_0^T / size cancels two
terms of magnitude num_img * size * mean(y)^2 down to the residual
scatter; for a near-constant voxel on a large DC offset (HCP background:
mean -0.76, across-image std 5e-4) that ratio is ~7e-8, below float32 eps,
and the result is the a55e7237 collapse -- E becomes rounding noise, the
inner-null std collapses, and the standardized z explodes into the max-z
null (see test_inner_perm_hcp.py). Instead T is assembled as

    T = W_r + S_r
    W_r = reg_sum(T_v - sum_a rho_a rho_a^T)
    S_r = reg_sum(sum_a s* s*^T) - gram(reg_sum s*) / size,  s* = rho + s0

an identity, not an approximation: the cross terms vanish because Q0 u = 0.
Expanding S_r over s* = rho + s0 then splits it four ways, and two of
those matter:

  - Scat(s0, s0), the within-region spatial scatter of the nuisance
    coefficients, is the ONLY term still carrying the DC offset -- and it
    does not depend on the permutation at all. It is therefore folded,
    with reg_sum(T_v), into a single hoisted T_inv computed once per tree
    at scan_dtype (float64 by default).
  - the rho-rho terms cancel between W_r and S_r, collapsing to a
    region-space outer product with no per-voxel scan.

What is left per draw is the rho / s0 cross term, whose two halves cancel
only to the spatial spread of s0 relative to its DC level -- a benign
ratio, and it rides a term that is itself a small fraction of T. So the
per-draw loop needs no float64 at all: acc_dtype carries everything, and
scan_dtype touches only prep. Setting scan_dtype=float32 reproduces the
collapse, which is how the regression test pins it.
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


def _gram_a(x):
    """Contract a (Pc, a, b, M) stack over its a axis into (Pc, b, b, M).

    Args:
        x (torch.Tensor): (Pc, a, b, M) -- M is the voxel or region axis,
            kept last

    Returns:
        g (torch.Tensor): (Pc, b, b, M) with
            g[p, i, j, m] = sum_a x[p, a, i, m] x[p, a, j, m]

    Implemented as unsqueeze-multiply-sum, never einsum or matmul: those
    dispatch to batched cuBLAS GEMM, which is ~500x slower than the
    elementwise reduce at these batch counts and tiny (b, b) outputs (see
    the module docstring).
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

def _prep_state(*, y, q0, q1, children, min_vox, device, acc_dtype,
                scan_dtype=np.float64):
    """Build the permutation-invariant device state.

    Splits y once, in float64, into the nuisance coefficients and the
    per-voxel residual:

        s0 = Q0 y                (a0, b, num_vox)
        u  = y - Q0^T s0         (b, num_img, num_vox),   Q0 u = 0

    and forms the residual's per-voxel Gram T_v = sum_n u u^T. Everything
    the hot loop needs is then a contraction against u, never against y --
    which is what makes float32 viable (see the module docstring) and what
    lets a caller hold u once for a whole fit.

    Uploads and reorders to DFS pre-order on device: a host-side
    y[:, :, leaf_ord] gather costs ~14ms at num_vox=25k and blocks the CPU
    that could be building the next tree.

    Args:
        y (np.array): (b, num_img, num_vox) imaging features, unpermuted
        q0 (np.array): (a0, num_img) nuisance subspace from decompose
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are inactive
        device (str): torch device string
        acc_dtype: hot-loop dtype (module docstring)
        scan_dtype: dtype for the region scans of s_star, the one group
            that still carries the DC offset (module docstring)

    Returns:
        state (dict):
            {dev, torch_dtype, scan_torch, b, num_img, num_vox, a0, a1, ak,
             num_reg, U (b, num_img, num_vox) DFS-ordered residual,
             Q01 (a0 + a1, num_img), s0 (a0, b, num_vox) DFS-ordered,
             T_v (b, b, num_vox), region_l_t, region_h_t, size_f,
             inv_size_scan, active}
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
    leaf_ord_t = torch.from_numpy(leaf_ord.astype(np.int64)).to(dev)
    num_reg = int(region_l.shape[0])

    # The one place the DC offset is removed, done in float64 before the
    # narrowing cast: u for a near-constant voxel is ~1e-4 of y, so
    # centring in float32 would keep only ~3 digits of it.
    y64 = np.ascontiguousarray(y).astype(np.float64, copy=False)
    q064 = np.ascontiguousarray(q0).astype(np.float64, copy=False)
    s0_np = np.einsum('an,bnv->abv', q064, y64, optimize=True)
    u_np = y64 - np.einsum('an,abv->bnv', q064, s0_np, optimize=True)

    u_dfs = torch.from_numpy(
        np.ascontiguousarray(u_np).astype(np_dtype, copy=False)
    ).to(dev).index_select(2, leaf_ord_t).contiguous()
    s0 = torch.from_numpy(
        np.ascontiguousarray(s0_np)
    ).to(dev).to(scan_torch).index_select(2, leaf_ord_t).contiguous()
    del y64, q064, s0_np, u_np

    q01 = torch.from_numpy(
        np.ascontiguousarray(np.vstack([q0, q1])).astype(np_dtype, copy=False)
    ).to(dev)

    size_d = region_h_t - region_l_t
    inv_size_scan = torch.where(
        size_d > 0, 1.0 / size_d.to(scan_torch),
        torch.zeros(num_reg, dtype=scan_torch, device=dev))

    # T_inv = reg_sum(T_v) + Scat(s0, s0): the whole permutation-invariant
    # part of T = E + H, and the whole of its DC-carrying cancellation.
    # Both region scans run once per tree at scan_dtype, so the per-draw
    # loop never touches float64 (see the module docstring).
    t_v = torch.einsum('bnv,cnv->bcv', u_dfs, u_dfs).to(scan_torch)
    t_v_r = _reg_sum_cumsum(t_v, -1, region_l_t, region_h_t)
    del t_v
    s0_sq_r = _reg_sum_cumsum(
        (s0.unsqueeze(2) * s0.unsqueeze(1)).sum(dim=0), -1,
        region_l_t, region_h_t)
    s0_r = _reg_sum_cumsum(s0, -1, region_l_t, region_h_t)
    t_inv = (t_v_r + s0_sq_r
             - (s0_r.unsqueeze(1) * s0_r.unsqueeze(2)).sum(dim=0)
               * inv_size_scan[None, None, :])
    del t_v_r, s0_sq_r

    return dict(
        dev=dev, torch_dtype=torch_dtype, scan_torch=scan_torch,
        b=b, num_img=num_img, num_vox=num_vox,
        a0=int(q0.shape[0]), a1=int(q1.shape[0]),
        ak=int(q0.shape[0]) + int(q1.shape[0]), num_reg=num_reg,
        U=u_dfs, Q01=q01,
        s0=s0.to(torch_dtype), s0_r=s0_r.to(torch_dtype),
        T_inv=t_inv.to(torch_dtype),
        region_l_t=region_l_t, region_h_t=region_h_t,
        size_f=size_d.to(torch_dtype),
        inv_size=inv_size_scan.to(torch_dtype),
        active=size_d >= min_vox)


def _chunk_llr(chunk_inv_t, state):
    """Compute one chunk of inner draws.

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
    term but T_inv vanishes, so the result is the hoisted T_u a
    nuisance-specific fast path would compute -- reached without a second
    code path, as glow.graph.iter_llr_perm reaches it with its own rho /
    X_v terms. Note the WORK does not vanish there, only its value: the
    terms are still computed, which is the price of one path.

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
    # axis is ~16x slower (module docstring).
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

    Matches inner_perm.cpu_perm's construction exactly, so a given
    (base_seed, n_perm) yields the same draws on both backends.
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
                  acc_dtype=np.float32, scan_dtype=np.float64):
    """Compute the raw (n_perm, num_reg) inner-perm LLR draws on device.

    The materializing counterpart of gpu_perm, for tests and for callers
    that inspect individual draws. Peak memory carries the whole draws
    matrix, so prefer gpu_perm at production n_perm.

    Args match inner_perm.cpu_reliable_full plus perm_chunk, device,
    acc_dtype and scan_dtype (see gpu_perm).

    Returns:
        draws (np.array): (n_perm, num_reg) per-draw LLR, NaN where a
            region is below min_vox or not positive definite
    """
    import torch

    state = _prep_state(
        y=exp.y, q0=q0, q1=q1, children=children, min_vox=min_vox,
        device=device, acc_dtype=acc_dtype, scan_dtype=scan_dtype)
    perms = _build_perms(base_seed, n_perm, exp.y.shape[1])
    src = _build_perm_inv_tensor(perms, state['dev'])

    out = np.full((n_perm, state['num_reg']), np.nan, dtype=np.float64)
    for s in range(0, n_perm, perm_chunk):
        llr, valid = _chunk_llr(src[s:s + perm_chunk], state)
        nan = torch.full_like(llr, float('nan'))
        out[s:s + perm_chunk] = torch.where(valid, llr, nan) \
                                     .to(torch.float64).cpu().numpy()
    return out


def gpu_perm(*, exp, base_seed: int, n_perm: int, q0, q1, children,
             min_vox: int, perm_chunk: int = 8, device: str = 'cuda',
             acc_dtype=np.float32, scan_dtype=np.float64):
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
        acc_dtype: hot-loop dtype, default float32 (module docstring)
        scan_dtype: dtype for the s_star region scans, default float64 --
            the one group still carrying the DC offset

    Returns:
        mu (np.array): (num_reg,) inner-null mean per region
        std (np.array): (num_reg,) inner-null std per region
    """
    import torch

    state = _prep_state(
        y=exp.y, q0=q0, q1=q1, children=children, min_vox=min_vox,
        device=device, acc_dtype=acc_dtype, scan_dtype=scan_dtype)
    perms = _build_perms(base_seed, n_perm, exp.y.shape[1])
    src = _build_perm_inv_tensor(perms, state['dev'])

    num_reg = state['num_reg']
    dev = state['dev']
    n = torch.zeros(num_reg, dtype=torch.float64, device=dev)
    mean = torch.zeros(num_reg, dtype=torch.float64, device=dev)
    m2 = torch.zeros(num_reg, dtype=torch.float64, device=dev)

    for s in range(0, n_perm, perm_chunk):
        llr, valid = _chunk_llr(src[s:s + perm_chunk], state)
        n, mean, m2 = _chan_combine(llr, valid, n, mean, m2)
    return _chan_finalize(n, mean, m2)
