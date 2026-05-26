"""GPU implementation of the inner Freedman-Lane permutation loop.

For one outer permutation: given the Ward dendrogram (built on host),
run all ``n_perm_inner`` inner perms on GPU and return per-region
Welford moments ``(mu, M2, n_per_reg)``.  Replaces the CPU race in
``AnalysisGLOW._process_permutation``.

Two implementations live here, dispatched from ``run_inner_perms`` by
``is_intercept_only_nuisance``:

  - **Intercept-only path** (``_prep_intercept`` + captured replay).
    Q0 is the span of the all-ones vector, so it commutes with every
    permutation: ``yout`` and ``log|T_u|`` are FL-invariant and hoist
    out of the inner loop entirely.  Each inner perm reduces to a
    row permutation of Q1 against the constant per-region precomputes.

  - **General-Q0 path** (``_prep_general`` + captured replay).
    Q0 has additional nuisance lanes so ``F = freed_lane @ freed_lane.T``
    is a rank-2a perturbation of identity.  ``yout_perm`` and
    ``a0_perm`` vary per perm but expand as ``α + β``, where ``β`` is
    perm-invariant and ``α`` is one (N, b, a+k) matmul per batch.

Both paths share the sweep primitive (``sweep``), the welford
reduction (``welford_reduce``), and the captured-graph replay
discipline.  The general path additionally needs Chan's parallel
Welford merge (``_chan_welford_combine``) because its per-batch chain
typically replays multiple times per outer perm (one capture per
``batch_size`` ≤ ``n_perm_inner``).

End-to-end at paper config (b=2, k=2, V=25k, N=100, n_perm_inner=250)
on an RTX 4060: intercept-only ~10 ms / outer perm; general ~60 ms
(both vs CPU race at ~250 ms).

Naming: ``sweep`` is the GPU equivalent of
``glow.graph.compute_phase1``'s bottom-up tree walk.  Where
``compute_phase1`` walks a giant ``(R, b, N)`` ysum tensor, here we
sweep small per-voxel projections through the same tree.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import glow.graph
from glow.analysis.mancova import is_intercept_only_nuisance


def is_available() -> bool:
    """True iff PyTorch is importable and a CUDA device is visible.

    Cheap import probe — no GPU work, no error raised when torch is
    missing.  Used by ``AnalysisGLOW`` to resolve ``use_gpu=None``.
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# The rest of this module unconditionally imports torch — only do so
# once a caller has asked for the GPU path.  Defer the import to
# function bodies so `import glow.analysis.inner_perm_gpu` is cheap
# from CPU-only callers (e.g. AWS workers).
def _torch():
    import torch
    return torch


# ---------------------------------------------------------------------------
# Sweep primitive — shared by both paths

@dataclass
class SweepPlan:
    """Pre-computed layer buckets for a single dendrogram.

    Same dendrogram drives every sweep within an outer perm, so we
    build this once and reuse across the precompute sweeps and the
    per-inner-perm sweep.

    Attributes:
        num_vox:  V — voxel (leaf) count.
        num_reg:  R = 2V - 1 — total regions.
        layers:   list of (nodes_t, c0_t, c1_t) triples, one per
                  internal tree depth in ascending order.  ``nodes_t``
                  holds region indices (R-indexed, i.e. V+offset);
                  ``c0_t`` and ``c1_t`` hold their children's region
                  indices.
        device:   torch.device the bucket tensors live on.
    """
    num_vox: int
    num_reg: int
    # Each entry: (nodes_t, c0_t, c1_t).  Typed loosely so torch is
    # only imported at construction time.
    layers: list
    device: object


def build_sweep_plan(children: np.ndarray, layer: np.ndarray,
                      num_vox: int, device='cuda') -> SweepPlan:
    """Bucket internal nodes by depth into a SweepPlan."""
    torch = _torch()
    num_internal = children.shape[0]
    num_reg = num_vox + num_internal
    internal_layer = layer[num_vox:]
    max_L = int(internal_layer.max()) if num_internal else 0
    c0_all = children[:, 0]
    c1_all = children[:, 1]

    dev = torch.device(device)
    plan_layers = []
    for L in range(1, max_L + 1):
        mask = internal_layer == L
        if not mask.any():
            continue
        offsets = np.where(mask)[0]
        nodes = num_vox + offsets
        c0 = c0_all[offsets]
        c1 = c1_all[offsets]
        plan_layers.append((
            torch.from_numpy(nodes.astype(np.int64)).to(dev),
            torch.from_numpy(c0.astype(np.int64)).to(dev),
            torch.from_numpy(c1.astype(np.int64)).to(dev),
        ))

    return SweepPlan(num_vox=num_vox, num_reg=num_reg,
                      layers=plan_layers, device=dev)


def sweep(leaf_vals, plan: SweepPlan):
    """GPU sweep: ``(..., V)`` leaves → ``(..., R)`` with internal-node sums.

    Equivalent to ``glow.graph.compute_phase1`` but on an arbitrary
    leading-dim tensor (no fixed (b, N) layout).  Layer-by-layer::

        out[..., nodes_L] = out[..., c0_L] + out[..., c1_L]

    Internally operates on a ``(R, F)`` buffer (R as leading dim) where
    F = product of leading dims.  Fancy indexing on PyTorch's leading
    dim is ~3× faster than on a trailing dim, so this layout determines
    steady-state sweep cost.  Two contiguous transposes (one in, one
    out) cost ~1 ms total at paper config and pay for themselves many
    times over.
    """
    torch = _torch()
    V = plan.num_vox
    R = plan.num_reg
    assert leaf_vals.shape[-1] == V, \
        f'expected last dim = V={V}, got {leaf_vals.shape[-1]}'

    leading = leaf_vals.shape[:-1]
    if not leading:
        out = torch.empty(R, dtype=leaf_vals.dtype, device=plan.device)
        out[:V] = leaf_vals
        for nodes, c0, c1 in plan.layers:
            out[nodes] = out[c0] + out[c1]
        return out

    F = int(np.prod(leading))
    flat_VF = leaf_vals.reshape(F, V).t().contiguous()
    buf = torch.empty(R, F, dtype=flat_VF.dtype, device=plan.device)
    buf[:V] = flat_VF
    for nodes, c0, c1 in plan.layers:
        buf[nodes] = buf[c0] + buf[c1]
    return buf.t().contiguous().reshape(*leading, R)


# ---------------------------------------------------------------------------
# Welford reductions — shared by both paths

def welford_reduce(llr_b):
    """Reduce ``(B, R)`` fp32 LLR (with NaNs) to per-region (mu, M2, n).

    One-pass form::

        sum  = Σ x            (over finite x in the B axis)
        sumq = Σ x²
        mu   = sum / n
        M2   = sumq − n·mu²   ≡ Σ (x − mu)²

    Less stable than two-pass Welford but adequate for LLR values of
    O(1) over B≈250 perms — fp32 cancellation here is ~7 digits, ample.
    Falls back to (0, 0, 0) on n=0.
    """
    torch = _torch()
    finite = torch.isfinite(llr_b)
    n_per_reg = finite.sum(dim=0).to(torch.int64)
    safe_n = torch.clamp(n_per_reg, min=1).to(torch.float32)
    clean = torch.nan_to_num(llr_b, nan=0.0)
    sum_b = clean.sum(dim=0)
    sumsq_b = (clean * clean).sum(dim=0)
    mu = sum_b / safe_n
    M2 = sumsq_b - safe_n * mu * mu
    keep = n_per_reg > 0
    mu = torch.where(keep, mu, torch.zeros_like(mu))
    M2 = torch.where(keep, M2, torch.zeros_like(M2))
    M2 = torch.clamp(M2, min=0.0)
    return mu, M2, n_per_reg


def _batch_welford(llr_b):
    """Per-batch (mu, M2, n) over the B axis, finite-only.  Returns
    fp32 mu/M2 and int64 n.  Used by the general path's multi-batch
    Chan merge."""
    torch = _torch()
    finite = torch.isfinite(llr_b)
    n_b = finite.sum(dim=0).to(torch.int64)
    safe = torch.clamp(n_b, min=1).float()
    zero = torch.zeros_like(llr_b)
    clean = torch.where(finite, llr_b, zero)
    sum_b = clean.sum(dim=0)
    mu_b = sum_b / safe
    mu_b = torch.where(n_b > 0, mu_b, torch.zeros_like(mu_b))
    diff = torch.where(finite, llr_b - mu_b.unsqueeze(0), zero)
    M2_b = (diff * diff).sum(dim=0)
    return mu_b, M2_b, n_b


def _chan_welford_combine(mu_a, M2_a, n_a, mu_b, M2_b, n_b):
    """Chan et al.'s parallel Welford merge.

    Combines two partial moments (mu, M2, n) into one.  All tensors
    are per-region (R,).  Numerically equivalent to one-pass Welford
    on the concatenated samples.
    """
    torch = _torch()
    n_new = n_a + n_b
    safe = torch.clamp(n_new, min=1).float()
    delta = mu_b - mu_a
    n_b_f = n_b.float()
    n_a_f = n_a.float()
    mu_new = mu_a + delta * (n_b_f / safe)
    M2_new = M2_a + M2_b + delta * delta * (n_a_f * n_b_f / safe)
    keep = n_new > 0
    mu_new = torch.where(keep, mu_new, torch.zeros_like(mu_new))
    M2_new = torch.where(keep, M2_new, torch.zeros_like(M2_new))
    return mu_new, M2_new, n_new


def moments_from_partials(sum_b, sumsq_b, n_b):
    """Compute (mu, M2, n) from one-batch (sum, sumsq, n)."""
    torch = _torch()
    safe_n = torch.clamp(n_b, min=1).to(torch.float32)
    mu = sum_b / safe_n
    M2 = sumsq_b - safe_n * mu * mu
    keep = n_b > 0
    mu = torch.where(keep, mu, torch.zeros_like(mu))
    M2 = torch.where(keep, M2, torch.zeros_like(M2))
    M2 = torch.clamp(M2, min=0.0)
    return mu, M2, n_b


def _moments_to_host(mu_d, M2_d, n_d, size_d):
    """Convert on-device fp32 moments to host fp64 + derived sigma."""
    mu = mu_d.to(_torch().float64).cpu().numpy()
    M2 = M2_d.to(_torch().float64).cpu().numpy()
    n_per_reg = n_d.cpu().numpy()
    size = size_d.cpu().numpy()
    sigma = np.where(
        n_per_reg >= 2,
        np.sqrt(np.maximum(M2 / np.maximum(n_per_reg - 1, 1), 0.0)),
        np.nan)
    return {'mu': mu, 'sigma': sigma, 'M2': M2,
            'n_per_reg': n_per_reg, 'size': size}


def _sigmas_for(perm_idx: int, n_perm_inner: int, n_img: int,
                base_seed: Optional[int]) -> np.ndarray:
    """Generate the (B, N) sigma matrix using the codebase's seed scheme.

    Mirrors ``_process_permutation``'s seed layout: each outer perm
    reserves a 100k-wide block, far above any realistic n_perm_inner.
    """
    if base_seed is None:
        base_seed = (perm_idx + 1) * 100_000
    sigmas = np.empty((n_perm_inner, n_img), dtype=np.int64)
    for i in range(n_perm_inner):
        rng = np.random.default_rng(base_seed + i)
        sigmas[i] = rng.permutation(n_img)
    return sigmas


# ===========================================================================
# Intercept-only path
#
# Q0 = span(1/√N · 1).  Q0 Q0ᵀ is a constant projector that commutes
# with every permutation; T_u = yout − a₀ a₀ᵀ / |r| and log|T_u| are
# perm-invariant and hoist out of the inner loop.  Each inner perm
# reduces to one row permutation of Q1 plus the closed-form 2×2 det.
# ===========================================================================

@dataclass
class _PrepIntercept:
    """Per-outer-perm GPU precompute (intercept-only path)."""
    num_vox: int
    num_reg: int
    n_img: int
    b: int
    k: int
    a: int
    min_vox: int
    plan: SweepPlan
    size_d: object        # int64 (R,)
    size_f: object        # fp32 (R,)
    active_d: object      # bool (R,)
    T_u: object           # (b, b, R) fp32
    log_T_u: object       # (R,) fp32
    Y_fp32: object        # (b, N, V) fp32
    Q1: object            # (k, N) fp32
    device: object


def _slogdet_2x2_psd(M):
    """log|det| for a stack of 2x2 symmetric-PSD matrices.

    M: (..., 2, 2) fp32.  Clamps det >= 1e-30 to avoid -inf for
    inactive / degenerate regions; downstream masks via ``active``.
    """
    torch = _torch()
    det = M[..., 0, 0] * M[..., 1, 1] - M[..., 0, 1] * M[..., 1, 0]
    return torch.log(torch.clamp(det, min=1e-30))


def _prep_intercept(
    exp,
    *,
    q0: np.ndarray,
    q1: np.ndarray,
    children: np.ndarray,
    layer: Optional[np.ndarray],
    min_vox: int,
    device: str,
) -> _PrepIntercept:
    """Build all perm-invariant GPU state for one outer permutation.

    Caller must have already applied ``exp.permute(perm_idx)``.
    """
    torch = _torch()
    assert is_intercept_only_nuisance(exp.x, exp.contrast), \
        "intercept path requires intercept-only nuisance"

    y = exp.y
    b, n_img, num_vox = y.shape
    a = q0.shape[0]
    k = q1.shape[0]
    if layer is None:
        layer = glow.graph.compute_tree_layers(children, num_vox)

    dev = torch.device(device)
    plan = build_sweep_plan(children, layer, num_vox, device=dev)

    # Region sizes via the sweep (broadcast 1s up the tree).
    ones_leaf = torch.ones(num_vox, dtype=torch.float32, device=dev)
    size_f = sweep(ones_leaf, plan)
    size_d = size_f.to(torch.int64)
    active_d = size_d >= min_vox

    # Voxel-gram → yout_u sweep.
    Y_fp32 = torch.from_numpy(y.astype(np.float32)).to(dev)
    g_voxel = torch.einsum('bnv,cnv->bcv', Y_fp32, Y_fp32)
    yout_u = sweep(g_voxel, plan)

    # Q0·Y projection per voxel → sweep → a_0 → outer_0.
    Q0 = torch.from_numpy(np.ascontiguousarray(q0).astype(np.float32)).to(dev)
    Q1 = torch.from_numpy(np.ascontiguousarray(q1).astype(np.float32)).to(dev)
    z_0 = torch.einsum('an,bnv->abv', Q0, Y_fp32)
    a_0 = sweep(z_0, plan)
    outer_0 = torch.einsum('abr,acr->bcr', a_0, a_0)

    # T_u and log|T_u| — stay in fp32 throughout: V-dim ops are fp32,
    # and using fp64 per-region ops would mismatch the precision of
    # H (fp32 sweep output) computed downstream, breaking error-
    # cancellation in the E = T_u − H subtraction.
    inv_size = torch.where(active_d, 1.0 / size_f, torch.zeros_like(size_f))
    T_u = yout_u - outer_0 * inv_size
    log_T_u = _slogdet_2x2_psd(T_u.permute(2, 0, 1).contiguous())
    log_T_u = torch.where(active_d, log_T_u, torch.zeros_like(log_T_u))

    torch.cuda.synchronize()

    return _PrepIntercept(
        num_vox=num_vox, num_reg=plan.num_reg,
        n_img=n_img, b=b, k=k, a=a,
        min_vox=min_vox, plan=plan,
        size_d=size_d, size_f=size_f, active_d=active_d,
        T_u=T_u, log_T_u=log_T_u,
        Y_fp32=Y_fp32, Q1=Q1, device=dev)


def _make_intercept_llr_kernel():
    """Compile and return the per-batch (B, R) LLR kernel.

    Closed-form b=2, k=2.  Lazy so the compile only fires when the
    GPU path is actually used (paying torch.compile JIT cost on import
    would slow CPU-only runs).
    """
    torch = _torch()

    @torch.compile(dynamic=False)
    def _kernel(a_1, T_u, log_T_u, size_f, active):
        # a_1 entries: index (k, b).
        a00 = a_1[:, 0, 0, :]
        a01 = a_1[:, 0, 1, :]
        a10 = a_1[:, 1, 0, :]
        a11 = a_1[:, 1, 1, :]

        inv_size = torch.where(active, 1.0 / size_f, torch.zeros_like(size_f))
        # H[b, b'] = Σ_k a_1[k, b] a_1[k, b']   — three unique entries.
        h00 = (a00 * a00 + a10 * a10) * inv_size
        h01 = (a00 * a01 + a10 * a11) * inv_size
        h11 = (a01 * a01 + a11 * a11) * inv_size

        e00 = T_u[0, 0] - h00
        e01 = T_u[0, 1] - h01
        e11 = T_u[1, 1] - h11

        det_E = e00 * e11 - e01 * e01
        valid = active & (det_E > 0)
        log_E = torch.log(torch.clamp(det_E, min=1e-30))
        llr = 0.5 * size_f * (log_T_u - log_E)
        return torch.where(valid, llr, torch.full_like(llr, float('nan')))
    return _kernel


_INTERCEPT_LLR_KERNEL = None


def _intercept_llr_kernel():
    global _INTERCEPT_LLR_KERNEL
    if _INTERCEPT_LLR_KERNEL is None:
        _INTERCEPT_LLR_KERNEL = _make_intercept_llr_kernel()
    return _INTERCEPT_LLR_KERNEL


@dataclass
class _GraphedIntercept:
    """Captured-graph state for the intercept-only path, fixed B."""
    B: int
    sigmas_static: object
    sum_static: object
    sumsq_static: object
    n_static: object
    graph: object


def _intercept_batch_chain(prep: _PrepIntercept, gb: _GraphedIntercept):
    """One per-batch step, reads gb.sigmas_static, writes gb.{sum,sumsq,n}."""
    torch = _torch()
    sigma_inv = torch.argsort(gb.sigmas_static, dim=1)
    Q1_T_perm = prep.Q1.T[sigma_inv]
    Q1_perm = Q1_T_perm.permute(0, 2, 1).contiguous()
    z_1 = torch.einsum('Bkn,bnv->Bkbv', Q1_perm, prep.Y_fp32)
    a_1 = sweep(z_1, prep.plan)
    llr = _intercept_llr_kernel()(
        a_1, prep.T_u, prep.log_T_u, prep.size_f, prep.active_d)

    finite = torch.isfinite(llr)
    n_b = finite.sum(dim=0).to(torch.int64)
    clean = torch.nan_to_num(llr, nan=0.0)
    gb.sum_static.copy_(clean.sum(dim=0))
    gb.sumsq_static.copy_((clean * clean).sum(dim=0))
    gb.n_static.copy_(n_b)


def _capture_intercept(prep: _PrepIntercept, B: int) -> _GraphedIntercept:
    """Capture the per-batch chain as a CUDA Graph for fixed B."""
    torch = _torch()
    R = prep.num_reg
    N = prep.n_img
    dev = prep.device

    gb = _GraphedIntercept(
        B=B,
        sigmas_static=torch.zeros(B, N, dtype=torch.int64, device=dev),
        sum_static=torch.zeros(R, dtype=torch.float32, device=dev),
        sumsq_static=torch.zeros(R, dtype=torch.float32, device=dev),
        n_static=torch.zeros(R, dtype=torch.int64, device=dev),
        graph=None,
    )

    # Warm: trigger torch.compile JIT of the LLR kernel at this B
    # before capture.  Two passes to ensure steady-state.
    rng = np.random.default_rng(0)
    warm_sigmas = np.stack([rng.permutation(N) for _ in range(B)]).astype(np.int64)
    gb.sigmas_static.copy_(torch.from_numpy(warm_sigmas))
    for _ in range(2):
        _intercept_batch_chain(prep, gb)
    torch.cuda.synchronize()

    # Capture on a side stream (PyTorch requirement).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _intercept_batch_chain(prep, gb)
    torch.cuda.current_stream().wait_stream(s)

    gb.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gb.graph):
        _intercept_batch_chain(prep, gb)

    gb.sum_static.zero_()
    gb.sumsq_static.zero_()
    gb.n_static.zero_()
    return gb


def _run_intercept(exp, perm_idx, *, q0, q1, children, layer,
                    n_perm_inner, min_vox, base_seed, device):
    """Full intercept-only run: prep → single-batch capture → replay."""
    torch = _torch()
    prep = _prep_intercept(
        exp, q0=q0, q1=q1, children=children, layer=layer,
        min_vox=min_vox, device=device)

    sigmas = _sigmas_for(perm_idx, n_perm_inner, prep.n_img, base_seed)
    gb = _capture_intercept(prep, B=n_perm_inner)
    gb.sigmas_static.copy_(torch.from_numpy(sigmas))
    gb.graph.replay()
    mu_d, M2_d, n_d = moments_from_partials(
        gb.sum_static, gb.sumsq_static, gb.n_static)
    return _moments_to_host(mu_d, M2_d, n_d, prep.size_d)


# ===========================================================================
# General-Q0 path
#
# Q0 has non-trivial nuisance lanes, so the FL freed_lane =
# (I - Q0Q0ᵀ)[:, perm] + Q0Q0ᵀ no longer commutes with permutations.
# yout_perm and a0_perm vary per perm; we decompose them as α + β where
# β is perm-invariant (one matmul against y at prep time) and α is one
# batched (N, B(a+k)) matmul per inner-perm batch:
#
#     α[b, v, B, a+k] = (y · R[:, perm])[b, v, B, a+k]
#                     = y_2d · X,   X = (R[:, perm] · [Q0ᵀ | Q1ᵀ])
#
# T_C = α·βᵀ then sweeps with the tree exactly like the intercept
# version sweeps a single voxel-gram.
# ===========================================================================

@dataclass
class _PrepGeneral:
    """Per-outer-perm GPU precompute (general-Q0 path)."""
    num_vox: int
    num_reg: int
    n_img: int
    b: int
    k: int
    a: int
    min_vox: int
    plan: SweepPlan
    size_d: object
    size_f: object
    active_d: object
    y_d: object           # (b, N, V) fp32
    q0_d: object          # (a, N) fp32
    q1_d: object          # (k, N) fp32
    R_d: object           # (N, N) fp32 — I − Q0Q0ᵀ
    q01_T: object         # (N, a+k) fp32 — [q0ᵀ | q1ᵀ]
    y_2d_mm: object       # (b·V, N) matmul-dtype (bf16 typically)
    beta_a0: object       # (R, b, a) fp32 — perm-invariant α₀ + walk
    yout_u: object        # (R, b, b) fp32 — perm-invariant
    matmul_dtype: object
    device: object


def _prep_general(
    exp,
    *,
    q0: np.ndarray,
    q1: np.ndarray,
    children: np.ndarray,
    layer: Optional[np.ndarray],
    min_vox: int,
    device: str,
    matmul_dtype=None,
) -> _PrepGeneral:
    """Build all perm-invariant GPU state for one outer permutation.

    Caller must have already applied ``exp.permute(perm_idx)``.
    """
    torch = _torch()
    if matmul_dtype is None:
        matmul_dtype = torch.bfloat16

    y = exp.y
    b, n_img, num_vox = y.shape
    a = q0.shape[0]
    k = q1.shape[0]
    if layer is None:
        layer = glow.graph.compute_tree_layers(children, num_vox)

    dev = torch.device(device)
    plan = build_sweep_plan(children, layer, num_vox, device=dev)

    y_d = torch.from_numpy(np.ascontiguousarray(y)).to(dev, torch.float32)
    q0_d = torch.from_numpy(np.ascontiguousarray(q0)).to(dev, torch.float32)
    q1_d = torch.from_numpy(np.ascontiguousarray(q1)).to(dev, torch.float32)
    Q0Q0T = q0_d.T @ q0_d
    eye_n = torch.eye(n_img, dtype=torch.float32, device=dev)
    R_d = eye_n - Q0Q0T

    # Region sizes via sweep on a leaf ones-vector.
    ones_leaf = torch.ones(num_vox, dtype=torch.float32, device=dev)
    size_f = sweep(ones_leaf, plan)
    size_d = size_f.to(torch.int64)
    active_d = size_d >= min_vox

    # Perm-invariant β·a₀ and yout_u.  Both are (R, b, b) / (R, b, a)
    # tensors built by sweeping leaf-level tensors up the tree.
    y_2d = y_d.permute(0, 2, 1).reshape(b * num_vox, n_img)        # (b·V, N)
    beta_leaf_2d = y_2d @ q0_d.T                                    # (b·V, a)
    beta_leaf = (beta_leaf_2d.reshape(b, num_vox, a)
                              .permute(1, 0, 2).contiguous())       # (V, b, a)
    beta_a0 = sweep(beta_leaf.permute(1, 2, 0).contiguous(), plan) \
                  .permute(2, 0, 1).contiguous()                    # (R, b, a)

    # yout_u: closed-form for b=2 (three unique entries).
    yu_00 = (y_d[0] * y_d[0]).sum(dim=0)
    yu_01 = (y_d[0] * y_d[1]).sum(dim=0)
    yu_11 = (y_d[1] * y_d[1]).sum(dim=0)
    yout_leaf = torch.stack([
        torch.stack([yu_00, yu_01], dim=-1),
        torch.stack([yu_01, yu_11], dim=-1),
    ], dim=-2)                                                       # (V, b, b)
    yout_u = sweep(yout_leaf.permute(1, 2, 0).contiguous(), plan) \
                 .permute(2, 0, 1).contiguous()                      # (R, b, b)

    y_2d_mm = y_2d.to(matmul_dtype)
    q01_T = torch.cat([q0_d.T, q1_d.T], dim=1).contiguous()           # (N, a+k)

    torch.cuda.synchronize()

    return _PrepGeneral(
        num_vox=num_vox, num_reg=plan.num_reg,
        n_img=n_img, b=b, k=k, a=a,
        min_vox=min_vox, plan=plan,
        size_d=size_d, size_f=size_f, active_d=active_d,
        y_d=y_d, q0_d=q0_d, q1_d=q1_d, R_d=R_d, q01_T=q01_T,
        y_2d_mm=y_2d_mm,
        beta_a0=beta_a0, yout_u=yout_u,
        matmul_dtype=matmul_dtype, device=dev)


def _make_general_llr_kernel():
    torch = _torch()

    @torch.compile(dynamic=False)
    def _kernel(a0, a1, yout_u, T_C, size_f, active):
        """Fused per-batch (B, R) LLR for the general-Q0 path.

        Inputs (b=2 closed-form):
          a0     (B, R, 2, a) fp32 — α₀ + β₀ (region-walked)
          a1     (B, R, 2, k) fp32 — α₁ (β₁ vanishes because q0⊥q1)
          yout_u (R, 2, 2)    fp32 — perm-invariant
          T_C    (B, R, 2, 2) fp32 — perm-dependent; NOT symmetric;
                                      yout_perm = yout_u + T_C + T_Cᵀ
          size_f (R,)         fp32
          active (R,)         bool
        """
        # yout = yout_u + T_C + T_Cᵀ (symmetric).
        yout_00 = yout_u[..., 0, 0] + 2.0 * T_C[..., 0, 0]
        yout_01 = yout_u[..., 0, 1] + T_C[..., 0, 1] + T_C[..., 1, 0]
        yout_11 = yout_u[..., 1, 1] + 2.0 * T_C[..., 1, 1]

        # a0 a0ᵀ (symmetric).
        a0_0 = a0[..., 0, :]
        a0_1 = a0[..., 1, :]
        aa_00 = (a0_0 * a0_0).sum(dim=-1)
        aa_01 = (a0_0 * a0_1).sum(dim=-1)
        aa_11 = (a0_1 * a0_1).sum(dim=-1)

        inv_size = 1.0 / size_f
        t_00 = yout_00 - aa_00 * inv_size
        t_01 = yout_01 - aa_01 * inv_size
        t_11 = yout_11 - aa_11 * inv_size

        # H = a1 a1ᵀ / |r|.
        a1_0 = a1[..., 0, :]
        a1_1 = a1[..., 1, :]
        h_00 = (a1_0 * a1_0).sum(dim=-1) * inv_size
        h_01 = (a1_0 * a1_1).sum(dim=-1) * inv_size
        h_11 = (a1_1 * a1_1).sum(dim=-1) * inv_size

        e_00 = t_00 - h_00
        e_01 = t_01 - h_01
        e_11 = t_11 - h_11

        det_t = t_00 * t_11 - t_01 * t_01
        det_e = e_00 * e_11 - e_01 * e_01

        valid = active & (det_t > 0) & (det_e > 0)
        llr = 0.5 * size_f * (torch.log(det_t) - torch.log(det_e))
        return torch.where(valid, llr, torch.full_like(llr, float('nan')))
    return _kernel


_GENERAL_LLR_KERNEL = None


def _general_llr_kernel():
    global _GENERAL_LLR_KERNEL
    if _GENERAL_LLR_KERNEL is None:
        _GENERAL_LLR_KERNEL = _make_general_llr_kernel()
    return _GENERAL_LLR_KERNEL


@dataclass
class _GraphedGeneral:
    """Captured-graph state for the general-Q0 path, fixed B."""
    B: int
    sigmas_static: object
    mu_static: object
    M2_static: object
    n_static: object
    graph: object


def _general_batch_chain(prep: _PrepGeneral, gb: _GraphedGeneral):
    """One per-batch step for the general path.  Accumulates moments
    into static buffers via in-place Chan merge so the captured graph
    can be replayed many times per outer perm."""
    torch = _torch()
    B = gb.B
    N = prep.n_img
    V = prep.num_vox
    R = prep.num_reg
    b = prep.b
    a = prep.a
    k = prep.k
    ak = a + k

    sigma_b = gb.sigmas_static
    perm_b = sigma_b.argsort(dim=1)
    perm_exp = perm_b.unsqueeze(1).expand(B, N, N)
    R_perm = torch.gather(prep.R_d.unsqueeze(0).expand(B, N, N),
                           dim=2, index=perm_exp)
    U01 = torch.matmul(R_perm, prep.q01_T)                           # (B, N, a+k)

    # Shape-J: (B, N, a+k) → (N, B(a+k)) so the heavy alpha matmul is
    # one wide 2D mm.
    X = U01.permute(1, 0, 2).reshape(N, B * ak).contiguous() \
           .to(prep.matmul_dtype)
    alpha_2d = torch.mm(prep.y_2d_mm, X).to(torch.float32)            # (b·V, B·(a+k))
    alpha_leaf = (alpha_2d.reshape(b, V, B, ak)
                            .permute(2, 1, 0, 3))                     # (B, V, b, a+k) strided
    alpha_a0_leaf = alpha_leaf[..., :a].contiguous()                  # (B, V, b, a)
    alpha_a1_leaf = alpha_leaf[..., a:].contiguous()                  # (B, V, b, k)

    # T_C leaf-level: (α₀)·(β₀)ᵀ per voxel, b=2 entries explicit.
    beta_leaf = (prep.beta_a0[:V])                                    # (V, b, a)
    a0_0 = alpha_a0_leaf[:, :, 0, :]
    a0_1 = alpha_a0_leaf[:, :, 1, :]
    b0_0 = beta_leaf[:, 0, :]
    b0_1 = beta_leaf[:, 1, :]
    T_C_leaf = torch.empty((B, V, b, b), device=prep.device,
                            dtype=torch.float32)
    T_C_leaf[:, :, 0, 0] = (a0_0 * b0_0).sum(dim=-1)
    T_C_leaf[:, :, 0, 1] = (a0_0 * b0_1).sum(dim=-1)
    T_C_leaf[:, :, 1, 0] = (a0_1 * b0_0).sum(dim=-1)
    T_C_leaf[:, :, 1, 1] = (a0_1 * b0_1).sum(dim=-1)

    # Allocate (B, R, *) sweep buffers and copy leaves, then walk.
    alpha_a0 = torch.empty((B, R, b, a), device=prep.device,
                            dtype=torch.float32)
    alpha_a1 = torch.empty((B, R, b, k), device=prep.device,
                            dtype=torch.float32)
    T_C = torch.empty((B, R, b, b), device=prep.device,
                       dtype=torch.float32)
    alpha_a0[:, :V, :, :] = alpha_a0_leaf
    alpha_a1[:, :V, :, :] = alpha_a1_leaf
    T_C[:, :V, :, :] = T_C_leaf

    children_d = None  # children layout already lives in plan.layers
    for nodes, c0, c1 in prep.plan.layers:
        alpha_a0[:, nodes, :, :] = alpha_a0[:, c0, :, :] + alpha_a0[:, c1, :, :]
        alpha_a1[:, nodes, :, :] = alpha_a1[:, c0, :, :] + alpha_a1[:, c1, :, :]
        T_C[:, nodes, :, :] = T_C[:, c0, :, :] + T_C[:, c1, :, :]

    a0_full = alpha_a0 + prep.beta_a0.unsqueeze(0)                    # (B, R, b, a)
    llr_b = _general_llr_kernel()(
        a0_full, alpha_a1, prep.yout_u, T_C, prep.size_f, prep.active_d)

    # In-place Chan merge into running moments.
    mu_b, M2_b, n_b = _batch_welford(llr_b)
    n_a_f = gb.n_static.float()
    n_b_f = n_b.float()
    n_new = gb.n_static + n_b
    safe = torch.clamp(n_new, min=1).float()
    delta = mu_b - gb.mu_static
    new_M2 = gb.M2_static + M2_b + delta * delta * (n_a_f * n_b_f / safe)
    new_mu = (n_a_f * gb.mu_static + n_b_f * mu_b) / safe
    keep = n_new > 0
    new_mu = torch.where(keep, new_mu, torch.zeros_like(new_mu))
    new_M2 = torch.where(keep, new_M2, torch.zeros_like(new_M2))
    gb.mu_static.copy_(new_mu)
    gb.M2_static.copy_(new_M2)
    gb.n_static.copy_(n_new)


def _capture_general(prep: _PrepGeneral, B: int,
                      warm_sigmas: np.ndarray) -> _GraphedGeneral:
    """Capture the per-batch general-path chain as a CUDA Graph."""
    torch = _torch()
    R = prep.num_reg
    N = prep.n_img
    dev = prep.device

    gb = _GraphedGeneral(
        B=B,
        sigmas_static=torch.zeros(B, N, dtype=torch.int64, device=dev),
        mu_static=torch.zeros(R, dtype=torch.float32, device=dev),
        M2_static=torch.zeros(R, dtype=torch.float32, device=dev),
        n_static=torch.zeros(R, dtype=torch.int64, device=dev),
        graph=None,
    )

    gb.sigmas_static.copy_(torch.from_numpy(warm_sigmas))
    _general_batch_chain(prep, gb)
    torch.cuda.synchronize()
    gb.mu_static.zero_(); gb.M2_static.zero_(); gb.n_static.zero_()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _general_batch_chain(prep, gb)
    torch.cuda.current_stream().wait_stream(s)
    gb.mu_static.zero_(); gb.M2_static.zero_(); gb.n_static.zero_()

    gb.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gb.graph):
        _general_batch_chain(prep, gb)
    gb.mu_static.zero_(); gb.M2_static.zero_(); gb.n_static.zero_()
    return gb


def _largest_divisor(n: int, candidates) -> int:
    """Largest c in candidates s.t. n % c == 0.  Used to pick a
    batch size that evenly divides n_perm_inner (CUDA Graphs are
    shape-specialised)."""
    for c in candidates:
        if c <= n and n % c == 0:
            return c
    return 1


def _probe_general_batch_size(prep: _PrepGeneral, n_perm_inner: int,
                                warm_sigmas_for) -> Tuple[int, _GraphedGeneral]:
    """Pick the largest B (dividing n_perm_inner) that captures without OOM.

    Tries B candidates in descending order.  The largest successful
    capture is returned, along with the captured GraphedGeneral
    (so the caller doesn't recapture).  ``warm_sigmas_for(B)`` must
    return a (B, N) int64 ndarray.
    """
    torch = _torch()
    # Sensible divisors of paper-config n_perm_inner=250.  At V≈25k
    # and B=50 the per-batch tensors total ~210 MB (well under 8 GB);
    # B=125 fits on 24 GB.  Start large and step down.
    candidates = [125, 50, 25, 10, 5, 2, 1]
    candidates = [c for c in candidates if c <= n_perm_inner
                   and n_perm_inner % c == 0]
    if not candidates:
        candidates = [1]

    last_oom = None
    for B in candidates:
        try:
            gb = _capture_general(prep, B, warm_sigmas_for(B))
            return B, gb
        except torch.cuda.OutOfMemoryError as e:
            last_oom = e
            torch.cuda.empty_cache()
            continue
    raise RuntimeError(
        f'no batch size from {candidates} fit in GPU memory') from last_oom


def _run_general(exp, perm_idx, *, q0, q1, children, layer,
                  n_perm_inner, min_vox, base_seed, device):
    """Full general-Q0 run: prep → batch-size probe → capture → replay loop."""
    torch = _torch()
    prep = _prep_general(
        exp, q0=q0, q1=q1, children=children, layer=layer,
        min_vox=min_vox, device=device)

    sigmas = _sigmas_for(perm_idx, n_perm_inner, prep.n_img, base_seed)
    sigmas_d = torch.from_numpy(sigmas).to(prep.device)

    # Probe biggest B that fits.  The probe's capture is reusable.
    B, gb = _probe_general_batch_size(
        prep, n_perm_inner, lambda B: sigmas[:B])

    # Replay across all batches.
    for batch_start in range(0, n_perm_inner, B):
        gb.sigmas_static.copy_(sigmas_d[batch_start:batch_start + B])
        gb.graph.replay()
    torch.cuda.synchronize()

    out = _moments_to_host(gb.mu_static, gb.M2_static, gb.n_static,
                            prep.size_d)
    out['batch_size'] = B
    return out


# ===========================================================================
# Public dispatcher

def run_inner_perms(
    exp,
    perm_idx: int,
    *,
    q0: np.ndarray,
    q1: np.ndarray,
    children: np.ndarray,
    layer: np.ndarray,
    n_perm_inner: int,
    min_vox: int = 4,
    base_seed: Optional[int] = None,
    device: str = 'cuda',
) -> dict:
    """Run ``n_perm_inner`` inner FL perms on GPU; return Welford moments.

    Dispatches on ``is_intercept_only_nuisance(exp.x, exp.contrast)``:
    intercept-only experiments take the fast closed-form path; the rest
    take the general-Q0 path with α/β decomposition.  Both return the
    same contract.

    Args:
        exp:           Already-outer-permuted Experiment (caller does
                       ``exp.permute(perm_idx)``).
        perm_idx:      outer permutation index (used to seed inner
                       perms when ``base_seed`` is None).
        q0, q1:        contrast subspaces from ``decompose``.
        children:      (num_internal, 2) Ward children.
        layer:         (num_reg,) depth table.
        n_perm_inner:  number of inner perms.  Must be >= 1.
        min_vox:       regions with size < min_vox return n_per_reg = 0.
        base_seed:     RNG seed base; defaults to
                       ``(perm_idx + 1) * 100_000`` to match the CPU
                       race's seed scheme.
        device:        torch device string.

    Returns:
        dict with ``mu``, ``sigma``, ``M2``, ``n_per_reg``, ``size``
        (all fp64 / int64 host arrays).  General path additionally
        returns ``batch_size`` (the captured-graph B that fit).
    """
    assert n_perm_inner >= 1, 'n_perm_inner must be >= 1 for GPU path'

    if is_intercept_only_nuisance(exp.x, exp.contrast):
        return _run_intercept(
            exp, perm_idx, q0=q0, q1=q1, children=children, layer=layer,
            n_perm_inner=n_perm_inner, min_vox=min_vox,
            base_seed=base_seed, device=device)
    return _run_general(
        exp, perm_idx, q0=q0, q1=q1, children=children, layer=layer,
        n_perm_inner=n_perm_inner, min_vox=min_vox,
        base_seed=base_seed, device=device)
