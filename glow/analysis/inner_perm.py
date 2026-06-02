"""Inner Freedman-Lane permutation drivers.

For one outer permutation: given the outer-perm tree (Ward children)
and an experiment, draw n_perm inner-perm LLR samples (Freedman & Lane
1983) and reduce them to per-region (mu, std). Two production backends:

  - cpu_perm -- the work horse. Rides glow.graph.iter_llr_perm:
    per-voxel sufficient statistics are computed once; per perm-chunk
    the only work is one big Q^T P r_v GEMM plus cumsum-and-diff
    aggregation on the DFS pre-order axis. Each chunk of draws is folded
    into a Welford / Chan-parallel accumulator and dropped -- the full
    (n_perm, num_reg) draws matrix is never materialized. Handles
    intercept-only and general-Q0 nuisance on the same code path --
    under intercept-only Q0 commutes with P, so the rho and X_v terms
    come out numerically zero and the rest of the algorithm is
    unaffected.
  - cpu_reliable -- trust anchor for tests. Drives the per-region
    iter_mancova + get_llr path per draw -- an independent code path
    used to cross-validate cpu_perm.

Both share a single keyword-only signature and return (mu, std):

    mu, std = inner_perm.cpu_perm(
        exp=exp, base_seed=base_seed, n_perm=n_perm,
        q0=q0, q1=q1, children=children, min_vox=min_vox)

exp is whatever experiment the inner perms should operate on -- the
backends don't need to know whether it carries an outer permutation;
they just sample iid permutations from it.

base_seed is the starting RNG seed: each draw i uses base_seed + i. The
caller is responsible for choosing a non-colliding base across outer
perms (_glow.py reserves a 100_000-wide block per outer perm).

cpu_reliable_full returns the raw (n_perm, num_reg) LLR draws matrix;
tests that want to inspect individual draws call it directly. cpu_perm
no longer has a _full variant -- callers that need per-draw output
materialize via np.vstack(list(glow.graph.iter_llr_perm(...))).
"""
import numpy as np
from scipy.special import ndtri

import glow.graph
from glow.analysis import mancova
from glow.experiment import permute


def _welford_combine(chunk, n, mean, M2):
    """Fold one (Pc, num_reg) NaN-aware draw-chunk into running moments.

    One step of Chan's parallel-combine rule (Chan, Golub & LeVeque 1979);
    NaN cells are excluded from the per-region count. Returns the updated
    (n, mean, M2) -- shared by the fixed-budget _welford_moments loop and the
    inner-perm race, so both reduce draws with identical semantics.

    Args:
        chunk (np.array): (Pc, num_reg) NaN-aware LLR draws
        n (np.array): (num_reg,) running valid-sample count
        mean (np.array): (num_reg,) running mean
        M2 (np.array): (num_reg,) running sum of squared deviations

    Returns:
        n, mean, M2 (np.array): the updated (num_reg,) accumulators
    """
    valid = ~np.isnan(chunk)
    chunk_safe = np.where(valid, chunk, 0.0)
    n_b = valid.sum(axis=0).astype(np.float64)
    sum_b = chunk_safe.sum(axis=0)

    # Guarded divisions: when n_b == 0 the chunk contributes nothing;
    # mean_b can be anything (multiplied by 0 below).
    safe_nb = np.where(n_b > 0, n_b, 1.0)
    mean_b = sum_b / safe_nb
    dev = np.where(valid, chunk_safe - mean_b[None, :], 0.0)
    M2_b = (dev * dev).sum(axis=0)

    new_n = n + n_b
    safe_new_n = np.where(new_n > 0, new_n, 1.0)
    delta = mean_b - mean
    mean = mean + delta * (n_b / safe_new_n)
    M2 = M2 + M2_b + delta * delta * (n * n_b / safe_new_n)
    return new_n, mean, M2


def _welford_finalize(n, mean, M2):
    """Reduce running (n, mean, M2) to (mu, std).

    Returns NaN mu where no valid sample accumulated (n < 1) and NaN std
    where fewer than 2 did (n < 2); std uses ddof=1. ULP-level negative
    variance is clamped to 0 before the sqrt.

    Args:
        n (np.array): (num_reg,) valid-sample count
        mean (np.array): (num_reg,) running mean
        M2 (np.array): (num_reg,) running sum of squared deviations

    Returns:
        mu (np.array): (num_reg,) per-region mean
        std (np.array): (num_reg,) per-region std (ddof=1)
    """
    mu = np.where(n > 0, mean, np.nan)
    safe_dof = np.where(n > 1, n - 1, 1.0)
    var = np.where(n > 1, M2 / safe_dof, np.nan)
    var = np.where((~np.isnan(var)) & (var < 0), 0.0, var)
    return mu, np.sqrt(var)


def _welford_moments(chunks, num_reg: int):
    """Reduce an iterator of LLR draw-chunks to per-region (mu, std).

    Per-region accumulators (n, mean, M2) updated via Chan's
    parallel-combine rule (Chan, Golub & LeVeque 1979). NaN cells are
    excluded from the count, matching the old nanmean / nanstd(ddof=1)
    semantics. Returns NaN mu where no chunk had a valid sample, NaN std
    where fewer than 2 valid samples accumulated.

    Numerically stable across chunk boundaries: the naive single-pass
    (sumsq - sum^2/n)/(n-1) formula loses precision when var << mean^2;
    Chan-parallel updates avoid the cancellation by maintaining the
    centered M2 directly.

    Args:
        chunks: iterator of (Pc, num_reg) NaN-aware LLR draw chunks,
            where Pc is the number of draws in the chunk
        num_reg (int): number of regions (column dimension of each chunk)

    Returns:
        mu (np.array): (num_reg,) per-region mean
        std (np.array): (num_reg,) per-region std (ddof=1)
    """
    n = np.zeros(num_reg, dtype=np.float64)
    mean = np.zeros(num_reg, dtype=np.float64)
    M2 = np.zeros(num_reg, dtype=np.float64)
    for chunk in chunks:
        n, mean, M2 = _welford_combine(chunk, n, mean, M2)
    return _welford_finalize(n, mean, M2)


def cpu_perm(*, exp, base_seed: int, n_perm: int, q0, q1, children,
             min_vox: int):
    """Compute inner-perm (mu, std) via the streaming perm-LLR backend.

    Builds Phase-1 sufficient statistics once, then folds each
    perm-chunk of LLR draws into a Welford / Chan-parallel accumulator.
    Peak memory is Phase-1 state + per-chunk Phase-2 temporaries; the
    full (n_perm, num_reg) draws matrix is never materialized.

    Handles intercept-only and general-Q0 nuisance on the same code
    path.

    Args:
        exp (Experiment): experiment to sample inner perms from
        base_seed (int): draw i uses RNG seed base_seed + i
        n_perm (int): number of inner FL draws
        q0 (np.array): (a0, num_img) nuisance subspace
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN

    Returns:
        mu (np.array): (num_reg,) inner-null mean per region
        std (np.array): (num_reg,) inner-null std per region
    """
    num_vox = exp.y.shape[2]
    num_img = exp.y.shape[1]
    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=num_vox)
    perms = np.empty((n_perm, num_img), dtype=np.int64)
    for i in range(n_perm):
        perms[i] = permute._perm_indices(base_seed + i, num_img)
    num_reg = int(region_l.shape[0])
    chunks = glow.graph.iter_llr_perm(
        y=exp.y, q0=q0, q1=q1, perms=perms,
        leaf_ord=leaf_ord, region_l=region_l, region_h=region_h,
        min_size=min_vox)
    return _welford_moments(chunks, num_reg)


def cpu_reliable_full(*, exp, base_seed: int, n_perm: int,
                      q0, q1, children, min_vox: int):
    """Compute trust-anchor (n_perm, num_reg) LLR draws.

    Thin wrapper over the per-region iter_mancova + get_llr path: for
    each draw, FL-permute via exp.permute(base_seed + i), then iterate
    (reg_idx, size, e, h) per region and finish with
    get_llr(e, h, n=size). No batching, no closed-form 2x2 slogdet, no
    Phase-1 hoisting -- an independent code path from
    compute_llr_batched for cross-validating the optimised backend.

    q0 / q1 are accepted for interface parity but unused: iter_mancova
    recomputes them internally via decompose(exp.x, exp.contrast), which
    is exactly the same (q0, q1) the caller would have passed in.

    Args:
        exp (Experiment): experiment to sample inner perms from
        base_seed (int): draw i uses RNG seed base_seed + i
        n_perm (int): number of inner FL draws
        q0 (np.array): accepted for interface parity, unused
        q1 (np.array): accepted for interface parity, unused
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN

    Returns:
        draws (np.array): (n_perm, num_reg) per-draw LLR, NaN where a
            region is smaller than min_vox or could not be computed
    """
    del q0, q1
    num_reg = exp.y.shape[2] + children.shape[0]
    draws = np.full((n_perm, num_reg), np.nan, dtype=np.float64)
    for i in range(n_perm):
        _exp = exp.permute(base_seed + i)
        for reg_idx, size, e, h in glow.graph.iter_mancova(
                _exp, children=children):
            if size < min_vox:
                continue
            draws[i, reg_idx] = mancova.get_llr(e, h, n=size)
    return draws


def cpu_reliable(*, exp, base_seed: int, n_perm: int, q0, q1, children,
                 min_vox: int):
    """Compute trust-anchor (mu, std) via the reliable draws backend.

    Folds cpu_reliable_full draws through the same Welford accumulator
    cpu_perm uses, so both backends share identical moment-reduction
    semantics. Args match cpu_reliable_full; returns per-region
    (mu, std), each (num_reg,).
    """
    draws = cpu_reliable_full(
        exp=exp, base_seed=base_seed, n_perm=n_perm,
        q0=q0, q1=q1, children=children, min_vox=min_vox)
    return _welford_moments([draws], draws.shape[1])


def _race_keep(llr_obs, mu, std, n, active, p_keep_thresh: float):
    """Pick the survivor set: regions that could still be the max-z region.

    Keep region r if its probability of beating the current leader L (the
    interim arg-max of z) exceeds p_keep_thresh, under a normal model for each
    region's true z: z_r ~ N(z_r_hat, se_r^2), se = sqrt((1 + z^2/2)/n) the
    delta-method SE of the standardised z. With z_r, z_L independent that is

        P(z_r > z_L) = Phi((z_r - z_L) / sqrt(se_r^2 + se_L^2)) > p_keep_thresh
                   <=> (z_r - z_L) / sqrt(se_r^2 + se_L^2) > Phi^{-1}(p_keep_thresh).

    This is a scale-free reparameterisation of a k-sigma band
    (p_keep_thresh = Phi(-k_sigma)); the survivor COUNT adapts per perm instead
    of being a fixed number, so it behaves consistently across num_vox. The
    leader is always retained, guaranteeing a non-empty survivor set whose max
    equals the full-tree max -- what keeps the FWER null exact.

    Recall note: this anchors on the leader, so it protects observed
    discoveries (regions above the FWER cutoff, which can sit well below the
    leader) only while the burn-in SE is wide enough -- empirically recall is
    1.0 at race_init=15, p_keep_thresh<=1e-6. A longer burn-in tightens the CIs
    and would need a smaller p_keep_thresh to stay lossless.

    Args:
        llr_obs (np.array): (num_reg,) observed LLR for this outer-perm tree
        mu (np.array): (num_reg,) burn-in inner-null mean
        std (np.array): (num_reg,) burn-in inner-null std
        n (np.array): (num_reg,) burn-in sample count per region
        active (np.array): (num_reg,) bool, size >= min_vox and finite llr
        p_keep_thresh (float): keep regions with > this probability of beating
            the leader (smaller => keep more)

    Returns:
        kept (np.array): (num_reg,) bool survivor mask
    """
    std_safe = np.where(std < 1e-12, 1.0, std)
    z = np.where(active & (n >= 2), (llr_obs - mu) / std_safe, np.nan)
    se = np.sqrt((1.0 + z ** 2 / 2.0) / np.maximum(n, 1.0))
    eligible = active & np.isfinite(z) & (n >= 2)

    kept = np.zeros_like(active)
    if not eligible.any():
        return kept
    lead = int(np.argmax(np.where(eligible, z, -np.inf)))
    z_crit = ndtri(p_keep_thresh)              # Phi^{-1}(p), negative
    denom = np.sqrt(se ** 2 + se[lead] ** 2)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    kept = eligible & ((z - z[lead]) / denom > z_crit)
    kept[lead] = True
    return kept


def cpu_perm_race(*, exp, llr_obs, base_seed: int, n_perm: int, q0, q1,
                  children, min_vox: int, race_init: int = 15,
                  p_keep_thresh: float = 1e-6):
    """Compute inner-perm (mu, std, kept) via the racing fast-kernel backend.

    The same fixed-budget statistic as cpu_perm, but inner FL draws are spent
    mostly on regions that could plausibly be the per-perm max-z region -- the
    only thing FWER's max-z null needs. The procedure is IDENTICAL on every
    outer perm (the caller must apply it to the observed perm too), which keeps
    the max-z statistics exchangeable and the FWER bound exact (Lehmann & Romano
    Thm 15.2.1; Hemerik & Goeman 2018) -- so race_init / p_keep_thresh are
    speed/power knobs, never validity knobs.

    Three stages, all on the intercept-only fast kernel
    (compute_llr_inner_fast; the caller must have checked
    is_intercept_only_nuisance):

      1. Burn-in: race_init draws over all active regions -> per-region
         (mu, std, n) via the shared Welford accumulator.
      2. Trim: keep the survivor set S (see _race_keep) -- regions with
         > p_keep_thresh probability of beating the interim leader.
      3. Tail: draw the remaining n_perm - race_init perms for S only, folding
         them into the same accumulator. Non-survivors stay frozen at their
         race_init samples (their mu/std are coarse but they are, by
         construction, confidently below the max).

    ysum and t = yout - a0 a0^T / size are FL-invariant under intercept-only,
    so they are built once (region_stats_arrays) and every draw is a single
    permuted-q1 hypothesis term -- per-draw cost in the tail scales with |S|,
    not num_reg.

    Args:
        exp (Experiment): experiment to sample inner perms from
        llr_obs (np.array): (num_reg,) observed LLR for this tree -- needed to
            score z = (llr_obs - mu)/std for the trim
        base_seed (int): draw i uses RNG seed base_seed + i
        n_perm (int): total inner FL draws for survivors (race_init counts
            toward this)
        q0 (np.array): (a0, num_img) nuisance subspace (intercept-only)
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN / inactive
        race_init (int): burn-in draws over all regions before the trim
        p_keep_thresh (float): keep regions with > this probability of beating
            the interim leader (scale-free; smaller keeps more). See
            _race_keep for the recall/race_init coupling.

    Returns:
        mu (np.array): (num_reg,) inner-null mean (survivors: n_perm samples;
            others: race_init samples)
        std (np.array): (num_reg,) inner-null std, same sampling split
        kept (np.array): (num_reg,) bool survivor mask -- the caller restricts
            the max-z (and discoveries) to these regions
    """
    num_vox = exp.y.shape[2]
    num_img = exp.y.shape[1]
    num_reg = num_vox + int(children.shape[0])

    ysum, yout, size = glow.graph.region_stats_arrays(exp.y, children)
    dtype = exp.y.dtype if exp.y.dtype == np.float32 else np.float64
    a0 = np.einsum('rbn,an->rba', ysum, q0, optimize=True)
    t = yout - np.einsum('rba,rca->rbc', a0, a0,
                         optimize=True) / size.astype(dtype)[:, None, None]
    q1_T = q1.T.astype(dtype, copy=False)
    active = (size >= min_vox) & np.isfinite(llr_obs)

    n = np.zeros(num_reg, dtype=np.float64)
    mean = np.zeros(num_reg, dtype=np.float64)
    M2 = np.zeros(num_reg, dtype=np.float64)

    def accumulate(idx, draw_range):
        # Per-draw Welford (Chan's rule at n_b=1) restricted to idx, updated
        # in place: cost scales with |idx|, so the tail touches only
        # survivors instead of a full num_reg-wide chunk per draw.
        for i in draw_range:
            # argsort(_perm_indices) matches the FL convention iter_llr_perm /
            # cpu_reliable use (verified to fp round-off), so race draws are
            # valid inner-null samples.
            perm = np.argsort(permute._perm_indices(base_seed + i, num_img))
            x = glow.graph.compute_llr_inner_fast(
                t[idx], ysum[idx], size[idx], q1_T[perm, :])
            fin = np.isfinite(x)
            sub = idx[fin]
            xv = x[fin]
            n[sub] += 1.0
            delta = xv - mean[sub]
            mean[sub] += delta / n[sub]
            M2[sub] += delta * (xv - mean[sub])

    # Stage 1: burn-in over all active regions.
    act_idx = np.where(active)[0]
    burn = min(race_init, n_perm)
    accumulate(act_idx, range(burn))

    # Stage 2: trim to survivors using the burn-in moments.
    mu_b, std_b = _welford_finalize(n, mean, M2)
    kept = _race_keep(llr_obs, mu_b, std_b, n, active, p_keep_thresh)

    # Stage 3: draw survivors to n_perm.
    kept_idx = np.where(kept)[0]
    if kept_idx.size:
        accumulate(kept_idx, range(burn, n_perm))

    mu, std = _welford_finalize(n, mean, M2)
    return mu, std, kept


def cpu_perm_race_general(*, exp, llr_obs, base_seed: int, n_perm: int, q0, q1,
                          children, min_vox: int, race_init: int = 15,
                          p_keep_thresh: float = 1e-6,
                          max_kernel_bytes: int = 16 * 2 ** 30):
    """Compute inner-perm (mu, std, kept) for GENERAL nuisance via the race.

    The general-Q0 counterpart of cpu_perm_race. Same burn-in / trim / tail
    structure and the same exact-FWER guarantee (identical procedure on every
    outer perm), but the intercept-only fast kernel does not apply, so:

      - Burn-in rides the general streaming backend (graph.iter_llr_perm, what
        cpu_perm uses) over ALL active regions for race_init draws.
      - Tail draws the survivors via the par/perp M-kernel
        (graph.build_survivor_kernels once, then compute_llr_inner_kernel per
        draw) -- valid for any Q0, per-draw cost scaling with the survivor
        count rather than num_reg.

    Burn-in and tail use identical permutations per draw index
    (permute._perm_indices(base_seed + i)), so a survivor's combined moments
    equal a full cpu_perm's to fp round-off. The M tensor is
    num_surv * b^2 * num_img^2 floats; if it would exceed max_kernel_bytes the
    tail falls back to streaming over all regions (kept := active, no survivor
    speedup) -- correctness and the FWER bound are unchanged. See
    docs/notes/general_q0_race_kernel.md.

    Args:
        exp (Experiment): experiment to sample inner perms from
        llr_obs (np.array): (num_reg,) observed LLR for this tree (scores the
            trim z = (llr_obs - mu)/std)
        base_seed (int): draw i uses RNG seed base_seed + i
        n_perm (int): total inner FL draws for survivors
        q0 (np.array): (a0, num_img) nuisance subspace
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN / inactive
        race_init (int): burn-in draws over all regions before the trim
        p_keep_thresh (float): keep regions with > this probability of beating
            the interim leader (scale-free; smaller keeps more)
        max_kernel_bytes (int): cap on the M-kernel allocation before the
            streaming fallback

    Returns:
        mu (np.array): (num_reg,) inner-null mean (survivors: n_perm samples;
            others: race_init samples)
        std (np.array): (num_reg,) inner-null std, same sampling split
        kept (np.array): (num_reg,) bool survivor mask
    """
    num_vox = exp.y.shape[2]
    num_img = exp.y.shape[1]
    b = exp.y.shape[0]
    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=num_vox)
    num_reg = int(region_l.shape[0])
    size = (region_h - region_l).astype(np.int64)
    active = (size >= min_vox) & np.isfinite(llr_obs)

    n = np.zeros(num_reg, dtype=np.float64)
    mean = np.zeros(num_reg, dtype=np.float64)
    M2 = np.zeros(num_reg, dtype=np.float64)

    def stream(draw_range):
        # Fold whole-tree draws from the general streaming backend into the
        # shared Welford accumulator (Chan parallel-combine per chunk).
        nonlocal n, mean, M2
        idxs = list(draw_range)
        if not idxs:
            return
        perms = np.empty((len(idxs), num_img), dtype=np.int64)
        for j, i in enumerate(idxs):
            perms[j] = permute._perm_indices(base_seed + i, num_img)
        for chunk in glow.graph.iter_llr_perm(
                y=exp.y, q0=q0, q1=q1, perms=perms,
                leaf_ord=leaf_ord, region_l=region_l, region_h=region_h,
                min_size=min_vox):
            n, mean, M2 = _welford_combine(chunk, n, mean, M2)

    # Stage 1: burn-in over ALL regions via the general streaming backend.
    burn = min(race_init, n_perm)
    stream(range(burn))

    # Stage 2: trim to survivors using the burn-in moments.
    mu_b, std_b = _welford_finalize(n, mean, M2)
    kept = _race_keep(llr_obs, mu_b, std_b, n, active, p_keep_thresh)
    kept_idx = np.where(kept)[0]

    # Stage 3: tail -- survivors only via the M-kernel, else stream-fallback.
    n_tail = n_perm - burn
    if n_tail > 0 and kept_idx.size:
        itemsize = 4 if exp.y.dtype == np.float32 else 8
        m_bytes = int(kept_idx.size) * b * b * num_img * num_img * itemsize
        if m_bytes <= max_kernel_bytes:
            kernels = glow.graph.build_survivor_kernels(
                exp.y, children, kept_idx, q0)
            # freed_lane built inline from q0 (= get_freed_lane), avoiding a
            # re-decompose per draw; proj is the (num_img, num_img) Q0Q0^T.
            proj = q0.T @ q0
            eye = np.eye(num_img, dtype=proj.dtype)
            for i in range(burn, n_perm):
                perm = permute._perm_indices(base_seed + i, num_img)
                freed_lane = (eye - proj)[:, perm] + proj
                x = glow.graph.compute_llr_inner_kernel(
                    kernels, q0, q1, freed_lane, perm, num_reg,
                    min_size=min_vox)
                xk = x[kept_idx]
                fin = np.isfinite(xk)
                sub = kept_idx[fin]
                xv = xk[fin]
                n[sub] += 1.0
                delta = xv - mean[sub]
                mean[sub] += delta / n[sub]
                M2[sub] += delta * (xv - mean[sub])
        else:
            # M would be too large: stream the tail over all regions (no trim).
            stream(range(burn, n_perm))
            kept = active.copy()

    mu, std = _welford_finalize(n, mean, M2)
    return mu, std, kept
