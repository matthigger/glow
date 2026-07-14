"""Inner Freedman-Lane permutation drivers.

For one outer permutation: given the outer-perm tree (Ward children)
and an experiment, draw n_perm inner-perm LLR samples (Freedman & Lane
1983) and reduce them to per-region (mu, std). Three production backends:

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
  - cpu_perm_race -- the fast path. Draws a short burn-in over all
    regions on cpu_perm's streaming kernel, then spends the remaining
    draws on a shrinking confusion set of contenders for the
    per-permutation max-z, drawn in geometrically growing rounds via the
    low-rank general-Q0 kernel (glow.graph.compute_llr_inner_kernel) and
    re-trimmed / re-admitted each round on an LUCB confidence-interval
    rule. Same contiguous draws as cpu_perm per seed, so a contender's
    moments match to float round-off; the trim only costs power, never
    validity (see cpu_perm_race).
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
returns only the reduced (mu, std); callers that need per-draw output
materialize via np.vstack(list(glow.graph.iter_llr_perm(...))).
"""
import numpy as np
from scipy.stats import norm

import glow.graph
from glow.analysis import mancova
from glow.experiment import permute


# Standard race knobs (speed/power, never validity -- see cpu_perm_race). The
# single source of truth for the survivor race's burn-in depth and keep-
# probability floor: AnalysisGLOW's recipe defaults and the benchmark config
# both reference these, so there is one canonical value, not a literal repeated
# per call site.
N_PERM_INNER_RACE = 15
RACE_P_KEEP_THRESH = 1e-6

# Tail draws are folded into the Welford accumulator this many at a time
# rather than materialized as one (n_perm - n_perm_inner_race, num_reg)
# array: one _welford_combine over the whole tail allocates several
# temporaries that size (chunk_safe, the mean-centred deviations, their
# square), which dominates the inner-perm peak at large num_reg. Matches the
# burn-in's iter_llr_perm streaming granularity.
_TAIL_FOLD_CHUNK = 8


def _welford_combine(chunk, n, mean, M2):
    """Fold one (Pc, num_reg) NaN-aware draw-chunk into running moments.

    One step of Chan's parallel-combine rule (Chan, Golub & LeVeque 1979);
    NaN cells are excluded from the per-region count. Returns the updated
    (n, mean, M2) -- the step _welford_moments folds every draw-chunk through,
    so the cpu_perm and cpu_reliable backends reduce draws identically.

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
    excluded from the count, giving nanmean / nanstd(ddof=1) semantics.
    Returns NaN mu where no chunk had a valid sample, NaN std where fewer
    than 2 valid samples accumulated.

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


def _race_keep(*, llr_obs, mu, std, n, size, min_vox: int,
               p_keep_thresh: float):
    """Return the survivor mask: the LUCB confusion set around the leader.

    Models each region's true standardised score as z_r ~ N(z_hat_r,
    se_r^2), with the delta-method standard error
    se_r = sqrt((1 + z_hat_r^2 / 2) / n_r) -- the mean contributes 1/n, the
    std estimate z_hat^2 / 2n. With radius k_sigma = -Phi^{-1}(p_keep_thresh),
    give each region the interval [z_hat_r - k se_r, z_hat_r + k se_r] and keep
    it iff its upper bound reaches the lower bound of the most-confident region
    (best = argmax (z_hat_r - k se_r)): ucb_r >= lcb_best. That is the LUCB /
    racing rule -- keep every region whose interval still overlaps the
    leader's. The survivor count adapts per permutation rather than a fixed
    top-k. Inactive regions (size < min_vox, non-finite z_hat, or n < 2) are
    dropped; the interim arg-max-z region is always retained so the reported
    max-z is never trimmed.

    Args:
        llr_obs (np.array): (num_reg,) observed (unpermuted) region LLR
        mu (np.array): (num_reg,) burn-in inner-null mean
        std (np.array): (num_reg,) burn-in inner-null std
        n (np.array): (num_reg,) burn-in valid-sample count per region
        size (np.array): (num_reg,) voxel count per region
        min_vox (int): regions smaller than this are inactive
        p_keep_thresh (float): survivor keep-probability floor (e.g. 1e-6)

    Returns:
        keep (np.array): (num_reg,) bool survivor mask
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        z_hat = (llr_obs - mu) / std
        se = np.sqrt((1.0 + z_hat ** 2 / 2.0) / n)
    active = ((size >= min_vox) & np.isfinite(z_hat)
              & np.isfinite(se) & (n >= 2))
    keep = np.zeros_like(active)
    if not active.any():
        return keep
    k = float(-norm.ppf(p_keep_thresh))
    lcb = z_hat - k * se
    ucb = z_hat + k * se
    best = int(np.argmax(np.where(active, lcb, -np.inf)))
    keep = active & (ucb >= lcb[best])
    # always keep the reported max-z region (arg-max z_hat) so retention holds
    keep[int(np.argmax(np.where(active, z_hat, -np.inf)))] = True
    return keep


def _draw_survivors(kernels, local_idx, i_lo, i_hi, *, base_seed, num_img,
                    q0, q1, q0q0, eye, num_reg, min_vox, n, mean, M2):
    """Fold inner FL draws [i_lo, i_hi) for a subset of the kernel's survivors.

    local_idx selects rows of the survivor kernels (build_survivor_kernels
    output), so only those regions are drawn and every other region stays NaN
    in each draw (untouched by the Welford fold). Draws fold _TAIL_FOLD_CHUNK
    at a time rather than materialize; draw i uses seed base_seed + i.

    Args:
        kernels (dict): build_survivor_kernels output for the whole band
        local_idx (np.array): (m,) int rows of kernels to draw
        i_lo (int): first perm index to draw
        i_hi (int): stop before this perm index
        n (np.array): (num_reg,) running Welford valid-sample count
        mean (np.array): (num_reg,) running Welford mean
        M2 (np.array): (num_reg,) running Welford sum of squared deviations

    Returns:
        n, mean, M2 (np.array): the accumulators with these draws folded in
    """
    sub = {k: kernels[k][local_idx] for k in
           ('K', 'ysum_u_S', 'yout_u_S', 'size_S', 'survivor_idx')}
    buf = np.empty((_TAIL_FOLD_CHUNK, num_reg), dtype=np.float64)
    filled = 0
    for i in range(i_lo, i_hi):
        perm = permute._perm_indices(base_seed + i, num_img)
        fl = (eye - q0q0)[:, perm] + q0q0
        buf[filled] = glow.graph.compute_llr_inner_kernel(
            sub, q0, q1, fl, perm, num_reg, min_size=min_vox)
        filled += 1
        if filled == _TAIL_FOLD_CHUNK:
            n, mean, M2 = _welford_combine(buf, n, mean, M2)
            filled = 0
    if filled:
        n, mean, M2 = _welford_combine(buf[:filled], n, mean, M2)
    return n, mean, M2


def cpu_perm_race(*, exp, llr_obs, base_seed: int, n_perm: int, q0, q1,
                  children, min_vox: int,
                  n_perm_inner_race: int = N_PERM_INNER_RACE,
                  p_keep_thresh: float = RACE_P_KEEP_THRESH):
    """Compute inner-perm (mu, std) via a progressive survivor race.

    Burn-in draws n_perm_inner_race permutations over all regions on cpu_perm's
    streaming iter_llr_perm path, folding into one Welford / Chan accumulator
    (Chan, Golub & LeVeque 1979). The tail then draws the surviving band in
    geometrically growing rounds via the low-rank general-Q0 kernel
    (graph.build_survivor_kernels once, graph.compute_llr_inner_kernel per
    draw): _race_keep re-trims after each round to the regions that could
    still be the per-permutation max-z, so the field shrinks, and a region
    whose band re-opens (the interim leader's z drifts down) re-enters and
    closes the gap it sat out. Every active region is drawn to the round's
    frontier, so the eventual arg-max reaches the full n_perm with contiguous
    draws. Regions outside the burn-in band stay frozen at their burn-in
    moments. Draw i uses seed base_seed + i, exactly as cpu_perm, so a
    survivor's moments equal the full-run moments to float round-off. Valid
    for any nuisance design.

    The trim only costs power (dropping a region that would have won), never
    validity: the race is applied identically on every outer permutation, so
    the Westfall-Young / Freedman-Lane FWER bound is untouched (Lehmann &
    Romano Thm 15.2.1; Hemerik & Goeman 2018). The leader is always kept, so
    the max over survivors equals the whole-tree max whenever the true
    arg-max survives. n_perm_inner_race and p_keep_thresh are speed/power
    knobs, never validity knobs.

    Args match cpu_perm plus the observed region LLR (the trim's z_hat
    numerator) and the two race knobs.

    Args:
        exp (Experiment): experiment to sample inner perms from
        llr_obs (np.array): (num_reg,) observed region LLR, from
            compute_llr_batched on the (this-outer-perm) exp
        base_seed (int): draw i uses seed base_seed + i
        n_perm (int): number of inner FL draws
        q0 (np.array): (a0, num_img) nuisance subspace
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN
        n_perm_inner_race (int): burn-in draws over all regions before the trim
        p_keep_thresh (float): survivor keep-probability floor

    Returns:
        mu (np.array): (num_reg,) inner-null mean per region
        std (np.array): (num_reg,) inner-null std per region
    """
    num_vox = exp.y.shape[2]
    num_img = exp.y.shape[1]
    n_perm_inner_race = min(n_perm_inner_race, n_perm)
    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=num_vox)
    num_reg = int(region_l.shape[0])
    size = region_h - region_l

    # burn-in: n_perm_inner_race streamed draws over ALL regions, == cpu_perm
    perms = np.empty((n_perm_inner_race, num_img), dtype=np.int64)
    for i in range(n_perm_inner_race):
        perms[i] = permute._perm_indices(base_seed + i, num_img)
    n = np.zeros(num_reg, dtype=np.float64)
    mean = np.zeros(num_reg, dtype=np.float64)
    M2 = np.zeros(num_reg, dtype=np.float64)
    for chunk in glow.graph.iter_llr_perm(
            y=exp.y, q0=q0, q1=q1, perms=perms,
            leaf_ord=leaf_ord, region_l=region_l, region_h=region_h,
            min_size=min_vox):
        n, mean, M2 = _welford_combine(chunk, n, mean, M2)
    mu_bi, std_bi = _welford_finalize(n, mean, M2)

    # Initial band: survivors that could be the max-z at burn-in. Build their
    # FL kernels once ((I - Q0Q0^T)[:, perm] + Q0Q0^T needs no get_freed_lane
    # QR); re-admission stays within this band, regions outside it stay frozen
    # at their burn-in moments.
    keep = _race_keep(llr_obs=llr_obs, mu=mu_bi, std=std_bi, n=n, size=size,
                      min_vox=min_vox, p_keep_thresh=p_keep_thresh)
    band = np.where(keep)[0]
    kernels = glow.graph.build_survivor_kernels(exp.y, children, band, q0)
    q0q0 = q0.T @ q0
    eye = np.eye(num_img, dtype=q0q0.dtype)

    # Progressive race: draw the band in geometrically growing rounds and
    # re-trim after each, so the field shrinks (and a region whose band
    # re-opens, as the interim leader's z drifts down, re-enters). d is each
    # region's contiguous draw count; a round draws every active region from
    # its own d up to the frontier -- a continuously-active region just
    # advances, a re-admitted one also closes the gap it sat out -- so the
    # eventual max-z region always reaches the full n_perm.
    d = np.full(num_reg, n_perm_inner_race, dtype=np.int64)
    frontier = n_perm_inner_race
    while frontier < n_perm:
        new_frontier = min(2 * frontier, n_perm)
        mu, std = _welford_finalize(n, mean, M2)
        keep = _race_keep(llr_obs=llr_obs, mu=mu, std=std, n=n, size=size,
                          min_vox=min_vox, p_keep_thresh=p_keep_thresh)
        active = band[keep[band]]
        for d_val in np.unique(d[active]):
            grp = active[d[active] == d_val]
            local = np.searchsorted(band, grp)
            n, mean, M2 = _draw_survivors(
                kernels, local, int(d_val), new_frontier,
                base_seed=base_seed, num_img=num_img, q0=q0, q1=q1,
                q0q0=q0q0, eye=eye, num_reg=num_reg, min_vox=min_vox,
                n=n, mean=mean, M2=M2)
            d[grp] = new_frontier
        frontier = new_frontier
    return _welford_finalize(n, mean, M2)


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
