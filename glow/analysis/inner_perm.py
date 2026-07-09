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
    regions on cpu_perm's streaming kernel, trims to the survivors that
    could still be the per-permutation max-z region, then draws the
    remaining perms for survivors only via the low-rank general-Q0
    kernel (glow.graph.compute_llr_inner_kernel). Same draws as cpu_perm
    per seed, so survivor moments match to float round-off; the trim
    only costs power, never validity (see cpu_perm_race).
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
    """Return the survivor mask over regions after the race burn-in.

    Models each region's true standardised score as z_r ~ N(z_hat_r,
    se_r^2), with the delta-method standard error
    se_r = sqrt((1 + z_hat_r^2 / 2) / n_r) -- the mean contributes 1/n, the
    std estimate z_hat^2 / 2n. A region is kept iff it still has more than
    p_keep_thresh probability of beating the interim leader L = argmax z_hat,
    i.e. (z_hat_r - z_hat_L) / sqrt(se_r^2 + se_L^2) > Phi^{-1}(p_keep_thresh).
    Inactive regions (size < min_vox, non-finite z_hat, or n < 2) are dropped;
    the leader is always retained.

    Writing p_keep_thresh = Phi(-k_sigma) makes this a scale-free k_sigma
    band, so the survivor count adapts per permutation rather than a fixed
    top-k.

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
    leader = int(np.argmax(np.where(active, z_hat, -np.inf)))
    thresh = float(norm.ppf(p_keep_thresh))
    with np.errstate(invalid='ignore'):
        score = (z_hat - z_hat[leader]) / np.sqrt(se ** 2 + se[leader] ** 2)
    keep = active & (score > thresh)
    keep[leader] = True
    return keep


def cpu_perm_race(*, exp, llr_obs, base_seed: int, n_perm: int, q0, q1,
                  children, min_vox: int, race_init: int = 15,
                  p_keep_thresh: float = 1e-6):
    """Compute inner-perm (mu, std) via the burn-in / trim / tail race.

    Three stages, all folding into one Welford / Chan accumulator (Chan,
    Golub & LeVeque 1979): burn-in draws race_init permutations over all
    regions on cpu_perm's streaming iter_llr_perm path; _race_keep trims to
    the survivors that could still be the per-permutation max-z region; the
    tail draws the remaining n_perm - race_init permutations for survivors
    only via the low-rank general-Q0 kernel (graph.build_survivor_kernels
    once, graph.compute_llr_inner_kernel per draw). Non-survivors stay frozen
    at their burn-in moments. Draw i uses seed base_seed + i, exactly as
    cpu_perm, so survivor moments equal the full-run moments to float
    round-off. Valid for any nuisance design.

    The trim only costs power (dropping a region that would have won), never
    validity: the race is applied identically on every outer permutation, so
    the Westfall-Young / Freedman-Lane FWER bound is untouched (Lehmann &
    Romano Thm 15.2.1; Hemerik & Goeman 2018). The leader is always kept, so
    the max over survivors equals the whole-tree max whenever the true
    arg-max survives, and discoveries are restricted to the survivor set.
    race_init and p_keep_thresh are speed/power knobs, never validity knobs.

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
        race_init (int): burn-in draws over all regions before the trim
        p_keep_thresh (float): survivor keep-probability floor

    Returns:
        mu (np.array): (num_reg,) inner-null mean per region
        std (np.array): (num_reg,) inner-null std per region
    """
    num_vox = exp.y.shape[2]
    num_img = exp.y.shape[1]
    race_init = min(race_init, n_perm)
    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=num_vox)
    num_reg = int(region_l.shape[0])
    size = region_h - region_l

    # burn-in: race_init streamed draws over ALL regions (== cpu_perm draws)
    perms = np.empty((race_init, num_img), dtype=np.int64)
    for i in range(race_init):
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

    # trim to survivors
    keep = _race_keep(llr_obs=llr_obs, mu=mu_bi, std=std_bi, n=n, size=size,
                      min_vox=min_vox, p_keep_thresh=p_keep_thresh)
    survivor_idx = np.where(keep)[0]

    # tail: kernel draws for survivors only, into the same accumulator. Build
    # the FL matrix inline from the cached nuisance projector -- get_freed_lane
    # re-runs decompose (a QR) per call; (I - Q0Q0^T)[:, perm] + Q0Q0^T is
    # identical and needs no seed-0 guard (it never calls get_freed_lane).
    kernels = glow.graph.build_survivor_kernels(
        exp.y, children, survivor_idx, q0)
    q0q0 = q0.T @ q0
    eye = np.eye(num_img, dtype=q0q0.dtype)
    tail = np.empty((n_perm - race_init, num_reg), dtype=np.float64)
    for j, i in enumerate(range(race_init, n_perm)):
        perm = permute._perm_indices(base_seed + i, num_img)
        fl = (eye - q0q0)[:, perm] + q0q0
        tail[j] = glow.graph.compute_llr_inner_kernel(
            kernels, q0, q1, fl, perm, num_reg, min_size=min_vox)
    n, mean, M2 = _welford_combine(tail, n, mean, M2)
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
