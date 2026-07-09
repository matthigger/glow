"""Prototype inner-perm race, wired alongside glow (glow itself untouched).

The race accelerates the inner Freedman-Lane loop by spending a short burn-in
over all regions, trimming to the regions that could still be the
per-permutation max-z (survivors), then drawing the remaining permutations for
survivors only via the low-rank general-Q0 kernel (glow.graph. build_survivor
_kernels / compute_llr_inner_kernel). Non-survivors stay frozen at their
burn-in moments. Valid for ANY nuisance design; the FWER bound is untouched
because the same procedure runs on every outer permutation (race_init and
p_keep_thresh are speed/power knobs, never validity knobs).

cpu_perm_race mirrors glow.analysis.inner_perm.cpu_perm's keyword-only
signature (plus race_init, p_keep_thresh, and an observed-LLR arg the trim
needs), so it drops into inner_perm.py verbatim on the next commit. It returns
(mu, std, info); the drop-in discards info.

run_experiment() compares the race against the full streaming cpu_perm at
num_vox=1024, n_perm_inner=1000, on a general-Q0 (non-intercept) design with a
planted effect, and reports the speedup plus that the max-z is unchanged.
"""
import time

import numpy as np
from scipy.stats import norm

import glow.graph
from glow.analysis import inner_perm
from glow.analysis.inner_perm import _welford_combine, _welford_finalize
from glow.analysis.mancova import decompose
from glow.analysis.cluster import cluster, ClusterMode
from glow.experiment.exper import Experiment
from glow.experiment import permute


def _race_keep(*, llr_obs, mu, std, n, size, min_vox, p_keep_thresh):
    """Return the boolean survivor mask over regions after burn-in.

    Models each region's true standardised score as
    z_r ~ N(z_hat_r, se_r^2) with the delta-method se_r =
    sqrt((1 + z_hat_r^2 / 2) / n_r) (mean contributes 1/n, the std estimate
    z_hat^2 / 2n). A region is kept iff it still has more than p_keep_thresh
    probability of beating the interim leader L = argmax z_hat, i.e.
    (z_hat_r - z_hat_L) / sqrt(se_r^2 + se_L^2) > Phi^{-1}(p_keep_thresh).
    Inactive regions (size < min_vox, non-finite z_hat, or n < 2) are dropped;
    the leader is always retained.

    Args:
        llr_obs (np.array): (num_reg,) observed (unpermuted) region LLR
        mu (np.array): (num_reg,) burn-in inner-null mean
        std (np.array): (num_reg,) burn-in inner-null std
        n (np.array): (num_reg,) burn-in valid-sample count per region
        size (np.array): (num_reg,) voxel count per region
        min_vox (int): regions smaller than this are inactive
        p_keep_thresh (float): keep-probability floor (e.g. 1e-6)

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


def cpu_perm_race(*, exp, llr_obs, base_seed, n_perm, q0, q1, children,
                  min_vox, race_init=15, p_keep_thresh=1e-6):
    """Compute inner-perm (mu, std) via the burn-in / trim / tail race.

    Burn-in rides glow's streaming iter_llr_perm (identical draws to cpu_perm);
    the tail rides the low-rank general-Q0 kernel over survivors only. Both
    stages fold into the shared Welford / Chan accumulator, and draw i uses the
    same seed base_seed + i as cpu_perm, so survivor moments equal the full-run
    moments to float round-off.

    Args match inner_perm.cpu_perm plus llr_obs (the trim's z_hat numerator),
    race_init, and p_keep_thresh. Returns (mu, std, info); the glow drop-in
    returns just (mu, std).

    Args:
        exp (Experiment): experiment to sample inner perms from
        llr_obs (np.array): (num_reg,) observed region LLR (from
            compute_llr_batched on the unpermuted exp)
        base_seed (int): draw i uses seed base_seed + i (>= 1; seed 0 is
            reserved for unpermuted data by get_freed_lane)
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
        info (dict): {survivor_idx, n_survivors, n_active, leader}
    """
    assert base_seed >= 1, 'base_seed must be >= 1 (seed 0 reserved)'
    num_vox = exp.y.shape[2]
    num_img = exp.y.shape[1]
    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=num_vox)
    num_reg = int(region_l.shape[0])
    size = region_h - region_l

    # ---- burn-in: race_init streamed draws over ALL regions ----
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

    # ---- trim ----
    keep = _race_keep(llr_obs=llr_obs, mu=mu_bi, std=std_bi, n=n, size=size,
                      min_vox=min_vox, p_keep_thresh=p_keep_thresh)
    survivor_idx = np.where(keep)[0]
    leader = int(np.argmax(np.where(keep, (llr_obs - mu_bi) / std_bi,
                                    -np.inf)))

    # ---- tail: kernel draws for survivors only, same accumulator ----
    kernels = glow.graph.build_survivor_kernels(
        exp.y, children, survivor_idx, q0)
    # Build the FL matrix inline from the cached nuisance projector -- the
    # get_freed_lane helper re-runs decompose (a QR) on every call, which
    # would dominate the tail; (I - Q0Q0^T)[:, perm] + Q0Q0^T is identical.
    q0q0 = q0.T @ q0
    eye = np.eye(num_img, dtype=q0q0.dtype)
    tail = np.empty((n_perm - race_init, num_reg), dtype=np.float64)
    for j, i in enumerate(range(race_init, n_perm)):
        seed = base_seed + i
        perm = permute._perm_indices(seed, num_img)
        fl = (eye - q0q0)[:, perm] + q0q0
        tail[j] = glow.graph.compute_llr_inner_kernel(
            kernels, q0, q1, fl, perm, num_reg, min_size=min_vox)
    n, mean, M2 = _welford_combine(tail, n, mean, M2)
    mu, std = _welford_finalize(n, mean, M2)

    active = (size >= min_vox)
    info = dict(survivor_idx=survivor_idx, n_survivors=int(survivor_idx.size),
                n_active=int(active.sum()), leader=leader)
    return mu, std, info


def _build_exp(*, num_img=40, b=2, num_vox=1024, seed=0,
               effect_beta=1.0, effect_slice=slice(480, 544)):
    """Build a general-Q0 (non-intercept) 1D experiment with a planted effect.

    Design: bias + one non-constant nuisance regressor + one interest
    regressor (so is_intercept_only_nuisance is False and the general kernel
    engages). A contiguous voxel block gets a mean shift along the interest
    regressor on both features, creating a clear max-z region.
    """
    rng = np.random.default_rng(seed)
    x = np.empty((3, num_img))
    x[0] = 1.0
    x[1] = rng.standard_normal(num_img)
    x[2] = rng.standard_normal(num_img)
    contrast = np.array([False, False, True])

    y = rng.standard_normal((b, num_img, num_vox))
    y[:, :, effect_slice] += effect_beta * x[2][None, :, None]

    mask_idx = np.arange(num_vox).reshape(1, 1, num_vox)
    exp = Experiment(x=x, y=y, contrast=contrast, mask_idx=mask_idx,
                     add_bias=False)
    return exp


def run_experiment(*, n_perm=1000, race_init=15, p_keep_thresh=1e-6,
                   min_vox=4, base_seed=1):
    """Compare the race vs full cpu_perm at 1024 voxels, n_perm_inner=1000."""
    exp = _build_exp()
    from glow.analysis.mancova import is_intercept_only_nuisance
    children = cluster(exp=exp, mode=ClusterMode.FOCUS)
    q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)
    llr_obs, size = glow.graph.compute_llr_batched(
        exp, children=children, q0=q0, q1=q1, min_size=min_vox)
    active = (size >= min_vox) & np.isfinite(llr_obs)

    print('=== inner-perm race vs full cpu_perm ===')
    print(f'num_vox={exp.y.shape[2]}  num_img={exp.y.shape[1]}  '
          f'b={exp.y.shape[0]}  num_reg={size.size}  active={int(active.sum())}')
    print(f'intercept_only_nuisance={is_intercept_only_nuisance(exp.x, exp.contrast)}'
          f'  (False => general kernel exercised)')
    print(f'n_perm_inner={n_perm}  race_init={race_init}  '
          f'p_keep_thresh={p_keep_thresh}  min_vox={min_vox}')

    full_fn = lambda: inner_perm.cpu_perm(
        exp=exp, base_seed=base_seed, n_perm=n_perm,
        q0=q0, q1=q1, children=children, min_vox=min_vox)
    race_fn = lambda: cpu_perm_race(
        exp=exp, llr_obs=llr_obs, base_seed=base_seed, n_perm=n_perm,
        q0=q0, q1=q1, children=children, min_vox=min_vox,
        race_init=race_init, p_keep_thresh=p_keep_thresh)

    def _best_time(fn, n_rep=3):
        fn()  # warmup (BLAS / cache), then report the fastest of n_rep
        best = float('inf')
        for _ in range(n_rep):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    # results are deterministic; compute once for correctness, time separately
    mu_f, std_f = full_fn()
    mu_r, std_r, info = race_fn()
    t_full = _best_time(full_fn)
    t_race = _best_time(race_fn)

    with np.errstate(divide='ignore', invalid='ignore'):
        z_f = (llr_obs - mu_f) / std_f
        z_r = (llr_obs - mu_r) / std_r
    valid = active & np.isfinite(z_f) & np.isfinite(z_r)

    argmax_full = int(np.argmax(np.where(valid, z_f, -np.inf)))
    maxz_full = float(z_f[argmax_full])
    maxz_race = float(np.nanmax(np.where(valid, z_r, np.nan)))
    argmax_survived = argmax_full in set(info['survivor_idx'].tolist())
    z_argmax_reldiff = abs(z_r[argmax_full] - z_f[argmax_full]) / abs(maxz_full)
    maxz_reldiff = abs(maxz_race - maxz_full) / abs(maxz_full)

    print('\n--- timing ---')
    print(f'full cpu_perm : {t_full:7.3f} s')
    print(f'race          : {t_race:7.3f} s')
    print(f'speedup       : {t_full / t_race:6.2f}x')
    print('\n--- survivors ---')
    print(f'survivors     : {info["n_survivors"]} / {info["n_active"]} active '
          f'({100.0 * info["n_survivors"] / info["n_active"]:.1f}%)')
    print('\n--- max-z (z = (llr_obs - mu) / std) ---')
    print(f'full  max-z   : {maxz_full:.6f}  at region {argmax_full} '
          f'(size {int(size[argmax_full])})')
    print(f'race  max-z   : {maxz_race:.6f}')
    print(f'argmax region survived the trim : {argmax_survived}')
    print(f'race z at full argmax vs full   : rel-diff {z_argmax_reldiff:.2e}')
    print(f'max-z agreement                 : rel-diff {maxz_reldiff:.2e}')

    ok = argmax_survived and maxz_reldiff < 1e-3 and t_race < t_full
    print(f'\nRESULT: {"PASS" if ok else "FAIL"} '
          f'(argmax survived, max-z unchanged, race faster)')
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if run_experiment() else 1)
