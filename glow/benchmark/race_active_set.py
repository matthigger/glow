"""Race active-set benchmark: trace how the inner-perm comparison set
collapses as the race runs, and plot it.

Usage:
    python -m glow.benchmark.race_active_set
    python -m glow.benchmark.race_active_set --effect-llr 0.03 --n-outer 10
    python -m glow.benchmark.race_active_set --n-inner-max 400 --k-sigma 2.0

Produces (under <user_data_dir>/glow/results/race_active_set/):
    race_active_set.png   trajectory plot (n_active vs inner perm)
    Phase A / Phase B region-sample totals on stdout.
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from platformdirs import user_data_dir

import glow
from glow.analysis.cluster import cluster
from glow.analysis.mancova import decompose, get_llr, is_intercept_only_nuisance
from glow.benchmark.hcp_data import get_hcp_path


def default_out_dir():
    return Path(user_data_dir('glow', 'glow_author')) / 'results' / 'race_active_set'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--target-vox', type=int, default=25_000)
    p.add_argument('--n-outer', type=int, default=10)
    p.add_argument('--n-inner-max', type=int, default=200,
                   help='Hard ceiling on inner perms per outer perm.')
    p.add_argument('--effect-llr', type=float, default=0.0,
                   help='0 = null HCP; >0 = imposed effect.')
    p.add_argument('--effect-perc', type=float, default=0.10)
    p.add_argument('--min-vox', type=int, default=4)
    p.add_argument('--race-init', type=int, default=50)
    p.add_argument('--race-batch', type=int, default=25)
    p.add_argument('--k-sigma', type=float, default=3.0)
    p.add_argument('--z-threshold', type=float, default=None,
                   help='If set, switch to threshold-race mode: drop '
                        'regions whose ±k*SE band commits to a side of '
                        'this z value.  Loop ends when no region is still '
                        'ambiguous.  Implies --n-outer 1 and runs against '
                        'unpermuted data (perm_idx=0).')
    p.add_argument('--out-dir', type=Path, default=default_out_dir(),
                   help='Directory to write the plot into.')
    args = p.parse_args()
    if args.z_threshold is not None and args.n_outer != 1:
        print(f'note: --z-threshold given, forcing --n-outer 1 (unpermuted).')
        args.n_outer = 1
    return args


def build_exp(args):
    path = get_hcp_path()
    exp = glow.experiment.ExperimentImageOnly.from_search(
        folder=path,
        sbj_regex=r'[\d]{6}',
        img_glob_dict={'fa': '*_fa.nii.gz', 'md': '*_md.nii.gz'})
    exp = exp.sample_x(a=2, seed=0, add_bias=True)
    extenter = glow.effect.ExtenterSphere(n_vox=args.target_vox, connected=True)
    mask = extenter(mask_idx=exp.mask_idx, seed=0, contiguous=True)
    exp = exp.apply_mask(mask)
    exp = glow.experiment.ExperimentScaled.from_exp(exp)
    if args.effect_llr > 0:
        n_eff = max(1, int(exp.y.shape[2] * args.effect_perc))
        eff_ext = glow.effect.ExtenterSphere(n_vox=n_eff)
        exp, _ = glow.effect.EffectSynthetic.impose(
            exp, effect_llr=args.effect_llr, extenter=eff_ext, seed=0)
    return exp


def race_with_log(draw_one, n_max, llr_outer, size, min_vox,
                  race_init, race_batch, race_k_sigma,
                  z_threshold=None):
    """Instrumented Welford copy of glow.analysis._glow._run_inner_race.

    Returns a profiling bundle:
        n_inner_used : int — total inner perms run
        active_per_perm : (n_inner_used,) int — n_active during each draw
        batches : list of dicts, one per batch (warmup + each pruning batch):
            n_run_start, n_in_batch, n_active_during, t_draw, t_check

    Mirrors both modes of _run_inner_race (leader race when
    z_threshold is None; threshold race otherwise).
    """
    from time import perf_counter
    num_reg = llr_outer.shape[0]
    mean = np.zeros(num_reg, dtype=float)
    M2 = np.zeros(num_reg, dtype=float)
    n_per_reg = np.zeros(num_reg, dtype=np.int64)

    def update(x):
        finite = np.isfinite(x)
        n_per_reg[finite] += 1
        delta = np.empty_like(mean)
        delta[finite] = x[finite] - mean[finite]
        mean[finite] += delta[finite] / n_per_reg[finite]
        delta2 = np.empty_like(mean)
        delta2[finite] = x[finite] - mean[finite]
        M2[finite] += delta[finite] * delta2[finite]

    def current_z_se():
        with np.errstate(invalid='ignore', divide='ignore'):
            var = np.where(n_per_reg >= 2, M2 / (n_per_reg - 1), np.nan)
        sigma = np.sqrt(var)
        sigma_safe = np.where(sigma < 1e-12, 1.0, sigma)
        z = (llr_outer - mean) / sigma_safe
        se = np.sqrt((1 + z ** 2 / 2) / np.maximum(n_per_reg, 1))
        return z, sigma, se

    active = (size >= min_vox).copy()
    n_active_initial = int(active.sum())

    batches = []

    # Warmup batch (no race-check before it)
    init_n = min(race_init, n_max)
    t0 = perf_counter()
    for i in range(init_n):
        update(draw_one(i))
    t_draw_warm = perf_counter() - t0
    batches.append({'n_run_start': 0, 'n_in_batch': init_n,
                    'n_active_during': n_active_initial,
                    't_draw': t_draw_warm, 't_check': 0.0})
    n_run = init_n
    active_per_perm = [n_active_initial] * n_run

    while n_run < n_max:
        t_c0 = perf_counter()
        z, _, se = current_z_se()
        eligible = active & np.isfinite(z)
        stop = False

        if z_threshold is None:
            if eligible.sum() <= 1:
                stop = True
            else:
                z_for_max = np.where(eligible, z, -np.inf)
                leader = int(np.argmax(z_for_max))
                leader_lower = z[leader] - race_k_sigma * se[leader]
                upper = z + race_k_sigma * se
                cant_catch = eligible & (upper < leader_lower)
                cant_catch[leader] = False
                if cant_catch.any():
                    active &= ~cant_catch
                    if (active & np.isfinite(z)).sum() <= 1:
                        stop = True
        else:
            if eligible.sum() == 0:
                stop = True
            else:
                upper = z + race_k_sigma * se
                lower = z - race_k_sigma * se
                decided = eligible & ((upper < z_threshold) |
                                      (lower > z_threshold))
                if decided.any():
                    active &= ~decided
                    if (active & np.isfinite(z)).sum() == 0:
                        stop = True
        t_check = perf_counter() - t_c0

        if stop:
            # Account for the check in the previous batch (no draws followed).
            if batches:
                batches[-1]['t_check'] += t_check
            break

        n_batch = min(race_batch, n_max - n_run)
        n_active_during = int(active.sum())
        t_d0 = perf_counter()
        for i in range(n_run, n_run + n_batch):
            update(draw_one(i))
        t_draw = perf_counter() - t_d0
        batches.append({'n_run_start': n_run, 'n_in_batch': n_batch,
                        'n_active_during': n_active_during,
                        't_draw': t_draw, 't_check': t_check})
        active_per_perm.extend([n_active_during] * n_batch)
        n_run += n_batch

    return {
        'n_inner_used': n_run,
        'active_per_perm': active_per_perm,
        'batches': batches,
    }


def run_perm(args, exp, q0, q1, perm_idx):
    """One outer perm: build draw_one (fast path), run race, return trace."""
    _exp = exp.permute(perm_idx)
    children = cluster(exp=_exp, mode='q1')
    num_vox = _exp.y.shape[2]
    layer = glow.graph.compute_tree_layers(children, num_vox)
    llr_outer, size = glow.graph.compute_llr_batched(
        _exp, children=children, q0=q0, q1=q1, layer=layer)
    del _exp

    dtype = exp.y.dtype if exp.y.dtype == np.float32 else np.float64
    ysum_u, yout_u, _ = glow.graph.compute_phase1(exp.y, children, layer=layer)
    sz_3d = size.astype(dtype)[:, None, None]
    a0 = np.einsum('rbn,an->rba', ysum_u, q0, optimize=True)
    t_u = (yout_u - np.einsum('rba,rca->rbc', a0, a0, optimize=True) / sz_3d)
    del a0
    n_img = exp.y.shape[1]
    base = (perm_idx + 1) * 100_000

    def draw_one(i):
        rng = np.random.default_rng(base + i)
        perm = np.argsort(rng.permutation(n_img))
        q1_T_perm = q1.T[perm, :].astype(dtype, copy=False)
        llr_i, _ = glow.graph.compute_llr_inner_fast(
            t_u, ysum_u, size, q1_T_perm, min_size=args.min_vox)
        return llr_i

    t0 = time.time()
    race_out = race_with_log(
        draw_one, args.n_inner_max, llr_outer, size, args.min_vox,
        args.race_init, args.race_batch, args.k_sigma,
        z_threshold=args.z_threshold)
    elapsed = time.time() - t0
    return {
        'perm_idx': perm_idx,
        'num_reg': int(llr_outer.shape[0]),
        'n_used': int(race_out['n_inner_used']),
        'active_per_perm': np.asarray(race_out['active_per_perm'], dtype=int),
        'batches': race_out['batches'],
        'elapsed': elapsed,
    }


def make_plot(args, traces, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(traces)))
    for trace, color in zip(traces, cmap):
        x = np.arange(1, len(trace['active_per_perm']) + 1)
        ax.plot(x, trace['active_per_perm'], color=color, alpha=0.85,
                lw=1.3, label=f'perm {trace["perm_idx"]}')

    num_reg = traces[0]['num_reg']
    ax.axhline(num_reg, color='grey', ls=':', lw=0.8,
               label=f'total regions in tree ({num_reg:,})')
    ax.axvline(args.n_inner_max, color='red', ls='--', lw=1.0,
               label=f'budget: n_inner_max={args.n_inner_max}')

    if args.z_threshold is None:
        ax.axhline(1, color='black', ls=':', lw=0.6,
                   label='leader alone (stop condition)')
        mode_label = 'leader race'
    else:
        ax.axhline(0.5, color='black', ls=':', lw=0.6,
                   label='all regions decided (stop condition)')
        mode_label = f'threshold race @ z={args.z_threshold}'

    ax.set_yscale('log')
    ax.set_xlabel('inner permutation index')
    ax.set_ylabel('# regions still active (log)')
    eff_label = 'null (H0)' if args.effect_llr == 0 else f'LLR={args.effect_llr}'
    ax.set_title(f'race active-set trajectory — HCP fa+md '
                 f'{args.target_vox:,} vox, {eff_label}, {mode_label}\n'
                 f'k_σ={args.k_sigma}, init={args.race_init}, '
                 f'batch={args.race_batch}')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f'\nplot saved -> {out_path}')


def main():
    args = parse_args()
    eff_label = 'null (H0)' if args.effect_llr == 0 else f'LLR={args.effect_llr}'
    print(f'config: HCP fa+md  vox={args.target_vox:,}  effect={eff_label}  '
          f'n_outer={args.n_outer}  n_inner_max={args.n_inner_max}  '
          f'k_σ={args.k_sigma}  init={args.race_init}  batch={args.race_batch}')

    print('\nbuild exp...')
    exp = build_exp(args)
    print(f'  y.shape={exp.y.shape}')
    q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)
    assert is_intercept_only_nuisance(exp.x, exp.contrast), \
        'this benchmark assumes the fast path applies'

    traces = []
    for i in range(args.n_outer):
        t = run_perm(args, exp, q0, q1, i)
        traces.append(t)
        rs_A = t['n_used'] * t['num_reg']
        rs_B = int(t['active_per_perm'].sum())
        rs_flat = args.n_inner_max * t['num_reg']
        print(f'  perm {i}: {t["elapsed"]:5.2f}s  '
              f'n_inner_used={t["n_used"]:>3}/{args.n_inner_max}  '
              f'n_reg={t["num_reg"]:>5}  '
              f'active 1st->last: {t["active_per_perm"][0]:>5}'
              f'->{t["active_per_perm"][-1]:>5}  '
              f'rs_A={rs_A:>10,} ({(1-rs_A/rs_flat)*100:+5.1f}%)  '
              f'rs_B={rs_B:>10,} ({(1-rs_B/rs_flat)*100:+5.1f}%)')

    print_profile_first_perm(traces[0])
    print_timing_summary(args, traces)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ('threshold' if args.z_threshold is not None else 'leader')
    out_path = args.out_dir / f'race_active_set_{suffix}.png'
    make_plot(args, traces, out_path)


def print_profile_first_perm(trace):
    print(f'\n=== per-batch profile (outer perm {trace["perm_idx"]}) ===')
    print(f'  {"#":>3} {"n_run":>10} {"n_in":>5} {"n_active":>8} '
          f'{"t_draw":>8} {"t_check":>8} {"draw_per_perm":>14}')
    for j, b in enumerate(trace['batches']):
        per_perm = b['t_draw'] / max(b['n_in_batch'], 1)
        rng = f'{b["n_run_start"]:>4}-{b["n_run_start"]+b["n_in_batch"]-1:<4}'
        print(f'  {j:>3} {rng:>10} {b["n_in_batch"]:>5} '
              f'{b["n_active_during"]:>8} {b["t_draw"]*1000:>7.1f}ms '
              f'{b["t_check"]*1000:>7.1f}ms '
              f'{per_perm*1000:>13.2f}ms')


def print_timing_summary(args, traces):
    flat_rs = sum(args.n_inner_max * t['num_reg'] for t in traces)
    rs_A_tot = sum(t['n_used'] * t['num_reg'] for t in traces)
    rs_B_tot = sum(int(t['active_per_perm'].sum()) for t in traces)
    wall = sum(t['elapsed'] for t in traces)

    # Per-perm draw cost averaged across all batches/perms.
    total_draws = sum(sum(b['n_in_batch'] for b in t['batches'])
                      for t in traces)
    total_t_draw = sum(sum(b['t_draw'] for b in t['batches'])
                       for t in traces)
    total_t_check = sum(sum(b['t_check'] for b in t['batches'])
                        for t in traces)
    per_perm_draw = total_t_draw / max(total_draws, 1)
    flat_200_wall = 200 * per_perm_draw * args.n_outer
    # Phase-B projected: each batch's draw time amortised by n_active/num_reg
    phase_b_wall = 0.0
    for t in traces:
        for b in t['batches']:
            phase_b_wall += (b['t_draw'] * b['n_active_during']
                              / t['num_reg'])

    print(f'\n=== totals across {args.n_outer} outer perms ===')
    print(f'  wall time (race)        : {wall:>7.2f}s  '
          f'(draw {total_t_draw:.2f}s + check {total_t_check:.2f}s)')
    print(f'  mean n_inner_used       : '
          f'{np.mean([t["n_used"] for t in traces]):.1f} / '
          f'{args.n_inner_max}')
    print(f'  per-perm draw cost      : {per_perm_draw*1000:>7.2f}ms '
          f'(avg across batches; ~constant in Phase A)')
    print(f'\n  flat n=200 (current default), projected: '
          f'{flat_200_wall:>7.2f}s  '
          f'= 200 perms × {per_perm_draw*1000:.2f}ms × {args.n_outer} outer')
    print(f'  race n_inner_max={args.n_inner_max:<5} (Phase A)         : '
          f'{wall:>7.2f}s  '
          f'= ~{wall/flat_200_wall:.1f}× flat-200')
    print(f'  race n_inner_max={args.n_inner_max:<5} (Phase B proj.)   : '
          f'{phase_b_wall:>7.2f}s  '
          f'= ~{phase_b_wall/flat_200_wall:.2f}× flat-200  '
          f'[needs kernel region_mask]')
    print(f'\n  region-samples flat     : {flat_rs:>14,}')
    print(f'  region-samples A        : {rs_A_tot:>14,}  '
          f'({(1 - rs_A_tot/flat_rs)*100:+5.1f}%)')
    print(f'  region-samples B        : {rs_B_tot:>14,}  '
          f'({(1 - rs_B_tot/flat_rs)*100:+5.1f}%)  '
          f'[ceiling; needs kernel region_mask]')


if __name__ == '__main__':
    main()
