"""Validate the studentized size-adjustment vs the legacy mean-only path.

Three side-by-side comparisons under matched seeds (same data, same
permutation indices, only ``score_method`` differs):

    E1  Null FWER calibration on HCP and WGN.
        Expected: rejection rate at alpha=0.05 should sit at ~5% under
        z_score; mean_adj on HCP currently undercovers (~2-3%).

    E2  LLR catastrophic-failure rate on HCP at high effect.
        Expected: under mean_adj, ~7% of high-effect HCP seeds fail to
        detect (Dice=0).  Under z_score, this fraction should drop.

    E3  Detection power across an effect-strength sweep on HCP.
        Expected: z_score should not regress mean Dice anywhere; on
        the moderate-effect band where the LLR plateau hurts mean_adj,
        z_score should improve.

Usage::

    python -m test.validate_studentize                  # all 3
    python -m test.validate_studentize --skip e3        # quick(er)
    python -m test.validate_studentize --hcp-only       # skip WGN E1
    python -m test.validate_studentize --n-seed 5       # quick smoke

Tunables at the top of the file mirror the segment_*/null_* benchmark
configs at 1k-voxel scale so timings stay tractable for an interactive
test.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

import glow
from glow.analysis import AnalysisGLOW
from glow.benchmark.config import Config


warnings.filterwarnings('ignore', category=RuntimeWarning)
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', None)


# matched to segment_*/null_* configs but reduced for an interactive test
DEFAULTS = dict(
    crop_n_vox=1000,
    effect_perc=0.1,
    n_perm_fwer=200,
    n_perm_fwer_size_adjust=50,
    alpha_fwer=0.05,
)


def _make_config(source, **overrides):
    cfg_kw = {**DEFAULTS, **overrides}
    crop = cfg_kw['crop_n_vox']
    # WGN shape must enclose the crop: pick a cube that overshoots
    # the requested crop voxel count so ExtenterSphere(n_vox=crop) can
    # always grow a contiguous sphere within it.
    side = max(10, int(np.ceil((crop * 1.5) ** (1 / 3))))
    cfg = Config(
        source=source,
        crop_n_vox=cfg_kw['crop_n_vox'],
        effect_perc=cfg_kw['effect_perc'],
        hcp_feats=['fa', 'md'] if source == 'hcp' else None,
        wgn_a=2, wgn_b=2, wgn_num_img=100,
        wgn_shape=(side, side, side),
        n_seed=10,
        effect_llr_all=[0.03, 0.075, 0.119, 0.189],
        label=f'studentize_{source}',
    )
    return cfg


def _run_one(cfg, seed, effect_llr, score_method):
    """Run AnalysisGLOW once at fixed seed + effect_llr + mode."""
    exp_eff, effect = cfg.get_exp_eff(seed=seed, effect_llr=effect_llr)
    ana = AnalysisGLOW(
        exp=exp_eff,
        n_perm_fwer=DEFAULTS['n_perm_fwer'],
        n_perm_fwer_size_adjust=DEFAULTS['n_perm_fwer_size_adjust'],
        alpha_fwer=DEFAULTS['alpha_fwer'],
        n_jobs_perm=-1,
        score_method=score_method,
        verbose=False,
    )
    if effect.mask.any():
        dice, sens, spec = glow.graph.get_dice_sens_spec(
            mask=effect.mask, mask_idx=exp_eff.mask_idx,
            children=ana.children)
        # report best (oracle-Dice) over the discovered antichain only
        sel = [eff.reg_idx for eff in ana.effect_list]
        if sel:
            best = max(dice[r] for r in sel)
            best_sens = sens[sel[np.argmax([dice[r] for r in sel])]]
            best_spec = spec[sel[np.argmax([dice[r] for r in sel])]]
        else:
            best = best_sens = 0.0
            best_spec = 1.0
    else:
        best = best_sens = 0.0
        best_spec = 1.0
    return {
        'n_sig': len(ana.sig_reg_list),
        'n_eff': len(ana.effect_list),
        'best_dice': float(best),
        'sens': float(best_sens),
        'spec': float(best_spec),
        'min_pval': float(np.nanmin(ana.pval)) if np.any(np.isfinite(ana.pval)) else float('nan'),
    }


def e1_null(source, n_seed):
    """Rejection rate under H0 — should land at alpha=0.05 with z_score."""
    cfg = _make_config(source)
    rows = []
    for seed in range(n_seed):
        for sm in ('mean_adj', 'z_score'):
            t0 = time.time()
            r = _run_one(cfg, seed=seed, effect_llr=0.0, score_method=sm)
            r.update(score_method=sm, seed=seed, source=source,
                     elapsed=time.time() - t0)
            rows.append(r)
            print(f'  E1 {source} seed={seed} {sm:9s}  '
                  f'n_sig={r["n_sig"]:3d}  '
                  f'rejected={r["n_sig"]>0}  '
                  f'({r["elapsed"]:.1f}s)')
    df = pd.DataFrame(rows)
    summary = df.groupby('score_method').agg(
        rejected_frac=('n_sig', lambda s: float((s > 0).mean())),
        mean_n_sig=('n_sig', 'mean'),
        seeds=('seed', 'count'),
    )
    print(f'\nE1 {source}: null FWER calibration (alpha=0.05)')
    print(summary.to_string(float_format=lambda x: f'{x:.3f}'))
    return df


def e2_catastrophic(n_seed):
    """High-effect HCP: count seeds where each mode fails (Dice=0)."""
    cfg = _make_config('hcp')
    rows = []
    for seed in range(n_seed):
        for sm in ('mean_adj', 'z_score'):
            t0 = time.time()
            r = _run_one(cfg, seed=seed, effect_llr=0.119, score_method=sm)
            r.update(score_method=sm, seed=seed, effect_llr=0.119,
                     elapsed=time.time() - t0)
            rows.append(r)
            print(f'  E2 hcp seed={seed} {sm:9s}  '
                  f'n_sig={r["n_sig"]:3d}  dice={r["best_dice"]:.3f}  '
                  f'pval_min={r["min_pval"]:.4f}  '
                  f'({r["elapsed"]:.1f}s)')
    df = pd.DataFrame(rows)
    summary = df.groupby('score_method').agg(
        catastrophic_frac=('best_dice', lambda s: float((s == 0).mean())),
        mean_dice=('best_dice', 'mean'),
        median_dice=('best_dice', 'median'),
        mean_sens=('sens', 'mean'),
        seeds=('seed', 'count'),
    )
    print(f'\nE2 hcp catastrophic-failure rate at effect_llr=0.119')
    print(summary.to_string(float_format=lambda x: f'{x:.3f}'))
    return df


def e3_sweep(n_seed):
    """Effect-strength sweep on HCP: mean Dice by mode and effect_llr."""
    cfg = _make_config('hcp')
    llrs = cfg.effect_llr_all
    rows = []
    for seed in range(n_seed):
        for llr in llrs:
            for sm in ('mean_adj', 'z_score'):
                t0 = time.time()
                r = _run_one(cfg, seed=seed, effect_llr=llr,
                             score_method=sm)
                r.update(score_method=sm, seed=seed, effect_llr=llr,
                         elapsed=time.time() - t0)
                rows.append(r)
        print(f'  E3 seed={seed} done '
              f'({sum(rr["elapsed"] for rr in rows[-len(llrs)*2:]):.1f}s)')
    df = pd.DataFrame(rows)
    summary = df.groupby(['effect_llr', 'score_method'])['best_dice'].agg(
        ['mean', 'median']).round(3).unstack('score_method')
    print(f'\nE3 hcp detection sweep (mean / median best_dice in pruned set)')
    print(summary.to_string())
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n-seed', type=int, default=10)
    parser.add_argument('--skip', nargs='*', default=[],
                        choices=['e1', 'e2', 'e3'])
    parser.add_argument('--hcp-only', action='store_true',
                        help='skip the WGN E1 cell')
    parser.add_argument('--out', type=str, default=None,
                        help='dump per-row CSV(s) to this directory')
    parser.add_argument('--crop-n-vox', type=int, default=None,
                        help='override crop_n_vox (default 1000)')
    parser.add_argument('--n-perm', type=int, default=None,
                        help='override n_perm_fwer (default 200)')
    parser.add_argument('--n-perm-fit', type=int, default=None,
                        help='override n_perm_fwer_size_adjust (default 50)')
    args = parser.parse_args()

    if args.crop_n_vox is not None:
        DEFAULTS['crop_n_vox'] = args.crop_n_vox
    if args.n_perm is not None:
        DEFAULTS['n_perm_fwer'] = args.n_perm
    if args.n_perm_fit is not None:
        DEFAULTS['n_perm_fwer_size_adjust'] = args.n_perm_fit

    print(f'studentize validation @ {DEFAULTS}')
    print(f'n_seed={args.n_seed}; skipping {args.skip}\n')

    bundles = {}
    t_total = time.time()

    if 'e1' not in args.skip:
        bundles['e1_hcp'] = e1_null('hcp', args.n_seed)
        if not args.hcp_only:
            bundles['e1_wgn'] = e1_null('wgn', args.n_seed)
    if 'e2' not in args.skip:
        bundles['e2_hcp'] = e2_catastrophic(args.n_seed)
    if 'e3' not in args.skip:
        bundles['e3_hcp'] = e3_sweep(args.n_seed)

    if args.out:
        import pathlib
        out = pathlib.Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in bundles.items():
            df.to_csv(out / f'{name}.csv', index=False)
        print(f'\nCSVs saved to {out}/')

    total = time.time() - t_total
    print(f'\ntotal wall time: {total:.0f}s ({total/60:.1f} min)')


if __name__ == '__main__':
    main()
