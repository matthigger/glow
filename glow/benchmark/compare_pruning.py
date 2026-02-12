"""Compare homogeneity-test pruning vs greedy pruning.

Runs AnalysisGLOW on a subset of vba_hcp and vba_wgn experiments,
then applies both pruning strategies to the same significance results
and compares F1 / sensitivity / specificity.

Usage:
    python -m glow.benchmark.compare_pruning [--n_seeds 10] [--n_perm 100]
"""
import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tabulate import tabulate

import glow
import glow.graph
import glow.mask
from glow.benchmark.paper_config import (
    ANALYSES, COMMON, make_config,
)
from glow.experiment.prune import prune_greedy


def _score_regions(reg_list, children, exp, effect):
    """Build a predicted mask from region list and score vs ground truth."""
    mask_pred = np.zeros(exp.mask_idx.shape, dtype=bool)
    for reg_idx in reg_list:
        label_map = glow.graph.get_label_map(
            reg_idx_list=[reg_idx],
            mask_idx=exp.mask_idx,
            children=children)
        mask_pred |= (label_map > -1)
    f1, sens, spec = glow.mask.get_score(
        mask_pred=mask_pred,
        mask_target=effect.mask,
        mask_active=exp.mask_idx > -1)
    return f1, sens, spec


def _run_one(config, seed, hotel_tr, glow_kwargs, source):
    """Run a single (seed, hotel_tr) experiment and return results dict."""
    exp, effect = config.get_exp_eff(seed=seed, hotel_tr=hotel_tr)

    # run GLOW analysis (uses homogeneity pruning internally)
    ana = glow.experiment.AnalysisGLOW(exp=exp, **glow_kwargs)

    children = ana.child_dict[0]
    num_vox = exp.y.shape[2]

    # --- method 1: homogeneity pruning (already done) ---
    homo_regs = [e.reg_idx for e in ana.effect_list]
    f1_h, sens_h, spec_h = _score_regions(
        homo_regs, children, exp, effect)

    # --- method 2: greedy pruning on same significance set ---
    greedy_regs = prune_greedy(
        sig_reg_list=ana.sig_reg_list,
        children=children,
        num_leaf=num_vox,
        stat_adj=ana.hotel_tr_adjusted[0, :])
    f1_g, sens_g, spec_g = _score_regions(
        greedy_regs, children, exp, effect)

    return {
        'source': source,
        'seed': seed,
        'hotel_tr': hotel_tr,
        'n_sig': len(ana.sig_reg_list),
        'n_homo': len(homo_regs),
        'n_greedy': len(greedy_regs),
        'f1_homo': f1_h,
        'sens_homo': sens_h,
        'spec_homo': spec_h,
        'f1_greedy': f1_g,
        'sens_greedy': sens_g,
        'spec_greedy': spec_g,
    }


def run_comparison(source, n_seeds, n_perm, hotel_tr_all, n_jobs):
    """Run comparison for a single source (wgn or hcp)."""
    glow_kwargs = ANALYSES['GLOW'].copy()
    glow_kwargs['n_perm'] = n_perm
    glow_kwargs['verbose'] = False
    # each experiment runs serial perms; parallelism is across experiments
    glow_kwargs['n_jobs_perm'] = 1

    # build config just to get exp/effect via get_exp_eff
    config = make_config(
        label=f'_compare_{source}',
        source=source,
        run_fnc=None,
        n_seed=n_seeds,
    )
    config.prep_exp_orig()

    # build job list
    jobs = [(seed, ht) for seed in range(n_seeds) for ht in hotel_tr_all]
    total = len(jobs)
    print(f'  {total} experiments, n_jobs={n_jobs}')

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_one)(config, seed, ht, glow_kwargs, source)
        for seed, ht in jobs
    )
    elapsed = time.time() - t0
    print(f'  done in {elapsed:.0f}s ({elapsed / total:.1f}s per experiment)')

    return pd.DataFrame(results)


def summarise(df):
    """Print a summary table grouped by (source, hotel_tr)."""
    groups = df.groupby(['source', 'hotel_tr']).agg(
        f1_homo=('f1_homo', 'mean'),
        f1_greedy=('f1_greedy', 'mean'),
        sens_homo=('sens_homo', 'mean'),
        sens_greedy=('sens_greedy', 'mean'),
        spec_homo=('spec_homo', 'mean'),
        spec_greedy=('spec_greedy', 'mean'),
        n_homo=('n_homo', 'mean'),
        n_greedy=('n_greedy', 'mean'),
    ).reset_index()

    # add delta columns
    groups['df1'] = groups['f1_greedy'] - groups['f1_homo']
    groups['dsens'] = groups['sens_greedy'] - groups['sens_homo']
    groups['dspec'] = groups['spec_greedy'] - groups['spec_homo']

    for source in groups['source'].unique():
        sub = groups[groups['source'] == source]
        print(f'\n===  {source.upper()}  (mean over seeds)  ===')
        tbl = sub[['hotel_tr',
                    'f1_homo', 'f1_greedy', 'df1',
                    'sens_homo', 'sens_greedy', 'dsens',
                    'spec_homo', 'spec_greedy', 'dspec',
                    'n_homo', 'n_greedy']].copy()
        tbl.columns = ['hotel_tr',
                        'F1_homo', 'F1_grdy', 'dF1',
                        'Sn_homo', 'Sn_grdy', 'dSn',
                        'Sp_homo', 'Sp_grdy', 'dSp',
                        '#homo', '#grdy']
        for c in tbl.columns:
            if c == 'hotel_tr':
                tbl[c] = tbl[c].map('{:.4f}'.format)
            elif c.startswith('#'):
                tbl[c] = tbl[c].map('{:.1f}'.format)
            else:
                tbl[c] = tbl[c].map('{:+.3f}'.format if c.startswith('d')
                                    else '{:.3f}'.format)
        print(tabulate(tbl, headers='keys', tablefmt='simple',
                        showindex=False))

    # overall summary
    print('\n===  OVERALL  ===')
    for source in df['source'].unique():
        sub = df[df['source'] == source]
        print(f'  {source.upper():4s}  '
              f'F1: homo={sub["f1_homo"].mean():.3f}  '
              f'greedy={sub["f1_greedy"].mean():.3f}  '
              f'delta={sub["f1_greedy"].mean() - sub["f1_homo"].mean():+.3f}')


def main():
    parser = argparse.ArgumentParser(
        description='Compare homogeneity vs greedy pruning')
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='seeds per hotel_tr value (default: 10)')
    parser.add_argument('--n_perm', type=int, default=100,
                        help='permutations for GLOW (default: 100)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='parallel jobs (default: -1 = all cores)')
    args = parser.parse_args()

    hotel_tr_all = COMMON['hotel_tr_all']

    print(f'Comparing pruning strategies: {args.n_seeds} seeds x '
          f'{len(hotel_tr_all)} hotel_tr values, n_perm={args.n_perm}')

    frames = []
    for source in ('wgn', 'hcp'):
        print(f'\n--- {source.upper()} ---')
        df = run_comparison(source, args.n_seeds, args.n_perm, hotel_tr_all,
                            args.n_jobs)
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    summarise(df_all)


if __name__ == '__main__':
    main()
