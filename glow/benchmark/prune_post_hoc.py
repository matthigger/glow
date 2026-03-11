"""Post-hoc pruning inspection: compare all pruning methods.

Usage::

    # list experiments sorted by reference method's advantage
    python -m glow.benchmark.prune_post_hoc

    # choose reference method via CLI
    python -m glow.benchmark.prune_post_hoc --ref node

    # open viewer for experiment at rank 0
    python -m glow.benchmark.prune_post_hoc 0

    # restrict to one source
    python -m glow.benchmark.prune_post_hoc --source hcp
    python -m glow.benchmark.prune_post_hoc 0 --source wgn
"""

import argparse
import sys

import pandas as pd

import glow.benchmark
from glow.benchmark.paper_config import CONFIG_BY_LABEL


_ALL_METHODS = ['GLOW-homo', 'GLOW-node', 'GLOW-node_fl',
                'GLOW-tree', 'GLOW-tree_dp', 'VBA', 'VBA-TFCE']


def _add_gap(pivot, ref_method):
    """Add f1_gap column and sort by reference method advantage."""
    method_cols = [c for c in _ALL_METHODS if c in pivot.columns]
    if ref_method not in pivot.columns:
        print(f'Reference method {ref_method!r} not in results. '
              f'Available: {method_cols}')
        sys.exit(1)

    other_cols = [c for c in method_cols if c != ref_method]
    pivot['best_other'] = pivot[other_cols].idxmax(axis=1)
    gap_col = f'{ref_method} - best_other'
    pivot[gap_col] = (pivot[ref_method]
                      - pivot[other_cols].max(axis=1))
    pivot = pivot.sort_values(gap_col, ascending=True).reset_index(drop=True)

    return pivot, method_cols, gap_col


def _run_and_view(source, seed, effect_llr):
    """Re-run a single experiment and open the viewer with diagnostics."""
    from glow.benchmark.run import _prune_diagnostics_df
    from glow.experiment.prune import (prune, prune_node,
                                       prune_tree, prune_tree_dp)
    from glow.viewer import launch

    label = f'prune_method_{source}'
    config = CONFIG_BY_LABEL[label]

    print(f'Building experiment: source={source}, seed={seed}, '
          f'effect_llr={effect_llr:.4f} ...')
    exp, effect = config.get_exp_eff(seed=int(seed), effect_llr=float(effect_llr))

    _, (Ana, ana_kw) = next(iter(config.ana_kwargs_dict.items()))
    n_perm_prune = ana_kw.get('n_perm_prune', 100)
    alpha_prune = ana_kw.get('alpha_prune', 0.05)
    exp_eff = ana_kw.get('prune_geom_exp_eff')

    print(f'Running AnalysisGLOW ({exp.y.shape[2]} voxels) ...')
    ana = Ana(exp=exp, **ana_kw)
    print(f'  {len(ana.sig_reg_list)} significant regions, '
          f'{len(ana.effect_list)} effects (node)')

    sig = ana.sig_reg_list
    children = ana.children

    if sig:
        methods = {
            'homo': prune(sig, children, exp,
                          n_perm=n_perm_prune, alpha_prune=alpha_prune),
            'node': prune_node(sig, children, exp,
                               n_perm=n_perm_prune,
                               alpha=alpha_prune,
                               exp_eff=exp_eff),
            'node_fl': prune_node(sig, children, exp,
                                  n_perm=n_perm_prune,
                                  alpha=alpha_prune),
            'tree': prune_tree(sig, children, exp),
        }
        if exp_eff is not None:
            methods['tree_dp'] = prune_tree_dp(
                sig, children, exp, exp_eff=exp_eff)

        for name, (regs, _) in methods.items():
            idx_str = ', '.join(str(r) for r in sorted(regs))
            print(f'  {name}: {len(regs)} regions — [{idx_str}]')

        extra_df = _prune_diagnostics_df(sig, methods)
    else:
        extra_df = None

    print('Launching viewer ...')
    launch(ana, mask_target=effect.mask, extra_df=extra_df)


def _load_pivot():
    """Load benchmark results and pivot F1 by method (no gap computation)."""
    frames = []
    for label in ('prune_compare_wgn', 'prune_compare_hcp'):
        df, _, _ = glow.benchmark.load_update_all(label, verbose=False)
        if df.empty:
            continue
        config = CONFIG_BY_LABEL[label]
        if 'config_hash' in df.columns:
            df = df[df['config_hash'] == config._config_hash()]
        df['source'] = config.source
        frames.append(df)

    if not frames:
        print('No prune_compare results found. Run the benchmark first.')
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    pivot = df.pivot_table(
        index=['source', 'seed', 'effect_llr'],
        columns='label',
        values='f1',
    ).reset_index()
    return pivot


def _prompt_ref_method(pivot):
    """Interactively ask the user which method to use as reference."""
    available = [m for m in _ALL_METHODS if m in pivot.columns]
    mean_f1 = {m: pivot[m].mean() for m in available}
    ranked = sorted(available, key=lambda m: mean_f1[m], reverse=True)

    print('Available pruning methods (sorted by mean F1):')
    for i, m in enumerate(ranked):
        print(f'  {i}: {m}  (mean F1 = {mean_f1[m]:.4f})')
    while True:
        try:
            choice = input(f'Reference method (name or number, '
                           f'default={ranked[0]}): ')
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        choice = choice.strip()
        if not choice:
            return ranked[0]
        if choice in available:
            return choice
        try:
            idx = int(choice)
            if 0 <= idx < len(ranked):
                return ranked[idx]
        except ValueError:
            pass
        print(f'  invalid choice: {choice!r}')


def main():
    parser = argparse.ArgumentParser(
        description='Inspect pruning benchmark results.')
    parser.add_argument('rank', nargs='?', type=int, default=None,
                        help='Rank index to open in viewer (0 = largest gap)')
    parser.add_argument('--source', choices=['wgn', 'hcp'], default=None,
                        help='Restrict to one data source')
    parser.add_argument('--ref', choices=_ALL_METHODS, default=None,
                        help='Reference method for f1_gap (default: prompt)')
    args = parser.parse_args()

    pivot = _load_pivot()
    ref_method = args.ref if args.ref else _prompt_ref_method(pivot)
    pivot, method_cols, gap_col = _add_gap(pivot, ref_method)

    if args.source:
        pivot = pivot[pivot['source'] == args.source].reset_index(drop=True)

    cols = ['source', 'seed', 'effect_llr'] + method_cols + ['best_other', gap_col]
    avail = [c for c in cols if c in pivot.columns]
    with pd.option_context('display.max_rows', None, 'display.width', 160,
                           'display.float_format', '{:.4f}'.format):
        print(pivot[avail].to_string())
    print(f'\n{len(pivot)} experiments (worst {ref_method} cases first).')

    rank = args.rank
    while True:
        if rank is None:
            try:
                rank = int(input('\nEnter rank to open in viewer (q to quit): '))
            except (ValueError, EOFError):
                break

        if rank < 0 or rank >= len(pivot):
            print(f'Rank {rank} out of range (0-{len(pivot) - 1})')
            rank = None
            continue

        row = pivot.iloc[rank]
        method_strs = ', '.join(f'{m}={row[m]:.4f}'
                                for m in method_cols if m in row.index)
        print(f'\nRank {rank}: source={row["source"]}, '
              f'seed={int(row["seed"])}, '
              f'effect_llr={row["effect_llr"]:.4f}, '
              f'{method_strs}, '
              f'{gap_col}={row[gap_col]:.4f}')

        _run_and_view(row['source'], row['seed'], row['effect_llr'])
        rank = None


if __name__ == '__main__':
    main()
