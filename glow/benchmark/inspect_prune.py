"""Inspect pruning benchmark: find where node beats tree.

Usage::

    # list experiments sorted by node advantage (both sources)
    python -m glow.benchmark.inspect_prune

    # open viewer for experiment at rank 0
    python -m glow.benchmark.inspect_prune 0

    # restrict to one source
    python -m glow.benchmark.inspect_prune --source hcp
    python -m glow.benchmark.inspect_prune 0 --source wgn
"""

import argparse
import sys
import tempfile

import numpy as np
import pandas as pd

import glow.benchmark
from glow.benchmark.paper_config import CONFIG_BY_LABEL


def _load_comparison():
    """Load prune_method results and pivot node vs tree."""
    frames = []
    for label in ('prune_method_wgn', 'prune_method_hcp'):
        df, _, _ = glow.benchmark.load_update_all(label, verbose=False)
        if df.empty:
            continue
        config = CONFIG_BY_LABEL[label]
        if 'config_hash' in df.columns:
            df = df[df['config_hash'] == config._config_hash()]
        df['source'] = config.source
        frames.append(df)

    if not frames:
        print('No prune_method results found. Run the benchmark first.')
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)

    pivot = df.pivot_table(
        index=['source', 'seed', 'hotel_tr'],
        columns='label',
        values='f1',
    ).reset_index()

    if 'node' not in pivot.columns or 'tree' not in pivot.columns:
        print('Missing node or tree results.')
        sys.exit(1)

    pivot['f1_gap'] = pivot['node'] - pivot['tree']
    pivot = pivot.sort_values('f1_gap', ascending=False).reset_index(drop=True)

    return pivot


def _run_and_view(source, seed, hotel_tr):
    """Re-run a single experiment and open the viewer with diagnostics."""
    from glow.benchmark.run import _prune_diagnostics_df
    from glow.experiment.prune import (prune, prune_node,
                                       prune_tree, prune_tree_dp)
    from glow.viewer import launch

    label = f'prune_method_{source}'
    config = CONFIG_BY_LABEL[label]

    print(f'Building experiment: source={source}, seed={seed}, '
          f'hotel_tr={hotel_tr:.4f} ...')
    exp, effect = config.get_exp_eff(seed=int(seed), hotel_tr=float(hotel_tr))

    _, (Ana, ana_kw) = next(iter(config.ana_kwargs_dict.items()))
    n_perm_prune = ana_kw.get('n_perm_prune', 100)
    alpha_prune = ana_kw.get('alpha_prune', 0.05)
    exp_eff = ana_kw.get('prune_geom_exp_eff')

    print(f'Running AnalysisGLOW ({exp.y.shape[2]} voxels) ...')
    ana = Ana(exp=exp, **ana_kw)
    print(f'  {len(ana.sig_reg_list)} significant regions, '
          f'{len(ana.effect_list)} effects (node)')

    sig = ana.sig_reg_list
    children = ana.child_dict[0]

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


def main():
    parser = argparse.ArgumentParser(
        description='Inspect pruning benchmark results.')
    parser.add_argument('rank', nargs='?', type=int, default=None,
                        help='Rank index to open in viewer (0 = largest gap)')
    parser.add_argument('--source', choices=['wgn', 'hcp'], default=None,
                        help='Restrict to one data source')
    args = parser.parse_args()

    pivot = _load_comparison()

    if args.source:
        pivot = pivot[pivot['source'] == args.source].reset_index(drop=True)

    cols = ['source', 'seed', 'hotel_tr', 'node', 'tree',
            'f1_gap']
    avail = [c for c in cols if c in pivot.columns]
    with pd.option_context('display.max_rows', None, 'display.width', 120,
                           'display.float_format', '{:.4f}'.format):
        print(pivot[avail].to_string())
    print(f'\n{len(pivot)} experiments (sorted by f1_gap, node - tree).')

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
        print(f'\nRank {rank}: source={row["source"]}, '
              f'seed={int(row["seed"])}, '
              f'hotel_tr={row["hotel_tr"]:.4f}, '
              f'node={row["node"]:.4f}, '
              f'tree={row["tree"]:.4f}, '
              f'f1_gap={row["f1_gap"]:.4f}')

        _run_and_view(row['source'], row['seed'], row['hotel_tr'])
        rank = None


if __name__ == '__main__':
    main()
