"""Plot method comparison results from the prune_compare benchmark.

Produces a multi-panel figure:
  Row 1: F1 / sensitivity / specificity vs effect_llr
  Row 2: n_regions / largest region coverage / total volume ratio
  Rows 3+: region-size strip chart per method (3 columns per row)

Method names, colors, and styles are discovered from the result data.

Usage::

    python -m glow.benchmark.prune_compare_plot
"""

import argparse
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from glow.benchmark.file import load_update_all, get_path_result


RESULTS_SUBDIR = 'prune_compare'

_LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
_MARKERS = ['o', 's', '^', 'D', 'v', 'X', 'P', 'h', '*', 'd']


def _sort_labels(labels):
    """GLOW methods first (alphabetically), then the rest (alphabetically)."""
    glow = sorted(l for l in labels if l.startswith('GLOW'))
    other = sorted(l for l in labels if not l.startswith('GLOW'))
    return glow + other


def _build_styles(method_order):
    """Assign a unique color, line style, and marker to each method."""
    n = len(method_order)
    colors = sns.color_palette('tab10', n_colors=max(n, 10))[:n]
    styles = {}
    ls_cycle = itertools.cycle(_LINE_STYLES)
    mk_cycle = itertools.cycle(_MARKERS)
    for i, label in enumerate(method_order):
        styles[label] = dict(
            color=colors[i],
            ls=next(ls_cycle),
            marker=next(mk_cycle),
        )
    return styles


def load_results(label: str) -> pd.DataFrame:
    """Load results from CSV + any new JSONs via load_update_all."""
    import ast

    df, _folder, _n_new = load_update_all(label, verbose=False)
    if df.empty:
        return df
    for col in ('effect_llr', 'f1', 'sens', 'spec'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ('region_sizes', 'region_tp_fracs'):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df


def _derive_columns(df):
    """Add derived columns for row 2 plots."""
    df = df.copy()

    def _largest_frac(row):
        sizes = row.get('region_sizes')
        if not sizes:
            return 0.0
        return sizes[0] / row['vox_effect'] if row['vox_effect'] > 0 else 0.0

    def _total_frac(row):
        sizes = row.get('region_sizes')
        if not sizes:
            return 0.0
        return sum(sizes) / row['vox_effect'] if row['vox_effect'] > 0 else 0.0

    df['largest_frac'] = df.apply(_largest_frac, axis=1)
    df['total_frac'] = df.apply(_total_frac, axis=1)
    return df


def _plot_metric_row(axes, df, metric_cols, titles, method_order, styles):
    """Row of metric-vs-effect_llr line plots (shared logic for rows 1 & 2)."""
    for ax, col, title in zip(axes, metric_cols, titles):
        for label in method_order:
            sub = df[df['label'] == label]
            if sub.empty:
                continue
            s = styles[label]

            stats = (sub.groupby('effect_llr')[col]
                     .mean()
                     .reset_index()
                     .sort_values('effect_llr'))

            ax.plot(stats['effect_llr'], stats[col],
                    lw=2.5, color=s['color'], label=label,
                    ls=s['ls'], marker=s['marker'],
                    markersize=5, markevery=2)

        ax.set_title(title)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.4, linewidth=1.0)


def _plot_strip_row(axes, df, method_order, styles):
    """Strip chart rows: per-method region-size scatter."""
    n_methods = len(method_order)

    sc = None
    for idx, method in enumerate(method_order):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            break
        sub = df[df['label'] == method]

        xs, ys, cs = [], [], []
        for _, row in sub.iterrows():
            sizes = row.get('region_sizes') or []
            tp_fracs = row.get('region_tp_fracs') or []
            vox_eff = row['vox_effect']
            for sz, tp in zip(sizes, tp_fracs):
                xs.append(row['effect_llr'])
                ys.append(sz / vox_eff * 100 if vox_eff > 0 else 0)
                cs.append(tp)

        if xs:
            sc = ax.scatter(xs, ys, c=cs, cmap='RdYlGn', vmin=0, vmax=1,
                            s=18, alpha=0.7, edgecolors='none')
        ax.set_title(method, fontsize=9)
        ax.set_xscale('log')
        ax.set_ylabel('% of true effect')
        ax.grid(True, alpha=0.3)

    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')

    return sc


def plot_compare(df, pdf_path: Path):
    sns.set_theme(context='paper', style='whitegrid', font_scale=1.0)
    df = _derive_columns(df)

    method_order = _sort_labels(df['label'].unique())
    styles = _build_styles(method_order)
    n_methods = len(method_order)

    strip_cols = 3
    strip_rows = math.ceil(n_methods / strip_cols)
    total_rows = 2 + strip_rows
    fig_height = 4 * 2 + 3.5 * strip_rows

    fig = plt.figure(figsize=(15, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(total_rows, strip_cols,
                          height_ratios=[1, 1] + [1] * strip_rows)

    # row 1: F1 / sens / spec
    ax_r1 = [fig.add_subplot(gs[0, j]) for j in range(3)]
    _plot_metric_row(ax_r1, df,
                     ['f1', 'sens', 'spec'],
                     ['F1 score', 'Sensitivity', 'Specificity'],
                     method_order, styles)
    ax_r1[0].legend(fontsize=7, frameon=False, ncol=2)
    ax_r1[0].set_ylabel('score')
    for ax in ax_r1:
        ax.set_xlabel('effect_llr')

    # row 2: n_regions / largest frac / total frac
    ax_r2 = [fig.add_subplot(gs[1, j]) for j in range(3)]
    _plot_metric_row(ax_r2, df,
                     ['n_regions', 'largest_frac', 'total_frac'],
                     ['Regions discovered', 'Largest region / true size',
                      'Total volume / true size'],
                     method_order, styles)
    ax_r2[0].set_ylabel('count')
    ax_r2[1].set_ylabel('fraction')
    ax_r2[1].axhline(1.0, ls='--', lw=0.8, color='black', alpha=0.5)
    ax_r2[2].set_ylabel('fraction')
    ax_r2[2].axhline(1.0, ls='--', lw=0.8, color='black', alpha=0.5)
    for ax in ax_r2:
        ax.set_xlabel('effect_llr')

    # rows 3+: strip chart per method, 3 columns per row
    ax_r3 = []
    for idx in range(n_methods):
        r = 2 + idx // strip_cols
        c = idx % strip_cols
        ax_r3.append(fig.add_subplot(gs[r, c]))
    for idx in range(n_methods, strip_rows * strip_cols):
        r = 2 + idx // strip_cols
        c = idx % strip_cols
        fig.add_subplot(gs[r, c]).axis('off')

    sc = _plot_strip_row(ax_r3, df, method_order, styles)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax_r3, location='right', shrink=0.8,
                            pad=0.02)
        cbar.set_label('TP fraction', fontsize=9)
    for ax in ax_r3:
        ax.set_xlabel('effect_llr')

    fig.savefig(pdf_path, bbox_inches='tight')
    print(f'plot saved: {pdf_path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--redo', action='store_true',
                        help='re-consolidate JSONs into CSV before plotting')
    args = parser.parse_args()

    results_base = get_path_result()

    found = sorted(results_base.glob(f'{RESULTS_SUBDIR}_*'))
    if not found:
        print(f'no {RESULTS_SUBDIR}_* directories in {results_base}')
        raise SystemExit(1)

    for results_dir in found:
        label = results_dir.name
        source = label.removeprefix(f'{RESULTS_SUBDIR}_')

        if args.redo:
            load_update_all(label, verbose=True)

        df = load_results(label)
        if df.empty:
            print(f'[{source}] no results found, skipping')
            continue

        print(f'[{source}] loaded {len(df)} results '
              f'({df["label"].nunique()} methods, '
              f'{df["seed"].nunique()} seeds, '
              f'{df["effect_llr"].nunique()} effect levels)')

        pdf_path = results_dir / 'prune_compare.pdf'
        plot_compare(df, pdf_path)


if __name__ == '__main__':
    main()
