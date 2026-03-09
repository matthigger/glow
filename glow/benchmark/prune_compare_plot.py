"""Plot method comparison results from the prune_compare benchmark.

Produces a multi-panel figure:
  Row 1: F1 / sensitivity / specificity vs effect_llr
  Row 2: n_regions / largest region coverage / total volume ratio
  Row 3: region-size strip chart per method

Usage::

    python -m glow.benchmark.prune_compare_plot
    python -m glow.benchmark.prune_compare_plot --source hcp
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from platformdirs import user_data_dir


RESULTS_SUBDIR = 'prune_compare'

METHOD_ORDER = [
    'GLOW-node', 'GLOW-node_fl', 'GLOW-homo',
    'GLOW-tree', 'GLOW-tree_dp',
    'VBA', 'VBA-TFCE',
]


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob('out/*_result.json')):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ('effect_llr', 'f1', 'sens', 'spec'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


_GLOW_COLORS = {
    'GLOW-node': '#1b9e77',
    'GLOW-node_fl': '#d95f02',
    'GLOW-homo': '#7570b3',
    'GLOW-tree': '#e7298a',
    'GLOW-tree_dp': '#66a61e',
}

_VBA_COLORS = {
    'VBA': '#e41a1c',
    'VBA-TFCE': '#984ea3',
}

_METHOD_STYLE = {
    'GLOW-node':    dict(ls='-',      marker='o'),
    'GLOW-node_fl': dict(ls='--',     marker='s'),
    'GLOW-homo':    dict(ls='-.',     marker='^'),
    'GLOW-tree':    dict(ls=(0, (3, 1, 1, 1)), marker='D'),
    'GLOW-tree_dp': dict(ls=':',      marker='v'),
    'VBA':          dict(ls='-',      marker='X'),
    'VBA-TFCE':     dict(ls='--',     marker='P'),
}


def _get_palette(labels):
    palette = {}
    for label in labels:
        if label in _GLOW_COLORS:
            palette[label] = _GLOW_COLORS[label]
        elif label in _VBA_COLORS:
            palette[label] = _VBA_COLORS[label]
        else:
            palette[label] = 'grey'
    return palette


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


def _plot_metric_row(axes, df, metric_cols, titles, palette):
    """Row of metric-vs-effect_llr line plots (shared logic for rows 1 & 2)."""
    for ax, col, title in zip(axes, metric_cols, titles):
        for label in METHOD_ORDER:
            sub = df[df['label'] == label]
            if sub.empty:
                continue
            color = palette.get(label, 'grey')
            style = _METHOD_STYLE.get(label, {})

            stats = (sub.groupby('effect_llr')[col]
                     .mean()
                     .reset_index()
                     .sort_values('effect_llr'))

            ax.plot(stats['effect_llr'], stats[col],
                    lw=2.5, color=color, label=label,
                    ls=style.get('ls', '-'),
                    marker=style.get('marker'),
                    markersize=5, markevery=2)

        ax.set_title(title)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.4, linewidth=1.0)


def _plot_strip_row(axes, df, palette):
    """Row 3: per-method region-size strip chart."""
    methods_present = [m for m in METHOD_ORDER if m in df['label'].values]
    n_methods = len(methods_present)

    for idx, method in enumerate(methods_present):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            break
        sub = df[df['label'] == method]
        color = palette.get(method, 'grey')

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

    return sc if xs else None


def plot_compare(df, pdf_path: Path):
    sns.set_theme(context='paper', style='whitegrid', font_scale=1.0)
    df = _derive_columns(df)

    labels = [m for m in METHOD_ORDER if m in df['label'].values]
    palette = _get_palette(labels)
    n_methods = len(labels)

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
                     palette)
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
                     palette)
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

    sc = _plot_strip_row(ax_r3, df, palette)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax_r3, location='right', shrink=0.8,
                            pad=0.02)
        cbar.set_label('TP fraction', fontsize=9)
    for ax in ax_r3:
        ax.set_xlabel('effect_llr')

    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(pdf_path.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f'plot saved: {pdf_path}')
    plt.close(fig)


def main():
    base = Path(user_data_dir('glow', 'glow_author'))
    results_base = base / 'results'

    found = sorted(results_base.glob(f'{RESULTS_SUBDIR}_*'))
    if not found:
        print(f'no {RESULTS_SUBDIR}_* directories in {results_base}')
        raise SystemExit(1)

    for results_dir in found:
        source = results_dir.name.removeprefix(f'{RESULTS_SUBDIR}_')
        df = load_results(results_dir)
        if df.empty:
            print(f'[{source}] no result files in {results_dir / "out"}, skipping')
            continue

        print(f'[{source}] loaded {len(df)} results '
              f'({df["label"].nunique()} methods, '
              f'{df["seed"].nunique()} seeds, '
              f'{df["effect_llr"].nunique()} effect levels)')

        pdf_path = results_dir / 'prune_compare.pdf'
        plot_compare(df, pdf_path)


if __name__ == '__main__':
    main()
