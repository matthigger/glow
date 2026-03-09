"""Plot method comparison results from the prune_compare benchmark.

Produces a multi-panel figure:
  Row 1: F1 / sensitivity / specificity vs effect_llr
  Row 2: n_regions / largest region coverage / total volume ratio
  Row 3: region-size strip chart per method

Usage::

    python -m glow.benchmark.prune_compare_plot
    python -m glow.benchmark.prune_compare_plot --source hcp
"""

import argparse
import json
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


def _get_palette(labels):
    glow_labels = sorted(l for l in labels if l.startswith('GLOW'))
    vba_labels = sorted(l for l in labels if not l.startswith('GLOW'))
    n_glow = len(glow_labels)
    n_vba = len(vba_labels)
    glow_colors = sns.color_palette('Blues_d', n_colors=max(n_glow, 3))[:n_glow]
    vba_colors = sns.color_palette('Reds_d', n_colors=max(n_vba, 2))[:n_vba]
    return dict(zip(glow_labels + vba_labels, glow_colors + vba_colors))


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


def _plot_metric_row(axes, df, metric_cols, titles, palette, ci=90):
    """Row of metric-vs-effect_llr line plots (shared logic for rows 1 & 2)."""
    lower_q = (100 - ci) / 2
    upper_q = 100 - lower_q

    for ax, col, title in zip(axes, metric_cols, titles):
        for label in METHOD_ORDER:
            sub = df[df['label'] == label]
            if sub.empty:
                continue
            color = palette.get(label, 'grey')

            for _seed, g in sub.groupby('seed', sort=False):
                g = g.sort_values('effect_llr')
                ax.plot(g['effect_llr'], g[col],
                        lw=0.4, alpha=0.35, color=color)

            stats = (sub.groupby('effect_llr')[col]
                     .agg(['mean',
                           lambda s: np.percentile(s, lower_q),
                           lambda s: np.percentile(s, upper_q)])
                     .reset_index()
                     .sort_values('effect_llr'))
            stats.columns = ['effect_llr', 'mean', 'lo', 'hi']

            ax.plot(stats['effect_llr'], stats['mean'],
                    lw=2.5, color=color, label=label)
            ax.fill_between(stats['effect_llr'], stats['lo'], stats['hi'],
                            color=color, alpha=0.15)

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

    fig = plt.figure(figsize=(15, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, max(3, n_methods), height_ratios=[1, 1, 1])

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

    # row 3: strip chart per method
    ax_r3 = [fig.add_subplot(gs[2, j]) for j in range(n_methods)]
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


def parse_args():
    p = argparse.ArgumentParser(description='Plot method comparison results')
    p.add_argument('--source', choices=['wgn', 'hcp'], default='wgn')
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(user_data_dir('glow', 'glow_author'))
    results_dir = base / 'results' / f'{RESULTS_SUBDIR}_{args.source}'

    if not results_dir.exists():
        print(f'results directory not found: {results_dir}')
        raise SystemExit(1)

    df = load_results(results_dir)
    if df.empty:
        print(f'no result files in {results_dir / "out"}')
        raise SystemExit(1)

    print(f'loaded {len(df)} results '
          f'({df["label"].nunique()} methods, '
          f'{df["seed"].nunique()} seeds, '
          f'{df["effect_llr"].nunique()} effect levels)')

    pdf_path = results_dir / 'prune_compare.pdf'
    plot_compare(df, pdf_path)


if __name__ == '__main__':
    main()
