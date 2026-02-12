"""Plot pruning comparison results from compare_pruning_results.csv.

Usage:
    python -m glow.benchmark.compare_pruning          # run experiments first
    python -m glow.benchmark.plot_pruning_comparison   # then plot
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load():
    csv = Path(__file__).with_name('compare_pruning_results.csv')
    if not csv.exists():
        raise FileNotFoundError(
            f'{csv} not found. Run compare_pruning.py first.')
    return pd.read_csv(csv)


def _aggregate(df):
    """Group by (source, hotel_tr) and compute means."""
    return df.groupby(['source', 'hotel_tr']).agg(
        f1_homo=('f1_homo', 'mean'),
        f1_greedy=('f1_greedy', 'mean'),
        sens_homo=('sens_homo', 'mean'),
        sens_greedy=('sens_greedy', 'mean'),
        spec_homo=('spec_homo', 'mean'),
        spec_greedy=('spec_greedy', 'mean'),
        n_homo=('n_homo', 'mean'),
        n_greedy=('n_greedy', 'mean'),
    ).reset_index()


def main():
    df = _load()
    agg = _aggregate(df)

    colors = {'homo': '#2166ac', 'greedy': '#b2182b'}
    lw = 2.2
    ms = 7

    # --- Figure 1: F1 / Sensitivity / Specificity ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Homogeneity Pruning vs Greedy Pruning', fontsize=15,
                 fontweight='bold', y=0.98)

    for row, source in enumerate(('wgn', 'hcp')):
        sub = agg[agg['source'] == source]
        ht = sub['hotel_tr'].values

        for col, (metric, ylabel) in enumerate([
            ('f1', 'F1 Score'), ('sens', 'Sensitivity'), ('spec', 'Specificity')
        ]):
            ax = axes[row, col]
            h = sub[f'{metric}_homo'].values
            g = sub[f'{metric}_greedy'].values

            ax.semilogx(ht, h, 'o-', color=colors['homo'],
                         lw=lw, ms=ms, label='Homogeneity', zorder=3)
            ax.semilogx(ht, g, 's--', color=colors['greedy'],
                         lw=lw, ms=ms, label='Greedy', zorder=3)
            # shade delta
            if metric in ('f1', 'sens'):
                ax.fill_between(ht, h, g, alpha=0.15, color=colors['greedy'],
                                 zorder=1)
            else:
                ax.fill_between(ht, g, h, alpha=0.15, color=colors['homo'],
                                 zorder=1)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'{source.upper()} -- {ylabel}', fontsize=13,
                         fontweight='bold')
            if metric == 'spec':
                ax.set_ylim(0.78, 1.005)
            else:
                ax.set_ylim(-0.02, 1.05)
            ax.legend(loc='lower right' if metric != 'spec' else 'lower left',
                      fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('hotel_tr (effect size)', fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # --- Figure 2: delta F1 + region counts ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle('Greedy - Homogeneity: Differences', fontsize=14,
                  fontweight='bold', y=1.0)

    src_colors = {'wgn': '#d6604d', 'hcp': '#4393c3'}

    ax = axes2[0]
    for source in ('wgn', 'hcp'):
        sub = agg[agg['source'] == source]
        ht = sub['hotel_tr'].values
        delta = sub['f1_greedy'].values - sub['f1_homo'].values
        ax.semilogx(ht, delta, 'o-' if source == 'wgn' else 's-',
                     color=src_colors[source], lw=lw, ms=ms,
                     label=source.upper())
        ax.fill_between(ht, 0, delta, alpha=0.12, color=src_colors[source])
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylabel('delta F1 (greedy - homo)', fontsize=12)
    ax.set_xlabel('hotel_tr (effect size)', fontsize=11)
    ax.set_title('F1 Improvement from Greedy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    for source in ('wgn', 'hcp'):
        sub = agg[agg['source'] == source]
        ht = sub['hotel_tr'].values
        marker = 'o' if source == 'wgn' else 's'
        ax.semilogx(ht, sub['n_homo'].values, f'{marker}-',
                     color=src_colors[source], lw=lw, ms=ms,
                     label=f'{source.upper()} homo')
        ax.semilogx(ht, sub['n_greedy'].values, f'{marker}--',
                     color=src_colors[source], lw=1.5, ms=ms, alpha=0.6,
                     label=f'{source.upper()} greedy')
    ax.set_ylabel('# regions selected (mean)', fontsize=12)
    ax.set_xlabel('hotel_tr (effect size)', fontsize=11)
    ax.set_title('Number of Output Regions', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    fig2.tight_layout()

    out_dir = Path(__file__).parent
    out1 = out_dir / 'pruning_comparison.png'
    out2 = out_dir / 'pruning_delta.png'
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f'saved: {out1}')
    print(f'saved: {out2}')


if __name__ == '__main__':
    main()
