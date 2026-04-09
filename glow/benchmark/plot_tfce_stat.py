"""Visualise the TFCE stat comparison benchmark (5 stats x {raw, z}).

Usage:
    python -m glow.benchmark.plot_tfce_stat
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import glow.benchmark

STAT_ORDER = ['llr', 'pillai', 'wilks', 'hotel_tr', 'roys_root']
STAT_NICE = {
    'llr': 'LLR', 'pillai': 'Pillai', 'wilks': 'Wilks',
    'hotel_tr': 'Hotelling', 'roys_root': "Roy's root",
}
SOURCES = [
    ('mancova_vba_wgn', 'WGN (synthetic)'),
    ('mancova_vba_hcp', 'HCP (real)'),
]


def _load(label):
    df, folder, _ = glow.benchmark.load_update_all(label, verbose=False)
    return df, folder


def _parse_label(label):
    """'VBA-TFCE-pillai-z' -> ('pillai', True)"""
    rest = label.removeprefix('VBA-TFCE-')
    if rest.endswith('-z'):
        return rest[:-2], True
    return rest, False


def _agg(df):
    """Mean F1 per (label, effect_llr)."""
    return (df.groupby(['label', 'effect_llr'])['f1']
            .agg(['mean', 'std', 'count'])
            .reset_index())


# ------------------------------------------------------------------
# Figure 1: faceted grid  (2 sources x 5 stats)
# ------------------------------------------------------------------

def plot_facet_grid(datasets):
    """5 columns (one per stat) x 2 rows (WGN, HCP).

    Each cell: raw (solid) + z-scored (dashed).
    A thin grey line shows LLR-raw as a common reference.
    """
    n_src = len(datasets)
    n_stat = len(STAT_ORDER)
    fig, axes = plt.subplots(n_src, n_stat, figsize=(3.2 * n_stat, 3.4 * n_src),
                             sharex=True, sharey=True)

    for row, (df_agg, src_title) in enumerate(datasets):
        llr_ref = df_agg[df_agg['label'] == 'VBA-TFCE-llr']

        for col, stat in enumerate(STAT_ORDER):
            ax = axes[row, col]

            ax.plot(llr_ref['effect_llr'], llr_ref['mean'],
                    color='0.65', lw=1.5, ls=':', label='LLR ref', zorder=1)

            for z_flag in [False, True]:
                suffix = '-z' if z_flag else ''
                lab = f'VBA-TFCE-{stat}{suffix}'
                sub = df_agg[df_agg['label'] == lab]
                if sub.empty:
                    continue
                style = '--' if z_flag else '-'
                nice = f'{STAT_NICE[stat]}{" (z)" if z_flag else ""}'
                color = 'C0' if not z_flag else 'C1'
                ax.plot(sub['effect_llr'], sub['mean'], style,
                        color=color, lw=2.2, label=nice, zorder=2)
                ax.fill_between(
                    sub['effect_llr'],
                    sub['mean'] - sub['std'] / np.sqrt(sub['count']),
                    sub['mean'] + sub['std'] / np.sqrt(sub['count']),
                    color=color, alpha=0.15, zorder=0)

            ax.set_xscale('log')
            if row == 0:
                ax.set_title(STAT_NICE[stat], fontsize=11)
            if col == 0:
                ax.set_ylabel(f'{src_title}\nF1', fontsize=10)
            if row == n_src - 1:
                ax.set_xlabel('effect LLR', fontsize=9)
            ax.legend(fontsize=7, frameon=False, loc='upper left')
            ax.grid(True, alpha=0.25)
            ax.set_ylim(-0.02, 1.02)

    fig.suptitle('TFCE stat comparison: raw vs z-scored (±1 SE shading)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Figure 2: summary — best contenders on one clean plot per source
# ------------------------------------------------------------------

def plot_summary(datasets):
    """Side-by-side panels showing the top stats for each source."""
    top_labels = [
        ('VBA-TFCE-llr',       'LLR',         'C0', '-'),
        ('VBA-TFCE-llr-z',     'LLR (z)',      'C0', '--'),
        ('VBA-TFCE-pillai',    'Pillai',       'C1', '-'),
        ('VBA-TFCE-pillai-z',  'Pillai (z)',   'C1', '--'),
        ('VBA-TFCE-wilks-z',   'Wilks (z)',    'C2', '--'),
        ('VBA-TFCE-hotel_tr',  'Hotelling',    'C4', '-'),
        ('VBA-TFCE-hotel_tr-z','Hotelling (z)','C4', '--'),
        ('VBA-TFCE-roys_root', "Roy's root",   'C3', '-'),
    ]

    n_src = len(datasets)
    fig, axes = plt.subplots(1, n_src, figsize=(7 * n_src, 4.5), sharey=True)
    if n_src == 1:
        axes = [axes]

    for ax, (df_agg, src_title) in zip(axes, datasets):
        for lab, nice, color, style in top_labels:
            sub = df_agg[df_agg['label'] == lab]
            if sub.empty:
                continue
            ax.plot(sub['effect_llr'], sub['mean'], style,
                    color=color, lw=2.2, label=nice)
            ax.fill_between(
                sub['effect_llr'],
                sub['mean'] - sub['std'] / np.sqrt(sub['count']),
                sub['mean'] + sub['std'] / np.sqrt(sub['count']),
                color=color, alpha=0.10)

        ax.set_xscale('log')
        ax.set_xlabel('effect LLR')
        ax.set_ylabel('F1')
        ax.set_title(src_title, fontsize=12)
        ax.legend(fontsize=8, frameon=False, ncol=2)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle('VBA-TFCE: F1 by statistic (mean ± 1 SE)', fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Figure 3: z-scoring delta (F1_z - F1_raw) per stat
# ------------------------------------------------------------------

def plot_z_delta(datasets):
    """Per-stat z-scoring improvement, one panel per source."""
    n_src = len(datasets)
    fig, axes = plt.subplots(1, n_src, figsize=(7 * n_src, 4), sharey=True)
    if n_src == 1:
        axes = [axes]

    colors = dict(zip(STAT_ORDER, ['C0', 'C1', 'C2', 'C4', 'C3']))

    for ax, (df_agg, src_title) in zip(axes, datasets):
        for stat in STAT_ORDER:
            raw = df_agg[df_agg['label'] == f'VBA-TFCE-{stat}']
            z = df_agg[df_agg['label'] == f'VBA-TFCE-{stat}-z']
            if raw.empty or z.empty:
                continue
            merged = raw.merge(z, on='effect_llr', suffixes=('_raw', '_z'))
            delta = merged['mean_z'] - merged['mean_raw']
            ax.plot(merged['effect_llr'], delta, '-o', color=colors[stat],
                    lw=2, markersize=4, label=STAT_NICE[stat])

        ax.axhline(0, color='black', lw=0.8, ls=':')
        ax.set_xscale('log')
        ax.set_xlabel('effect LLR')
        ax.set_ylabel('$\\Delta$ F1  (z-scored $-$ raw)')
        ax.set_title(src_title, fontsize=12)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25)

    fig.suptitle('Effect of z-scoring on F1 by statistic', fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------

def main():
    datasets = []
    folders = {}
    for label, nice in SOURCES:
        df, folder = _load(label)
        if df.empty:
            print(f'skipping {label}: no data')
            continue
        datasets.append((_agg(df), nice))
        folders[label] = folder

    if not datasets:
        print('no data found')
        return

    out = folders.get(SOURCES[0][0], folders[next(iter(folders))])

    fig1 = plot_facet_grid(datasets)
    p1 = out / 'tfce_facet.pdf'
    fig1.savefig(p1, bbox_inches='tight')
    print(f'saved: {p1}')

    fig2 = plot_summary(datasets)
    p2 = out / 'tfce_summary.pdf'
    fig2.savefig(p2, bbox_inches='tight')
    print(f'saved: {p2}')

    fig3 = plot_z_delta(datasets)
    p3 = out / 'tfce_z_delta.pdf'
    fig3.savefig(p3, bbox_inches='tight')
    print(f'saved: {p3}')

    plt.close('all')


if __name__ == '__main__':
    main()
