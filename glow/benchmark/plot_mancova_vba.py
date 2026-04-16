"""Visualise the MANCOVA stat comparison across all methods.

Covers VBA, VBA-TFCE, CET, and GLOW (5 stats each, ±z where applicable).

Usage:
    python -m glow.benchmark.plot_mancova_vba
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
METHOD_ORDER = ['VBA', 'VBA-TFCE', 'CET', 'GLOW']

# VBA/TFCE/CET results live in mancova_vba_*, GLOW in mancova_glow_*
SOURCES_VBA = [
    ('mancova_vba_wgn', 'WGN'),
    ('mancova_vba_hcp', 'HCP'),
]
SOURCES_GLOW = [
    ('mancova_glow_wgn', 'WGN'),
    ('mancova_glow_hcp', 'HCP'),
]
# legacy alias used by the VBA-TFCE detail plots
SOURCES = [(l, f'{n} ({"synthetic" if "wgn" in l else "real"})') for l, n in SOURCES_VBA]


def _load(label):
    df, folder, _ = glow.benchmark.load_update_all(label, verbose=False)
    return df, folder


def _parse_label(label):
    """'VBA-TFCE-pillai-z' -> ('pillai', True)"""
    rest = label.removeprefix('VBA-TFCE-')
    if rest.endswith('-z'):
        return rest[:-2], True
    return rest, False


def _parse_label_full(label):
    """Parse any mancova label into (method, stat, z_flag).

    Examples:
        'VBA-TFCE-llr-z' -> ('VBA-TFCE', 'llr', True)
        'VBA-llr'        -> ('VBA',      'llr', False)
        'CET-pillai'     -> ('CET',      'pillai', False)
        'GLOW-wilks'     -> ('GLOW',     'wilks', False)
    """
    z_flag = label.endswith('-z')
    if z_flag:
        label = label[:-2]

    if label.startswith('VBA-TFCE-'):
        return 'VBA-TFCE', label.removeprefix('VBA-TFCE-'), z_flag
    if label.startswith('VBA-'):
        return 'VBA', label.removeprefix('VBA-'), z_flag
    if label.startswith('CET-'):
        return 'CET', label.removeprefix('CET-'), z_flag
    if label.startswith('GLOW-'):
        return 'GLOW', label.removeprefix('GLOW-'), z_flag

    return label, '', z_flag


def _agg(df):
    """Mean Dice per (label, effect_llr)."""
    return (df.groupby(['label', 'effect_llr'])['dice']
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
                ax.set_ylabel(f'{src_title}\nDice', fontsize=10)
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
        ax.set_ylabel('Dice')
        ax.set_title(src_title, fontsize=12)
        ax.legend(fontsize=8, frameon=False, ncol=2)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle('VBA-TFCE: Dice by statistic (mean ± 1 SE)', fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------
# Figure 3: z-scoring delta (Dice_z - Dice_raw) per stat
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
        ax.set_ylabel('$\\Delta$ Dice  (z-scored $-$ raw)')
        ax.set_title(src_title, fontsize=12)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(True, alpha=0.25)

    fig.suptitle('Effect of z-scoring on Dice by statistic', fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------

def build_summary_table(raw_dfs):
    """Build a summary CSV: mean Dice, std, and win rate by stat and z-score.

    Parameters
    ----------
    raw_dfs : list of (pd.DataFrame, str)
        Each entry is (raw df with 'label' column, source nice-name).
        Labels must be VBA-TFCE-{stat}[-z] format.

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for df, source_nice in raw_dfs:
        # parse stat and z flag from label
        parsed = df['label'].map(_parse_label)
        df = df.copy()
        df['stat'] = [s for s, _ in parsed]
        df['z_scored'] = [z for _, z in parsed]

        # mean dice by (stat, z_scored)
        grp = df.groupby(['stat', 'z_scored'])['dice']
        means = grp.mean()
        stds = grp.std()

        # win rate: which (stat, z) combo has highest dice per
        # (seed, effect_llr)?  sums to 1 across all 10 variants.
        df_sig = df[df['effect_llr'] > 0.01]
        best_idx = df_sig.groupby(['seed', 'effect_llr'])['dice'].idxmax()
        win_counts = (df_sig.loc[best_idx]
                      .groupby(['stat', 'z_scored']).size())
        wins = win_counts / win_counts.sum()

        for z_flag in [False, True]:
            for stat in STAT_ORDER:
                rows.append({
                    'source': source_nice,
                    'stat': STAT_NICE.get(stat, stat),
                    'z_scored': z_flag,
                    'mean_dice': means.get((stat, z_flag), np.nan),
                    'std_dice': stds.get((stat, z_flag), np.nan),
                    'win_rate': wins.get((stat, z_flag), 0.0),
                })

    return pd.DataFrame(rows)


def build_best_stat_table(all_dfs):
    """Build a table of best stat per (method, source).

    Parameters
    ----------
    all_dfs : list of (pd.DataFrame, str)
        Each entry is (raw df, source nice-name).  Labels can be any method.

    Returns
    -------
    pd.DataFrame  with columns: source, method, stat, z_scored, mean_dice,
                                std_dice, win_rate, tie_rate, mean_loss
    """
    rows = []
    for df, source_nice in all_dfs:
        df = df.copy()
        parsed = df['label'].map(_parse_label_full)
        df['method'] = [m for m, _, _ in parsed]
        df['stat'] = [s for _, s, _ in parsed]
        df['z_scored'] = [z for _, _, z in parsed]

        for method in METHOD_ORDER:
            mdf = df[df['method'] == method]
            if mdf.empty:
                continue

            # mean dice per variant
            grp = mdf.groupby(['stat', 'z_scored'])['dice']
            means = grp.mean()
            stds = grp.std()

            # per-(seed, effect_llr): best dice across all variants of this
            # method, then regret = best - this variant's dice
            mdf_sig = mdf[mdf['effect_llr'] > 0.01]
            if mdf_sig.empty:
                continue

            # best dice per trial
            best_per_trial = mdf_sig.groupby(
                ['seed', 'effect_llr'])['dice'].transform('max')
            mdf_sig = mdf_sig.copy()
            mdf_sig['is_max'] = mdf_sig['dice'] == best_per_trial

            # number of stats sharing the max per trial
            n_at_max = mdf_sig.groupby(
                ['seed', 'effect_llr'])['is_max'].transform('sum')
            mdf_sig['is_tie'] = mdf_sig['is_max'] & (n_at_max > 1)

            # win rate (includes ties) and tie rate, per variant
            n_trials = mdf_sig.groupby(['stat', 'z_scored']).size()
            win_counts = (mdf_sig.groupby(['stat', 'z_scored'])['is_max']
                          .sum())
            wins = win_counts / n_trials
            tie_counts = (mdf_sig.groupby(['stat', 'z_scored'])['is_tie']
                          .sum())
            ties = tie_counts / n_trials

            # mean loss: average regret conditioned on NOT being the max
            mdf_sig['regret'] = best_per_trial - mdf_sig['dice']
            loss_mask = ~mdf_sig['is_max']
            loss_regret = (mdf_sig[loss_mask]
                           .groupby(['stat', 'z_scored'])['regret'].mean())

            z_options = [False, True] if method in ('VBA', 'VBA-TFCE', 'CET') else [False]
            for z_flag in z_options:
                for stat in STAT_ORDER:
                    if (stat, z_flag) not in means.index:
                        continue
                    rows.append({
                        'source': source_nice,
                        'method': method,
                        'stat': STAT_NICE.get(stat, stat),
                        'z_scored': z_flag,
                        'mean_dice': means.get((stat, z_flag), np.nan),
                        'std_dice': stds.get((stat, z_flag), np.nan),
                        'win_rate': wins.get((stat, z_flag), 0.0),
                        'tie_rate': ties.get((stat, z_flag), 0.0),
                        'mean_loss': loss_regret.get((stat, z_flag), 0.0),
                    })

    return pd.DataFrame(rows)


def main():
    # --- load all sources (VBA/TFCE/CET from mancova_vba_*, GLOW from mancova_glow_*) ---
    # keyed by source nice-name -> combined df
    combined = {}  # source_nice -> list of dfs
    for label, nice in SOURCES_VBA + SOURCES_GLOW:
        df, folder = _load(label)
        if df.empty:
            print(f'skipping {label}: no data')
            continue
        combined.setdefault(nice, []).append(df)

    all_dfs = []
    for nice, dfs in combined.items():
        all_dfs.append((pd.concat(dfs, ignore_index=True), nice))

    if not all_dfs:
        print('no data found')
        return

    out = glow.benchmark.get_path_result() / '_latest'
    out.mkdir(exist_ok=True)

    # --- VBA-TFCE detail plots (existing) ---
    tfce_datasets = []
    tfce_raw = []
    for label, nice in SOURCES:
        df, folder = _load(label)
        if df.empty:
            continue
        df_tfce = df[df['label'].str.startswith('VBA-TFCE-')]
        if df_tfce.empty:
            continue
        tfce_raw.append((df_tfce, nice))
        tfce_datasets.append((_agg(df_tfce), nice))

    if tfce_datasets:
        fig1 = plot_facet_grid(tfce_datasets)
        p1 = out / 'mancova_vba_facet.pdf'
        fig1.savefig(p1, bbox_inches='tight')
        print(f'saved: {p1}')

        fig2 = plot_summary(tfce_datasets)
        p2 = out / 'mancova_vba_summary.pdf'
        fig2.savefig(p2, bbox_inches='tight')
        print(f'saved: {p2}')

        fig3 = plot_z_delta(tfce_datasets)
        p3 = out / 'mancova_vba_z_delta.pdf'
        fig3.savefig(p3, bbox_inches='tight')
        print(f'saved: {p3}')

    # --- best stat per method (all methods) ---
    best = build_best_stat_table(all_dfs)
    csv_path = out / 'mancova_best_stat.csv'
    best.to_csv(csv_path, index=False, float_format='%.4f')
    print(f'saved: {csv_path}')

    # print best-stat summary to stdout
    for source in best['source'].unique():
        print(f'\n## {source}\n')
        for method in METHOD_ORDER:
            sub = best[(best['source'] == source) & (best['method'] == method)]
            if sub.empty:
                continue
            sub = sub.sort_values('mean_dice', ascending=False)
            winner = sub.iloc[0]
            z_str = ' (z)' if winner['z_scored'] else ''
            print(f'  {method:<10s}  best={winner["stat"]}{z_str:<14s}  '
                  f'dice={winner["mean_dice"]:.4f}  '
                  f'win={winner["win_rate"]:.1%}  '
                  f'tie={winner["tie_rate"]:.1%}  '
                  f'loss={winner["mean_loss"]:.4f}')

        # full table
        sub_all = best[best['source'] == source].sort_values(
            ['method', 'mean_dice'], ascending=[True, False])
        print(f'\n| method     | stat         | z   | mean_dice | win_rate | tie_rate | mean_loss |')
        print(f'|------------|--------------|-----|-----------|----------|----------|-----------|')
        for _, r in sub_all.iterrows():
            z = 'yes' if r['z_scored'] else 'no'
            print(f'| {r["method"]:<10s} | {r["stat"]:<12s} | {z:<3s} '
                  f'| {r["mean_dice"]:.4f}    '
                  f'| {r["win_rate"]:.4f}   '
                  f'| {r["tie_rate"]:.4f}   '
                  f'| {r["mean_loss"]:.4f}    |')

    plt.close('all')


if __name__ == '__main__':
    main()
