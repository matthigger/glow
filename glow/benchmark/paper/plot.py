"""Paper plots: shared palette, generic helpers, and the plot-everything entrypoint.

python -m glow.benchmark.paper.plot (main) walks the config catalogue
(CACHE_BY_LABEL) rather than the result folders: for each cache it keeps
only the complete trials the current config defines (_load_in_config
drops anything left over from an older config) and writes one figure set
per cache into results/_latest:

  - run_ana caches get a compute-time boxplot plus either a FWER
    calibration curve (null caches) or a dice/sens/spec metric sweep,
    via plot_ana_cache.
  - the MANCOVA stat-comparison caches (run_mancova) are combined by
    source (WGN / HCP) and plotted by _plot_mancova with the
    faceted-grid / summary / z-delta figures.

The generic helpers (plot_compute_time, plot_calibration,
plot_x_vs_metrics) stay reusable so notebooks can call them directly.
"""
import colorsys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import glow.benchmark
from glow.util import stable_hash


# ---------------------------------------------------------------------------
# Consistent paper colour palette
# ---------------------------------------------------------------------------
# Base: teal from Fig. 3 (#4DA6A6), H=180° S=0.37 L=0.48 in HLS.
# Analysis methods: 4 hues evenly spaced (90° apart), same S/L.
# Segmentation methods: R/G/B hues (0°/120°/240°), same S/L.
_H, _L, _S = 0.500, 0.476, 0.366  # HLS of #4DA6A6


def _hls_hex(h: float, l: float = _L, s: float = _S) -> str:
    """Convert an HLS triple to a #rrggbb hex string."""
    r, g, b = colorsys.hls_to_rgb(h % 1.0, l, s)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


COLOR_ANALYSIS = {
    # teal (180 deg)
    'GLOW-Focus': _hls_hex(0/4 + _H),
    # darker teal
    'GLOW-GLM':   _hls_hex(0/4 + _H, l=_L * 0.6),
    # purple (270 deg)
    'VBA-TFCE':   _hls_hex(1/4 + _H),
    # coral (0 deg)
    'VBA':        _hls_hex(2/4 + _H),
    # olive (90 deg)
    'CET':        _hls_hex(3/4 + _H),
}

COLOR_SEGMENT = {
    # red (0 deg)
    'Naive':     _hls_hex(0/3),
    # green (120 deg)
    'GLM Error': _hls_hex(1/3),
    # blue (240 deg)
    'Focus':     _hls_hex(2/3),
}


def get_cmap_dict(label_list) -> dict:
    """Map each label to a color, using the fixed palette where possible.

    Args:
        label_list: labels to assign colors to

    Returns:
        out (dict): label -> color; labels outside the fixed palettes get a
            seaborn fallback color
    """
    out = {}
    for lab in label_list:
        if lab in COLOR_ANALYSIS:
            out[lab] = COLOR_ANALYSIS[lab]
        elif lab in COLOR_SEGMENT:
            out[lab] = COLOR_SEGMENT[lab]
        else:
            out[lab] = None

    # fall back to seaborn for labels not in the fixed palettes
    missing = [lab for lab in sorted(label_list) if out[lab] is None]
    if missing:
        fallback = sns.husl_palette(n_colors=len(missing), h=0.9)
        for lab, c in zip(missing, fallback):
            out[lab] = c
    return out


def plot_compute_time(df, title: str = 'Computation Time (per Experiment)') -> None:
    """Draw a per-experiment compute-time boxplot, one row per method label.

    Args:
        df: results DataFrame with label and time_sec columns
        title (str): axes title
    """
    labels_sorted = sorted(df['label'].unique().tolist())
    color_map = get_cmap_dict(labels_sorted)

    plt.title(title)
    plt.xlabel('time (sec)')
    plt.ylabel('')
    sns.boxplot(data=df, x='time_sec', y='label', palette=color_map,
                hue='label')


def plot_calibration(df, alpha_max: float = 0.20, n_pts: int = 200,
                     title: str = None) -> None:
    """Plot FWER calibration curve: nominal alpha vs empirical rejection rate.

    Each method (label) gets its own curve; the diagonal is the reference.

    Args:
        df: results DataFrame; requires a min_pval column (minimum
            FWER-corrected p-value per seed)
        alpha_max (float): right edge of the nominal-alpha axis
        n_pts (int): number of nominal-alpha sample points
        title (str): plot title, or None for the default
    """
    if 'min_pval' not in df.columns:
        print('  (no min_pval column — skipping calibration plot)')
        return

    df2 = df.copy()
    df2['min_pval'] = pd.to_numeric(df2['min_pval'], errors='coerce')
    df2 = df2.dropna(subset=['min_pval'])
    if df2.empty:
        return

    labels_sorted = sorted(df2['label'].unique().tolist())
    color_map = get_cmap_dict(labels_sorted)

    alphas = np.linspace(0, alpha_max, n_pts)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, alpha_max], [0, alpha_max], ls='--', color='grey', lw=1,
            label='ideal')

    for label in labels_sorted:
        pvals = df2.loc[df2['label'] == label, 'min_pval'].values
        n = len(pvals)
        if n == 0:
            continue
        rates = np.array([(pvals <= a).mean() for a in alphas])

        ax.plot(alphas, rates, lw=2.5, color=color_map[label], label=label)

        # binomial 95% CI at nominal alpha = 0.05
        a05 = 0.05
        r05 = (pvals <= a05).mean()
        se = np.sqrt(r05 * (1 - r05) / n) if n > 1 else 0
        ax.errorbar(a05, r05, yerr=1.96 * se, fmt='o', ms=5,
                    color=color_map[label], capsize=3)

    ax.set_xlabel('nominal $\\alpha$')
    ax.set_ylabel('empirical rejection rate')
    ax.set_title(title if title else 'FWER Calibration (Null)')
    ax.legend(frameon=False)
    ax.set_xlim(0, alpha_max)
    ax.set_ylim(0, alpha_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


_METRIC_TITLES = {
    'dice': 'Dice',
    'sens': 'Sensitivity',
    'spec': 'Specificity',
    'pct_max_dice': r'Dice$(\hat{r})\;/\;\max_r$ Dice$(r)$',
    'n_selected': 'Num Selected Regions',
}

_X_PARAM_LABELS = {
    'effect_llr': 'Effect LLR',
    'effect_perc': 'Effect Size (% of Volume)',
    'num_img': 'Number of Subjects',
}


def plot_x_vs_metrics(df, x_param: str = 'effect_llr',
                      metrics: list = ['dice', 'sens', 'spec'],
                      one_vs_rest: bool = False, one_labels: list = None,
                      alpha: float = .5, ci: int = 90,
                      title: str = None, ylabel: str = None) -> None:
    """Plot each metric vs x_param, one bold mean curve per method label.

    The top row shows mean + shaded percentile band per label. When
    one_vs_rest is set, extra rows show each one_label's metric minus the
    best of the other methods, per trial and on average.

    Args:
        df: results DataFrame with label, seed, x_param, and metric columns
        x_param (str): column to use for the x-axis
        metrics (list): metric column names, one subplot column each
        one_vs_rest (bool): add per-label difference rows below the top row
        one_labels (list): labels to difference; defaults to all GLOW variants
        alpha (float): grid line alpha
        ci (int): central percentile width for the shaded band
        title (str): per-subplot title override, or None for the metric name
        ylabel (str): y-axis label of the top-left subplot, or None for "score"
    """
    # ensure numeric x + metrics (prevents lexicographic sorts)
    df2 = df.copy()
    df2[x_param] = pd.to_numeric(df2[x_param], errors='coerce')
    for m in metrics:
        df2[m] = pd.to_numeric(df2[m], errors='coerce')
    df2 = df2.dropna(subset=[x_param])

    # aggregate to unique (label, seed, x_param)
    df_agg = (
        df2.groupby(['label', 'seed', x_param], as_index=False)[metrics]
        .mean()
    )

    labels_sorted = sorted(df_agg['label'].unique().tolist())
    color_map = get_cmap_dict(labels_sorted)

    # one diff row per one_label; default to all GLOW variants present
    if one_vs_rest:
        if one_labels is None:
            one_labels = [l for l in labels_sorted if l.startswith('GLOW')]
        diff_labels = [l for l in one_labels if l in labels_sorted]
    else:
        diff_labels = []

    nrows = 1 + len(diff_labels)
    fig, axes = plt.subplots(
        nrows, len(metrics),
        figsize=(14, 3.0 + 2.5 * len(diff_labels)),
        sharex='col', squeeze=False,
    )

    # percentiles for shading
    lower_q = (100 - ci) / 2
    upper_q = 100 - lower_q

    for j, metric in enumerate(metrics):
        ax_top = axes[0, j]

        # top: bold mean + shaded percentile band
        for label, sub in df_agg.groupby('label'):
            color = color_map[label]

            g_stats = (
                sub.groupby(x_param)[metric]
                .agg(['mean',
                      lambda s: np.percentile(s, lower_q),
                      lambda s: np.percentile(s, upper_q)])
                .reset_index()
                .sort_values(x_param)
            )
            g_stats.columns = [x_param, 'mean', 'q_low', 'q_high']

            ax_top.plot(
                g_stats[x_param], g_stats['mean'],
                lw=3, color=color, label=label
            )
            ax_top.fill_between(
                g_stats[x_param], g_stats['q_low'], g_stats['q_high'],
                color=color, alpha=0.2
            )

        if j == 0:
            ax_top.legend(frameon=False)
        ax_top.set_title(title if title else _METRIC_TITLES.get(metric, metric))
        if metric == 'spec':
            ax_top.set_ylim(0, 1)
        ax_top.grid(True, alpha=alpha, linewidth=1.2)

        # diff rows: each one_label - best of non-(diff_labels) methods
        if diff_labels:
            pivot = (
                df_agg.pivot_table(
                    index=['seed', x_param], columns='label', values=metric
                )
                .reset_index()
            )
            others = [c for c in pivot.columns
                      if c not in {'seed', x_param}
                      and c not in diff_labels]

            for i, one_label in enumerate(diff_labels):
                ax_bot = axes[1 + i, j]

                if one_label not in pivot.columns or not others:
                    ax_bot.axis('off')
                    continue

                valid = pivot[one_label].notna() & pivot[others].notna().any(axis=1)
                pv = pivot.loc[valid].copy()
                if pv.empty:
                    ax_bot.axis('off')
                    continue

                pv['best_other'] = pv[others].max(axis=1, skipna=True)
                pv['diff'] = pv[one_label] - pv['best_other']

                for _, seed_df in pv.groupby('seed'):
                    seed_df = seed_df.sort_values(x_param)
                    ax_bot.plot(
                        seed_df[x_param], seed_df['diff'],
                        lw=0.5, color='black', alpha=0.3,
                    )

                mean_diff = (
                    pv.groupby(x_param)['diff'].mean()
                    .reset_index()
                    .sort_values(x_param)
                )
                ax_bot.plot(
                    mean_diff[x_param], mean_diff['diff'],
                    lw=3, color='black',
                )

                if j == 0:
                    ax_bot.set_ylabel(f'{one_label} - best other')
                ax_bot.set_ylim(-1, 1)
                ax_bot.axhline(0, lw=.5, color='black', alpha=alpha)
                ax_bot.grid(True, alpha=alpha, linewidth=1.2)

        # x label only on the bottom-most row
        axes[-1, j].set_xlabel(_X_PARAM_LABELS.get(x_param, x_param))

    # log x-axis if strictly positive
    xmin = df_agg[x_param].min()
    if pd.notnull(xmin) and xmin > 0:
        for r in range(nrows):
            for ax in axes[r]:
                ax.set_xscale('log')

    axes[0, 0].set_ylabel(ylabel if ylabel else 'score')
    plt.tight_layout()


# ---------------------------------------------------------------------------
# run_ana: per-cache dispatch (compute time + calibration / metric sweep)
# ---------------------------------------------------------------------------

def _savefig(path) -> None:
    """Save the current matplotlib figure to path (tight bbox) and close it."""
    plt.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'saved: {path}')


def plot_ana_cache(label: str, df, cache, out) -> None:
    """Write the run_ana figures for one cache into out.

    Always writes a compute-time boxplot, then one sweep / diagnostic
    figure chosen from what the cache varies (read off cache.iter_kwargs,
    not the data, so the choice matches the config exactly):

      - null cache (effect_llr grid is all 0): FWER calibration from min_pval
      - effect_llr swept: dice / sens / spec vs effect_llr
      - effect extent swept (extenter in iter_kwargs): metrics vs effect
        size (realized support as a fraction of the volume)
      - data source swept (ds in iter_kwargs): metrics vs number of subjects

    The metric sweeps difference each GLOW variant against the best other
    method (plot_x_vs_metrics one_vs_rest). A cache with nothing to sweep
    gets only the compute-time plot.

    Args:
        label (str): cache label; used in titles and output filenames
        df: the cache's results DataFrame (run_ana schema: label, seed,
            effect_llr, dice/sens/spec, min_pval, time_sec, vox_* columns),
            already filtered to the complete in-config trials
        cache (TrialCache): the cache from the config catalogue; its
            iter_kwargs say which parameter the cache sweeps
        out (pathlib.Path): directory the figures are written into
    """
    iter_kwargs = cache.iter_kwargs or {}
    effect_llr_grid = list(iter_kwargs.get('effect_llr', []))

    n_method = df['label'].nunique()
    plt.figure(figsize=(7, 0.5 * n_method + 1.5))
    plot_compute_time(df, title=f'Compute time — {label}')
    _savefig(out / f'{label}_time.pdf')

    if effect_llr_grid and all(v == 0 for v in effect_llr_grid):
        plot_calibration(df, title=f'FWER calibration — {label}')
        _savefig(out / f'{label}_calibration.pdf')
        return

    if len(set(effect_llr_grid)) > 1:
        x_param = 'effect_llr'
    elif 'extenter' in iter_kwargs:
        df = df.copy()
        df['effect_perc'] = df['vox_effect'] / df['vox_total']
        x_param = 'effect_perc'
    elif 'ds' in iter_kwargs:
        x_param = 'num_img'
    else:
        return

    plot_x_vs_metrics(df, x_param=x_param, one_vs_rest=True)
    plt.gcf().suptitle(label, y=1.02, fontsize=13)
    _savefig(out / f'{label}_metrics.pdf')


# ---------------------------------------------------------------------------
# MANCOVA stat comparison (mancova_* caches)
# ---------------------------------------------------------------------------
STAT_ORDER = ['llr', 'pillai', 'wilks', 'hotel_tr', 'roys_root']
STAT_NICE = {
    'llr': 'LLR', 'pillai': 'Pillai', 'wilks': 'Wilks',
    'hotel_tr': 'Hotelling', 'roys_root': "Roy's root",
}
METHOD_ORDER = ['VBA', 'VBA-TFCE', 'CET', 'GLOW']

def _parse_label(label: str):
    """Parse a VBA-TFCE label into (stat, z_flag): 'VBA-TFCE-pillai-z' -> ('pillai', True)."""
    rest = label.removeprefix('VBA-TFCE-')
    if rest.endswith('-z'):
        return rest[:-2], True
    return rest, False


def _parse_label_full(label: str):
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
    """Aggregate to mean / std / count of Dice per (label, effect_llr)."""
    return (df.groupby(['label', 'effect_llr'])['dice']
            .agg(['mean', 'std', 'count'])
            .reset_index())


# ------------------------------------------------------------------
# Figure 1: faceted grid  (2 sources x 5 stats)
# ------------------------------------------------------------------

def plot_facet_grid(datasets):
    """Plot a faceted grid: 5 columns (one per stat) x one row per source.

    Each cell shows raw (solid) + z-scored (dashed); a thin grey line
    shows LLR-raw as a common reference.

    Args:
        datasets: list of (aggregated df, source nice-name); each df comes
            from _agg and has label, effect_llr, mean, std, count columns

    Returns:
        the matplotlib Figure
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
    """Plot side-by-side panels showing the top stats for each source.

    Args:
        datasets: list of (aggregated df, source nice-name) from _agg

    Returns:
        the matplotlib Figure
    """
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
    """Plot per-stat z-scoring improvement (Dice_z - Dice_raw), one panel per source.

    Args:
        datasets: list of (aggregated df, source nice-name) from _agg

    Returns:
        the matplotlib Figure
    """
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
    """Build a summary table: mean Dice, std, and win rate by stat and z-score.

    Args:
        raw_dfs: list of (raw df with a label column, source nice-name);
            labels must be VBA-TFCE-{stat}[-z] format

    Returns:
        a DataFrame with one row per (source, stat, z_scored)
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
    """Build a per-(method, source) table comparing each method's stat variants.

    Args:
        all_dfs: list of (raw df, source nice-name); labels can be any method

    Returns:
        a DataFrame with columns source, method, stat, z_scored, mean_dice,
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


def _plot_mancova(sources, out) -> None:
    """Write the MANCOVA stat-comparison figures + CSV, print the summaries.

    Writes the faceted-grid / summary / z-delta figures plus the best-stat
    CSV into out, from the per-source results that main has already loaded
    and filtered to the in-config trials.

    Args:
        sources (dict): source nice-name (WGN / HCP) -> the combined,
            in-config results DataFrame for that source's mancova caches
        out (pathlib.Path): directory the figures and CSV are written into
    """
    all_dfs = [(df, nice) for nice, df in sources.items()]

    # --- VBA-TFCE detail plots ---
    tfce_datasets = []
    for df, nice in all_dfs:
        df_tfce = df[df['label'].str.startswith('VBA-TFCE-')]
        if df_tfce.empty:
            continue
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


def _load_in_config(label: str, cache):
    """Load a cache's results, keeping only complete trials in the current config.

    Folds any pending per-trial json into the csv (load_update_all), then
    drops rows whose trial_hash is not one cache.iter_trial() would
    produce -- i.e. trials left over from a different config. The kept rows
    are exactly the complete, in-config trials: save_result writes all of a
    trial's rows under one trial_hash, so a present hash means complete.

    Args:
        label (str): cache label / result subfolder name
        cache (TrialCache): the config-catalogue cache whose iter_trial()
            defines the in-config trial set

    Returns:
        the filtered results DataFrame (empty if nothing on disk matches)
    """
    df, _, _ = glow.benchmark.load_update_all(label, verbose=False)
    if df.empty or 'trial_hash' not in df.columns:
        return pd.DataFrame()
    expected = {stable_hash(trial) for trial in cache.iter_trial()}
    return df[df['trial_hash'].astype(str).isin(expected)]


def main() -> None:
    """Plot every cache in the config catalogue from the default results folder.

    Iterates CACHE_BY_LABEL; for each cache keeps only the complete,
    in-config trials (_load_in_config) and dispatches by run_fnc: run_ana
    caches go to plot_ana_cache, run_mancova caches are combined by source
    (WGN / HCP) and handed to _plot_mancova. All figures land in
    results/_latest. Caches with no in-config results on disk are skipped.
    """
    import matplotlib
    matplotlib.use('Agg')
    from .config import CACHE_BY_LABEL
    from .run import run_mancova

    out = glow.benchmark.get_path_result() / '_latest'
    out.mkdir(exist_ok=True)

    mancova_sources = {}
    n_plotted = 0
    for label, (cache, run_fnc) in CACHE_BY_LABEL.items():
        df = _load_in_config(label, cache)
        if df.empty:
            continue
        if getattr(run_fnc, 'func', run_fnc) is run_mancova:
            nice = 'HCP' if 'hcp' in label else 'WGN'
            mancova_sources.setdefault(nice, []).append(df)
        else:
            print(f'\n=== {label} ({len(df)} rows) ===')
            plot_ana_cache(label, df, cache, out)
        n_plotted += 1

    if mancova_sources:
        combined = {nice: pd.concat(dfs, ignore_index=True)
                    for nice, dfs in mancova_sources.items()}
        _plot_mancova(combined, out)

    if n_plotted == 0:
        print(f'no in-config results found in {out.parent}')


if __name__ == '__main__':
    main()
