"""Plot the run_ana benchmark caches from the shared provenance records.

The plotting layer for the run_ana caches. It reads each cache's
provenance frame (results.config_results_df: one wide row per run_ana leaf,
namespaced by the producing function -- run_ana.out.score, the swept
data_factory / effect_factory inputs), normalises it to one tidy row per
(trial, recipe) with tidy_run_ana, and writes one figure set per cache into
results/_latest.

The tidy frame is what the plotters consume: a label (method), a source
(WGN / HCP, which share each cache and face apart here), the swept axes
(effect_llr, b, num_img, the realized effect fraction), and the metrics
(dice / sens / ppv / spec) derived from the score's confusion counts (see
score_effects / glow.mask.stats_from_counts). The x-axis is inferred from
what actually varies in the cache (no config spec): an all-null effect grid
is the FWER calibration path, else the first of effect_llr / b / num_img /
effect_perc that varies is swept.

Each cache then gets either a faceted FWER calibration curve (null) or a
faceted dice/sens/ppv sweep plus the GLOW-Focus head-to-head diff grid, WGN
and HCP side by side. With no arguments it plots every run_ana cache in the
catalogue; passing names restricts it. The mancova / segment / prune figures
are out of scope -- their leaf functions do not exist in this layer yet
(see config).
"""
import colorsys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import glow._extra.benchmark
from .file import add_metric_cols


# ---------------------------------------------------------------------------
# Consistent paper colour palette
# ---------------------------------------------------------------------------
# Base: teal from Fig. 3 (#4DA6A6), H=180 deg S=0.37 L=0.48 in HLS.
# Analysis methods: 4 hues evenly spaced (90 deg apart), same S/L.
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


def get_cmap_dict(label_list) -> dict:
    """Map each label to a color, using the fixed palette where possible.

    Args:
        label_list: labels to assign colors to

    Returns:
        out (dict): label -> color; labels outside the fixed palette get a
            seaborn fallback color
    """
    out = {lab: COLOR_ANALYSIS.get(lab) for lab in label_list}

    # fall back to seaborn for labels not in the fixed palette
    missing = [lab for lab in sorted(label_list) if out[lab] is None]
    if missing:
        fallback = sns.husl_palette(n_colors=len(missing), h=0.9)
        for lab, c in zip(missing, fallback):
            out[lab] = c
    return out


_METRIC_TITLES = {
    'dice': 'Dice',
    'sens': 'Sensitivity',
    'ppv': 'PPV (Precision)',
    'spec': 'Specificity',
}

_X_PARAM_LABELS = {
    'effect_llr': 'Effect LLR',
    'effect_perc': 'Effect Size (% of Volume)',
    'num_img': 'Number of Subjects',
    'b': 'Number of Imaging Features',
}


# ---------------------------------------------------------------------------
# Normalise the provenance frame to one tidy row per (trial, recipe)
# ---------------------------------------------------------------------------

def tidy_run_ana(raw):
    """Normalise a run_ana provenance frame to a tidy per-trial results frame.

    Collapses the wide, function-namespaced frame from
    results.config_results_df (run_ana leaf + its data_factory / effect_factory
    ancestors) into the flat schema the plotters consume. The source is read
    off which data_factory produced the row (wgn / hcp), the swept axes off the
    relevant ancestor inputs, and the metrics off the run_ana.out.score dict's
    confusion counts (glow.mask.stats_from_counts via add_metric_cols).

    Args:
        raw: the provenance DataFrame (one row per run_ana leaf), with
            run_ana.in.label / run_ana.out.score, data_factory_{wgn,hcp}.in.*
            and (when an effect was planted) effect_factory.in.* columns.

    Returns:
        a tidy DataFrame, one row per (trial, recipe), with columns label,
        source (WGN / HCP), seed, b, num_img, effect_llr, time_sec, the four
        confusion counts (tp/fp/tn/fn), min_pval, num_vox, n_pred, the realized
        vox_effect / vox_total / effect_perc, and the derived
        dice/sens/ppv/spec (empty in, empty out).
    """
    if raw.empty:
        return raw

    def col(name):
        """Return raw[name], or an all-NaN column when absent."""
        if name in raw.columns:
            return raw[name]
        return pd.Series(np.nan, index=raw.index)

    wgn_seed = pd.to_numeric(col('data_factory_wgn.in.seed'), errors='coerce')
    hcp_seed = pd.to_numeric(col('data_factory_hcp.in.seed'), errors='coerce')

    out = pd.DataFrame(index=raw.index)
    out['label'] = col('run_ana.in.label')
    # the row's source is whichever data_factory produced its clean experiment
    out['source'] = np.where(hcp_seed.notna(), 'HCP', 'WGN')
    out['seed'] = wgn_seed.fillna(hcp_seed)

    # b: WGN carries it directly; HCP is the length of its feature subset
    hcp_b = col('data_factory_hcp.in.hcp_feats').map(
        lambda v: len(v) if isinstance(v, (list, tuple)) else np.nan)
    out['b'] = pd.to_numeric(col('data_factory_wgn.in.b'),
                             errors='coerce').fillna(hcp_b)
    # num_img is a WGN axis only (HCP's N is its cohort), so HCP rows stay NaN
    out['num_img'] = pd.to_numeric(col('data_factory_wgn.in.num_img'),
                                   errors='coerce')
    out['effect_llr'] = pd.to_numeric(col('effect_factory.in.effect_llr'),
                                      errors='coerce')
    out['time_sec'] = pd.to_numeric(col('run_ana.time_sec'), errors='coerce')

    # explode the score dict: the union-target confusion counts, plus the
    # global min_pval / num_vox / n_pred
    score = col('run_ana.out.score')

    def field(key, subkey=None):
        """Pull score[key] (or score[key][subkey]) per row, NaN when absent."""
        def get(s):
            if not isinstance(s, dict):
                return np.nan
            value = s.get(key)
            if subkey is not None:
                value = (value.get(subkey)
                         if isinstance(value, dict) else np.nan)
            return value
        return score.map(get)

    for cnt in ('tp', 'fp', 'tn', 'fn'):
        out[cnt] = pd.to_numeric(field('target', cnt), errors='coerce')
    out['min_pval'] = pd.to_numeric(field('min_pval'), errors='coerce')
    out['num_vox'] = pd.to_numeric(field('num_vox'), errors='coerce')
    out['n_pred'] = pd.to_numeric(field('n_pred'), errors='coerce')

    # realized effect support (target positives) over the analyzed volume
    out['vox_effect'] = out['tp'] + out['fn']
    out['vox_total'] = out['num_vox']
    out['effect_perc'] = out['vox_effect'] / out['vox_total']

    # dice/sens/ppv/spec from the four counts (glow.mask is the source)
    return add_metric_cols(out)


def _infer_x(df) -> str:
    """Infer the swept x-axis column from what varies in a tidy frame.

    No config spec is read: an all-null effect grid is the FWER calibration
    path (returns None); otherwise the first of effect_llr / b / num_img /
    effect_perc that takes more than one value is the swept axis. The shared
    baseline cell holds every other axis fixed, so exactly one varies.

    Args:
        df: a tidy_run_ana frame.

    Returns:
        the x-axis column name, or None for the null / calibration path.
    """
    if not df['effect_llr'].notna().any():
        return None
    for cand in ('effect_llr', 'b', 'num_img', 'effect_perc'):
        if df[cand].dropna().nunique() > 1:
            return cand
    return 'effect_llr'


# ---------------------------------------------------------------------------
# FWER calibration (null caches)
# ---------------------------------------------------------------------------

def plot_calibration(df, alpha_max: float = 0.20, n_pts: int = 200,
                     title: str = None, ax=None) -> None:
    """Plot FWER calibration curve: nominal alpha vs empirical rejection rate.

    Each method (label) gets its own curve; the diagonal is the reference.

    Args:
        df: tidy results frame; requires a min_pval column (smallest
            FWER-corrected region p-value per trial)
        alpha_max (float): right edge of the nominal-alpha axis
        n_pts (int): number of nominal-alpha sample points
        title (str): plot title, or None for the default
        ax: matplotlib Axes to draw into; None makes its own square figure
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

    owns_fig = ax is None
    if owns_fig:
        _, ax = plt.subplots(figsize=(5, 5))
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
    if owns_fig:
        plt.tight_layout()


def _plot_calibration_faceted(label: str, df, out,
                              facet: str = 'source') -> None:
    """Lay out one FWER-calibration axes per facet value, side by side.

    Args:
        label (str): cache name; used in titles and the output filename
        df: the null cache's tidy results (needs min_pval, label, facet)
        out (pathlib.Path): directory the figure is written into
        facet (str): categorical column to split across axes
    """
    sources = sorted(df[facet].dropna().unique().tolist())
    fig, axes = plt.subplots(1, len(sources), figsize=(5 * len(sources), 5),
                             squeeze=False)
    for ax, src in zip(axes[0], sources):
        plot_calibration(df[df[facet] == src], title=f'{label} — {src}', ax=ax)
    fig.tight_layout()
    path = out / f'{label}_calibration.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'saved: {path}')


# ---------------------------------------------------------------------------
# Metric sweeps (faceted by source: WGN | HCP)
# ---------------------------------------------------------------------------

def plot_metric_grid(label: str, df, *, x: str, metrics: list,
                     facet: str = 'source', hue: str = 'label',
                     ci: int = 90, out=None) -> None:
    """Plot a faceted metric sweep: col=facet, row=metric, one curve per hue.

    Puts the swept axis on the x, the facet (source) across panel columns,
    and each metric on its own panel row, aggregating the seed replicates
    into a mean + percentile band -- so WGN and HCP sit side by side.

    Args:
        label (str): cache name; used in the title and output filename
        df: the cache's tidy results
        x (str): column for the x-axis
        metrics (list): metric columns, one panel row each
        facet (str): categorical column spread across panel columns
        hue (str): column mapped to line colour (the method label)
        ci (int): central percentile-interval width for the band
        out (pathlib.Path): directory the figure is written into
    """
    df = df.copy()
    for c in [x, *metrics]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=[x])

    long = df.melt(id_vars=[x, facet, hue], value_vars=metrics,
                   var_name='metric', value_name='value')
    long['metric'] = long['metric'].map(lambda m: _METRIC_TITLES.get(m, m))

    palette = get_cmap_dict(sorted(df[hue].dropna().unique().tolist()))
    g = sns.relplot(
        data=long, x=x, y='value', hue=hue, col=facet, row='metric',
        kind='line', estimator='mean', errorbar=('pi', ci), palette=palette,
        facet_kws=dict(sharey='row', sharex=True), height=2.6, aspect=1.5)

    xmin = df[x].min()
    if pd.notnull(xmin) and xmin > 0:
        g.set(xscale='log')
    g.set_axis_labels(_X_PARAM_LABELS.get(x, x), 'score')
    g.figure.suptitle(label, y=1.02, fontsize=13)
    path = out / f'{label}_metrics.pdf'
    g.figure.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'saved: {path}')


def plot_metric_diff_grid(label: str, df, *, x: str, metrics: list,
                          one_label: str = 'GLOW-Focus',
                          facet: str = 'source', hue: str = 'label',
                          alpha: float = .5, out=None) -> None:
    """Plot one_label minus the best alternative: col=facet, row=metric.

    The head-to-head companion to plot_metric_grid. Each panel shows the
    per-trial advantage of one_label (default GLOW-Focus) over the best
    competing method -- the largest metric among the non-GLOW labels (VBA /
    VBA-TFCE / CET) at the same (seed, x). A thin black line per seed plus a
    bold black mean make the win / loss against the field legible; the dashed
    zero line is break-even. PPV is undefined for trials with no detections
    (nan; see glow.mask.stats_from_counts) so those drop out of the difference.

    Also writes {label}_diff.csv: one row per (facet, method, x) -- a block for
    every GLOW variant present, each against the same best non-GLOW
    alternative -- with a glow_<m> / other_<m> / <m>_diff / <m>_win block per
    metric. Only one_label's line is drawn. The figure is skipped (nothing
    written) when one_label is absent or no non-GLOW alternative exists.

    Args:
        label (str): cache name; used in the title and output filename
        df: the cache's tidy results
        x (str): column for the x-axis
        metrics (list): metric columns, one panel row each
        one_label (str): the method differenced against the field
        facet (str): categorical column spread across panel columns
        hue (str): the method-label column; its GLOW* values are excluded
            from the "best alternative" pool
        alpha (float): grid / zero-line alpha
        out (pathlib.Path): directory the figure and CSV are written into
    """
    df = df.copy()
    for col in [x, *metrics]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[x])

    if one_label not in set(df[hue].unique()):
        print(f'  ({one_label} absent — skipping {label} diff grid)')
        return

    sources = sorted(df[facet].dropna().unique().tolist())
    nrows, ncols = len(metrics), len(sources)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 2.8 * nrows),
                             sharex=True, sharey='row', squeeze=False)

    # tidy rows behind the bold mean line, for the companion CSV
    diff_rows = []

    for j, src in enumerate(sources):
        # collapse replicate rows to one value per (label, seed, x)
        agg = (df[df[facet] == src]
               .groupby([hue, 'seed', x], as_index=False)[metrics].mean())
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            pivot = agg.pivot_table(index=['seed', x], columns=hue,
                                    values=metric).reset_index()
            others = [c for c in pivot.columns
                      if c not in {'seed', x}
                      and not str(c).startswith('GLOW')]
            glow_cols = [c for c in pivot.columns
                         if c not in {'seed', x} and str(c).startswith('GLOW')]

            if not others or not glow_cols:
                ax.axis('off')
                continue
            pivot['best_other'] = pivot[others].max(axis=1, skipna=True)

            # each GLOW variant vs the best non-GLOW alternative: the CSV gets
            # a row block per variant (method column); the panel draws only
            # one_label's per-seed + bold mean line
            drew = False
            for gl in glow_cols:
                valid = pivot[gl].notna() & pivot['best_other'].notna()
                pv = pivot.loc[valid, ['seed', x, gl, 'best_other']].copy()
                if pv.empty:
                    continue
                pv['diff'] = pv[gl] - pv['best_other']
                pv['win'] = (pv[gl] > pv['best_other']).astype(float)
                agg_x = (pv.groupby(x).agg(
                            mean_diff=('diff', 'mean'),
                            glow=(gl, 'mean'),
                            other=('best_other', 'mean'),
                            win=('win', 'mean'),
                            n_seed=('diff', 'size'))
                         .reset_index().sort_values(x))
                for _, row in agg_x.iterrows():
                    diff_rows.append({facet: src, 'method': gl, x: row[x],
                                      'metric': metric, 'glow': row['glow'],
                                      'other': row['other'],
                                      'mean_diff': row['mean_diff'],
                                      'win': row['win'],
                                      'n_seed': int(row['n_seed'])})

                if gl == one_label:
                    for _, seed_df in pv.groupby('seed'):
                        seed_df = seed_df.sort_values(x)
                        ax.plot(seed_df[x], seed_df['diff'],
                                lw=0.5, color='black', alpha=0.3)
                    ax.plot(agg_x[x], agg_x['mean_diff'], lw=3, color='black')
                    drew = True

            if not drew:
                ax.axis('off')
                continue

            ax.axhline(0, lw=.5, color='black', alpha=alpha)
            ax.set_ylim(-1, 1)
            ax.grid(True, alpha=alpha, linewidth=1.2)
            if i == 0:
                ax.set_title(f'{facet} = {src}')
            if j == 0:
                ax.set_ylabel(_METRIC_TITLES.get(metric, metric))
            if i == nrows - 1:
                ax.set_xlabel(_X_PARAM_LABELS.get(x, x))

    xmin = df[x].min()
    if pd.notnull(xmin) and xmin > 0:
        for ax in axes.flat:
            ax.set_xscale('log')

    fig.suptitle(f'{label}: {one_label} − best alternative',
                 y=1.02, fontsize=13)
    fig.tight_layout()
    path = out / f'{label}_diff.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'saved: {path}')

    if diff_rows:
        _write_diff_csv(label, pd.DataFrame(diff_rows), x=x, facet=facet,
                        metrics=metrics, out=out)


def _write_diff_csv(label: str, diff_long, *, x: str, facet: str,
                    metrics: list, out) -> None:
    """Write the diff-grid mean line to CSV and print where Dice peaks.

    diff_long is the tidy mean line behind plot_metric_diff_grid, one row per
    (facet, method, x, metric): the seed-averaged absolute scores (glow = the
    GLOW variant named in method, other = best non-GLOW alternative), their
    difference (mean_diff), and the win rate (win = fraction of trials with
    the GLOW variant strictly above the best alternative). It is reshaped to
    one row per (facet, method, x) with a glow_<m> / other_<m> / <m>_diff /
    <m>_win block per metric, written to {label}_diff.csv; then the x of the
    maximum mean Dice delta is printed per (facet, method).

    Args:
        label (str): cache name; used in the output filename
        diff_long: tidy DataFrame (facet, method, x, metric, glow, other,
            mean_diff, win, n_seed)
        x (str): the x-axis column name (becomes a CSV column)
        facet (str): the facet column name (becomes a CSV column)
        metrics (list): metric names, fixing the column block order
        out (pathlib.Path): directory the CSV is written into
    """
    # one column block per metric, grouped: glow / other / diff / win
    keys = [facet, 'method', x]
    wide = diff_long[keys].drop_duplicates().sort_values(keys)
    for m in metrics:
        sub = diff_long[diff_long['metric'] == m]
        if sub.empty:
            continue
        sub = sub.rename(columns={'glow': f'glow_{m}', 'other': f'other_{m}',
                                  'mean_diff': f'{m}_diff', 'win': f'{m}_win'})
        wide = wide.merge(
            sub[keys + [f'glow_{m}', f'other_{m}', f'{m}_diff', f'{m}_win']],
            on=keys, how='left')
    csv_path = out / f'{label}_diff.csv'
    wide.to_csv(csv_path, index=False, float_format='%.4f')
    print(f'saved: {csv_path}')

    if 'dice_diff' not in wide.columns:
        return
    # peak mean Dice delta per (facet, method): scores, delta, win, other
    # deltas
    n_dice = (diff_long[diff_long['metric'] == 'dice']
              .set_index([facet, 'method', x])['n_seed'])
    print('  GLOW variant − best alternative, peak mean Dice delta:')
    for (src, method), sub in wide.groupby([facet, 'method']):
        sub = sub.dropna(subset=['dice_diff'])
        if sub.empty:
            continue
        peak = sub.loc[sub['dice_diff'].idxmax()]
        n = int(n_dice.get((src, method, peak[x]), 0))
        others = '  '.join(
            f'{m}={peak[f"{m}_diff"]:+.4f}'
            for m in metrics if m != 'dice' and f'{m}_diff' in wide.columns
            and pd.notnull(peak[f'{m}_diff']))
        print(f'    {facet}={src} {method}: at {x}={peak[x]:g} (n={n})  '
              f'dice {peak["glow_dice"]:.4f} vs {peak["other_dice"]:.4f} '
              f'(Δ{peak["dice_diff"]:+.4f}, win {peak["dice_win"]:.0%})  '
              + others)


# ---------------------------------------------------------------------------
# Per-cache dispatch + CLI
# ---------------------------------------------------------------------------

def plot_cache(label: str, df, out,
               metrics: list = ['dice', 'sens', 'ppv']) -> None:
    """Write one run_ana cache's figures, dispatching on its swept axis.

    The null path (no effect planted) gets a faceted FWER calibration curve;
    every other cache gets the faceted metric sweep plus the GLOW-Focus
    head-to-head diff grid. The x-axis is inferred from the data (_infer_x),
    so no config plot spec is needed.

    Args:
        label (str): cache name; used in titles and output filenames
        df: the cache's tidy_run_ana results
        out (pathlib.Path): directory the figures are written into
        metrics (list): metric columns plotted as the sweep panel rows
    """
    if df.empty:
        print(f'  (no rows for {label} — skipping)')
        return

    x = _infer_x(df)
    if x is None:
        _plot_calibration_faceted(label, df, out)
        return

    plot_metric_grid(label, df, x=x, metrics=metrics, out=out)
    plot_metric_diff_grid(label, df, x=x, metrics=metrics, out=out)


def main(argv=None) -> None:
    """Plot the run_ana caches from the shared provenance records.

    For each selected run_ana cache (every one in CONFIG by default, or the
    names given on the command line), reads its provenance frame
    (results.config_results_df), normalises it with tidy_run_ana, and hands it
    to plot_cache. Figures land in results/_latest, so a mid-benchmark run
    yields intermediate figures. Non-run_ana CONFIG entries are skipped.

    Args:
        argv (list | None): CLI args to parse; None reads sys.argv. Positional
            args are cache names (e.g. sweep_llr); with none, every run_ana
            cache in the catalogue is plotted.
    """
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    from .config import CONFIG
    from .run import run_ana
    from . import results

    parser = argparse.ArgumentParser(
        description='Plot run_ana benchmark figures from the records.')
    parser.add_argument(
        'names', nargs='*',
        help='cache names to plot (e.g. sweep_llr); '
             'default: every run_ana cache in the catalogue')
    args = parser.parse_args(argv)

    # the leaf function fixes whether a cache is a run_ana cache (CONFIG values
    # are (data, effect, fnc_kwargs, fnc))
    ana_names = [n for n, cfg in CONFIG.items() if cfg[3] is run_ana]
    if args.names:
        unknown = [n for n in args.names if n not in CONFIG]
        if unknown:
            parser.error(f'unknown cache name(s): {", ".join(unknown)}')
        names = args.names
    else:
        names = ana_names

    out = glow._extra.benchmark.get_path_result() / '_latest'
    out.mkdir(exist_ok=True)

    n_plotted = 0
    for name in names:
        if name not in ana_names:
            print(f'  ({name} is not a run_ana cache — skipping)')
            continue
        df = tidy_run_ana(results.config_results_df(name))
        if df.empty:
            print(f'  (no records for {name} — skipping)')
            continue
        print(f'\n=== {name}: {len(df)} run_ana rows ===')
        plot_cache(name, df, out)
        n_plotted += 1

    if n_plotted == 0:
        print(f'no run_ana results found under {out.parent}')


if __name__ == '__main__':
    main()
