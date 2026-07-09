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

Each cache then gets either a faceted FWER calibration curve (null) or one
stacked detection figure: an HCP block over a WGN block, each a 2 x 3 grid
whose top row is the per-method mean score + central 95% percentile band and
whose bottom row is the GLOW-Focus head-to-head diff, over the dice / sens /
ppv columns. Alongside it a discovery-threshold table (write_threshold_table)
records the effect strength at which each method's mean Dice crosses 0.5,
normalised to GLOW-Focus (thr_method / thr_glow) with a column per b.

The runtime family is plotted apart (tidy_runtime / plot_runtime): those caches
hold detection fixed and sweep one cost knob, so the signal is the leaf wall
time (RECORDER time_sec), not a score. Each is one wall-time-vs-knob curve per
method on log axes -- runtime (the paper figure) over the num_vox sweep to the
full HCP support, and the diagnostic caches over the segmentation / permutation
/ feature-count knobs (see _RUNTIME_SPEC).

The inner-edge cache (sweep_n_perm_inner) is plotted apart too (tidy_inner_edge
/ plot_inner_edge): its run_inner_edge leaf records the per-outer-perm max-z at
each num_inner_perm, from which the FWER critical value is derived, giving one
threshold-vs-num_inner_perm convergence curve per GLOW arm (and the companion
threshold JSON the recommended n_perm_inner is read off).

The race-retention cache (race_maxz) is plotted apart too (tidy_race_maxz /
plot_race_maxz): its run_race_maxz leaf records each outer perm's max-z under
the full cpu_perm and the survivor race, giving a race-vs-full scatter on y=x
per source (and the companion retention JSON: the max abs difference and
mismatch count that certify the race keeps the max-z).

With no arguments the CLI plots every detection, runtime, inner-edge, and
race-retention cache in the catalogue; passing names restricts it. The
remaining caches (segment / stat / prune / min_size) carry different leaf and
score shapes, so this layer does not plot them (see config).
"""
import colorsys
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import glow._extra.benchmark
from .config import ana_kwargs_dict, RUNTIME_GLOW_MODES
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

# recover a run_ana leaf's method name from its recorded recipe: config's
# label -> ana map, inverted on the ana repr (the address-free recipe id
# Analysis.__repr__ renders, = the recorded run_ana.in.ana cell). config owns
# the labels; run_ana neither takes nor records one (see config / run).
_LABEL_OF_ANA = {repr(ana): label for label, ana in ana_kwargs_dict.items()}

# recover a run_inner_edge leaf's GLOW arm from its recorded cluster_mode.
# ClusterMode is a StrEnum, so the recorded in.cluster_mode cell is its string
# ('Focus' / 'GLM Error'); no label is stored (see run.run_inner_edge / config).
_ARM_OF_MODE = {str(mode): label for label, mode in RUNTIME_GLOW_MODES}


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
    'num_vox': 'Number of Voxels',
    'n_perm_fwer': 'FWER Permutations',
    'n_perm_inner': 'Inner (Freedman-Lane) Permutations',
}

# runtime caches: name -> (leaf column prefix, swept x-axis column). The
# runtime family plots wall time (leaf.time_sec) against one swept cost knob;
# unlike the detection sweeps the x is not inferred (time is the signal, the
# effect is held at the moderate default). run_ana_time (runtime / runtime_b)
# carries the method in the recipe (in.ana); every timing leaf returns num_vox
# bare, and run_segment_time / run_perm_* record the method as an explicit
# label. See config's runtime section.
_RUNTIME_SPEC = {
    'runtime':              ('run_ana_time',     'num_vox'),
    'runtime_b':            ('run_ana_time',     'b'),
    'runtime_segment':      ('run_segment_time', 'num_vox'),
    'runtime_n_perm_fwer':  ('run_perm_fwer',    'n_perm_fwer'),
    'runtime_n_perm_inner': ('run_perm_inner',   'n_perm_inner'),
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
    relevant ancestor inputs, and the metrics off the recursed score columns
    (run_ana.out.score.target.{tp,fp,tn,fn}; glow.mask.stats_from_counts via
    add_metric_cols).

    Args:
        raw: the provenance DataFrame (one row per run_ana leaf), with
            run_ana.in.ana (mapped to the method label via _LABEL_OF_ANA), the
            recursed run_ana.out.score.* columns, data_factory_{wgn,hcp}.in.*
            and (when an effect was planted) effect_factory_single.in.* columns
            (the recorded builder, not the effect_factory dispatcher).

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
    out['label'] = col('run_ana.in.ana').map(_LABEL_OF_ANA)
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
    # the RECORDER logs the concrete builder effect_factory dispatches to
    # (single / split), not the dispatcher, so the column is prefixed by it
    out['effect_llr'] = pd.to_numeric(
        col('effect_factory_single.in.effect_llr'), errors='coerce').fillna(
        pd.to_numeric(col('effect_factory_split.in.effect_llr'),
                      errors='coerce'))
    out['time_sec'] = pd.to_numeric(col('run_ana.time_sec'), errors='coerce')

    # run_ana recurses 'score', so flatten_to_df expands the dict into
    # out.score.<path> columns: the union-target confusion counts plus the
    # global min_pval / num_vox / n_pred (col -> all-NaN when absent)
    base = 'run_ana.out.score'
    for cnt in ('tp', 'fp', 'tn', 'fn'):
        out[cnt] = pd.to_numeric(col(f'{base}.target.{cnt}'), errors='coerce')
    out['min_pval'] = pd.to_numeric(col(f'{base}.min_pval'), errors='coerce')
    out['num_vox'] = pd.to_numeric(col(f'{base}.num_vox'), errors='coerce')
    out['n_pred'] = pd.to_numeric(col(f'{base}.n_pred'), errors='coerce')

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
    effect_perc that takes more than one value is the swept axis. A cache may
    vary a second, structural axis alongside it (the llr sweep varies b too);
    that one is not the x -- plot_cache holds it fixed per figure via
    _split_by_secondary, so exactly one axis moves in any drawn frame.

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


# Structural axes a detection cache may vary alongside its swept x. The metric
# grids already spend both facet dimensions (col=source, row=metric) with the
# method as hue, so a cache that also moves one of these (the llr sweep varies
# b as well as effect_llr) is drawn one figure per value rather than crammed
# into a third facet -- see _split_by_secondary.
_SECONDARY_AXES = ('b', 'num_img')


def _split_by_secondary(label: str, df, x: str):
    """Yield (sub_label, sub_df) per value of a secondary axis that varies.

    _infer_x gives the swept x; a cache that also varies a structural axis
    (b / num_img) besides it is split so every drawn figure holds that axis
    fixed -- the combined llr sweep yields sweep_llr_b1 / _b2 / _b3, matching
    the per-b figures the separate caches used to produce. With nothing else
    varying, yields (label, df) unchanged.

    Args:
        label (str): the cache name; the sub-label's prefix
        df: a tidy_run_ana frame
        x (str): the swept x-axis column (never split on)

    Yields:
        (str, DataFrame): a label suffixed with the held value (e.g.
            sweep_llr_b1) and the matching sub-frame.
    """
    extra = [a for a in _SECONDARY_AXES
             if a != x and df[a].dropna().nunique() > 1]
    if not extra:
        yield label, df
        return
    for values, sub in df.groupby(extra):
        values = values if isinstance(values, tuple) else (values,)
        suffix = ''.join(f'_{a}{int(v)}' for a, v in zip(extra, values))
        yield f'{label}{suffix}', sub


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

    labels_sorted = sorted(df2['label'].dropna().unique().tolist())
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
# Metric sweeps (one stacked figure: an HCP block over a WGN block)
# ---------------------------------------------------------------------------

def _draw_metric_band(ax, df, x: str, metric: str, palette: dict, *,
                      ci: int = 95, hue: str = 'label') -> None:
    """Draw a mean line plus a central ci% percentile band per method into ax.

    One curve per method (hue), the band spanning the (100-ci)/2 .. (100+ci)/2
    percentiles of the seed replicates at each x (so the default ci=95 shades
    the 2.5th-97.5th percentile). PPV is undefined for trials with no
    detections (nan; glow.mask.stats_from_counts) and drops out of both.

    Args:
        ax: matplotlib Axes to draw into
        df: one source's tidy rows (numeric x / metric, dropna'd on x)
        x (str): the swept x-axis column
        metric (str): the metric column plotted on the y-axis
        palette (dict): label -> colour
        ci (int): central percentile-interval width for the band
        hue (str): the method-label column
    """
    lo_q, hi_q = (1 - ci / 100) / 2, (1 + ci / 100) / 2
    for label in sorted(df[hue].dropna().unique().tolist()):
        g = df[df[hue] == label].groupby(x)[metric]
        mean, lo, hi = g.mean(), g.quantile(lo_q), g.quantile(hi_q)
        ax.plot(mean.index, mean.values, lw=2, color=palette[label],
                label=label)
        ax.fill_between(mean.index, lo.values, hi.values,
                        color=palette[label], alpha=0.15)


def _draw_diff(ax, df, x: str, metric: str, *, one_label: str = 'GLOW-Focus',
               hue: str = 'label', alpha: float = .5) -> list:
    """Draw one_label minus the best non-GLOW method into ax; return CSV rows.

    The head-to-head panel: the per-trial advantage of one_label (default
    GLOW-Focus) over the best competing method -- the largest metric among the
    non-GLOW labels (VBA / VBA-TFCE / CET) at the same (seed, x). A thin line
    per seed plus a bold mean make the win / loss against the field legible;
    the zero line is break-even. Only one_label's line is drawn, but the
    returned rows cover every GLOW variant (each vs the same best alternative)
    for the companion CSV. The axis is turned off when one_label draws nothing.

    Args:
        ax: matplotlib Axes to draw into
        df: one source's tidy rows (numeric x / metric, dropna'd on x)
        x (str): the swept x-axis column
        metric (str): the metric column differenced on the y-axis
        one_label (str): the method whose line is drawn
        hue (str): the method-label column; GLOW* values are the "glow" pool,
            the rest the "best alternative" pool
        alpha (float): grid / zero-line alpha

    Returns:
        list[dict]: one row per (method, x) -- method (the GLOW variant), x,
            metric, glow / other (seed-mean scores), mean_diff, win, n_seed.
    """
    agg = df.groupby([hue, 'seed', x], as_index=False)[metric].mean()
    pivot = agg.pivot_table(index=['seed', x], columns=hue,
                            values=metric).reset_index()
    others = [c for c in pivot.columns
              if c not in {'seed', x} and not str(c).startswith('GLOW')]
    glow_cols = [c for c in pivot.columns
                 if c not in {'seed', x} and str(c).startswith('GLOW')]
    if not others or not glow_cols:
        ax.axis('off')
        return []
    pivot['best_other'] = pivot[others].max(axis=1, skipna=True)

    rows, drew = [], False
    for gl in glow_cols:
        valid = pivot[gl].notna() & pivot['best_other'].notna()
        pv = pivot.loc[valid, ['seed', x, gl, 'best_other']].copy()
        if pv.empty:
            continue
        pv['diff'] = pv[gl] - pv['best_other']
        pv['win'] = (pv[gl] > pv['best_other']).astype(float)
        agg_x = (pv.groupby(x).agg(
                    mean_diff=('diff', 'mean'), glow=(gl, 'mean'),
                    other=('best_other', 'mean'), win=('win', 'mean'),
                    n_seed=('diff', 'size'))
                 .reset_index().sort_values(x))
        for _, row in agg_x.iterrows():
            rows.append({'method': gl, x: row[x], 'metric': metric,
                         'glow': row['glow'], 'other': row['other'],
                         'mean_diff': row['mean_diff'], 'win': row['win'],
                         'n_seed': int(row['n_seed'])})
        if gl == one_label:
            for _, seed_df in pv.groupby('seed'):
                seed_df = seed_df.sort_values(x)
                ax.plot(seed_df[x], seed_df['diff'], lw=0.5, color='black',
                        alpha=0.3)
            ax.plot(agg_x[x], agg_x['mean_diff'], lw=3, color='black')
            drew = True

    if not drew:
        ax.axis('off')
        return rows
    ax.axhline(0, lw=.5, color='black', alpha=alpha)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=alpha, linewidth=1.2)
    return rows


# WGN / HCP stack top-to-bottom, so the fixed order puts HCP first; a source
# absent from the cache (sweep_nimg is WGN-only) just drops out.
_SOURCE_ORDER = ('HCP', 'WGN')


def plot_source_grid(label: str, df, *, x: str, metrics: list, out,
                     one_label: str = 'GLOW-Focus', ci: int = 95,
                     thresh_metric: str = 'dice', level: float = 0.5) -> None:
    """Plot the stacked per-source detection figure: two rows per source.

    One SubFigure per source (its banner the source name), stacked HCP over
    WGN; within each a 2 x len(metrics) grid whose top row is the mean score +
    central ci% percentile band per method (_draw_metric_band) and whose bottom
    row is one_label minus the best non-GLOW alternative (_draw_diff), with the
    metrics (dice / sens / ppv) across the columns. A dashed line marks the
    threshold level on the thresh_metric (Dice) panel, where the discovery
    thresholds (write_threshold_table, plotted once per cache) are read.

    Writes {label}.pdf and the companion {label}_diff.csv (one block per GLOW
    variant; see _write_diff_csv).

    Args:
        label (str): cache name; the output filename stem and figure title
        df: the cache's tidy_run_ana results (needs source / label / seed / x /
            the metric columns)
        x (str): the swept x-axis column
        metrics (list): metric columns, one panel column each
        out (pathlib.Path): directory the figure and CSV are written into
        one_label (str): the method the diff row draws against the field
        ci (int): central percentile-interval width for the top-row band
        thresh_metric (str): the metric whose level line is drawn (Dice)
        level (float): the threshold level line (0.5 = half-maximal Dice)
    """
    df = df.copy()
    for c in [x, *metrics]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=[x])

    have = set(df['source'].dropna().unique())
    sources = [s for s in _SOURCE_ORDER if s in have]
    sources += [s for s in sorted(have) if s not in sources]
    if not sources:
        print(f'  (no rows for {label} — skipping)')
        return

    palette = get_cmap_dict(sorted(df['label'].dropna().unique().tolist()))
    log_x = pd.notnull(df[x].min()) and df[x].min() > 0
    ncols = len(metrics)

    fig = plt.figure(figsize=(4.2 * ncols, 4.6 * len(sources)),
                     layout='constrained')
    fig.suptitle(label, fontsize=13)
    subfigs = np.atleast_1d(fig.subfigures(len(sources), 1))

    diff_rows = []
    for si, (subfig, src) in enumerate(zip(subfigs, sources)):
        subfig.suptitle(src, fontsize=14, fontweight='bold')
        axes = subfig.subplots(2, ncols, sharex=True, squeeze=False)
        dsrc = df[df['source'] == src]
        for j, metric in enumerate(metrics):
            _draw_metric_band(axes[0, j], dsrc, x, metric, palette, ci=ci)
            axes[0, j].set_title(_METRIC_TITLES.get(metric, metric))
            axes[0, j].set_ylim(0, 1)
            axes[0, j].grid(True, alpha=0.3)
            # the discovery-threshold level, read as a table below
            if metric == thresh_metric:
                axes[0, j].axhline(level, ls='--', lw=0.8, color='grey',
                                   alpha=0.7)

            rows = _draw_diff(axes[1, j], dsrc, x, metric, one_label=one_label)
            for r in rows:
                r['source'] = src
            diff_rows += rows
            axes[1, j].set_xlabel(_X_PARAM_LABELS.get(x, x))
            if log_x:
                axes[0, j].set_xscale('log')
                axes[1, j].set_xscale('log')

        axes[0, 0].set_ylabel(f'score (mean, {ci}% band)')
        axes[1, 0].set_ylabel(f'{one_label} − best')
        # one legend for the figure, on the first block's top-left panel
        if si == 0:
            axes[0, 0].legend(frameon=False, fontsize=8)

    path = out / f'{label}.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'saved: {path}')

    if diff_rows:
        _write_diff_csv(label, pd.DataFrame(diff_rows), x=x, facet='source',
                        metrics=metrics, out=out)


def _write_diff_csv(label: str, diff_long, *, x: str, facet: str,
                    metrics: list, out) -> None:
    """Write the diff-grid mean line to CSV and print where Dice peaks.

    diff_long is the tidy mean line behind the diff rows (_draw_diff), one row
    per (facet, method, x, metric): the seed-averaged absolute scores (glow =
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
# Discovery threshold: the effect strength at which mean Dice crosses a level
# ---------------------------------------------------------------------------

def _crossing(x_vals, y_vals, level: float):
    """Find the first upward crossing of level, interpolated in log10(x).

    The x grid is log-spaced and the mean-metric curve is a monotone-ish
    sigmoid, so the crossing is located by linear interpolation between the two
    bracketing grid points in log10(x) (equivalently, geometric interpolation
    in x). The first x with y >= level fixes the upper bracket; the crossing
    lies between it and its predecessor.

    Args:
        x_vals (np.array): swept-axis values, sorted ascending, all > 0
        y_vals (np.array): the mean metric at each x (same length)
        level (float): the crossing level (0.5 for half-maximal Dice)

    Returns:
        (float, str): (threshold x, status). status is 'ok' with a finite
            threshold; 'below' (nan) when the curve already sits at/above level
            at the weakest x; 'above' (nan) when it never reaches level.
    """
    if len(x_vals) == 0:
        return np.nan, 'above'
    if y_vals[0] >= level:
        return np.nan, 'below'
    for i in range(1, len(x_vals)):
        if y_vals[i] >= level:
            t = (level - y_vals[i - 1]) / (y_vals[i] - y_vals[i - 1])
            lx = (np.log10(x_vals[i - 1])
                  + t * (np.log10(x_vals[i]) - np.log10(x_vals[i - 1])))
            return float(10 ** lx), 'ok'
    return np.nan, 'above'


def threshold_ratio_table(df, *, x: str, metric: str = 'dice',
                          level: float = 0.5,
                          ref_label: str = 'GLOW-Focus'):
    """Wide table of each method's discovery threshold, ref-normalised, per b.

    A method's discovery threshold is the swept-axis value at which its mean
    metric (averaged across trials) first crosses level -- the effect strength
    at which it starts recovering the support. Using the half-maximal (level =
    0.5) point of a monotone performance curve as a threshold is the standard
    dose-response / psychometric convention (the EC50 / 50%-detection point,
    the steepest, most reproducible part of the sigmoid); Dice itself is Dice
    1945, with Dice > 0.7 the usual "good overlap" line (Zijdenbos 1994), so
    level is a parameter.

    Each entry is the raw threshold ratio thr_method / thr_ref (for x =
    effect_llr, the LLR at which the method reaches level Dice divided by the
    reference method's): ref_label is 1.0, and > 1 means the method needs a
    stronger effect than the reference. A structural axis that varies alongside
    x (b in the llr sweep) becomes the columns, so each b gets its own ratio
    column; with none varying the single ratio column is named by x.

    Args:
        df: a tidy_run_ana frame (needs source / label / x / metric, and any
            varying secondary axis such as b)
        x (str): the swept-axis column (effect strength when x is effect_llr)
        metric (str): the metric whose level crossing defines the threshold
        level (float): the crossing level (0.5 = half-maximal)
        ref_label (str): the reference method (its ratio is 1.0)

    Returns:
        a wide DataFrame with columns source, method, then one ratio column per
        varying secondary value (e.g. b=1 / b=2 / b=3), or a single column named
        by x when no secondary varies. Empty in, empty out.
    """
    df = df.copy()
    df[x] = pd.to_numeric(df[x], errors='coerce')
    # keyed by method label, so rows whose ana repr did not resolve to a
    # catalogue label (stale records from a since-changed knob such as
    # n_perm_inner) carry no method and are dropped; an all-unlabelled cache
    # then yields an empty table rather than a groupby that silently drops
    # every NaN-label row and leaves wide without a method column.
    df = df.dropna(subset=[x, 'label'])
    # every ratio normalises to ref_label, so a frame missing it (only the
    # non-reference methods resolved) has no reference to divide by and yields
    # an all-blank table; skip it as empty rather than emit one.
    if df.empty or ref_label not in set(df['label']):
        return df.iloc[0:0]

    secondary = [a for a in _SECONDARY_AXES
                 if a != x and a in df.columns and df[a].dropna().nunique() > 1]

    rows = []
    for keys, sub in df.groupby(['source', *secondary]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        cell = dict(zip(['source', *secondary], keys))
        mean_curve = sub.groupby(['label', x])[metric].mean()
        thr = {}
        for lab in mean_curve.index.get_level_values(0).unique().tolist():
            s = mean_curve.loc[lab].dropna().sort_index()
            thr[lab] = _crossing(np.asarray(s.index, dtype=float),
                                 np.asarray(s.values, dtype=float), level)[0]
        ref = thr.get(ref_label, np.nan)
        for lab, t in thr.items():
            ratio = (t / ref if np.isfinite(t) and np.isfinite(ref) and ref > 0
                     else np.nan)
            rows.append({**cell, 'method': lab, 'ratio': ratio})

    long = pd.DataFrame(rows)
    if secondary:
        long['_col'] = long[secondary].apply(
            lambda r: ' '.join(f'{s}={int(r[s])}' for s in secondary), axis=1)
        wide = long.pivot_table(index=['source', 'method'], columns='_col',
                                values='ratio').reset_index()
        wide.columns.name = None
    else:
        wide = long.rename(columns={'ratio': x})

    # reference method first per source, then others weakest-method first
    ratio_cols = [c for c in wide.columns if c not in ('source', 'method')]
    order = wide[ratio_cols].mean(axis=1)
    wide = (wide.assign(_ref=wide['method'].ne(ref_label), _order=order)
                .sort_values(['source', '_ref', '_order'],
                             ascending=[True, True, False])
                .drop(columns=['_ref', '_order']).reset_index(drop=True))
    return wide


def write_threshold_table(label: str, df, *, x: str, out, metric: str = 'dice',
                          level: float = 0.5,
                          ref_label: str = 'GLOW-Focus') -> None:
    """Write and print the ref-normalised discovery-threshold table.

    Args:
        label (str): cache name; used in the output filename
        df: the cache's tidy_run_ana results
        x (str): the swept-axis column
        out (pathlib.Path): directory the CSV is written into
        metric (str): the metric whose level crossing defines the threshold
        level (float): the crossing level (0.5 = half-maximal Dice)
        ref_label (str): the reference method (its ratio is 1.0)
    """
    wide = threshold_ratio_table(df, x=x, metric=metric, level=level,
                                 ref_label=ref_label)
    if wide.empty:
        return
    path = out / f'{label}_threshold.csv'
    wide.to_csv(path, index=False, float_format='%.3f')
    print(f'saved: {path}')

    ratio_cols = [c for c in wide.columns if c not in ('source', 'method')]
    print(f'  {x} at Dice >= {level:g} (mean across trials), relative to '
          f'{ref_label} (= 1.00; > 1 needs a stronger effect):')
    for src, sub in wide.groupby('source'):
        print(f'    source={src}')
        print(f'      {"method":<11} '
              + '  '.join(f'{c:>7}' for c in ratio_cols))
        for _, r in sub.iterrows():
            cells = '  '.join(
                (f'{r[c]:>7.2f}' if pd.notnull(r[c]) else f'{"—":>7}')
                for c in ratio_cols)
            tag = ' (ref)' if r['method'] == ref_label else ''
            print(f'      {r["method"]:<11} {cells}{tag}')


# ---------------------------------------------------------------------------
# Runtime sweeps (wall time vs one cost knob)
# ---------------------------------------------------------------------------

def tidy_runtime(name: str, raw):
    """Normalise a runtime cache's provenance frame to (label, x, time_sec).

    The runtime counterpart to tidy_run_ana: collapses the wide provenance
    frame to one tidy row per timed leaf, reading the method label, the swept
    x-axis value, and the wall time. The leaf prefix and swept axis come from
    _RUNTIME_SPEC (time is the signal, so unlike the detection path the x is
    not inferred from what varies). A run_ana_time cache (runtime / runtime_b)
    reads the method off the recipe (in.ana), as tidy_run_ana does; the
    dedicated leaves (run_segment_time / run_perm_*) record it as an explicit
    label. Every timing leaf returns num_vox bare, so only the run_ana
    detection path reads it from the recursed score. All runtime caches are
    HCP-only, so the seed is the HCP data seed.

    Args:
        name (str): the runtime cache name (a key of _RUNTIME_SPEC).
        raw: the provenance DataFrame (one row per leaf) for this cache.

    Returns:
        a tidy DataFrame, one row per (trial, method), with columns label, x
        (the swept-axis value), x_name (its column name), time_sec, num_vox,
        and seed; empty in, empty out.
    """
    if raw.empty:
        return raw

    leaf, x_name = _RUNTIME_SPEC[name]

    def col(c):
        """Return raw[c], or an all-NaN column when absent."""
        if c in raw.columns:
            return raw[c]
        return pd.Series(np.nan, index=raw.index)

    out = pd.DataFrame(index=raw.index)
    out['time_sec'] = pd.to_numeric(col(f'{leaf}.time_sec'), errors='coerce')
    out['seed'] = pd.to_numeric(col('data_factory_hcp.in.seed'),
                                errors='coerce')

    # method label: run_ana / run_ana_time carry the recipe (in.ana); the
    # dedicated timing leaves record an explicit label
    if leaf in ('run_ana', 'run_ana_time'):
        out['label'] = col(f'{leaf}.in.ana').map(_LABEL_OF_ANA)
    else:
        out['label'] = col(f'{leaf}.in.label')

    # num_vox: run_ana carries it inside the recursed score dict; every timing
    # leaf (run_ana_time included) returns it bare
    if leaf == 'run_ana':
        out['num_vox'] = pd.to_numeric(col('run_ana.out.score.num_vox'),
                                       errors='coerce')
    else:
        out['num_vox'] = pd.to_numeric(col(f'{leaf}.out.num_vox'),
                                       errors='coerce')

    # b is the HCP feature-subset length; every other knob is num_vox itself or
    # an explicit leaf input
    if x_name == 'num_vox':
        out['x'] = out['num_vox']
    elif x_name == 'b':
        out['x'] = col('data_factory_hcp.in.hcp_feats').map(
            lambda v: len(v) if isinstance(v, (list, tuple)) else np.nan)
    else:
        out['x'] = pd.to_numeric(col(f'{leaf}.in.{x_name}'), errors='coerce')
    out['x_name'] = x_name
    return out


def plot_runtime(name: str, df, out, log_x_ratio: float = 10.0) -> None:
    """Plot wall time vs the swept knob, one curve per method, on log axes.

    The runtime family's single plotter: the seed replicates collapse to a
    median line per method with a min-max band, wall time on a log y-axis and
    the swept knob on a log x-axis when it spans at least log_x_ratio (so the
    num_vox / permutation scaling reads as a slope; the small b sweep stays
    linear). Methods use the shared palette (COLOR_ANALYSIS); the Ward-mode
    labels of runtime_segment take a seaborn fallback (get_cmap_dict).

    Args:
        name (str): cache name; used in the title and output filename.
        df: the cache's tidy_runtime results.
        out (pathlib.Path): directory the figure is written into.
        log_x_ratio (float): x max/min ratio at or above which the x-axis is
            log-scaled.
    """
    df = df.dropna(subset=['x', 'time_sec', 'label'])
    if df.empty:
        print(f'  (no timed rows for {name} — skipping)')
        return

    x_name = df['x_name'].iloc[0]
    labels = sorted(df['label'].unique().tolist())
    palette = get_cmap_dict(labels)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label in labels:
        g = df[df['label'] == label].groupby('x')['time_sec']
        med, lo, hi = g.median(), g.min(), g.max()
        ax.plot(med.index, med.values, marker='o', ms=5, lw=2,
                color=palette[label], label=label)
        ax.fill_between(med.index, lo.values, hi.values,
                        color=palette[label], alpha=0.15)

    if df['x'].max() / max(df['x'].min(), 1) >= log_x_ratio:
        ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(_X_PARAM_LABELS.get(x_name, x_name))
    ax.set_ylabel('wall time (s)')
    ax.set_title(name)
    ax.legend(frameon=False)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    path = out / f'{name}_runtime.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'saved: {path}')


# ---------------------------------------------------------------------------
# Inner-perm edge (num_inner_perm convergence)
# ---------------------------------------------------------------------------

def _fwer_threshold(max_z_col, alpha_fwer: float) -> float:
    """Return the FWER critical value from one column of per-perm max-z.

    The max-z a region must exceed for its Westfall-Young p-value to fall at or
    below alpha_fwer, in Analysis.get_pval's convention: with n = len(max_z_col)
    permuted-or-observed maxima, reject iff at least ceil((1 - alpha) * n) of
    them are below the region's z, so the critical value is the
    ceil((1 - alpha) * n) th smallest max-z.

    Args:
        max_z_col (np.array): (n_perm_fwer + 1,) max-z, one per outer perm.
        alpha_fwer (float): the FWER level.

    Returns:
        the critical max-z (the significance bar) at alpha_fwer.
    """
    col = np.sort(np.asarray(max_z_col, dtype=float))
    n = col.shape[0]
    k = min(max(int(np.ceil((1 - alpha_fwer) * n)), 1), n)
    return float(col[k - 1])


def tidy_inner_edge(raw, alpha_fwer: float = 0.05):
    """Normalise the inner-edge cache to one tidy row per (trial, arm, m).

    Parses each run_inner_edge leaf's recorded curve JSON (the num_inner_perm
    grid and the per-outer-perm max_z_null) into long form, and per
    num_inner_perm derives the FWER critical value (_fwer_threshold) and the
    observed (k=0) max-z. The GLOW arm is recovered from the recorded
    cluster_mode (_ARM_OF_MODE); no label was stored. All cells are HCP or WGN,
    so the source is read off which data_factory produced the row.

    Args:
        raw: the provenance DataFrame (one row per run_inner_edge leaf).
        alpha_fwer (float): FWER level the critical value is reported at.

    Returns:
        a tidy DataFrame with columns source (WGN / HCP), seed, label (GLOW
        arm), num_inner_perm, threshold (FWER critical max-z), obs_max_z;
        empty in, empty out.
    """
    if raw.empty:
        return raw

    def col(c):
        """Return raw[c], or an all-NaN column when absent."""
        if c in raw.columns:
            return raw[c]
        return pd.Series(np.nan, index=raw.index)

    wgn_seed = pd.to_numeric(col('data_factory_wgn.in.seed'), errors='coerce')
    hcp_seed = pd.to_numeric(col('data_factory_hcp.in.seed'), errors='coerce')
    arm = col('run_inner_edge.in.cluster_mode').map(_ARM_OF_MODE)
    curve = col('run_inner_edge.out.curve')

    rows = []
    for idx in raw.index:
        cell = curve.get(idx)
        if not isinstance(cell, str):
            continue
        d = json.loads(cell)
        max_z = np.asarray(d['max_z_null'], dtype=float)
        src = 'HCP' if pd.notna(hcp_seed.get(idx)) else 'WGN'
        seed = hcp_seed.get(idx) if src == 'HCP' else wgn_seed.get(idx)
        for j, m in enumerate(d['num_inner_perm']):
            rows.append({'source': src, 'seed': seed, 'label': arm.get(idx),
                         'num_inner_perm': int(m),
                         'threshold': _fwer_threshold(max_z[:, j], alpha_fwer),
                         'obs_max_z': float(max_z[0, j])})
    return pd.DataFrame(rows)


def plot_inner_edge(name: str, df, out, alpha_fwer: float = 0.05) -> None:
    """Plot the FWER threshold vs num_inner_perm and write the threshold JSON.

    The convergence read for n_perm_inner: how the FWER max-z critical value
    (the significance bar) settles as num_inner_perm grows, one curve per GLOW
    arm and one axes per data source (HCP for sweep_n_perm_inner), seeds
    aggregated to a median line with a min-max band on a log x-axis. Also writes
    {name}_threshold.json -- the
    seed-median threshold per (source, arm, num_inner_perm) -- the artifact the
    recommended n_perm_inner is read off (the smallest num_inner_perm whose
    threshold has plateaued).

    Args:
        name (str): cache name; used in the title and output filenames.
        df: the tidy_inner_edge frame.
        out (pathlib.Path): directory the figure and JSON are written into.
        alpha_fwer (float): FWER level (annotated on the y-axis).
    """
    df = df.dropna(subset=['num_inner_perm', 'threshold', 'label'])
    if df.empty:
        print(f'  (no rows for {name} — skipping)')
        return

    sources = sorted(df['source'].dropna().unique().tolist())
    labels = sorted(df['label'].dropna().unique().tolist())
    palette = get_cmap_dict(labels)

    fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 4.5),
                             squeeze=False, sharey=True)
    summary = {}
    for ax, src in zip(axes[0], sources):
        sub = df[df['source'] == src]
        summary[src] = {}
        for lab in labels:
            g = sub[sub['label'] == lab].groupby('num_inner_perm')['threshold']
            if not len(g):
                continue
            med, lo, hi = g.median(), g.min(), g.max()
            ax.plot(med.index, med.values, marker='o', ms=5, lw=2,
                    color=palette[lab], label=lab)
            ax.fill_between(med.index, lo.values, hi.values,
                            color=palette[lab], alpha=0.15)
            summary[src][lab] = {int(m): float(v) for m, v in med.items()}
        ax.set_xscale('log')
        ax.set_xlabel(_X_PARAM_LABELS['n_perm_inner'])
        ax.set_title(f'{name} — {src}')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(frameon=False)
    axes[0][0].set_ylabel(f'FWER max-z threshold ($\\alpha$={alpha_fwer:g})')
    fig.tight_layout()
    path = out / f'{name}_threshold.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'saved: {path}')

    json_path = out / f'{name}_threshold.json'
    with open(json_path, 'w') as f:
        json.dump({'alpha_fwer': alpha_fwer, 'threshold': summary}, f, indent=2)
    print(f'saved: {json_path}')


# ---------------------------------------------------------------------------
# Max-z race retention (survivor race vs full cpu_perm)
# ---------------------------------------------------------------------------

# the max-z difference a race-retention pair may show and still count as a
# match: pure float round-off between the survivor race and the full cpu_perm
# (both reduce identical draws, see run.run_race_maxz).
_RACE_MAXZ_TOL = 1e-6


def tidy_race_maxz(raw):
    """Normalise the race-retention cache to one tidy row per (trial, arm, k).

    Parses each run_race_maxz leaf's recorded curve JSON (the per-outer-perm
    max_z_slow / max_z_race and their arg-max regions) into long form. The GLOW
    arm is recovered from the recorded cluster_mode (_ARM_OF_MODE); the source
    is read off which data_factory produced the row.

    Args:
        raw: the provenance DataFrame (one row per run_race_maxz leaf).

    Returns:
        a tidy DataFrame with columns source (WGN / HCP), seed, label (GLOW
        arm), k (outer-perm index), max_z_slow, max_z_race, diff
        (race - slow), same_reg (arg-max region agrees); empty in, empty out.
    """
    if raw.empty:
        return raw

    def col(c):
        """Return raw[c], or an all-NaN column when absent."""
        if c in raw.columns:
            return raw[c]
        return pd.Series(np.nan, index=raw.index)

    wgn_seed = pd.to_numeric(col('data_factory_wgn.in.seed'), errors='coerce')
    hcp_seed = pd.to_numeric(col('data_factory_hcp.in.seed'), errors='coerce')
    arm = col('run_race_maxz.in.cluster_mode').map(_ARM_OF_MODE)
    curve = col('run_race_maxz.out.curve')

    rows = []
    for idx in raw.index:
        cell = curve.get(idx)
        if not isinstance(cell, str):
            continue
        d = json.loads(cell)
        slow = np.asarray(d['max_z_slow'], dtype=float)
        race = np.asarray(d['max_z_race'], dtype=float)
        reg_s = d.get('reg_slow', [-1] * len(slow))
        reg_r = d.get('reg_race', [-1] * len(race))
        src = 'HCP' if pd.notna(hcp_seed.get(idx)) else 'WGN'
        seed = hcp_seed.get(idx) if src == 'HCP' else wgn_seed.get(idx)
        for k in range(len(slow)):
            rows.append({'source': src, 'seed': seed, 'label': arm.get(idx),
                         'k': int(k), 'max_z_slow': float(slow[k]),
                         'max_z_race': float(race[k]),
                         'diff': float(race[k] - slow[k]),
                         'same_reg': bool(reg_s[k] == reg_r[k])})
    return pd.DataFrame(rows)


def plot_race_maxz(name: str, df, out) -> None:
    """Plot race vs full-run max-z and write the retention summary JSON.

    The retention read for the survivor race: each outer perm's race max-z
    against the full cpu_perm max-z, one axes per data source, arms coloured,
    with the y=x line the points must lie on. Also writes {name}_retention.json
    -- per (source, arm) the max abs max-z difference, the count of outer perms
    whose max-z differs beyond float round-off (_RACE_MAXZ_TOL), and the arg-max
    region agreement rate -- the artifact that certifies the race keeps the
    max-z (max_abs_diff ~ 0, n_mismatch 0).

    Args:
        name (str): cache name; used in the title and output filenames.
        df: the tidy_race_maxz frame.
        out (pathlib.Path): directory the figure and JSON are written into.
    """
    df = df.dropna(subset=['max_z_slow', 'max_z_race', 'label'])
    df = df[np.isfinite(df['max_z_slow']) & np.isfinite(df['max_z_race'])]
    if df.empty:
        print(f'  (no rows for {name} — skipping)')
        return

    sources = sorted(df['source'].dropna().unique().tolist())
    labels = sorted(df['label'].dropna().unique().tolist())
    palette = get_cmap_dict(labels)

    fig, axes = plt.subplots(1, len(sources), figsize=(5.5 * len(sources), 5),
                             squeeze=False)
    summary = {}
    for ax, src in zip(axes[0], sources):
        sub = df[df['source'] == src]
        summary[src] = {}
        for lab in labels:
            g = sub[sub['label'] == lab]
            if not len(g):
                continue
            ax.scatter(g['max_z_slow'], g['max_z_race'], s=18, alpha=0.6,
                       color=palette[lab], label=lab)
            adiff = g['diff'].abs()
            summary[src][lab] = {
                'max_abs_diff': float(adiff.max()),
                'n_mismatch': int((adiff > _RACE_MAXZ_TOL).sum()),
                'n_perm': int(len(g)),
                'same_reg_rate': float(g['same_reg'].mean())}
        lo = float(min(sub['max_z_slow'].min(), sub['max_z_race'].min()))
        hi = float(max(sub['max_z_slow'].max(), sub['max_z_race'].max()))
        ax.plot([lo, hi], [lo, hi], color='0.5', lw=1, ls='--', zorder=0)
        ax.set_xlabel('full cpu_perm max-z')
        ax.set_title(f'{name} — {src}')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    axes[0][0].set_ylabel('survivor race max-z')
    fig.tight_layout()
    path = out / f'{name}_retention.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'saved: {path}')

    json_path = out / f'{name}_retention.json'
    with open(json_path, 'w') as f:
        json.dump({'tol': _RACE_MAXZ_TOL, 'retention': summary}, f, indent=2)
    print(f'saved: {json_path}')


# ---------------------------------------------------------------------------
# Per-cache dispatch + CLI
# ---------------------------------------------------------------------------

def plot_cache(label: str, df, out,
               metrics: list = ['dice', 'sens', 'ppv']) -> None:
    """Write one run_ana cache's figure, dispatching on its swept axis.

    The null path (no effect planted) gets a faceted FWER calibration curve;
    every other cache gets the stacked per-source detection figure
    (plot_source_grid: an HCP block over a WGN block, each a mean-band row and
    a GLOW-Focus diff row across the metric columns) plus one cache-level
    discovery-threshold table (write_threshold_table). The x-axis is inferred
    from the data (_infer_x), so no config plot spec is needed. A cache that
    also varies a structural axis besides x (the llr sweep varies b) is drawn
    one figure per value of it (_split_by_secondary), each suffixed into the
    label, while the threshold table spreads that axis across its columns.

    Args:
        label (str): cache name; used in titles and output filenames
        df: the cache's tidy_run_ana results
        out (pathlib.Path): directory the figures are written into
        metrics (list): metric columns plotted as the panel columns
    """
    if df.empty:
        print(f'  (no rows for {label} — skipping)')
        return

    x = _infer_x(df)
    if x is None:
        _plot_calibration_faceted(label, df, out)
        return

    for sub_label, sub in _split_by_secondary(label, df, x):
        plot_source_grid(sub_label, sub, x=x, metrics=metrics, out=out)

    # one discovery-threshold table for the whole cache, a column per secondary
    # (b in the llr sweep); normalised to GLOW-Focus (see threshold_ratio_table)
    write_threshold_table(label, df, x=x, out=out)


def main(argv=None) -> None:
    """Plot the detection and runtime caches from the shared records.

    For each selected cache (every detection and runtime cache in CONFIG by
    default, or the names given on the command line), reads its provenance
    frame (results.config_results_df) and plots it: a runtime cache is
    normalised with tidy_runtime and drawn by plot_runtime (wall time vs its
    cost knob); every other run_ana cache is normalised with tidy_run_ana and
    drawn by plot_cache (detection sweep / calibration). Figures land in
    results/_latest, so a mid-benchmark run yields intermediate figures. The
    remaining caches (segment / stat / prune / min_size) carry other leaf and
    score shapes and are skipped.

    Args:
        argv (list | None): CLI args to parse; None reads sys.argv. Positional
            args are cache names (e.g. sweep_llr, runtime); with none, every
            detection and runtime cache in the catalogue is plotted.
    """
    import argparse
    import fnmatch
    import matplotlib
    matplotlib.use('Agg')
    from .config import CONFIG
    from .run import run_ana, run_inner_edge, run_race_maxz
    from . import results

    parser = argparse.ArgumentParser(
        description='Plot detection and runtime benchmark figures from the '
                    'records.')
    parser.add_argument(
        'names', nargs='*',
        help='cache names or fnmatch patterns (e.g. sweep_llr, runtime*); '
             'default: every detection and runtime cache in the catalogue')
    args = parser.parse_args(argv)

    # a runtime cache is one in _RUNTIME_SPEC; the rest split on the leaf
    # function (CONFIG values are (data, effect, fnc_kwargs, fnc)) into the
    # run_ana detection caches this layer draws and the others it skips.
    runtime_names = [n for n in CONFIG if n in _RUNTIME_SPEC]
    detect_names = [n for n, cfg in CONFIG.items()
                    if cfg[3] is run_ana and n not in _RUNTIME_SPEC]
    edge_names = [n for n, cfg in CONFIG.items() if cfg[3] is run_inner_edge]
    race_names = [n for n, cfg in CONFIG.items() if cfg[3] is run_race_maxz]
    if args.names:
        # literal name, else fnmatch pattern; a pattern matching nothing is an
        # error (a typo surfaces rather than silently plotting nothing)
        names = []
        for pattern in args.names:
            matches = ([pattern] if pattern in CONFIG
                       else fnmatch.filter(CONFIG, pattern))
            if not matches:
                parser.error(f'no cache names match: {pattern}')
            names += [n for n in matches if n not in names]
    else:
        names = detect_names + runtime_names + edge_names + race_names

    out = glow._extra.benchmark.get_path_result() / '_latest'
    out.mkdir(exist_ok=True)

    n_plotted = 0
    for name in names:
        if name in _RUNTIME_SPEC:
            df = tidy_runtime(name, results.config_results_df(name))
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {len(df)} runtime rows ===')
            plot_runtime(name, df, out)
            n_plotted += 1
        elif name in detect_names:
            df = tidy_run_ana(results.config_results_df(name))
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {len(df)} run_ana rows ===')
            plot_cache(name, df, out)
            n_plotted += 1
        elif name in edge_names:
            df = tidy_inner_edge(results.config_results_df(name))
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {len(df)} inner-edge rows ===')
            plot_inner_edge(name, df, out)
            n_plotted += 1
        elif name in race_names:
            df = tidy_race_maxz(results.config_results_df(name))
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {len(df)} race-retention rows ===')
            plot_race_maxz(name, df, out)
            n_plotted += 1
        else:
            print(f'  ({name} is not a detection or runtime cache — skipping)')

    if n_plotted == 0:
        print(f'no results found under {out.parent}')


if __name__ == '__main__':
    main()
