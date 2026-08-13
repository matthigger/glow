"""Plot the run_ana benchmark caches from the shared provenance records.

The plotting layer for the run_ana caches. It reads each cache's
provenance frame (make_csv.write_config_csv: one wide row per run_ana leaf,
namespaced by the producing function -- run_ana.out.score, the swept
data_factory / effect_factory inputs), normalises it to one tidy row per
(trial, recipe) with tidy_run_ana, and writes one figure set per cache into
results/_latest. Reading through make_csv refreshes each plotted cache's CSV
in passing, so the exported table and the figures come from the same records.

The tidy frame is what the plotters consume: a label (method), a source
(WGN / HCP, which share each cache and face apart here), the swept axes
(effect_llr, b, num_img, the realized effect fraction), and the metrics
(dice / sens / ppv / spec) derived from the score's confusion counts (see
score_effects / glow.mask.stats_from_counts). The x-axis is inferred from
what actually varies in the cache (no config spec): an all-null effect grid
is the FWER calibration path, else the first of effect_llr / b / num_img /
effect_perc that varies is swept.

Every figure outside the segment / prune / race-retention families reports one
GLOW arm, the GLM-Error clustering, labelled plainly GLOW: the Focus arm is
dropped and the survivor relabelled at the plot layer (_select_glow_arm), not
in the caches or records. The three exceptions compare the arms, so they keep
both.

Each cache then gets either a GLOW-only FWER calibration row (null) -- a single
row of source x GLOW arm cells (HCP / WGN x GLOW),
each the nominal-alpha vs empirical-rejection-rate curve with a
Clopper-Pearson 95% band on 0..1 -- or one stacked detection figure: an HCP
block over a WGN block,
each a 2 x 3 grid
whose top row is the per-method mean score + central 95% percentile band and
whose bottom row is the GLOW head-to-head diff, over the dice / sens /
ppv columns. Alongside it a discovery-threshold table (write_threshold_table)
records, per method, the absolute effect strength at which its mean Dice first
reaches 0.5 -- rows the methods, a column per b.

The runtime family is plotted apart (tidy_runtime / plot_runtime): those caches
hold detection fixed and sweep one cost knob, so the signal is the leaf wall
time (RECORDER time_sec), not a score. Each is one time-vs-knob curve per
method on log axes -- runtime_num_vox for the wall clock on this machine, and
the runtime_1perm_* caches for the growth rate in each of the five cost knobs
(see _RUNTIME_SPEC, and config's runtime section for why the two are separate).

The inner-edge cache (sweep_n_perm_inner) is plotted apart too (tidy_inner_edge
/ plot_inner_edge): its run_inner_edge leaf records the per-outer-perm max-z at
each num_inner_perm, from which the FWER critical value is derived, giving one
threshold-vs-num_inner_perm convergence curve per GLOW arm (and the companion
threshold JSON the recommended n_perm_inner is read off).


The segment and prune caches share a flatter path (tidy_segment / tidy_prune /
plot_metric_grid): their leaves return a flat {tp, fp, tn, fn} score (the oracle
best-Dice Ward region per mode; one pruning rule's selection) rather than
run_ana's nested score.target block. Each is drawn as a source x metric grid --
HCP over WGN, the Dice / sensitivity / PPV columns -- of the per-method
seed-mean with a 95% CI error bar vs effect_llr, x-dodged and styled per method
(the Ward mode / prune rule) in the Okabe-Ito palette. Prune crosses its rules
with both Ward modes, so plot_prune draws one such grid per clustering mode
(prune_Focus / prune_GLM_Error), a line per rule within each.

With no arguments the CLI plots every cache in the catalogue: the detection
sweeps, the runtime family, the inner-edge and race-retention checks, the stat
bake-off tables, and the segment / prune metric grids; passing names restricts
it.
"""
import colorsys
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import beta

import glow._extra.benchmark
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import stat_dict
from .config import ana_kwargs_dict, GLOW_ARM_MODES, RUN_STAT_LIST
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
    # teal (180 deg) -- the Focus arm, which no figure reports (_ARMS_SKIP)
    'GLOW-Focus': _hls_hex(0/4 + _H),
    # darker teal, the reported arm under both its recipe and figure label
    'GLOW-GLM':   _hls_hex(0/4 + _H, l=_L * 0.6),
    'GLOW':       _hls_hex(0/4 + _H, l=_L * 0.6),
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
_ARM_OF_MODE = {str(mode): label for label, mode in GLOW_ARM_MODES}

# One GLOW arm is reported, the GLM-Error clustering, and the figures call it
# plainly GLOW: the Focus arm is dropped (_ARMS_SKIP) and the survivor
# relabelled (_ARM_LABEL), so the arm choice does not have to be re-argued on
# each panel. Both arms stay in the caches and the records -- Focus is a real
# recipe, and the segment / prune families exist to compare the two
# clusterings. Those families label by Ward mode (Focus / GLM Error) and prune
# rule (GLOW-greedy / GLOW-dp) rather than by analysis arm, so neither the drop
# nor the relabel reaches them.
_ARMS_SKIP = ('GLOW-Focus',)
_ARM_LABEL = {'GLOW-GLM': 'GLOW'}


def _select_glow_arm(df):
    """Drop the unreported GLOW arm and label the reported one GLOW.

    Args:
        df: a tidy frame from any family; needs a label column to filter on

    Returns:
        the frame without its _ARMS_SKIP rows, its GLOW-GLM rows relabelled
        GLOW (_ARM_LABEL); unchanged when empty or unlabelled (the segment /
        prune frames label by mode / rule).
    """
    if df.empty or 'label' not in df.columns:
        return df
    df = df[~df['label'].isin(_ARMS_SKIP)]
    return df.assign(label=df['label'].replace(_ARM_LABEL))

# stat bake-off vocabulary. The method (VBA / VBA-TFCE / CET) and the raw/z arm
# are recovered from the recorded recipe -- its class and tfce_flag / z_flag --
# the same way _LABEL_OF_ANA recovers a run_ana method; the stat is the recorded
# stat_name. stat_dict order (llr..roys_root) fixes the column order (= the
# grid.get_run_stat_list build order, so an interrupted cell drops the last stats).
_STAT_METHOD_ORDER = ['VBA', 'VBA-TFCE', 'CET']
_ZT_ORDER = ['raw', 'z']
_ZT_PRETTY = {'raw': 'raw', 'z': 'z-scored'}
_STAT_ORDER = list(stat_dict)
_STAT_PRETTY = {'llr': 'LLR', 'wilks': 'Wilks', 'pillai': 'Pillai',
                'hotel_tr': 'Hotelling', 'roys_root': 'Roy'}


def _stat_method_zt(ana):
    """Recover (method, zt) from a stat-cache recipe (see grid.get_run_stat_list)."""
    if type(ana).__name__ == 'AnalysisCET':
        method = 'CET'
    else:
        method = 'VBA-TFCE' if getattr(ana, 'tfce_flag', False) else 'VBA'
    return method, ('z' if getattr(ana, 'z_flag', False) else 'raw')


# (recipe repr, stat_name) -> (method, zt, stat), the address-free variant id
_STAT_VARIANT = {(repr(s['ana']), s['stat_name']):
                 (*_stat_method_zt(s['ana']), s['stat_name'])
                 for s in RUN_STAT_LIST}


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


# Okabe-Ito colourblind-safe qualitative palette, for the flat-score caches
# whose methods sit outside COLOR_ANALYSIS (segment Ward modes, prune rules).
# Paired with line styles so overlapping curves stay distinct where the colours
# muddy (Okabe & Ito 2008; Wong 2011). Ordered for high pairwise contrast at the
# small counts these caches use (3 methods).
_OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7',
              '#56B4E9', '#F0E442', '#000000']
_LINE_STYLES = ['-', '--', ':', '-.']


def _qual_style(label_list) -> dict:
    """Map labels to an Okabe-Ito (colour, line style), assigned in sorted order.

    A deterministic qualitative style for the flat-score caches: each label
    takes the next Okabe-Ito colour and line style, so a cache's methods are
    coloured the same across runs (sorted-order, not palette-position, stable).

    Args:
        label_list: the method labels to style.

    Returns:
        dict: label -> {'color': str, 'ls': str}.
    """
    out = {}
    for i, lab in enumerate(sorted(label_list)):
        out[lab] = {'color': _OKABE_ITO[i % len(_OKABE_ITO)],
                    'ls': _LINE_STYLES[i % len(_LINE_STYLES)]}
    return out


_METRIC_TITLES = {
    'dice': 'Dice',
    'sens': 'Sensitivity',
    'ppv': 'PPV (Precision)',
    'spec': 'Specificity',
}

# effect_llr is the size-normalized effect strength the factory plants -- the
# per-voxel LLR contribution, so a planted region's observed LLR is
# ~ effect_llr * |r| (glow.effect.impose.compute_offset). The axis is labelled
# in that quantity, LLR / |r|, not the bare region LLR.
_X_PARAM_LABELS = {
    'effect_llr': 'LLR / |r|',
    'effect_perc': 'Effect Size (% of Volume)',
    'num_img': 'Number of Subjects',
    'b': 'Number of Imaging Features',
    'num_vox': 'Number of Voxels',
    'n_perm_fwer': 'FWER Permutations',
    'n_perm_inner': 'Freedman-Lane Permutations',
}

# runtime caches: name -> (leaf column prefix, swept x-axis column). The
# runtime family plots time (leaf.time_sec) against one swept cost knob;
# unlike the detection sweeps the x is not inferred (time is the signal, the
# effect is held at the moderate default). Both leaves carry the method in the
# recipe (in.ana) and return num_vox bare. A knob is read wherever it was
# declared: num_vox off the leaf's own output, b off the data factory that
# shaped the array, num_img and the permutation counts off the 1perm leaf's
# explicit inputs. See config's runtime section.
_RUNTIME_SPEC = {
    'runtime_num_vox':            ('run_ana_time',       'num_vox'),
    'runtime_1perm_num_vox':      ('run_ana_time_1perm', 'num_vox'),
    'runtime_1perm_n_perm_fwer':  ('run_ana_time_1perm', 'n_perm_fwer'),
    'runtime_1perm_b':            ('run_ana_time_1perm', 'b'),
    'runtime_1perm_nimg':         ('run_ana_time_1perm', 'num_img'),
}


# ---------------------------------------------------------------------------
# Normalise the provenance frame to one tidy row per (trial, recipe)
# ---------------------------------------------------------------------------

def tidy_run_ana(raw):
    """Normalise a run_ana provenance frame to a tidy per-trial results frame.

    Collapses the wide, function-namespaced frame from
    make_csv.write_config_csv (run_ana leaf + its data_factory /
    effect_factory ancestors) into the flat schema the plotters consume. The
    source is read off which data_factory produced the row (wgn / hcp), the
    swept axes off the relevant ancestor inputs, and the metrics off the
    recursed score columns
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

def _binom_ci(n0, n: int, conf: float = 0.95):
    """Clopper-Pearson exact confidence interval for a binomial proportion.

    The exact interval (Clopper & Pearson 1934) for the success probability p
    given n0 successes in n trials, from the Beta-quantile identity: lo is the
    (1-conf)/2 quantile of Beta(n0, n-n0+1), hi the (1+conf)/2 quantile of
    Beta(n0+1, n-n0). The degenerate ends (n0 = 0, n0 = n) clamp to 0 and 1,
    where the Beta quantile is undefined.

    Args:
        n0 (np.array): (k,) successes, each 0..n
        n (int): trials
        conf (float): confidence level

    Returns:
        (lo, hi) (np.array, np.array): (k,) elementwise lower / upper bounds
    """
    n0 = np.asarray(n0, dtype=float)
    a = 1 - conf
    lo = beta.ppf(a / 2, n0, n - n0 + 1)
    hi = beta.ppf(1 - a / 2, n0 + 1, n - n0)
    lo = np.where(n0 <= 0, 0.0, lo)
    hi = np.where(n0 >= n, 1.0, hi)
    return lo, hi


def plot_calibration(pvals, color, *, alpha_max: float = 1.0, n_pts: int = 200,
                     conf: float = 0.95, ax=None) -> None:
    """Plot one method's FWER calibration curve with a binomial CI band.

    The empirical rejection rate -- the fraction of null trials whose smallest
    FWER-corrected region p-value falls at or below the nominal alpha --
    against nominal alpha, with the y=x ideal for reference. Rather than a
    single-alpha error bar, two lines trace the Clopper-Pearson 95% confidence
    interval for the true rejection probability p at each nominal alpha, given
    n0 of n null trials rejected there (_binom_ci); a well-calibrated method's
    curve tracks the diagonal and its band brackets it.

    Args:
        pvals (np.array): (n,) per-trial smallest FWER-corrected p-value for
            one (source, method) cell
        color: the method's curve / band colour
        alpha_max (float): right edge of the nominal-alpha axis
        n_pts (int): number of nominal-alpha sample points
        conf (float): confidence level for the CI band
        ax: matplotlib Axes to draw into; None makes its own square figure
    """
    owns_fig = ax is None
    if owns_fig:
        _, ax = plt.subplots(figsize=(4, 4))

    ax.plot([0, alpha_max], [0, alpha_max], ls='--', color='grey', lw=1,
            label='ideal', zorder=1)

    pvals = np.asarray(pvals, dtype=float)
    pvals = pvals[np.isfinite(pvals)]
    n = pvals.size
    if n:
        alphas = np.linspace(0, alpha_max, n_pts)
        n0 = (pvals[None, :] <= alphas[:, None]).sum(axis=1)
        rate = n0 / n
        lo, hi = _binom_ci(n0, n, conf=conf)
        ax.plot(alphas, rate, lw=2.5, color=color, zorder=3)
        # the CI as two lines (not an error bar): the interval for the true
        # rejection probability at each nominal alpha, faintly filled between
        ax.plot(alphas, lo, lw=1, color=color, alpha=0.7, zorder=2,
                label=f'{int(conf * 100)}% CI')
        ax.plot(alphas, hi, lw=1, color=color, alpha=0.7, zorder=2)
        ax.fill_between(alphas, lo, hi, color=color, alpha=0.12, zorder=0)

    ax.set_xlim(0, alpha_max)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    if owns_fig:
        plt.tight_layout()


# The null calibration figure shows only GLOW (the paper's FWER claim is about
# GLOW); the voxel-wise methods are dropped, and the Focus arm before that
# (_ARMS_SKIP / _select_glow_arm). Left-to-right column order.
_CALIB_METHODS = ('GLOW',)


def _plot_calibration_faceted(label: str, df, out,
                              facet: str = 'source') -> None:
    """Lay out the GLOW-only FWER calibration as a single source x arm row.

    One full row of source x GLOW-arm cells, source-major; every method
    outside _CALIB_METHODS is dropped (the voxel-wise arms; the Focus arm goes
    earlier, in _select_glow_arm). Each cell is one (source, arm) calibration
    curve with its Clopper-Pearson 95% band (plot_calibration), both axes on
    0..1, titled 'arm (source)'.

    Args:
        label (str): cache name; used in the output filename
        df: the null cache's tidy results (needs min_pval, label, facet)
        out (pathlib.Path): directory the figure is written into
        facet (str): categorical source column crossed with the GLOW arm
    """
    df = df.copy()
    df['min_pval'] = pd.to_numeric(df['min_pval'], errors='coerce')
    df = df.dropna(subset=['min_pval'])

    have_src = set(df[facet].dropna().unique())
    have_lab = set(df['label'].dropna().unique())
    sources = [s for s in _SOURCE_ORDER if s in have_src]
    methods = [m for m in _CALIB_METHODS if m in have_lab]
    if not sources or not methods:
        print(f'  (no GLOW null rows for {label} — skipping calibration)')
        return

    palette = get_cmap_dict(methods)
    cells = [(src, method) for src in sources for method in methods]
    fig, axes = plt.subplots(1, len(cells), figsize=(4 * len(cells), 4),
                             squeeze=False)
    for k, (src, method) in enumerate(cells):
        ax = axes[0, k]
        pvals = df.loc[(df[facet] == src) & (df['label'] == method),
                       'min_pval'].values
        plot_calibration(pvals, palette[method], ax=ax)
        ax.set_title(f'{method} ({src})')
        ax.set_xlabel('nominal $\\alpha$')
        if k == 0:
            ax.set_ylabel('empirical rejection rate')
    axes[0, 0].legend(frameon=False, fontsize=8, loc='lower right')
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


def _draw_diff(ax, df, x: str, metric: str, *, one_label: str = 'GLOW',
               hue: str = 'label', alpha: float = .5) -> list:
    """Draw one_label minus the best non-GLOW method into ax; return CSV rows.

    The head-to-head panel: the per-trial advantage of one_label (default
    GLOW) over the best competing method -- the largest metric among the
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
# absent from the cache (an HCP-only one, say) just drops out.
_SOURCE_ORDER = ('HCP', 'WGN')


def plot_source_grid(label: str, df, *, x: str, metrics: list, out,
                     one_label: str = 'GLOW', ci: int = 95,
                     thresh_metric: str = 'dice', level: float = 0.5) -> None:
    """Plot the stacked per-source detection figure: two rows per source.

    One SubFigure per source (its banner the source name), stacked HCP over
    WGN; within each a 2 x len(metrics) grid whose top row is the mean score +
    central ci% percentile band per method (_draw_metric_band) and whose bottom
    row is one_label (GLOW) minus the best non-GLOW alternative
    (_draw_diff), with the metrics (dice / sens / ppv) across the columns. A
    dashed line marks the threshold level on the thresh_metric (Dice) panel,
    where the discovery thresholds (write_threshold_table, plotted once per
    cache) are read.

    Writes {label}.pdf and the companion {label}_diff.csv (one block per GLOW
    variant; see _write_diff_csv).

    Args:
        label (str): cache name; the output filename stem
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
    """Write the diff-grid mean line to CSV.

    diff_long is the tidy mean line behind the diff rows (_draw_diff), one row
    per (facet, method, x, metric): the seed-averaged absolute scores (glow =
    GLOW variant named in method, other = best non-GLOW alternative), their
    difference (mean_diff), and the win rate (win = fraction of trials with
    the GLOW variant strictly above the best alternative). It is reshaped to
    one row per (facet, method, x) with a glow_<m> / other_<m> / <m>_diff /
    <m>_win block per metric, written to {label}_diff.csv.

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


def _order_threshold_rows(wide):
    """Sort a threshold table by source order, then canonical method order.

    Methods follow the config catalogue order under their figure labels (GLOW
    first, then the voxel-wise methods; see config.ana_kwargs_dict and
    _ARM_LABEL); a label outside the catalogue sorts last. Sources follow
    _SOURCE_ORDER (HCP over WGN).
    """
    m_rank = {_ARM_LABEL.get(m, m): i
              for i, m in enumerate(ana_kwargs_dict)}
    s_rank = {s: i for i, s in enumerate(_SOURCE_ORDER)}
    keyed = wide.assign(
        _s=wide['source'].map(lambda s: s_rank.get(s, len(s_rank))),
        _m=wide['method'].map(lambda m: m_rank.get(m, len(m_rank))))
    return (keyed.sort_values(['_s', '_m'])
                 .drop(columns=['_s', '_m']).reset_index(drop=True))


def threshold_table(df, *, x: str, metric: str = 'dice', level: float = 0.5):
    """Wide table of each method's discovery threshold in absolute x, per b.

    A method's discovery threshold is the swept-axis value at which its mean
    metric (averaged across trials) first crosses level -- the effect strength
    at which it starts recovering the support. Using the half-maximal (level =
    0.5) point of a monotone performance curve as a threshold is the standard
    dose-response / psychometric convention (the EC50 / 50%-detection point,
    the steepest, most reproducible part of the sigmoid); Dice itself is Dice
    1945, with Dice > 0.7 the usual "good overlap" line (Zijdenbos 1994), so
    level is a parameter.

    Each entry is the raw threshold, not a ratio: for x = effect_llr, the
    size-normalized LLR / |r| at which the method reaches level Dice, read
    directly off the swept axis. A structural axis that varies alongside x (b
    in the llr sweep) becomes the columns, so each b gets its own threshold
    column; with none varying the single column is named by x. A method whose
    mean curve never reaches level within the swept range,
    or already sits at/above it at the weakest x, has no finite crossing -- its
    cell is nan, the reason in the returned status ('above' / 'below').

    Args:
        df: a tidy_run_ana frame (needs source / label / x / metric, and any
            varying secondary axis such as b)
        x (str): the swept-axis column (effect strength when x is effect_llr)
        metric (str): the metric whose level crossing defines the threshold
        level (float): the crossing level (0.5 = half-maximal)

    Returns:
        (wide, status): wide is a DataFrame with columns source, method, then
            one threshold column per varying secondary value (b=1 / b=2 / b=3),
            or a single column named by x when none varies; status maps
            (source, method, column) to the _crossing status ('ok' / 'below' /
            'above'), for rendering the censored (nan) cells. Empty frame and
            empty dict in, empty out.
    """
    df = df.copy()
    df[x] = pd.to_numeric(df[x], errors='coerce')
    # keyed by method label, so rows whose ana repr did not resolve to a
    # catalogue label (stale records from a since-changed knob such as
    # n_perm_inner) carry no method and are dropped; an all-unlabelled cache
    # then yields an empty table rather than a groupby that silently drops
    # every NaN-label row and leaves wide without a method column.
    df = df.dropna(subset=[x, 'label'])
    if df.empty:
        return df.iloc[0:0], {}

    secondary = [a for a in _SECONDARY_AXES
                 if a != x and a in df.columns and df[a].dropna().nunique() > 1]

    rows = []
    status = {}
    for keys, sub in df.groupby(['source', *secondary]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        cell = dict(zip(['source', *secondary], keys))
        col = (' '.join(f'{s}={int(cell[s])}' for s in secondary)
               if secondary else x)
        mean_curve = sub.groupby(['label', x])[metric].mean()
        for lab in mean_curve.index.get_level_values(0).unique().tolist():
            s = mean_curve.loc[lab].dropna().sort_index()
            thr, st = _crossing(np.asarray(s.index, dtype=float),
                                np.asarray(s.values, dtype=float), level)
            rows.append({'source': cell['source'], 'method': lab,
                         '_col': col, 'thr': thr})
            status[(cell['source'], lab, col)] = st

    long = pd.DataFrame(rows)
    # dropna=False keeps a method whose crossing is nan at every b (censored),
    # so the row survives to be rendered from its status rather than vanishing
    wide = long.pivot_table(index=['source', 'method'], columns='_col',
                            values='thr', dropna=False).reset_index()
    wide.columns.name = None
    return _order_threshold_rows(wide), status


def write_threshold_table(label: str, df, *, x: str, out, metric: str = 'dice',
                          level: float = 0.5) -> None:
    """Write and print the absolute discovery-threshold table.

    Rows are the methods (the GLOW arms first), the columns the varying
    secondary axis (b in the llr sweep); each cell is the effect_llr (the
    size-normalized LLR / |r|) at which that method's mean Dice first reaches
    level (threshold_table). Printed once per source and written to
    {label}_threshold.csv. A censored cell prints as < the weakest tested
    effect (already above level there) or > the strongest (never reaches it).

    Args:
        label (str): cache name; used in the output filename
        df: the cache's tidy_run_ana results
        x (str): the swept-axis column
        out (pathlib.Path): directory the CSV is written into
        metric (str): the metric whose level crossing defines the threshold
        level (float): the crossing level (0.5 = half-maximal Dice)
    """
    wide, status = threshold_table(df, x=x, metric=metric, level=level)
    if wide.empty:
        return
    path = out / f'{label}_threshold.csv'
    wide.to_csv(path, index=False, float_format='%.4f')
    print(f'saved: {path}')

    val_cols = [c for c in wide.columns if c not in ('source', 'method')]
    xlo, xhi = float(df[x].min()), float(df[x].max())

    def render(src, method, col, value):
        """Format one threshold cell, censored ends read off the status map."""
        if pd.notnull(value):
            return f'{value:>10.4f}'
        st = status.get((src, method, col))
        if st == 'below':
            return f'{f"<{xlo:g}":>10}'
        if st == 'above':
            return f'{f">{xhi:g}":>10}'
        return f'{"—":>10}'

    metric_title = _METRIC_TITLES.get(metric, metric)
    print(f'  {_X_PARAM_LABELS.get(x, x)} at {metric_title} >= {level:g} '
          f'(mean across trials):')
    for src in wide['source'].drop_duplicates():
        sub = wide[wide['source'] == src]
        print(f'    source={src}')
        header = ' '.join(f'{c:>10}' for c in val_cols)
        print(f'      {"method":<11} {header}')
        for _, r in sub.iterrows():
            cells = ' '.join(render(src, r['method'], c, r[c])
                             for c in val_cols)
            print(f'      {r["method"]:<11} {cells}')


# ---------------------------------------------------------------------------
# MANCOVA stat bake-off (stat cache): VBA / VBA-TFCE / CET x 5 stats x {raw, z}
# ---------------------------------------------------------------------------

def tidy_stat(raw):
    """Normalise the stat cache's leaves to a tidy per-variant frame.

    Maps each run_stat leaf (results.stat_cell_df) to its (method, stat, zt)
    variant via the recorded recipe (_STAT_VARIANT) and derives Dice from the
    confusion counts. Rows whose recipe is not a stat-cache variant, or that
    lack counts, drop out; a variant recorded twice (a rerun) collapses to one
    row (the value is deterministic).

    Args:
        raw: the stat_cell_df frame (cell / ana / stat_name / count columns).

    Returns:
        a DataFrame with cell, method, stat, zt, dice (one row per variant).
    """
    if raw.empty:
        return raw
    var = [_STAT_VARIANT.get(k) for k in zip(raw['ana'], raw['stat_name'])]
    out = pd.DataFrame({'cell': raw['cell'].values})
    out['method'] = [v[0] if v else None for v in var]
    out['stat'] = [v[2] if v else None for v in var]
    out['zt'] = [v[1] if v else None for v in var]
    tp, fp, fn = (pd.to_numeric(raw[c], errors='coerce').values
                  for c in ('tp', 'fp', 'fn'))
    denom = 2 * tp + fp + fn
    out['dice'] = np.where(denom > 0, 2 * tp / denom, np.nan)
    out = out.dropna(subset=['method'])
    return out.drop_duplicates(['cell', 'method', 'stat', 'zt'])


def _stat_balanced(df):
    """Keep only cells recorded with the full variant grid (balanced N).

    A cell fit by an interrupted worker is missing its last-computed variants
    (grid.get_run_stat_list runs stat-major, llr..roys_root), which would give each
    method a different trial count. Restricting to cells with all
    len(RUN_STAT_LIST) variants makes every method's panel the same cells.

    Returns:
        (DataFrame, int, int): the filtered frame, kept cell count, dropped.
    """
    per_cell = df.groupby('cell')['dice'].size()
    full = per_cell.index[per_cell == len(RUN_STAT_LIST)]
    return df[df['cell'].isin(full)], len(full), df['cell'].nunique() - len(full)


def stat_tables(df, tol: float = 1e-9, decisive: float = 0.01):
    """Build the two stat-comparison frames from a tidy_stat frame.

    Restricts to the balanced panel (_stat_balanced), then per (cell, method,
    zt) group of the five stats takes the Dice range (max - min). Both frames
    are indexed by (method, zt) -- the raw and z-scored arms are separate
    methods, not two draws of one, and z-scoring interacts with the stat
    (it lifts TFCE far more than it lifts VBA / CET), so pooling them would
    average a good arm with a bad one. t1 summarises how often the stat choice
    matters; t2 gives mean Dice per stat with the best-worst gap. Both read the
    same panel, so their trial counts agree and their rows line up (one paper
    table each -- see write_stat_tables).

    Args:
        df: a tidy_stat frame.
        tol (float): Dice range at / below which the five stats count as tied.
        decisive (float): Dice range above which the stat choice counts as
            decisive on that trial -- best vs worst stat moves Dice by more
            than this, so a reader picking the wrong one pays for it.

    Returns:
        (t1, t2, meta): t1 indexed by (method, zt) with pct_all_tie,
            pct_decisive, mean_range, median_range, max_range; t2 indexed by
            (method, zt) with one column per stat in _STAT_ORDER, plus gap;
            meta is
            {n_cells, n_dropped, tol, decisive} -- the panel size, the partial
            cells excluded from it, and the two thresholds t1 was cut at.
    """
    kept, n_cells, n_drop = _stat_balanced(df)

    # the balanced panel scores every stat on the same trials; if not (a stat
    # missing across cells, or the filter let a lopsided cell through) the
    # per-stat means are not comparable, so surface it rather than average over
    # unequal supports
    per_stat = kept.groupby('stat').size().reindex(_STAT_ORDER)
    if per_stat.nunique(dropna=False) > 1:
        warnings.warn('stat bake-off: unequal trials per stat '
                      f'{per_stat.to_dict()} -- the balanced panel should give '
                      'every stat the same count; means are not comparable.')

    rows = pd.MultiIndex.from_product([_STAT_METHOD_ORDER, _ZT_ORDER],
                                      names=['method', 'zt'])
    g = kept.groupby(['cell', 'method', 'zt'])['dice']
    rng = (g.max() - g.min()).rename('range').reset_index()
    t1 = rng.groupby(['method', 'zt']).agg(
        pct_all_tie=('range', lambda s: 100 * (s <= tol).mean()),
        pct_decisive=('range', lambda s: 100 * (s > decisive).mean()),
        mean_range=('range', 'mean'),
        median_range=('range', 'median'),
        max_range=('range', 'max'),
    ).reindex(rows)
    t2 = (kept.groupby(['method', 'zt', 'stat'])['dice'].mean()
          .unstack()[_STAT_ORDER].reindex(rows))
    t2['gap'] = t2.max(axis=1) - t2.min(axis=1)
    return t1, t2, {'n_cells': n_cells, 'n_dropped': n_drop, 'tol': tol,
                    'decisive': decisive}


def _latex_table(path, colspec: str, header: list, rows: list,
                 group_header: list = None, note: list = None) -> None:
    """Write one booktabs tabular fragment for \\input into the paper.

    Just the tabular (no table float, caption or label) so the paper owns the
    surrounding environment and all prose; running .plot only refreshes the
    numbers.

    Args:
        path (pathlib.Path): destination .tex file.
        colspec (str): the tabular column spec (e.g. 'lrrrr').
        header (list): already-escaped column titles.
        rows (list): each an already-escaped list of cell strings.
        group_header (list): optional (title, span, align) triples banding the
            header columns into a \\multicolumn row above it; spans must cover
            every column. align is that cell's own column spec ('c', 'c|'), so
            the caller keeps whatever vertical rules colspec draws. A titled
            group also gets a \\cmidrule(lr); an empty title just spans.
        note (list): optional lines emitted as LaTeX comments above the
            tabular, to pin down how the numbers were computed without
            printing anything into the paper.
    """
    lines = ['% requires \\usepackage{booktabs}; \\input inside a table env']
    lines += [f'% {n}' for n in note or []]
    lines += [f'\\begin{{tabular}}{{{colspec}}}', '  \\toprule']
    if group_header:
        spans = sum(span for _, span, _ in group_header)
        if spans != len(header):
            raise ValueError(f'group_header spans {spans} columns, header has '
                             f'{len(header)}')
        cells, rules, col = [], [], 1
        for title, span, align in group_header:
            cells.append(f'\\multicolumn{{{span}}}{{{align}}}{{{title}}}')
            if title:
                rules.append(f'\\cmidrule(lr){{{col}-{col + span - 1}}}')
            col += span
        lines += ['  ' + ' & '.join(cells) + r' \\', '  ' + ' '.join(rules)]
    lines += ['  ' + ' & '.join(header) + r' \\', '  \\midrule']
    lines += ['  ' + ' & '.join(r) + r' \\' for r in rows]
    lines += ['  \\bottomrule', '\\end{tabular}', '']
    path.write_text('\n'.join(lines))


def write_stat_tables(label: str, df, out) -> None:
    """Write (and print) the stat bake-off's two paper tables as .tex.

    Both split the raw and z-scored arms -- z-scoring interacts with the stat,
    so they never pool (see stat_tables) -- and both read the same balanced
    panel (cells recorded with the full variant grid), so every stat is scored
    on the same trials (stat_tables warns otherwise).

    stat_dice.tex is one row per method, its five stat means banded under raw
    and again under z-scored, with one bold on the method's best (scaling,
    stat) cell of the ten. stat_matters.tex is one row per arm: whether the
    choice matters at all -- the all-tie / decisive fractions and the mean /
    median per-trial Dice range, which is where a reader sizes up a win. Two
    narrow tables rather than one wide one, to fit a single paper column. Each
    file is a bare booktabs tabular (no caption / label); the paper owns the
    table environment and prose (see _latex_table).

    Args:
        label (str): cache name; unused in the output but kept for the dispatch
            signature symmetry with plot_cache / plot_runtime.
        df: a tidy_stat frame.
        out (pathlib.Path): directory the .tex files are written into.
    """
    if df.empty:
        print(f'  (no rows for {label} — skipping)')
        return
    t1, t2, meta = stat_tables(df)
    panel = (f'{meta["n_cells"]} planted cells, b=2'
             + (f'; {meta["n_dropped"]} partial cells excluded'
                if meta['n_dropped'] else ''))
    scaling = ('scaling: z = each voxel z-scored across the permutations '
               'before the max-stat null (AnalysisVoxel.z_score_stat)')

    # one bold per method, not per row: the best (scaling, stat) cell over both
    # of its rows, so the mark reads as the best this method can do. Ties break
    # by (_ZT_ORDER, _STAT_ORDER) -- hotel_tr takes it over the bitwise-
    # identical roys_root of a rank-1 hypothesis matrix.
    best_cell = {m: t2.loc[m, _STAT_ORDER].stack().idxmax()
                 for m in _STAT_METHOD_ORDER}

    def dice_row(method):
        """Render a method's row: the five stats raw, then the five z-scored.

        Whether the bolded win means anything is stat_matters.tex's job -- the
        bold marks the max, not a margin.
        """
        cells = [method]
        for zt in _ZT_ORDER:
            row = t2.loc[(method, zt)]
            for s in _STAT_ORDER:
                v = f'{row[s]:.3f}'
                cells.append(f'\\textbf{{{v}}}'
                             if best_cell[method] == (zt, s) else v)
        return cells

    n_stat = len(_STAT_ORDER)
    stat_names = [_STAT_PRETTY[s] for s in _STAT_ORDER]
    _latex_table(
        out / 'stat_dice.tex',
        'l|' + '|'.join(['r' * n_stat] * len(_ZT_ORDER)),
        ['Method'] + stat_names * len(_ZT_ORDER),
        [dice_row(m) for m in _STAT_METHOD_ORDER],
        group_header=[('', 1, 'l|')] + [
            (_ZT_PRETTY[zt], n_stat, 'c|' if zt != _ZT_ORDER[-1] else 'c')
            for zt in _ZT_ORDER],
        note=[f'panel: {panel}', scaling])

    _latex_table(
        out / 'stat_matters.tex', 'llrrrr',
        ['Method', 'Scaling', 'All tie (\\%)', 'Decisive (\\%)',
         'Mean range', 'Median range'],
        [[m, zt, f'{r.pct_all_tie:.1f}', f'{r.pct_decisive:.1f}',
          f'{r.mean_range:.4f}', f'{r.median_range:.4f}']
         for (m, zt), r in t1.iterrows()],
        note=[f'panel: {panel}', scaling,
              'range: per-trial (max - min) Dice over the five stats; '
              f'all tie <= {meta["tol"]:g}, decisive > {meta["decisive"]:g}'])

    print(f'saved: {out / "stat_dice.tex"}, {out / "stat_matters.tex"}')
    print(f'  panel: {panel}')
    print('  does the stat matter:')
    print(t1.to_string(float_format=lambda v: f'{v:.4f}'))
    print('  mean Dice per stat:')
    print(t2.to_string(float_format=lambda v: f'{v:.3f}'))


# ---------------------------------------------------------------------------
# Flat-score caches (segment / prune): a source x metric grid vs effect_llr
# ---------------------------------------------------------------------------

def _tidy_flat_cache(raw, leaf: str, label_col: str, label_fn=None):
    """Normalise a flat-score cache to a tidy per-(trial, method) metric frame.

    The tidy path for leaves that return a flat {tp, fp, tn, fn} score
    (run_segment / run_prune) rather than run_ana's nested score.target block.
    Reads the shared data / effect ancestors and the flat counts, attaches the
    method label off a recorded input column (the Ward mode / prune rule -- the
    cache axis), and derives dice/sens/ppv/spec (add_metric_cols).

    Args:
        raw: the provenance DataFrame (one row per leaf).
        leaf (str): the leaf function name = the score column prefix
            (run_segment / run_prune).
        label_col (str): the recorded input column naming the method.
        label_fn (Callable | None): maps a label_col value to the method label
            (prune's 'greedy' -> 'GLOW-greedy'); None uses the value as-is
            (segment's Ward-mode string).

    Returns:
        a tidy DataFrame, one row per (trial, method), with columns label,
        source (WGN / HCP), seed, b, effect_llr, the four confusion counts, and
        the derived dice/sens/ppv/spec (empty in, empty out).
    """
    def col(name):
        """Return raw[name], or an all-NaN column when absent."""
        if name in raw.columns:
            return raw[name]
        return pd.Series(np.nan, index=raw.index)

    hcp_seed = pd.to_numeric(col('data_factory_hcp.in.seed'), errors='coerce')
    wgn_seed = pd.to_numeric(col('data_factory_wgn.in.seed'), errors='coerce')

    label = col(label_col)
    if label_fn is not None:
        label = label.map(lambda v: label_fn(v) if isinstance(v, str) else v)

    out = pd.DataFrame(index=raw.index)
    out['label'] = label
    out['source'] = np.where(hcp_seed.notna(), 'HCP', 'WGN')
    out['seed'] = wgn_seed.fillna(hcp_seed)
    hcp_b = col('data_factory_hcp.in.hcp_feats').map(
        lambda v: len(v) if isinstance(v, (list, tuple)) else np.nan)
    out['b'] = pd.to_numeric(col('data_factory_wgn.in.b'),
                             errors='coerce').fillna(hcp_b)
    out['effect_llr'] = pd.to_numeric(
        col('effect_factory_single.in.effect_llr'), errors='coerce')
    for cnt in ('tp', 'fp', 'tn', 'fn'):
        out[cnt] = pd.to_numeric(col(f'{leaf}.out.score.{cnt}'),
                                 errors='coerce')
    return add_metric_cols(out)


def tidy_segment(raw):
    """Normalise the segment cache to a tidy per-(trial, Ward mode) frame.

    The method label is the recorded Ward mode (config records str(mode), so
    the cell is already the mode string Naive / GLM Error / Focus).

    Args:
        raw: the segment cache's provenance frame (one row per run_segment leaf).

    Returns:
        a tidy_flat_cache frame (label = Ward mode); empty in, empty out.
    """
    if raw.empty:
        return raw
    return _tidy_flat_cache(raw, 'run_segment', 'run_segment.in.cluster_mode')


def tidy_prune(raw):
    """Normalise the prune cache to a tidy per-(trial, rule) frame.

    The method label is GLOW-<rule> off the recorded rule (maxllr / greedy /
    dp); the recorded Ward mode (run_prune.in.cluster_mode) rides along as the
    cluster_mode column, so plot_prune can split it into one figure per mode. A
    legacy record predating the mode axis carried the Focus default, so a
    missing mode reads back as Focus.

    Args:
        raw: the prune cache's provenance frame (one row per run_prune leaf).

    Returns:
        a tidy_flat_cache frame (label = GLOW-<rule>) plus a cluster_mode
        column (the Ward-mode string); empty in, empty out.
    """
    if raw.empty:
        return raw
    out = _tidy_flat_cache(raw, 'run_prune', 'run_prune.in.rule',
                           label_fn=lambda r: f'GLOW-{r}')
    mode_col = 'run_prune.in.cluster_mode'
    mode = (raw[mode_col] if mode_col in raw.columns
            else pd.Series(np.nan, index=raw.index))
    out['cluster_mode'] = mode.reindex(out.index).fillna(str(ClusterMode.FOCUS))
    return out


def _draw_metric_errbar(ax, df, x: str, metric: str, style: dict, *,
                        hue: str = 'label', z_mult: float = 1.96,
                        dodge: float = 0.03) -> None:
    """Draw each method's mean +/- 95% CI as x-dodged error bars into ax.

    Per x, the seed-mean of the metric with a 95% CI-of-the-mean bar
    (z_mult * SEM, SEM = std / sqrt(n_seed)), markers joined by a thin line in
    the method's style. Unlike a percentile band the CI narrows as sqrt(n_seed),
    so more seeds tighten it. The bars are dodged multiplicatively about each x
    (a fixed fraction per method, centred on the group) so overlapping methods
    stay legible on the log axis. PPV is nan for trials with no detections
    (glow.mask.stats_from_counts) and drops from its mean / SEM.

    Args:
        ax: matplotlib Axes to draw into.
        df: one source's tidy rows (numeric x / metric).
        x (str): the swept x-axis column.
        metric (str): the metric column plotted on the y-axis.
        style (dict): label -> {'color', 'ls'} (from _qual_style).
        hue (str): the method-label column.
        z_mult (float): SEM multiplier for the error bar (1.96 ~ 95% CI).
        dodge (float): fractional multiplicative x-dodge between methods.
    """
    labels = sorted(df[hue].dropna().unique().tolist())
    n = len(labels)
    for i, lab in enumerate(labels):
        g = df[df[hue] == lab].groupby(x)[metric]
        mean, sem = g.mean(), g.std() / np.sqrt(g.count())
        # centre the per-method dodge on the group so it sits over the true x
        factor = 1 + dodge * (i - (n - 1) / 2)
        st = style[lab]
        ax.errorbar(mean.index.values * factor, mean.values,
                    yerr=z_mult * sem.values, marker='o', ms=4, lw=1.5,
                    ls=st['ls'], color=st['color'], capsize=2, label=lab)


def plot_metric_grid(label: str, df, out, *, x: str = 'effect_llr',
                     metrics=('dice', 'sens', 'ppv')) -> None:
    """Plot a source x metric grid of per-method mean +/- 95% CI vs the swept x.

    A 2 x len(metrics) grid: one row per data source (HCP over WGN), one column
    per metric (Dice / Sensitivity / PPV). Each panel draws the per-method
    seed-mean with a 95% CI-of-the-mean error bar, x-dodged so the methods stay
    legible (_draw_metric_errbar), against effect_llr on a log x-axis, one series
    per method (the Ward mode for segment, the prune rule for prune). Methods
    take the Okabe-Ito qualitative style (_qual_style: colour + line style).
    Writes {label}.pdf.

    Args:
        label (str): cache name; the output filename stem.
        df: a tidy frame (needs source / label / seed / x / the metric columns).
        out (pathlib.Path): directory the figure is written into.
        x (str): the swept x-axis column (effect_llr).
        metrics (iterable): the metric columns, one panel column each.
    """
    metrics = list(metrics)
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

    style = _qual_style(df['label'].dropna().unique().tolist())
    log_x = pd.notnull(df[x].min()) and df[x].min() > 0
    ncols = len(metrics)

    fig, axes = plt.subplots(len(sources), ncols, sharex=True,
                             figsize=(4.2 * ncols, 3.8 * len(sources)),
                             squeeze=False)
    for i, src in enumerate(sources):
        dsrc = df[df['source'] == src]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            _draw_metric_errbar(ax, dsrc, x, metric, style)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            if log_x:
                ax.set_xscale('log')
            if i == 0:
                ax.set_title(_METRIC_TITLES.get(metric, metric))
            if i == len(sources) - 1:
                ax.set_xlabel(_X_PARAM_LABELS.get(x, x))
        axes[i, 0].set_ylabel(f'{src}\nmean (95% CI)')
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out / f'{label}.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'saved: {path}')


def _mode_slug(mode: str) -> str:
    """Filename-safe token for a Ward mode ('GLM Error' -> 'GLM_Error')."""
    return str(mode).replace(' ', '_')


# prune rules kept in the cache and records but dropped from the paper figures:
# maxllr (the single max-LLR region) is a diagnostic baseline, not a paper
# method, so plot_prune skips it (see PRUNE_RULES in config).
_PRUNE_LABELS_SKIP = ('GLOW-maxllr',)


def plot_prune(label: str, df, out) -> None:
    """Plot one metric grid per Ward clustering mode for the prune cache.

    The prune cache crosses the pruning rules with both Ward modes (Focus / GLM
    Error), so a single grid would overlay two clusterings. This draws one
    source x metric grid per mode (plot_metric_grid), writing {label}_{mode}.pdf
    so the clusterings are compared side by side rather than on one axis. The
    diagnostic maxllr rule is dropped (_PRUNE_LABELS_SKIP).

    Args:
        label (str): cache name; each figure's stem is {label}_{mode}.
        df: a tidy_prune frame (needs the cluster_mode column).
        out (pathlib.Path): directory the figures are written into.
    """
    df = df[~df['label'].isin(_PRUNE_LABELS_SKIP)]
    for mode, df_mode in df.groupby('cluster_mode'):
        plot_metric_grid(f'{label}_{_mode_slug(mode)}', df_mode, out)



# ---------------------------------------------------------------------------
# Runtime sweeps (wall time vs one cost knob)
# ---------------------------------------------------------------------------

def tidy_runtime(name: str, raw):
    """Normalise a runtime cache's provenance frame to (label, x, time_sec).

    The runtime counterpart to tidy_run_ana: collapses the wide provenance
    frame to one tidy row per timed leaf, reading the method label, the swept
    x-axis value, and the wall time. The leaf prefix and swept axis come from
    _RUNTIME_SPEC (time is the signal, so unlike the detection path the x is
    not inferred from what varies). Both timing leaves name the method in the
    recipe (in.ana), as tidy_run_ana does, and return num_vox bare.

    A knob is read wherever it was declared. b rides the data grid, so it comes
    off the HCP feature subset the factory was given; the rest are explicit
    1perm leaf inputs. Every runtime cache is HCP, so the seed is the HCP
    factory's.

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
    out['label'] = col(f'{leaf}.in.ana').map(_LABEL_OF_ANA)
    out['num_vox'] = pd.to_numeric(col(f'{leaf}.out.num_vox'), errors='coerce')

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
    linear). Methods use the shared palette (COLOR_ANALYSIS), with a seaborn
    fallback for any label outside it (get_cmap_dict).

    Args:
        name (str): cache name; used in the output filename.
        df: the cache's tidy_runtime results.
        out (pathlib.Path): directory the figure is written into.
        log_x_ratio (float): x max/min ratio at or above which the x-axis is
            log-scaled.
    """
    df = _select_glow_arm(df.dropna(subset=['x', 'time_sec', 'label']))
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
        name (str): cache name; used in the output filenames.
        df: the tidy_inner_edge frame.
        out (pathlib.Path): directory the figure and JSON are written into.
        alpha_fwer (float): FWER level (annotated on the y-axis).
    """
    df = _select_glow_arm(
        df.dropna(subset=['num_inner_perm', 'threshold', 'label']))
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
        ax.set_title(src)
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
# Per-cache dispatch + CLI
# ---------------------------------------------------------------------------

def plot_cache(label: str, df, out,
               metrics: list = ['dice', 'sens', 'ppv']) -> None:
    """Write one run_ana cache's figure, dispatching on its swept axis.

    The null path (no effect planted) gets a faceted FWER calibration curve;
    every other cache gets the stacked per-source detection figure
    (plot_source_grid: an HCP block over a WGN block, each a mean-band row and
    a GLOW diff row across the metric columns) plus one cache-level
    discovery-threshold table (write_threshold_table). The x-axis is inferred
    from the data (_infer_x), so no config plot spec is needed. A cache that
    also varies a structural axis besides x (the llr sweep varies b) is drawn
    one figure per value of it (_split_by_secondary), each suffixed into the
    label, while the threshold table spreads that axis across its columns.

    Args:
        label (str): cache name; used in output filenames
        df: the cache's tidy_run_ana results
        out (pathlib.Path): directory the figures are written into
        metrics (list): metric columns plotted as the panel columns
    """
    df = _select_glow_arm(df)
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
    # (b in the llr sweep); absolute effect_llr per method (threshold_table)
    write_threshold_table(label, df, x=x, out=out)


def main(argv=None) -> None:
    """Plot the detection and runtime caches from the shared records.

    For each selected cache (every detection and runtime cache in CONFIG by
    default, or the names given on the command line), reads its provenance
    frame (make_csv.write_config_csv, which refreshes that cache's CSV on the
    way past) and plots it: a runtime cache is normalised with tidy_runtime and
    drawn by plot_runtime (wall time vs its cost knob); every other run_ana
    cache is normalised with tidy_run_ana and drawn by plot_cache (detection
    sweep / calibration). The stat bake-off is read straight from the run_stat
    leaves (results.stat_cell_df) and written as two paper tables by
    write_stat_tables. The segment / prune caches are normalised with
    tidy_segment / tidy_prune and drawn as a source x metric grid vs
    effect_llr: segment by plot_metric_grid, prune by plot_prune (one grid per
    Ward clustering mode). Figures / tables land in results/_latest, so a
    mid-benchmark run yields intermediate output.

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
    from .run import (run_ana, run_inner_edge, run_prune,
                      run_segment, run_stat)
    from . import make_csv, results

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
    stat_names = [n for n, cfg in CONFIG.items() if cfg[3] is run_stat]
    segment_names = [n for n, cfg in CONFIG.items() if cfg[3] is run_segment]
    prune_names = [n for n, cfg in CONFIG.items() if cfg[3] is run_prune]
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
        names = (detect_names + runtime_names + edge_names
                 + stat_names + segment_names + prune_names)

    out = glow._extra.benchmark.get_path_result() / '_latest'
    out.mkdir(exist_ok=True)

    n_plotted = 0
    for name in names:
        if name in _RUNTIME_SPEC:
            df = tidy_runtime(name, make_csv.write_config_csv(name))
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {len(df)} runtime rows ===')
            plot_runtime(name, df, out)
            n_plotted += 1
        elif name in detect_names:
            df = tidy_run_ana(make_csv.write_config_csv(name))
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {len(df)} run_ana rows ===')
            plot_cache(name, df, out)
            n_plotted += 1
        elif name in edge_names:
            df = tidy_inner_edge(make_csv.write_config_csv(name))
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {len(df)} inner-edge rows ===')
            plot_inner_edge(name, df, out)
            n_plotted += 1
        elif name in stat_names:
            df = tidy_stat(results.stat_cell_df())
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {df["cell"].nunique()} cells, '
                  f'{len(df)} variant rows ===')
            write_stat_tables(name, df, out)
            n_plotted += 1
        elif name in segment_names:
            df = tidy_segment(make_csv.write_config_csv(name))
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {len(df)} segment rows ===')
            plot_metric_grid(name, df, out)
            n_plotted += 1
        elif name in prune_names:
            df = tidy_prune(make_csv.write_config_csv(name))
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {len(df)} prune rows ===')
            plot_prune(name, df, out)
            n_plotted += 1
        else:
            print(f'  ({name} is not a detection or runtime cache — skipping)')

    if n_plotted == 0:
        print(f'no results found under {out.parent}')


if __name__ == '__main__':
    main()
