"""Plot the run_ana benchmark caches from the shared provenance records.

Reads each cache's provenance frame (make_csv.write_config_csv: one wide row
per run_ana leaf, namespaced by the producing function), normalises it to one
tidy row per (trial, recipe) with tidy_run_ana, and writes one figure set per
cache into its own results/_latest/<cache> directory. Reading through
make_csv refreshes each plotted cache's CSV in passing, so table and figures
come from the same records.

The tidy frame is what the plotters consume: a label (method), a source
(WGN / HCP, which share each cache and face apart here), the swept axes
(effect_llr, b, num_img, the realized effect fraction) and the metrics
(dice / sens / ppv / spec) derived from the score's confusion counts. The
x-axis is inferred from what varies in the cache rather than declared: an
all-null effect grid is the FWER calibration path, else the first of
effect_llr / b / num_img / effect_perc that varies is swept.

Every figure outside the segment / prune / race-retention families reports
the arms config names (REPORTED_GLOW_LABEL_LIST), a lone one drawn plainly as
GLOW: any other GLOW recipe is dropped and the survivors renamed at the plot
layer (_select_glow_arm), never in the caches. Throughout, an arm's colour is
its Ward projection and its dash the selection rule (_ARM_STYLE).

Each cache then gets either a GLOW-only FWER calibration row (null) -- one
cell per source x reported arm, nominal alpha vs empirical rejection rate,
with a Clopper-Pearson 95% band -- or one stacked detection figure, an HCP
block over a WGN block, each a 2 x 3 grid whose top row is the per-method
mean score with a central 95% band and whose bottom row is the GLOW
head-to-head diff, over the dice / sens / ppv columns -- the diff row only
where the source has a non-GLOW method to diff against. Alongside it a
discovery-threshold table (write_threshold_table) records the absolute effect
strength at which each method's mean Dice first reaches 0.5, and
{label}_tables.txt (write_table_txt) carries the drawn figure's own numbers
as plain text: the per-method score matrices (its region counts among them,
where its methods select regions), those thresholds and the head-to-head
against the reported arm.

The runtime family is plotted apart (tidy_runtime / plot_runtime): those caches
hold detection fixed and sweep one cost knob, so the signal is the leaf wall
time (RECORDER time_sec), not a score. Each is one time-vs-knob curve per
method, minutes on a log y-axis, and where the knob spans a decade or more it
is log too and each curve carries its fitted exponent (_loglog_slope).

The segment and prune caches share a flatter path (tidy_segment / tidy_prune /
plot_metric_grid): their leaves return a flat {tp, fp, tn, fn} score, not
run_ana's nested target block. Each is drawn as a source x metric grid of the
per-method seed-mean with a 95% CI error bar vs effect_llr, styled per method
(Ward mode / prune rule). Prune crosses its rules with both Ward modes, so
plot_prune draws one grid per mode. segment_perc_llr crosses the Ward modes
with frac_segment, the share of the images the tree is built on, so
plot_segment_llr holds the Ward mode fixed and draws the llr sweep once per
fold share, and plot_segment_compare pages the same frame by fold share, with
the Ward modes and the cohorts side by side.

Prune also reports what its rules select, not just how well: plot_prune_regions
draws each rule's region count against the effect strength, both Ward modes on
one page, against the one-region reference the caches plant. The GLOW arm cache
gets the same read off run_ana's own count (plot_cache), since its methods are
those rules crossed with the Ward projections.

With no arguments the CLI plots every cache in the catalogue; passing names
restricts it.
"""
import colorsys
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import beta

import glow._extra.benchmark
import glow.mask
from glow.analysis import AnalysisGLOWBase
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import stat_dict
from .config import (ana_kwargs_dict, REPORTED_GLOW_LABEL,
                     REPORTED_GLOW_LABEL_LIST, RUN_STAT_LIST)
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
    # the GLOW arms are added below, off _ARM_STYLE, which is where their
    # projection / rule encoding lives
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


# config.REPORTED_GLOW_LABEL_LIST is what the figures report, and every other
# GLOW recipe is dropped (_ARMS_SKIP) so a panel does not re-argue a knob a
# tuning cache settles; the survivors are renamed (_arm_labels). Both taken
# off the catalogue, so adding or renaming a variant in config needs no edit
# here. Every variant stays in the caches and the records -- each is a
# real recipe, and the segment / prune families exist to compare the two
# clusterings. Those families label by Ward mode (Focus / GLM Error) and prune
# rule (GLOW-greedy / GLOW-dp) rather than by analysis arm, so neither the
# drop nor the rename reaches them.
_ARMS_SKIP = tuple(label for label, ana in ana_kwargs_dict.items()
                   if isinstance(ana, AnalysisGLOWBase)
                   and label not in REPORTED_GLOW_LABEL_LIST)


def _arm_labels(label_list) -> dict:
    """Map each reported GLOW recipe label to the name a figure gives it.

    Args:
        label_list: config's reported arms, GLOW-<projection>-<rule> labels.

    Returns:
        dict: recipe label -> figure label. One arm is plainly GLOW; several
            drop the rule they share and read as GLOW-<projection>. Empty
            when dropping the rule would collide two arms (a pair separated
            by the rule alone), where the recipe labels are the figures'.
    """
    label_list = tuple(label_list)
    if len(label_list) == 1:
        return {label_list[0]: 'GLOW'}
    out = {lab: lab.rsplit('-', 1)[0] for lab in label_list}
    return {} if len(set(out.values())) < len(out) else out


_ARM_LABEL = _arm_labels(REPORTED_GLOW_LABEL_LIST)

# the arm the head-to-head row draws when a caller names none, under the label
# the figures give it
_DIFF_DEFAULT = _ARM_LABEL.get(REPORTED_GLOW_LABEL, REPORTED_GLOW_LABEL)

def _select_glow_arm(df):
    """Drop the unreported GLOW arms and label the reported one GLOW.

    Args:
        df: a tidy frame from any family; needs a label column to filter on

    Returns:
        the frame without its _ARMS_SKIP rows, the reported arm's rows
        renamed by projection (_ARM_LABEL); unchanged when empty or
        unlabelled (the segment / prune frames label by mode / rule).
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

# Methods whose raw block stat_dice.tex bolds entire rather than at its
# argmax. With one contrast column rank(H) = 1, so the five stats are strictly
# increasing functions of a single eigenvalue, and CET rejects on their
# ordering alone (a quantile cluster-forming threshold, integer cluster sizes,
# a rank-count p-value). No raw cell is a win over the others.
_STAT_BOLD_RAW_ROW = frozenset({'CET'})


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


# The GLOW arms' own styling: a Ward projection in the colour, a selection
# rule in the dash, since two curves and their bands never separate by
# lightness alone. The Focus arm keeps the palette's teal, so the method a
# reader has followed through every figure keeps its colour. An arm appears
# under its figure label as well (_ARM_LABEL), since that is what a relabelled
# frame carries.
_TEAL = _hls_hex(0/4 + _H, l=_L * 0.6)
_ARM_STYLE = {
    'GLOW-Focus-greedy': {'color': _TEAL, 'ls': '-'},
}
_ARM_STYLE.update({fig: _ARM_STYLE[raw] for raw, fig in _ARM_LABEL.items()})

# an arm is a method like any other where a figure colours by palette alone
# (the runtime curves, the calibration cells), so the same colour reaches
# get_cmap_dict
COLOR_ANALYSIS.update({lab: st['color'] for lab, st in _ARM_STYLE.items()})


def _method_style(label_list) -> dict:
    """Map method labels to a (colour, line style) pair for one figure.

    A GLOW arm takes _ARM_STYLE -- the projection in the colour, the selection
    rule in the dash -- under either its recipe or its figure label; every
    other method takes its palette colour (get_cmap_dict), solid.

    Args:
        label_list: the method labels to style.

    Returns:
        dict: label -> {'color': str, 'ls': str}.
    """
    color = get_cmap_dict([lab for lab in label_list
                           if lab not in _ARM_STYLE])
    return {lab: (_ARM_STYLE[lab] if lab in _ARM_STYLE
                  else {'color': color[lab], 'ls': '-'})
            for lab in label_list}


def _seq_style(value_list) -> dict:
    """Map ordered values to a sequential colour ramp, one line style.

    The counterpart to _qual_style for a hue that is a quantity rather than a
    method (the segmentation fold share): the values are ordered, so they take
    a perceptually uniform ramp in that order -- dark for the smallest, light
    for the largest -- and one line style throughout, since the ordering is
    what the reader follows and a dash pattern would cut across it. The ramp
    stops short of viridis's brightest yellow, which washes out on white.

    Args:
        value_list: the hue values to style; ordered by sort order.

    Returns:
        dict: value -> {'color': rgba tuple, 'ls': str}.
    """
    values = sorted(value_list)
    ramp = plt.get_cmap('viridis')(np.linspace(0, 0.88, max(len(values), 1)))
    return {v: {'color': c, 'ls': '-'} for v, c in zip(values, ramp)}


def _hue_text(value) -> str:
    """Legend text for one hue value (a numeric hue as a short decimal)."""
    if isinstance(value, (float, np.floating)):
        return f'{value:g}'
    return str(value)


_METRIC_TITLES = {
    'dice': 'Dice',
    'sens': 'Sensitivity',
    'ppv': 'PPV (Precision)',
    'spec': 'Specificity',
    # region counts, not scores: how many regions a method hands back, the
    # selection's own size rather than how well it overlaps the plant. The
    # prune leaf calls it n_selected, run_ana n_pred (score.score_prune /
    # score.score_ana), and both are drawn against a one-region reference
    # (_ONE_REGION) since the caches plant a single effect.
    'n_selected': 'Regions selected',
    'n_pred': 'Regions detected',
}

# the count figures' reference line: the caches plant one effect, so a rule
# that returns one region is size-matched to the plant and anything above it
# is fragmentation (prune's dp oversegments by construction).
_ONE_REGION = 1.0

# the columns above that count regions rather than score them; they are
# tabulated to one decimal, a score to three.
_COUNT_COLS = ('n_selected', 'n_pred')

# effect_llr is the size-normalized effect strength the factory plants -- the
# per-voxel LLR contribution, so a planted region's observed LLR is
# ~ effect_llr * |r| (glow.effect.impose.compute_offset). The axis is labelled
# in that quantity, LLR / |r|, not the bare region LLR.
_X_PARAM_LABELS = {
    'effect_llr': 'LLR / |r|',
    'effect_perc': 'Effect Size (% of Volume)',
    'frac_segment': 'Segmentation Fold (share of images)',
    'num_img': 'Number of Subjects',
    'b': 'Number of Imaging Features',
    'num_vox': 'Number of Voxels',
    'n_perm_fwer': 'FWER Permutations',
    'n_perm_inner': 'Inner Draws per Tree',
}

# the metric grid's non-method hues: a column here draws one series per value
# of a quantity rather than per method, so it is styled with the sequential
# ramp (_seq_style) and titles its legend with this text -- short, since it
# sits inside a panel, unlike the same column's axis label above. Membership is
# the switch: a hue outside it is a method (_qual_style, no legend title).
_HUE_TITLES = {
    'frac_segment': 'Fold share',
    'effect_llr': 'LLR / |r|',
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


def tidy_pred_decomp(raw):
    """Split each trial's false-positive volume by where it came from.

    A region-inference method can be wrong two ways, and the two carry
    different diagnoses. It can declare a region that touches no effect voxel
    at all (spurious: the effect is invented), or declare a region that does
    hit the effect and drag the non-effect voxels around it in with it
    (leaked: the effect is real but its extent is over-stated, which is the
    price of a region rather than a voxel being the unit of inference).

    The two partition fp exactly, because the discovered regions are disjoint:
    GLOW's are an antichain of the tree by construction (prune_by_rule) and
    the baselines' are connected components (Analysis.discover_mask). With n_r
    the voxel count of discovered region r and o_r how many of its voxels land
    in the planted effect (score_effects records both, per region):

        spurious = sum of n_r over regions with o_r == 0
        leaked   = sum of (n_r - o_r) over regions with o_r > 0

    Args:
        raw: the provenance DataFrame tidy_run_ana consumes, with its
            run_ana.out.score.pred.<i>.{num_vox,target} block intact (the wide
            per-region columns, which tidy_run_ana itself drops).

    Returns:
        a tidy_run_ana frame with four columns added: spurious and leaked
        (voxels), frags (regions that hit the effect) and n_spur (regions that
        missed it). All four are NaN when raw carries no per-region block, so
        a caller handed the wrong frame draws nothing rather than reading a
        silent zero as "no spurious volume".
    """
    out = tidy_run_ana(raw)
    if out.empty:
        return out

    def block(field):
        """Return the per-region field as an (n_row, n_reg) float array."""
        pat = re.compile(rf'\.pred\.(\d+)\.{field}$')
        cols = sorted(((int(m.group(1)), c) for c in raw.columns
                       if (m := pat.search(c))))
        if not cols:
            return None
        return raw[[c for _, c in cols]].to_numpy(dtype='float64')

    n_r, o_r = block('num_vox'), block('target')
    if n_r is None or o_r is None:
        for c in ('spurious', 'leaked', 'frags', 'n_spur'):
            out[c] = np.nan
        return out

    # a region index past the trial's region count is absent, not empty
    seen = ~np.isnan(n_r)
    n_r = np.where(seen, n_r, 0.0)
    o_r = np.where(seen & ~np.isnan(o_r), o_r, 0.0)
    hit = seen & (o_r > 0)
    miss = seen & (o_r == 0)

    out['spurious'] = np.where(miss, n_r, 0.0).sum(axis=1)
    out['leaked'] = np.where(hit, n_r - o_r, 0.0).sum(axis=1)
    out['frags'] = hit.sum(axis=1)
    out['n_spur'] = miss.sum(axis=1)

    # the partition is guaranteed by disjointness, so a mismatch means the
    # region records and the confusion counts came from different scorings
    bad = ~np.isclose(out['spurious'] + out['leaked'], out['fp'])
    if bad.any():
        warnings.warn(f'pred decomposition: spurious + leaked != fp on '
                      f'{int(bad.sum())} of {len(out)} rows')
    return out


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
             if a != x and a in df.columns and df[a].dropna().nunique() > 1]
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
# GLOW); the voxel-wise methods are dropped, and the unreported arms before
# that (_ARMS_SKIP / _select_glow_arm). Left-to-right column order. Both
# spellings of the reported arm are listed -- its figure label and its recipe
# label -- so a frame reaches its cell whether or not it has been through
# _select_glow_arm; only one of the two can be present in a given frame.
_CALIB_METHODS = (tuple(_ARM_LABEL.get(lab, lab)
                        for lab in REPORTED_GLOW_LABEL_LIST)
                  + tuple(REPORTED_GLOW_LABEL_LIST))


def _plot_calibration_faceted(label: str, df, out,
                              facet: str = 'source') -> None:
    """Lay out the GLOW-only FWER calibration as a single source x arm row.

    One full row of source x GLOW-arm cells, source-major; every method
    outside _CALIB_METHODS is dropped (the voxel-wise arms; the dp arms go
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

def _draw_metric_band(ax, df, x: str, metric: str, style: dict, *,
                      ci: int = 95, hue: str = 'label') -> None:
    """Draw a mean line plus a central ci% percentile band per method into ax.

    One curve per method (hue) in catalogue order (_ordered_methods, so the
    legend reads in the tables' order), the band spanning the (100-ci)/2 ..
    (100+ci)/2 percentiles of the seed replicates at each x (so the default
    ci=95 shades the 2.5th-97.5th percentile). PPV is undefined for trials
    with no detections (nan; glow.mask.stats_from_counts) and drops out of
    both.

    Args:
        ax: matplotlib Axes to draw into
        df: one source's tidy rows (numeric x / metric, dropna'd on x)
        x (str): the swept x-axis column
        metric (str): the metric column plotted on the y-axis
        style (dict): label -> {'color', 'ls'} (_method_style)
        ci (int): central percentile-interval width for the band
        hue (str): the method-label column
    """
    lo_q, hi_q = (1 - ci / 100) / 2, (1 + ci / 100) / 2
    for label in _ordered_methods(df[hue].dropna().unique().tolist()):
        g = df[df[hue] == label].groupby(x)[metric]
        mean, lo, hi = g.mean(), g.quantile(lo_q), g.quantile(hi_q)
        ax.plot(mean.index, mean.values, lw=2, color=style[label]['color'],
                ls=style[label]['ls'], label=label)
        ax.fill_between(mean.index, lo.values, hi.values,
                        color=style[label]['color'], alpha=0.15)


def _draw_diff(ax, df, x: str, metric: str, *,
               one_label: str = _DIFF_DEFAULT,
               hue: str = 'label', alpha: float = .5) -> list:
    """Draw one_label minus the best non-GLOW method into ax; return CSV rows.

    The head-to-head panel: the per-trial advantage of one_label (the headline
    arm by default) over the best competing method -- the largest metric among
    the non-GLOW labels (VBA / VBA-TFCE / CET) at the same (seed, x). A thin
    line
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


def _has_diff(df, hue: str = 'label') -> bool:
    """Whether a frame holds both a GLOW and a non-GLOW method to diff."""
    labels = {str(v) for v in df[hue].dropna().unique()}
    return (any(v.startswith('GLOW') for v in labels)
            and any(not v.startswith('GLOW') for v in labels))


def plot_source_grid(label: str, df, *, x: str, metrics: list, out,
                     one_label: str = _DIFF_DEFAULT, ci: int = 95,
                     thresh_metric: str = 'dice', level: float = 0.5) -> None:
    """Plot the stacked per-source detection figure: a block per source.

    One SubFigure per source (its banner the source name), stacked HCP over
    WGN; within each a len(metrics)-wide grid whose top row is the mean score +
    central ci% percentile band per method (_draw_metric_band) and whose second
    row is one_label (the headline GLOW arm) minus the best non-GLOW
    alternative (_draw_diff),
    with the metrics (dice / sens / ppv) across the columns. That second row is
    drawn only for a source that has a non-GLOW method to diff against
    (_has_diff): a cache of GLOW arms alone is one row per source. A dashed
    line marks the threshold level on the thresh_metric (Dice) panel, where the
    discovery thresholds (write_threshold_table, plotted once per cache) are
    read.

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

    style = _method_style(df['label'].dropna().unique().tolist())
    log_x = pd.notnull(df[x].min()) and df[x].min() > 0
    ncols = len(metrics)

    # a source's block is the band row plus a diff row, or the band row alone
    nrow_src = [1 + _has_diff(df[df['source'] == src]) for src in sources]
    fig = plt.figure(figsize=(4.2 * ncols, 2.3 * sum(nrow_src)),
                     layout='constrained')
    subfigs = np.atleast_1d(fig.subfigures(len(sources), 1,
                                           height_ratios=nrow_src))

    diff_rows = []
    for si, (subfig, src, nrows) in enumerate(zip(subfigs, sources,
                                                 nrow_src)):
        subfig.suptitle(src, fontsize=14, fontweight='bold')
        axes = subfig.subplots(nrows, ncols, sharex=True, squeeze=False)
        dsrc = df[df['source'] == src]
        for j, metric in enumerate(metrics):
            _draw_metric_band(axes[0, j], dsrc, x, metric, style, ci=ci)
            axes[0, j].set_title(_METRIC_TITLES.get(metric, metric))
            axes[0, j].set_ylim(0, 1)
            axes[0, j].grid(True, alpha=0.3)
            # the discovery-threshold level, read as a table below
            if metric == thresh_metric:
                axes[0, j].axhline(level, ls='--', lw=0.8, color='grey',
                                   alpha=0.7)

            if nrows == 2:
                rows = _draw_diff(axes[1, j], dsrc, x, metric,
                                  one_label=one_label)
                for r in rows:
                    r['source'] = src
                diff_rows += rows
            axes[-1, j].set_xlabel(_X_PARAM_LABELS.get(x, x))
            if log_x:
                for ax in axes[:, j]:
                    ax.set_xscale('log')

        axes[0, 0].set_ylabel(f'score (mean, {ci}% band)')
        if nrows == 2:
            axes[1, 0].set_ylabel(f'{one_label} − best')
        # one legend for the figure, on the first block's top-left panel
        if si == 0:
            axes[0, 0].legend(frameon=False, fontsize=8, loc='upper left')

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


def _method_rank() -> dict:
    """Map every method label to its position in the config catalogue.

    Methods follow the config catalogue order (GLOW first, then the voxel-wise
    methods; see config.ana_kwargs_dict), under both their recipe label and
    their figure label, so a frame ranks whether or not it has been through
    _select_glow_arm's rename (_ARM_LABEL).
    """
    rank = {}
    for i, m in enumerate(ana_kwargs_dict):
        rank[m] = i
        rank[_ARM_LABEL.get(m, m)] = i
    return rank


def _ordered_methods(label_list) -> list:
    """Sort method labels into catalogue order, a label outside it last."""
    rank = _method_rank()
    return sorted(label_list, key=lambda m: (rank.get(m, len(rank)), str(m)))


def _order_threshold_rows(wide):
    """Sort a threshold table by source order, then catalogue method order.

    Sources follow _SOURCE_ORDER (HCP over WGN), methods _method_rank.
    """
    m_rank = _method_rank()
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
    # catalogue label (stale records from a since-changed knob) drop out; an
    # all-unlabelled cache then yields an empty table rather than a groupby
    # that leaves wide without a method column
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


def _threshold_lines(df, wide, status, *, x: str, metric: str,
                     level: float) -> list:
    """Render a threshold table as aligned plain-text lines.

    One caption line, then a block per source: a header of the table's value
    columns and a row per method. A censored cell reads as < the weakest
    tested x (already above level there) or > the strongest (never reaches
    it), off the status map.

    Args:
        df: the frame the table was built from; its x range names the
            censored ends
        wide: the threshold table (threshold_table)
        status (dict): (source, method, column) -> crossing status
        x (str): the swept-axis column
        metric (str): the metric whose level crossing the table holds
        level (float): the crossing level

    Returns:
        list[str]: the lines, indented two spaces below their block header.
    """
    val_cols = [c for c in wide.columns if c not in ('source', 'method')]
    xlo, xhi = float(df[x].min()), float(df[x].max())
    width = max([len(str(c)) for c in val_cols] + [10])
    name_w = max([len(str(m)) for m in wide['method']] + [11])

    def render(src, method, col, value):
        """Format one threshold cell, censored ends read off the status map."""
        if pd.notnull(value):
            return f'{value:>{width}.4f}'
        st = status.get((src, method, col))
        if st == 'below':
            return f'{f"<{xlo:g}":>{width}}'
        if st == 'above':
            return f'{f">{xhi:g}":>{width}}'
        return f'{"—":>{width}}'

    metric_title = _METRIC_TITLES.get(metric, metric)
    lines = [f'{_X_PARAM_LABELS.get(x, x)} at {metric_title} >= {level:g} '
             f'(mean across trials):']
    for src in wide['source'].drop_duplicates():
        sub = wide[wide['source'] == src]
        header = ' '.join(f'{c:>{width}}' for c in val_cols)
        lines += [f'  source={src}',
                  f'    {"method":<{name_w}} {header}']
        for _, r in sub.iterrows():
            cells = ' '.join(render(src, r['method'], c, r[c])
                             for c in val_cols)
            lines.append(f'    {r["method"]:<{name_w}} {cells}')
    return lines


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
    for line in _threshold_lines(df, wide, status, x=x, metric=metric,
                                 level=level):
        print(f'  {line}')


# ---------------------------------------------------------------------------
# The figure's numbers as plain text ({label}_tables.txt)
# ---------------------------------------------------------------------------

def _matrix_lines(piv, *, prec: int = 3) -> list:
    """Render a method x swept-value matrix as aligned plain-text lines.

    Args:
        piv: DataFrame indexed by method label, one column per swept value
        prec (int): decimals per cell

    Returns:
        list[str]: a header of the swept values, then a row per method. An
            absent cell is blank (PPV is undefined for a trial that detected
            nothing; glow.mask.stats_from_counts).
    """
    text = piv.map(lambda v: '' if pd.isnull(v) else f'{v:.{prec}f}')
    # wide enough for the widest cell, so a region count in the thousands
    # keeps the columns aligned with a 0..1 score's
    width = max([len(c) for c in text.to_numpy().ravel()]
                + [len(f'{v:.3g}') for v in piv.columns] + [prec + 5])
    name_w = max([len(str(m)) for m in piv.index] + [6])
    header = ' '.join(f'{v:>{width}.3g}' for v in piv.columns)
    lines = [f'{"method":<{name_w}} {header}']
    for name, row in text.iterrows():
        lines.append(f'{name:<{name_w}} '
                     + ' '.join(f'{c:>{width}}' for c in row))
    return lines


def _head_to_head_lines(df, *, x: str, metric: str, one_label: str,
                        hue: str = 'label') -> list:
    """Render each method's mean metric and its win rate against one_label.

    A cell is one (seed, x) trial pair, so the win rate is read on paired
    trials: the share of the cells a method scores strictly above one_label
    on, over the cells where both ran. The mean is over the same cells the
    method has, so it weights every swept value equally. one_label absent
    yields no lines.

    Args:
        df: one source's tidy rows (numeric x / metric)
        x (str): the swept-axis column
        metric (str): the metric compared
        one_label (str): the method every row is compared against
        hue (str): the method-label column

    Returns:
        list[str]: a caption line, a header, then a row per method.
    """
    agg = df.groupby([hue, 'seed', x], as_index=False)[metric].mean()
    piv = agg.pivot_table(index=['seed', x], columns=hue, values=metric)
    if one_label not in piv.columns:
        return []
    methods = _ordered_methods(piv.columns.tolist())
    name_w = max([len(str(m)) for m in methods] + [6])
    lines = [f'mean {_METRIC_TITLES.get(metric, metric)} and win rate vs '
             f'{one_label} (paired on seed x {x}):',
             f'  {"method":<{name_w}} {"mean":>8} {"win":>8} {"n_cell":>8}']
    for lab in methods:
        # keyed rather than sliced by label, so the reference row's own pair
        # is two columns and not one name twice
        pair = pd.concat([piv[lab], piv[one_label]], axis=1,
                         keys=['a', 'b']).dropna()
        beat = (pair['a'] > pair['b']).mean() if len(pair) else np.nan
        win = (f'{"—":>8}' if lab == one_label else f'{beat:>8.3f}')
        lines.append(f'  {lab:<{name_w}} {piv[lab].mean():>8.3f} '
                     f'{win} {len(pair):>8d}')
    return lines


def write_table_txt(label: str, df, *, x: str, out, metrics: list,
                    one_label: str = 'GLOW', level: float = 0.5) -> None:
    """Write one figure's numbers as a plain-text table file.

    {label}_tables.txt, the readable companion to {label}.pdf: a header naming
    the swept axis and the panel, the discovery thresholds
    (_threshold_lines), then a block per source holding a method x
    swept-value matrix of the seed-mean metric (_matrix_lines, one per metric)
    and the head-to-head against one_label (_head_to_head_lines). Written per
    drawn figure, so a cache split on a secondary axis gets one file per value
    of it and every table has exactly one axis moving.

    Args:
        label (str): the drawn figure's label; the output filename stem
        df: the tidy rows behind that figure
        x (str): the swept-axis column
        out (pathlib.Path): directory the file is written into
        metrics (list): the metric columns tabulated, one block each
        one_label (str): the method the head-to-head compares against
        level (float): the discovery-threshold level
    """
    df = df.copy()
    for c in [x, *metrics]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=[x, 'label'])
    if df.empty:
        return

    have = set(df['source'].dropna().unique())
    sources = [s for s in _SOURCE_ORDER if s in have]
    sources += [s for s in sorted(have) if s not in sources]
    x_vals = sorted(df[x].unique())

    lines = [label, '=' * len(label), '',
             f'x axis: {_X_PARAM_LABELS.get(x, x)} ({x})',
             f'panel: {len(df)} leaves, {df["label"].nunique()} methods, '
             f'{df["seed"].nunique()} seeds']
    # the axes this figure holds fixed, so a file read on its own says which
    # slice of the cache it is
    for c in _SECONDARY_AXES:
        if (c in df.columns and df[c].notna().all()
                and df[c].nunique() == 1):
            lines.append(f'held: {c} = {df[c].iloc[0]:g}')

    wide, status = threshold_table(df, x=x, metric=metrics[0], level=level)
    if not wide.empty:
        lines += ['', *_threshold_lines(df, wide, status, x=x,
                                        metric=metrics[0], level=level)]

    for src in sources:
        dsrc = df[df['source'] == src]
        n_cell = dsrc.pivot_table(index='label', columns=x,
                                  values=metrics[0], aggfunc='size')
        lines += ['', f'--- {src} ---',
                  f'seeds per cell: {int(n_cell.min().min())} to '
                  f'{int(n_cell.max().max())}']
        for metric in metrics:
            piv = dsrc.pivot_table(index='label', columns=x, values=metric,
                                   aggfunc='mean')
            piv = piv.reindex(index=_ordered_methods(piv.index.tolist()),
                              columns=x_vals)
            prec = 1 if metric in _COUNT_COLS else 3
            lines += ['', f'{_METRIC_TITLES.get(metric, metric)} '
                          f'(mean over seeds)',
                      *_matrix_lines(piv, prec=prec)]
        h2h = _head_to_head_lines(dsrc, x=x, metric=metrics[0],
                                  one_label=one_label)
        if h2h:
            lines += ['', *h2h]

    path = out / f'{label}_tables.txt'
    path.write_text('\n'.join(lines) + '\n')
    print(f'saved: {path}')


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

    Both split the raw and z-scored arms, which never pool because z-scoring
    interacts with the stat, and both read the same balanced panel, so every
    stat is scored on the same trials (stat_tables warns otherwise).

    stat_dice.tex is one row per method, its five stat means banded under raw
    and again under z-scored, bolding the method's best of the ten (or its
    whole raw block, for a _STAT_BOLD_RAW_ROW method).
    stat_matters.tex is one row per arm: the all-tie / decisive fractions and
    the mean / median per-trial Dice range. Two narrow tables rather than one
    wide, to fit a paper column. Each file is a bare booktabs tabular; the
    paper owns the table environment and prose.

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

        The bold marks the max, not a margin, except over the raw block of a
        _STAT_BOLD_RAW_ROW method, where it marks the tie.
        """
        cells = [method]
        for zt in _ZT_ORDER:
            row = t2.loc[(method, zt)]
            for s in _STAT_ORDER:
                v = f'{row[s]:.3f}'
                bold = (best_cell[method] == (zt, s)
                        or (zt == 'raw' and method in _STAT_BOLD_RAW_ROW))
                cells.append(f'\\textbf{{{v}}}' if bold else v)
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
        source (WGN / HCP), seed, b, effect_llr, the four confusion counts,
        n_selected (the leaf's own region count, nan for a leaf that returns
        none) and the derived dice/sens/ppv/spec (empty in, empty out).
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
    # the selection's size, which only the leaves that select one report
    # (run_prune / run_inner_perm; run_segment scores a whole Ward mode)
    out['n_selected'] = pd.to_numeric(col(f'{leaf}.out.score.n_selected'),
                                      errors='coerce')
    return add_metric_cols(out)


# the fold share a whole-cohort leaf stands at: it segments every image, so it
# is the frac_segment sweep's ceiling rather than a point beside it (the leaf
# grid stops at 0.9, since a split always holds a test fold back -- see
# config.SEGMENT_FRAC_GRID). Only the frames that draw both halves at once
# (tidy_segment(perc=None)) put it on the axis.
WHOLE_COHORT_FRAC = 1.0


def tidy_segment(raw, *, perc: bool = False):
    """Normalise the segment cache to a tidy per-(trial, Ward mode) frame.

    The method label is the recorded Ward mode (config records str(mode), so
    the cell is already the mode string Naive / GLM Error / Focus).

    The segment caches share the run_segment leaf and their moderate-effect
    cells, so the forward record walk reaches every one of their leaves from
    any one's cells (results.config_leaf_keys). frac_segment is what tells them
    apart -- a whole-cohort leaf records none, a fold leaf records the share it
    segmented on -- so perc selects the wanted half and the column rides
    through as the fold sweep's x-axis (or, where both halves are drawn, its
    hue).

    Args:
        raw: the segment cache's provenance frame (one row per run_segment leaf).
        perc (bool | None): True keeps the fold leaves (frac_segment recorded),
            False (default) the whole-cohort ones, None both -- the latter with
            the whole-cohort leaves reading WHOLE_COHORT_FRAC, so the ceiling
            sits on the fold axis as its last value.

    Returns:
        a tidy_flat_cache frame (label = Ward mode) plus a frac_segment column;
        empty in, empty out.
    """
    if raw.empty:
        return raw
    out = _tidy_flat_cache(raw, 'run_segment', 'run_segment.in.cluster_mode')
    col = 'run_segment.in.frac_segment'
    frac = (raw[col] if col in raw.columns
            else pd.Series(np.nan, index=raw.index))
    out['frac_segment'] = pd.to_numeric(frac.reindex(out.index),
                                        errors='coerce')
    if perc is None:
        return out.fillna({'frac_segment': WHOLE_COHORT_FRAC})
    keep = out['frac_segment'].notna() if perc else out['frac_segment'].isna()
    return out[keep]


def tidy_prune(raw):
    """Normalise the prune cache to a tidy per-(trial, rule) frame.

    The method label is GLOW-<rule> off the recorded rule (single_max /
    greedy / dp / oracle, the last a headroom line rather than a method --
    see prune.prune_oracle); the recorded Ward mode (run_prune.in.cluster_mode)
    rides along as the cluster_mode column, so plot_prune can split it into
    one figure per mode. A legacy record predating the mode axis carried the
    Focus default, so a missing mode reads back as Focus.

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
                        dodge: float = 0.03, y_floor: float = None) -> None:
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
        style (dict): hue value -> {'color', 'ls'} (from _qual_style /
            _seq_style).
        hue (str): the column the series are drawn per (the method label, or a
            quantity like the fold share).
        z_mult (float): SEM multiplier for the error bar (1.96 ~ 95% CI).
        dodge (float): fractional multiplicative x-dodge between methods.
        y_floor (float | None): clip the lower bar here (_log_y_floor). A
            count's symmetric CI reaches below zero, which a log axis cannot
            draw -- and left unclipped it stretches the panel over decades
            holding nothing.
    """
    labels = sorted(df[hue].dropna().unique().tolist())
    n = len(labels)
    for i, lab in enumerate(labels):
        g = df[df[hue] == lab].groupby(x)[metric]
        mean, sem = g.mean(), g.std() / np.sqrt(g.count())
        err = z_mult * sem.values
        if y_floor is None:
            yerr = err
        else:
            # a mean of zero sits below the floor and cannot be drawn on a log
            # axis at all, so its clipped bar is zero rather than negative
            lo = np.maximum(mean.values - np.maximum(mean.values - err,
                                                     y_floor), 0)
            yerr = np.vstack([lo, err])
        # centre the per-method dodge on the group so it sits over the true x
        factor = 1 + dodge * (i - (n - 1) / 2)
        st = style[lab]
        ax.errorbar(mean.index.values * factor, mean.values,
                    yerr=yerr, marker='o', ms=4, lw=1.5,
                    ls=st['ls'], color=st['color'], capsize=2,
                    label=_hue_text(lab))


def _log_y_floor(df, x: str, col: str, hue: str = 'label') -> float:
    """Pick a log-y floor for a count panel: half its smallest positive mean.

    Args:
        df: the frame the panel draws (numeric x / col).
        x (str): the swept x-axis column.
        col (str): the counted column.
        hue (str): the column one series is drawn per.

    Returns:
        float: the floor, or a default when no positive mean is present.
    """
    mean = df.groupby([hue, x])[col].mean()
    pos = mean[mean > 0]
    return float(pos.min()) / 2 if len(pos) else 0.1


def plot_metric_grid(label: str, df, out, *, x: str = 'effect_llr',
                     metrics=('dice', 'sens', 'ppv'),
                     log_x: bool = None, hue: str = 'label',
                     dodge: float = 0.03, ylim=(0, 1), log_y: bool = False,
                     hline: float = None, style: dict = None) -> None:
    """Plot a source x metric grid of per-series mean +/- 95% CI vs the x.

    A 2 x len(metrics) grid: one row per data source (HCP over WGN), one column
    per metric (Dice / Sensitivity / PPV). Each panel draws the per-series
    seed-mean with a 95% CI-of-the-mean error bar, x-dodged so the series stay
    legible (_draw_metric_errbar), against effect_llr on a log x-axis. A series
    is a method by default (the Ward mode for segment, the prune rule for
    prune), styled in the Okabe-Ito qualitative palette; a hue that is a
    quantity instead (segment_perc_llr's fold share) takes the sequential ramp
    _seq_style, and names itself in the legend title (_HUE_TITLES).
    Writes {label}.pdf.

    The y defaults suit a score; a region count is the same grid read on a log
    y with no clamp (ylim=None, log_y=True). A count column mixed in beside
    the scores (_COUNT_COLS) takes that axis for its own panel alone, plus the
    one-region reference the count figures draw (_ONE_REGION) -- see
    plot_prune.

    Args:
        label (str): cache name; the output filename stem.
        df: a tidy frame (needs source / hue / seed / x / the metric columns).
        out (pathlib.Path): directory the figure is written into.
        x (str): the swept x-axis column (effect_llr).
        metrics (iterable): the metric columns, one panel column each.
        log_x (bool | None): log x-axis; None (default) takes one wherever
            every x is positive, which suits the decade-wide llr sweeps and
            not a linear axis like segment_perc's fold share.
        hue (str): the column one series is drawn per; 'label' (default) is the
            method.
        dodge (float): fractional multiplicative x-dodge between series; the
            default suits the few-series figures, and many series want less
            (the spread grows with the count).
        ylim (tuple | None): y limits; the default is a score's 0..1, None
            leaves the axis to the data (a count). A count panel ignores it.
        log_y (bool): log y-axis, for a quantity spanning decades; a count
            panel takes one either way.
        hline (float | None): a horizontal reference line, drawn dashed; a
            count panel draws _ONE_REGION instead.
        style (dict | None): hue value -> {'color', 'ls'}; None styles the
            values here, so a caller passes one only to hold a figure's
            colours to another's (_method_style).
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

    values = df[hue].dropna().unique().tolist()
    if style is None:
        style = (_seq_style(values) if hue in _HUE_TITLES
                 else _qual_style(values))
    if log_x is None:
        log_x = pd.notnull(df[x].min()) and df[x].min() > 0
    ncols = len(metrics)
    floors = {m: _log_y_floor(df, x, m, hue=hue) for m in metrics
              if log_y or m in _COUNT_COLS}

    fig, axes = plt.subplots(len(sources), ncols, sharex=True,
                             figsize=(4.2 * ncols, 3.8 * len(sources)),
                             squeeze=False)
    for i, src in enumerate(sources):
        dsrc = df[df['source'] == src]
        for j, metric in enumerate(metrics):
            ax = axes[i, j]

            count = metric in _COUNT_COLS
            _draw_metric_errbar(ax, dsrc, x, metric, style, hue=hue,
                                dodge=dodge, y_floor=floors.get(metric))
            if ylim is not None and not count:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)
            if log_x:
                ax.set_xscale('log')
            if log_y or count:
                ax.set_yscale('log')
                ax.set_ylim(bottom=floors[metric])
            ref = _ONE_REGION if count else hline
            if ref is not None:
                ax.axhline(ref, ls='--', lw=0.8, color='grey', alpha=0.7)
            if i == 0:
                ax.set_title(_METRIC_TITLES.get(metric, metric))
            if i == len(sources) - 1:
                ax.set_xlabel(_X_PARAM_LABELS.get(x, x))
        axes[i, 0].set_ylabel(f'{src}\nmean (95% CI)')
    # a few methods leave room inside the first panel; a quantity's ramp is one
    # entry per swept value (ten fold shares), which no panel has room for --
    # that legend goes beside the grid, where it cannot cover a curve
    if hue in _HUE_TITLES:
        axes[0, -1].legend(frameon=False, fontsize=8, loc='upper left',
                           bbox_to_anchor=(1.02, 1.0),
                           title=_HUE_TITLES[hue], title_fontsize=8)
    else:
        axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out / f'{label}.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'saved: {path}')


def _mode_slug(mode: str) -> str:
    """Filename-safe token for a Ward mode ('GLM Error' -> 'GLM_Error')."""
    return str(mode).replace(' ', '_')


def _segment_perc(kwargs_effect_list, kwargs_fnc_list):
    """Read which half of the shared run_segment leaf a segment cache plots.

    The segment family's caches share one leaf and one label vocabulary, so the
    figure each asks for is read off its own grids rather than its name:
    frac_segment in the leaf grid makes it a fold sweep, and a fold sweep whose
    effect grid also varies the strength wants both halves -- the fold shares
    and the whole-cohort ceiling they are read against (plot_segment_llr).

    Args:
        kwargs_effect_list (list): the cache's effect cells (CONFIG[name][1]).
        kwargs_fnc_list (list): the cache's leaf-kwargs cells
            (CONFIG[name][2]).

    Returns:
        bool | None: the perc argument tidy_segment takes -- False the
            whole-cohort leaves, True the fold ones, None both.
    """
    fold = any('frac_segment' in kwargs for kwargs in kwargs_fnc_list)
    llrs = {kwargs['effect_llr'] for kwargs in kwargs_effect_list if kwargs}
    return None if fold and len(llrs) > 1 else fold


# the fold sweep's x-dodge: ten curves at the segment_perc_llr default of 0.03
# would spread +/- 13% about each x, comparable to the llr grid's own 26% step,
# so the series drift into each other's columns. A third of it keeps the error
# bars apart without moving a point off its grid value.
_FOLD_DODGE = 0.01


def plot_segment_llr(label: str, df, out) -> None:
    """Plot one llr metric grid per Ward mode, a curve per segmentation fold.

    The fold sweep read along the effect axis rather than across it: this
    fixes the Ward mode (one source x metric grid each) and draws the whole
    llr sweep once per fold share, so reading down a panel's curves gives
    what a smaller segmentation fold costs at that effect strength.

    Takes a both-halves frame (tidy_segment(perc=None)): the fold shares plus
    the whole-cohort ceiling at WHOLE_COHORT_FRAC, the segmentation every other
    GLOW figure clusters on and so the line the folds are read against.

    Args:
        label (str): cache name; each figure's stem is {label}_{mode}.
        df: a tidy_segment frame (label = Ward mode, plus frac_segment).
        out (pathlib.Path): directory the figures are written into.
    """
    for mode, df_mode in df.groupby('label'):
        plot_metric_grid(f'{label}_{_mode_slug(mode)}', df_mode, out,
                         hue='frac_segment', dodge=_FOLD_DODGE)


# the comparison pages' source order, left to right. Not _SOURCE_ORDER: the
# stacked grids put HCP in the top row, where a reader meets the real cohort
# first, but a page whose two panels sit side by side reads the other way --
# WGN on the left as the clean case, HCP on the right as the anatomy it has to
# survive.
_COMPARE_SOURCE_ORDER = ('WGN', 'HCP')


def _compare_page_fig(df, metrics, *, x: str = 'effect_llr',
                      dodge: float = 0.03):
    """Build one comparison page: a metric x source grid, WGN left, HCP right.

    The transpose of plot_metric_grid's layout, for a page that holds one thing
    fixed and compares the methods under it: a source is a column here rather
    than a row (_COMPARE_SOURCE_ORDER), so the two cohorts sit side by side and
    the metric names move from the panel titles to the y-labels. Rows are the
    metrics, so the default single metric is one row of two panels. The panels
    share both axes -- the point is reading one curve set against the other, so
    they must not be on different scales.

    Args:
        df: one page's tidy rows (source / label / seed / x / the metrics).
        metrics (iterable): the metric columns, one panel row each.
        x (str): the swept x-axis column.
        dodge (float): fractional multiplicative x-dodge between methods.

    Returns:
        fig | None: the page, or None when no row carries a source.
    """
    metrics = list(metrics)
    df = df.copy()
    for c in [x, *metrics]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=[x])

    have = set(df['source'].dropna().unique())
    sources = [s for s in _COMPARE_SOURCE_ORDER if s in have]
    sources += [s for s in sorted(have) if s not in sources]
    if not sources:
        return None

    style = _qual_style(df['label'].dropna().unique().tolist())
    log_x = pd.notnull(df[x].min()) and df[x].min() > 0

    fig, axes = plt.subplots(len(metrics), len(sources), sharex=True,
                             sharey=True, squeeze=False,
                             figsize=(4.2 * len(sources), 3.8 * len(metrics)))
    for i, metric in enumerate(metrics):
        for j, src in enumerate(sources):
            ax = axes[i, j]
            _draw_metric_errbar(ax, df[df['source'] == src], x, metric, style,
                                dodge=dodge)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            if log_x:
                ax.set_xscale('log')
            if i == 0:
                ax.set_title(src)
            if i == len(metrics) - 1:
                ax.set_xlabel(_X_PARAM_LABELS.get(x, x))
        axes[i, 0].set_ylabel(f'{_METRIC_TITLES.get(metric, metric)}\n'
                              'mean (95% CI)')
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_segment_compare(label: str, df, out,
                         metrics=('dice',)) -> list:
    """Write a page per fold share comparing the Ward modes, largest first.

    The third cut of the same frame, answering which clustering a given fold
    should buy: a page holds the fold share fixed and draws the three Ward
    modes against the llr sweep, one panel per source side by side
    (_compare_page_fig). Flipping the pages from 100% of the images down is
    then the same comparison losing data.

    Pages descend so the whole-cohort segmentation, the one every other GLOW
    figure clusters on, is the page a reader opens on.

    Args:
        label (str): cache name; the multipage file is {label}_compare.pdf.
        df: a tidy_segment frame (label = Ward mode, plus frac_segment).
        out (pathlib.Path): directory the file is written into.
        metrics (iterable): the metric columns to draw per page, one panel row
            each (_compare_page_fig); Dice alone by default, which is what the
            comparison is about (the per-mode figures carry sens / ppv beside
            it).

    Returns:
        list[float]: the fold shares drawn, in page order (empty if none).
    """
    from matplotlib.backends.backend_pdf import PdfPages

    fracs = sorted(df['frac_segment'].dropna().unique(), reverse=True)
    if not fracs:
        print(f'  (no fold shares for {label} — skipping the comparison)')
        return []
    path = out / f'{label}_compare.pdf'
    drawn = []
    with PdfPages(path) as pdf:
        for frac in fracs:
            fig = _compare_page_fig(df[df['frac_segment'] == frac],
                                    metrics)
            if fig is None:
                continue
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            drawn.append(float(frac))
    print(f'saved: {path} ({len(drawn)} page(s))')
    return drawn


# prune rules kept in the cache and records but dropped from the paper figures:
# single_max (the single max-LLR region) is a diagnostic baseline, not a paper
# method, so plot_prune skips it (see PRUNE_RULES in config).
_PRUNE_LABELS_SKIP = ('GLOW-single_max',)

# the rule the shipped recipe prunes with, under the prune cache's own
# labelling: read off the reported arm rather than spelled out, so the tables
# compare against whatever config ships.
_REPORTED_PRUNE_LABEL = (
    f'GLOW-{ana_kwargs_dict[REPORTED_GLOW_LABEL].prune_rule}')

# the prune grid's columns: the three scores plus what the rule selected, the
# count being half of what a rule is judged on (a Dice bought by handing back
# the support in pieces is not the same result as one region)
_PRUNE_METRICS = ('dice', 'sens', 'ppv', 'n_selected')


def plot_prune_regions(label: str, df, out, *, x: str = 'effect_llr',
                       count: str = 'n_selected') -> None:
    """Plot how many regions each pruning rule hands back, per Ward mode.

    A source x Ward-mode grid (HCP over WGN, Focus beside GLM Error): the
    per-rule seed-mean count with a 95% CI bar against effect_llr, on a log y
    since the rules span one region to thousands. Both modes sit on one page
    here rather than one figure each -- the count is the comparison, and the
    two clusterings are two columns of it.

    The single planted effect is the dashed reference (_ONE_REGION): a rule
    above it returns the support in pieces, which is how a rule can gain Dice
    without gaining anything a reader would report. The oracle line is the
    count a max-Dice antichain needs on the same fit. single_max is dropped,
    its count being one by construction (_PRUNE_LABELS_SKIP).

    Writes {label}_regions.pdf.

    Args:
        label (str): cache name; the figure's stem is {label}_regions.
        df: a tidy_prune frame (needs cluster_mode and the count column).
        out (pathlib.Path): directory the figure is written into.
        x (str): the swept x-axis column.
        count (str): the region-count column (run_prune's n_selected).
    """
    df = df[~df['label'].isin(_PRUNE_LABELS_SKIP)].copy()
    for c in (x, count):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=[x, count])

    have = set(df['source'].dropna().unique())
    sources = [s for s in _SOURCE_ORDER if s in have]
    sources += [s for s in sorted(have) if s not in sources]
    modes = sorted(df['cluster_mode'].dropna().unique().tolist())
    if not sources or not modes:
        print(f'  (no region counts for {label} — skipping)')
        return

    # styled over every rule in the cache, so a rule keeps the colour it has
    # in that mode's metric grid (plot_metric_grid styles the same label set)
    style = _qual_style(df['label'].dropna().unique().tolist())
    log_x = pd.notnull(df[x].min()) and df[x].min() > 0
    floor = _log_y_floor(df, x, count)

    fig, axes = plt.subplots(len(sources), len(modes), sharex=True,
                             sharey=True, squeeze=False,
                             figsize=(4.4 * len(modes), 3.8 * len(sources)))
    for i, src in enumerate(sources):
        for j, mode in enumerate(modes):
            ax = axes[i, j]
            _draw_metric_errbar(
                ax, df[(df['source'] == src) & (df['cluster_mode'] == mode)],
                x, count, style, y_floor=floor)
            ax.set_yscale('log')
            ax.set_ylim(bottom=floor)
            ax.axhline(_ONE_REGION, ls='--', lw=0.8, color='grey', alpha=0.7)
            ax.grid(True, alpha=0.3, which='both')
            if log_x:
                ax.set_xscale('log')
            if i == 0:
                ax.set_title(str(mode))
            if i == len(sources) - 1:
                ax.set_xlabel(_X_PARAM_LABELS.get(x, x))
        axes[i, 0].set_ylabel(f'{src}\n'
                              f'{_METRIC_TITLES.get(count, count)} '
                              '(mean, 95% CI)')
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out / f'{label}_regions.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'saved: {path}')


def plot_prune(label: str, df, out) -> None:
    """Plot one metric grid per Ward clustering mode for a prune cache.

    A prune cache crosses the pruning rules with both Ward modes (Focus / GLM
    Error), so a single grid would overlay two clusterings. This draws one
    source x metric grid per mode (plot_metric_grid), writing {label}_{mode}.pdf
    so the clusterings are compared side by side rather than on one axis. The
    diagnostic single_max rule is dropped (_PRUNE_LABELS_SKIP).

    The grid's fourth column is how many regions the rule hands back
    (_PRUNE_METRICS): what separates two rules that score the same Dice, read
    beside the score it bought. Both modes' counts also go out on one page of
    their own (plot_prune_regions), where the two clusterings share a y-axis.
    Each mode's numbers, counts included, go out as text beside its grid
    (write_table_txt).

    A cache that also varies b splits again on it (_split_by_secondary), one
    figure per (mode, b): the grid's facets are
    already spent on source x metric, and pooling the b's would average two
    power curves into one line. The b=1-only prune cache yields the one figure
    per mode it always did.

    Args:
        label (str): cache name; each figure's stem is {label}_{mode}, plus
            _b{b} where the cache sweeps b.
        df: a tidy_prune frame (needs the cluster_mode column).
        out (pathlib.Path): directory the figures are written into.
    """
    df = df[~df['label'].isin(_PRUNE_LABELS_SKIP)]
    for mode, df_mode in df.groupby('cluster_mode'):
        stem = f'{label}_{_mode_slug(mode)}'
        for sub_label, sub in _split_by_secondary(stem, df_mode, 'effect_llr'):
            plot_metric_grid(sub_label, sub, out, metrics=_PRUNE_METRICS)
            write_table_txt(sub_label, sub, x='effect_llr', out=out,
                            metrics=list(_PRUNE_METRICS),
                            one_label=_REPORTED_PRUNE_LABEL)
    for sub_label, sub in _split_by_secondary(label, df, 'effect_llr'):
        plot_prune_regions(sub_label, sub, out)


# ---------------------------------------------------------------------------
# Inner-draw sweep (what n_perm_inner buys)
# ---------------------------------------------------------------------------

def tidy_inner_perm(raw):
    """Normalise the inner-draw sweep to a tidy per-(trial, count) frame.

    The cache holds one GLOW variant at several inner-draw counts, so the
    count is the x-axis and the effect strength the hue -- unlike the
    method-comparison caches, whose hue is the recipe. Beyond the selection's
    counts it carries the max-z block (run.run_inner_perm): which region the
    statistic came from, its size, its z, and its own Dice against the plant.

    Args:
        raw: the cache's provenance frame (one row per run_inner_perm leaf).

    Returns:
        a _tidy_flat_cache frame plus n_perm_inner, n_sig, min_pval,
        max_z_reg_idx, max_z_num_vox, max_z_z and max_z_dice; empty in,
        empty out.
    """
    if raw.empty:
        return raw
    leaf = 'run_inner_perm'
    out = _tidy_flat_cache(raw, leaf, f'{leaf}.in.cluster_mode',
                           label_fn=lambda mode: f'GLOW-{mode}')

    def col(name):
        """raw[name] reindexed onto out, or all-NaN when absent."""
        if name not in raw.columns:
            return pd.Series(np.nan, index=out.index)
        return pd.to_numeric(raw[name].reindex(out.index), errors='coerce')

    out['n_perm_inner'] = col(f'{leaf}.in.n_perm_inner')
    out['n_sig'] = col(f'{leaf}.out.score.n_sig')
    out['min_pval'] = col(f'{leaf}.out.score.min_pval')
    for name in ('reg_idx', 'num_vox', 'z'):
        out[f'max_z_{name}'] = col(f'{leaf}.out.score.max_z.{name}')
    counts = {c: col(f'{leaf}.out.score.max_z.{c}')
              for c in ('tp', 'fp', 'tn', 'fn')}
    out['max_z_dice'] = glow.mask.stats_from_counts(**counts)['dice']
    return out


def _max_z_agreement(df):
    """Flag each row whose max-z region is the deepest count's.

    The stability read: the observed tree is the same at every count (the
    inner draws standardize it, they do not build it), so a region index is
    comparable across a trial's counts and the deepest count is the reference
    every shallower one is scored against. Two counts that both found no
    region agree.

    Args:
        df: a tidy_inner_perm frame.

    Returns:
        df with an agree column (bool).
    """
    key = ['source', 'seed', 'effect_llr']
    deep = df.loc[df.groupby(key, dropna=False)['n_perm_inner'].idxmax(),
                  key + ['max_z_reg_idx']]
    out = df.merge(deep.rename(columns={'max_z_reg_idx': '_reg_deep'}),
                   on=key, how='left')
    same = out['max_z_reg_idx'] == out['_reg_deep']
    both_none = out['max_z_reg_idx'].isna() & out['_reg_deep'].isna()
    return out.assign(agree=same | both_none).drop(columns='_reg_deep')


# the inner-draw figure's panels: (metric column, panel title). Two are the
# stability read -- whether the argmax has settled, and how well the region it
# settles on matches the plant -- and the third is why: z is capped at
# n_perm_inner / sqrt(n_perm_inner + 1) by the observed draw's own presence in
# its null, so a mean sitting on that ceiling says the count, not the effect,
# is what set the statistic.
_INNER_PANELS = (('agree', "Max-z region is the deepest count's"),
                 ('max_z_dice', 'Max-z region Dice'),
                 ('max_z_z', 'Observed max z'))


def plot_inner_perm(label: str, df, out) -> None:
    """Plot what n_perm_inner buys: detection, then the max-z region.

    Two figures, since the count acts on the test in two places.
    {label}.pdf is the usual metric grid with the count on the x-axis and the
    effect strength as its hue -- what the count costs detection.
    {label}_max_z.pdf is the stability read (_INNER_PANELS), and
    {label}_max_z.csv the same numbers per (effect strength, count), which is
    what a value of N_PERM_INNER is read off.

    Args:
        label (str): cache name; the figures' filename stems.
        df: a tidy_inner_perm frame.
        out (pathlib.Path): directory the figures are written into.
    """
    plot_metric_grid(label, df, out, x='n_perm_inner', hue='effect_llr')

    df = _max_z_agreement(df)
    llr_list = sorted(df['effect_llr'].dropna().unique().tolist())
    style = _seq_style(llr_list)
    fig, axes = plt.subplots(1, len(_INNER_PANELS), sharex=True,
                             figsize=(4.2 * len(_INNER_PANELS), 3.8),
                             squeeze=False)
    for ax, (metric, title) in zip(axes[0], _INNER_PANELS):
        _draw_metric_errbar(ax, df, 'n_perm_inner', metric, style,
                            hue='effect_llr')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel(_X_PARAM_LABELS['n_perm_inner'])
    for ax in axes[0][:2]:
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel('mean (95% CI)')

    n_perm_inner = np.sort(df['n_perm_inner'].dropna().unique())
    axes[0][-1].plot(n_perm_inner, n_perm_inner / np.sqrt(n_perm_inner + 1),
                     color='0.4', ls=':', label='self-inclusion ceiling')
    axes[0][-1].legend(frameon=False, fontsize=8, loc='upper left')
    axes[0][0].legend(frameon=False, fontsize=8,
                      title=_HUE_TITLES['effect_llr'], title_fontsize=8)
    fig.tight_layout()
    path = out / f'{label}_max_z.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'saved: {path}')

    metrics = ['agree', 'max_z_dice', 'max_z_z', 'max_z_num_vox', 'dice',
               'n_sig', 'min_pval']
    table = (df.groupby(['effect_llr', 'n_perm_inner'])[metrics]
             .mean().reset_index())
    table['n_seed'] = (df.groupby(['effect_llr', 'n_perm_inner'])['seed']
                       .nunique().values)
    table['z_ceiling'] = (table['n_perm_inner']
                          / np.sqrt(table['n_perm_inner'] + 1))
    path = out / f'{label}_max_z.csv'
    table.to_csv(path, index=False)
    print(f'saved: {path}')



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


def _loglog_slope(x, y, fit_decades: float = 1.0) -> float:
    """Fit the power-law exponent of y against x over the top decades of x.

    Args:
        x (np.array): (n,) swept-axis values
        y (np.array): (n,) the timed median at each x
        fit_decades (float): decades of x, counted down from the largest, the
            fit is restricted to

    Returns:
        slope (float): least-squares slope of log10(y) on log10(x) over that
            tail, so 1 is linear growth and 2 quadratic; nan where the tail
            holds fewer than two positive points
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = (x > 0) & (y > 0) & (x >= x.max() / 10 ** fit_decades)
    if keep.sum() < 2:
        return float('nan')
    return float(np.polyfit(np.log10(x[keep]), np.log10(y[keep]), 1)[0])


def plot_runtime(name: str, df, out, log_x_ratio: float = 10.0,
                 fit_decades: float = 1.0) -> None:
    """Plot wall time vs the swept knob, one curve per method, on log axes.

    The runtime family's single plotter: the seed replicates collapse to a
    median line per method with a min-max band, wall time in minutes on a log
    y-axis and the swept knob on a log x-axis when it spans at least
    log_x_ratio (so the num_vox / permutation scaling reads as a slope; the
    small b sweep stays linear). Methods use the shared palette
    (COLOR_ANALYSIS), with a seaborn fallback for any label outside it
    (get_cmap_dict).

    Where both axes come out log the reader's question is the exponent, so
    each method's slope is fitted (_loglog_slope) and printed in its legend
    entry rather than left to be eyeballed. The fit takes the largest
    fit_decades of the swept axis only: at the small end a fixed startup cost
    dominates and flattens the curve, so a whole-range fit understates the
    asymptotic scaling the claim is about. The legend title names that range,
    since the number means nothing without it.

    Args:
        name (str): cache name; used in the output filename.
        df: the cache's tidy_runtime results.
        out (pathlib.Path): directory the figure is written into.
        log_x_ratio (float): x max/min ratio at or above which the x-axis is
            log-scaled.
        fit_decades (float): decades of x, counted down from the largest, each
            slope is fitted over.
    """
    df = _select_glow_arm(df.dropna(subset=['x', 'time_sec', 'label']))
    if df.empty:
        print(f'  (no timed rows for {name} — skipping)')
        return

    x_name = df['x_name'].iloc[0]
    labels = sorted(df['label'].unique().tolist())
    palette = get_cmap_dict(labels)
    log_x = df['x'].max() / max(df['x'].min(), 1) >= log_x_ratio

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label in labels:
        g = df[df['label'] == label].groupby('x')['time_sec']
        med, lo, hi = g.median() / 60, g.min() / 60, g.max() / 60
        legend = label
        if log_x:
            slope = _loglog_slope(med.index.values, med.values, fit_decades)
            legend = f'{label}  (slope {slope:.2f})'
        ax.plot(med.index, med.values, marker='o', ms=5, lw=2,
                color=palette[label], label=legend)
        ax.fill_between(med.index, lo.values, hi.values,
                        color=palette[label], alpha=0.15)

    if log_x:
        ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(_X_PARAM_LABELS.get(x_name, x_name))
    ax.set_ylabel('wall time (min)')
    ax.legend(frameon=False,
              title=(f'slope over top {fit_decades:g} decade of x'
                     if log_x else None),
              title_fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    path = out / f'{name}_runtime.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.close('all')
    print(f'saved: {path}')


# ---------------------------------------------------------------------------
# Per-cache dispatch + CLI
# ---------------------------------------------------------------------------

def plot_cache(label: str, df, out,
               metrics: list = ['dice', 'sens', 'ppv']) -> None:
    """Write one run_ana cache's figure, dispatching on its swept axis.

    The null path (no effect planted) gets a faceted FWER calibration curve;
    every other cache gets the stacked per-source detection figure
    (plot_source_grid: an HCP block over a WGN block, each a mean-band row and
    a GLOW diff row across the metric columns), that figure's numbers as text
    (write_table_txt) plus one cache-level discovery-threshold table
    (write_threshold_table). The x-axis is inferred
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
        write_table_txt(sub_label, sub, x=x, out=out, metrics=metrics)

    # one discovery-threshold table for the whole cache, a column per secondary
    # (b in the llr sweep); absolute effect_llr per method (threshold_table)
    write_threshold_table(label, df, x=x, out=out)


def _cache_dir(out, name: str):
    """Return the cache's own output directory under out, creating it.

    One directory per catalogue cache, so a run's figures group by the
    question they answer instead of sharing one flat folder where a filename
    prefix is all that tells them apart. Created on demand, so a cache with no
    records leaves no empty directory behind.

    Args:
        out (pathlib.Path): the run's output root (results/_latest).
        name (str): the cache name (a CONFIG key).

    Returns:
        pathlib.Path: out / name.
    """
    path = out / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv=None) -> None:
    """Plot the detection and runtime caches from the shared records.

    For each selected cache (every detection and runtime cache in CONFIG by
    default, or the names given on the command line), reads its provenance
    frame -- make_csv.write_config_csv, which refreshes that cache's CSV on the
    way past -- and dispatches by family:

      - runtime: tidy_runtime + plot_runtime (wall time vs its cost knob)
      - other run_ana: tidy_run_ana + plot_cache (detection sweep /
        calibration)
      - stat bake-off: read straight from the run_stat leaves
        (results.stat_cell_df), written as two paper tables
      - segment / prune: tidy_segment / tidy_prune as a source x metric grid,
        prune one grid per Ward mode. Which segment cut a cache gets comes
        off its own grids (_segment_perc).
      - inner-draw sweep: tidy_inner_perm + plot_inner_perm (detection and the
        max-z region against the count)

    Figures and tables land in results/_latest/<cache>, one directory per
    cache (_cache_dir), so a mid-benchmark run yields intermediate output
    grouped by the question it answers.

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
    from .run import (run_ana, run_inner_perm, run_prune, run_segment,
                      run_stat)
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
    stat_names = [n for n, cfg in CONFIG.items() if cfg[3] is run_stat]
    segment_names = [n for n, cfg in CONFIG.items() if cfg[3] is run_segment]
    prune_names = [n for n, cfg in CONFIG.items() if cfg[3] is run_prune]
    inner_names = [n for n, cfg in CONFIG.items() if cfg[3] is run_inner_perm]
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
        names = (detect_names + runtime_names + stat_names
                 + segment_names + prune_names + inner_names)

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
            plot_runtime(name, df, _cache_dir(out, name))
            n_plotted += 1
        elif name in detect_names:
            df = tidy_run_ana(make_csv.write_config_csv(name))
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {len(df)} run_ana rows ===')
            plot_cache(name, df, _cache_dir(out, name))
            n_plotted += 1
        elif name in stat_names:
            df = tidy_stat(results.stat_cell_df())
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {df["cell"].nunique()} cells, '
                  f'{len(df)} variant rows ===')
            write_stat_tables(name, df, _cache_dir(out, name))
            n_plotted += 1
        elif name in segment_names:
            # which figure this cache asks for -- and so which half of the
            # shared record walk is its own -- is read off its grids, not its
            # name (_segment_perc)
            _, effect_list, fnc_list, _ = CONFIG[name]
            perc = _segment_perc(effect_list, fnc_list)
            df = tidy_segment(make_csv.write_config_csv(name), perc=perc)
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {len(df)} segment rows ===')
            out_cache = _cache_dir(out, name)
            if perc is None:
                plot_segment_llr(name, df, out_cache)
                plot_segment_compare(name, df, out_cache)
            else:
                x, log_x = (('frac_segment', False) if perc
                            else ('effect_llr', None))
                plot_metric_grid(name, df, out_cache, x=x, log_x=log_x)
            n_plotted += 1
        elif name in prune_names:
            df = tidy_prune(make_csv.write_config_csv(name))
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {len(df)} prune rows ===')
            plot_prune(name, df, _cache_dir(out, name))
            n_plotted += 1
        elif name in inner_names:
            df = tidy_inner_perm(make_csv.write_config_csv(name))
            if df.empty:
                print(f'  (no records for {name} — skipping)')
                continue
            print(f'\n=== {name}: {len(df)} inner-draw rows ===')
            plot_inner_perm(name, df, _cache_dir(out, name))
            n_plotted += 1
        else:
            print(f'  ({name} is not a detection or runtime cache — skipping)')

    if n_plotted == 0:
        print(f'no results found under {out.parent}')


if __name__ == '__main__':
    main()
