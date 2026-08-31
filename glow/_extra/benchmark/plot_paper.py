"""Manuscript figures for the GLOW paper, drawn to one visual grammar.

Three channels, one meaning each, never reused:

    hue             the method family (GLOW teal, VBA coral, VBA-TFCE purple,
                    CET gold)
    lightness/dash  a variant within GLOW: the teal ladder for the knob, a
                    dash for the rule that lost
    panel position  the data source, HCP the left column and WGN the right

Source is never a hue. It is column position, a bold column header on the top
row, and a faint axes tint on the synthetic (WGN) panels, so a reader who has
learned the encoding once never re-learns it. Metric is always the row, named
by the left column's y-label; the x-label appears on the bottom row only and
the legend once, below the axes.

GLOW's teal is the teal of image/tikz/summary.tex, so the schematic that
introduces the method and every curve that measures it agree. GLOW draws
heavier than any baseline (_LW_LEAD against _LW_BASE), which also survives a
grayscale print.

Figures are sized to the manuscript's 17.8 cm textwidth (_TEXTWIDTH_IN) so
they are included unscaled and the point sizes here are the ones a reader
sees. This module owns its drawing outright; plot.py stays the draft view,
stamps and all.

See publications/submissions/2026_glow/notes/figure_style.md for the
rationale, the attention devices, and the main-text / appendix split these
figures implement.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .config import REPORTED_GLOW_LABEL
from .plot import (_ARM_LABEL, _binom_ci, tidy_pred_decomp,
                   tidy_prune, tidy_run_ana, tidy_runtime,
                   tidy_segment)


# 17.8 cm textwidth (imag-ms-template.cls), in inches
_TEXTWIDTH_IN = 17.8 / 2.54

# Method hues: one wheel, the teal of summary.tex at H=0.5 and the three
# baselines spread over the warm half, so GLOW is the only cool line on the
# page. S/L are held at the teal's (0.366 / 0.476) so no method reads as
# louder than another.
COLOR_METHOD = {
    'GLOW':      '#4da6a6',
    'VBA':       '#a64d4d',
    'VBA-TFCE':  '#894da6',
    'CET':       '#a6884d',
}

# The teal ladder, taken verbatim from image/tikz/summary.tex, so a tuning
# curve is visibly the same method as the schematic that introduced it.
# Darkest is the setting the paper ships.
TEAL_LADDER = ('#2B7A78', '#4DA6A6', '#7FBBBB', '#B8DEDE')

# WGN panels carry a faint tint and HCP panels none: real data reads as the
# clean panel, synthetic as the tinted one. A tint, not a hue, so the colour
# channel stays spent on method alone.
_TINT = {'HCP': 'white', 'WGN': '#f2f2f2'}
_SOURCE_ORDER = ('HCP', 'WGN')

_LW_LEAD = 2.6
_LW_BASE = 1.5

# Markers repeat what hue and lightness already say. They are the redundant
# channel, not a fourth meaning: two methods whose curves coincide (the
# runtime baselines do, exactly) stay tellable apart, and so does a grayscale
# print. Never used to encode anything a colour does not already encode.
_MARKERS = ('o', 's', '^', 'D', 'v')

# The defaults every non-swept experiment sits at (Results preamble). Marked
# in each LLR sweep so a reader can see where the paper's numbers were taken.
_DEFAULT_LLR = 0.03

_METRIC_LABEL = {
    'dice': 'Dice',
    'sens': 'Sensitivity',
    'ppv': 'PPV',
    'spec': 'Specificity',
    'n_selected': 'Regions returned',
}

_X_LABEL = {
    'effect_llr': 'effect strength (LLR / voxel)',
    'effect_perc': 'effect extent (fraction of crop)',
    'b': 'imaging features $b$',
    'num_img': 'subjects $N$',
    'num_vox': 'voxels analyzed',
    'seed_rank': 'seed, ranked by false volume',
}


def _rc() -> dict:
    """Return the rcParams every paper figure is drawn under.

    Sans-serif to match the class's \\sfdefault body text, and point sizes
    that land below the 12pt body without going unreadable at true size.
    """
    return {
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 9,
        'axes.labelsize': 9.5,
        'axes.titlesize': 10.5,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 9,
        'axes.linewidth': 0.7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.35,
        'lines.solid_capstyle': 'round',
        # Not bbox='tight'. A trimmed page is narrower than the figure it came
        # from, so \includegraphics[width=\textwidth] scales it back up and
        # every point size here comes out larger than it says. Constrained
        # layout fits the content to the declared size instead, leaving the
        # saved page exactly _TEXTWIDTH_IN wide and the inclusion 1:1.
        'savefig.bbox': 'standard',
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.02,
        'figure.constrained_layout.w_pad': 0.02,
        'figure.constrained_layout.hspace': 0.03,
        'figure.constrained_layout.wspace': 0.02,
        'pdf.fonttype': 42,
    }


def method_style(labels) -> dict:
    """Map benchmark method labels to the paper's (colour, dash, width).

    A GLOW arm takes the teal and the lead width whatever its recipe suffix,
    since the arms are one method to a reader of the paper; a baseline takes
    its own hue at the base width. An unknown label falls back to grey rather
    than silently borrowing a method's colour.

    Args:
        labels: the method labels present in the frame.

    Returns:
        out (dict): label -> {'color': str, 'ls': str, 'lw': float}.
    """
    out = {}
    for lab in labels:
        name = str(lab)
        if name.startswith('GLOW'):
            out[lab] = {'color': COLOR_METHOD['GLOW'], 'ls': '-',
                        'lw': _LW_LEAD}
        elif name in COLOR_METHOD:
            out[lab] = {'color': COLOR_METHOD[name], 'ls': '-',
                        'lw': _LW_BASE}
        else:
            out[lab] = {'color': '#808080', 'ls': '-', 'lw': _LW_BASE}
    return out


def ladder_style(values, chosen=None, dashed=()) -> dict:
    """Map ordered GLOW variants onto the teal ladder, darkest first.

    The tuning figures compare settings of GLOW, not rival methods, so the
    variants share the teal and separate by lightness. The setting the paper
    ships is drawn darkest and at the lead width, which is what makes a glance
    say which one was picked.

    Args:
        values: the variant labels, in the order they should darken.
        chosen: the label the paper ships, drawn darkest and heaviest; None
            leaves every variant at the base width.
        dashed: labels drawn dashed (the rule that lost).

    Returns:
        out (dict): label -> {'color': str, 'ls': str, 'lw': float}.
    """
    values = list(values)
    # the chosen setting takes the ladder's darkest rung wherever it sits in
    # the caller's order, so "darkest = shipped" holds in every tuning panel
    order = ([chosen] + [v for v in values if v != chosen]
             if chosen in values else values)
    out = {}
    for i, val in enumerate(order):
        out[val] = {
            'color': TEAL_LADDER[i % len(TEAL_LADDER)],
            'ls': '--' if val in dashed else '-',
            'lw': _LW_LEAD if val == chosen else _LW_BASE,
        }
    return out


def _sources(df):
    """List the frame's sources in the paper's fixed left-to-right order."""
    have = set(df['source'].dropna().unique())
    return [s for s in _SOURCE_ORDER if s in have]


def _grid(nrow: int, sources: list, *, height: float, width_frac: float = 1.0):
    """Open a source-columned figure at the manuscript's text width.

    Args:
        nrow (int): number of metric rows.
        sources (list): the source column order (left to right).
        height (float): figure height in inches.
        width_frac (float): fraction of textwidth to occupy; a figure the tex
            includes at less than \\textwidth passes its own fraction so the
            point sizes still come out right.

    Returns:
        (fig, axes): axes is always 2-D, (nrow, len(sources)).
    """
    fig, axes = plt.subplots(
        nrow, len(sources), squeeze=False, sharex=True, sharey='row',
        figsize=(_TEXTWIDTH_IN * width_frac, height))
    return fig, axes


def _dress(axes, sources: list, *, x: str, row_labels: list,
           log_x: bool = True, mark_default: bool = False) -> None:
    """Apply the grammar's furniture rules to a drawn grid.

    Source headers on the top row, metric names on the left column, the
    x-label on the bottom row alone, and the WGN tint down its whole column.
    Everything a neighbouring panel already says is left off.

    Args:
        axes: the (nrow, ncol) axes array.
        sources (list): the source per column.
        x (str): the swept column, for the bottom-row label.
        row_labels (list): the y-label per row (metric names).
        log_x (bool): log the shared x-axis.
        mark_default (bool): draw the default operating-point rule.
    """
    nrow = axes.shape[0]
    for j, src in enumerate(sources):
        for i in range(nrow):
            ax = axes[i, j]
            ax.set_facecolor(_TINT.get(src, 'white'))
            ax.grid(True)
            ax.set_axisbelow(True)
            if log_x:
                ax.set_xscale('log')
            # the operating point the Results preamble fixes, unlabelled: it
            # is a reading aid, not a series
            if mark_default and x == 'effect_llr':
                ax.axvline(_DEFAULT_LLR, ls=':', lw=0.8, color='#999999',
                           zorder=0)
            if i == 0:
                ax.set_title(src, fontweight='bold', pad=6)
            if j == 0 and i < len(row_labels):
                ax.set_ylabel(row_labels[i])
            if i == nrow - 1:
                ax.set_xlabel(_X_LABEL.get(x, x))


def _legend(fig, handles, *, ncol: int = None, y: float = 0.0) -> None:
    """Put the figure's one legend below the axes, unframed.

    Placed outside the axes so constrained layout reserves its space rather
    than the legend landing on a curve, which is what the draft figures do.

    Args:
        fig: the figure to attach to.
        handles (list): Line2D proxies, in the order they should read.
        ncol (int | None): legend columns; None puts everything on one row.
        y (float): unused; kept so callers read as placement-agnostic.
    """
    fig.legend(handles=handles, loc='outside lower center',
               ncol=ncol or len(handles), frameon=False,
               handlelength=1.9, columnspacing=1.6)


def _proxy(label: str, st: dict):
    """Build one legend proxy line from a style dict."""
    return Line2D([], [], color=st['color'], ls=st['ls'],
                  lw=st.get('lw', _LW_BASE), marker=st.get('marker'),
                  ms=st.get('ms', 3.6), label=label)


def _band(ax, df, x: str, metric: str, st: dict, *, ci: int = 95,
          band: bool = True, errbar: bool = False,
          dodge: float = 0.0) -> None:
    """Draw one series as its seed mean, with the spread as band or error bar.

    Bands suit a continuous sweep. They stop working once several same-hue
    series overlap (the tuning panels) or the x is a handful of discrete
    values (the feature-count sweep), so those pass band=False / errbar=True
    and the spread reads as a bar at each point instead.

    Args:
        ax: the Axes to draw into.
        df: rows for one (source, series); numeric x and metric.
        x (str): the swept column.
        metric (str): the scored column.
        st (dict): {'color', 'ls', 'lw'} and optionally 'marker'.
        ci (int): central percentile width of the band / bar.
        band (bool): fill the central ci% as a band.
        errbar (bool): draw the central ci% as a vertical bar per x.
        dodge (float): fractional multiplicative x-offset for the error bars,
            so several series' bars at one x stay separable.
    """
    grp = df.groupby(x)[metric]
    mean = grp.mean()
    lo = grp.quantile((1 - ci / 100) / 2)
    hi = grp.quantile(1 - (1 - ci / 100) / 2)
    if band:
        ax.fill_between(mean.index, lo.values, hi.values, color=st['color'],
                        alpha=0.13, lw=0, zorder=1)
    if errbar:
        xv = mean.index.values * (1 + dodge)
        # the bar is the percentile interval, so it is centred on that
        # interval rather than on the mean: a mean can sit outside [lo, hi]
        # (a skewed series, or a float sliver where the two coincide), which
        # errorbar rejects outright as a negative yerr
        ax.errorbar(xv, (lo.values + hi.values) / 2,
                    yerr=(hi.values - lo.values) / 2,
                    fmt='none', ecolor=st['color'], elinewidth=0.9,
                    capsize=2, alpha=0.7, zorder=2)
    ax.plot(mean.index, mean.values, color=st['color'], ls=st['ls'],
            lw=st['lw'], marker=st.get('marker'), ms=st.get('ms', 3.6),
            markevery=st.get('markevery', 1), zorder=3)


def _numeric(df, cols: list):
    """Coerce columns to numeric and drop rows missing any of them."""
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out.dropna(subset=cols)


def _save(fig, out, stem: str) -> None:
    """Write one figure as {stem}.pdf and a matching png for quick review."""
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{stem}.{ext}', dpi=200)
    plt.close(fig)
    print(f'  wrote {stem}.pdf')


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

# The Ward objectives, ordered as the paper argues them
_SEGMENT_ORDER = ('GLM Error', 'Focus', 'Naive')
# The prune rules the paper compares; oracle and single_max are diagnostics
_PRUNE_KEEP = ('GLOW-greedy', 'GLOW-dp')

# The recipe suffix config ships spells the Ward mode the way the analysis
# names it; the segment / prune frames spell it the way the clustering does
_MODE_OF_TOKEN = {'GLM': 'GLM Error', 'Focus': 'Focus', 'Naive': 'Naive'}


def _shipped(label: str = REPORTED_GLOW_LABEL):
    """Split config's reported arm into the settings the tuning figure marks.

    Everything downstream reads the shipped setting from config rather than
    restating it, so the bolded curve in the tuning figure and the single
    curve in every detection figure cannot drift apart or from the benchmark.

    Args:
        label (str): a GLOW recipe label, GLOW-<mode>-<rule>.

    Returns:
        (mode, rule): the Ward objective as the segment / prune frames spell
            it, and the prune rule under its frame label (GLOW-<rule>).
    """
    parts = str(label).split('-')
    mode = _MODE_OF_TOKEN.get(parts[1], parts[1]) if len(parts) > 2 else None
    rule = f'GLOW-{parts[-1]}' if len(parts) > 2 else None
    return mode, rule


_SEGMENT_CHOSEN, _PRUNE_CHOSEN = _shipped()


def fig_tuning(seg_df, prune_df, out, *, metric: str = 'dice') -> None:
    """Draw the two tuning decisions as one 2 x 2 figure.

    Row 1 is the segmentation objective (three Ward variants, oracle Dice of
    the best-matching region); row 2 is the pruning rule crossed with the
    objective it ran on. Columns are the sources. Both rows read on the teal
    ladder, so the figure says at a glance that every curve is GLOW under a
    different setting, and the shipped setting is the darkest, heaviest line.

    This replaces the three full metric grids (segment plus one prune grid per
    Ward mode, 18 panels) that the draft view writes; those move to the
    appendix figure (fig_appendix_grids).

    Args:
        seg_df: tidy_segment frame (source / label / seed / effect_llr).
        prune_df: tidy_prune frame (adds cluster_mode and the rule label).
        out (pathlib.Path): directory to write into.
        metric (str): the deciding metric drawn in both rows.
    """
    x = 'effect_llr'
    seg = _numeric(seg_df, [x, metric])
    pru = _numeric(prune_df, [x, metric])
    pru = pru[pru['label'].isin(_PRUNE_KEEP)]
    sources = _sources(seg)

    seg_labels = [s for s in _SEGMENT_ORDER
                  if s in set(seg['label'].dropna().unique())]
    seg_style = ladder_style(seg_labels, chosen=_SEGMENT_CHOSEN)
    # legend order follows the ladder, shipped setting (darkest) first
    if _SEGMENT_CHOSEN in seg_labels:
        seg_labels = ([_SEGMENT_CHOSEN]
                      + [s for s in seg_labels if s != _SEGMENT_CHOSEN])
    for i, lab in enumerate(seg_labels):
        seg_style[lab]['marker'] = _MARKERS[i % len(_MARKERS)]

    # the prune row crosses rule with objective: the objective keeps the
    # ladder rung and marker it had in row 1, the rule rides the dash, so a
    # reader tracks one colour down the figure
    modes = [m for m in _SEGMENT_ORDER
             if m in set(pru['cluster_mode'].dropna().unique())]
    pru_style = {}
    for mode in modes:
        for rule in _PRUNE_KEEP:
            base = seg_style.get(mode, {'color': TEAL_LADDER[1],
                                        'marker': 'o'})
            pru_style[(mode, rule)] = {
                'color': base['color'],
                'marker': base.get('marker'),
                'ls': '-' if rule == _PRUNE_CHOSEN else '--',
                'lw': (_LW_LEAD if (mode == _SEGMENT_CHOSEN
                                    and rule == _PRUNE_CHOSEN)
                       else _LW_BASE),
            }

    # no bands in this figure: four same-hue series overlapping turn the fill
    # to mush, and the per-metric spread is what the appendix grids are for
    fig, axes = _grid(2, sources, height=4.4)
    for j, src in enumerate(sources):
        for lab in seg_labels:
            sub = seg[(seg['source'] == src) & (seg['label'] == lab)]
            if len(sub):
                _band(axes[0, j], sub, x, metric, seg_style[lab], band=False)
        for (mode, rule), st in pru_style.items():
            sub = pru[(pru['source'] == src) & (pru['cluster_mode'] == mode)
                      & (pru['label'] == rule)]
            if len(sub):
                _band(axes[1, j], sub, x, metric, st, band=False)

    lab = _METRIC_LABEL.get(metric, metric)
    _dress(axes, sources, x=x, mark_default=True,
           row_labels=[f'{lab} (oracle region)', f'{lab} (after pruning)'])
    for ax in axes.ravel():
        ax.set_ylim(0, 1)

    # a legend per row, inside the left panel: the rows compare different
    # things, and one shared legend below would not say which entry is whose
    axes[0, 0].legend(
        handles=[_proxy(l, seg_style[l]) for l in seg_labels],
        title='Ward objective', loc='upper left', frameon=False,
        fontsize=8, title_fontsize=8, handlelength=2.0, labelspacing=0.25)
    axes[1, 0].legend(
        handles=[_proxy(f'{m}, {r.split("-")[1]}', st)
                 for (m, r), st in pru_style.items()],
        title='objective, prune rule', loc='upper left', frameon=False,
        fontsize=8, title_fontsize=8, handlelength=2.0, labelspacing=0.25)
    _save(fig, out, 'tuning')


def fig_null(df, out) -> None:
    """Draw the FWER calibration as one half-width panel, both sources.

    The draft view spends a full-width 1 x 2 grid on what is a diagonal line.
    Collapsed here to a single panel with the sources separated by dash (the
    one place source is not a column, because there is only one), plus the
    Clopper-Pearson band on each.

    Args:
        df: the null cache's tidy_run_ana frame (needs min_pval / label /
            source).
        out (pathlib.Path): directory to write into.
    """
    df = df.copy()
    df['min_pval'] = pd.to_numeric(df['min_pval'], errors='coerce')
    df = df.dropna(subset=['min_pval'])
    glow = df[df['label'].astype(str).str.startswith('GLOW')]
    sources = _sources(glow)

    fig, ax = plt.subplots(figsize=(_TEXTWIDTH_IN * 0.5,
                                    _TEXTWIDTH_IN * 0.5))
    ax.plot([0, 1], [0, 1], ls=':', lw=1, color='#999999', zorder=1)

    teal = COLOR_METHOD['GLOW']
    alphas = np.linspace(0, 1, 200)
    handles = []
    for src in sources:
        pvals = glow.loc[glow['source'] == src, 'min_pval'].to_numpy(float)
        pvals = pvals[np.isfinite(pvals)]
        if not pvals.size:
            continue
        n0 = (pvals[None, :] <= alphas[:, None]).sum(axis=1)
        rate = n0 / pvals.size
        lo, hi = _binom_ci(n0, pvals.size)
        ls = '-' if src == 'HCP' else '--'
        # both sources are teal here (source is the dash, not the hue), so two
        # filled bands would be one indistinguishable wash: the interval is
        # drawn as a faint envelope carrying its own series' dash instead
        ax.plot(alphas, lo, lw=0.6, ls=ls, color=teal, alpha=0.4, zorder=2)
        ax.plot(alphas, hi, lw=0.6, ls=ls, color=teal, alpha=0.4, zorder=2)
        ax.plot(alphas, rate, lw=_LW_LEAD, ls=ls, color=teal, zorder=3)
        # the rate at the paper's alpha, read off rather than left to the eye
        at05 = float((pvals <= 0.05).mean())
        handles.append(Line2D(
            [], [], color=teal, ls=ls, lw=_LW_LEAD,
            label=f'{src}  (n={pvals.size}, rate {at05:.2f} at 0.05)'))

    # the paper's claim is at one alpha; mark it rather than leave the reader
    # to find 0.05 on a 0..1 axis
    ax.axvline(0.05, ls=':', lw=0.8, color='#999999', zorder=0)
    ax.annotate(r'$\alpha=0.05$', xy=(0.05, 1.0), xytext=(3, -2),
                textcoords='offset points', fontsize=7.5, color='#666666',
                ha='left', va='top', rotation=90)

    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=r'nominal $\alpha$',
           ylabel='empirical rejection rate')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_axisbelow(True)
    handles.append(Line2D([], [], color='#999999', ls=':', lw=1,
                          label='ideal'))
    ax.legend(handles=handles, loc='lower right', frameon=False,
              fontsize=8.5, handlelength=1.8)
    _save(fig, out, 'null_calibration')


_BASELINE_ORDER = ('VBA', 'VBA-TFCE', 'CET')


def fig_detection(df, out, stem: str, *, x: str,
                  metrics=('dice', 'sens', 'ppv'),
                  diff_metric: str = 'dice', log_x: bool = True,
                  arm: str = REPORTED_GLOW_LABEL, errbar: bool = False,
                  int_x: bool = False) -> None:
    """Draw a detection sweep as a metric x source grid, GLOW against the field.

    The headline layout: one row per metric, one column per source, and (when
    diff_metric is given and the frame has a baseline to diff against) a final
    row holding GLOW minus the best baseline at each x. That diff row carries
    the seed median and an interquartile band rather than one thin line per
    seed, which at print size reads as noise.

    Exactly one GLOW curve is drawn, config's reported arm. Its two knobs --
    the Ward projection and the selection rule -- are argued in the tuning
    figure, off the segment and prune caches; a second teal line here would
    re-open that choice in the figure meant to settle GLOW against the field.

    Args:
        df: a tidy_run_ana frame (source / label / seed / x / the metrics).
        out (pathlib.Path): directory to write into.
        stem (str): output filename stem.
        x (str): the swept column.
        metrics (iterable): the metric columns, one row each.
        diff_metric (str | None): the metric the head-to-head row differences;
            None omits the row.
        log_x (bool): log the x-axis.
        arm (str): the GLOW recipe label drawn as GLOW.
        errbar (bool): spread as a per-point bar rather than a band, for a
            sweep whose x is a handful of discrete values.
        int_x (bool): force integer x ticks (a count on the x-axis).
    """
    metrics = list(metrics)
    df = _numeric(df, [x, *metrics])
    sources = _sources(df)
    have = set(df['label'].dropna().unique())

    glow = arm if arm in have else next(
        (c for c in sorted(have) if str(c).startswith('GLOW')), None)
    others = [c for c in _BASELINE_ORDER if c in have]
    others += [c for c in sorted(have)
               if c not in others and not str(c).startswith('GLOW')]
    labels = ([glow] if glow else []) + others
    style = method_style(labels)
    if glow:
        style[glow]['marker'] = _MARKERS[0]
    for i, lab in enumerate(others):
        style[lab]['marker'] = _MARKERS[(i + 1) % len(_MARKERS)]

    want_diff = bool(diff_metric and glow and others)
    nrow = len(metrics) + (1 if want_diff else 0)

    fig, axes = _grid(nrow, sources, height=1.55 * nrow + 1.15)
    for j, src in enumerate(sources):
        dsrc = df[df['source'] == src]
        for i, metric in enumerate(metrics):
            for lab in labels:
                sub = dsrc[dsrc['label'] == lab]
                if len(sub):
                    _band(axes[i, j], sub, x, metric, style[lab],
                          band=not errbar, errbar=errbar)
            axes[i, j].set_ylim(0, 1)
        if want_diff:
            _diff_band(axes[-1, j], dsrc, x, diff_metric, glow, others)

    rows = [_METRIC_LABEL.get(m, m) for m in metrics]
    if want_diff:
        rows.append(f'GLOW $-$ best\nbaseline ({_METRIC_LABEL[diff_metric]})')
    _dress(axes, sources, x=x, row_labels=rows, log_x=log_x,
           mark_default=True)
    if int_x:
        for ax in axes.ravel():
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    handles = ([_proxy('GLOW', style[glow])] if glow else [])
    handles += [_proxy(lab, style[lab]) for lab in others]
    # a one-row figure has far less height for the legend to sit under
    bottom = 0.30 if nrow == 1 else 0.13 - 0.012 * (nrow - 2)
    _legend(fig, handles, y=0.055 if nrow == 1 else 0.035)
    _save(fig, out, stem)


def _diff_band(ax, dsrc, x: str, metric: str, glow: str, others: list,
               ci: int = 50) -> None:
    """Draw GLOW minus the best baseline: seed median and an IQR band.

    Args:
        ax: the Axes to draw into.
        dsrc: one source's rows.
        x (str): the swept column.
        metric (str): the differenced metric.
        glow (str): the GLOW label whose line is drawn.
        others (list): baseline labels the per-seed maximum is taken over.
        ci (int): central percentile width of the band (50 = IQR).
    """
    agg = dsrc.groupby(['label', 'seed', x], as_index=False)[metric].mean()
    pivot = agg.pivot_table(index=['seed', x], columns='label',
                            values=metric).reset_index()
    have = [c for c in others if c in pivot.columns]
    if glow not in pivot.columns or not have:
        ax.axis('off')
        return
    pivot['best'] = pivot[have].max(axis=1, skipna=True)
    pivot = pivot.dropna(subset=[glow, 'best'])
    pivot['d'] = pivot[glow] - pivot['best']

    grp = pivot.groupby(x)['d']
    med = grp.median()
    lo = grp.quantile((1 - ci / 100) / 2)
    hi = grp.quantile(1 - (1 - ci / 100) / 2)
    teal = COLOR_METHOD['GLOW']
    ax.fill_between(med.index, lo.values, hi.values, color=teal, alpha=0.18,
                    lw=0, zorder=1)
    ax.plot(med.index, med.values, color=teal, lw=_LW_LEAD, zorder=3)
    ax.axhline(0, lw=0.8, color='black', alpha=0.6, zorder=2)
    ax.set_ylim(-0.6, 0.6)


# The two ways a region method can be wrong, one colour each, held across
# every panel of the three precision figures: over-inclusion takes the
# ladder's darkest rung and spurious a light one, both plainly GLOW's teal.
_COLOR_LEAK = TEAL_LADDER[0]
_COLOR_SPUR = TEAL_LADDER[2]

# A false-volume ratio spans four decades and is exactly zero whenever a trial
# makes that error not at all, which a log axis cannot draw. The volume rows
# are symlog with this linear threshold, so zero is a plotted value and not a
# gap: a trial that leaks nothing is a result, not missing data. The threshold
# sits below one voxel of a planted effect.
_VOL_LINTHRESH = 1e-3

# Below this a trial is called a precision failure rather than a soft one.
# Arbitrary, and only ever used to count how many seeds land there.
_PPV_FLOOR = 0.5


def _decomp_cols(df, arm: str = None):
    """Add the precision quantities the false-volume figures share.

    Keeps the detecting trials only (a trial that discovered nothing has no
    PPV to explain), then derives the two counterfactual precisions: what PPV
    would have been had each error source contributed nothing. Whichever
    counterfactual recovers the deficit names the mechanism.

    Args:
        df: a tidy_pred_decomp frame.
        arm (str | None): keep this method's rows alone; None keeps all.

    Returns:
        the frame with pred_vol, true_vol, vol_leak / vol_spur / vol_fp (each
        as a multiple of the planted effect volume), ppv_noleak and ppv_nospur.
    """
    d = _numeric(df, ['tp', 'fp', 'fn', 'leaked', 'spurious'])
    if arm is not None:
        d = d[d['label'] == arm]
    d = d[d['tp'] + d['fp'] > 0].copy()

    d['pred_vol'] = d['tp'] + d['fp']
    d['true_vol'] = d['tp'] + d['fn']
    for name, num in (('vol_leak', d['leaked']), ('vol_spur', d['spurious']),
                      ('vol_fp', d['fp'])):
        d[name] = num / d['true_vol']
    # each counterfactual zeroes one error source and keeps the other
    d['ppv_noleak'] = d['tp'] / (d['tp'] + d['spurious'])
    d['ppv_nospur'] = d['tp'] / (d['tp'] + d['leaked'])
    return d


def _vol_axis(ax) -> None:
    """Scale one false-volume row: symlog, decade ticks, the parity rule.

    The dotted rule at 1 is where a method's false volume equals the volume of
    the effect it found, which is the reading the row exists to support.
    """
    ax.set_yscale('symlog', linthresh=_VOL_LINTHRESH, linscale=0.4)
    ax.set_ylim(0, 20)
    ax.set_yticks([0, 1e-3, 1e-2, 1e-1, 1, 10])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.axhline(1.0, ls=':', lw=0.8, color='#999999', zorder=0)


def _rate_line(ax, df, x: str, mask, st: dict) -> None:
    """Draw the percentage of trials satisfying mask against x.

    Args:
        ax: the Axes to draw into.
        df: rows for one (source, series).
        x (str): the swept column.
        mask: a boolean Series aligned to df.
        st (dict): {'color', 'ls', 'lw'} and optionally 'marker'.
    """
    rate = 100 * mask.groupby(df[x]).mean()
    ax.plot(rate.index, rate.values, color=st['color'], ls=st['ls'],
            lw=st['lw'], marker=st.get('marker'), ms=st.get('ms', 3.6),
            zorder=3)


def fig_ppv_decomp(df, out, *, x: str = 'effect_llr',
                   arm: str = REPORTED_GLOW_LABEL,
                   stem: str = 'ppv_decomp') -> None:
    """Attribute GLOW's false volume to invention or to over-inclusion.

    The PPV panel of the headline sweep says GLOW is the least precise arm; it
    does not say why, and the two candidate reasons call for different fixes.
    Either GLOW declares regions that miss the effect entirely (invention, a
    testing failure) or it declares regions that hit the effect and carry
    their non-effect surroundings along (over-inclusion, the cost of a region
    being the unit of inference, addressable by the selection rule).

    Three rows answer it three ways, so no single summary carries the claim:
    the counterfactual precisions (row 1) say which source, if removed, would
    close the gap; the volumes (row 2) say how much of each there is against
    the size of the planted effect; and the incidence (row 3) separates a
    mechanism that fires on nearly every trial from one that fires rarely and
    expensively. A baseline is drawn in row 1 as the reference the deficit is
    measured against.

    Args:
        df: a tidy_pred_decomp frame.
        out (pathlib.Path): directory to write into.
        x (str): the swept column.
        arm (str): the GLOW recipe label to decompose.
        stem (str): output filename stem.
    """
    d = _decomp_cols(df, arm=None)
    if d.empty or arm not in set(d['label']):
        print(f'  (no {arm} rows to decompose - skipping {stem})')
        return
    sources = _sources(d)

    ref = next((c for c in _BASELINE_ORDER if c in set(d['label'])), None)
    st_glow = {'color': COLOR_METHOD['GLOW'], 'ls': '-', 'lw': _LW_LEAD,
               'marker': _MARKERS[0]}
    # rows 2 and 3 mark their points, so a reader can tell a value at the
    # symlog zero from a line passing through it
    st_leak = {'color': _COLOR_LEAK, 'ls': '-', 'lw': _LW_BASE,
               'marker': _MARKERS[0], 'ms': 3.0}
    st_spur = {'color': _COLOR_SPUR, 'ls': '-', 'lw': _LW_BASE,
               'marker': _MARKERS[1], 'ms': 3.0}
    st_ref = (dict(method_style([ref])[ref], marker=_MARKERS[1])
              if ref else None)

    fig, axes = _grid(3, sources, height=5.9)
    for j, src in enumerate(sources):
        g = d[(d['source'] == src) & (d['label'] == arm)]

        # row 1: observed PPV, then the same trials with one error zeroed
        _band(axes[0, j], g, x, 'ppv', st_glow, band=False)
        for metric, st in (('ppv_noleak', st_leak), ('ppv_nospur', st_spur)):
            _band(axes[0, j], g, x, metric,
                  dict(st, ls='--', lw=1.8, marker=None), band=False)
        if ref is not None:
            _band(axes[0, j], d[(d['source'] == src) & (d['label'] == ref)],
                  x, 'ppv', st_ref, band=False)
        axes[0, j].set_ylim(0, 1.03)

        # row 2: how much of each, in units of the planted effect's volume
        for metric, st in (('vol_leak', st_leak), ('vol_spur', st_spur)):
            _band(axes[1, j], g, x, metric, st, band=False)
        _vol_axis(axes[1, j])

        # row 3: how often each fires at all, which the means above hide
        _rate_line(axes[2, j], g, x, g['leaked'] > 0, st_leak)
        _rate_line(axes[2, j], g, x, g['spurious'] > 0, st_spur)
        axes[2, j].set_ylim(0, 103)

    _dress(axes, sources, x=x, mark_default=True,
           row_labels=['PPV', 'false volume\n($\\times$ effect volume)',
                       'trials affected (%)'])
    handles = [_proxy(f'{_ARM_LABEL.get(arm, "GLOW")}, observed', st_glow),
               _proxy('over-inclusion onto a true region', st_leak),
               _proxy('spurious region, no effect voxels', st_spur),
               _proxy('dashed: PPV with that source removed',
                      {'color': '#777777', 'ls': '--', 'lw': 1.8})]
    if ref is not None:
        handles.append(_proxy(f'{ref} (reference)', st_ref))
    _legend(fig, handles, ncol=2)
    _save(fig, out, stem)


def fig_ppv_seed(df, out, *, x: str = 'effect_llr',
                 arm: str = REPORTED_GLOW_LABEL,
                 stem: str = 'ppv_seed') -> None:
    """Draw precision one seed at a time, against the mean that hides them.

    A seed fixes the planted extent, so a per-seed line is one geometry
    followed across effect strength. The mean and its percentile band imply a
    single population being shifted; if instead a minority of geometries fail
    badly and the rest are near-perfect, the mean is describing neither, and
    what needs fixing is whatever those geometries have in common rather than
    the method's operating point.

    Rows 1 and 2 draw GLOW alone (one thin line per seed, the across-seed
    median heavy). Row 3 counts the seeds below a precision floor for every
    method, which is where a bimodal spread shows up as a number and where a
    baseline's absence from the failure mode is visible.

    Args:
        df: a tidy_pred_decomp frame.
        out (pathlib.Path): directory to write into.
        x (str): the swept column.
        arm (str): the GLOW recipe label whose seeds are drawn.
        stem (str): output filename stem.
    """
    d = _decomp_cols(df, arm=None)
    if d.empty or arm not in set(d['label']):
        print(f'  (no {arm} rows to decompose - skipping {stem})')
        return
    sources = _sources(d)
    teal = COLOR_METHOD['GLOW']

    others = [c for c in _BASELINE_ORDER if c in set(d['label'])]
    style = method_style([arm] + others)
    style[arm].update(marker=_MARKERS[0])
    for i, lab in enumerate(others):
        style[lab]['marker'] = _MARKERS[(i + 1) % len(_MARKERS)]

    fig, axes = _grid(3, sources, height=5.9)
    for j, src in enumerate(sources):
        g = d[(d['source'] == src) & (d['label'] == arm)]
        # every seed is drawn on the full swept grid, so a seed that detected
        # nothing at some x breaks its line there rather than having a segment
        # drawn straight across the strengths it missed
        grid = np.sort(g[x].unique())
        for i, metric in enumerate(('ppv', 'vol_fp')):
            for _, sub in g.groupby('seed'):
                s = sub.groupby(x)[metric].mean().reindex(grid)
                axes[i, j].plot(grid, s.values, color=teal, lw=0.5,
                                alpha=0.3, zorder=2)
            med = g.groupby(x)[metric].median()
            axes[i, j].plot(med.index, med.values, color=teal, lw=_LW_LEAD,
                            zorder=4)
        axes[0, j].set_ylim(0, 1.03)
        _vol_axis(axes[1, j])

        for lab in [arm] + others:
            sub = d[(d['source'] == src) & (d['label'] == lab)]
            if len(sub):
                _rate_line(axes[2, j], sub, x, sub['ppv'] < _PPV_FLOOR,
                           style[lab])
        axes[2, j].set_ylim(0, 103)

    _dress(axes, sources, x=x, mark_default=True,
           row_labels=['PPV\n(per seed, median heavy)',
                       'false volume\n($\\times$ effect volume)',
                       f'trials at PPV $<$ {_PPV_FLOOR:g} (%)'])
    handles = [Line2D([], [], color=teal, lw=0.7, alpha=0.5,
                      label='GLOW, one seed'),
               _proxy('GLOW', dict(style[arm], marker=None))]
    handles += [_proxy(lab, style[lab]) for lab in others]
    _legend(fig, handles, ncol=len(handles))
    _save(fig, out, stem)


def fig_seed_pareto(df, out, *, arm: str = REPORTED_GLOW_LABEL,
                    llr_min: float = None,
                    stem: str = 'seed_pareto') -> None:
    """Rank the seeds by the false volume each contributes, and split it.

    Answers how concentrated the precision failure is. Every seed's whole
    sweep is pooled into one bar, given as its share of the source's total
    false volume so both panels read on one scale, and the bar is split by the
    mechanism of the two the volume came from. The cumulative share on the
    right axis is the same data as a Lorenz curve: a diagonal would mean every
    geometry is equally hard, and a steep rise means a handful of planted
    extents are the whole result.

    Args:
        df: a tidy_pred_decomp frame.
        out (pathlib.Path): directory to write into.
        arm (str): the GLOW recipe label to rank.
        llr_min (float | None): restrict to effects at least this strong;
            None pools the whole sweep.
        stem (str): output filename stem.
    """
    d = _decomp_cols(df, arm=arm)
    if llr_min is not None:
        d = d[pd.to_numeric(d['effect_llr'], errors='coerce') >= llr_min]
    if d.empty:
        print(f'  (no {arm} rows to rank - skipping {stem})')
        return
    sources = _sources(d)

    fig, axes = _grid(1, sources, height=3.0)
    for j, src in enumerate(sources):
        ax = axes[0, j]
        t = (d[d['source'] == src].groupby('seed')[['leaked', 'spurious']]
             .sum())
        t['fp'] = t['leaked'] + t['spurious']
        t = t.sort_values('fp', ascending=False)
        pct = 100 * t / t['fp'].sum()
        rank = np.arange(1, len(t) + 1)

        ax.bar(rank, pct['leaked'], width=0.86, color=_COLOR_LEAK, lw=0,
               zorder=2)
        ax.bar(rank, pct['spurious'], width=0.86, bottom=pct['leaked'],
               color=_COLOR_SPUR, lw=0, zorder=2)

        cum = ax.twinx()
        cum.patch.set_visible(False)
        cum.plot(rank, pct['fp'].cumsum(), color='#555555', lw=1.2, zorder=3)
        cum.set_ylim(0, 101)
        cum.spines['right'].set_visible(True)
        cum.spines['top'].set_visible(False)
        if j == len(sources) - 1:
            cum.set_ylabel('cumulative (%)')
        else:
            cum.set_yticklabels([])

        # a rule against the cumulative curve, not the bars, so it is drawn
        # on the rank axis the two share rather than on either y
        n_half = int((pct['fp'].cumsum() < 50).sum()) + 1
        ax.axvline(n_half, ls=':', lw=0.8, color='#999999', zorder=1)
        ax.set_xlim(0.3, len(t) + 0.7)
        # the share is scale-free, so name the volume it is a share of
        ax.text(0.97, 0.72, f'{n_half} of {len(t)} seeds carry half\n'
                            f"of {t['fp'].sum() / 1e3:.0f}k false voxels",
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                color='#555555')

    _dress(axes, sources, x='seed_rank', log_x=False,
           row_labels=['share of all false volume (%)'])
    _legend(fig, [_proxy('over-inclusion onto a true region',
                         {'color': _COLOR_LEAK, 'ls': '-', 'lw': 4}),
                  _proxy('spurious region, no effect voxels',
                         {'color': _COLOR_SPUR, 'ls': '-', 'lw': 4}),
                  _proxy('cumulative share',
                         {'color': '#555555', 'ls': '-', 'lw': 1.2})])
    _save(fig, out, stem)


def fig_runtime(df, out, *, fit_decades: float = 1.0) -> None:
    """Draw wall time against voxel count, annotating each method's slope.

    The reader's question on a log-log cost plot is the exponent, so it is
    fitted and printed rather than left to be eyeballed. The fit runs over the
    largest fit_decades of voxel count only: at small crops a fixed startup
    cost dominates and flattens the curve, so a whole-range fit understates
    the asymptotic scaling that the claim is about. The legend says which
    range it used, since the number is meaningless without it.

    Args:
        df: tidy_runtime frame (label / x / time_sec).
        out (pathlib.Path): directory to write into.
        fit_decades (float): decades of x, counted down from the largest, the
            slope is fitted over.
    """
    df = _numeric(df, ['x', 'time_sec'])
    labels = [c for c in _BASELINE_ORDER
              if c in set(df['label'].dropna().unique())]
    labels = [c for c in sorted(df['label'].dropna().unique())
              if str(c).startswith('GLOW')] + labels
    style = method_style(labels)
    for i, lab in enumerate(labels):
        style[lab]['marker'] = _MARKERS[i % len(_MARKERS)]

    fig, ax = plt.subplots(figsize=(_TEXTWIDTH_IN * 0.66,
                                    _TEXTWIDTH_IN * 0.44))
    handles = []
    for lab in labels:
        sub = df[df['label'] == lab]
        grp = sub.groupby('x')['time_sec']
        med = grp.median() / 60.0
        lo, hi = grp.min() / 60.0, grp.max() / 60.0
        st = style[lab]
        ax.fill_between(med.index, lo.values, hi.values, color=st['color'],
                        alpha=0.15, lw=0, zorder=1)
        ax.plot(med.index, med.values, color=st['color'], ls=st['ls'],
                lw=st['lw'], marker=st['marker'], ms=3.6, zorder=3)

        xv, yv = med.index.values.astype(float), med.values.astype(float)
        tail = xv >= xv.max() / 10 ** fit_decades
        slope = (np.polyfit(np.log10(xv[tail]), np.log10(yv[tail]), 1)[0]
                 if tail.sum() > 1 else float('nan'))
        name = 'GLOW' if str(lab).startswith('GLOW') else str(lab)
        handles.append(Line2D([], [], color=st['color'], ls=st['ls'],
                              lw=st['lw'], marker=st['marker'], ms=3.6,
                              label=f'{name}  (slope {slope:.2f})'))

    ax.set(xscale='log', yscale='log',
           xlabel=_X_LABEL['num_vox'], ylabel='wall time (min)')
    ax.grid(True, which='both')
    ax.set_axisbelow(True)
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=8,
              title=f'slope over top {fit_decades:g} decade',
              title_fontsize=7.5)
    _save(fig, out, 'runtime')


def fig_appendix_grids(frames: dict, out) -> None:
    """Draw the full metric grids the main text demotes, one page per cache.

    Fairness, not compression: the main-text figures show the deciding metric,
    and every metric the benchmark scores stays available here under the same
    grammar (source columns, metric rows, teal for GLOW).

    Args:
        frames (dict): stem -> (frame, x, metrics, hue) for each demoted grid;
            hue is the column one series is drawn per.
        out (pathlib.Path): directory to write into.
    """
    for stem, (df, x, metrics, hue) in frames.items():
        metrics = list(metrics)
        d = _numeric(df, [x, *metrics])
        sources = _sources(d)
        if not sources:
            continue
        labels = sorted(d[hue].dropna().unique().tolist())
        chosen = (_SEGMENT_CHOSEN if hue == 'label' and _SEGMENT_CHOSEN
                  in labels else
                  _PRUNE_CHOSEN if _PRUNE_CHOSEN in labels else None)
        style = ladder_style(labels, chosen=chosen)
        # the ladder puts the shipped setting first, so the legend reads
        # darkest to lightest in the order the colours were assigned
        order = ([chosen] + [l for l in labels if l != chosen]
                 if chosen in labels else labels)
        for i, lab in enumerate(order):
            style[lab]['marker'] = _MARKERS[i % len(_MARKERS)]

        fig, axes = _grid(len(metrics), sources,
                          height=1.55 * len(metrics) + 1.1)
        for j, src in enumerate(sources):
            dsrc = d[d['source'] == src]
            for i, metric in enumerate(metrics):
                for k, lab in enumerate(order):
                    sub = dsrc[dsrc[hue] == lab]
                    if len(sub):
                        # same-hue series: overlapping fills turn to mush, so
                        # the spread is a dodged bar per point instead
                        _band(axes[i, j], sub, x, metric, style[lab],
                              band=False, errbar=True,
                              dodge=0.05 * (k - (len(order) - 1) / 2))
                if metric in ('n_selected',):
                    axes[i, j].set_yscale('log')
                else:
                    axes[i, j].set_ylim(0, 1)
        _dress(axes, sources, x=x, mark_default=True,
               row_labels=[_METRIC_LABEL.get(m, m) for m in metrics])
        _legend(fig, [_proxy(str(l), style[l]) for l in order], y=0.03)
        _save(fig, out, stem)


def main(argv=None) -> None:
    """Draw every manuscript figure into one output directory.

    Reads the same provenance records plot.py does (make_csv.write_config_csv
    refreshes each cache's CSV on the way past), so this adds a view and never
    a re-run.

    Args:
        argv (list | None): CLI args; None reads sys.argv. Use --out to name
            the output directory under results/ (default _latest_paper), so a
            preview never overwrites the draft figures in _latest.
    """
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import glow._extra.benchmark as bench
    from . import make_csv

    parser = argparse.ArgumentParser(
        description='Draw the GLOW manuscript figures from the records.')
    parser.add_argument('--out', default='_latest_paper',
                        help='output directory name under results/')
    args = parser.parse_args(argv)

    out = bench.get_path_result() / args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f'writing paper figures to {out}')

    with plt.rc_context(_rc()):
        seg = tidy_segment(make_csv.write_config_csv('segment'), perc=None)
        pru = tidy_prune(make_csv.write_config_csv('prune'))
        print('\n=== tuning ===')
        fig_tuning(seg, pru, out)

        print('\n=== null ===')
        fig_null(tidy_run_ana(make_csv.write_config_csv('null')), out)

        # a sweep whose x is a handful of integers reads as points with bars,
        # not as a band over a continuum
        llr_raw = None
        for cache, stem, x, metrics, diff in (
                ('sweep_llr', 'sweep_llr', 'effect_llr',
                 ('dice', 'sens', 'ppv'), 'dice'),
                ('sweep_b', 'sweep_b', 'b', ('dice',), None),
                ('sweep_extent', 'sweep_extent', 'effect_perc',
                 ('dice',), None)):
            raw = make_csv.write_config_csv(cache)
            df = tidy_run_ana(raw)
            if df.empty:
                print(f'  (no records for {cache} - skipping)')
                continue
            print(f'\n=== {cache} ===')
            # the llr sweep is recorded at more than one b; the paper's
            # headline is b=1 and the rest is the sweep_b figure's job
            if cache == 'sweep_llr' and 'b' in df.columns:
                df = df[pd.to_numeric(df['b'], errors='coerce') == 1]
                llr_raw = raw
            discrete = x == 'b'
            fig_detection(df, out, stem, x=x, metrics=metrics,
                          diff_metric=diff, log_x=not discrete,
                          errbar=discrete, int_x=discrete)

        # the precision figures re-tidy the headline sweep's records: they
        # need the wide per-region block tidy_run_ana drops, and they explain
        # the PPV row that sweep_llr just drew
        if llr_raw is not None:
            print('\n=== precision decomposition ===')
            dec = tidy_pred_decomp(llr_raw)
            dec = dec[pd.to_numeric(dec['b'], errors='coerce') == 1]
            fig_ppv_decomp(dec, out)
            fig_ppv_seed(dec, out)
            fig_seed_pareto(dec, out)

        print('\n=== runtime ===')
        rt = tidy_runtime('runtime_num_vox',
                          make_csv.write_config_csv('runtime_num_vox'))
        if not rt.empty:
            fig_runtime(rt, out)

        print('\n=== appendix grids ===')
        fig_appendix_grids({
            'app_segment': (seg, 'effect_llr',
                            ('dice', 'sens', 'ppv'), 'label'),
            'app_prune': (pru[pru['label'].isin(_PRUNE_KEEP)], 'effect_llr',
                          ('dice', 'sens', 'ppv', 'n_selected'), 'label'),
        }, out)

    print(f'\ndone: {len(list(out.glob("*.pdf")))} figures in {out}')


if __name__ == '__main__':
    main()
