"""Tests for the benchmark plotters (detection + runtime)."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from glow._extra.benchmark import plot
from glow._extra.benchmark.config import (ana_kwargs_dict, CONFIG,
                                          REPORTED_GLOW_LABEL,
                                          REPORTED_GLOW_LABEL_LIST,
                                          RUN_STAT_LIST)
from glow.analysis import AnalysisVBA


# The catalogue's method names, taken by role rather than spelled out: a
# provenance row has to carry a recipe the read path can map back to a label
# (plot._LABEL_OF_ANA), and these tests need the reported GLOW arms and one
# voxel-wise arm -- not whichever names config carries this month, nor how many
# arms it ships. What the figures then CALL an arm is plot's own contract, so
# output assertions read the name off the same mapping the figures do
# (_ARM_LABEL) rather than spelling it out.
GLOW_LABEL = REPORTED_GLOW_LABEL
VBA_LABEL = next(label for label, ana in ana_kwargs_dict.items()
                 if isinstance(ana, AnalysisVBA) and not ana.tfce_flag)
GLOW_FIGURE = plot._ARM_LABEL[REPORTED_GLOW_LABEL]
# the other reported arm: the same recipe under the other Ward projection,
# which every detection figure carries beside the headline one
ARM_OTHER = next(lab for lab in REPORTED_GLOW_LABEL_LIST
                 if lab != REPORTED_GLOW_LABEL)
ARM_OTHER_FIGURE = plot._ARM_LABEL[ARM_OTHER]


def _score(tp, fp, tn, fn, min_pval=0.5, n_pred=1):
    """Build the recursed run_ana.out.score.* columns for one row.

    run_ana declares recurse_out_list=['score'], so flatten_to_df expands the
    score dict into one out.score.<path> column per scalar leaf; the fixtures
    mirror that flat layout rather than a single dict cell.
    """
    base = 'run_ana.out.score'
    return {f'{base}.num_vox': tp + fp + tn + fn, f'{base}.min_pval': min_pval,
            f'{base}.n_pred': n_pred,
            f'{base}.target.tp': tp, f'{base}.target.fp': fp,
            f'{base}.target.tn': tn, f'{base}.target.fn': fn}


def _wgn_row(label, seed, effect_llr, score, b=1, num_img=100):
    """Build one WGN provenance row (run_ana leaf + its ancestors)."""
    return {'run_ana.in.ana': repr(ana_kwargs_dict[label]),
            'run_ana.time_sec': 1.0, **score,
            'data_factory_wgn.in.b': b, 'data_factory_wgn.in.num_img': num_img,
            'data_factory_wgn.in.seed': seed,
            'effect_factory_single.in.effect_llr': effect_llr,
            'effect_factory_single.in.n_vox_frac': 0.1}


def _hcp_row(label, seed, effect_llr, score, hcp_feats=('od',)):
    """Build one HCP provenance row (run_ana leaf + its ancestors)."""
    return {'run_ana.in.ana': repr(ana_kwargs_dict[label]),
            'run_ana.time_sec': 1.0, **score,
            'data_factory_hcp.in.hcp_feats': list(hcp_feats),
            'data_factory_hcp.in.seed': seed,
            'effect_factory_single.in.effect_llr': effect_llr,
            'effect_factory_single.in.n_vox_frac': 0.1}


def test_tidy_run_ana_empty():
    """An empty frame in gives an empty frame out (no derived columns)."""
    assert plot.tidy_run_ana(pd.DataFrame()).empty


def test_tidy_run_ana_columns_and_source():
    """tidy_run_ana derives the canonical schema and the per-row source."""
    raw = pd.DataFrame([
        _wgn_row(GLOW_LABEL, seed=0, effect_llr=0.03,
                 score=_score(80, 10, 890, 20)),
        _hcp_row(VBA_LABEL, seed=1, effect_llr=0.03,
                 score=_score(40, 30, 870, 60), hcp_feats=('od', 'mk')),
    ])
    df = plot.tidy_run_ana(raw)

    # source read off which data_factory produced the row; HCP b is the
    # feature-subset length, and HCP carries no num_img
    assert list(df['source']) == ['WGN', 'HCP']
    assert list(df['b']) == [1, 2]
    assert df.loc[df['source'] == 'HCP', 'num_img'].isna().all()

    # metrics derive from the four counts (glow.mask.stats_from_counts)
    glow = df[df['label'] == GLOW_LABEL].iloc[0]
    assert glow['vox_effect'] == 100  # tp + fn
    assert glow['vox_total'] == 1000
    assert glow['effect_perc'] == pytest.approx(0.1)
    assert glow['dice'] == pytest.approx(2 * 80 / (2 * 80 + 10 + 20))
    assert {'dice', 'sens', 'ppv', 'spec'}.issubset(df.columns)


def test_infer_x():
    """_infer_x returns None for the null path, else the swept axis."""
    null = pd.DataFrame({'effect_llr': [np.nan, np.nan],
                         'b': [1, 1], 'num_img': [100, 100],
                         'effect_perc': [0.0, 0.0]})
    assert plot._infer_x(null) is None

    swept = pd.DataFrame({'effect_llr': [0.01, 0.1],
                          'b': [1, 1], 'num_img': [100, 100],
                          'effect_perc': [0.1, 0.1]})
    assert plot._infer_x(swept) == 'effect_llr'

    extent = pd.DataFrame({'effect_llr': [0.03, 0.03],
                           'b': [1, 1], 'num_img': [100, 100],
                           'effect_perc': [0.05, 0.2]})
    assert plot._infer_x(extent) == 'effect_perc'


def test_plot_cache_null_writes_calibration(tmp_path):
    """A null cache (no effect) writes only the faceted calibration figure."""
    rows = []
    for src in ('wgn', 'hcp'):
        mk = _wgn_row if src == 'wgn' else _hcp_row
        for label in (GLOW_LABEL, VBA_LABEL):
            for seed in range(3):
                rows.append(mk(label, seed=seed, effect_llr=np.nan,
                               score=_score(0, 0, 1000, 0, min_pval=0.3)))
    df = plot.tidy_run_ana(pd.DataFrame(rows))

    plot.plot_cache('null', df, tmp_path)
    assert (tmp_path / 'null_calibration.pdf').exists()
    assert not (tmp_path / 'null.pdf').exists()


def test_plot_cache_sweep_writes_grid_and_diff_csv(tmp_path):
    """A swept cache writes the stacked per-source figure and the diff CSV."""
    rng = np.random.default_rng(0)
    rows = []
    for src in ('wgn', 'hcp'):
        mk = _wgn_row if src == 'wgn' else _hcp_row
        for llr in (0.01, 0.1):
            for seed in range(4):
                # GLOW beats VBA, more so at the stronger effect
                tp_glow = int(40 + 400 * llr + rng.integers(0, 5))
                tp_vba = int(20 + 200 * llr + rng.integers(0, 5))
                rows.append(mk(GLOW_LABEL, seed, llr,
                               _score(tp_glow, 10, 800, 100 - tp_glow)))
                rows.append(mk(VBA_LABEL, seed, llr,
                               _score(tp_vba, 30, 800, 100 - tp_vba)))
    df = plot.tidy_run_ana(pd.DataFrame(rows))
    assert plot._infer_x(df) == 'effect_llr'

    plot.plot_cache('sweep_llr', df, tmp_path)
    # one stacked figure (HCP over WGN) plus the companion diff CSV
    assert (tmp_path / 'sweep_llr.pdf').exists()
    assert (tmp_path / 'sweep_llr_diff.csv').exists()

    # the diff CSV carries one block per GLOW variant vs the best
    # alternative; the llr sweep reports one variant, relabelled GLOW
    diff = pd.read_csv(tmp_path / 'sweep_llr_diff.csv')
    assert set(diff['method'].unique()) == {GLOW_FIGURE}
    assert {'source', 'effect_llr', 'dice_diff', 'dice_win'}.issubset(
        diff.columns)


def test_plot_cache_splits_on_secondary_axis(tmp_path):
    """A sweep that also varies b is drawn one figure-set per b value."""
    rng = np.random.default_rng(0)
    rows = []
    # the combined llr sweep: effect_llr is the x, b the secondary axis
    for b in (1, 2, 3):
        feats = ('od', 'fa', 'md')[:b]
        for llr in (0.01, 0.1):
            for seed in range(4):
                score = _score(50, 10, 800, 50)
                rows.append(_wgn_row(GLOW_LABEL, seed, llr, score, b=b))
                rows.append(_wgn_row(VBA_LABEL, seed, llr, score, b=b))
                rows.append(_hcp_row(GLOW_LABEL, seed, llr, score,
                                     hcp_feats=feats))
                rows.append(_hcp_row(VBA_LABEL, seed, llr, score,
                                     hcp_feats=feats))
    df = plot.tidy_run_ana(pd.DataFrame(rows))
    assert plot._infer_x(df) == 'effect_llr'

    plot.plot_cache('sweep_llr', df, tmp_path)
    # one figure per b, suffixed into the label; no un-split figure
    for b in (1, 2, 3):
        assert (tmp_path / f'sweep_llr_b{b}.pdf').exists()
    assert not (tmp_path / 'sweep_llr.pdf').exists()


def test_plot_cache_sweep_writes_threshold_csv(tmp_path):
    """A swept cache writes the absolute threshold table, a column per b."""
    rows = []
    for b in (1, 2, 3):
        feats = ('od', 'fa', 'md')[:b]
        for llr in (0.01, 0.03, 0.1):
            for seed in range(3):
                # both climb with llr; GLOW reaches 0.5 Dice the earlier
                tp_glow = {0.01: 20, 0.03: 80, 0.1: 95}[llr]
                tp_vba = {0.01: 5, 0.03: 20, 0.1: 90}[llr]
                rows.append(_hcp_row(GLOW_LABEL, seed, llr,
                                     _score(tp_glow, 10, 800, 100 - tp_glow),
                                     hcp_feats=feats))
                rows.append(_hcp_row(VBA_LABEL, seed, llr,
                                     _score(tp_vba, 10, 800, 100 - tp_vba),
                                     hcp_feats=feats))
    df = plot.tidy_run_ana(pd.DataFrame(rows))

    plot.plot_cache('sweep_llr', df, tmp_path)
    assert (tmp_path / 'sweep_llr_threshold.csv').exists()
    thr = pd.read_csv(tmp_path / 'sweep_llr_threshold.csv')
    # one threshold column per b value; entries are absolute effect_llr
    assert {'source', 'method'}.issubset(thr.columns)
    assert {'b=1', 'b=2', 'b=3'}.issubset(thr.columns)
    bcols = ['b=1', 'b=2', 'b=3']
    # the llr sweep reports one variant, relabelled GLOW
    glow = thr[thr['method'] == GLOW_FIGURE][bcols].to_numpy()
    vba = thr[thr['method'] == VBA_LABEL][bcols].to_numpy()
    # thresholds fall inside the swept 0.01..0.1 range
    assert glow.size and vba.size
    assert ((glow > 0.01) & (glow < 0.1)).all()
    # GLOW reaches Dice 0.5 at a weaker effect than VBA
    assert (glow < vba).all()


def test_crossing_interpolates_in_log_x():
    """_crossing geometric-interpolates the level, flagging the censored ends."""
    thr, status = plot._crossing(np.array([1.0, 10.0]),
                                 np.array([0.0, 1.0]), 0.5)
    assert status == 'ok'
    assert thr == pytest.approx(10 ** 0.5)  # geometric midpoint

    # censored ends: already above level at the weakest x, or never reaching it
    thr_lo, below = plot._crossing(np.array([1.0, 10.0]),
                                   np.array([0.6, 0.9]), 0.5)
    thr_hi, above = plot._crossing(np.array([1.0, 10.0]),
                                   np.array([0.1, 0.3]), 0.5)
    assert (below, above) == ('below', 'above')
    assert np.isnan(thr_lo) and np.isnan(thr_hi)


def test_threshold_absolute_crossing():
    """Entry = the absolute x at which mean Dice reaches level (log-interp)."""
    df = pd.DataFrame({
        'source': ['HCP'] * 6,
        'b': [1] * 6,
        'label': ['VBA'] * 3 + ['GLOW-Focus'] * 3,
        'effect_llr': [0.01, 0.03, 0.1] * 2,
        # VBA crosses between 0.03 and 0.1; GLOW between 0.01 and 0.03
        'dice': [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    })
    wide, status = plot.threshold_table(df, x='effect_llr', metric='dice',
                                        level=0.5)
    # b is constant here, so the single threshold column is named by x
    w = wide.set_index('method')
    assert w.loc['GLOW-Focus', 'effect_llr'] == pytest.approx(
        (0.01 * 0.03) ** 0.5)
    assert w.loc['VBA', 'effect_llr'] == pytest.approx((0.03 * 0.1) ** 0.5)
    assert status[('HCP', 'GLOW-Focus', 'effect_llr')] == 'ok'


def test_threshold_column_per_b():
    """A sweep varying b yields one absolute-threshold column per b value."""
    grid = [0.01, 0.02, 0.04, 0.08]
    # (glow_cross, vba_cross) per b: VBA falls a step further behind at b=2
    spec = {1: (0.02, 0.04), 2: (0.02, 0.08)}
    rows = []
    for b, (glow_cross, vba_cross) in spec.items():
        for lab, cross in (('GLOW-Focus', glow_cross), ('VBA', vba_cross)):
            for llr in grid:
                rows.append({'source': 'HCP', 'b': b, 'label': lab,
                             'effect_llr': llr,
                             'dice': 1.0 if llr >= cross else 0.0})
    wide, _ = plot.threshold_table(pd.DataFrame(rows), x='effect_llr')
    assert {'b=1', 'b=2'}.issubset(wide.columns)
    assert 'effect_llr' not in wide.columns  # b became the columns
    w = wide.set_index('method')
    # VBA needs a stronger effect at b=2 (crosses a grid step later)
    assert w.loc['VBA', 'b=2'] > w.loc['VBA', 'b=1']
    # GLOW-Focus reaches 0.5 at a weaker effect than VBA at both b
    assert w.loc['GLOW-Focus', 'b=1'] < w.loc['VBA', 'b=1']
    assert w.loc['GLOW-Focus', 'b=2'] < w.loc['VBA', 'b=2']


def test_threshold_unlabelled_returns_empty():
    """All-NaN labels (stale records) yield an empty table, not a KeyError."""
    df = pd.DataFrame({
        'source': ['WGN'] * 3,
        'b': [3] * 3,
        'label': [np.nan] * 3,
        'effect_llr': [0.01, 0.03, 0.1],
        'dice': [0.0, 0.5, 1.0],
    })
    wide, status = plot.threshold_table(df, x='effect_llr')
    assert wide.empty and status == {}


def test_threshold_needs_no_reference():
    """Absolute thresholds need no reference: a lone method still tables."""
    df = pd.DataFrame({
        'source': ['HCP'] * 3,
        'b': [1] * 3,
        'label': ['VBA'] * 3,
        'effect_llr': [0.01, 0.03, 0.1],
        'dice': [0.0, 1.0, 1.0],
    })
    wide, _ = plot.threshold_table(df, x='effect_llr')
    assert list(wide['method']) == ['VBA']
    assert wide.set_index('method').loc['VBA', 'effect_llr'] == pytest.approx(
        (0.01 * 0.03) ** 0.5)


def test_threshold_censored_status():
    """A method already above level at the weakest x is 'below'; one that never
    reaches it is 'above' -- both nan thresholds, split by the status map."""
    df = pd.DataFrame({
        'source': ['HCP'] * 6,
        'b': [1] * 6,
        'label': ['GLOW-Focus'] * 3 + ['VBA'] * 3,
        'effect_llr': [0.01, 0.03, 0.1] * 2,
        # GLOW already >= 0.5 at the weakest effect; VBA never reaches 0.5
        'dice': [0.6, 0.8, 0.9, 0.0, 0.1, 0.2],
    })
    wide, status = plot.threshold_table(df, x='effect_llr')
    w = wide.set_index('method')
    assert pd.isna(w.loc['GLOW-Focus', 'effect_llr'])
    assert pd.isna(w.loc['VBA', 'effect_llr'])
    assert status[('HCP', 'GLOW-Focus', 'effect_llr')] == 'below'
    assert status[('HCP', 'VBA', 'effect_llr')] == 'above'


# ---------------------------------------------------------------------------
# Runtime plotters
# ---------------------------------------------------------------------------

def _runtime_ana_row(label, seed, num_vox, time_sec, hcp_feats=('od',),
                     leaf='run_ana_time'):
    """One timed HCP row: the method rides in the recipe, num_vox is bare."""
    return {f'{leaf}.in.ana': repr(ana_kwargs_dict[label]),
            f'{leaf}.time_sec': time_sec,
            f'{leaf}.out.num_vox': num_vox,
            'data_factory_hcp.in.hcp_feats': list(hcp_feats),
            'data_factory_hcp.in.seed': seed}


def _runtime_1perm_row(label, seed, time_sec, num_vox=1000,
                       hcp_feats=('od',), **knobs):
    """One run_ana_time_1perm row: the swept count is an explicit input."""
    row = _runtime_ana_row(label, seed, num_vox, time_sec,
                           hcp_feats=hcp_feats, leaf='run_ana_time_1perm')
    row.update({f'run_ana_time_1perm.in.{k}': v for k, v in knobs.items()})
    return row


def test_tidy_runtime_empty():
    """An empty frame in gives an empty frame out."""
    assert plot.tidy_runtime('runtime_num_vox', pd.DataFrame()).empty


def test_tidy_runtime_run_ana_leaf():
    """A run_ana_time runtime cache reads num_vox bare and label off ana."""
    raw = pd.DataFrame([
        _runtime_ana_row(GLOW_LABEL, seed=0, num_vox=1000, time_sec=25.0),
        _runtime_ana_row(VBA_LABEL, seed=0, num_vox=224619, time_sec=480.0),
    ])
    df = plot.tidy_runtime('runtime_num_vox', raw)
    assert list(df['label']) == [GLOW_LABEL, VBA_LABEL]
    assert list(df['x']) == [1000, 224619]
    assert list(df['x_name'].unique()) == ['num_vox']
    assert list(df['time_sec']) == [25.0, 480.0]


def test_tidy_runtime_b_from_feature_count():
    """The b sweep's x is the HCP feature-subset length, not num_vox."""
    raw = pd.DataFrame([
        _runtime_1perm_row(GLOW_LABEL, 0, time_sec=2.0, hcp_feats=('od',)),
        _runtime_1perm_row(GLOW_LABEL, 0, time_sec=6.0,
                           hcp_feats=('od', 'mk', 'fa')),
    ])
    df = plot.tidy_runtime('runtime_1perm_b', raw)
    assert list(df['x_name'].unique()) == ['b']
    assert list(df['x']) == [1, 3]


def test_tidy_runtime_1perm_knob_from_leaf_input():
    """A swept permutation count is read off the 1perm leaf's own inputs."""
    raw = pd.DataFrame([
        _runtime_1perm_row(GLOW_LABEL, 0, time_sec=5.0, n_perm_fwer=1),
        _runtime_1perm_row(GLOW_LABEL, 0, time_sec=80.0, n_perm_fwer=16),
    ])
    df = plot.tidy_runtime('runtime_1perm_n_perm_fwer', raw)
    assert list(df['label']) == [GLOW_LABEL, GLOW_LABEL]
    assert list(df['x']) == [1, 16]
    assert list(df['x_name'].unique()) == ['n_perm_fwer']


def test_tidy_runtime_num_img_from_leaf_input():
    """num_img is cut by the leaf, so it is read from the leaf's inputs."""
    raw = pd.DataFrame([
        _runtime_1perm_row(GLOW_LABEL, seed=0, time_sec=1.0, num_img=10),
        _runtime_1perm_row(GLOW_LABEL, seed=1, time_sec=30.0, num_img=100),
    ])
    df = plot.tidy_runtime('runtime_1perm_nimg', raw)
    assert list(df['x_name'].unique()) == ['num_img']
    assert list(df['x']) == [10, 100]
    assert list(df['seed']) == [0, 1]


def test_plot_runtime_writes_figure(tmp_path):
    """plot_runtime writes one wall-time curve figure per runtime cache."""
    rows = []
    for label in (GLOW_LABEL, VBA_LABEL):
        for num_vox in (1000, 8000, 64000):
            for seed in range(3):
                rows.append(_runtime_ana_row(
                    label, seed, num_vox, time_sec=num_vox * 0.01 + seed))
    df = plot.tidy_runtime('runtime_num_vox', pd.DataFrame(rows))

    plot.plot_runtime('runtime_num_vox', df, tmp_path)
    assert (tmp_path / 'runtime_num_vox_runtime.pdf').exists()


# --- stat bake-off (stat cache) ------------------------------------------

def _stat_counts(dice, size=100):
    """Confusion counts (a stat_cell_df row) realising a target Dice.

    With tp = round(dice*size) and fp = fn = size - tp, Dice = tp / size, so a
    round-number Dice comes out exactly.
    """
    tp = round(dice * size)
    k = size - tp
    return {'num_vox': 4 * size, 'tp': tp, 'fp': k, 'fn': k,
            'tn': 4 * size - tp - 2 * k}


def _stat_rows(cell, dice_map=None, default=0.4, drop=(), size=100):
    """One stat_cell_df row per RUN_STAT_LIST variant for a planted cell.

    dice_map overrides the Dice of specific (method, stat, zt) variants; the
    rest take default. drop omits variants (a partial, interrupted cell). size
    sets the Dice granularity (1 / size), so finer targets need a bigger size.
    """
    rows = []
    for spec in RUN_STAT_LIST:
        method, zt, stat = plot._STAT_VARIANT[(repr(spec['ana']),
                                               spec['stat_name'])]
        if (method, stat, zt) in drop:
            continue
        dice = (dice_map or {}).get((method, stat, zt), default)
        rows.append({'cell': cell, 'ana': repr(spec['ana']),
                     'stat_name': spec['stat_name'],
                     **_stat_counts(dice, size=size)})
    return rows


def test_tidy_stat_empty():
    """An empty frame in gives an empty frame out."""
    assert plot.tidy_stat(pd.DataFrame()).empty


def test_tidy_stat_recovers_variant_and_dice():
    """tidy_stat recovers (method, stat, zt) from the recipe and derives Dice."""
    df = plot.tidy_stat(pd.DataFrame(_stat_rows('cellA')))
    # every RUN_STAT_LIST variant is mapped (nothing dropped)
    assert set(df['method']) == {'VBA', 'VBA-TFCE', 'CET'}
    assert set(df['stat']) == {'llr', 'wilks', 'pillai', 'hotel_tr',
                               'roys_root'}
    assert set(df['zt']) == {'raw', 'z'}
    assert len(df) == len(RUN_STAT_LIST)
    assert df['dice'].eq(0.4).all()


def test_stat_tables_balanced_panel_and_range():
    """Partial cells drop out; ties vs a decisive winner read per arm."""
    # two complete cells: VBA / CET all-tie, VBA-TFCE decided by wilks/pillai
    win = {(m, s, zt): 0.6 for m in ['VBA-TFCE'] for zt in ['raw', 'z']
           for s in ['wilks', 'pillai']}
    rows = _stat_rows('cellA', dice_map=win) + _stat_rows('cellB', dice_map=win)
    # a third cell missing its last stat is not a full grid -> excluded
    rows += _stat_rows('cellC', drop=[('CET', 'roys_root', 'z')])
    t1, t2, meta = plot.stat_tables(plot.tidy_stat(pd.DataFrame(rows)))

    assert meta == {'n_cells': 2, 'n_dropped': 1, 'tol': 1e-9,
                    'decisive': 0.01}
    # both frames are indexed by (method, zt): the raw and z arms stay apart
    assert list(t1.index) == list(t2.index)
    assert list(t1.index) == [(m, zt) for m in plot._STAT_METHOD_ORDER
                              for zt in ('raw', 'z')]
    assert t1.loc[('VBA', 'raw'), 'pct_all_tie'] == 100.0
    assert t1.loc[('VBA', 'z'), 'pct_decisive'] == 0.0
    assert t1.loc[('VBA-TFCE', 'raw'), 'pct_decisive'] == 100.0
    for col in ('mean_range', 'median_range', 'max_range'):
        assert t1.loc[('VBA-TFCE', 'z'), col] == pytest.approx(0.2)
        assert t1.loc[('VBA', 'raw'), col] == pytest.approx(0.0)

    assert t2.loc[('VBA-TFCE', 'z'), 'wilks'] == pytest.approx(0.6)
    assert t2.loc[('VBA-TFCE', 'raw'), 'llr'] == pytest.approx(0.4)
    assert t2.loc[('VBA-TFCE', 'z'), 'gap'] == pytest.approx(0.2)
    assert t2.loc[('VBA', 'raw'), 'gap'] == pytest.approx(0.0)


def test_write_stat_tables_bolds_one_cell_even_when_near_tied(tmp_path):
    """A method's one bold lands on its max, however close the rest print."""
    # z-scored wilks wins; pillai trails by 0.002 and roys_root by 0.0004
    # (which prints 0.600 too); the raw arm sits 0.1 below throughout
    win = {('VBA-TFCE', 'wilks', 'z'): 0.6, ('VBA-TFCE', 'pillai', 'z'): 0.598,
           ('VBA-TFCE', 'roys_root', 'z'): 0.5996,
           ('VBA-TFCE', 'wilks', 'raw'): 0.5}
    df = plot.tidy_stat(pd.DataFrame(
        _stat_rows('cellA', dice_map=win, size=10000)))
    plot.write_stat_tables('stat', df, tmp_path)

    dice = (tmp_path / 'stat_dice.tex').read_text()
    rows = [r.split(' & ') for r in _tex_body(dice)]
    assert dice.count('\\textbf') == len(plot._STAT_METHOD_ORDER)
    # the bold sits in the z-scored half on wilks -- the raw block runs columns
    # 2-6, so z-scored wilks is column 8 -- and near-tied roys_root stays plain
    tfce = next(c for c in rows if c[0] == 'VBA-TFCE')
    assert tfce[7] == '\\textbf{0.600}'
    assert '\\textbf{0.598}' not in dice and '\\textbf{0.500}' not in dice


def test_stat_tables_max_range_is_the_worst_trial():
    """max_range reports the worst single trial, not the average one."""
    a = _stat_rows('cellA', dice_map={('CET', 'wilks', 'z'): 0.5})
    b = _stat_rows('cellB', dice_map={('CET', 'wilks', 'z'): 0.9})
    t1, _, _ = plot.stat_tables(plot.tidy_stat(pd.DataFrame(a + b)))

    row = t1.loc[('CET', 'z')]
    assert row['max_range'] == pytest.approx(0.5)
    assert row['mean_range'] == pytest.approx(0.3)
    assert row['median_range'] == pytest.approx(0.3)


def test_stat_tables_keeps_zt_arms_apart():
    """A stat winning only in the z arm does not bleed into the raw arm."""
    # wilks wins for z-scored VBA-TFCE only; its raw arm ties everywhere
    win = {('VBA-TFCE', 'wilks', 'z'): 0.9}
    df = plot.tidy_stat(pd.DataFrame(_stat_rows('cellA', dice_map=win)))
    t1, t2, _ = plot.stat_tables(df)

    assert t2.loc[('VBA-TFCE', 'z'), 'wilks'] == pytest.approx(0.9)
    assert t2.loc[('VBA-TFCE', 'raw'), 'wilks'] == pytest.approx(0.4)
    assert t2.loc[('VBA-TFCE', 'z'), 'gap'] == pytest.approx(0.5)
    assert t2.loc[('VBA-TFCE', 'raw'), 'gap'] == pytest.approx(0.0)
    assert t1.loc[('VBA-TFCE', 'z'), 'pct_decisive'] == 100.0
    assert t1.loc[('VBA-TFCE', 'raw'), 'pct_all_tie'] == 100.0


def test_stat_tables_warns_on_unequal_stat_trials():
    """A panel that scores the stats on unequal trial counts warns."""
    recs = [{'cell': 'A', 'method': m, 'stat': s, 'zt': zt, 'dice': 0.4}
            for m in plot._STAT_METHOD_ORDER for zt in ('raw', 'z')
            for s in plot._STAT_ORDER]
    # relabel one wilks trial as llr: cell A still has the full-grid row count
    # (so it survives the panel) but the stats no longer share a trial count
    df = pd.DataFrame(recs)
    df.loc[df.index[df['stat'] == 'wilks'][0], 'stat'] = 'llr'
    with pytest.warns(UserWarning, match='unequal trials per stat'):
        plot.stat_tables(df)


def _tex_body(tex):
    """The data rows of a written tabular, stripped, method rows only."""
    return [ln.strip() for ln in tex.splitlines()
            if ln.strip().split(' &')[0] in plot._STAT_METHOD_ORDER]


def test_write_stat_tables_writes_bare_tabular(tmp_path):
    """write_stat_tables emits two bare booktabs tabulars, one per table."""
    win = {(m, s, zt): 0.6 for m in ['VBA-TFCE'] for zt in ['raw', 'z']
           for s in ['wilks', 'pillai']}
    df = plot.tidy_stat(pd.DataFrame(
        _stat_rows('cellA', dice_map=win) + _stat_rows('cellB', dice_map=win)))
    plot.write_stat_tables('stat', df, tmp_path)

    dice = (tmp_path / 'stat_dice.tex').read_text()
    matters = (tmp_path / 'stat_matters.tex').read_text()
    # two narrow tabulars, one table each; the paper owns float/caption/label
    for tex in (dice, matters):
        assert tex.count('\\begin{tabular}') == 1
        assert '\\toprule' in tex
        assert '\\begin{table}' not in tex
        assert '\\caption' not in tex and '\\label' not in tex
        assert 'Trials' not in tex
    # the dice table glues a method's two arms into one banded, ruled row
    assert '\\begin{tabular}{l|rrrrr|rrrrr}' in dice
    assert '\\multicolumn{5}{c|}{raw}' in dice
    assert '\\multicolumn{5}{c}{z-scored}' in dice
    assert '\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}' in dice
    dice_rows = _tex_body(dice)
    assert len(dice_rows) == len(plot._STAT_METHOD_ORDER)
    assert all(r.count('&') == 10 for r in dice_rows)
    # the matters table stays one row per arm, labelled with its scaling
    matters_rows = _tex_body(matters)
    assert len(matters_rows) == 2 * len(plot._STAT_METHOD_ORDER)
    assert [r.split(' & ')[1] for r in matters_rows] == ['raw', 'z'] * 3
    assert all(r.count('&') == 5 for r in matters_rows)
    assert '\\cmidrule' not in matters
    # one bold per method: its best (scaling, stat) cell of the ten
    assert dice.count('\\textbf') == len(plot._STAT_METHOD_ORDER)
    assert all(r.count('\\textbf') == 1 for r in dice_rows)
    assert '\\textbf{0.600}' in dice
    # sizing a win is the matters table's job, and max range is not in it
    assert 'Gap' not in dice
    assert 'Max range' not in matters
    tfce_z = next(r for r in matters_rows if r.startswith('VBA-TFCE & z'))
    assert tfce_z.endswith('0.2000 & 0.2000 \\\\')


# ---------------------------------------------------------------------------
# flat-score caches (segment / prune): source x metric grid vs effect_llr
# ---------------------------------------------------------------------------

def _flat_score(leaf, tp, fp, tn, fn, extra=None):
    """Build the recursed {leaf}.out.score.* columns for one flat-score row.

    run_segment / run_prune return a flat {tp,fp,tn,fn} score (prune adds
    n_selected), so the columns sit directly under out.score, not out.score.target.
    """
    base = f'{leaf}.out.score'
    out = {f'{base}.tp': tp, f'{base}.fp': fp,
           f'{base}.tn': tn, f'{base}.fn': fn}
    if extra:
        out.update({f'{base}.{k}': v for k, v in extra.items()})
    return out


def _segment_row(mode, seed, effect_llr, tp, fp, tn, fn, source='wgn',
                 frac_segment=None):
    """Build one segment provenance row (run_segment leaf + its ancestors).

    frac_segment is the segment_perc half of the shared leaf: None is a
    whole-cohort leaf, which records no such input at all.
    """
    row = {'run_segment.in.cluster_mode': mode,
           'effect_factory_single.in.effect_llr': effect_llr,
           **_flat_score('run_segment', tp, fp, tn, fn)}
    if frac_segment is not None:
        row['run_segment.in.frac_segment'] = frac_segment
    if source == 'wgn':
        row.update({'data_factory_wgn.in.b': 1,
                    'data_factory_wgn.in.seed': seed})
    else:
        row.update({'data_factory_hcp.in.hcp_feats': ['od'],
                    'data_factory_hcp.in.seed': seed})
    return row


def _prune_row(rule, seed, effect_llr, tp, fp, tn, fn, source='wgn',
               cluster_mode='Focus', n_selected=1):
    """Build one prune provenance row (run_prune leaf + its ancestors)."""
    row = {'run_prune.in.rule': rule,
           'run_prune.in.cluster_mode': cluster_mode,
           'effect_factory_single.in.effect_llr': effect_llr,
           **_flat_score('run_prune', tp, fp, tn, fn,
                         extra={'n_selected': n_selected})}
    if source == 'wgn':
        row.update({'data_factory_wgn.in.b': 1,
                    'data_factory_wgn.in.seed': seed})
    else:
        row.update({'data_factory_hcp.in.hcp_feats': ['od'],
                    'data_factory_hcp.in.seed': seed})
    return row


def test_tidy_segment_empty():
    """An empty frame in gives an empty frame out."""
    assert plot.tidy_segment(pd.DataFrame()).empty


def test_tidy_segment_label_source_and_metrics():
    """tidy_segment reads the Ward mode as-is and the source off data_factory."""
    df = plot.tidy_segment(pd.DataFrame([
        _segment_row('Focus', 0, 0.03, 80, 10, 890, 20, source='wgn'),
        _segment_row('GLM Error', 1, 0.03, 40, 30, 870, 60, source='hcp'),
    ]))
    assert list(df['label']) == ['Focus', 'GLM Error']
    assert list(df['source']) == ['WGN', 'HCP']
    focus = df[df['label'] == 'Focus'].iloc[0]
    assert focus['dice'] == pytest.approx(2 * 80 / (2 * 80 + 10 + 20))
    assert {'dice', 'sens', 'ppv', 'spec'}.issubset(df.columns)


def test_tidy_segment_splits_the_shared_leaf_on_frac_segment():
    """segment keeps the whole-cohort leaves, segment_perc the fold ones.

    Both caches record run_segment on the same moderate-effect cells, so the
    record walk hands either cache a frame holding both.
    """
    raw = pd.DataFrame([
        _segment_row('Focus', 0, 0.03, 80, 10, 890, 20),
        _segment_row('Focus', 0, 0.03, 40, 30, 870, 60, frac_segment=0.3),
    ])
    whole = plot.tidy_segment(raw)
    assert list(whole['tp']) == [80]
    assert whole['frac_segment'].isna().all()

    perc = plot.tidy_segment(raw, perc=True)
    assert list(perc['tp']) == [40]
    assert list(perc['frac_segment']) == [0.3]


def test_tidy_prune_label_prefixes_rule():
    """tidy_prune maps the recorded rule to the GLOW-<rule> method label."""
    df = plot.tidy_prune(pd.DataFrame([
        _prune_row('single_max', 0, 0.03, 90, 5, 900, 5, source='wgn'),
        _prune_row('dp', 0, 0.03, 60, 40, 880, 20, source='hcp'),
    ]))
    assert set(df['label']) == {'GLOW-single_max', 'GLOW-dp'}
    assert list(df['source']) == ['WGN', 'HCP']


def test_tidy_prune_carries_cluster_mode():
    """tidy_prune rides the recorded Ward mode through as cluster_mode.

    A row missing the mode column (a legacy record predating the axis) reads
    back as the Focus default.
    """
    df = plot.tidy_prune(pd.DataFrame([
        _prune_row('greedy', 0, 0.03, 90, 5, 900, 5, cluster_mode='Focus'),
        _prune_row('greedy', 1, 0.03, 60, 40, 880, 20,
                   cluster_mode='GLM Error'),
    ]))
    assert set(df['cluster_mode']) == {'Focus', 'GLM Error'}

    legacy = _prune_row('greedy', 0, 0.03, 90, 5, 900, 5)
    del legacy['run_prune.in.cluster_mode']
    assert plot.tidy_prune(pd.DataFrame([legacy]))['cluster_mode'].iloc[0] \
        == 'Focus'


def test_plot_metric_grid_writes_figure(tmp_path):
    """plot_metric_grid writes one {label}.pdf for a flat-score cache."""
    rows = []
    for mode in ('Focus', 'GLM Error', 'Naive'):
        for source in ('wgn', 'hcp'):
            for effect_llr in (0.003, 0.03, 0.3):
                for seed in range(3):
                    rows.append(_segment_row(mode, seed, effect_llr,
                                             80, 10, 890, 20, source=source))
    df = plot.tidy_segment(pd.DataFrame(rows))
    plot.plot_metric_grid('segment', df, tmp_path)
    assert (tmp_path / 'segment.pdf').exists()


def test_plot_metric_grid_writes_fold_sweep_figure(tmp_path):
    """segment_perc draws the same grid against frac_segment, on a linear x."""
    rows = []
    for mode in ('Focus', 'GLM Error', 'Naive'):
        for source in ('wgn', 'hcp'):
            for frac in (0.1, 0.5, 0.9):
                for seed in range(3):
                    rows.append(_segment_row(mode, seed, 0.03, 80, 10, 890, 20,
                                             source=source, frac_segment=frac))
    df = plot.tidy_segment(pd.DataFrame(rows), perc=True)
    plot.plot_metric_grid('segment_perc', df, tmp_path, x='frac_segment',
                          log_x=False)
    assert (tmp_path / 'segment_perc.pdf').exists()


def _inner_row(n_perm_inner, seed, effect_llr, tp, fp, tn, fn, *,
               max_z_reg_idx=7, max_z_z=5.0, source='hcp'):
    """Build one inner-draw provenance row (run_inner_perm + its ancestors)."""
    leaf = 'run_inner_perm'
    row = {f'{leaf}.in.n_perm_inner': n_perm_inner,
           f'{leaf}.in.cluster_mode': 'Focus',
           'effect_factory_single.in.effect_llr': effect_llr,
           **_flat_score(leaf, tp, fp, tn, fn,
                         extra={'n_selected': 1, 'n_sig': 3,
                                'min_pval': 0.002,
                                'max_z.reg_idx': max_z_reg_idx,
                                'max_z.num_vox': 90,
                                'max_z.z': max_z_z,
                                'max_z.tp': tp, 'max_z.fp': fp,
                                'max_z.tn': tn, 'max_z.fn': fn})}
    if source == 'wgn':
        row.update({'data_factory_wgn.in.b': 1,
                    'data_factory_wgn.in.seed': seed})
    else:
        row.update({'data_factory_hcp.in.hcp_feats': ['od'],
                    'data_factory_hcp.in.seed': seed})
    return row


def test_tidy_inner_perm_empty():
    """An empty frame in gives an empty frame out."""
    assert plot.tidy_inner_perm(pd.DataFrame()).empty


def test_tidy_inner_perm_reads_the_count_and_the_max_z_block():
    df = plot.tidy_inner_perm(pd.DataFrame([
        _inner_row(25, 0, 0.03, 80, 10, 890, 20, max_z_z=4.9),
        _inner_row(1000, 0, 0.03, 80, 10, 890, 20, max_z_z=27.5),
    ]))
    assert list(df['n_perm_inner']) == [25, 1000]
    assert list(df['max_z_z']) == [4.9, 27.5]
    assert list(df['label']) == ['GLOW-Focus'] * 2
    # the max-z region's Dice comes off its own counts, like every metric
    assert df['max_z_dice'].iloc[0] == pytest.approx(
        2 * 80 / (2 * 80 + 10 + 20))


def test_max_z_agreement_scores_against_the_deepest_count():
    """The deepest count is the reference; a shared 'no region' agrees."""
    raw = pd.DataFrame([
        _inner_row(25, 0, 0.03, 80, 10, 890, 20, max_z_reg_idx=3),
        _inner_row(1000, 0, 0.03, 80, 10, 890, 20, max_z_reg_idx=7),
        _inner_row(25, 1, 0.03, 0, 0, 970, 30, max_z_reg_idx=None),
        _inner_row(1000, 1, 0.03, 0, 0, 970, 30, max_z_reg_idx=None),
    ])
    out = plot._max_z_agreement(plot.tidy_inner_perm(raw))
    assert list(out['agree']) == [False, True, True, True]


def test_plot_inner_perm_writes_both_figures_and_the_table(tmp_path):
    rows = []
    for effect_llr in (0.012, 0.03, 0.075):
        for n_perm_inner in (25, 250, 1000):
            for seed in range(3):
                rows.append(_inner_row(n_perm_inner, seed, effect_llr,
                                       80, 10, 890, 20,
                                       max_z_reg_idx=n_perm_inner))
    df = plot.tidy_inner_perm(pd.DataFrame(rows))
    plot.plot_inner_perm('sweep_n_perm_inner', df, tmp_path)
    assert (tmp_path / 'sweep_n_perm_inner.pdf').exists()
    assert (tmp_path / 'sweep_n_perm_inner_max_z.pdf').exists()
    table = pd.read_csv(tmp_path / 'sweep_n_perm_inner_max_z.csv')
    assert len(table) == 9
    assert set(table['n_seed']) == {3}
    assert (table['z_ceiling'] > 0).all()


def test_tidy_segment_keeps_both_halves_at_the_whole_cohort_ceiling():
    """perc=None keeps every leaf, the whole-cohort one at frac_segment 1.0.

    The llr figure reads the fold curves against the whole-cohort
    segmentation, so that leaf has to sit on the same axis: it segments every
    image, which is where the fold grid (stopping at 0.9) is heading.
    """
    raw = pd.DataFrame([
        _segment_row('Focus', 0, 0.03, 80, 10, 890, 20),
        _segment_row('Focus', 0, 0.03, 40, 30, 870, 60, frac_segment=0.3),
    ])
    both = plot.tidy_segment(raw, perc=None)
    assert list(both['tp']) == [80, 40]
    assert list(both['frac_segment']) == [plot.WHOLE_COHORT_FRAC, 0.3]


# the segment_perc_llr case is commented out with its catalogue entry
# (config.CONFIG); restore both together
@pytest.mark.parametrize('name, perc', [('segment', False)])
def test_segment_perc_reads_the_figure_off_the_catalogue_grids(name, perc):
    """Each segment cache's figure is derived from its own CONFIG grids.

    Both share the run_segment leaf and the Ward-mode labels, so nothing but
    the grids tells them apart: a fold cache is one whose leaf grid sets
    frac_segment, and one that also sweeps the strength is the both-halves llr
    figure. Read off CONFIG rather than a name list here, so a cache added
    later is classified rather than skipped -- including a fold sweep at one
    effect strength, which _segment_perc still returns True for.
    """
    _, effect_list, fnc_list, _ = CONFIG[name]
    assert plot._segment_perc(effect_list, fnc_list) is perc


def test_plot_segment_llr_writes_one_figure_per_mode(tmp_path):
    """The fold x llr figure is one {label}_{mode}.pdf per Ward mode.

    A curve per fold share within each, the whole-cohort leaves among them
    (frac_segment 1.0), so a mode's panels compare fold shares rather than
    modes -- the transpose of the segment_perc figure.
    """
    rows = []
    for mode in ('Focus', 'GLM Error', 'Naive'):
        for source in ('wgn', 'hcp'):
            for effect_llr in (0.003, 0.03, 0.3):
                for frac in (0.1, 0.5, 0.9, None):
                    for seed in range(3):
                        rows.append(_segment_row(
                            mode, seed, effect_llr, 80, 10, 890, 20,
                            source=source, frac_segment=frac))
    df = plot.tidy_segment(pd.DataFrame(rows), perc=None)
    plot.plot_segment_llr('segment_perc_llr', df, tmp_path)
    for mode in ('Focus', 'GLM_Error', 'Naive'):
        assert (tmp_path / f'segment_perc_llr_{mode}.pdf').exists()


def test_plot_segment_compare_writes_a_page_per_fold_largest_first(tmp_path):
    """The comparison is one multipage PDF, whole cohort first.

    The transpose of plot_segment_llr: the fold share is what a page holds
    fixed, the Ward modes are its lines. Pages descend so the whole-cohort
    segmentation opens the file, and the returned order is what a reader flips
    through (nothing here reads the PDF back).
    """
    rows = []
    for mode in ('Focus', 'GLM Error', 'Naive'):
        for source in ('wgn', 'hcp'):
            for effect_llr in (0.003, 0.03, 0.3):
                for frac in (0.1, 0.5, None):
                    for seed in range(3):
                        rows.append(_segment_row(
                            mode, seed, effect_llr, 80, 10, 890, 20,
                            source=source, frac_segment=frac))
    df = plot.tidy_segment(pd.DataFrame(rows), perc=None)

    pages = plot.plot_segment_compare('segment_perc_llr', df, tmp_path)
    assert pages == [plot.WHOLE_COHORT_FRAC, 0.5, 0.1]
    assert (tmp_path / 'segment_perc_llr_compare.pdf').exists()


def test_compare_page_puts_the_cohorts_side_by_side():
    """A page's panels are the sources, WGN left and HCP right.

    The transpose of the stacked grids, so it is the panel titles (not the row
    y-labels) that name the cohorts, and the metric moves to the y-label. Read
    off the figure rather than the PDF, which nothing here can parse back.
    """
    rows = []
    for mode in ('Focus', 'Naive'):
        for source in ('hcp', 'wgn'):  # built HCP-first, so order is the fig's
            for effect_llr in (0.003, 0.3):
                for seed in range(3):
                    rows.append(_segment_row(mode, seed, effect_llr,
                                             80, 10, 890, 20, source=source,
                                             frac_segment=0.5))
    df = plot.tidy_segment(pd.DataFrame(rows), perc=True)

    fig = plot._compare_page_fig(df, ('dice',), suptitle='half the images')
    axes = fig.get_axes()
    assert [ax.get_title() for ax in axes] == ['WGN', 'HCP']
    assert axes[0].get_ylabel().startswith('Dice')
    assert fig._suptitle.get_text() == 'half the images'
    plt.close(fig)


def test_plot_segment_compare_without_fold_rows_writes_nothing(tmp_path):
    """A frame of whole-cohort leaves alone has no fold axis to page over."""
    rows = [_segment_row('Focus', seed, 0.03, 80, 10, 890, 20)
            for seed in range(3)]
    df = plot.tidy_segment(pd.DataFrame(rows))

    assert plot.plot_segment_compare('segment', df, tmp_path) == []
    assert not (tmp_path / 'segment_compare.pdf').exists()


def test_plot_prune_writes_one_figure_per_mode(tmp_path):
    """plot_prune writes one {label}_{mode}.pdf per Ward clustering mode."""
    rows = []
    for mode in ('Focus', 'GLM Error'):
        for rule in ('single_max', 'greedy', 'dp'):
            for source in ('wgn', 'hcp'):
                for effect_llr in (0.003, 0.03, 0.3):
                    for seed in range(3):
                        rows.append(_prune_row(rule, seed, effect_llr,
                                               80, 10, 890, 20, source=source,
                                               cluster_mode=mode))
    df = plot.tidy_prune(pd.DataFrame(rows))
    plot.plot_prune('prune', df, tmp_path)
    assert (tmp_path / 'prune_Focus.pdf').exists()
    assert (tmp_path / 'prune_GLM_Error.pdf').exists()


# ---------------------------------------------------------------------------
# Reported GLOW arm
# ---------------------------------------------------------------------------

def test_select_glow_arm_spares_mode_and_rule_labels():
    """The rename reaches the analysis arms, not a mode or rule label.

    Each reported arm is renamed by its projection. segment labels by Ward
    mode ('Focus') and prune by rule ('GLOW-greedy'), so neither family can be
    caught by a filter keyed on the analysis arm.
    """
    df = pd.DataFrame({'label': [GLOW_LABEL, ARM_OTHER, 'Focus',
                                 'GLOW-greedy', 'VBA']})
    assert list(plot._select_glow_arm(df)['label']) == [
        GLOW_FIGURE, ARM_OTHER_FIGURE, 'Focus', 'GLOW-greedy', 'VBA']


def _both_arm_frame():
    """Build a one-source tidy frame carrying both GLOW arms plus VBA.

    The arms are injected into the tidy frame rather than round-tripped
    through a recipe: what the plot layer promises is about the labels handed
    to it, not about how many arms the catalogue ships, so the raw names here
    are plot's own vocabulary (_ARM_LABEL).
    """
    rows = []
    for llr in (0.01, 0.03, 0.1):
        for seed in range(3):
            tp = {0.01: 20, 0.03: 80, 0.1: 95}[llr]
            for label in (GLOW_LABEL, VBA_LABEL):
                rows.append(_hcp_row(label, seed, llr,
                                     _score(tp, 10, 800, 100 - tp)))
    tidy = plot.tidy_run_ana(pd.DataFrame(rows))
    glow = tidy[tidy['label'] == GLOW_LABEL]
    df = pd.concat([tidy[tidy['label'] != GLOW_LABEL],
                    glow.assign(label=GLOW_LABEL),
                    glow.assign(label=ARM_OTHER)], ignore_index=True)
    assert {GLOW_LABEL, ARM_OTHER} <= set(df['label'])
    return df


def test_plot_cache_names_the_arms_by_projection(tmp_path):
    """A detection cache reports both arms under their projection names."""
    df = _both_arm_frame()

    plot.plot_cache('sweep_extent', df, tmp_path)
    thr = pd.read_csv(tmp_path / 'sweep_extent_threshold.csv')
    # no recipe label reaches the output; each arm reads as its projection
    assert not {GLOW_LABEL, ARM_OTHER} & set(thr['method'])
    assert {GLOW_FIGURE, ARM_OTHER_FIGURE} <= set(thr['method'])
    # catalogue order survives: the arms lead, the voxel-wise method follows
    assert list(thr['method']) == [GLOW_FIGURE, ARM_OTHER_FIGURE, VBA_LABEL]
    # the head-to-head covers both arms, one block each against the same
    # best alternative
    diff = pd.read_csv(tmp_path / 'sweep_extent_diff.csv')
    assert set(diff['method'].unique()) == {GLOW_FIGURE, ARM_OTHER_FIGURE}


def test_plot_cache_glow_only_loses_the_diff_row(tmp_path):
    """With no non-GLOW method, the head-to-head row is not drawn at all.

    Nothing is left to diff the arms against (_has_diff): the figure is one
    band row per source and no diff CSV is written.
    """
    df = _both_arm_frame()
    df = df[df['label'] != VBA_LABEL]

    assert not plot._has_diff(df)
    plot.plot_cache('sweep_extent', df, tmp_path)
    assert (tmp_path / 'sweep_extent.pdf').exists()
    assert not (tmp_path / 'sweep_extent_diff.csv').exists()
    # the arms still reach the tables
    thr = pd.read_csv(tmp_path / 'sweep_extent_threshold.csv')
    assert {GLOW_FIGURE, ARM_OTHER_FIGURE} <= set(thr['method'])


def test_method_style_separates_the_arms():
    """Every recipe arm draws apart, and its figure label matches it.

    The recipes take one colour per Ward projection, and a renamed arm keeps
    the colour its recipe has, so a frame drawn before the rename and one
    drawn after agree on what a curve looks like.
    """
    arms = [lab for lab in plot._ARM_STYLE if lab in ana_kwargs_dict]
    style = plot._method_style(arms)
    assert len({(s['color'], s['ls']) for s in style.values()}) == len(arms)
    assert style[GLOW_LABEL] == plot._method_style([GLOW_FIGURE])[GLOW_FIGURE]

    plain = plot._method_style([GLOW_LABEL, VBA_LABEL])
    assert plain[VBA_LABEL] == {'color': plot.COLOR_ANALYSIS[VBA_LABEL],
                                'ls': '-'}


def test_write_table_txt_holds_the_figures_numbers(tmp_path):
    """The txt companion carries a matrix per metric, thresholds and wins."""
    df = _both_arm_frame()

    plot.write_table_txt('sweep_extent', df, x='effect_llr', out=tmp_path,
                         metrics=['dice', 'sens', 'ppv'],
                         one_label=GLOW_LABEL)
    text = (tmp_path / 'sweep_extent_tables.txt').read_text()
    assert 'x axis: LLR / |r| (effect_llr)' in text
    for title in ('Dice', 'Sensitivity', 'PPV (Precision)'):
        assert f'{title} (mean over seeds)' in text
    assert 'at Dice >= 0.5' in text
    assert f'win rate vs {GLOW_LABEL}' in text
    # a row per method, in catalogue order, under one shared swept-value header
    body = text[text.index('Dice (mean over seeds)'):]
    rows = body.splitlines()[1:5]
    assert rows[0].split()[0] == 'method'
    assert [r.split()[0] for r in rows[1:]] == [GLOW_LABEL, ARM_OTHER,
                                                VBA_LABEL]


def test_tidy_prune_carries_the_selection_size():
    """The rule's own region count rides through as n_selected.

    What a rule selects is the count the region figures read
    (plot_prune_regions); it is a leaf output like the confusion counts, not
    something derived from them.
    """
    df = plot.tidy_prune(pd.DataFrame([
        _prune_row('greedy', 0, 0.03, 90, 5, 900, 5, n_selected=1),
        _prune_row('dp', 0, 0.03, 60, 40, 880, 20, n_selected=347),
    ]))
    assert dict(zip(df['label'], df['n_selected'])) == {
        'GLOW-greedy': 1, 'GLOW-dp': 347}


def test_plot_prune_writes_the_region_count_figure(tmp_path):
    """plot_prune adds one region-count figure holding both Ward modes."""
    rows = []
    for mode in ('Focus', 'GLM Error'):
        for rule, n_sel in (('greedy', 2), ('dp', 400)):
            for source in ('wgn', 'hcp'):
                for effect_llr in (0.003, 0.03, 0.3):
                    for seed in range(3):
                        rows.append(_prune_row(rule, seed, effect_llr,
                                               80, 10, 890, 20, source=source,
                                               cluster_mode=mode,
                                               n_selected=n_sel))
    df = plot.tidy_prune(pd.DataFrame(rows))
    plot.plot_prune('prune', df, tmp_path)
    # one page for the counts, next to the per-mode metric grids
    assert (tmp_path / 'prune_regions.pdf').exists()
    assert (tmp_path / 'prune_Focus.pdf').exists()


def test_plot_cache_draws_no_region_counts(tmp_path):
    """A run_ana cache reports scores; region counts belong to plot_prune.

    What a selection rule hands back is the prune cache's question, read off
    its own n_selected (plot_prune_regions), so no detection figure carries a
    count column.
    """
    df = _both_arm_frame()

    plot.plot_cache('sweep_extent', df, tmp_path)
    assert not (tmp_path / 'sweep_extent_regions.pdf').exists()
    plain = (tmp_path / 'sweep_extent_tables.txt').read_text()
    assert 'Regions detected' not in plain


def test_cache_dir_is_one_directory_per_cache(tmp_path):
    """Each cache's output goes under its own directory, created on demand."""
    assert not (tmp_path / 'prune').exists()
    got = plot._cache_dir(tmp_path, 'prune')
    assert got == tmp_path / 'prune' and got.is_dir()
    # idempotent: a second cache leaves the first alone
    plot._cache_dir(tmp_path, 'sweep_llr')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['prune', 'sweep_llr']
