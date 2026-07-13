"""Tests for the benchmark plotters (detection + runtime)."""
import json

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from glow._extra.benchmark import plot
from glow._extra.benchmark.config import ana_kwargs_dict, RUN_STAT_LIST


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
        _wgn_row('GLOW-Focus', seed=0, effect_llr=0.03,
                 score=_score(80, 10, 890, 20)),
        _hcp_row('VBA', seed=1, effect_llr=0.03,
                 score=_score(40, 30, 870, 60), hcp_feats=('od', 'mk')),
    ])
    df = plot.tidy_run_ana(raw)

    # source read off which data_factory produced the row; HCP b is the
    # feature-subset length, and HCP carries no num_img
    assert list(df['source']) == ['WGN', 'HCP']
    assert list(df['b']) == [1, 2]
    assert df.loc[df['source'] == 'HCP', 'num_img'].isna().all()

    # metrics derive from the four counts (glow.mask.stats_from_counts)
    glow = df[df['label'] == 'GLOW-Focus'].iloc[0]
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
        for label in ('GLOW-Focus', 'VBA'):
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
                # GLOW-Focus beats VBA, more so at the stronger effect
                tp_glow = int(40 + 400 * llr + rng.integers(0, 5))
                tp_vba = int(20 + 200 * llr + rng.integers(0, 5))
                rows.append(mk('GLOW-Focus', seed, llr,
                               _score(tp_glow, 10, 800, 100 - tp_glow)))
                rows.append(mk('VBA', seed, llr,
                               _score(tp_vba, 30, 800, 100 - tp_vba)))
    df = plot.tidy_run_ana(pd.DataFrame(rows))
    assert plot._infer_x(df) == 'effect_llr'

    plot.plot_cache('sweep_llr', df, tmp_path)
    # one stacked figure (HCP over WGN) plus the companion diff CSV
    assert (tmp_path / 'sweep_llr.pdf').exists()
    assert (tmp_path / 'sweep_llr_diff.csv').exists()

    # the diff CSV carries one block per GLOW variant vs the best alternative
    diff = pd.read_csv(tmp_path / 'sweep_llr_diff.csv')
    assert set(diff['method'].unique()) == {'GLOW-Focus'}
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
                rows.append(_wgn_row('GLOW-Focus', seed, llr, score, b=b))
                rows.append(_wgn_row('VBA', seed, llr, score, b=b))
                rows.append(_hcp_row('GLOW-Focus', seed, llr, score,
                                     hcp_feats=feats))
                rows.append(_hcp_row('VBA', seed, llr, score, hcp_feats=feats))
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
                # both climb with llr; GLOW-Focus reaches 0.5 Dice the earlier
                tp_glow = {0.01: 20, 0.03: 80, 0.1: 95}[llr]
                tp_vba = {0.01: 5, 0.03: 20, 0.1: 90}[llr]
                rows.append(_hcp_row('GLOW-Focus', seed, llr,
                                     _score(tp_glow, 10, 800, 100 - tp_glow),
                                     hcp_feats=feats))
                rows.append(_hcp_row('VBA', seed, llr,
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
    glow = thr[thr['method'] == 'GLOW-Focus'][bcols].to_numpy()
    vba = thr[thr['method'] == 'VBA'][bcols].to_numpy()
    # thresholds fall inside the swept 0.01..0.1 range
    assert ((glow > 0.01) & (glow < 0.1)).all()
    # GLOW-Focus reaches Dice 0.5 at a weaker effect than VBA
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

def _runtime_ana_row(label, seed, num_vox, time_sec, hcp_feats=('od',)):
    """One run_ana_time runtime row (runtime / runtime_b): method in recipe."""
    return {'run_ana_time.in.ana': repr(ana_kwargs_dict[label]),
            'run_ana_time.time_sec': time_sec,
            'run_ana_time.out.num_vox': num_vox,
            'data_factory_hcp.in.hcp_feats': list(hcp_feats),
            'data_factory_hcp.in.seed': seed}


def _runtime_perm_row(label, seed, n_perm_fwer, time_sec, num_vox=1000):
    """One run_perm_fwer runtime row: method + knob are explicit inputs."""
    return {'run_perm_fwer.in.label': label,
            'run_perm_fwer.in.n_perm_fwer': n_perm_fwer,
            'run_perm_fwer.time_sec': time_sec,
            'run_perm_fwer.out.num_vox': num_vox,
            'data_factory_hcp.in.seed': seed}


def test_tidy_runtime_empty():
    """An empty frame in gives an empty frame out."""
    assert plot.tidy_runtime('runtime', pd.DataFrame()).empty


def test_tidy_runtime_run_ana_leaf():
    """A run_ana_time runtime cache reads num_vox bare and label off ana."""
    raw = pd.DataFrame([
        _runtime_ana_row('GLOW-Focus', seed=0, num_vox=1000, time_sec=25.0),
        _runtime_ana_row('VBA', seed=0, num_vox=224619, time_sec=480.0),
    ])
    df = plot.tidy_runtime('runtime', raw)
    assert list(df['label']) == ['GLOW-Focus', 'VBA']
    assert list(df['x']) == [1000, 224619]
    assert list(df['x_name'].unique()) == ['num_vox']
    assert list(df['time_sec']) == [25.0, 480.0]


def test_tidy_runtime_b_from_feature_count():
    """runtime_b's x is the HCP feature-subset length (b), not num_vox."""
    raw = pd.DataFrame([
        _runtime_ana_row('GLOW-Focus', 0, num_vox=1000, time_sec=2.0,
                         hcp_feats=('od',)),
        _runtime_ana_row('GLOW-Focus', 0, num_vox=1000, time_sec=6.0,
                         hcp_feats=('od', 'mk', 'fa')),
    ])
    df = plot.tidy_runtime('runtime_b', raw)
    assert list(df['x_name'].unique()) == ['b']
    assert list(df['x']) == [1, 3]


def test_tidy_runtime_timed_leaf():
    """A timed leaf reads the method + swept knob off its explicit inputs."""
    raw = pd.DataFrame([
        _runtime_perm_row('GLOW-Focus', seed=0, n_perm_fwer=50, time_sec=5.0),
        _runtime_perm_row('GLOW-GLM', seed=0, n_perm_fwer=800, time_sec=80.0),
    ])
    df = plot.tidy_runtime('runtime_n_perm_fwer', raw)
    assert list(df['label']) == ['GLOW-Focus', 'GLOW-GLM']
    assert list(df['x']) == [50, 800]
    assert list(df['x_name'].unique()) == ['n_perm_fwer']


def test_plot_runtime_writes_figure(tmp_path):
    """plot_runtime writes one wall-time curve figure per runtime cache."""
    rows = []
    for label in ('GLOW-Focus', 'VBA'):
        for num_vox in (1000, 8000, 64000):
            for seed in range(3):
                rows.append(_runtime_ana_row(
                    label, seed, num_vox, time_sec=num_vox * 0.01 + seed))
    df = plot.tidy_runtime('runtime', pd.DataFrame(rows))

    plot.plot_runtime('runtime', df, tmp_path)
    assert (tmp_path / 'runtime_runtime.pdf').exists()


# ---------------------------------------------------------------------------
# inner-perm edge (num_inner_perm convergence)
# ---------------------------------------------------------------------------

def _edge_curve(num_inner_perm, max_z_null) -> str:
    """Serialize one run_inner_edge leaf's recorded curve JSON (row 0 observed)."""
    return json.dumps({'num_inner_perm': list(num_inner_perm),
                       'max_z_null': [[float(v) for v in row]
                                      for row in max_z_null],
                       'min_vox': 1})


def test_tidy_inner_edge_empty():
    assert plot.tidy_inner_edge(pd.DataFrame()).empty


def test_tidy_inner_edge_threshold_and_source():
    # 3 outer perms (rows, k=0 observed), 2 grid points; with n=3 the FWER
    # critical value (alpha=0.05) is the column max (k = ceil(0.95*3) = 3)
    mz = [[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]]
    raw = pd.DataFrame([{'run_inner_edge.in.cluster_mode': 'Focus',
                         'run_inner_edge.out.curve': _edge_curve([50, 100], mz),
                         'data_factory_wgn.in.seed': 0}])
    df = plot.tidy_inner_edge(raw)

    assert set(df['num_inner_perm']) == {50, 100}
    assert list(df['source'].unique()) == ['WGN']
    assert list(df['label'].unique()) == ['GLOW-Focus']  # arm from cluster_mode
    at100 = df[df['num_inner_perm'] == 100].iloc[0]
    assert at100['threshold'] == 6.0        # max of column [6, 2, 4]
    assert at100['obs_max_z'] == 6.0        # row 0


def test_tidy_inner_edge_hcp_and_glm_arm():
    raw = pd.DataFrame([{'run_inner_edge.in.cluster_mode': 'GLM Error',
                         'run_inner_edge.out.curve':
                             _edge_curve([50], [[2.0], [1.0]]),
                         'data_factory_hcp.in.seed': 3}])
    df = plot.tidy_inner_edge(raw)
    assert list(df['source']) == ['HCP']
    assert list(df['label']) == ['GLOW-GLM']


def test_plot_inner_edge_writes_figure_and_json(tmp_path):
    mz = [[5.0, 5.5], [1.0, 2.0], [3.0, 4.0]]
    raw = pd.DataFrame([
        {'run_inner_edge.in.cluster_mode': 'Focus',
         'run_inner_edge.out.curve': _edge_curve([50, 100], mz),
         'data_factory_wgn.in.seed': seed}
        for seed in range(2)])
    df = plot.tidy_inner_edge(raw)

    plot.plot_inner_edge('sweep_n_perm_inner', df, tmp_path)
    assert (tmp_path / 'sweep_n_perm_inner_threshold.pdf').exists()

    js = json.loads((tmp_path / 'sweep_n_perm_inner_threshold.json').read_text())
    # the threshold JSON: seed-median FWER critical value per (source, arm, m)
    assert js['threshold']['WGN']['GLOW-Focus'] == {'50': 5.0, '100': 5.5}


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


def _stat_rows(cell, dice_map=None, default=0.4, drop=()):
    """One stat_cell_df row per RUN_STAT_LIST variant for a planted cell.

    dice_map overrides the Dice of specific (method, stat, zt) variants; the
    rest take default. drop omits variants (a partial, interrupted cell).
    """
    rows = []
    for spec in RUN_STAT_LIST:
        method, zt, stat = plot._STAT_VARIANT[(repr(spec['ana']),
                                               spec['stat_name'])]
        if (method, stat, zt) in drop:
            continue
        dice = (dice_map or {}).get((method, stat, zt), default)
        rows.append({'cell': cell, 'ana': repr(spec['ana']),
                     'stat_name': spec['stat_name'], **_stat_counts(dice)})
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


def test_stat_tables_balanced_panel_and_spread():
    """Partial cells drop out; ties vs a decisive winner read per method."""
    # two complete cells: VBA / CET all-tie, VBA-TFCE decided by wilks/pillai
    win = {(m, s, zt): 0.6 for m in ['VBA-TFCE'] for zt in ['raw', 'z']
           for s in ['wilks', 'pillai']}
    rows = _stat_rows('cellA', dice_map=win) + _stat_rows('cellB', dice_map=win)
    # a third cell missing its last stat is not a full grid -> excluded
    rows += _stat_rows('cellC', drop=[('CET', 'roys_root', 'z')])
    t1, t2, meta = plot.stat_tables(plot.tidy_stat(pd.DataFrame(rows)))

    assert meta == {'n_cells': 2, 'n_dropped': 1}
    assert t1.loc['VBA', 'pct_all_tie'] == 100.0
    assert t1.loc['VBA', 'pct_decisive'] == 0.0
    assert t1.loc['VBA-TFCE', 'pct_decisive'] == 100.0
    assert t1.loc['VBA-TFCE', 'mean_spread'] == pytest.approx(0.2)

    assert t2.loc['VBA-TFCE', 'wilks'] == pytest.approx(0.6)
    assert t2.loc['VBA-TFCE', 'llr'] == pytest.approx(0.4)
    assert t2.loc['VBA-TFCE', 'gap'] == pytest.approx(0.2)
    assert t2.loc['VBA', 'gap'] == pytest.approx(0.0)


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


def test_write_stat_tables_writes_bare_tabular(tmp_path):
    """write_stat_tables emits bare booktabs tabulars (no float/caption/label)."""
    win = {(m, s, zt): 0.6 for m in ['VBA-TFCE'] for zt in ['raw', 'z']
           for s in ['wilks', 'pillai']}
    df = plot.tidy_stat(pd.DataFrame(
        _stat_rows('cellA', dice_map=win) + _stat_rows('cellB', dice_map=win)))
    plot.write_stat_tables('stat', df, tmp_path)

    matters = (tmp_path / 'stat_matters.tex').read_text()
    dice = (tmp_path / 'stat_dice.tex').read_text()
    # bare tabular: the paper owns the float, caption and label
    for tex in (matters, dice):
        assert '\\begin{tabular}' in tex and '\\toprule' in tex
        assert '\\begin{table}' not in tex
        assert '\\caption' not in tex and '\\label' not in tex
    # Table 1 no longer carries a Trials column
    assert 'Trials' not in matters
    # the decisive winner is bolded; the tied-everywhere VBA row is not
    assert '\\textbf{0.600}' in dice
    vba_line = next(ln for ln in dice.splitlines()
                    if ln.strip().startswith('VBA &'))
    assert '\\textbf' not in vba_line
