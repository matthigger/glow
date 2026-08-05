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
        for label in ('GLOW-GLM', 'GLOW-Focus', 'VBA'):
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
                # GLOW-GLM beats VBA, more so at the stronger effect
                tp_glow = int(40 + 400 * llr + rng.integers(0, 5))
                tp_vba = int(20 + 200 * llr + rng.integers(0, 5))
                rows.append(mk('GLOW-GLM', seed, llr,
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
    assert set(diff['method'].unique()) == {'GLOW'}
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
                rows.append(_wgn_row('GLOW-GLM', seed, llr, score, b=b))
                rows.append(_wgn_row('VBA', seed, llr, score, b=b))
                rows.append(_hcp_row('GLOW-GLM', seed, llr, score,
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
                # both climb with llr; GLOW-GLM reaches 0.5 Dice the earlier
                tp_glow = {0.01: 20, 0.03: 80, 0.1: 95}[llr]
                tp_vba = {0.01: 5, 0.03: 20, 0.1: 90}[llr]
                rows.append(_hcp_row('GLOW-GLM', seed, llr,
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
    glow = thr[thr['method'] == 'GLOW'][bcols].to_numpy()
    vba = thr[thr['method'] == 'VBA'][bcols].to_numpy()
    # thresholds fall inside the swept 0.01..0.1 range
    assert glow.size and vba.size
    assert ((glow > 0.01) & (glow < 0.1)).all()
    # GLOW-GLM reaches Dice 0.5 at a weaker effect than VBA
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
        {'run_inner_edge.in.cluster_mode': 'GLM Error',
         'run_inner_edge.out.curve': _edge_curve([50, 100], mz),
         'data_factory_wgn.in.seed': seed}
        for seed in range(2)])
    df = plot.tidy_inner_edge(raw)

    plot.plot_inner_edge('sweep_n_perm_inner', df, tmp_path)
    assert (tmp_path / 'sweep_n_perm_inner_threshold.pdf').exists()

    js = json.loads((tmp_path / 'sweep_n_perm_inner_threshold.json').read_text())
    # the threshold JSON: seed-median FWER critical value per (source, arm, m)
    assert js['threshold']['WGN']['GLOW'] == {'50': 5.0, '100': 5.5}


# ---------------------------------------------------------------------------
# max-z race retention (survivor race vs full cpu_perm)
# ---------------------------------------------------------------------------

def _race_curve(max_z_slow, max_z_race, reg_slow=None, reg_race=None) -> str:
    """Serialize one run_race_maxz leaf's curve JSON (row 0 observed).

    The regions default to agreeing on every outer perm, so a fixture only
    spells out the arg-max regions when it plants a leader miss.
    """
    n = len(max_z_slow)
    return json.dumps({'n_perm_inner': 1000, 'n_perm_inner_race': 15,
                       'p_keep_thresh': 1e-6, 'min_vox': 1,
                       'max_z_slow': [float(v) for v in max_z_slow],
                       'max_z_race': [float(v) for v in max_z_race],
                       'reg_slow': list(reg_slow or range(n)),
                       'reg_race': list(reg_race or range(n))})


def _race_row(cluster_mode, seed, curve, source='wgn') -> dict:
    """Build one run_race_maxz provenance row (leaf + its data ancestor)."""
    return {'run_race_maxz.in.cluster_mode': cluster_mode,
            'run_race_maxz.out.curve': curve,
            f'data_factory_{source}.in.seed': seed}


# two fits: one the race reproduces exactly, one where its max-z drifts past
# round-off on k=2 (same region) and displaces the leader on k=3.
_RACE_RAW = pd.DataFrame([
    _race_row('Focus', 0, _race_curve([10, 1, 2, 3], [10, 1, 2, 3])),
    _race_row('GLM Error', 1,
              _race_curve([5, 1, 2, 4], [5, 1, 2.5, 6],
                          reg_slow=[0, 1, 2, 3], reg_race=[0, 1, 2, 9]),
              source='hcp')])


def test_tidy_race_maxz_empty():
    assert plot.tidy_race_maxz(pd.DataFrame()).empty


def test_tidy_race_maxz_pairs_and_source():
    """One row per (fit, outer perm), carrying the diff and region agreement."""
    df = plot.tidy_race_maxz(_RACE_RAW)

    assert len(df) == 8
    assert set(df['source']) == {'WGN', 'HCP'}
    assert set(df['label']) == {'GLOW-Focus', 'GLOW-GLM'}
    # the planted miss: race 2 above the full null, on a disagreeing region
    miss = df[~df['same_reg']]
    assert list(miss['k']) == [3]
    assert miss['diff'].iloc[0] == pytest.approx(2.0)


def test_summarize_race_maxz_counts_agreement_and_region():
    """The summary counts the round-off agreements and the shared arg-maxes."""
    s = plot.summarize_race_maxz(plot.tidy_race_maxz(_RACE_RAW))

    # 8 pairs, 2 off by more than round-off (k=2 and the k=3 leader miss), of
    # which only the miss changes the arg-max region
    assert (s['n_pair'], s['n_fit']) == (8, 2)
    assert (s['n_agree'], s['n_same_reg']) == (6, 7)
    # both arms are kept: the check is over the segmentation objectives
    assert s['per_arm']['WGN']['GLOW-Focus'] == {'n_pair': 4, 'n_agree': 4,
                                                'n_same_reg': 4}
    assert s['per_arm']['HCP']['GLOW-GLM'] == {'n_pair': 4, 'n_agree': 2,
                                               'n_same_reg': 3}


def test_summarize_race_maxz_tolerance_is_relative():
    """A gap is judged against the magnitude, not on one absolute bar."""
    # the same absolute gap (1e-3) on a max-z of 1e6 and of 1.0
    raw = pd.DataFrame([
        _race_row('Focus', 0, _race_curve([1e6, 1.0], [1e6 + 1e-3, 1.001]))])
    s = plot.summarize_race_maxz(plot.tidy_race_maxz(raw))

    assert (s['n_pair'], s['n_agree'], s['n_same_reg']) == (2, 1, 2)


def test_summarize_race_maxz_empty():
    assert plot.summarize_race_maxz(pd.DataFrame()) == {}
    assert plot.format_race_maxz_summary({}) == ''


def test_plot_race_maxz_writes_figure_json_and_text(tmp_path):
    """The figure, the JSON, and the line the paper quotes all land."""
    df = plot.tidy_race_maxz(_RACE_RAW)

    plot.plot_race_maxz('race_maxz', df, tmp_path)
    assert (tmp_path / 'race_maxz_retention.pdf').exists()

    js = json.loads((tmp_path / 'race_maxz_retention.json').read_text())
    assert js['tol'] == plot._RACE_MAXZ_TOL
    assert js['retention']['n_same_reg'] == 7
    assert js['retention']['per_arm']['HCP']['GLOW-GLM']['n_agree'] == 2

    text = (tmp_path / 'race_maxz_retention.txt').read_text()
    assert '8 max-z pairs (2 fits)' in text
    assert '6 (75.000%) agree to float round-off' in text
    assert '7 (87.500%) take the max in the same region' in text


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


def _segment_row(mode, seed, effect_llr, tp, fp, tn, fn, source='wgn'):
    """Build one segment provenance row (run_segment leaf + its ancestors)."""
    row = {'run_segment.in.cluster_mode': mode,
           'effect_factory_single.in.effect_llr': effect_llr,
           **_flat_score('run_segment', tp, fp, tn, fn)}
    if source == 'wgn':
        row.update({'data_factory_wgn.in.b': 1,
                    'data_factory_wgn.in.seed': seed})
    else:
        row.update({'data_factory_hcp.in.hcp_feats': ['od'],
                    'data_factory_hcp.in.seed': seed})
    return row


def _prune_row(rule, seed, effect_llr, tp, fp, tn, fn, source='wgn',
               cluster_mode='Focus'):
    """Build one prune provenance row (run_prune leaf + its ancestors)."""
    row = {'run_prune.in.rule': rule,
           'run_prune.in.cluster_mode': cluster_mode,
           'effect_factory_single.in.effect_llr': effect_llr,
           **_flat_score('run_prune', tp, fp, tn, fn, extra={'n_selected': 1})}
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


def test_tidy_prune_label_prefixes_rule():
    """tidy_prune maps the recorded rule to the GLOW-<rule> method label."""
    df = plot.tidy_prune(pd.DataFrame([
        _prune_row('maxllr', 0, 0.03, 90, 5, 900, 5, source='wgn'),
        _prune_row('dp', 0, 0.03, 60, 40, 880, 20, source='hcp'),
    ]))
    assert set(df['label']) == {'GLOW-maxllr', 'GLOW-dp'}
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


def test_plot_prune_writes_one_figure_per_mode(tmp_path):
    """plot_prune writes one {label}_{mode}.pdf per Ward clustering mode."""
    rows = []
    for mode in ('Focus', 'GLM Error'):
        for rule in ('maxllr', 'greedy', 'dp'):
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
    """The filter drops the GLOW-Focus arm, not the Ward mode of that name.

    The surviving arm is relabelled GLOW. segment labels by Ward mode
    ('Focus') and prune by rule ('GLOW-greedy'), so neither family can be
    caught by a filter keyed on the analysis arm.
    """
    df = pd.DataFrame({'label': ['GLOW-GLM', 'GLOW-Focus', 'Focus',
                                 'GLOW-greedy', 'VBA']})
    assert list(plot._select_glow_arm(df)['label']) == [
        'GLOW', 'Focus', 'GLOW-greedy', 'VBA']


def test_plot_cache_drops_focus_arm(tmp_path):
    """A detection cache reports one arm, labelled GLOW, in figure + CSVs."""
    rows = []
    for llr in (0.01, 0.03, 0.1):
        for seed in range(3):
            tp = {0.01: 20, 0.03: 80, 0.1: 95}[llr]
            for label in ('GLOW-GLM', 'GLOW-Focus', 'VBA'):
                rows.append(_hcp_row(label, seed, llr,
                                     _score(tp, 10, 800, 100 - tp)))
    df = plot.tidy_run_ana(pd.DataFrame(rows))
    # the arm is present in the tidy frame; only the plot layer drops it
    assert 'GLOW-Focus' in set(df['label'])

    plot.plot_cache('sweep_llr', df, tmp_path)
    thr = pd.read_csv(tmp_path / 'sweep_llr_threshold.csv')
    # neither raw arm label reaches the output; the reported arm reads GLOW
    assert not {'GLOW-Focus', 'GLOW-GLM'} & set(thr['method'])
    assert 'GLOW' in set(thr['method'])
    diff = pd.read_csv(tmp_path / 'sweep_llr_diff.csv')
    assert set(diff['method'].unique()) == {'GLOW'}
