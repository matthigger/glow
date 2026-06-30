"""Tests for the run_ana benchmark plotters (tidy_run_ana + plot_cache)."""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

from glow._extra.benchmark import plot


def _score(tp, fp, tn, fn, min_pval=0.5, n_pred=1):
    """Build a run_ana.out.score cell with the given union-target counts."""
    return {'num_vox': tp + fp + tn + fn, 'min_pval': min_pval,
            'n_pred': n_pred,
            'target': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}}


def _wgn_row(label, seed, effect_llr, score, b=1, num_img=100):
    """Build one WGN provenance row (run_ana leaf + its ancestors)."""
    return {'run_ana.in.label': label, 'run_ana.time_sec': 1.0,
            'run_ana.out.score': score,
            'data_factory_wgn.in.b': b, 'data_factory_wgn.in.num_img': num_img,
            'data_factory_wgn.in.seed': seed,
            'effect_factory.in.effect_llr': effect_llr,
            'effect_factory.in.n_vox': 100}


def _hcp_row(label, seed, effect_llr, score, hcp_feats=('od',)):
    """Build one HCP provenance row (run_ana leaf + its ancestors)."""
    return {'run_ana.in.label': label, 'run_ana.time_sec': 1.0,
            'run_ana.out.score': score,
            'data_factory_hcp.in.hcp_feats': list(hcp_feats),
            'data_factory_hcp.in.seed': seed,
            'effect_factory.in.effect_llr': effect_llr,
            'effect_factory.in.n_vox': 100}


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
    assert not (tmp_path / 'null_metrics.pdf').exists()


def test_plot_cache_sweep_writes_metrics_and_diff(tmp_path):
    """A swept cache writes the metric grid, the diff grid, and diff CSV."""
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
    assert (tmp_path / 'sweep_llr_metrics.pdf').exists()
    assert (tmp_path / 'sweep_llr_diff.pdf').exists()
    assert (tmp_path / 'sweep_llr_diff.csv').exists()

    # the diff CSV carries one block per GLOW variant vs the best alternative
    diff = pd.read_csv(tmp_path / 'sweep_llr_diff.csv')
    assert set(diff['method'].unique()) == {'GLOW-Focus'}
    assert {'source', 'effect_llr', 'dice_diff', 'dice_win'}.issubset(
        diff.columns)
