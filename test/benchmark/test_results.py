"""Tests for glow._extra.benchmark.results: per-CONFIG CSV slicing.

results re-attaches the CONFIG catalogue to the config-agnostic provenance frame
(RECORDER.flatten_to_df): for one cache name it recomputes the run_ana args-hashes
that name's grid produces (config_leaf_hashes) and keeps the matching rows. The
behaviour worth covering is the *sharing* -- a run_ana call in two caches' grids
is one record, so a shared row lands in both caches' frames, which is exactly why
membership is recomputed rather than stamped. These run against a small
monkeypatched CONFIG (the real grids run 15-1000 seeds). The recorder folder is
redirected to a tmp dir and fresh seeds keep every cell a cache miss, so each
cell really records (a hit would not).
"""
import random

import pandas as pd
import pytest

from glow._extra.benchmark import data, results
from glow._extra.benchmark.driver import drive
from glow._extra.benchmark.run import run_ana
from glow.analysis import AnalysisVBA


@pytest.fixture(autouse=True)
def _records_to_tmp(monkeypatch, tmp_path):
    """Mirror the shared recorder to a tmp dir and start from empty records."""
    monkeypatch.setattr(data.RECORDER, 'folder', tmp_path)
    data.RECORDER.records.clear()


def _data_cell(seed):
    return dict(source='wgn', shape=(4, 4, 4), b=2, num_img=8, a=2, seed=seed)


def _ana_grid():
    # two recipes, each tagged with its method label (the record-only kwarg)
    return [dict(ana=AnalysisVBA(n_perm_fwer=6), label='VBA-6'),
            dict(ana=AnalysisVBA(n_perm_fwer=7), label='VBA-7')]


@pytest.fixture
def small_config(monkeypatch):
    """Two caches sharing one data cell; null effect, two labelled recipes.

    cacheA runs only the shared cell; cacheB runs the shared cell plus one of
    its own. Driving both records every cell (fresh seeds -> all miss); the
    shared cell is one record reused by both caches.
    """
    s_shared, s_only_b = random.randrange(2 ** 31), random.randrange(2 ** 31)
    cfg = {
        'cacheA': ([_data_cell(s_shared)], [None], _ana_grid(), run_ana),
        'cacheB': ([_data_cell(s_shared), _data_cell(s_only_b)], [None],
                   _ana_grid(), run_ana),
    }
    monkeypatch.setattr(results, 'CONFIG', cfg)
    for entry in cfg.values():
        drive(*entry)
    return cfg


def test_config_leaf_hashes_match_recorded_rows(small_config):
    # the recomputed hashes are exactly run_ana records the cache's grid ran
    all_hashes = set(data.RECORDER.flatten_to_df()['run_ana.hash'])
    assert results.config_leaf_hashes('cacheA') <= all_hashes
    assert results.config_leaf_hashes('cacheB') <= all_hashes


def test_config_results_df_filters_and_labels(small_config):
    a = results.config_results_df('cacheA')
    b = results.config_results_df('cacheB')
    assert len(a) == 1 * 2           # 1 data cell x 2 recipes
    assert len(b) == 2 * 2           # 2 data cells x 2 recipes
    # the method label rides through as the in.label column
    assert set(a['run_ana.in.label']) == {'VBA-6', 'VBA-7'}


def test_shared_cell_appears_in_both_caches(small_config):
    # the shared cell is one record, so its rows land in BOTH frames -- the
    # many-to-one membership a single stamped "which cache" column could not hold
    a = results.config_results_df('cacheA')
    b = results.config_results_df('cacheB')
    shared = set(a['run_ana.hash']) & set(b['run_ana.hash'])
    assert len(shared) == 2                              # shared cell x 2 recipes
    assert set(a['run_ana.hash']) < set(b['run_ana.hash'])  # B is a superset


def test_write_config_csvs(small_config, tmp_path):
    out = tmp_path / 'out'
    written = results.write_config_csvs(out_dir=out)
    assert set(written) == {'cacheA', 'cacheB'}
    a = pd.read_csv(written['cacheA'])
    assert len(a) == 2
    assert set(a['run_ana.in.label']) == {'VBA-6', 'VBA-7'}


def test_empty_cache_is_skipped(monkeypatch, tmp_path):
    # a cache whose cells have not run contributes no rows -> no csv written
    monkeypatch.setattr(results, 'CONFIG',
                        {'unrun': ([_data_cell(random.randrange(2 ** 31))],
                                   [None], _ana_grid(), run_ana)})
    written = results.write_config_csvs(out_dir=tmp_path / 'out')
    assert written == {}
