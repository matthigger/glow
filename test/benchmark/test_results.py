"""Tests for glow._extra.benchmark.results: per-CONFIG CSV slicing.

results re-attaches the CONFIG catalogue to the config-agnostic provenance frame
(RECORDER.flatten_to_df): for one cache name it selects the leaves the driver
*tagged* with that cache (config_leaf_keys, reading the record's ``configs``
list) and walks each up to its ancestors. The behaviour worth covering is the
*sharing* -- a run_ana call in two caches is one record carrying both names, so
a shared row lands in both caches' frames, which is exactly why membership is a
tag list and not a single stamped field. These run against a small monkeypatched
CONFIG (the real grids run 15-1000 seeds). The recorder folder is redirected to a
tmp dir and fresh seeds keep the first run of every cell a cache miss (so it
records); the *second* cache to want the shared cell hits the cache, and the
driver's above-the-cache tag is what adds its name to the existing record.
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
    # two recipes, each tagged with its method label (the ignored 'label' kwarg)
    return [dict(ana=AnalysisVBA(n_perm_fwer=6), label='VBA-6'),
            dict(ana=AnalysisVBA(n_perm_fwer=7), label='VBA-7')]


@pytest.fixture
def small_config(monkeypatch):
    """Two caches sharing one data cell; null effect, two labelled recipes.

    cacheA runs only the shared cell; cacheB runs the shared cell plus one of
    its own. Driving each under ``RECORDER.collecting(name)`` tags every leaf
    with its cache; the shared cell is one record that cacheB tags on a cache
    hit, so it ends up carrying both names.
    """
    s_shared, s_only_b = random.randrange(2 ** 31), random.randrange(2 ** 31)
    cfg = {
        'cacheA': ([_data_cell(s_shared)], [None], _ana_grid(), run_ana),
        'cacheB': ([_data_cell(s_shared), _data_cell(s_only_b)], [None],
                   _ana_grid(), run_ana),
    }
    monkeypatch.setattr(results, 'CONFIG', cfg)
    for name, entry in cfg.items():
        with data.RECORDER.collecting(name):
            drive(*entry)
    return cfg


def test_config_leaf_keys_select_tagged_records(small_config):
    # every selected key is a real record tagged with that cache
    for name in ('cacheA', 'cacheB'):
        keys = results.config_leaf_keys(name)
        assert keys                                  # non-empty
        for key in keys:
            assert name in data.RECORDER.records[key]['configs']


def test_config_results_df_filters_and_labels(small_config):
    a = results.config_results_df('cacheA')
    b = results.config_results_df('cacheB')
    assert len(a) == 1 * 2           # 1 data cell x 2 recipes
    assert len(b) == 2 * 2           # 2 data cells x 2 recipes
    # the method label rides through as the in.label column
    assert set(a['run_ana.in.label']) == {'VBA-6', 'VBA-7'}


def test_shared_cell_appears_in_both_caches(small_config):
    # the shared cell is one record carrying BOTH names, so its rows land in
    # both frames -- the many-to-one membership a single stamped column could
    # not hold
    a = results.config_results_df('cacheA')
    b = results.config_results_df('cacheB')
    shared = set(a['run_ana.hash']) & set(b['run_ana.hash'])
    assert len(shared) == 2                              # shared cell x 2 recipes
    assert set(a['run_ana.hash']) < set(b['run_ana.hash'])  # B is a superset
    # and that shared record really carries both cache names
    for key in shared:
        assert set(data.RECORDER.records[key]['configs']) == {'cacheA', 'cacheB'}


def test_write_config_csvs(small_config, tmp_path):
    out = tmp_path / 'out'
    written = results.write_config_csvs(out_dir=out)
    assert set(written) == {'cacheA', 'cacheB'}
    a = pd.read_csv(written['cacheA'])
    assert len(a) == 2
    assert set(a['run_ana.in.label']) == {'VBA-6', 'VBA-7'}


def test_empty_cache_is_skipped(monkeypatch, tmp_path):
    # a cache whose cells have not run contributes no tagged leaves -> no csv
    monkeypatch.setattr(results, 'CONFIG',
                        {'unrun': ([_data_cell(random.randrange(2 ** 31))],
                                   [None], _ana_grid(), run_ana)})
    written = results.write_config_csvs(out_dir=tmp_path / 'out')
    assert written == {}
