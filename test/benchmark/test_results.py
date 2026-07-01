"""Tests for glow._extra.benchmark.results: per-CONFIG CSV slicing.

results recomputes each cache's membership from the current CONFIG by walking
the recorded provenance DAG forward from the cache's data cells -- no stored tag
(see results). These cover: config_leaf_keys selecting exactly a cache's leaves
down the null path and the planted (effect-matching) path; a cell shared by two
caches landing in both; that editing the config (dropping a cell) is reflected
at once (the staleness fix a tag could not give); and the CSV writer.

Run against a small monkeypatched CONFIG (the real grids run 15-1000 seeds); the
recorder folder is redirected to a tmp dir and fresh seeds keep every cell a
cache miss (so it records). Driving a cache just records; results walks the
records to recover which leaves are its.
"""
import random

import pandas as pd
import pytest

from glow._extra.benchmark import config, data, results
from glow._extra.benchmark.driver import drive
from glow._extra.benchmark.run import run_ana
from glow.analysis import AnalysisVBA
from glow.effect import ExtenterSphere


@pytest.fixture(autouse=True)
def _records_to_tmp(monkeypatch, tmp_path):
    """Mirror the shared recorder to a tmp dir and start from empty records."""
    monkeypatch.setattr(data.RECORDER, 'folder', tmp_path)
    data.RECORDER.records.clear()


def _data_cell(seed):
    return dict(source='wgn', shape=(5, 5, 5), b=2, num_img=16, a=2, seed=seed)


def _ana_grid():
    # two recipes, each named by its (ignored) label kwarg -- the leaf-grid
    # discriminator results matches on
    return [dict(ana=AnalysisVBA(n_perm_fwer=6), label='VBA-6'),
            dict(ana=AnalysisVBA(n_perm_fwer=7), label='VBA-7')]


def _effect_cell():
    return dict(effect_llr=0.05, extenter_cls=ExtenterSphere, n_vox=10, seed=0)


@pytest.fixture
def small_config(monkeypatch):
    """Three caches over fresh cells: two null caches sharing a data cell, plus
    one planted cache.

    cacheA runs only the shared cell; cacheB runs the shared cell plus one of
    its own (both null). planted runs its own data cell with an effect. Driving
    each just records (no tag); results walks the records for membership.
    """
    s_shared, s_only_b, s_plant = (random.randrange(2 ** 31) for _ in range(3))
    cfg = {
        'cacheA': ([_data_cell(s_shared)], [None], _ana_grid(), run_ana),
        'cacheB': ([_data_cell(s_shared), _data_cell(s_only_b)], [None],
                   _ana_grid(), run_ana),
        'planted': ([_data_cell(s_plant)], [_effect_cell()],
                    _ana_grid(), run_ana),
    }
    monkeypatch.setattr(config, 'CONFIG', cfg)
    for entry in cfg.values():
        drive(*entry)
    return cfg


def test_config_leaf_keys_selects_a_caches_leaves(small_config):
    # 1 data cell x null x 2 recipes = 2 real run_ana leaf records
    keys = results.config_leaf_keys('cacheA')
    assert len(keys) == 1 * 2
    for key in keys:
        assert data.RECORDER.records[key]['function'] == 'run_ana'


def test_config_results_df_counts_and_labels(small_config):
    a = results.config_results_df('cacheA')
    b = results.config_results_df('cacheB')
    assert len(a) == 1 * 2           # 1 data cell x 2 recipes
    assert len(b) == 2 * 2           # 2 data cells x 2 recipes
    # the method label rides through as the in.label column
    assert set(a['run_ana.in.label']) == {'VBA-6', 'VBA-7'}


def test_shared_cell_appears_in_both_caches(small_config):
    # the shared cell's leaves are reached by both caches' walks, so its rows
    # land in both frames; cacheB is a strict superset (its own extra cell)
    a = results.config_results_df('cacheA')
    b = results.config_results_df('cacheB')
    shared = set(a['run_ana.hash']) & set(b['run_ana.hash'])
    assert len(shared) == 2                                 # shared cell x 2
    assert set(a['run_ana.hash']) < set(b['run_ana.hash'])


def test_planted_cache_matches_via_effect_cell(small_config):
    # the planted cache is found down the data -> effect -> run_ana chain, so
    # its effect-cell match (not just the data anchor) is exercised
    keys = results.config_leaf_keys('planted')
    assert len(keys) == 1 * 1 * 2                  # 1 data x 1 effect x 2 recipes
    # disjoint from the null caches (own data cell, and a plant node on the way)
    assert not set(keys) & set(results.config_leaf_keys('cacheA'))
    # the plant reached run_ana: every leaf's target is the planted support
    df = results.config_results_df('planted')
    assert (df['run_ana.out.score.target.tp']
            + df['run_ana.out.score.target.fn'] == 10).all()


def test_membership_follows_config_edits(small_config, monkeypatch):
    # dropping cacheB's own data cell (the staleness case) drops its leaves at
    # once: the records still hold them, but the current config no longer asks
    assert len(results.config_leaf_keys('cacheB')) == 2 * 2
    shrunk = dict(small_config)
    shrunk['cacheB'] = small_config['cacheA']       # only the shared cell now
    monkeypatch.setattr(config, 'CONFIG', shrunk)
    assert len(results.config_leaf_keys('cacheB')) == 1 * 2


def test_write_config_csvs(small_config, tmp_path):
    out = tmp_path / 'out'
    written = results.write_config_csvs(out_dir=out)
    assert set(written) == {'cacheA', 'cacheB', 'planted'}
    a = pd.read_csv(written['cacheA'])
    assert len(a) == 2
    assert set(a['run_ana.in.label']) == {'VBA-6', 'VBA-7'}


def test_empty_cache_is_skipped(monkeypatch, tmp_path):
    # a cache whose cells have not run has no leaves -> no csv
    monkeypatch.setattr(config, 'CONFIG',
                        {'unrun': ([_data_cell(random.randrange(2 ** 31))],
                                   [None], _ana_grid(), run_ana)})
    written = results.write_config_csvs(out_dir=tmp_path / 'out')
    assert written == {}
