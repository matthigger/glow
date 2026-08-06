"""Tests for glow._extra.benchmark.results: per-CONFIG membership.

results recomputes each cache's membership from the current CONFIG by walking
the recorded provenance DAG forward from the cache's data cells -- no stored tag
(see results). These cover: config_leaf_keys selecting exactly a cache's leaves
down the null path and the planted (effect-matching) path; a cell shared by two
caches landing in both; that editing the config (dropping a cell) is reflected
at once (the staleness fix a tag could not give); and incomplete_cell_indices
(the AWS driver's local-records skip -- empty when a cache is fully recorded,
flagging an unrun cell down both the null and planted paths, a cell missing one
recipe's leaf, and judging completeness against a narrowed leaf grid when one
is passed). Exporting these leaves as CSVs is make_csv's (test_make_csv).

Run against a small monkeypatched CONFIG (the real grids run 15-1000 seeds); the
recorder folder is redirected to a tmp dir and fresh seeds keep every cell a
cache miss (so it records). Driving a cache just records; results walks the
records to recover which leaves are its.
"""
import random

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
    # two recipes (distinct n_perm_fwer -> distinct ana -> distinct leaf record)
    return [dict(ana=AnalysisVBA(n_perm_fwer=6)),
            dict(ana=AnalysisVBA(n_perm_fwer=7))]


def _effect_cell():
    return dict(effect_llr=0.05, extenter_cls=ExtenterSphere, n_vox_frac=0.1,
                seed=0)


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


def test_shared_cell_appears_in_both_caches(small_config):
    # the shared cell's leaves are reached by both caches' walks, so they are
    # selected by both; cacheB is a strict superset (its own extra cell)
    a = set(results.config_leaf_keys('cacheA'))
    b = set(results.config_leaf_keys('cacheB'))
    assert len(a & b) == 2                                  # shared cell x 2
    assert a < b


def test_planted_cache_matches_via_effect_cell(small_config):
    # the planted cache is found down the data -> effect -> run_ana chain, so
    # its effect-cell match (not just the data anchor) is exercised
    keys = results.config_leaf_keys('planted')
    assert len(keys) == 1 * 1 * 2                  # 1 data x 1 effect x 2 recipes
    # disjoint from the null caches (own data cell, and a plant node on the way)
    assert not set(keys) & set(results.config_leaf_keys('cacheA'))
    # the plant reached run_ana: every leaf's target is the planted support
    # (n_vox_frac=0.1 of the 5x5x5=125-voxel volume -> round(12.5)=12 voxels)
    for key in keys:
        target = data.RECORDER.records[key]['outputs']['score']['target']
        assert target['tp'] + target['fn'] == 12


def test_membership_follows_config_edits(small_config, monkeypatch):
    # dropping cacheB's own data cell (the staleness case) drops its leaves at
    # once: the records still hold them, but the current config no longer asks
    assert len(results.config_leaf_keys('cacheB')) == 2 * 2
    shrunk = dict(small_config)
    shrunk['cacheB'] = small_config['cacheA']       # only the shared cell now
    monkeypatch.setattr(config, 'CONFIG', shrunk)
    assert len(results.config_leaf_keys('cacheB')) == 1 * 2


def test_incomplete_empty_when_fully_recorded(small_config):
    # every cell of each cache was driven, so nothing is left to run
    for name in ('cacheA', 'cacheB', 'planted'):
        assert results.incomplete_cell_indices(name) == []


def test_incomplete_flags_unrun_null_cell(small_config, monkeypatch):
    # append a never-driven data cell to cacheB: only its index is returned
    # (the driven cells are complete), in grid order
    base = small_config['cacheB']
    grown = dict(small_config)
    grown['cacheB'] = ([*base[0], _data_cell(random.randrange(2 ** 31))],
                       *base[1:])
    monkeypatch.setattr(config, 'CONFIG', grown)
    assert results.incomplete_cell_indices('cacheB') == [len(base[0])]


def test_incomplete_flags_planted_unrun_cell(small_config, monkeypatch):
    # the planted cache: a never-driven cell is flagged down the
    # data -> effect -> run_ana chain, the driven cell stays complete
    base = small_config['planted']
    grown = dict(small_config)
    grown['planted'] = ([*base[0], _data_cell(random.randrange(2 ** 31))],
                        *base[1:])
    monkeypatch.setattr(config, 'CONFIG', grown)
    assert results.incomplete_cell_indices('planted') == [len(base[0])]


def test_incomplete_flags_cell_missing_a_recipe(monkeypatch):
    # a cell driven under one recipe is incomplete once the grid asks for two:
    # the second recipe's leaf is missing (per-fnc-kwargs matching, not a count)
    seed = random.randrange(2 ** 31)
    one = [dict(ana=AnalysisVBA(n_perm_fwer=6))]
    monkeypatch.setattr(config, 'CONFIG',
                        {'c': ([_data_cell(seed)], [None], one, run_ana)})
    drive(*config.CONFIG['c'])
    assert results.incomplete_cell_indices('c') == []
    monkeypatch.setattr(
        config, 'CONFIG',
        {'c': ([_data_cell(seed)], [None], _ana_grid(), run_ana)})
    assert results.incomplete_cell_indices('c') == [0]


def test_incomplete_judged_against_a_narrowed_grid(monkeypatch):
    # the rerun-one-recipe skip: completeness is judged against the grid handed
    # in, so a cell holding the narrowed grid's leaf reads complete though the
    # cache's own grid asks for two -- the recipe already computed is not resub-
    # mitted. The recipe that never ran is still flagged.
    seed = random.randrange(2 ** 31)
    ran, missing = [_ana_grid()[0]], [_ana_grid()[1]]
    monkeypatch.setattr(config, 'CONFIG',
                        {'c': ([_data_cell(seed)], [None], ran, run_ana)})
    drive(*config.CONFIG['c'])
    monkeypatch.setattr(
        config, 'CONFIG',
        {'c': ([_data_cell(seed)], [None], _ana_grid(), run_ana)})
    assert results.incomplete_cell_indices('c') == [0]
    assert results.incomplete_cell_indices('c', kwargs_fnc_list=ran) == []
    assert results.incomplete_cell_indices('c', kwargs_fnc_list=missing) == [0]


def test_incomplete_indexes_planted_cells(monkeypatch):
    # the unit is the planted cell (a data cell x one effect), so a cache with
    # two effects per data cell has two indices per data cell: growing the data
    # grid by one cell flags one index per effect, not a single data-cell index
    seed_a, seed_b = (random.randrange(2 ** 31) for _ in range(2))
    eff_two = [_effect_cell(), dict(_effect_cell(), effect_llr=0.1)]
    one = [dict(ana=AnalysisVBA(n_perm_fwer=6))]
    monkeypatch.setattr(config, 'CONFIG',
                        {'c': ([_data_cell(seed_a)], eff_two, one, run_ana)})
    drive(*config.CONFIG['c'])
    assert results.incomplete_cell_indices('c') == []
    # a fresh data cell adds two planted cells (its two effects): indices 2, 3
    monkeypatch.setattr(
        config, 'CONFIG',
        {'c': ([_data_cell(seed_a), _data_cell(seed_b)], eff_two, one,
               run_ana)})
    assert results.incomplete_cell_indices('c') == [2, 3]


def test_cell_leaf_uids_named_without_building_or_hashing(no_array_hashing):
    # a cell's leaf ids come from its kwargs alone: nothing is built, no record
    # is read, and no array is hashed (so the ids are the same on any machine)
    uids = results.cell_leaf_uids(_data_cell(7), _effect_cell(), _ana_grid(),
                                  run_ana)
    assert len(uids) == len(_ana_grid())
    assert len(set(uids)) == len(uids)
    # the null path names the clean exp as the parent instead of a plant
    assert results.cell_parent_uid(_data_cell(7), None) != \
        results.cell_parent_uid(_data_cell(7), _effect_cell())


# ---------------------------------------------------------------------------
# a leaf computed on another machine still counts for its cell
# ---------------------------------------------------------------------------

def test_divergent_experiment_leaf_still_completes_its_cell(monkeypatch):
    """The regression the declared uids exist for.

    Two CPUs plant the same effect to different last bits, so the leaves an AWS
    worker ships reference an Experiment whose content hash nothing local
    produces. Simulated here by rewriting the recorded content hashes after the
    fact: the legacy walk cannot link those leaves to the local ancestor, and
    the cell must still read complete on its declared uids -- otherwise the
    driver resubmits it on every sweep, forever.
    """
    seed = random.randrange(2 ** 31)
    one = [dict(ana=AnalysisVBA(n_perm_fwer=6))]
    monkeypatch.setattr(
        config, 'CONFIG',
        {'c': ([_data_cell(seed)], [_effect_cell()], one, run_ana)})
    drive(*config.CONFIG['c'])
    assert results.incomplete_cell_indices('c') == []

    # Diverge the ancestors' produced-exp hashes, leaving what the leaves
    # reference untouched: exactly the asymmetry a second CPU's planting
    # creates, where the local record says it produced exp A and the shipped
    # leaves consume exp B. The legacy join has nothing to match on.
    for rec in data.RECORDER.records.values():
        rec['output_hashes'] = {name: (h and f'divergent-{h}') for name, h
                                in rec.get('output_hashes', {}).items()}

    # the declared uids carry the cell...
    assert results.config_leaf_keys('c')
    assert results.incomplete_cell_indices('c') == []

    # ...and they are what carries it: with the recipe fields stripped, only
    # the legacy join is left and the finished cell reads unrun (the loop)
    stripped = {key: {k: v for k, v in rec.items()
                      if k not in ('uid', 'parents')}
                for key, rec in data.RECORDER.records.items()}
    monkeypatch.setattr(data.RECORDER, 'records', stripped)
    assert results.incomplete_cell_indices('c') == [0]
    monkeypatch.undo()

    # and a genuinely unrun cell is still flagged (the test is not vacuous)
    monkeypatch.setattr(
        config, 'CONFIG',
        {'c': ([_data_cell(seed), _data_cell(random.randrange(2 ** 31))],
               [_effect_cell()], one, run_ana)})
    assert results.incomplete_cell_indices('c') == [1]
