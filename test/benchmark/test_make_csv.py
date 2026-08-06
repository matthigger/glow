"""Tests for glow._extra.benchmark.make_csv: the records -> CSV export.

Which leaves a cache owns is results' job (test_results); these cover what
make_csv adds on top: the per-cache frame (row counts, the recipe column, a
cell shared by two caches landing in both), the writer (one <name>.csv per
non-empty cache, an unrun cache skipped), write_config_csv handing back the
frame it wrote, and the CLI (fnmatch names, --out-dir).

Run against a small monkeypatched CONFIG (the real grids run 15-1000 seeds);
the recorder folder is redirected to a tmp dir and fresh seeds keep every cell
a cache miss (so it records).
"""
import random

import pandas as pd
import pytest

from glow._extra.benchmark import config, data, make_csv
from glow._extra.benchmark.driver import drive
from glow._extra.benchmark.run import run_ana
from glow.analysis import AnalysisVBA


@pytest.fixture(autouse=True)
def _records_to_tmp(monkeypatch, tmp_path):
    """Mirror the shared recorder to a tmp dir and start from empty records."""
    monkeypatch.setattr(data.RECORDER, 'folder', tmp_path)
    data.RECORDER.records.clear()


def _data_cell(seed):
    return dict(source='wgn', shape=(5, 5, 5), b=2, num_img=16, a=2, seed=seed)


def _ana_grid():
    # two recipes (distinct n_perm_fwer -> distinct ana -> distinct leaves)
    return [dict(ana=AnalysisVBA(n_perm_fwer=6)),
            dict(ana=AnalysisVBA(n_perm_fwer=7))]


@pytest.fixture
def small_config(monkeypatch):
    """Two driven null caches sharing a data cell, plus one never run.

    cacheA runs only the shared cell; cacheB runs the shared cell plus one of
    its own. unrun declares a cell nothing ever drove, so it has no leaves.
    """
    s_shared, s_only_b, s_unrun = (random.randrange(2 ** 31) for _ in range(3))
    cfg = {
        'cacheA': ([_data_cell(s_shared)], [None], _ana_grid(), run_ana),
        'cacheB': ([_data_cell(s_shared), _data_cell(s_only_b)], [None],
                   _ana_grid(), run_ana),
        'unrun': ([_data_cell(s_unrun)], [None], _ana_grid(), run_ana),
    }
    monkeypatch.setattr(config, 'CONFIG', cfg)
    for name in ('cacheA', 'cacheB'):
        drive(*cfg[name])
    return cfg


def test_config_results_df_counts_and_recipes(small_config):
    a = make_csv.config_results_df('cacheA')
    b = make_csv.config_results_df('cacheB')
    assert len(a) == 1 * 2           # 1 data cell x 2 recipes
    assert len(b) == 2 * 2           # 2 data cells x 2 recipes
    # the recipe rides through as the in.ana column (the label source), one per
    # recipe -- no label is stored (see run / plot)
    assert a['run_ana.in.ana'].nunique() == 2


def test_shared_cell_appears_in_both_frames(small_config):
    # the shared cell's leaves are reached by both caches' walks, so its rows
    # land in both frames; cacheB is a strict superset (its own extra cell)
    a = make_csv.config_results_df('cacheA')
    b = make_csv.config_results_df('cacheB')
    shared = set(a['run_ana.hash']) & set(b['run_ana.hash'])
    assert len(shared) == 2                                 # shared cell x 2
    assert set(a['run_ana.hash']) < set(b['run_ana.hash'])


def test_unrun_cache_has_an_empty_frame(small_config):
    assert make_csv.config_results_df('unrun').empty


def test_write_config_csvs(small_config, tmp_path):
    out = tmp_path / 'out'
    written = make_csv.write_config_csvs(out_dir=out)
    # the cache nothing drove has no leaves -> no csv (no empty file either)
    assert set(written) == {'cacheA', 'cacheB'}
    assert sorted(p.name for p in out.iterdir()) == ['cacheA.csv',
                                                     'cacheB.csv']
    a = pd.read_csv(written['cacheA'])
    assert len(a) == 2
    assert a['run_ana.in.ana'].nunique() == 2
    # what lands on disk is exactly the read path's frame
    assert set(a['run_ana.hash']) == set(
        make_csv.config_results_df('cacheA')['run_ana.hash'])


def test_write_config_csvs_honours_names(small_config, tmp_path):
    out = tmp_path / 'out'
    assert set(make_csv.write_config_csvs(out_dir=out, names=['cacheB'])) == \
        {'cacheB'}
    assert [p.name for p in out.iterdir()] == ['cacheB.csv']


def test_write_config_csv_returns_the_frame_it_wrote(small_config, tmp_path):
    # the read-and-refresh path plot takes: the frame handed back is the one
    # on disk, and it matches the plain read
    out = tmp_path / 'out'
    df = make_csv.write_config_csv('cacheB', out_dir=out)
    pd.testing.assert_frame_equal(df, make_csv.config_results_df('cacheB'))
    on_disk = pd.read_csv(out / 'cacheB.csv')
    assert len(on_disk) == len(df)
    assert list(on_disk.columns) == list(df.columns)


def test_write_config_csv_writes_nothing_for_an_unrun_cache(small_config,
                                                            tmp_path):
    out = tmp_path / 'out'
    assert make_csv.write_config_csv('unrun', out_dir=out).empty
    assert list(out.iterdir()) == []


# ---------------------------------------------------------------------------
# the CLI: python -m glow._extra.benchmark.make_csv [names...] [--out-dir]
# ---------------------------------------------------------------------------

def test_main_resolves_patterns_and_reports(small_config, tmp_path, capsys):
    out = tmp_path / 'out'
    make_csv.main(['cache*', '--out-dir', str(out)])
    assert sorted(p.name for p in out.iterdir()) == ['cacheA.csv',
                                                     'cacheB.csv']
    printed = capsys.readouterr().out
    assert 'cacheA' in printed and 'cacheB' in printed


def test_main_says_so_when_there_is_nothing_to_write(small_config, tmp_path,
                                                     capsys):
    make_csv.main(['unrun', '--out-dir', str(tmp_path / 'out')])
    assert 'no CSVs written' in capsys.readouterr().out


def test_main_rejects_an_unknown_name(small_config, tmp_path):
    # the sweep CLI's resolver, so a typo surfaces rather than exporting
    # nothing (see __main__.resolve_names)
    with pytest.raises(ValueError):
        make_csv.main(['nope', '--out-dir', str(tmp_path / 'out')])
