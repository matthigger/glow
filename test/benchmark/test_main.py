"""Tests for glow._extra.benchmark.__main__: the paper-benchmark CLI.

The catalogue (config.CONFIG), the sweep (driver.drive) and the records->CSV
slice (results.write_config_csvs) are covered by test_config / test_driver /
test_results, so these cover only what is novel to the CLI: name resolution
(literal / glob / dedup, with typos surfacing as errors), argument parsing
(including the --no-csv / --csv-only mutual exclusion), and that ``run`` wires
those pieces together -- driving a sweep and writing its CSV, versus the
csv-only path that rebuilds the CSV from the records on disk without running
anything.

CONFIG is monkeypatched to a tiny WGN + cheap-VBA sweep so a real ``run`` is
fast; fresh seeds keep each cell a cache miss (so it really runs and records),
and ensure_hcp_data is stubbed (the WGN cells need no HCP data).
"""
import random

import pandas as pd
import pytest

from glow._extra.benchmark import __main__ as cli
from glow._extra.benchmark import config, data, hcp
from glow._extra.benchmark.run import run_ana
from glow.analysis import AnalysisVBA
from glow.effect import ExtenterSphere


def _fresh_seed() -> int:
    """A seed unlikely to be in the on-disk cache, so the cell is a miss."""
    return random.randrange(2 ** 31)


def _tiny_entry():
    """A one-cell WGN sweep: the CONFIG four-tuple ``drive`` consumes.

    A single data cell (fresh seed -> cache miss), one fixed sphere plant, and
    one cheap VBA fit -- so ``run`` does a real but fast sweep that records.
    """
    data_list = [dict(source='wgn', shape=(5, 5, 5), b=2, num_img=16, a=2,
                      seed=_fresh_seed())]
    effect_list = [dict(effect_llr=0.05, extenter_cls=ExtenterSphere,
                        n_vox_frac=0.1, seed=0)]
    fnc_list = [dict(ana=AnalysisVBA(n_perm_fwer=15))]
    return (data_list, effect_list, fnc_list, run_ana)


@pytest.fixture(autouse=True)
def _records_to_tmp(monkeypatch, tmp_path):
    """Isolate the shared recorder: tmp folder, empty in-memory records."""
    monkeypatch.setattr(data.RECORDER, 'folder', tmp_path)
    data.RECORDER.records.clear()


# ---------------------------------------------------------------------------
# name resolution
# ---------------------------------------------------------------------------

class TestResolveNames:
    @pytest.fixture(autouse=True)
    def _catalogue(self, monkeypatch):
        # an ordered known catalogue so first-seen / glob order is assertable
        monkeypatch.setattr(config, 'CONFIG', dict.fromkeys(
            ['null', 'sweep_llr', 'sweep_b', 'sweep_extent']))

    def test_empty_selects_all(self):
        assert cli.resolve_names([]) == [
            'null', 'sweep_llr', 'sweep_b', 'sweep_extent']

    def test_literal(self):
        assert cli.resolve_names(['sweep_b']) == ['sweep_b']

    def test_glob(self):
        assert cli.resolve_names(['sweep_*']) == [
            'sweep_llr', 'sweep_b', 'sweep_extent']

    def test_dedup_first_seen(self):
        # the literal pins first position; the glob's later repeats are dropped
        assert cli.resolve_names(['sweep_llr', 'sweep_*', 'null']) == [
            'sweep_llr', 'sweep_b', 'sweep_extent', 'null']

    def test_unknown_literal_raises(self):
        with pytest.raises(ValueError):
            cli.resolve_names(['nope'])

    def test_zero_match_glob_raises(self):
        with pytest.raises(ValueError):
            cli.resolve_names(['xyz_*'])


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self):
        ns = cli.parse_args([])
        assert ns.names == [] and ns.n_jobs == 1 and not ns.quiet
        assert not ns.no_csv and not ns.csv_only and not ns.list_names
        assert ns.out_dir is None

    def test_flags(self):
        ns = cli.parse_args(
            ['sweep_llr', '-j', '4', '-q', '--csv-only', '--out-dir', '/x'])
        assert ns.names == ['sweep_llr'] and ns.n_jobs == 4 and ns.quiet
        assert ns.csv_only and ns.out_dir == '/x'

    def test_no_csv_and_csv_only_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            cli.parse_args(['--no-csv', '--csv-only'])


# ---------------------------------------------------------------------------
# run: drive a sweep then write its CSV, vs. csv-only rebuild
# ---------------------------------------------------------------------------

class TestRun:
    @pytest.fixture
    def _tiny(self, monkeypatch):
        """Patch CONFIG to one tiny cache; count ensure_hcp_data calls."""
        monkeypatch.setattr(config, 'CONFIG', {'tiny': _tiny_entry()})
        calls = []
        monkeypatch.setattr(hcp, 'ensure_hcp_data', lambda: calls.append(1))
        return calls

    def test_run_writes_csv(self, _tiny, tmp_path):
        written = cli.run(names=['tiny'], out_dir=tmp_path / 'csv',
                          verbose=False)
        assert set(written) == {'tiny'}
        # one data x effect x fnc cell -> one leaf row
        assert len(pd.read_csv(written['tiny'])) == 1
        # the HCP dataset was ensured once up front
        assert _tiny == [1]

    def test_no_csv_runs_but_writes_nothing(self, _tiny, tmp_path):
        written = cli.run(names=['tiny'], out_dir=tmp_path / 'csv',
                          write_csv=False, verbose=False)
        assert written == {}
        # the sweep still ran and recorded (only the CSV write was skipped)
        assert _tiny == [1]
        assert any(r['function'] == 'run_ana'
                   for r in data.RECORDER.records.values())

    def test_csv_only_rebuilds_without_running(self, _tiny, tmp_path):
        # a first sweep populates the records (no CSV written)
        cli.run(names=['tiny'], out_dir=tmp_path / 'csv', write_csv=False,
                verbose=False)
        n_records = len(data.RECORDER.records)
        _tiny.clear()

        # csv-only rebuilds the CSV from those records: no HCP load, no new run
        written = cli.run(names=['tiny'], out_dir=tmp_path / 'rebuild',
                          csv_only=True, verbose=False)
        assert set(written) == {'tiny'}
        assert len(pd.read_csv(written['tiny'])) == 1
        assert _tiny == []
        assert len(data.RECORDER.records) == n_records


# ---------------------------------------------------------------------------
# main: --list and dispatch
# ---------------------------------------------------------------------------

class TestMain:
    def test_list_prints_names(self, monkeypatch, capsys):
        monkeypatch.setattr(config, 'CONFIG', dict.fromkeys(['a', 'b']))
        cli.main(['--list'])
        assert capsys.readouterr().out.split() == ['a', 'b']

    def test_main_dispatches_to_run(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(cli, 'run', lambda **kw: captured.update(kw))
        cli.main(['sweep_llr', '-j', '3', '-q', '--no-csv'])
        assert captured['names'] == ['sweep_llr'] and captured['n_jobs'] == 3
        assert captured['verbose'] is False and captured['write_csv'] is False
        assert captured['csv_only'] is False
