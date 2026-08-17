"""Tests for glow._extra.benchmark.__main__: the paper-benchmark CLI.

The catalogue (config.CONFIG), the sweep (driver.drive) and the records->CSV
export (make_csv) are covered by test_config / test_driver / test_make_csv, so
these cover only what is novel to the CLI: name resolution (literal / glob /
dedup, with typos surfacing as errors), argument parsing, and that run wires
those pieces together -- driving a sweep into the records and nothing else
(no CSV: aggregation is make_csv's separate step).

CONFIG is monkeypatched to a tiny WGN + cheap-VBA sweep so a real ``run`` is
fast; fresh seeds keep each cell a cache miss (so it really runs and records),
and ensure_hcp_data is stubbed (the WGN cells need no HCP data).
"""
import random

import pytest

from glow._extra.benchmark import __main__ as cli
from glow._extra.benchmark import config, data, hcp, make_csv
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
            ['null', 'sweep_llr_b1', 'sweep_llr_b2', 'sweep_extent']))

    def test_empty_selects_all(self):
        assert cli.resolve_names([]) == [
            'null', 'sweep_llr_b1', 'sweep_llr_b2', 'sweep_extent']

    def test_literal(self):
        assert cli.resolve_names(['sweep_llr_b2']) == ['sweep_llr_b2']

    def test_glob(self):
        assert cli.resolve_names(['sweep_*']) == [
            'sweep_llr_b1', 'sweep_llr_b2', 'sweep_extent']

    def test_dedup_first_seen(self):
        # the literal pins first position; the glob's later repeats are dropped
        assert cli.resolve_names(['sweep_llr_b1', 'sweep_*', 'null']) == [
            'sweep_llr_b1', 'sweep_llr_b2', 'sweep_extent', 'null']

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
        assert not ns.list_names and not ns.no_skip

    def test_flags(self):
        ns = cli.parse_args(['sweep_llr_b1', '-j', '4', '-q', '--no-skip'])
        assert ns.names == ['sweep_llr_b1'] and ns.n_jobs == 4 and ns.quiet
        assert ns.no_skip

    def test_csv_flags_are_gone(self):
        # a sweep is records-only, so the CSV knobs it used to carry must not
        # come back silently (aggregation is python -m ...benchmark.make_csv)
        for argv in (['--no-csv'], ['--csv-only'], ['--out-dir', '/x']):
            with pytest.raises(SystemExit):
                cli.parse_args(argv)

    def test_method_repeats_into_a_list(self):
        assert cli.parse_args([]).methods is None
        assert cli.parse_args(['--method', 'VBA', '--method', 'CET']).methods \
            == ['VBA', 'CET']


# ---------------------------------------------------------------------------
# run: drive a sweep into the records, writing nothing else
# ---------------------------------------------------------------------------

class TestRun:
    @pytest.fixture
    def _tiny(self, monkeypatch):
        """Patch CONFIG to one tiny cache; count ensure_hcp_data calls."""
        monkeypatch.setattr(config, 'CONFIG', {'tiny': _tiny_entry()})
        calls = []
        monkeypatch.setattr(hcp, 'ensure_hcp_data', lambda: calls.append(1))
        return calls

    def test_run_records_the_swept_cache(self, _tiny):
        assert cli.run(names=['tiny'], verbose=False) == ['tiny']
        # one data x effect x fnc cell -> one recorded run_ana leaf
        assert len([r for r in data.RECORDER.records.values()
                    if r['function'] == 'run_ana']) == 1
        # the HCP dataset was ensured once up front
        assert _tiny == [1]

    def test_sweep_writes_no_csv(self, _tiny, tmp_path, monkeypatch):
        # a sweep is records-only: nothing lands in the CSV destination until
        # the separate export step is asked for it
        results_dir = tmp_path / 'results'
        monkeypatch.setattr(make_csv, 'get_path_result', lambda: results_dir)
        cli.run(names=['tiny'], verbose=False)
        assert not results_dir.exists()
        assert make_csv.write_config_csv('tiny').shape[0] == 1
        assert [p.name for p in results_dir.iterdir()] == ['tiny.csv']

    @pytest.fixture
    def _two_recipes(self, monkeypatch):
        """A tiny two-recipe cache, its recipes labelled as --method resolves.

        The catalogue's own recipes are far too costly to fit here, so
        ana_kwargs_dict -- the label source the filter reads -- is patched to
        two cheap VBA fits (distinct n_perm_fwer, hence distinct leaves).
        """
        ana_a, ana_b = AnalysisVBA(n_perm_fwer=6), AnalysisVBA(n_perm_fwer=7)
        monkeypatch.setattr(config, 'ana_kwargs_dict',
                            {'A': ana_a, 'B': ana_b})
        data_list, effect_list, _, fnc = _tiny_entry()
        monkeypatch.setattr(config, 'CONFIG', {'tiny': (
            data_list, effect_list, [dict(ana=ana_a), dict(ana=ana_b)], fnc)})
        monkeypatch.setattr(hcp, 'ensure_hcp_data', lambda: None)
        return ana_a, ana_b

    def _ana_reprs(self):
        """The recipes the recorded run_ana leaves were fit under."""
        return [rec['inputs']['ana']
                for rec in data.RECORDER.records.values()
                if rec['function'] == 'run_ana']

    def test_methods_runs_only_the_named_recipe(self, _two_recipes):
        # the rerun-one-method path: the sibling recipe is never called, so its
        # fit is not recomputed (see run / config.filter_ana_list)
        _, ana_b = _two_recipes
        cli.run(names=['tiny'], verbose=False, methods=['B'])
        assert self._ana_reprs() == [repr(ana_b)]

    def test_methods_skips_a_cache_with_no_named_recipe(self, _two_recipes,
                                                       monkeypatch, capsys):
        # a leaf grid with no recipe of that name (a segment / prune grid in the
        # real catalogue) has no per-method axis, so the cache is skipped whole
        data_list, effect_list, _, fnc = _tiny_entry()
        monkeypatch.setattr(config, 'CONFIG', {'other': (
            data_list, effect_list, [dict(ana=AnalysisVBA(n_perm_fwer=8))],
            fnc)})
        # the skipped cache is not reported as driven
        assert cli.run(names=['other'], methods=['A']) == []
        assert 'skipped' in capsys.readouterr().out
        assert self._ana_reprs() == []


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
        cli.main(['sweep_llr_b1', '-j', '3', '-q', '--no-skip'])
        assert captured['names'] == ['sweep_llr_b1'] and captured['n_jobs'] == 3
        assert captured['verbose'] is False
        assert captured['skip_recorded'] is False
        assert captured['methods'] is None
