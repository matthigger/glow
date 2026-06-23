"""Tests for the recorder-wired paper driver and run_ana.

The TrialCache owns a recorder and writes one json per trial under its records/
dir; driver_paper passes the recorder to a recorder-wired fn (run_ana) and falls
back to results.csv for a legacy DataFrame-returning fn. Serial drives
cache.iter_trial(record=True, flush=True); parallel scopes each trial in a
worker. The run_ana cases use a tiny WGN cell with a cheap VBA fit.
"""
import json
from functools import partial

from glow.analysis import AnalysisVBA
from glow.benchmark.trial_cache import TrialCache, RECORDS_DIR
from glow.benchmark.paper import run as paper_run
from glow.benchmark.paper.driver import driver_paper

VBA = {'VBA': (AnalysisVBA, dict(n_perm_fwer=1))}


def _cache(tmp_path, **iter_kwargs):
    """A tiny WGN run_ana cache rooted at tmp_path."""
    return TrialCache(folder=str(tmp_path),
                      kwargs=dict(source='wgn', num_img=20, n_vox_eff=6,
                                  effect_llr=0.5),
                      iter_kwargs=iter_kwargs)


def _ana(ana_kwargs_dict):
    return partial(paper_run.run_ana, ana_kwargs_dict=ana_kwargs_dict)


class TestDriverRecords:
    """Per-trial record files: writing, stamping, resume, load/consolidate,
    parallel, and the legacy csv fallback."""

    def test_per_trial_files_written_and_stamped(self, tmp_path):
        cache = _cache(tmp_path, b=[2], seed=[0, 1])
        driver_paper(cache, _ana(VBA), verbose=False)

        files = sorted((tmp_path / RECORDS_DIR).glob('*.json'))
        assert len(files) == 2  # one file per trial

        recs = json.loads(files[0].read_text())
        # the file holds that trial's records, each stamped with the axes and
        # sharing the trial_id (the cache hash = the file stem)
        assert all(r['source'] == 'wgn' and r['b'] == 2 for r in recs)
        assert {r['trial_id'] for r in recs} == {files[0].stem}
        fns = {r['function'].rsplit('.', 1)[-1] for r in recs}
        assert fns == {'_setup_trial', 'fit'}

    def test_resume_skips_completed(self, tmp_path):
        cache = _cache(tmp_path, b=[2], seed=[0])
        driver_paper(cache, _ana(VBA), verbose=False)
        n1 = len(list((tmp_path / RECORDS_DIR).glob('*.json')))
        driver_paper(cache, _ana(VBA), verbose=False)  # resume: no-op
        n2 = len(list((tmp_path / RECORDS_DIR).glob('*.json')))
        assert n1 == n2 == 1

    def test_load_records_and_consolidate(self, tmp_path):
        cache = _cache(tmp_path, b=[2], seed=[0, 1])
        driver_paper(cache, _ana(VBA), verbose=False)

        recs = cache.load_records()
        assert len(recs) == 4  # 2 trials x (setup + fit)
        # consolidate writes one combined json outside the per-trial dir
        cache.load_records(consolidate=True)
        combined = json.loads((tmp_path / 'records.json').read_text())
        assert len(combined) == 4

    def test_parallel_writes_per_trial_files(self, tmp_path):
        cache = _cache(tmp_path, b=[2], seed=[0, 1, 2])
        driver_paper(cache, _ana(VBA), n_jobs=2, verbose=False)
        assert len(list((tmp_path / RECORDS_DIR).glob('*.json'))) == 3


class TestRunAnaRecords:
    """run_ana given the recorder: setup provenance + per-label fit records."""

    def test_setup_and_fit_recipes(self, tmp_path):
        cache = _cache(tmp_path, b=[2], seed=[0])
        driver_paper(cache, _ana(VBA), verbose=False)
        recs = cache.load_records()

        # setup + fit share one trial_id (the cache hash)
        assert len({r['trial_id'] for r in recs}) == 1

        setup = next(r for r in recs if r['function'].endswith('_setup_trial'))
        assert setup['outputs']['exp_eff']['kind'] == 'Experiment'
        assert setup['outputs']['exp_eff']['b'] == 2
        assert 'meta' not in setup['outputs']['exp_eff']  # dropped
        eff = setup['outputs']['effect_list'][0]
        assert eff['kind'] == 'EffectSynthetic' and 'mask' not in eff
        assert eff['extenter']['kind'] == 'ExtenterMinVar'

        fit = next(r for r in recs if r['function'].endswith('.fit'))
        assert fit['inputs']['self']['kind'] == 'AnalysisVBA'
        assert isinstance(fit['time_sec'], float)
        assert 'pval' not in fit['inputs']['self']  # config recipe only

    def test_fit_failure_short_circuits_the_trial(self, tmp_path):
        # one shared trial scope -> the first failing fit marks the trial failed
        # and later fits short-circuit; the records stop at the failure
        class _Boom(AnalysisVBA):
            def fit(self, _stat=None):
                raise RuntimeError('boom')

        ana = {'boom': (_Boom, dict(n_perm_fwer=1)),
               'VBA': (AnalysisVBA, dict(n_perm_fwer=1))}
        cache = _cache(tmp_path, b=[2], seed=[0])
        driver_paper(cache, _ana(ana), verbose=False)

        fits = [r for r in cache.load_records() if r['function'].endswith('.fit')]
        assert len(fits) == 1
        assert 'error' in fits[0] and 'RuntimeError' in fits[0]['error']
