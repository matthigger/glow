"""Tests for the recorder-wired paper driver (records.json) and run_ana.

driver_paper clears the shared recorder per trial, runs the trial fn, stamps
the cache hash and scalar axes onto each captured record, and appends to
records.json -- with a legacy csv fallback for fns that still return a
DataFrame. The driver-mechanics tests use trivial fns (no analysis fit); the
run_ana tests exercise the real recorder wiring on a tiny WGN cell.
"""
import json
from functools import partial

import pandas as pd
import pytest

from glow.analysis import AnalysisVBA
from glow.benchmark.trial_cache import TrialCache, CSV_NAME, RECORDS_NAME
from glow.benchmark.paper import run as paper_run
from glow.benchmark.paper.driver import driver_paper


def _cache(tmp_path, **iter_kwargs):
    """A tiny WGN run_ana cache rooted at tmp_path."""
    return TrialCache(folder=str(tmp_path),
                      kwargs=dict(source='wgn', num_img=20, n_vox_eff=6,
                                  effect_llr=0.5),
                      iter_kwargs=iter_kwargs)


def _records(tmp_path):
    return json.loads((tmp_path / RECORDS_NAME).read_text())


class TestDriverMechanics:
    """records.json IO, hash/axis stamping, resume, and the legacy csv path,
    exercised with trivial fns (no analysis fit)."""

    def test_records_stamped_and_written(self, tmp_path):
        @paper_run.recorder(output_name='out')
        def trivial(*, source, b, num_img, n_vox_eff, seed, effect_llr=None):
            return b + seed

        driver_paper(_cache(tmp_path, b=[2], seed=[0, 1]), trivial,
                     verbose=False)

        recs = _records(tmp_path)
        assert len(recs) == 2
        # every record carries the scalar axes; trial_id is the cache hash
        # (the driver's trial scope)
        for r in recs:
            assert r['source'] == 'wgn' and r['b'] == 2 and r['n_vox_eff'] == 6
            assert r['seed'] in (0, 1)
            assert r['outputs']['out'] == 2 + r['seed']
        # one distinct trial id (cache hash) per trial (seed)
        assert len({r['trial_id'] for r in recs}) == 2

    def test_resume_skips_done_trials(self, tmp_path):
        @paper_run.recorder(output_name='out')
        def trivial(*, source, b, num_img, n_vox_eff, seed, effect_llr=None):
            return seed

        cache = _cache(tmp_path, b=[2], seed=[0])
        driver_paper(cache, trivial, verbose=False)
        n1 = len(_records(tmp_path))
        # a second run finds the hash already in records.json -> no-op
        driver_paper(cache, trivial, verbose=False)
        n2 = len(_records(tmp_path))
        assert n1 == n2 == 1

    def test_legacy_dataframe_goes_to_csv(self, tmp_path):
        # a fn that returns a DataFrame (not recorder-wired) still persists via
        # the legacy results.csv path, and writes no records.json
        def legacy(*, source, b, num_img, n_vox_eff, seed, effect_llr=None):
            return pd.DataFrame([{'label': 'X', 'val': seed}])

        driver_paper(_cache(tmp_path, b=[2], seed=[0]), legacy, verbose=False)

        assert not (tmp_path / RECORDS_NAME).exists()
        assert (tmp_path / CSV_NAME).exists()


class TestRunAna:
    """run_ana wired to the recorder: setup provenance + per-label fit records."""

    def test_setup_and_fit_share_one_trial_id(self, tmp_path):
        ana = {'VBA': (AnalysisVBA, dict(n_perm_fwer=1))}
        driver_paper(_cache(tmp_path, b=[2], seed=[0]),
                     partial(paper_run.run_ana, ana_kwargs_dict=ana),
                     verbose=False)

        recs = _records(tmp_path)
        # setup + fit share one trial_id (the cache hash, from the driver scope)
        assert len({r['trial_id'] for r in recs}) == 1

        setup = next(r for r in recs if r['function'].endswith('_setup_trial'))
        assert setup['outputs']['exp_eff']['kind'] == 'Experiment'
        assert setup['outputs']['exp_eff']['b'] == 2
        eff = setup['outputs']['effect_list'][0]
        assert eff['kind'] == 'EffectSynthetic' and 'mask' not in eff
        assert eff['extenter']['kind'] == 'ExtenterMinVar'

        fit = next(r for r in recs if r['function'].endswith('.fit'))
        assert fit['trial_id'] == setup['trial_id']  # same trial as setup
        # the analysis recipe (self) identifies the variant
        assert fit['inputs']['self']['kind'] == 'AnalysisVBA'
        assert isinstance(fit['time_sec'], float)
        # config recipe only -- no fitted arrays in the record
        assert 'pval' not in fit['inputs']['self']

    def test_fit_failure_short_circuits_the_trial(self, tmp_path):
        # under one trial scope, the first failing fit marks the trial failed,
        # so later fits short-circuit (no per-label isolation): the trial's
        # records stop at the failure
        class _Boom(AnalysisVBA):
            def fit(self, _stat=None):
                raise RuntimeError('boom')

        ana = {'boom': (_Boom, dict(n_perm_fwer=1)),
               'VBA': (AnalysisVBA, dict(n_perm_fwer=1))}
        driver_paper(_cache(tmp_path, b=[2], seed=[0]),
                     partial(paper_run.run_ana, ana_kwargs_dict=ana),
                     verbose=False)

        recs = _records(tmp_path)
        fits = [r for r in recs if r['function'].endswith('.fit')]
        # only the failing fit is recorded; the trailing VBA short-circuited
        assert len(fits) == 1
        assert 'error' in fits[0] and 'RuntimeError' in fits[0]['error']
