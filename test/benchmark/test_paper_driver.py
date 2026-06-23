"""Tests for the recorder-wired paper driver and trial fns.

The TrialCache owns a recorder and writes one json per trial under its records/
dir; driver_paper passes the recorder to a recorder-wired fn and serial drives
cache.iter_trial(record=True, flush=True) while parallel scopes each trial in a
worker. Covers run_ana plus the converted fit-shaped fns (run_mancova /
run_prune / run_two_effect), on a tiny WGN cell with cheap fits.
"""
import json
from functools import partial

from glow.analysis import AnalysisVBA
from glow.benchmark.trial_cache import TrialCache
from glow.benchmark.recorder import RECORDS_DIR
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
        assert fns == {'_setup_trial', 'fit', 'score_effects'}

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
        assert len(recs) == 6  # 2 trials x (setup + fit + score)
        # consolidate writes one combined json outside the per-trial dir
        cache.load_records(consolidate=True)
        combined = json.loads((tmp_path / 'records.json').read_text())
        assert len(combined) == 6

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

    def test_score_step_records_detection_dict(self, tmp_path):
        # the score step run beside each fit records the detection dict
        # (score_effects) against the recipe it scored
        cache = _cache(tmp_path, b=[2], seed=[0])
        driver_paper(cache, _ana(VBA), verbose=False)
        recs = cache.load_records()

        score = next(r for r in recs
                     if r['function'].endswith('score_effects'))
        s = score['outputs']['score']
        assert {'num_vox', 'min_pval', 'n_pred', 'pred', 'target'} <= set(s)
        assert set(s['target']) == {'tp', 'fp', 'tn', 'fn'}
        assert score['inputs']['ana']['kind'] == 'AnalysisVBA'

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


def _fns(recs):
    """Leaf names of each record's function (bare for free fns / methods)."""
    return [r['function'].rsplit('.', 1)[-1] for r in recs]


class TestConvertedFitFns:
    """run_mancova / run_prune / run_two_effect, recorder-wired (config only)."""

    def test_mancova_records_setup_walk_and_fits(self, tmp_path):
        cache = _cache(tmp_path, b=[2], seed=[0])
        driver_paper(cache, partial(paper_run.run_mancova, n_perm_fwer=1),
                     verbose=False)
        recs = cache.load_records()
        fns = _fns(recs)
        # setup + the shared voxel walk (recorded once) + one fit per variant
        assert '_setup_trial' in fns and '_shared_voxel_walk' in fns
        assert sum(f == 'fit' for f in fns) == 30
        walk = next(r for r in recs if r['function'].endswith('_shared_voxel_walk'))
        # walk output is keyed by stat name (json-friendly), each matrix a hash
        assert 'llr' in walk['outputs']['stat']
        fit = next(r for r in recs if r['function'].endswith('.fit'))
        assert fit['inputs']['self']['kind'] in ('AnalysisVBA', 'AnalysisCET')

    def test_prune_records_fit_and_both_rules(self, tmp_path, monkeypatch):
        # the real _GLOW_BASE is 250x1000 perms; shrink it for the test
        from glow.benchmark.paper import config as cfg
        monkeypatch.setattr(cfg, '_GLOW_BASE',
                            dict(n_perm_fwer=1, n_perm_inner=2))
        cache = _cache(tmp_path, b=[2], seed=[0])
        driver_paper(cache, paper_run.run_prune, verbose=False)
        recs = cache.load_records()
        fns = _fns(recs)
        assert 'fit' in fns  # the one shared GLOW fit
        prunes = [r for r in recs if 'prune' in r['function']]
        assert {r['function'].rsplit('.', 1)[-1] for r in prunes} == {
            'prune_greedy', 'prune_dp'}
        assert all('reg_out_list' in r['outputs'] for r in prunes)

    def test_two_effect_records_splitter_and_two_effects(self, tmp_path):
        cache = _cache(tmp_path, b=[2], seed=[0], angle=[30.0])
        driver_paper(cache, partial(paper_run.run_two_effect,
                                    ana_kwargs_dict=VBA), verbose=False)
        recs = cache.load_records()
        setup = next(r for r in recs
                     if r['function'].endswith('_setup_two_effect'))
        # provenance: the ExtenterSplit plus the two effect specs (angles 0/30)
        assert setup['outputs']['splitter']['kind'] == 'ExtenterSplit'
        effects = setup['outputs']['effect_list']
        assert len(effects) == 2 and effects[1]['angle'] == 30.0
        assert any(r['function'].endswith('AnalysisVBA.fit') for r in recs)
