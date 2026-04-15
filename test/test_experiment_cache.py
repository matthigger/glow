"""Tests for experiment-level caching in Config and load_update_all."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from glow.benchmark.config import Config
from glow.benchmark.file import load_update_all, OUT
from glow.benchmark.run import run_ana, run_segment


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_result_json(folder, label, seed, effect_llr, extra=None):
    """Write a minimal result JSON into folder/out/."""
    out = folder / OUT
    out.mkdir(parents=True, exist_ok=True)
    d = {
        'label': label,
        'seed': int(seed),
        'effect_llr': float(effect_llr),
        'dice': 0.5,
        'sens': 0.5,
        'spec': 0.5,
        'uuid': 'test1234',
        'vox_total': 100,
        'vox_effect': 20,
        'time_sec': 1.0,
    }
    if extra:
        d.update(extra)
    path = out / f'{d["uuid"]}_{label}_{seed}_{effect_llr}_result.json'
    with open(path, 'w') as f:
        json.dump(d, f)
    return path


def _make_config(label='test_cache', ana_labels=('A', 'B')):
    """Build a tiny Config with mock analyses."""
    import glow
    ana_kwargs_dict = {
        lbl: (glow.analysis.AnalysisGLOW, {
            'n_perm_fwer': 1,
            'min_size': 1, 'alpha_fwer': 0.05,
            'n_jobs_perm': 1,
        })
        for lbl in ana_labels
    }
    return Config(
        label=label,
        source='wgn',
        run_fnc=run_ana,
        ana_kwargs_dict=ana_kwargs_dict,
        n_seed=2,
        effect_llr_all=np.array([0.05, 0.2]),
        wgn_shape=(3, 3),
        wgn_a=2, wgn_b=2, wgn_num_img=10,
        exp_seed=0,
        n_jobs=1,
        detail_save=False,
        error_save=False,
    )


# ---------------------------------------------------------------------------
# load_update_all
# ---------------------------------------------------------------------------

class TestLoadUpdateAll:
    def test_missing_folder_returns_empty(self, tmp_path):
        """load_update_all returns empty df when folder doesn't exist."""
        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            df, folder, n_new = load_update_all('nonexistent', verbose=False)
        assert df.empty
        assert n_new == 0

    def test_aggregates_json_into_csv(self, tmp_path):
        """JSON files are aggregated into results.csv and deleted."""
        label = 'test_agg'
        label_dir = tmp_path / label
        label_dir.mkdir()

        _make_result_json(label_dir, 'GLOW', seed=0, effect_llr=0.05)
        _make_result_json(label_dir, 'VBA', seed=0, effect_llr=0.05)

        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            df, folder, n_new = load_update_all(label, verbose=False)

        assert len(df) == 2
        assert n_new == 2
        assert (label_dir / 'results.csv').exists()
        # JSON files should be deleted after aggregation
        assert list((label_dir / OUT).glob('*result.json')) == []

    def test_incremental_load(self, tmp_path):
        """New JSONs are merged with existing CSV on subsequent loads."""
        label = 'test_incr'
        label_dir = tmp_path / label
        label_dir.mkdir()

        # first load: one result
        _make_result_json(label_dir, 'GLOW', seed=0, effect_llr=0.05)
        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            df1, _, _ = load_update_all(label, verbose=False)
        assert len(df1) == 1

        # second load: add another result
        _make_result_json(label_dir, 'VBA', seed=0, effect_llr=0.05)
        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            df2, _, n_new = load_update_all(label, verbose=False)
        assert len(df2) == 2
        assert n_new == 1


# ---------------------------------------------------------------------------
# _config_hash_for_label
# ---------------------------------------------------------------------------

class TestConfigHash:
    def test_deterministic(self):
        """Same config produces same hash for same label."""
        c1 = _make_config(ana_labels=('GLOW', 'VBA'))
        c2 = _make_config(ana_labels=('GLOW', 'VBA'))
        assert c1._config_hash_for_label('GLOW') == c2._config_hash_for_label('GLOW')

    def test_different_labels_different_hash(self):
        """Different labels produce different hashes."""
        c = _make_config(ana_labels=('GLOW', 'VBA'))
        assert c._config_hash_for_label('GLOW') != c._config_hash_for_label('VBA')

    def test_different_nperm(self):
        """Changing n_perm_fwer for a label changes that label's hash."""
        c1 = _make_config(ana_labels=('GLOW',))
        c2 = _make_config(ana_labels=('GLOW',))
        _, kw = c2.ana_kwargs_dict['GLOW']
        kw['n_perm_fwer'] = 999
        assert c1._config_hash_for_label('GLOW') != c2._config_hash_for_label('GLOW')

    def test_changing_other_label_no_effect(self):
        """Changing VBA's config does not affect GLOW's hash."""
        c1 = _make_config(ana_labels=('GLOW', 'VBA'))
        c2 = _make_config(ana_labels=('GLOW', 'VBA'))
        _, kw = c2.ana_kwargs_dict['VBA']
        kw['n_perm_fwer'] = 999
        assert c1._config_hash_for_label('GLOW') == c2._config_hash_for_label('GLOW')
        assert c1._config_hash_for_label('VBA') != c2._config_hash_for_label('VBA')

    def test_different_source(self):
        """Changing source changes the hash (base params changed)."""
        c1 = _make_config()
        c2 = _make_config()
        c2.source = 'hcp'
        assert c1._config_hash_for_label('GLOW') != c2._config_hash_for_label('GLOW')

    def test_different_wgn_shape(self):
        """Changing wgn_shape changes the hash (base params changed)."""
        c1 = _make_config()
        c2 = _make_config()
        c2.wgn_shape = (5, 5)
        assert c1._config_hash_for_label('GLOW') != c2._config_hash_for_label('GLOW')

    def test_hash_length(self):
        """Hash should be a 12 character hex string."""
        c = _make_config()
        h = c._config_hash_for_label('GLOW')
        assert len(h) == 12
        assert all(ch in '0123456789abcdef' for ch in h)

    def test_lazy_loads_exp_orig(self):
        """_config_hash_for_label() auto-loads exp_orig if None."""
        c = _make_config()
        assert c.exp_orig is None
        c._config_hash_for_label('GLOW')
        assert c.exp_orig is not None

    def test_job_hash_deterministic(self):
        """_job_hash() is deterministic."""
        c1 = _make_config(ana_labels=('GLOW', 'VBA'))
        c2 = _make_config(ana_labels=('GLOW', 'VBA'))
        assert c1._job_hash() == c2._job_hash()

    def test_job_hash_subset(self):
        """_job_hash() with a subset of labels differs from full set."""
        c = _make_config(ana_labels=('GLOW', 'VBA'))
        assert c._job_hash(['GLOW']) != c._job_hash(['GLOW', 'VBA'])


# ---------------------------------------------------------------------------
# _get_expected_labels
# ---------------------------------------------------------------------------

class TestGetExpectedLabels:
    def test_run_ana_labels(self):
        config = _make_config(ana_labels=('GLOW', 'VBA', 'VBA-TFCE'))
        assert config._get_expected_labels() == {'GLOW', 'VBA', 'VBA-TFCE'}

    def test_run_segment_labels(self):
        config = Config(label='seg', run_fnc=run_segment, source='wgn')
        from glow.analysis.cluster import MODE_LABELS
        assert config._get_expected_labels() == set(MODE_LABELS.values())

    def test_unknown_run_fnc(self):
        config = Config(label='x', run_fnc=lambda: None, source='wgn')
        assert config._get_expected_labels() == set()


# ---------------------------------------------------------------------------
# _is_experiment_cached
# ---------------------------------------------------------------------------

class TestIsExperimentCached:
    def setup_method(self):
        self.config = _make_config(ana_labels=('GLOW', 'VBA'))
        self.expected = {'GLOW', 'VBA'}
        self.label_hashes = {
            lab: self.config._config_hash_for_label(lab)
            for lab in self.expected
        }

    def test_empty_df(self):
        assert not self.config._is_experiment_cached(
            {'seed': 0, 'effect_llr': 0.05}, pd.DataFrame(), self.expected,
            self.label_hashes)

    def test_fully_cached(self):
        df = pd.DataFrame([
            {'seed': 0, 'effect_llr': 0.05, 'label': 'GLOW', 'dice': 0.5,
             'config_hash': self.label_hashes['GLOW']},
            {'seed': 0, 'effect_llr': 0.05, 'label': 'VBA', 'dice': 0.6,
             'config_hash': self.label_hashes['VBA']},
        ])
        assert self.config._is_experiment_cached(
            {'seed': 0, 'effect_llr': 0.05}, df, self.expected,
            self.label_hashes)

    def test_partially_cached(self):
        df = pd.DataFrame([
            {'seed': 0, 'effect_llr': 0.05, 'label': 'GLOW', 'dice': 0.5,
             'config_hash': self.label_hashes['GLOW']},
        ])
        assert not self.config._is_experiment_cached(
            {'seed': 0, 'effect_llr': 0.05}, df, self.expected,
            self.label_hashes)

    def test_different_seed_not_cached(self):
        df = pd.DataFrame([
            {'seed': 1, 'effect_llr': 0.05, 'label': 'GLOW', 'dice': 0.5,
             'config_hash': self.label_hashes['GLOW']},
            {'seed': 1, 'effect_llr': 0.05, 'label': 'VBA', 'dice': 0.6,
             'config_hash': self.label_hashes['VBA']},
        ])
        assert not self.config._is_experiment_cached(
            {'seed': 0, 'effect_llr': 0.05}, df, self.expected,
            self.label_hashes)

    def test_wrong_config_hash_not_cached(self):
        """Results with a different config_hash should not count as cached."""
        df = pd.DataFrame([
            {'seed': 0, 'effect_llr': 0.05, 'label': 'GLOW', 'dice': 0.5,
             'config_hash': 'wrong_hash_x'},
            {'seed': 0, 'effect_llr': 0.05, 'label': 'VBA', 'dice': 0.6,
             'config_hash': 'wrong_hash_x'},
        ])
        assert not self.config._is_experiment_cached(
            {'seed': 0, 'effect_llr': 0.05}, df, self.expected,
            self.label_hashes)

    def test_no_hash_column_not_cached(self):
        """Legacy data without config_hash column should not count as cached."""
        df = pd.DataFrame([
            {'seed': 0, 'effect_llr': 0.05, 'label': 'GLOW', 'dice': 0.5},
            {'seed': 0, 'effect_llr': 0.05, 'label': 'VBA', 'dice': 0.6},
        ])
        assert not self.config._is_experiment_cached(
            {'seed': 0, 'effect_llr': 0.05}, df, self.expected,
            self.label_hashes)

    def test_float_rounding(self):
        """effect_llr floats should match after rounding to 14 decimals."""
        df = pd.DataFrame([
            {'seed': 0, 'effect_llr': 0.050000000000001, 'label': 'GLOW',
             'dice': 0.5, 'config_hash': self.label_hashes['GLOW']},
            {'seed': 0, 'effect_llr': 0.050000000000001, 'label': 'VBA',
             'dice': 0.6, 'config_hash': self.label_hashes['VBA']},
        ])
        assert self.config._is_experiment_cached(
            {'seed': 0, 'effect_llr': 0.05}, df, self.expected,
            self.label_hashes)

    def test_stale_hash_for_one_label(self):
        """If one label has a stale hash, only that label is uncached."""
        df = pd.DataFrame([
            {'seed': 0, 'effect_llr': 0.05, 'label': 'GLOW', 'dice': 0.5,
             'config_hash': self.label_hashes['GLOW']},
            {'seed': 0, 'effect_llr': 0.05, 'label': 'VBA', 'dice': 0.6,
             'config_hash': 'stale_hash_x'},
        ])
        cached = self.config._cached_labels(
            {'seed': 0, 'effect_llr': 0.05}, df, self.label_hashes)
        assert cached == {'GLOW'}
        assert not self.config._is_experiment_cached(
            {'seed': 0, 'effect_llr': 0.05}, df, self.expected,
            self.label_hashes)


# ---------------------------------------------------------------------------
# _filter_uncached
# ---------------------------------------------------------------------------

class TestFilterUncached:
    def test_no_cache_returns_all(self, tmp_path):
        config = _make_config()
        kwargs_list = list(config.iter_kwargs())

        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            uncached = config._filter_uncached(kwargs_list, verbose=False)

        assert len(uncached) == len(kwargs_list)
        # indices should be sequential
        assert [idx for idx, _, _ in uncached] == list(range(len(kwargs_list)))

    def test_partial_cache(self, tmp_path):
        config = _make_config(ana_labels=('A', 'B'))
        kwargs_list = list(config.iter_kwargs())
        n_total = len(kwargs_list)

        # cache the first experiment (seed=0, effect_llr=0.05)
        label_dir = tmp_path / config.label
        label_dir.mkdir()
        _make_result_json(label_dir, 'A', seed=0, effect_llr=0.05,
                          extra={'config_hash': config._config_hash_for_label('A')})
        _make_result_json(label_dir, 'B', seed=0, effect_llr=0.05,
                          extra={'config_hash': config._config_hash_for_label('B')})
        # aggregate into CSV
        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            load_update_all(config.label, verbose=False)

        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            uncached = config._filter_uncached(kwargs_list, verbose=False)

        assert len(uncached) == n_total - 1

    def test_all_cached(self, tmp_path):
        config = _make_config(ana_labels=('A',))
        config.n_seed = 1
        config.effect_llr_all = np.array([0.05])
        kwargs_list = list(config.iter_kwargs())

        label_dir = tmp_path / config.label
        label_dir.mkdir()
        _make_result_json(label_dir, 'A', seed=0, effect_llr=0.05,
                          extra={'config_hash': config._config_hash_for_label('A')})
        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            load_update_all(config.label, verbose=False)

        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            uncached = config._filter_uncached(kwargs_list, verbose=False)

        assert len(uncached) == 0

    def test_partial_label_cache(self, tmp_path):
        """Experiment with one label cached returns with missing={B}."""
        config = _make_config(ana_labels=('A', 'B'))
        config.n_seed = 1
        config.effect_llr_all = np.array([0.05])
        kwargs_list = list(config.iter_kwargs())

        # cache only label A for the single experiment
        label_dir = tmp_path / config.label
        label_dir.mkdir()
        _make_result_json(label_dir, 'A', seed=0, effect_llr=0.05,
                          extra={'config_hash': config._config_hash_for_label('A')})
        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            load_update_all(config.label, verbose=False)

        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            uncached = config._filter_uncached(kwargs_list, verbose=False)

        assert len(uncached) == 1
        exp_idx, kwargs, missing = uncached[0]
        assert exp_idx == 0
        assert missing == {'B'}

    def test_stale_hash_not_cached(self, tmp_path):
        """Results with old config_hash are ignored."""
        config = _make_config(ana_labels=('A',))
        config.n_seed = 1
        config.effect_llr_all = np.array([0.05])
        kwargs_list = list(config.iter_kwargs())

        label_dir = tmp_path / config.label
        label_dir.mkdir()
        _make_result_json(label_dir, 'A', seed=0, effect_llr=0.05,
                          extra={'config_hash': 'old_stale_hsh'})
        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            load_update_all(config.label, verbose=False)

        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            uncached = config._filter_uncached(kwargs_list, verbose=False)

        assert len(uncached) == len(kwargs_list)


# ---------------------------------------------------------------------------
# submit_cloud_jobs interface
# ---------------------------------------------------------------------------

class TestSubmitCloudJobs:
    """Verify submit_cloud_jobs passes correct data shapes to AWSBatchRunner."""

    def test_upload_receives_2tuples(self, tmp_path):
        """upload_all_kwargs must get list of (int, dict), not 3-tuples."""
        from glow.aws.aws_batch import CloudConfig

        config = _make_config(ana_labels=('A',))
        config.n_seed = 1
        config.effect_llr_all = np.array([0.05])
        config.cloud_config = CloudConfig(
            s3_bucket='fake', s3_prefix='fake',
            job_queue='fake', job_definition='fake',
        )

        with (
            patch('glow.benchmark.file.get_path_result',
                  return_value=tmp_path),
            patch('glow.aws.aws_batch.boto3'),
            patch('glow.benchmark.config.path_result', tmp_path),
        ):
            # prep so exp_orig exists (needed for memory estimation)
            config.prep_exp_orig()

            # mock the runner methods called by submit_cloud_jobs
            with patch('glow.aws.aws_batch.AWSBatchRunner') as MockRunner:
                runner = MockRunner.return_value
                runner.config = config.cloud_config
                runner.estimate_experiment_memory_mb.return_value = None
                runner.submit_array_job.return_value = {
                    'child_job_ids': ['job-1'],
                    'index_map': {'job-1': 0},
                }

                config.submit_cloud_jobs(verbose=False)

                # verify upload_all_kwargs got (int, dict) pairs
                args = runner.upload_all_kwargs.call_args[0]
                kwargs_list = args[1]
                assert len(kwargs_list) >= 1
                for item in kwargs_list:
                    assert len(item) == 2, f'expected 2-tuple, got {len(item)}'
                    assert isinstance(item[0], (int, np.integer))
                    assert isinstance(item[1], dict)

                # verify submit_array_job got integer indices
                sa_kwargs = runner.submit_array_job.call_args[1]
                for idx in sa_kwargs['indices']:
                    assert isinstance(idx, (int, np.integer))


# ---------------------------------------------------------------------------
# prep_folder
# ---------------------------------------------------------------------------

class TestPrepFolder:
    def test_creates_flat_folder(self, tmp_path):
        """prep_folder creates results/{label}/ with no timestamp."""
        with patch('glow.benchmark.config.path_result', tmp_path):
            config = Config(label='test_flat', source='wgn', run_fnc=run_ana)
            config.prep_folder()

        assert config.folder == tmp_path / 'test_flat'
        assert config.folder.exists()

    def test_idempotent(self, tmp_path):
        """calling prep_folder twice doesn't raise."""
        with patch('glow.benchmark.config.path_result', tmp_path):
            config = Config(label='test_idem', source='wgn', run_fnc=run_ana)
            config.prep_folder()
            config.prep_folder()  # should not raise

        assert config.folder.exists()
