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

def _make_result_json(folder, label, seed, hotel_tr, extra=None):
    """Write a minimal result JSON into folder/out/."""
    out = folder / OUT
    out.mkdir(parents=True, exist_ok=True)
    d = {
        'label': label,
        'seed': int(seed),
        'hotel_tr': float(hotel_tr),
        'f1': 0.5,
        'sens': 0.5,
        'spec': 0.5,
        'uuid': 'test1234',
        'vox_total': 100,
        'vox_effect': 20,
        'time_sec': 1.0,
    }
    if extra:
        d.update(extra)
    path = out / f'{d["uuid"]}_{label}_{seed}_{hotel_tr}_result.json'
    with open(path, 'w') as f:
        json.dump(d, f)
    return path


def _make_config(label='test_cache', ana_labels=('A', 'B')):
    """Build a tiny Config with mock analyses."""
    import glow
    ana_kwargs_dict = {
        lbl: (glow.experiment.AnalysisGLOW, {
            'n_perm': 1, 'n_perm_prune': 1,
            'min_size': 1, 'alpha_prune': 0.05, 'alpha_fwer': 0.05,
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
        hotel_tr_all=np.array([0.1, 0.5]),
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

        _make_result_json(label_dir, 'GLOW', seed=0, hotel_tr=0.1)
        _make_result_json(label_dir, 'VBA', seed=0, hotel_tr=0.1)

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
        _make_result_json(label_dir, 'GLOW', seed=0, hotel_tr=0.1)
        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            df1, _, _ = load_update_all(label, verbose=False)
        assert len(df1) == 1

        # second load: add another result
        _make_result_json(label_dir, 'VBA', seed=0, hotel_tr=0.1)
        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            df2, _, n_new = load_update_all(label, verbose=False)
        assert len(df2) == 2
        assert n_new == 1


# ---------------------------------------------------------------------------
# _config_hash
# ---------------------------------------------------------------------------

class TestConfigHash:
    def test_deterministic(self):
        """Same config produces same hash."""
        c1 = _make_config(ana_labels=('GLOW', 'VBA'))
        c2 = _make_config(ana_labels=('GLOW', 'VBA'))
        assert c1._config_hash() == c2._config_hash()

    def test_different_nperm(self):
        """Changing n_perm in ana_kwargs_dict produces a different hash."""
        c1 = _make_config(ana_labels=('GLOW',))
        c2 = _make_config(ana_labels=('GLOW',))
        # mutate c2's n_perm
        _, kw = c2.ana_kwargs_dict['GLOW']
        kw['n_perm'] = 999
        assert c1._config_hash() != c2._config_hash()

    def test_different_source(self):
        """Changing source changes the hash."""
        c1 = _make_config()
        c2 = _make_config()
        c2.source = 'hcp'
        assert c1._config_hash() != c2._config_hash()

    def test_different_wgn_shape(self):
        """Changing wgn_shape changes the hash."""
        c1 = _make_config()
        c2 = _make_config()
        c2.wgn_shape = (5, 5)
        assert c1._config_hash() != c2._config_hash()

    def test_hash_length(self):
        """Hash should be a 12 character hex string."""
        c = _make_config()
        h = c._config_hash()
        assert len(h) == 12
        assert all(ch in '0123456789abcdef' for ch in h)

    def test_lazy_loads_exp_orig(self):
        """_config_hash() auto-loads exp_orig if None."""
        c = _make_config()
        assert c.exp_orig is None
        c._config_hash()
        assert c.exp_orig is not None


# ---------------------------------------------------------------------------
# _get_expected_labels
# ---------------------------------------------------------------------------

class TestGetExpectedLabels:
    def test_run_ana_labels(self):
        config = _make_config(ana_labels=('GLOW', 'VBA', 'VBA-TFCE'))
        assert config._get_expected_labels() == {'GLOW', 'VBA', 'VBA-TFCE'}

    def test_run_segment_labels(self):
        config = Config(label='seg', run_fnc=run_segment, source='wgn')
        assert config._get_expected_labels() == {'ward-naive', 'ward-glm'}

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
        self.hash = self.config._config_hash()

    def test_empty_df(self):
        assert not self.config._is_experiment_cached(
            {'seed': 0, 'hotel_tr': 0.1}, pd.DataFrame(), self.expected,
            self.hash)

    def test_fully_cached(self):
        df = pd.DataFrame([
            {'seed': 0, 'hotel_tr': 0.1, 'label': 'GLOW', 'f1': 0.5,
             'config_hash': self.hash},
            {'seed': 0, 'hotel_tr': 0.1, 'label': 'VBA', 'f1': 0.6,
             'config_hash': self.hash},
        ])
        assert self.config._is_experiment_cached(
            {'seed': 0, 'hotel_tr': 0.1}, df, self.expected, self.hash)

    def test_partially_cached(self):
        df = pd.DataFrame([
            {'seed': 0, 'hotel_tr': 0.1, 'label': 'GLOW', 'f1': 0.5,
             'config_hash': self.hash},
        ])
        assert not self.config._is_experiment_cached(
            {'seed': 0, 'hotel_tr': 0.1}, df, self.expected, self.hash)

    def test_different_seed_not_cached(self):
        df = pd.DataFrame([
            {'seed': 1, 'hotel_tr': 0.1, 'label': 'GLOW', 'f1': 0.5,
             'config_hash': self.hash},
            {'seed': 1, 'hotel_tr': 0.1, 'label': 'VBA', 'f1': 0.6,
             'config_hash': self.hash},
        ])
        assert not self.config._is_experiment_cached(
            {'seed': 0, 'hotel_tr': 0.1}, df, self.expected, self.hash)

    def test_wrong_config_hash_not_cached(self):
        """Results with a different config_hash should not count as cached."""
        df = pd.DataFrame([
            {'seed': 0, 'hotel_tr': 0.1, 'label': 'GLOW', 'f1': 0.5,
             'config_hash': 'wrong_hash_x'},
            {'seed': 0, 'hotel_tr': 0.1, 'label': 'VBA', 'f1': 0.6,
             'config_hash': 'wrong_hash_x'},
        ])
        assert not self.config._is_experiment_cached(
            {'seed': 0, 'hotel_tr': 0.1}, df, self.expected, self.hash)

    def test_no_hash_column_not_cached(self):
        """Legacy data without config_hash column should not count as cached."""
        df = pd.DataFrame([
            {'seed': 0, 'hotel_tr': 0.1, 'label': 'GLOW', 'f1': 0.5},
            {'seed': 0, 'hotel_tr': 0.1, 'label': 'VBA', 'f1': 0.6},
        ])
        assert not self.config._is_experiment_cached(
            {'seed': 0, 'hotel_tr': 0.1}, df, self.expected, self.hash)

    def test_float_rounding(self):
        """hotel_tr floats should match after rounding to 14 decimals."""
        df = pd.DataFrame([
            {'seed': 0, 'hotel_tr': 0.100000000000001, 'label': 'GLOW',
             'f1': 0.5, 'config_hash': self.hash},
            {'seed': 0, 'hotel_tr': 0.100000000000001, 'label': 'VBA',
             'f1': 0.6, 'config_hash': self.hash},
        ])
        assert self.config._is_experiment_cached(
            {'seed': 0, 'hotel_tr': 0.1}, df, self.expected, self.hash)


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
        assert [idx for idx, _ in uncached] == list(range(len(kwargs_list)))

    def test_partial_cache(self, tmp_path):
        config = _make_config(ana_labels=('A', 'B'))
        kwargs_list = list(config.iter_kwargs())
        n_total = len(kwargs_list)
        ch = config._config_hash()

        # cache the first experiment (seed=0, hotel_tr=0.1)
        label_dir = tmp_path / config.label
        label_dir.mkdir()
        _make_result_json(label_dir, 'A', seed=0, hotel_tr=0.1,
                          extra={'config_hash': ch})
        _make_result_json(label_dir, 'B', seed=0, hotel_tr=0.1,
                          extra={'config_hash': ch})
        # aggregate into CSV
        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            load_update_all(config.label, verbose=False)

        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            uncached = config._filter_uncached(kwargs_list, verbose=False)

        assert len(uncached) == n_total - 1

    def test_all_cached(self, tmp_path):
        config = _make_config(ana_labels=('A',))
        config.n_seed = 1
        config.hotel_tr_all = np.array([0.1])
        kwargs_list = list(config.iter_kwargs())
        ch = config._config_hash()

        label_dir = tmp_path / config.label
        label_dir.mkdir()
        _make_result_json(label_dir, 'A', seed=0, hotel_tr=0.1,
                          extra={'config_hash': ch})
        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            load_update_all(config.label, verbose=False)

        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            uncached = config._filter_uncached(kwargs_list, verbose=False)

        assert len(uncached) == 0

    def test_stale_hash_not_cached(self, tmp_path):
        """Results with old config_hash are ignored."""
        config = _make_config(ana_labels=('A',))
        config.n_seed = 1
        config.hotel_tr_all = np.array([0.1])
        kwargs_list = list(config.iter_kwargs())

        label_dir = tmp_path / config.label
        label_dir.mkdir()
        _make_result_json(label_dir, 'A', seed=0, hotel_tr=0.1,
                          extra={'config_hash': 'old_stale_hsh'})
        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            load_update_all(config.label, verbose=False)

        with patch('glow.benchmark.file.get_path_result', return_value=tmp_path):
            uncached = config._filter_uncached(kwargs_list, verbose=False)

        assert len(uncached) == len(kwargs_list)


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
