"""Tests for ``glow.benchmark.TrialCache``."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import glow.benchmark.trial_cache as tc_mod
from glow.benchmark import TrialCache
from glow.benchmark.trial_cache import HASH_COL
from glow.util import HashBySlots


class _Toy(HashBySlots):
    """Small HashBySlots fixture standing in for DataSource / Effect."""
    __slots__ = ('a', 'b')

    def __init__(self, a, b):
        self.a = a
        self.b = b


class TestConstruction:
    def test_folder_kwarg_creates_dir(self, tmp_path):
        target = tmp_path / 'fresh'
        assert not target.exists()
        cache = TrialCache(folder=target)
        assert cache.folder == target
        assert target.is_dir()

    def test_folder_accepts_str(self, tmp_path):
        cache = TrialCache(folder=str(tmp_path / 'as_str'))
        assert isinstance(cache.folder, Path)
        assert cache.folder.is_dir()

    def test_name_kwarg_resolves_under_default_results_dir(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(tc_mod, 'get_path_result', lambda: tmp_path)
        cache = TrialCache(name='exp_foo')
        assert cache.folder == tmp_path / 'exp_foo'
        assert cache.folder.is_dir()

    def test_both_name_and_folder_rejected(self, tmp_path):
        with pytest.raises(ValueError, match='exactly one'):
            TrialCache(name='x', folder=tmp_path)

    def test_neither_name_nor_folder_rejected(self):
        with pytest.raises(ValueError, match='exactly one'):
            TrialCache()

    def test_overlap_between_iter_and_const_kwargs_rejected(self, tmp_path):
        with pytest.raises(ValueError, match='share keys'):
            TrialCache(
                folder=tmp_path,
                iter_kwargs={'seed': [1, 2]},
                kwargs={'seed': 5})

    def test_defaults_iter_and_kwargs_none(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        assert cache.iter_kwargs is None
        assert cache.kwargs is None

    def test_df_initialized_empty_for_fresh_folder(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        assert isinstance(cache.df, pd.DataFrame)
        assert cache.df.empty

    def test_df_loaded_from_existing_csv(self, tmp_path):
        cache1 = TrialCache(folder=tmp_path, iter_kwargs={'seed': [0, 1]})
        for trial in cache1.iter_trial():
            cache1.save_result({'v': trial['seed']}, trial)
        cache2 = TrialCache(folder=tmp_path, iter_kwargs={'seed': [0, 1]})
        assert len(cache2.df) == 2
        assert sorted(cache2.df['v']) == [0, 1]


class TestIterTrial:
    def test_empty_yields_single_empty_dict(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        assert list(cache.iter_trial()) == [{}]

    def test_only_constant_kwargs_yields_one_trial(self, tmp_path):
        cache = TrialCache(folder=tmp_path, kwargs={'a': 1, 'b': 2})
        assert list(cache.iter_trial()) == [{'a': 1, 'b': 2}]

    def test_cartesian_product_count(self, tmp_path):
        cache = TrialCache(
            folder=tmp_path,
            iter_kwargs={'seed': range(3), 'llr': [0.1, 0.2]})
        assert len(list(cache.iter_trial())) == 6

    def test_constant_merged_into_each_trial(self, tmp_path):
        ds = _Toy(1, 2)
        cache = TrialCache(
            folder=tmp_path,
            iter_kwargs={'seed': [0, 1]},
            kwargs={'ds': ds})
        out = list(cache.iter_trial())
        assert all(t['ds'] is ds for t in out)
        assert sorted(t['seed'] for t in out) == [0, 1]

    def test_keys_match_spec(self, tmp_path):
        cache = TrialCache(
            folder=tmp_path,
            iter_kwargs={'a': [1], 'b': [2]},
            kwargs={'c': 3})
        assert list(cache.iter_trial()) == [{'a': 1, 'b': 2, 'c': 3}]


class TestCaching:
    def test_is_cached_false_before_save(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        assert cache.is_cached({'seed': 0}) is False

    def test_is_cached_true_after_save(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        trial = {'seed': 0, 'llr': 0.1}
        cache.save_result({'val': 42}, trial)
        assert cache.is_cached(trial) is True

    def test_iter_trial_no_repeat_skips_cached(self, tmp_path):
        cache = TrialCache(folder=tmp_path, iter_kwargs={'seed': [0, 1, 2]})
        cache.save_result({'val': 1}, {'seed': 1})
        remaining = list(cache.iter_trial_no_repeat())
        assert sorted(t['seed'] for t in remaining) == [0, 2]

    def test_iter_trial_no_repeat_empty_when_all_cached(self, tmp_path):
        cache = TrialCache(folder=tmp_path, iter_kwargs={'seed': [0, 1]})
        for trial in cache.iter_trial():
            cache.save_result({'v': trial['seed']}, trial)
        assert list(cache.iter_trial_no_repeat()) == []

    def test_complex_kwarg_affects_cache_key(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        cache.save_result({'val': 1}, {'ds': _Toy(1, 1), 'seed': 0})
        assert cache.is_cached({'ds': _Toy(1, 1), 'seed': 0}) is True
        assert cache.is_cached({'ds': _Toy(2, 2), 'seed': 0}) is False

    def test_cache_survives_re_open(self, tmp_path):
        TrialCache(folder=tmp_path).save_result({'v': 1}, {'seed': 0})
        cache2 = TrialCache(folder=tmp_path)
        assert cache2.is_cached({'seed': 0}) is True


class TestSaveResult:
    def test_save_dict_writes_csv(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        cache.save_result({'metric': 0.9}, {'seed': 0, 'llr': 0.1})
        assert (tmp_path / 'results.csv').exists()
        assert len(cache.df) == 1
        row = cache.df.iloc[0]
        assert row['seed'] == 0
        assert row['llr'] == 0.1
        assert row['metric'] == 0.9

    def test_save_dataframe_emits_one_row_per_record(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        result = pd.DataFrame([{'k': 'a', 'v': 1}, {'k': 'b', 'v': 2}])
        cache.save_result(result, {'seed': 0})
        assert len(cache.df) == 2
        assert sorted(cache.df['k']) == ['a', 'b']
        assert (cache.df['seed'] == 0).all()
        # both rows came from one trial → share index value
        assert cache.df.index.nunique() == 1
        assert cache.df.index.name == HASH_COL

    def test_save_appends_across_trials(self, tmp_path):
        cache = TrialCache(folder=tmp_path, iter_kwargs={'seed': [0, 1, 2]})
        for trial in cache.iter_trial():
            cache.save_result({'v': trial['seed'] ** 2}, trial)
        assert len(cache.df) == 3
        assert sorted(cache.df['v']) == [0, 1, 4]
        on_disk = pd.read_csv(tmp_path / 'results.csv')
        assert len(on_disk) == 3

    def test_invalid_result_type_raises(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        with pytest.raises(TypeError):
            cache.save_result(42, {'seed': 0})

    def test_complex_kwarg_stored_as_hash_string(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        cache.save_result({'v': 1}, {'ds': _Toy(1, 2), 'seed': 0})
        ds_val = cache.df.iloc[0]['ds']
        assert isinstance(ds_val, str) and len(ds_val) == 16

    def test_ndarray_kwarg_stored_as_hash_string(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        cache.save_result({'v': 1}, {'x': np.arange(6), 'seed': 0})
        x_val = cache.df.iloc[0]['x']
        assert isinstance(x_val, str) and len(x_val) == 16

    def test_trial_hash_is_dataframe_index(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        cache.save_result({'v': 1}, {'seed': 0})
        assert cache.df.index.name == HASH_COL
        assert HASH_COL not in cache.df.columns

    def test_trial_hash_round_trips_through_csv(self, tmp_path):
        cache1 = TrialCache(folder=tmp_path)
        cache1.save_result({'v': 1}, {'seed': 0})
        cache2 = TrialCache(folder=tmp_path)
        assert cache2.df.index.name == HASH_COL
        assert list(cache2.df.index) == list(cache1.df.index)
