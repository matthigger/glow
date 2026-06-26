"""Tests for ``glow._extra.benchmark.TrialCache``.

The cache iterates a scalar-axis grid, hashes trials, and owns a Recorder; it
does no disk IO of its own. A trial is "done" when a per-trial record file
(records/<hash>.json) exists, which the recorder writes on flush -- so the tests
mark a trial done by recording a trivial call under its id and flushing it.
"""
from pathlib import Path

import pytest

import glow._extra.benchmark.recorder as rec_mod
from glow._extra.benchmark import TrialCache
from glow._extra.benchmark.recorder import RECORDS_DIR
from glow.util import stable_hash


class _Toy:
    """Small fixture standing in for DataSource / Effect."""
    def __init__(self, a, b):
        self.a = a
        self.b = b


def _complete(cache, trial):
    """Mark a trial done: record a trivial call under its id and flush it."""
    cache.recorder.records.clear()
    with cache.recorder.trial(trial_id=cache.hash(trial)):
        cache.recorder(output_name='x')(lambda: 1)()
    cache.recorder.flush()


class TestConstruction:
    def test_folder_kwarg_creates_dir(self, tmp_path):
        # the cache has no folder of its own; the recorder owns it (created)
        target = tmp_path / 'fresh'
        assert not target.exists()
        cache = TrialCache(folder=target)
        assert not hasattr(cache, 'folder')
        assert cache.recorder.folder == target
        assert target.is_dir()

        cache_str = TrialCache(folder=str(tmp_path / 'as_str'))
        assert isinstance(cache_str.recorder.folder, Path)
        assert cache_str.recorder.folder.is_dir()

    def test_name_kwarg_resolves_under_default_results_dir(
            self, tmp_path, monkeypatch):
        # name resolution lives in the recorder now
        monkeypatch.setattr(rec_mod, 'get_path_result', lambda: tmp_path)
        cache = TrialCache(name='exp_foo')
        assert cache.recorder.folder == tmp_path / 'exp_foo'
        assert cache.recorder.folder.is_dir()

    @pytest.mark.parametrize('kw', [
        dict(name='x', folder='FOLDER'),  # both -> rejected
        dict(),                           # neither -> rejected
    ])
    def test_exactly_one_of_name_folder_required(self, tmp_path, kw):
        if 'folder' in kw:
            kw['folder'] = tmp_path
        with pytest.raises(ValueError, match='exactly one'):
            TrialCache(**kw)

    def test_overlap_between_iter_and_const_kwargs_rejected(self, tmp_path):
        with pytest.raises(ValueError, match='share keys'):
            TrialCache(folder=tmp_path,
                       iter_kwargs={'seed': [1, 2]}, kwargs={'seed': 5})

    def test_per_trial_files_under_records_subdir(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        _complete(cache, {'seed': 0})
        # the recorder writes per-trial files to the records/ subdir
        assert (tmp_path / RECORDS_DIR).is_dir()
        assert list((tmp_path / RECORDS_DIR).glob('*.json'))


class TestIterTrial:
    def test_empty_yields_single_empty_dict(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        assert list(cache.iter_trial()) == [{}]

    def test_only_constant_kwargs_yields_one_trial(self, tmp_path):
        cache = TrialCache(folder=tmp_path, kwargs={'a': 1, 'b': 2})
        assert list(cache.iter_trial()) == [{'a': 1, 'b': 2}]

    def test_cartesian_product_count(self, tmp_path):
        cache = TrialCache(folder=tmp_path,
                           iter_kwargs={'seed': range(3), 'llr': [0.1, 0.2]})
        assert len(list(cache.iter_trial())) == 6
        assert len(cache) == 6

    def test_constant_merged_into_each_trial(self, tmp_path):
        ds = _Toy(1, 2)
        cache = TrialCache(folder=tmp_path,
                           iter_kwargs={'seed': [0, 1]}, kwargs={'ds': ds})
        out = list(cache.iter_trial())
        assert all(t['ds'] is ds for t in out)
        assert sorted(t['seed'] for t in out) == [0, 1]


class TestCompletion:
    def test_is_cached_false_before(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        assert cache.is_cached({'seed': 0}) is False

    def test_is_cached_true_after_flush(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        trial = {'seed': 0, 'llr': 0.1}
        _complete(cache, trial)
        assert cache.is_cached(trial) is True

    def test_iter_trial_skips_completed(self, tmp_path):
        cache = TrialCache(folder=tmp_path, iter_kwargs={'seed': [0, 1, 2]})
        _complete(cache, {'seed': 1})
        remaining = list(cache.iter_trial())
        assert sorted(t['seed'] for t in remaining) == [0, 2]

    def test_include_completed_yields_all(self, tmp_path):
        cache = TrialCache(folder=tmp_path, iter_kwargs={'seed': [0, 1, 2]})
        _complete(cache, {'seed': 1})
        allt = list(cache.iter_trial(include_completed=True))
        assert sorted(t['seed'] for t in allt) == [0, 1, 2]

    def test_complex_kwarg_affects_completion(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        _complete(cache, {'ds': _Toy(1, 1), 'seed': 0})
        assert cache.is_cached({'ds': _Toy(1, 1), 'seed': 0}) is True
        assert cache.is_cached({'ds': _Toy(2, 2), 'seed': 0}) is False


class TestLoadRecords:
    def test_load_and_consolidate(self, tmp_path):
        cache = TrialCache(folder=tmp_path, iter_kwargs={'seed': [0, 1]})
        for trial in ({'seed': 0}, {'seed': 1}):
            _complete(cache, trial)

        recs = cache.load_records()
        assert len(recs) == 2  # one trivial record per trial

        cache.load_records(consolidate=True)
        assert (tmp_path / 'records.json').exists()


class TestTrialAliasMap:
    """trial_alias_map redirects a trial's hash (the trial id) so a swapped
    twin keys the same record file its original would (the AWS driver use)."""

    def test_default_is_identity(self, tmp_path):
        cache = TrialCache(folder=tmp_path)
        assert cache.trial_alias_map is None
        assert cache.hash({'seed': 0}) == stable_hash({'seed': 0})

    def test_hash_follows_alias(self, tmp_path):
        new_trial = {'seed': 0, 'ds': _Toy(9, 9)}
        new_hash = stable_hash(new_trial)
        cache = TrialCache(folder=tmp_path,
                           trial_alias_map={new_hash: 'OLDHASH'})
        assert cache.hash(new_trial) == 'OLDHASH'

    def test_completion_follows_alias(self, tmp_path):
        new_trial = {'seed': 0, 'ds': _Toy(9, 9)}
        new_hash = stable_hash(new_trial)
        twin = TrialCache(folder=tmp_path,
                          iter_kwargs={'seed': [0]}, kwargs={'ds': _Toy(9, 9)},
                          trial_alias_map={new_hash: 'OLDHASH'})
        _complete(twin, new_trial)  # writes OLDHASH.json (the aliased id)
        assert twin.is_cached(new_trial) is True
        assert list(twin.iter_trial()) == []

    def test_unmapped_hash_passes_through(self, tmp_path):
        cache = TrialCache(folder=tmp_path,
                           trial_alias_map={'SOMETHING_ELSE': 'OLDHASH'})
        assert cache.hash({'seed': 0}) == stable_hash({'seed': 0})
