"""Tests for glow._extra.benchmark.recorder: the args-hash-keyed Recorder.

The Recorder stores one record per *successful* decorated call, keyed by
joblib's own args hash (``joblib.hash(filter_args(fnc, [], args, kwargs))``) so
a record matches the cache entry the same call writes. A repeat hash overwrites
the prior record and warns; a call that raises propagates and records nothing;
and -- when a folder is given -- each record mirrors to ``<hash>.json``. These
tests exercise capture, the keying, the overwrite / failure semantics, and the
on-disk persistence.
"""
import json

import joblib
import pytest
from joblib.func_inspect import filter_args

from glow._extra.benchmark.recorder import Recorder


@pytest.fixture
def rec():
    return Recorder()


def _only(records):
    """The sole record in a (single-entry) records dict."""
    (record,) = records.values()
    return record


# Module-level receiver classes for the bound-method tests: the key hashes the
# receiver (filter_args includes it), and joblib.hash pickles it -- so, exactly
# as with joblib.Memory, the receiver must be picklable (a class defined inside
# a test function is not).

class _Thing:
    def __init__(self, k):
        self.k = k
    def go(self, x):
        return self.k + x


class _A:
    def go(self):
        return 1


class _B:
    def go(self):
        return 2


# --- basic capture ----------------------------------------------------------

def test_single_output_and_defaults(rec):
    @rec(output_name='out')
    def f(a, b=10):
        return a + b

    assert f(5) == 15
    assert len(rec.records) == 1
    record = _only(rec.records)
    # function identity is the qualname; locally-defined fns carry a <locals> prefix
    assert record["function"].split(".")[-1] == "f"
    assert record["inputs"] == {"a": 5, "b": 10}
    assert record["outputs"] == {"out": 15}
    # every success record is timed
    assert isinstance(record["time_sec"], float) and record["time_sec"] >= 0
    # the record carries its hash, and that hash is its dict key
    assert record["hash"] in rec.records


def test_explicit_kwarg_overrides_default(rec):
    @rec(output_name='out')
    def f(a, b=10):
        return a + b

    f(5, b=2)
    assert _only(rec.records)["inputs"] == {"a": 5, "b": 2}


def test_output_name_list_unpacks_tuple(rec):
    @rec(output_name_list=('lo', 'hi'))
    def f(x):
        return (x - 1, x + 1)

    assert f(5) == (4, 6)
    assert _only(rec.records)["outputs"] == {"lo": 4, "hi": 6}


def test_output_name_stores_tuple_whole(rec):
    # single output_name keeps the return as-is, even when it's a tuple
    @rec(output_name='pair')
    def f(x):
        return (x, x)

    f(7)
    assert _only(rec.records)["outputs"] == {"pair": (7, 7)}


# --- keying (joblib args hash) ----------------------------------------------

def test_record_keyed_by_joblib_args_hash(rec):
    # the key is exactly joblib.hash(filter_args(raw_fn, [], args, kwargs)),
    # i.e. the id joblib.Memory would file the same call under
    @rec(output_name='out')
    def f(a, b=10):
        return a + b

    f(5, b=2)
    key = joblib.hash(filter_args(f.__wrapped__, [], (5,), {'b': 2}))
    assert list(rec.records) == [key]
    assert rec.records[key]["hash"] == key
    # and the keying staticmethod agrees with the wrapper's keying
    assert Recorder._args_hash(f.__wrapped__, (5,), {'b': 2}) == key


def test_distinct_args_distinct_records(rec):
    @rec(output_name='out')
    def f(a):
        return a

    f(1)
    f(2)
    assert len(rec.records) == 2
    assert {r["outputs"]["out"] for r in rec.records.values()} == {1, 2}


def test_same_args_overwrites_and_warns(rec):
    # same fn + same args -> same hash -> the second call overwrites the first
    # (latest kept) and warns, mirroring joblib.Memory's single cache entry
    calls = []

    @rec(output_name='out')
    def f(a):
        calls.append(a)
        return len(calls)   # 1 on the first call, 2 on the second

    assert f(1) == 1
    with pytest.warns(UserWarning, match='overwriting'):
        assert f(1) == 2
    assert len(rec.records) == 1
    assert _only(rec.records)["outputs"] == {"out": 2}   # latest wins


# --- label tagging ----------------------------------------------------------

def test_label_recorded_top_level(rec):
    # label names the method/variant this call belongs to; it lands as a
    # top-level field on the record (not nested under inputs)
    @rec(output_name='out', label='GLOW-GLM')
    def f(a):
        return a

    f(5)
    record = _only(rec.records)
    assert record['label'] == 'GLOW-GLM'
    assert 'label' not in record['inputs']


def test_label_absent_when_not_given(rec):
    # an unlabelled call carries no label key
    @rec(output_name='out')
    def f(a):
        return a

    f(5)
    assert 'label' not in _only(rec.records)


def test_label_with_output_name_list(rec):
    @rec(output_name_list=('lo', 'hi'), label='prune')
    def f(x):
        return (x - 1, x + 1)

    f(5)
    record = _only(rec.records)
    assert record['label'] == 'prune'
    assert record['outputs'] == {'lo': 4, 'hi': 6}


def test_label_must_be_str(rec):
    with pytest.raises(TypeError):
        @rec(output_name='out', label=123)
        def f():
            return 1


def test_label_not_part_of_key(rec):
    # label is metadata, not part of the hash: two distinct functions with the
    # same filtered args (here, both `(a=1)`) collide regardless of label --
    # the flat map does not namespace by function, so the overwrite-warning is
    # the guard (see module docstring)
    @rec(output_name='out', label='A')
    def f(a):
        return a

    @rec(output_name='out', label='B')
    def g(a):
        return a

    f(1)
    with pytest.warns(UserWarning, match='overwriting'):
        g(1)
    assert len(rec.records) == 1


# --- decoration-time validation --------------------------------------------

def test_requires_exactly_one_of_output_args(rec):
    with pytest.raises(ValueError):
        @rec()
        def f():
            return 1

    with pytest.raises(ValueError):
        @rec(output_name='a', output_name_list=('a',))
        def g():
            return 1


def test_duplicate_output_names_rejected_at_decoration(rec):
    with pytest.raises(ValueError):
        @rec(output_name_list=('x', 'x'))
        def f():
            return (1, 2)


def test_output_name_must_be_str(rec):
    with pytest.raises(TypeError):
        @rec(output_name=123)
        def f():
            return 1


def test_empty_output_name_list_rejected(rec):
    with pytest.raises(TypeError):
        @rec(output_name_list=())
        def f():
            return ()


# --- runtime output validation ----------------------------------------------

def test_output_name_list_length_mismatch(rec):
    @rec(output_name_list=('a', 'b'))
    def f():
        return (1, 2, 3)

    with pytest.raises(ValueError):
        f()
    assert rec.records == {}  # misconfiguration is not recorded


def test_output_name_list_requires_sequence_return(rec):
    @rec(output_name_list=('a', 'b'))
    def f():
        return 5  # not a tuple/list

    with pytest.raises(TypeError):
        f()
    assert rec.records == {}


# --- variadic parameters ----------------------------------------------------

def test_var_keyword_spliced_up_a_level(rec):
    @rec(output_name='out')
    def f(a, b=10, **kwargs):
        return a

    f(1, b=2, x=99, y=100)
    # x and y are recorded as first-class named inputs, not nested under "kwargs"
    assert _only(rec.records)["inputs"] == {"a": 1, "b": 2, "x": 99, "y": 100}


def test_var_keyword_empty_when_unused(rec):
    @rec(output_name='out')
    def f(a, **kwargs):
        return a

    f(1)
    assert _only(rec.records)["inputs"] == {"a": 1}


def test_var_positional_rejected_at_decoration(rec):
    with pytest.raises(TypeError):
        @rec(output_name='out')
        def f(a, *args):
            return a


def test_keyword_only_params_recorded_normally(rec):
    # keyword-only params are named, so they bind to their own name (not **kwargs)
    @rec(output_name='out')
    def f(a, *, b=10, **kwargs):
        return a

    f(1, b=2, z=3)
    assert _only(rec.records)["inputs"] == {"a": 1, "b": 2, "z": 3}


# --- introspection (functools.wraps) ----------------------------------------

def test_wraps_preserves_metadata_and_signature(rec):
    import inspect

    @rec(output_name='out')
    def f(a, b=10):
        """docstring."""
        return a + b

    assert f.__name__ == "f"
    assert f.__doc__ == "docstring."
    assert list(inspect.signature(f).parameters) == ["a", "b"]


def test_function_identity_uses_qualname(rec):
    # qualname disambiguates same-named methods on different classes; the
    # receiver (different classes) also keys the two records distinctly
    rec(output_name='out')(_A().go)()
    rec(output_name='out')(_B().go)()
    funcs = sorted(r["function"] for r in rec.records.values())
    assert funcs == ["_A.go", "_B.go"]


# --- method / self capture ---------------------------------------------------

def test_bound_method_captures_self(rec):
    # decorating a bound method inline records its receiver as the 'self' input
    # (the bound signature omits self, so it is taken from __self__)
    assert rec(output_name='r')(_Thing(10).go)(5) == 15

    record = _only(rec.records)
    assert record['function'].split('.')[-1] == 'go'
    assert record['outputs']['r'] == 15
    # in memory the captured self is the live object
    assert isinstance(record['inputs']['self'], _Thing)
    assert record['inputs']['x'] == 5


def test_distinct_receivers_distinct_records(rec):
    # filter_args hashes the receiver, so two receivers in different states key
    # to two records even though the (bound) call args are identical
    rec(output_name='r')(_Thing(1).go)(0)
    rec(output_name='r')(_Thing(2).go)(0)
    assert len(rec.records) == 2


def test_plain_function_has_no_self(rec):
    # a non-method recorded call gets no injected 'self'
    @rec(output_name='out')
    def f(x):
        return x

    f(7)
    assert 'self' not in _only(rec.records)['inputs']


# --- failure capture ---------------------------------------------------------

def test_exception_propagates_and_records_nothing(rec):
    # a call that raises propagates the exception (no swallowing) and records
    # nothing, like joblib.Memory on a failed call
    @rec(output_name='x')
    def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        boom()
    assert rec.records == {}


def test_output_validation_error_is_not_recorded(rec):
    # the fnc succeeds; the recorder's own arity check raises afterwards, so
    # nothing is recorded (it's misconfiguration, not a recorded result)
    @rec(output_name_list=('a', 'b'))
    def f():
        return (1, 2, 3)

    with pytest.raises(ValueError):
        f()
    assert rec.records == {}


def test_failure_writes_no_file(tmp_path):
    rec = Recorder(folder=tmp_path)

    @rec(output_name='x')
    def boom():
        raise ValueError('x')

    with pytest.raises(ValueError):
        boom()
    assert list(tmp_path.glob('*.json')) == []


# --- persistence (per-hash files) --------------------------------------------

def test_record_mirrored_to_per_hash_file(tmp_path):
    rec = Recorder(folder=tmp_path)

    @rec(output_name='out')
    def f(a, b=10):
        return a + b

    f(5)
    record = _only(rec.records)
    f_json = tmp_path / f"{record['hash']}.json"
    assert f_json.exists()
    on_disk = json.loads(f_json.read_text())
    assert on_disk['hash'] == record['hash']
    assert on_disk['inputs'] == {'a': 5, 'b': 10}
    assert on_disk['outputs'] == {'out': 15}


def test_load_reads_per_hash_files(tmp_path):
    rec = Recorder(folder=tmp_path)

    @rec(output_name='out')
    def f(a):
        return a

    f(1)
    f(2)
    keys = set(rec.records)

    # a fresh recorder over the same folder reconstitutes the records via load()
    rec2 = Recorder(folder=tmp_path)
    loaded = rec2.load()
    assert loaded is rec2.records
    assert set(loaded) == keys
    assert {r['outputs']['out'] for r in loaded.values()} == {1, 2}


def test_overwrite_replaces_file(tmp_path):
    rec = Recorder(folder=tmp_path)
    calls = []

    @rec(output_name='out')
    def f(a):
        calls.append(a)
        return len(calls)

    f(1)
    with pytest.warns(UserWarning, match='overwriting'):
        f(1)
    files = list(tmp_path.glob('*.json'))
    assert len(files) == 1
    assert json.loads(files[0].read_text())['outputs'] == {'out': 2}


def test_non_json_output_falls_back_to_repr_on_disk(tmp_path):
    rec = Recorder(folder=tmp_path)

    class Thing:
        def __repr__(self):
            return '<Thing>'

    @rec(output_name='out')
    def f():
        return Thing()

    f()
    record = _only(rec.records)
    on_disk = json.loads((tmp_path / f"{record['hash']}.json").read_text())
    assert on_disk['outputs']['out'] == '<Thing>'


def test_in_memory_recorder_writes_no_files(rec):
    # the default (folder=None) recorder is in-memory; load() is then a no-op
    @rec(output_name='out')
    def f(a):
        return a

    f(1)
    assert rec.folder is None
    assert rec.load() is rec.records
    assert len(rec.records) == 1
