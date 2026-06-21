import json
import threading
import uuid

import pytest

from glow.benchmark.recorder import Recorder


@pytest.fixture
def rec():
	return Recorder()


# --- example pipeline (mirrors the original sketch) -------------------------

def make_pipeline(rec):
	"""Build the big_func -> part0 -> part1 pipeline against a given recorder."""

	@rec(output_name='e')
	def part0(c, d=3):
		return c * d

	@rec(output_name='g')
	def part1(e, f):
		return e + f

	@rec(output_name='big_out')
	def big_func(a, b=10):
		variable_c = a + b
		e = part0(variable_c)
		g = part1(e, f=123)
		return g * 100

	return big_func, part0, part1


# --- basic capture ----------------------------------------------------------

def test_single_output_and_defaults(rec):
	@rec(output_name='out')
	def f(a, b=10):
		return a + b

	assert f(5) == 15
	assert len(rec.records) == 1
	(record,) = rec.records
	# function identity is the qualname; locally-defined fns carry a <locals> prefix
	assert record["function"].split(".")[-1] == "f"
	assert record["inputs"] == {"a": 5, "b": 10}
	assert record["outputs"] == {"out": 15}
	# trial_ids are uuid4 strings
	uuid.UUID(record["trial_id"])


def test_explicit_kwarg_overrides_default(rec):
	@rec(output_name='out')
	def f(a, b=10):
		return a + b

	f(5, b=2)
	assert rec.records[0]["inputs"] == {"a": 5, "b": 2}


def test_output_name_list_unpacks_tuple(rec):
	@rec(output_name_list=('lo', 'hi'))
	def f(x):
		return (x - 1, x + 1)

	assert f(5) == (4, 6)
	assert rec.records[0]["outputs"] == {"lo": 4, "hi": 6}


def test_output_name_stores_tuple_whole(rec):
	# single output_name keeps the return as-is, even when it's a tuple
	@rec(output_name='pair')
	def f(x):
		return (x, x)

	f(7)
	assert rec.records[0]["outputs"] == {"pair": (7, 7)}


# --- run / trial_id semantics -----------------------------------------------

def test_nested_calls_share_trial_id(rec):
	big_func, _, _ = make_pipeline(rec)

	assert big_func(a=3) == 16200
	trial_ids = {r["trial_id"] for r in rec.records}
	assert len(trial_ids) == 1
	funcs = [r["function"].split(".")[-1] for r in rec.records]
	# inner calls complete before the outer one
	assert funcs == ["part0", "part1", "big_func"]


def test_separate_top_level_calls_get_distinct_trial_ids(rec):
	big_func, _, _ = make_pipeline(rec)

	big_func(a=3)
	big_func(a=3)
	big_ids = [r["trial_id"] for r in rec.records if r["function"].split(".")[-1] == "big_func"]
	assert big_ids[0] != big_ids[1]


def test_run_uses_explicit_trial_id(rec):
	big_func, _, _ = make_pipeline(rec)

	with rec.run(trial_id='asdf'):
		big_func(1, b=2)
	assert all(r["trial_id"] == 'asdf' for r in rec.records)


def test_run_groups_multiple_top_level_calls(rec):
	@rec(output_name='out')
	def f(a):
		return a

	with rec.run() as trial_id:
		f(1)
		f(2)
	assert [r["trial_id"] for r in rec.records] == [trial_id, trial_id]


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
	assert rec.records == []  # nothing recorded on failure


def test_output_name_list_requires_sequence_return(rec):
	@rec(output_name_list=('a', 'b'))
	def f():
		return 5  # not a tuple/list

	with pytest.raises(TypeError):
		f()


# --- variadic parameters ----------------------------------------------------

def test_var_keyword_spliced_up_a_level(rec):
	@rec(output_name='out')
	def f(a, b=10, **kwargs):
		return a

	f(1, b=2, x=99, y=100)
	# x and y are recorded as first-class named inputs, not nested under "kwargs"
	assert rec.records[0]["inputs"] == {"a": 1, "b": 2, "x": 99, "y": 100}


def test_var_keyword_empty_when_unused(rec):
	@rec(output_name='out')
	def f(a, **kwargs):
		return a

	f(1)
	assert rec.records[0]["inputs"] == {"a": 1}


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
	assert rec.records[0]["inputs"] == {"a": 1, "b": 2, "z": 3}


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
	# qualname disambiguates same-named functions (e.g. methods on different classes)
	class A:
		@rec(output_name='out')
		def go(self):
			return 1

	class B:
		@rec(output_name='out')
		def go(self):
			return 2

	A().go()
	B().go()
	assert [r["function"] for r in rec.records] == [
		"test_function_identity_uses_qualname.<locals>.A.go",
		"test_function_identity_uses_qualname.<locals>.B.go",
	]


# --- failure cleanup ---------------------------------------------------------

def test_exception_cleans_up_trial_id_and_records_nothing(rec):
	@rec(output_name='x')
	def boom():
		raise ValueError("boom")

	with pytest.raises(ValueError):
		boom()

	assert rec.records == []
	# trial id must be released so the next top-level call starts fresh
	assert rec._trial_id_current.get() is None

	@rec(output_name='y')
	def ok():
		return 1

	ok()
	assert len(rec.records) == 1


# --- export ------------------------------------------------------------------

def test_to_json_string_and_file(rec, tmp_path):
	@rec(output_name='out')
	def f(a):
		return a * 2

	f(21)

	loaded = json.loads(rec.to_json())
	assert loaded[0]["outputs"]["out"] == 42

	path = tmp_path / "rec.json"
	rec.to_json(file=str(path))
	on_disk = json.loads(path.read_text())
	assert on_disk == loaded


def test_to_json_falls_back_to_repr(rec):
	class Thing:
		def __repr__(self):
			return "<Thing>"

	@rec(output_name='out')
	def f():
		return Thing()

	f()
	loaded = json.loads(rec.to_json())
	assert loaded[0]["outputs"]["out"] == "<Thing>"


# --- concurrency -------------------------------------------------------------

def test_concurrent_runs_do_not_bleed_trial_ids(rec):
	@rec(output_name='out')
	def f(x):
		return x

	barrier = threading.Barrier(2)

	def worker(trial_id):
		with rec.run(trial_id=trial_id):
			barrier.wait()  # force the two runs to overlap
			for i in range(50):
				f(i)

	t1 = threading.Thread(target=worker, args=('A',))
	t2 = threading.Thread(target=worker, args=('B',))
	t1.start(); t2.start()
	t1.join(); t2.join()

	a = [r for r in rec.records if r["trial_id"] == 'A']
	b = [r for r in rec.records if r["trial_id"] == 'B']
	assert len(a) == 50 and len(b) == 50
	assert len(rec.records) == 100  # every call accounted for, none cross-contaminated
