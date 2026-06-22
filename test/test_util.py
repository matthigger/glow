"""Tests for glow.util: hash_array and the DataclassJSON mixin."""

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

from glow.util import (
    HASH_SAMPLE_SIZE, DataclassJSON, hash_array, stable_hash, value_id,
)


# ---------------------------------------------------------------------------
# hash_array
# ---------------------------------------------------------------------------

class TestHashArray:
    def test_same_content_same_hash(self):
        # also covers determinism: equal-content arrays (incl. the same
        # array hashed twice) hash identically
        a = np.arange(20).astype(np.float64)
        b = np.arange(20).astype(np.float64)
        assert hash_array(a) == hash_array(a)
        assert hash_array(a) == hash_array(b)

    def test_different_content_different_hash(self):
        a = np.arange(20).astype(np.float64)
        b = np.arange(20).astype(np.float64)
        b[0] = -1
        assert hash_array(a) != hash_array(b)

    def test_shape_change_changes_hash(self):
        # same flat bytes, different shape → different hash
        a = np.arange(12).astype(np.float64).reshape(3, 4)
        b = np.arange(12).astype(np.float64).reshape(4, 3)
        assert hash_array(a) != hash_array(b)

    def test_dtype_change_changes_hash(self):
        # same logical contents, different dtype → different hash
        a = np.arange(12, dtype=np.float64)
        b = np.arange(12, dtype=np.float32)
        assert hash_array(a) != hash_array(b)

    def test_multi_array(self):
        a = np.arange(5)
        b = np.arange(7)
        # combined hash differs from either single-array hash
        assert hash_array(a, b) != hash_array(a)
        assert hash_array(a, b) != hash_array(b)
        # deterministic
        assert hash_array(a, b) == hash_array(a, b)

    def test_multi_array_order_matters(self):
        a = np.arange(5)
        b = np.arange(7)
        assert hash_array(a, b) != hash_array(b, a)

    def test_non_contiguous_input(self):
        # slicing produces a non-contiguous view; hash_array handles it
        full = np.arange(100).reshape(10, 10)
        view = full[::2, ::2]
        assert not view.flags.c_contiguous
        # equal contents (forced to contiguous) → equal hashes
        assert hash_array(view) == hash_array(np.ascontiguousarray(view))

    def test_large_array_sampled(self):
        # array larger than HASH_SAMPLE_SIZE triggers the sampling path
        n = HASH_SAMPLE_SIZE * 3
        a = np.arange(n).astype(np.float64)
        # deterministic across calls (sampling seed is fixed at 0)
        assert hash_array(a) == hash_array(a)
        # changing a sampled position changes the hash
        b = a.copy()
        # bump every position the sampler would have hit
        idx = np.random.default_rng(0).integers(0, n, HASH_SAMPLE_SIZE)
        b[idx] += 1
        assert hash_array(a) != hash_array(b)


# ---------------------------------------------------------------------------
# DataclassJSON — frozen dataclasses (native __hash__ / __eq__) + to_dict/json
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class _Toy(DataclassJSON):
    a: object
    b: object


@dataclass(frozen=True, slots=True)
class _ToyOther(DataclassJSON):
    """different class with the same field names."""
    a: object
    b: object


@dataclass(frozen=True, slots=True)
class _Nested(DataclassJSON):
    inner: object
    label: object


@dataclass(frozen=True, slots=True)
class _ToyBase(DataclassJSON):
    a: object


@dataclass(frozen=True, slots=True)
class _ToySub(_ToyBase):
    b: object


class TestNativeHashEq:
    @pytest.mark.parametrize('other, equal', [
        (_Toy(1, 2), True),   # equal fields → equal value and hash
        (_Toy(1, 3), False),  # differing field → distinct value and hash
    ])
    def test_eq_and_hash_follow_fields(self, other, equal):
        ref = _Toy(1, 2)
        assert (ref == other) is equal
        assert (hash(ref) == hash(other)) is equal

    def test_neq_other_class(self):
        # same field names + values, different class → not equal, and the
        # stable id (to_json, folds in 'kind') stays distinct. The builtin
        # hash may collide (it excludes the class), which is harmless: __eq__
        # still separates them in dicts / sets.
        assert _Toy(1, 2) != _ToyOther(1, 2)
        assert _Toy(1, 2).to_json() != _ToyOther(1, 2).to_json()

    def test_neq_unrelated_type(self):
        assert _Toy(1, 2) != (1, 2)
        assert _Toy(1, 2) != {'a': 1, 'b': 2}

    def test_usable_as_collection_key(self):
        d = {_Toy(1, 2): 'x'}
        assert d[_Toy(1, 2)] == 'x'
        assert len({_Toy(1, 2), _Toy(1, 2)}) == 1


class TestNested:
    def test_nested_spec_in_identity(self):
        x = _Nested(_Toy(1, 2), 'L')
        y = _Nested(_Toy(1, 2), 'L')
        assert x == y
        assert hash(x) == hash(y)

    def test_nested_inner_change_propagates(self):
        x = _Nested(_Toy(1, 2), 'L')
        y = _Nested(_Toy(1, 3), 'L')
        assert x != y
        assert hash(x) != hash(y)

    def test_to_dict_recurses_nested(self):
        d = _Nested(_Toy(1, 2), 'L').to_dict()
        assert d['kind'] == '_Nested'
        assert d['label'] == 'L'
        assert d['inner'] == {'kind': '_Toy', 'a': 1, 'b': 2}


class TestInheritance:
    """native __hash__ / __eq__ cover inherited fields, so a subclass's
    identity includes its base field."""

    def test_base_field_in_identity(self):
        assert _ToySub(1, 2) != _ToySub(9, 2)
        assert hash(_ToySub(1, 2)) != hash(_ToySub(9, 2))

    def test_base_and_subclass_distinct(self):
        # _ToyBase(1) and _ToySub(1, 2) are different classes
        assert _ToyBase(1) != _ToySub(1, 2)
        assert _ToyBase(1).to_json() != _ToySub(1, 2).to_json()

    def test_to_dict_includes_base_field(self):
        d = _ToySub(1, 2).to_dict()
        assert d == {'kind': '_ToySub', 'a': 1, 'b': 2}


class TestFrozen:
    def test_cannot_reassign_field(self):
        t = _Toy(1, 2)
        with pytest.raises(FrozenInstanceError):
            t.a = 99

    def test_no_instance_dict(self):
        # slots=True + the mixin's __slots__ = () means no __dict__
        assert not hasattr(_Toy(1, 2), '__dict__')

    def test_hash_stable_over_lifetime(self):
        t = _Toy(1, 2)
        assert hash(t) == hash(t)


class TestToJson:
    def test_to_dict_has_kind_and_fields(self):
        assert _Toy(1, 2).to_dict() == {'kind': '_Toy', 'a': 1, 'b': 2}

    def test_to_json_is_sorted_json(self):
        import json
        s = _Toy(1, 2).to_json()
        assert json.loads(s) == {'kind': '_Toy', 'a': 1, 'b': 2}
        # sort_keys → deterministic across processes
        assert _Toy(1, 2).to_json() == _Toy(1, 2).to_json()


class TestValueId:
    """value_id dispatch table: each value kind routes to its identity."""

    def test_unknown_type_falls_back_to_repr(self):
        class Other:
            def __repr__(self):
                return 'OTHER'
        assert value_id(Other()) == 'OTHER'

    @pytest.mark.parametrize('value, expected', [
        # scalar passthrough
        (0, 0),
        ('x', 'x'),
        (None, None),
        (True, True),
        # numpy scalar is unboxed to a native Python scalar
        (np.float64(1.5), 1.5),
        # ndarray routes through hash_array (16-hex digest)
        (np.arange(10), hash_array(np.arange(10))),
        # DataclassJSON routes through its stable identity hash
        (_Toy(1, 2), value_id(_Toy(1, 2))),
    ])
    def test_dispatch(self, value, expected):
        assert value_id(value) == expected

    def test_numpy_scalar_is_native(self):
        assert not isinstance(value_id(np.float64(1.5)), np.floating)

    def test_dataclass_value_id_class_distinct(self):
        # value_id folds in 'kind', so same-field different-class specs differ
        assert value_id(_Toy(1, 2)) != value_id(_ToyOther(1, 2))


class TestStableHash:
    def test_same_dict_same_hash(self):
        d = {'seed': 0, 'llr': 0.1}
        assert stable_hash(d) == stable_hash(d)

    def test_key_order_independent(self):
        a = {'seed': 0, 'llr': 0.1}
        b = {'llr': 0.1, 'seed': 0}
        assert stable_hash(a) == stable_hash(b)

    def test_different_values_different_hash(self):
        # the only stable_hash-specific behavior: dict values flow through
        # value_id (proven in TestValueId), so a value change changes the hash
        assert stable_hash({'seed': 0}) != stable_hash({'seed': 1})
