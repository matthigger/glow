"""Tests for glow.util: hash_array and HashBySlots."""

import numpy as np
import pytest

from glow.util import (
    HASH_SAMPLE_SIZE, HashBySlots, hash_array, stable_hash, value_id,
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
# HashBySlots — exercised via a tiny test subclass
# ---------------------------------------------------------------------------

class _Toy(HashBySlots):
    __slots__ = ('a', 'b')

    def __init__(self, a, b):
        self.a = a
        self.b = b


class _ToyOther(HashBySlots):
    """different class with same slot names — must hash distinctly."""
    __slots__ = ('a', 'b')

    def __init__(self, a, b):
        self.a = a
        self.b = b


class _Nested(HashBySlots):
    __slots__ = ('inner', 'label')

    def __init__(self, inner, label):
        self.inner = inner
        self.label = label


class _WithArray(HashBySlots):
    __slots__ = ('arr', 'tag')

    def __init__(self, arr, tag):
        self.arr = arr
        self.tag = tag


class TestHashBySlotsBasic:
    @pytest.mark.parametrize('other, equal', [
        (_Toy(1, 2), True),   # equal slots → equal value and hash
        (_Toy(1, 3), False),  # differing slot → distinct value and hash
    ])
    def test_eq_and_hash_follow_slots(self, other, equal):
        ref = _Toy(1, 2)
        assert (ref == other) is equal
        assert (hash(ref) == hash(other)) is equal

    def test_neq_other_subclass(self):
        # same slot names, different class → not equal, different hash
        assert _Toy(1, 2) != _ToyOther(1, 2)
        assert hash(_Toy(1, 2)) != hash(_ToyOther(1, 2))

    def test_neq_unrelated_type(self):
        assert _Toy(1, 2) != (1, 2)
        assert _Toy(1, 2) != {'a': 1, 'b': 2}


class TestHashBySlotsCanon:
    def test_ndarray_slot_routes_through_hash_array(self):
        # equal-content array slots compare equal; a content change is
        # detected — i.e. the slot is canonicalized via hash_array
        a = np.arange(10)
        b = np.arange(10)
        assert _WithArray(a, 't') == _WithArray(b, 't')
        assert hash(_WithArray(a, 't')) == hash(_WithArray(b, 't'))
        b[0] = -1
        assert _WithArray(a, 't') != _WithArray(b, 't')

    def test_nested_hashbyslots(self):
        inner = _Toy(1, 2)
        x = _Nested(inner, 'L')
        y = _Nested(_Toy(1, 2), 'L')
        assert x == y
        assert hash(x) == hash(y)

    def test_nested_inner_change_propagates(self):
        x = _Nested(_Toy(1, 2), 'L')
        y = _Nested(_Toy(1, 3), 'L')
        assert x != y
        assert hash(x) != hash(y)


class TestHashBySlotsMutation:
    """The mixin doesn't pretend to be frozen — hash follows the slots.
    Document that contract."""

    def test_hash_changes_when_slot_mutated(self):
        t = _Toy(1, 2)
        h0 = hash(t)
        t.a = 99
        h1 = hash(t)
        assert h0 != h1


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
        # HashBySlots routes through its stable identity hash
        (_Toy(1, 2), value_id(_Toy(1, 2))),
    ])
    def test_dispatch(self, value, expected):
        assert value_id(value) == expected

    def test_numpy_scalar_is_native(self):
        assert not isinstance(value_id(np.float64(1.5)), np.floating)


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

# Subclass hierarchy + a class with an underscore slot, used to exercise
# the MRO walk and the _-prefix skip in HashBySlots._identity_dict.
class _ToyBase(HashBySlots):
    __slots__ = ('a',)

    def __init__(self, a):
        self.a = a


class _ToySub(_ToyBase):
    __slots__ = ('b',)

    def __init__(self, a, b):
        super().__init__(a)
        self.b = b


class TestHashBySlotsMroWalk:
    """identity walks the full MRO so subclass identity covers base slots
    (otherwise a subclass with extra slots would silently drop its base
    identity)."""

    def test_base_slot_change_changes_hash(self):
        # a base-class slot participates in the subclass's identity
        assert _ToySub(1, 2) != _ToySub(9, 2)
        assert hash(_ToySub(1, 2)) != hash(_ToySub(9, 2))

    def test_base_and_subclass_distinct(self):
        # _ToyBase(1) and _ToySub(1, 2) must not collide even when their
        # base slot agrees — class name differs, and __eq__ checks type
        assert _ToyBase(1) != _ToySub(1, 2)
        assert hash(_ToyBase(1)) != hash(_ToySub(1, 2))


class TestHashBySlotsRequiresSlots:
    """__init_subclass__ rejects subclasses that don't declare __slots__,
    so identity is never silently empty.  This fires at any depth."""

    def test_subclass_without_slots_errors_at_any_depth(self):
        # direct subclass of HashBySlots
        with pytest.raises(TypeError, match='__slots__'):
            class _NoSlots(HashBySlots):
                pass

        # and a deeper subclass of a slotted intermediate
        class _Mid(HashBySlots):
            __slots__ = ('x',)

        with pytest.raises(TypeError, match='__slots__'):
            class _Leaf(_Mid):
                pass

    def test_subclass_with_empty_slots_ok(self):
        # explicit empty tuple is the documented opt-out for "no new
        # slots at this layer"
        class _Empty(HashBySlots):
            __slots__ = ()

        assert hash(_Empty()) == hash(_Empty())