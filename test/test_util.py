"""Tests for glow.util: hash_array and HashBySlots."""

import numpy as np
import pytest

from glow.util import HASH_SAMPLE_SIZE, HashBySlots, hash_array


# ---------------------------------------------------------------------------
# hash_array
# ---------------------------------------------------------------------------

class TestHashArray:
    def test_deterministic(self):
        a = np.arange(20).astype(np.float64)
        assert hash_array(a) == hash_array(a)

    def test_same_content_same_hash(self):
        a = np.arange(20).astype(np.float64)
        b = np.arange(20).astype(np.float64)
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

    def test_returns_16char_hex(self):
        h = hash_array(np.arange(5))
        assert isinstance(h, str)
        assert len(h) == 16
        int(h, 16)  # raises ValueError if not valid hex

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
    def test_equal_when_slots_equal(self):
        assert _Toy(1, 2) == _Toy(1, 2)
        assert hash(_Toy(1, 2)) == hash(_Toy(1, 2))

    def test_differ_when_slot_differs(self):
        assert _Toy(1, 2) != _Toy(1, 3)
        assert hash(_Toy(1, 2)) != hash(_Toy(1, 3))

    def test_neq_other_subclass(self):
        # same slot names, different class → not equal, different hash
        assert _Toy(1, 2) != _ToyOther(1, 2)
        assert hash(_Toy(1, 2)) != hash(_ToyOther(1, 2))

    def test_neq_unrelated_type(self):
        assert _Toy(1, 2) != (1, 2)
        assert _Toy(1, 2) != {'a': 1, 'b': 2}

    def test_dict_key(self):
        d = {_Toy(1, 2): 'x'}
        assert d[_Toy(1, 2)] == 'x'

    def test_slots_block_stray_attrs(self):
        t = _Toy(1, 2)
        with pytest.raises(AttributeError):
            t.weird = 99

    def test_identity_dict_contains_kind(self):
        d = _Toy(1, 2)._identity_dict()
        assert d['kind'] == '_Toy'
        assert d['a'] == 1
        assert d['b'] == 2


class TestHashBySlotsCanon:
    def test_ndarray_slot(self):
        a = np.arange(10)
        b = np.arange(10)
        assert _WithArray(a, 't') == _WithArray(b, 't')
        assert hash(_WithArray(a, 't')) == hash(_WithArray(b, 't'))

    def test_ndarray_content_change(self):
        a = np.arange(10)
        b = a.copy()
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
