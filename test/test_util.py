"""Tests for glow.util: hash_array, value_id, and stable_hash."""

import numpy as np
import pytest

from glow.util import HASH_SAMPLE_SIZE, hash_array, stable_hash, value_id


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
# value_id / stable_hash
# ---------------------------------------------------------------------------

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
