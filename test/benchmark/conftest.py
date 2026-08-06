"""Shared fixtures for the benchmark tests."""

import numpy as np
import pytest
from joblib import hashing


@pytest.fixture
def no_array_hashing(monkeypatch):
    """Fail the test if any ndarray reaches joblib's hasher.

    The guardrail behind glow._extra.benchmark.recipe's invariant: an identity
    (cache key, record key, provenance edge, RNG seed) may never derive from the
    bytes of a computed array, because two CPUs compute the same array to
    different last bits. Patches NumpyHasher.save, the single point every
    joblib.hash of an array passes through, so a violation raises where it
    happens rather than silently keying on bytes.
    """
    original = hashing.NumpyHasher.save

    def guarded(hasher_self, obj):
        """Raise on an ndarray, else hash as usual."""
        if isinstance(obj, np.ndarray):
            raise AssertionError(
                f'joblib hashed an ndarray (shape {obj.shape}, dtype '
                f'{obj.dtype}): identities must come from a declared recipe, '
                f'not from array bytes')
        return original(hasher_self, obj)

    monkeypatch.setattr(hashing.NumpyHasher, 'save', guarded)
