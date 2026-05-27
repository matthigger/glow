"""Shared utilities for hashing and identity.

``hash_array`` is a fast probabilistic SHA-256 over one or more
ndarrays. ``HashBySlots`` is a mixin that derives ``__hash__`` /
``__eq__`` from a class's ``__slots__`` and handles ndarray /
nested-mixin slot values via ``hash_array`` / recursion.
"""

import hashlib
import json

import numpy as np


HASH_SAMPLE_SIZE = 10_000


def hash_array(*arrs):
    """fast probabilistic SHA-256 hash over one or more ndarrays.

    Each array contributes its shape, dtype, and either its full byte
    buffer (when ``size <= HASH_SAMPLE_SIZE``) or a deterministic
    fixed-size sample. Mixing shape + dtype in catches differences a
    content sample would miss.

    Returns:
        16-character hex digest combining all input arrays.
    """
    h = hashlib.sha256()
    for arr in arrs:
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        flat = a.ravel()
        if flat.size > HASH_SAMPLE_SIZE:
            # reseed per-array so each array's sample positions are
            # a pure function of its own size (independent of order
            # / membership of arrs)
            idx = np.random.default_rng(0).integers(
                0, flat.size, HASH_SAMPLE_SIZE)
            sample = np.ascontiguousarray(flat[idx])
        else:
            sample = flat
        h.update(memoryview(sample).cast('B'))
    return h.hexdigest()[:16]


class HashBySlots:
    """Mixin: ``__hash__`` / ``__eq__`` derived from ``__slots__``.

    Subclasses declare ``__slots__`` as the canonical attribute list;
    every slot participates in identity. Non-JSON-friendly slot values
    are canonicalized inside ``_canon`` (ndarrays → ``hash_array``,
    nested ``HashBySlots`` instances → their own identity dict).

    Hash is NOT stable across mutating method calls — if a subclass
    writes to a slot after construction (e.g. an sklearn-style
    ``.fit()`` that populates result attrs), its hash will change.
    Callers that put mutated instances in dicts/sets are responsible
    for that contract.
    """

    __slots__ = ()

    def _identity_dict(self):
        """JSON-friendly identity dict used by __hash__ / __eq__."""
        return {'kind': type(self).__name__,
                **{f: self._canon(getattr(self, f))
                   for f in type(self).__slots__}}

    @staticmethod
    def _canon(v):
        if isinstance(v, np.ndarray):
            return hash_array(v)
        if isinstance(v, HashBySlots):
            return v._identity_dict()
        return v

    def __hash__(self):
        return hash(json.dumps(self._identity_dict(), sort_keys=True))

    def __eq__(self, other):
        return (type(self) is type(other)
                and self._identity_dict() == other._identity_dict())
