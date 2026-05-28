"""Shared utilities for hashing and identity.

hash_array is a fast probabilistic SHA-256 over one or more ndarrays.
HashBySlots is a mixin that derives __hash__ / __eq__ from a class's
__slots__ and handles ndarray / nested-mixin slot values via hash_array
and recursion.
"""

import hashlib
import json

import numpy as np


HASH_SAMPLE_SIZE = 10_000


def hash_array(*arrs) -> str:
    """Compute a fast probabilistic SHA-256 hash over one or more ndarrays.

    Each array contributes its shape, dtype, and either its full byte
    buffer (when size <= HASH_SAMPLE_SIZE) or a deterministic fixed-size
    sample. Mixing shape and dtype in catches differences a content sample
    would miss.

    Args:
        arrs (np.array): one or more ndarrays of any shape/dtype to hash

    Returns:
        digest (str): 16-character hex digest combining all input arrays
    """
    h = hashlib.sha256()
    for arr in arrs:
        a = np.ascontiguousarray(arr)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        flat = a.ravel()
        if flat.size > HASH_SAMPLE_SIZE:
            # reseed per-array so each array's sample positions are a pure
            # function of its own size (independent of the order / membership
            # of arrs)
            idx = np.random.default_rng(0).integers(
                0, flat.size, HASH_SAMPLE_SIZE)
            sample = np.ascontiguousarray(flat[idx])
        else:
            sample = flat
        h.update(memoryview(sample).cast('B'))
    return h.hexdigest()[:16]


class HashBySlots:
    """Mixin deriving __hash__ / __eq__ from a class's __slots__.

    Every slot listed in __slots__ is part of identity — there is no
    private / non-identity slot convention. Non-identity state (memo
    caches, etc.) lives off the instance, typically as a class attribute.

    Subclasses MUST declare __slots__ (use () if no new slots are added);
    enforced at class-creation time by __init_subclass__.

    Non-JSON-friendly slot values are canonicalised in _canon (ndarrays
    become hash_array digests, nested HashBySlots instances become their
    own identity dict). Hash is NOT stable across mutating method calls —
    if a subclass writes to a slot after construction (e.g. an
    sklearn-style .fit() that populates result attrs), its hash will
    change. Callers that put mutated instances in dicts/sets are
    responsible for that contract.
    """

    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if '__slots__' not in cls.__dict__:
            raise TypeError(
                f'{cls.__name__}: HashBySlots subclasses must declare '
                '__slots__ (use () if no new slots are added)')

    def _identity_dict(self) -> dict:
        """Build the JSON-friendly identity dict used by __hash__ / __eq__.

        Walks the full MRO so subclass identity covers every slot
        declared along the inheritance chain.
        """
        out = {'kind': type(self).__name__}
        for cls in reversed(type(self).__mro__):
            for slot in getattr(cls, '__slots__', ()):
                out[slot] = self._canon(getattr(self, slot))
        return out

    @staticmethod
    def _canon(v):
        """Canonicalise one slot value into a JSON-friendly identity token."""
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


def value_id(v):
    """Build a stable, JSON-friendly identity for one value.

    Simple scalars pass through; ndarrays and HashBySlots instances get a
    stable hash; anything else falls back to repr. Unlike Python's builtin
    hash, the output is the same across processes.
    """
    if isinstance(v, np.ndarray):
        return hash_array(v)
    if isinstance(v, HashBySlots):
        return hashlib.sha256(
            json.dumps(v._identity_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return repr(v)


def stable_hash(d: dict) -> str:
    """Compute an 8-hex digest of a dict's contents, stable across processes.

    Unlike Python's builtin hash, the output does not depend on
    PYTHONHASHSEED, so it's safe to embed in cached filenames or CSV
    columns and compare across runs. 8 hex chars (~32 bits) is plenty for
    our trial-cache scale.
    """
    sig = json.dumps({k: value_id(d[k]) for k in sorted(d)},
                     sort_keys=True)
    return hashlib.sha256(sig.encode()).hexdigest()[-8:]
