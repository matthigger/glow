"""Shared utilities for hashing and identity.

hash_array is a fast probabilistic SHA-256 over one or more ndarrays.
DataclassJSON is a mixin adding to_dict / to_json to a frozen dataclass;
the dataclass itself supplies native __hash__ / __eq__ (so every field
must be hashable -- scalars, tuples, nested dataclasses -- no ndarrays).
"""

import hashlib
import json
from dataclasses import fields

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


class DataclassJSON:
    """Mixin adding to_dict / to_json to a frozen dataclass.

    The dataclass itself supplies immutability and native __hash__ / __eq__
    (so every field must be hashable -- scalars, tuples, nested
    DataclassJSON specs -- and no field may hold an ndarray). This mixin
    adds only a JSON-friendly serialisation of the fields, used for stable
    cross-process identity (value_id / stable_hash) and provenance. Unlike
    the builtin hash, to_json folds in the class name under 'kind', so two
    classes with identical field tuples get distinct stable ids.

    Declares __slots__ = () so it does not reintroduce a __dict__ on
    dataclasses built with slots=True (a single non-slotted ancestor would
    silently defeat slots=True on every subclass).
    """

    __slots__ = ()

    def to_dict(self) -> dict:
        """Build the JSON-friendly identity dict, keyed by field name.

        The class name is under 'kind'; nested DataclassJSON fields recurse.
        """
        out = {'kind': type(self).__name__}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v.to_dict() if isinstance(v, DataclassJSON) else v
        return out

    def to_json(self) -> str:
        """Serialise the identity dict to a sorted-key JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_record(self) -> dict:
        """Return this spec's JSON-friendly record (its to_dict identity).

        The benchmark Recorder serialises any value exposing to_record (see
        glow._extra.benchmark.recorder._json_default); for a DataclassJSON spec that
        is just its identity dict, so the record nests the full stable recipe.
        """
        return self.to_dict()


def value_id(v):
    """Build a stable, JSON-friendly identity for one value.

    Simple scalars pass through; ndarrays and DataclassJSON instances get a
    stable hash; anything else falls back to repr. Unlike Python's builtin
    hash, the output is the same across processes.
    """
    if isinstance(v, np.ndarray):
        return hash_array(v)
    if isinstance(v, DataclassJSON):
        return hashlib.sha256(v.to_json().encode()).hexdigest()[:16]
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
