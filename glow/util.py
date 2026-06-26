"""Shared utilities for hashing and identity.

hash_array is a fast probabilistic SHA-256 over one or more ndarrays.
value_id / stable_hash build a JSON-friendly identity (stable across
processes, unlike the builtin hash) for scalars, ndarrays, and dicts of them.
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


def value_id(v):
    """Build a stable, JSON-friendly identity for one value.

    Simple scalars pass through; ndarrays get a stable content hash; anything
    else falls back to repr. Unlike Python's builtin hash, the output is the
    same across processes.
    """
    if isinstance(v, np.ndarray):
        return hash_array(v)
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
