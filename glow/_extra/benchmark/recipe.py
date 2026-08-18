"""Declarative identity for benchmark artifacts, never a content hash.

An artifact's identity is the recipe that built it -- an operation name, its
declarative kwargs, and its parents' ids -- not the bytes it happens to hold.
recipe_id hashes that recipe's canonical JSON, so one cell resolves to the same
id on any machine: no pickle, no array bytes, no float payload. That is what
makes a cache key and a provenance edge portable across a heterogeneous fleet,
where the same computation lands last-bit-different results.

    uid_data = recipe_id('data_factory_wgn', kwargs_data)
    uid_eff = recipe_id('effect_factory_single', kwargs_effect,
                        parents=[uid_data])
    uid_leaf = recipe_id('run_stat', kwargs_fnc, parents=[uid_eff])

Canonical form (canon): a dict renders key-sorted, a tuple / set as a list, a
numpy scalar as its Python value, a class as module.qualname, an object with a
canon_spec() as that, anything else as its repr -- the Extenter / Analysis
convention, where a parameter-only repr IS the declaration. A computed array can
never be an identity, so an ndarray above MAX_CANON_SIZE raises rather than
quietly hashing; a small declarative one (a contrast) renders as nested lists.

A declared key does not invalidate itself when the numerics change, so
IMPL_VERSION is how a numerics change that leaves the recipe untouched still
invalidates everything built from it: bump the op's entry.
"""

import hashlib
import inspect
import json

import numpy as np

# Bump only for a change to canon / canon_json themselves (a re-render of every
# id); IMPL_VERSION is the per-op knob, this is the format's.
CANON_VERSION = 1

# Hex chars kept from the sha256 digest. 32 keeps a record file name the same
# shape as the md5 keys it replaces, at 128 bits of collision resistance.
UID_LEN = 32

# An ndarray this size or smaller is treated as a declaration (a contrast, a
# small index list) and renders structurally; anything larger is a computed
# payload and raises, which is the invariant this module exists to enforce.
MAX_CANON_SIZE = 1024

# Per-op implementation version, folded into every id the op produces. Bump an
# entry when the op's numerics or semantics change without its kwargs changing;
# an op absent here is version 0.
IMPL_VERSION = {
    'data_factory_wgn': 1,
    'data_factory_hcp': 1,
    'effect_factory_single': 1,
    'effect_factory_split': 1,
    'run_ana': 1,
    'run_stat': 1,
    'run_prune': 1,
    'run_segment': 1,
    'run_inner_perm': 1,
    'run_ana_time': 1,
    'run_ana_time_1perm': 1,
}


class ComputedArrayError(ValueError):
    """Raised when an array too large to be a declaration reaches canon."""


def canon(value):
    """Return value in canonical, JSON-serialisable, machine-stable form.

    The one rule: nothing here may depend on the bytes of a computed array (see
    the module docstring). Containers recurse; a set renders as its sorted
    canonical JSON strings, so element order cannot leak in.

    Args:
        value: any declarative value from a recipe's kwargs.

    Returns:
        a value composed only of dict / list / str / int / float / bool / None.

    Raises:
        ComputedArrayError: value holds an ndarray larger than MAX_CANON_SIZE.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return canon(value.item())
    if isinstance(value, np.ndarray):
        if value.size > MAX_CANON_SIZE:
            raise ComputedArrayError(
                f'refusing to canonicalise a {value.size}-element array: an '
                f'identity may not depend on computed array bytes (limit '
                f'{MAX_CANON_SIZE}; pass the recipe that built it instead)')
        return value.tolist()
    if isinstance(value, type):
        return f'{value.__module__}.{value.__qualname__}'
    if isinstance(value, dict):
        return {str(k): canon(v)
                for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [canon(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(json.dumps(canon(v), sort_keys=True) for v in value)
    spec = getattr(value, 'canon_spec', None)
    if callable(spec):
        return canon(spec())
    return repr(value)


def canon_json(op: str, kwargs=None, parents=(), impl_version=None) -> str:
    """Render one recipe as the canonical JSON string its id hashes.

    Exposed alongside recipe_id so a mismatch is debuggable: two ids differ iff
    these strings differ, and the diff names the field.

    Args:
        op (str): the operation's recorded name (a function __qualname__).
        kwargs (dict | None): the op's declarative kwargs; None is empty.
        parents (iterable[str]): parent uids, order significant.
        impl_version (int | None): override the op's IMPL_VERSION entry (for
            tests); None looks it up, defaulting to 0.

    Returns:
        the canonical JSON string (key-sorted, no insignificant whitespace).
    """
    if impl_version is None:
        impl_version = IMPL_VERSION.get(op, 0)
    payload = {
        'canon_version': CANON_VERSION,
        'op': op,
        'impl_version': impl_version,
        'kwargs': canon(dict(kwargs or {})),
        'parents': [str(p) for p in parents],
    }
    return json.dumps(payload, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True)


def recipe_id(op: str, kwargs=None, parents=(), impl_version=None) -> str:
    """Return the portable uid of one recipe.

    Args:
        op (str): the operation's recorded name.
        kwargs (dict | None): the op's declarative kwargs.
        parents (iterable[str]): parent uids, order significant.
        impl_version (int | None): override the op's IMPL_VERSION entry.

    Returns:
        uid (str): UID_LEN hex chars of sha256 over canon_json.
    """
    digest = hashlib.sha256(
        canon_json(op, kwargs, parents, impl_version).encode('utf-8'))
    return digest.hexdigest()[:UID_LEN]


class Recipe:
    """One artifact's declared identity: op + kwargs + parent uids.

    The value a record stores so a leaf is self-describing -- its cell is
    readable off the leaf alone, with no walk up a content-hash join (see
    glow._extra.benchmark.recorder).

    Attributes:
        op (str): the operation's recorded name.
        kwargs (dict): the op's declarative kwargs.
        parents (tuple[str]): parent uids, order significant.
    """

    __slots__ = ('op', 'kwargs', 'parents')

    def __init__(self, op: str, kwargs=None, parents=()):
        self.op = op
        self.kwargs = dict(kwargs or {})
        self.parents = tuple(str(p) for p in parents)

    @property
    def uid(self) -> str:
        """The recipe's portable id (recipe_id over its fields)."""
        return recipe_id(self.op, self.kwargs, self.parents)

    def child(self, op: str, kwargs=None) -> 'Recipe':
        """Return the recipe of an op consuming this one's output."""
        return Recipe(op, kwargs, parents=(self.uid,))

    def as_dict(self) -> dict:
        """Return the record form: {uid, op, kwargs, parents, impl_version}."""
        return {
            'uid': self.uid,
            'op': self.op,
            'kwargs': canon(self.kwargs),
            'parents': list(self.parents),
            'impl_version': IMPL_VERSION.get(self.op, 0),
        }

    def __eq__(self, other) -> bool:
        return isinstance(other, Recipe) and self.uid == other.uid

    def __hash__(self) -> int:
        return hash(self.uid)

    def __repr__(self) -> str:
        return (f'Recipe(op={self.op!r}, kwargs={self.kwargs!r}, '
                f'parents={self.parents!r})')


def raw_fnc(fnc):
    """Return the undecorated function under a memoised + recorded callable.

    Peels joblib.Memory's .func then the recorder wrapper, so the signature and
    qualname match what the recorder recorded under.
    """
    return inspect.unwrap(getattr(fnc, 'func', fnc))


def declared_ignore(fnc):
    """Return the ignore list fnc was decorated with (() if undecorated).

    The recorder stamps its list onto the wrapper, so a reader naming a call's
    uid filters exactly what the writer filtered without being told twice.
    """
    for obj in (fnc, getattr(fnc, 'func', None)):
        names = getattr(obj, '_recipe_ignore', None)
        if names is not None:
            return names
    return ()


def recipe_for_call(fnc, kwargs, parents=(), ignore=None) -> Recipe:
    """Return the recipe a recorded call to fnc(**kwargs) files under.

    The caller-side twin of what the recorder computes at record time: bind
    kwargs to the signature and apply defaults (so an omitted default keys the
    same either way), then drop the non-declarative names. A parameter with no
    default that kwargs omits -- the linked exp, its mask_target_list companion
    -- stays absent, which is why the two agree.

    Use it to name a cell's uid without building anything: a CONFIG cell's
    kwargs are enough to say what its artifacts will be called.

    Args:
        fnc (Callable): the memoised + recorded op (or the raw function).
        kwargs (dict): the call's declarative kwargs.
        parents (iterable[str]): parent uids, order significant.
        ignore (iterable[str] | None): parameter names to leave out of the
            recipe; None (default) reads the decorator's own list
            (declared_ignore), which is what keeps a reader in step with it.

    Returns:
        recipe (Recipe): the call's declared identity.
    """
    if ignore is None:
        ignore = declared_ignore(fnc)
    fnc = raw_fnc(fnc)
    bound = inspect.signature(fnc).bind_partial(**kwargs)
    bound.apply_defaults()
    drop = set(ignore) | {'parent_uid'}
    return Recipe(fnc.__qualname__,
                  {k: v for k, v in bound.arguments.items() if k not in drop},
                  parents)


def seed_from_uid(uid: str, bits: int = 32) -> int:
    """Derive a reproducible RNG seed from a uid.

    The declared replacement for seeding off joblib.hash(exp): same intent (a
    placement that varies per data realization but is fixed across the effect
    grid), now a function of the declaration rather than of the array bytes, so
    it cannot drift with a BLAS kernel.

    Args:
        uid (str): the parent artifact's uid.
        bits (int): width of the returned seed.

    Returns:
        a non-negative int below 2**bits.
    """
    digest = hashlib.sha256(f'seed:{uid}'.encode('utf-8')).hexdigest()
    return int(digest, 16) % (1 << bits)
