"""Record each decorated call's inputs and outputs, keyed by joblib's args hash.

A ``Recorder`` is a decorator factory. Decorating a function makes every
*successful* call store one record

    {"hash": ..., "function": ..., "inputs": {...}, "outputs": {...},
     "input_hashes": {...}, "output_hashes": {...}, "time_sec": ...}

keyed by ``joblib.hash(filter_args(fnc, [], args, kwargs))`` -- byte-for-byte
the key ``joblib.Memory`` files the same call's result under (its
``MemorizedFunc._get_args_id``: ``hash(filter_args(self.func, self.ignore,
args, kwargs))`` with the default ``ignore=[]`` / ``mmap_mode=None``). So a
record lines up one-to-one with the cached artifact: put the recorder *inside*
``@MEMORY.cache`` (the inner decorator) and a cache miss records exactly the
calls that actually ran, each under the hash joblib filed its result by. An
optional top-level ``"label"`` names the method/variant the call belongs to
(e.g. 'GLOW-GLM', 'VBA-TFCE'); it is metadata only, never part of the key.

``time_sec`` is the wall-clock duration of the wrapped call.

A recorded call may be a bound method: its receiver is captured as the
``self`` input (serialized via its ``repr``). ``filter_args`` includes the
receiver in the hashed arguments, so distinct receivers key to distinct
records -- recording an object's method needs no passthrough wrapper. The
experiment is passed to fit rather than stored on the analysis, so it is
captured as fit's own input rather than nested in the analysis recipe, e.g.
``recorder(output_name='ana')(AnalysisGLOW(n_perm_fwer=n).fit)(exp)``
records the analysis recipe, its ``exp`` fit input, the fitted result and the
timing.

Records are *keyed*, not appended. The same call recorded twice (same fnc and
the same filtered args -> same hash) overwrites the first and warns, mirroring
joblib.Memory, where a recompute replaces the one cache entry. In the
inside-the-cache layout a joblib hit never reaches the recorder, so a repeat
hash only arises if the cache was cleared and the call recomputed. NB the key
is the args hash *alone* (as joblib's is): joblib namespaces it by a per-
function directory, but this flat map does not, so two *different* functions
with identical filtered args would collide -- the overwrite-warning is the
guard for that.

A call that *raises* records nothing and the exception propagates -- again
like joblib.Memory, which stores nothing on a failed call. The caller sees the
real error; there is no failure record and no swallowing. (The output-shape
checks below likewise raise without recording: they are misconfiguration, not
a recorded result.)

Persistence: when a ``folder`` is given, each record is mirrored to
``folder/<hash>.json`` as it is made, written atomically (temp file +
``os.replace``). The hash partitions the filename namespace, so parallel
writers -- local joblib pools or AWS Batch jobs -- never contend on a file, and
merging is just listing the folder; this mirrors joblib.Memory's own per-hash
on-disk layout. ``load()`` reads the files back into ``records``. With
``folder=None`` the recorder is in-memory only.

``_json_default`` is the serialisation fallback: any value json can't serialise
natively becomes its ``repr`` string (a richer per-object form may replace repr
later). Raw numpy is kept compact -- a scalar becomes its Python value and an
ndarray its stable content hash -- so a stray heavy array does not dump
wholesale. Every record is snapshotted to this form *as it is made* (``_cell``),
so it holds no live reference to the (often heavy) inputs/outputs: the Experiment
etc. is free to be collected once the call returns, and the record stays light to
hold, pickle, and persist. The DAG hashes (below) are taken from the live values
first, before the snapshot -- so a record reads identically whether built here or
reloaded from disk, and joblib.Memory remains the store for the live objects
themselves (a cache hit hands them back).

Provenance DAG: each record also stores ``input_hashes`` / ``output_hashes``
-- ``joblib.hash`` of every named input and output whose type is in the
Recorder's ``link_types``. A call B *depends on* a call A when one of B's input
hashes equals one of A's output hashes (B consumed the value A produced).
Restricting the hashing to a few meaningful domain classes (e.g.
``(Experiment,)``) is deliberate: it keeps trivial values -- a scalar ``2``, a
shared all-True mask -- from forging spurious edges between unrelated calls.
Because the link is by content hash, not object ``id()``, it survives the
per-hash on-disk layout above: producer and consumer may be different processes
or runs, yet their shared value hashes the same. ``flatten_to_df`` reads
``records`` as that directed acyclic graph and returns one row per *leaf* (a
record whose outputs feed no other record) with every ancestor's fields
appended -- e.g. a score leaf carries the fit it scored and the setup that built
the data, so one row holds a whole trial.
"""

import functools
import inspect
import json
import os
import tempfile
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
from joblib.func_inspect import filter_args


# Sentinel `label` values for trials that produced no scored result, used by
# the trial fns / plotting (not by the recorder itself). SKIP: an infeasible
# cell, e.g. an HCP feature count beyond the pool. ERROR: a failed trial.
ERROR_LABEL = 'ERROR'
SKIP_LABEL = 'SKIP'
NON_RESULT_LABELS = (ERROR_LABEL, SKIP_LABEL)


def _json_default(obj):
    """json.dumps fallback: repr(obj), a placeholder serialisation.

    Stop-gap: any value json can't serialise natively records as its repr
    string rather than a structured recipe. Raw numpy is kept compact: a
    scalar becomes its Python value and an ndarray its stable content hash, so
    a stray array (e.g. in Experiment.meta) does not dump wholesale.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return joblib.hash(obj)
    return repr(obj)


def _cell(value):
    """Coerce a recorded value to its serialised snapshot form.

    The single coercion used everywhere a recorded value is kept: when a record
    is made (snapshotting its inputs/outputs, so no live object is retained), on
    disk, and when flatten_to_df lays a value into a DataFrame cell. Routing
    through the same json serialisation (``_json_default``) means all three agree
    -- a scalar stays itself, an ndarray / opaque object collapses to its
    content-hash / repr string, a small list/dict is kept structurally (its
    arrays hashed; note json renders tuples as lists). Object *identity* for the
    DAG edges lives in the records' hash maps, not in these cells.
    """
    return json.loads(json.dumps(value, default=_json_default))


def _flatten_record(record, prefix, sep='.') -> dict:
    """One record as a flat ``{column: cell}`` dict, every column under ``prefix``.

    ``prefix`` is the record's role in the row (its short function name, e.g.
    ``fit``); every column it contributes -- the ``hash`` / ``function`` /
    ``label`` / ``time_sec`` metadata and one ``in.<name>`` / ``out.<name>`` per
    named input / output -- is namespaced by it (``fit.time_sec``,
    ``fit.in.exp``), so a leaf and its ancestors never collide in one row. Each
    value is coerced to a cell by ``_cell``; ``label`` is None when the call
    carried none.
    """
    flat = {
        f'{prefix}{sep}hash': record['hash'],
        f'{prefix}{sep}function': record['function'],
        f'{prefix}{sep}label': record.get('label'),
        f'{prefix}{sep}time_sec': record.get('time_sec'),
    }
    for name, value in record.get('inputs', {}).items():
        flat[f'{prefix}{sep}in.{name}'] = _cell(value)
    for name, value in record.get('outputs', {}).items():
        flat[f'{prefix}{sep}out.{name}'] = _cell(value)
    return flat


def _ancestor_depths(start, parents_of) -> dict:
    """Breadth-first map of every transitive ancestor of ``start`` to its depth.

    ``parents_of(key)`` gives a record's immediate producers; this walks them
    transitively, recording each ancestor's shortest hop count from ``start``
    (its first visit, breadth-first). The visited set (``depth`` keys) makes it
    cycle-safe should a content-hash collision forge a back edge. ``start``
    itself is excluded from the result.

    Args:
        start: the leaf record key to walk up from.
        parents_of: maps a record key to the set of its immediate-producer keys.

    Returns:
        ancestor record key -> depth (hops from ``start``), ``start`` omitted.
    """
    depth = {start: 0}
    queue = deque([start])
    while queue:
        key = queue.popleft()
        for parent in parents_of(key):
            if parent not in depth:
                depth[parent] = depth[key] + 1
                queue.append(parent)
    del depth[start]
    return depth


class Recorder:
    """Decorator factory recording each decorated call, keyed by its args hash.

    Attributes:
        records (dict): hash -> recorded call dict, one per executed decorated
            call. Each carries ``{hash, function, inputs, outputs, input_hashes,
            output_hashes, time_sec}`` and, when a ``label`` was given, that too.
            ``inputs`` / ``outputs`` are snapshots (the serialised ``_cell``
            form), not live objects, so the record retains nothing heavy.
            ``input_hashes`` / ``output_hashes`` hold ``joblib.hash`` of only
            the inputs / outputs whose type is in ``link_types`` -- the
            provenance DAG edges flatten_to_df reads. A repeat hash overwrites
            (and warns); see the module docstring.
        folder (Path | None): the directory record files are mirrored to (one
            ``<hash>.json`` per record), created on construction, or None for
            in-memory only. The Recorder is the only object that reads/writes
            these files.
        link_types (tuple[type]): the classes whose values are content-hashed to
            form DAG edges (e.g. ``(Experiment,)``). Restricting to a few
            meaningful domain types stops trivial values (a scalar ``2``, a
            shared mask) from forging spurious edges. Empty (the default) links
            nothing, so flatten_to_df returns one ancestor-less row per record.
            Registered types must be joblib-hashable, else recording raises --
            that is a misconfiguration, deliberately not caught.
    """

    def __init__(self, folder=None, link_types=()):
        self.records = {}
        self.folder = Path(folder) if folder is not None else None
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
        self.link_types = tuple(link_types)
        if not all(isinstance(t, type) for t in self.link_types):
            raise TypeError("link_types must be classes (a value is linked when "
                            "isinstance(value, link_types))")

    @staticmethod
    def _args_hash(fnc, args, kwargs) -> str:
        """The args hash joblib keys a cached call by (its _get_args_id).

        joblib's filter_args (bind the call, apply defaults, drop ignored)
        then joblib.hash -- so a record keys by the same id joblib caches the
        call under without reaching into the MemorizedFunc. ignore is []
        (MEMORY.cache uses no ignore list) and the default coerce_mmap=False
        matches MEMORY's mmap_mode=None. For a bound method filter_args
        includes the receiver, so distinct receivers hash distinctly.

        Args:
            fnc: the function being hashed (its signature drives filter_args).
            args (tuple): positional call arguments.
            kwargs (dict): keyword call arguments.

        Returns:
            the joblib args hash (hex digest), the on-disk cache-entry key.
        """
        return joblib.hash(filter_args(fnc, [], args, kwargs))

    def __call__(self, output_name=None, output_name_list=None, label=None):
        """Build a decorator that records calls under one or more output names.

        Pass exactly one of ``output_name`` (the whole return is recorded
        under that single name) or ``output_name_list`` (the return is a
        tuple/list, unpacked positionally onto those names).

        ``label`` is optional per-call metadata: the method/variant name this
        call belongs to (e.g. 'GLOW-GLM', 'VBA-TFCE'), recorded as a top-level
        field. It is not part of the key -- variant identity rides in the
        receiver state filter_args already hashes.

        Args:
            output_name (str | None): single name for the whole return.
            output_name_list (tuple | list | None): names for an unpacked
                tuple/list return; non-empty, no duplicates.
            label (str | None): method/variant name for this call, recorded
                top-level; None to record no label.

        Returns:
            a decorator that wraps a function for recording.

        Raises:
            ValueError: neither or both of the two output args given, or
                ``output_name_list`` has duplicate names.
            TypeError: a name is not a str, ``output_name_list`` is empty, or
                ``label`` is not a str.
        """
        # require exactly one of output_name / output_name_list (be explicit)
        if (output_name is None) == (output_name_list is None):
            raise ValueError("provide exactly one of output_name or output_name_list")

        if label is not None and not isinstance(label, str):
            raise TypeError("label must be a str")

        if output_name is not None:
            if not isinstance(output_name, str):
                raise TypeError("output_name must be a str")
        else:
            if not isinstance(output_name_list, (tuple, list)) or len(output_name_list) == 0:
                raise TypeError("output_name_list must be a non-empty tuple/list")
            if not all(isinstance(n, str) for n in output_name_list):
                raise TypeError("output_name_list entries must be str")
            # the only possible name collision: a repeated output name. assert once, here.
            if len(set(output_name_list)) != len(output_name_list):
                raise ValueError("output_name_list has duplicate names")

        def decorator(fnc):
            sig = inspect.signature(fnc)

            # a bound method's signature already excludes self, so record it
            # separately from its __self__ -- the call captures the receiver
            # (serialised via its repr) as the 'self' input without the caller
            # threading it through a passthrough function.
            is_method = inspect.ismethod(fnc)

            # *args holds positional values with no names, so we can't record
            # them as named inputs -> reject it at decoration time. (**kwargs is
            # fine: its values are named, and we splice them up a level below.)
            var_kw = None
            for name, param in sig.parameters.items():
                if param.kind is inspect.Parameter.VAR_POSITIONAL:
                    raise TypeError(
                        f"{fnc.__name__} uses *{name}; recorder requires named "
                        f"parameters (no *args)"
                    )
                if param.kind is inspect.Parameter.VAR_KEYWORD:
                    var_kw = name

            @functools.wraps(fnc)
            def wrapped(*args, **kwargs):
                # capture inputs (incl. defaults). NB: shallow copy of
                # references, serialised only when written to disk (see module
                # docstring).
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                inputs = dict(bound.arguments)

                # record the receiver of a bound method as the 'self' input
                # (first, for readability), captured from __self__ since the
                # bound signature omits it
                if is_method:
                    inputs = {'self': fnc.__self__, **inputs}

                # splice **kwargs up a level: {'kwargs': {'x': 1}} -> {'x': 1}.
                # sig.bind already routes any keyword matching a named param to
                # that param (and rejects duplicates), so extra can't collide.
                if var_kw is not None:
                    inputs.update(inputs.pop(var_kw))

                # time only the wrapped call; an exception propagates (nothing
                # is recorded), like joblib.Memory on a failed call.
                t0 = time.perf_counter()
                out = fnc(*args, **kwargs)
                time_sec = time.perf_counter() - t0

                # name the outputs. The shape checks raise without recording:
                # misconfiguration, not a recorded result.
                if output_name is not None:
                    outputs = {output_name: out}
                else:
                    if not isinstance(out, (tuple, list)):
                        raise TypeError(
                            f"{fnc.__name__} declared output_name_list but returned "
                            f"{type(out).__name__}; expected a tuple/list"
                        )
                    if len(out) != len(output_name_list):
                        raise ValueError(
                            f"{fnc.__name__} returned {len(out)} values but "
                            f"output_name_list has {len(output_name_list)} names"
                        )
                    outputs = dict(zip(output_name_list, out))

                # content hashes of the link_types inputs / outputs, taken from
                # the *live* values (before the snapshot below): an input that
                # *is* another call's output shares that output's joblib.hash, so
                # flatten_to_df() links the two into a provenance DAG. Hashing
                # only the registered domain types keeps trivial values from
                # forging spurious edges (see module docs).
                input_hashes = {n: joblib.hash(v) for n, v in inputs.items()
                                if isinstance(v, self.link_types)}
                output_hashes = {n: joblib.hash(v) for n, v in outputs.items()
                                 if isinstance(v, self.link_types)}

                # snapshot inputs / outputs to their serialised form *now* (the
                # same _cell coercion the disk files and flatten_to_df use), so
                # the record keeps no live reference to the (often heavy) call
                # values -- the Experiment etc. is free to be collected once the
                # call returns, and the record stays light to hold, pickle, and
                # persist. The DAG hashes above were already taken from the live
                # values, so the snapshot loses nothing the graph needs.
                inputs = {n: _cell(v) for n, v in inputs.items()}
                outputs = {n: _cell(v) for n, v in outputs.items()}

                # key by joblib's own args hash, so the record matches the
                # cache entry the same call writes (see module docstring)
                key = self._args_hash(fnc, args, kwargs)
                self._store(key, {
                    "hash": key,
                    "function": fnc.__qualname__,
                    **({"label": label} if label is not None else {}),
                    "inputs": inputs,
                    "outputs": outputs,
                    "input_hashes": input_hashes,
                    "output_hashes": output_hashes,
                    "time_sec": time_sec,
                })
                return out

            return wrapped

        return decorator

    def _store(self, key, record) -> None:
        """Add a record under its hash key, overwriting (and warning) on repeat.

        A repeat key is the collision the module docstring describes (the same
        call recomputed, or two functions whose filtered args coincide); we
        keep the latest and warn so it is never silent. When a folder is set
        the record is also mirrored to ``<hash>.json``.
        """
        if key in self.records:
            warnings.warn(
                f"recorder overwriting record for hash {key} "
                f"({record['function']}); a prior call shared this args hash")
        self.records[key] = record
        if self.folder is not None:
            self._write(key, record)

    def _write(self, key, record) -> None:
        """Atomically mirror one record to ``folder/<hash>.json``.

        Written via a temp file in the same dir + ``os.replace`` so a reader
        (or a parallel worker) never sees a half-written file; the hash names
        the final file, so distinct records never contend.
        """
        text = json.dumps(record, indent=2, default=_json_default)
        fd, tmp = tempfile.mkstemp(dir=self.folder, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(text)
            os.replace(tmp, self.folder / f'{key}.json')
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def load(self) -> dict:
        """Read every ``<hash>.json`` under folder into ``records``; return it.

        Merges the on-disk records into the in-memory map (keyed by hash), so a
        reader process reconstitutes what parallel writers produced. A no-op
        returning the current records when no folder is set.
        """
        if self.folder is None:
            return self.records
        for p in self.folder.glob('*.json'):
            rec = json.loads(p.read_text())
            self.records[rec['hash']] = rec
        return self.records

    def flatten_to_df(self, prefix_sep='.'):
        """Flatten the records into a leaf-per-row provenance DataFrame.

        Reads ``records`` as a directed acyclic graph: a call B depends on a
        call A when one of B's ``input_hashes`` equals one of A's
        ``output_hashes`` (B consumed the value A produced -- the same
        ``joblib.hash`` content identity, so the edge holds across processes and
        runs; see the module docstring). Each *leaf* -- a record whose outputs
        feed no other record -- becomes one row, carrying its own fields plus
        those of every ancestor (transitive producer) appended as prefixed
        columns. A typical benchmark leaf is a score step; its ancestors are the
        fit it scored and the setup that built the data, so one row holds the
        whole trial: the swept axes (setup inputs), the recipe and timing (fit),
        and the result (the score outputs).

        Every record in a row -- the leaf and each ancestor -- contributes its
        ``hash`` / ``function`` / ``label`` / ``time_sec`` and one
        ``in.<name>`` / ``out.<name>`` column per named input / output (values
        coerced by ``_cell``), all namespaced by the record's short function
        name (``fit.time_sec``, ``setup.in.seed``). A name repeated within one
        row is suffixed ``#2`` / ``#3``; the leaf is laid down first (claiming
        the un-suffixed name) and ancestors follow shallowest-first, so the
        prefixes are deterministic. A record whose link_types produced no hashes
        contributes no edges -- it is then its own ancestor-less leaf.

        Operates on the in-memory ``records``; call ``load()`` first to fold in
        a folder's on-disk records from other writers.

        Args:
            prefix_sep (str): separator between an ancestor's prefix and its
                column name (e.g. the ``.`` in ``fit.time_sec``).

        Returns:
            a pandas DataFrame with one row per leaf record; columns are the
            union across leaves, NaN where a leaf lacks an ancestor's column.
        """
        import pandas as pd

        records = self.records

        # output content hash -> the record that produced it. A collision (two
        # calls emitting an equal value, see module docstring) keeps the last
        # writer; in practice the heavy domain objects are unique per trial. A
        # None hash (an unhashable value, see _safe_hash) forms no edge.
        producer_of = {}
        for key, rec in records.items():
            for h in rec.get('output_hashes', {}).values():
                if h is not None:
                    producer_of[h] = key

        def parents_of(key):
            """The records that produced this record's inputs (no self-edge)."""
            ps = set()
            for h in records[key].get('input_hashes', {}).values():
                producer = producer_of.get(h)
                if producer is not None and producer != key:
                    ps.add(producer)
            return ps

        # input content hash -> the records consuming it. A leaf is a record
        # none of whose (hashable) outputs is consumed by *another* record; the
        # self exclusion keeps an identity call (input value == output value) a
        # leaf, and a None output hash never disqualifies one.
        consumers_of = defaultdict(set)
        for key, rec in records.items():
            for h in rec.get('input_hashes', {}).values():
                if h is not None:
                    consumers_of[h].add(key)
        leaves = [key for key, rec in records.items()
                  if all(not (consumers_of[h] - {key})
                         for h in rec.get('output_hashes', {}).values()
                         if h is not None)]

        rows = []
        for leaf in leaves:
            depth = _ancestor_depths(leaf, parents_of)
            # the leaf first (it claims the un-suffixed prefix), then ancestors
            # shallowest-first (then function, hash) so prefixes are
            # deterministic; a short name repeated within the row gets #2 / #3
            ordered = [leaf] + sorted(
                depth, key=lambda k: (depth[k], records[k]['function'], k))
            row = {}
            seen = {}
            for key in ordered:
                role = records[key]['function'].rsplit('.', 1)[-1]
                seen[role] = seen.get(role, 0) + 1
                prefix = role if seen[role] == 1 else f'{role}#{seen[role]}'
                row.update(_flatten_record(records[key], prefix, sep=prefix_sep))
            rows.append(row)

        return pd.DataFrame(rows)
