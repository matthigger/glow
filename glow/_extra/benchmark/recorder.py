"""Record each decorated call's inputs and outputs, keyed by joblib's args hash.

A ``Recorder`` is a decorator factory. Decorating a function makes every
*successful* call store one record

    {"hash": ..., "function": ..., "inputs": {...}, "outputs": {...},
     "time_sec": ...}

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
records -- recording an object's method needs no passthrough wrapper, e.g.
``recorder(output_name='ana')(AnalysisGLOW(exp=exp, n_perm_fwer=n).fit)()``
records the analysis, the fitted result and the timing.

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

``_json_default`` is the on-disk serialisation fallback: any value json can't
serialise natively is written as its ``repr`` string (a richer per-object form
may replace repr later). Raw numpy is kept compact -- a scalar becomes its
Python value and an ndarray its stable content hash -- so a stray heavy array
does not dump wholesale. Records held in memory keep live references (the
inputs/outputs themselves), serialised only when written to disk.
"""

import functools
import inspect
import json
import os
import tempfile
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from joblib.func_inspect import filter_args

from glow.util import hash_array


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
        return hash_array(obj)
    return repr(obj)


class Recorder:
    """Decorator factory recording each decorated call, keyed by its args hash.

    Attributes:
        records (dict): hash -> recorded call dict, one per executed decorated
            call. Each carries ``{hash, function, inputs, outputs, time_sec}``
            and, when a ``label`` was given, that too. A repeat hash overwrites
            (and warns); see the module docstring.
        folder (Path | None): the directory record files are mirrored to (one
            ``<hash>.json`` per record), created on construction, or None for
            in-memory only. The Recorder is the only object that reads/writes
            these files.
    """

    def __init__(self, folder=None):
        self.records = {}
        self.folder = Path(folder) if folder is not None else None
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)

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

                # key by joblib's own args hash, so the record matches the
                # cache entry the same call writes (see module docstring)
                key = self._args_hash(fnc, args, kwargs)
                self._store(key, {
                    "hash": key,
                    "function": fnc.__qualname__,
                    **({"label": label} if label is not None else {}),
                    "inputs": inputs,
                    "outputs": outputs,
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
