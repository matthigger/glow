"""Record each decorated call's inputs and outputs, grouped by trial.

A ``Recorder`` is a decorator factory. Decorating a function makes every
call append one record

    {"trial_id": ..., "function": ..., "inputs": {...}, "outputs": {...},
     "time_sec": ...}

to ``records``. Calls nested inside one top-level invocation (or inside an
explicit ``with recorder.trial(trial_id=...):`` block) share a single
``trial_id``, so all the work of one benchmark trial groups together. In
the benchmark the ``trial_id`` is the trial's cache hash
(``TrialCache.hash``), so records from independent workers -- local joblib
pools or AWS Batch jobs -- merge without collision and line up with the
results.csv row the same trial would write.

``time_sec`` is the wall-clock duration of the wrapped call, so the
benchmark no longer hand-times each method.

A recorded call may be a bound method: when the wrapped callable is one, its
receiver is captured as the ``self`` input (serialized via its ``to_record``
recipe), so recording an object's method needs no passthrough wrapper -- apply
the decorator to the bound method inline, e.g.
``recorder(output_name='ana')(AnalysisGLOW(exp=exp, n_perm_fwer=n).fit)()``
records the analysis recipe, the fitted result, the timing and any failure.

A call that *raises* records a failure record instead -- same shape but
with ``error`` (the traceback) in place of ``outputs`` -- and the exception
is then *swallowed*: the call returns None and the trial is marked failed,
so the caller never has to wrap each step in try/except. Every later
recorded call in that same trial short-circuits -- it neither runs nor
records, returning None -- so a trial fn written as a straight-line
sequence of recorded steps just fast-forwards to its end and the driver
moves on to the next trial. A failure therefore leaves exactly the records
up to and including the call that raised, and nothing after, all under the
one trial id. The SKIP-vs-ERROR distinction is a domain decision left to
the benchmark, not encoded here.

Records are append-only: a function called twice in a trial yields two
records, never an overwrite. A call short-circuited after a same-trial
failure adds none.

``to_json`` serializes via ``_json_default``, which consults one protocol --
``to_record()``: any input/output exposing it (a DataclassJSON spec such as a
DataSource / Extenter, an Experiment, an Analysis) records as its
JSON-friendly identity/recipe dict rather than an opaque ``repr`` string.
Each glow class chooses the subset it records -- Experiment its shapes / dtype
/ content hash / meta (never the large ``y``), Analysis its ``__init__`` config
knobs (never ``exp``, the fitted arrays, or ``pval``) -- so a record carries
the stable recipe without dumping any heavy array. Raw numpy still falls back
safely (scalar -> value, ndarray -> content hash); everything else -> ``repr``.

One known limitation, deferred to the object-representation work (see
docs/notes/recorder_object_repr.md): inputs/outputs hold *references*,
serialized only at ``to_json`` time, so an in-place mutation by the decorated
function (e.g. an ``Analysis.fit`` populating its fitted outputs) is reflected
post-call rather than snapshotted at call time. The recorded subsets above are
all set at ``__init__`` and never mutated, so this does not affect them.
"""

import contextlib
import contextvars
import functools
import inspect
import json
import time
import traceback
import uuid

import numpy as np

from glow.util import hash_array


def _json_default(obj):
    """json.dumps fallback: a value's own to_record() dict, else a safe form.

    The single serialisation protocol is to_record(): any value exposing it
    -- a DataclassJSON spec (DataSource / Extenter / ...), an Experiment, an
    Analysis -- records as its JSON-friendly identity/recipe dict rather than
    an opaque repr string. Raw numpy is never dumped wholesale: a scalar
    becomes its Python value and an ndarray its stable content hash (so a
    stray array in, e.g., Experiment.meta records compactly). A StrEnum like
    ClusterMode is already JSON-native (its value string), so json handles it
    directly. Anything else falls back to repr.
    """
    to_record = getattr(obj, "to_record", None)
    if callable(to_record):
        return to_record()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return hash_array(obj)
    return repr(obj)


class Recorder:
    """Decorator factory recording each decorated call's inputs and outputs.

    Attributes:
        records (list): append-only list of recorded call dicts, one per
            executed decorated call. A success carries ``{trial_id, function,
            inputs, outputs, time_sec}``; a failure swaps ``outputs`` for
            ``error`` (the traceback). Calls short-circuited after a same-trial
            failure add none.
        _trial_id_current (contextvars.ContextVar): the active trial id for
            the current execution context (thread / asyncio task); None when
            no trial is open. A ContextVar so concurrent runs never see each
            other's id.
        _failed_trials (set): trial ids that have recorded a failure and are
            still open. Membership makes every later recorded call in the trial
            short-circuit; the id is dropped when its trial scope closes.
    """

    def __init__(self):
        self.records = []
        self._trial_id_current = contextvars.ContextVar(
            "recorder_trial_id", default=None)
        self._failed_trials = set()

    def get_trial_id(self):
        """Mint a fresh trial id.

        uuid4: globally unique with no coordination, so JSON outputs from
        independent workers (e.g. on AWS) merge without collision. The
        benchmark instead supplies the cache hash explicitly via
        ``trial(trial_id=...)``; this default covers standalone use.
        """
        return str(uuid.uuid4())

    @contextlib.contextmanager
    def trial(self, trial_id=None):
        """Scope a trial id; all decorated calls in the block share it.

        A fresh id is minted when ``trial_id`` is None. Uses the ContextVar
        so two concurrent ``run`` blocks (threads or async tasks) never see
        each other's id; the reset token restores any outer value on exit,
        so nesting is safe.

        Args:
            trial_id: the id to scope, or None to mint one.

        Yields:
            the scoped trial id.
        """
        if trial_id is None:
            trial_id = self.get_trial_id()
        reset = self._trial_id_current.set(trial_id)
        try:
            yield trial_id
        finally:
            self._trial_id_current.reset(reset)
            # drop any failure flag raised during this trial: the id can't
            # carry a stale "failed" state into a reuse, and the set stays
            # bounded by the number of trials currently open (not ever-failed).
            self._failed_trials.discard(trial_id)

    def __call__(self, output_name=None, output_name_list=None):
        """Build a decorator that records calls under one or more output names.

        Pass exactly one of ``output_name`` (the whole return is recorded
        under that single name) or ``output_name_list`` (the return is a
        tuple/list, unpacked positionally onto those names).

        Args:
            output_name (str | None): single name for the whole return.
            output_name_list (tuple | list | None): names for an unpacked
                tuple/list return; non-empty, no duplicates.

        Returns:
            a decorator that wraps a function for recording.

        Raises:
            ValueError: neither or both of the two args given, or
                ``output_name_list`` has duplicate names.
            TypeError: a name is not a str, or ``output_name_list`` is empty.
        """
        # require exactly one of output_name / output_name_list (be explicit)
        if (output_name is None) == (output_name_list is None):
            raise ValueError("provide exactly one of output_name or output_name_list")

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
            # (its to_record() recipe) as the 'self' input without the caller
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
                # once a trial has failed, every later step in it is a no-op:
                # it neither runs nor records and returns None, so the trial fn
                # keeps its straight-line shape and just fast-forwards to its
                # end (the driver then moves on to the next trial).
                active = self._trial_id_current.get()
                if active is not None and active in self._failed_trials:
                    return None

                # capture inputs (incl. defaults). NB: shallow copy of
                # references, not a snapshot -- in-place mutation by fnc is
                # reflected later (see module docstring).
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

                # reuse an open trial id (nested call), else open a fresh one
                # for this call via trial() (set/reset, even if fnc raises)
                cm = contextlib.nullcontext(active) if active is not None else self.trial()
                with cm as trial_id:
                    # time only the wrapped call. On failure record it (so the
                    # failed trial is auditable, not vanished), mark the trial
                    # failed, and *swallow* -- return None instead of re-raising
                    # -- so the caller need not try/except each step and later
                    # steps short-circuit above. Only fnc's own exceptions are
                    # caught; the output-validation errors below are
                    # misconfiguration, not a trial failure, and stay unrecorded.
                    t0 = time.perf_counter()
                    try:
                        out = fnc(*args, **kwargs)
                    except Exception:
                        self.records.append({
                            "trial_id": trial_id,
                            "function": fnc.__qualname__,
                            "inputs": inputs,
                            "error": traceback.format_exc(),
                            "time_sec": time.perf_counter() - t0,
                        })
                        self._failed_trials.add(trial_id)
                        return None
                    time_sec = time.perf_counter() - t0

                    # a nested wrapped call may have failed (and swallowed)
                    # while fnc ran, marking the trial failed: suppress this
                    # success so a trial's records stop at its first failure.
                    if trial_id in self._failed_trials:
                        return out

                    # only successful calls in a still-clean trial reach here;
                    # record their named outputs
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

                    self.records.append({
                        "trial_id": trial_id,
                        "function": fnc.__qualname__,
                        "inputs": inputs,
                        "outputs": outputs,
                        "time_sec": time_sec,
                    })

                    return out

            return wrapped

        return decorator

    def to_json(self, file=None, indent=2):
        """Serialize records to JSON. Returns the string if no file is given.

        Inputs/outputs exposing to_record() (DataclassJSON specs, Experiment,
        Analysis) nest as their record dict; raw numpy serialises safely and
        any other non-serializable value falls back to repr() (see module
        docstring and _json_default).

        Args:
            file (str | None): path to write; None returns the JSON string.
            indent (int): json.dump indent.

        Returns:
            the JSON string when ``file`` is None, else None.
        """
        if file is None:
            return json.dumps(self.records, indent=indent, default=_json_default)
        with open(file, "w") as f:
            json.dump(self.records, f, indent=indent, default=_json_default)
