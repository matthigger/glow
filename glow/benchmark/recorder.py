"""Record each decorated call's inputs and outputs, grouped by trial.

A ``Recorder`` is a decorator factory. Decorating a function makes every
call append one record

    {"trial_id": ..., "function": ..., "inputs": {...}, "outputs": {...}}

to ``records``. Calls nested inside one top-level invocation (or inside an
explicit ``with recorder.run(trial_id=...):`` block) share a single
``trial_id``, so all the work of one benchmark trial groups together. In
the benchmark the ``trial_id`` is the trial's cache hash
(``TrialCache.hash``), so records from independent workers -- local joblib
pools or AWS Batch jobs -- merge without collision and line up with the
results.csv row the same trial would write.

Records are append-only: a function called twice in a trial yields two
records, never an overwrite.

Two known limitations, both deferred to the object-representation work
(see docs/notes/recorder_object_repr.md):

  - inputs/outputs hold *references*, serialized only at ``to_json`` time
    via the ``repr`` fallback, so an in-place mutation by the decorated
    function is reflected post-call rather than snapshotted at call time;
  - heavy glow objects (Experiment, Analysis) have no JSON-friendly,
    stable repr yet, so they currently serialize via ``repr``.
"""

import contextlib
import contextvars
import functools
import inspect
import json
import uuid


class Recorder:
    """Decorator factory recording each decorated call's inputs and outputs.

    Attributes:
        records (list): append-only list of recorded call dicts, one per
            decorated call, each ``{trial_id, function, inputs, outputs}``.
        _trial_id_current (contextvars.ContextVar): the active trial id for
            the current execution context (thread / asyncio task); None when
            no trial is open. A ContextVar so concurrent runs never see each
            other's id.
    """

    def __init__(self):
        self.records = []
        self._trial_id_current = contextvars.ContextVar(
            "recorder_trial_id", default=None)

    def get_trial_id(self):
        """Mint a fresh trial id.

        uuid4: globally unique with no coordination, so JSON outputs from
        independent workers (e.g. on AWS) merge without collision. The
        benchmark instead supplies the cache hash explicitly via
        ``run(trial_id=...)``; this default covers standalone use.
        """
        return str(uuid.uuid4())

    @contextlib.contextmanager
    def run(self, trial_id=None):
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
                # references, not a snapshot -- in-place mutation by fnc is
                # reflected later (see module docstring).
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                inputs = dict(bound.arguments)

                # splice **kwargs up a level: {'kwargs': {'x': 1}} -> {'x': 1}.
                # sig.bind already routes any keyword matching a named param to
                # that param (and rejects duplicates), so extra can't collide.
                if var_kw is not None:
                    inputs.update(inputs.pop(var_kw))

                # reuse an open trial id (nested call), else open a fresh one
                # for this call via run() (set/reset, even if fnc raises)
                active = self._trial_id_current.get()
                cm = contextlib.nullcontext(active) if active is not None else self.run()
                with cm as trial_id:
                    out = fnc(*args, **kwargs)

                # only successful calls are recorded (a failed call has no
                # outputs)
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
                })

                return out

            return wrapped

        return decorator

    def to_json(self, file=None, indent=2):
        """Serialize records to JSON. Returns the string if no file is given.

        Non-serializable values fall back to repr() (see module docstring).

        Args:
            file (str | None): path to write; None returns the JSON string.
            indent (int): json.dump indent.

        Returns:
            the JSON string when ``file`` is None, else None.
        """
        if file is None:
            return json.dumps(self.records, indent=indent, default=repr)
        with open(file, "w") as f:
            json.dump(self.records, f, indent=indent, default=repr)
