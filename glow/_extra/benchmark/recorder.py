"""Record each decorated call's inputs and outputs, keyed by its args hash.

A Recorder decorates a function so every successful call stores one record:

    {hash, cache_key, function, inputs, outputs, input_hashes, output_hashes,
     time_sec, uid?, op?, kwargs?, parents?, impl_version?, recurse?}

The key is joblib.hash(filter_args(fnc, [], args, kwargs)) -- the key
joblib.Memory files the result under, also stored as cache_key so the cache
entry stays locatable however the record is named. Nest the recorder inside
@MEMORY.cache and a miss records just the calls that ran, each matching its
cached artifact. A repeat key overwrites and warns; a raising call records
nothing.

inputs/outputs are _cell snapshots, so a record holds no live Experiment.
With a folder, each mirrors to <hash>.json atomically (one file per hash, so
parallel workers never contend); load() reads them back.

Declared identity: uid / op / kwargs / parents / impl_version hold the call's
recipe (see glow._extra.benchmark.recipe) -- a portable id built from the
declaration rather than from any array's bytes, and lineage the consumer
declares rather than a reader rediscovering it. A call whose parent cannot be
named (it takes a link_types input but no parent_uid) records no recipe
fields, so a uid is never a guess. Pass ignore to keep a non-declarative
parameter (an array companion) out of the recipe, mirroring the ignore list
given to @MEMORY.cache.

Provenance DAG: an edge is declared -- B depends on A when B lists A's uid in
parents -- so it holds on any machine. flatten_to_df walks it to one row per
leaf carrying its ancestors' fields, so a trial's setup, fit and score land in
one row.

input_hashes/output_hashes are the legacy edge: joblib.hash of the inputs and
outputs whose type is in link_types (e.g. (Experiment,)), matched by equality.
They are still written, and read for any record lacking a recipe, so records
written before the recipe fields stay linkable; being a hash of computed arrays
they are not portable across machines, and they retire with those records.
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

from .recipe import Recipe


# Sentinel label values for trials that produced no scored result, used by
# the trial fns / plotting (not by the recorder itself). SKIP: an infeasible
# cell, e.g. an HCP feature count beyond the pool. ERROR: a failed trial.
ERROR_LABEL = 'ERROR'
SKIP_LABEL = 'SKIP'
NON_RESULT_LABELS = (ERROR_LABEL, SKIP_LABEL)


def _json_default(obj):
    """Serialise a value json cannot encode natively (the json.dumps default).

    A numpy scalar becomes its Python value and an ndarray its content hash
    (so a stray array does not dump wholesale); anything else becomes its repr.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return joblib.hash(obj)
    return repr(obj)


def _cell(value):
    """Coerce a recorded value to its serialised snapshot form.

    Used wherever a recorded value is kept -- at record time, on disk, and in
    a flatten_to_df cell -- all via _json_default, so they agree: a scalar
    stays itself, an ndarray or opaque object becomes its content-hash / repr,
    a small list/dict stays structural (arrays hashed; tuples render as lists).
    DAG-edge identity lives in the hash maps, not these cells.
    """
    return json.loads(json.dumps(value, default=_json_default))


def _recurse_cell(value, base, sep='.') -> dict:
    """Expand a nested cell value into a flat {column: scalar} by key-path.

    The opt-in counterpart to _cell, for outputs in a record's recurse list
    (see Recorder): one column per scalar leaf, named by its path. A dict
    descends per key (base.key), a list/tuple per index (base.0, base.1) so
    the row count holds and the structure widens, a scalar ends as
    {base: value}. An empty dict/list yields no column. value is the _cell
    form, so only dict/list/scalar arise.
    """
    if isinstance(value, dict):
        flat = {}
        for k, v in value.items():
            flat.update(_recurse_cell(v, f'{base}{sep}{k}', sep))
        return flat
    if isinstance(value, (list, tuple)):
        flat = {}
        for i, v in enumerate(value):
            flat.update(_recurse_cell(v, f'{base}{sep}{i}', sep))
        return flat
    return {base: value}


def _flatten_record(record, prefix, sep='.') -> dict:
    """Flatten one record to a {column: cell} dict, all columns under prefix.

    prefix is the record's role (its short function name, e.g. fit). Each
    column -- the hash/function/time_sec metadata and one in.<name> /
    out.<name> per input/output -- is namespaced by it (fit.time_sec), so a
    leaf and its ancestors never collide. Values go through _cell, except an
    output in the recurse list, which _recurse_cell expands into per-key-path
    columns under out.<name> (e.g. run_ana.out.score.target.tp); lists widen
    via index suffixes, so ragged lists NaN-fill.
    """
    flat = {
        f'{prefix}{sep}hash': record['hash'],
        f'{prefix}{sep}function': record['function'],
        f'{prefix}{sep}time_sec': record.get('time_sec'),
    }
    for name, value in record.get('inputs', {}).items():
        flat[f'{prefix}{sep}in.{name}'] = _cell(value)
    recurse = record.get('recurse', [])
    for name, value in record.get('outputs', {}).items():
        base = f'{prefix}{sep}out.{name}'
        if name in recurse:
            flat.update(_recurse_cell(_cell(value), base, sep))
        else:
            flat[base] = _cell(value)
    return flat


def _ancestor_depths(start, parents_of) -> dict:
    """Map every transitive ancestor of start to its depth, breadth-first.

    Walks parents_of transitively from start, recording each ancestor's first
    (shortest) hop count. Cycle-safe via the visited set, should a hash
    collision forge a back edge. start itself is excluded.

    Args:
        start: the leaf record key to walk up from.
        parents_of: maps a record key to its immediate-producer keys.

    Returns:
        ancestor record key -> depth in hops from start (start omitted).
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
        records (dict): hash -> record dict, one per executed decorated call;
            its fields are the record layout in the module docstring.
            inputs/outputs are _cell snapshots, not live objects;
            input_hashes/output_hashes hash only link_types values (the DAG
            edges). A repeat hash overwrites and warns.
        folder (Path | None): directory the records mirror to (one <hash>.json
            each), made on construction, or None for in-memory only.
        link_types (tuple[type]): classes whose values are content-hashed into
            DAG edges (e.g. (Experiment,)); narrowing to a few domain types
            keeps trivial values (a scalar 2, a shared mask) from forging
            spurious edges. Empty (default) links nothing. Values must be
            joblib-hashable or recording raises.
    """

    def __init__(self, folder=None, link_types=()):
        """Set up the records map and optional mirror folder.

        Args:
            folder: directory to mirror records to as folder/<hash>.json
                (made if absent), or None for in-memory only.
            link_types (tuple[type]): classes content-hashed into DAG edges;
                see the class Attributes.

        Raises:
            TypeError: if any link_types entry is not a class.
        """
        self.records = {}
        self.folder = Path(folder) if folder is not None else None
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
        self.link_types = tuple(link_types)
        if not all(isinstance(t, type) for t in self.link_types):
            raise TypeError(
                "link_types must be classes (a value is linked when "
                "isinstance(value, link_types))")

    @staticmethod
    def _args_hash(fnc, args, kwargs, ignore=()) -> str:
        """Compute the args hash joblib keys a cached call by.

        filter_args (bind, apply defaults, drop ignored) then joblib.hash, so
        a record keys by the same id joblib caches under, without touching the
        MemorizedFunc. Pass the same ignore list given to MEMORY.cache, or the
        two keys diverge; coerce_mmap=False matches its mmap_mode=None. A bound
        method's receiver is included, so distinct receivers hash distinctly.

        Args:
            fnc: the function being hashed (its signature drives filter_args).
            args (tuple): positional call arguments.
            kwargs (dict): keyword call arguments.
            ignore (iterable[str]): parameter names dropped before hashing,
                matching MEMORY.cache's ignore list.

        Returns:
            the joblib args hash (hex digest), the on-disk cache-entry key.
        """
        return joblib.hash(filter_args(fnc, list(ignore), args, kwargs))

    def __call__(self, output_name=None, output_name_list=None,
                 recurse_out_list=None, ignore=()):
        """Build a decorator that records calls under one or more output names.

        Pass exactly one of output_name (the whole return under one name) or
        output_name_list (a tuple/list return unpacked onto those names).
        recurse_out_list is top-level metadata, never part of the key: it names
        outputs whose nested dict/list flatten_to_df expands per key-path into
        out.<name>.<path> columns (e.g. out.score.target.tp) not one cell.

        Args:
            output_name (str | None): single name for the whole return.
            output_name_list (tuple | list | None): names for an unpacked
                tuple/list return; non-empty, no duplicates.
            recurse_out_list (tuple | list | None): output names to expand per
                key-path; each must be a declared output. None recurses none.
            ignore (tuple | list): parameter names to drop from both the
                record key and the recipe kwargs -- the non-declarative
                arguments (the linked Experiment, an array companion, a label).
                Must be the same list given to @MEMORY.cache: the key is
                joblib's args hash, so filtering anything different would name
                a cache entry that does not exist. link_types inputs and
                parent_uid leave the recipe regardless.

        Returns:
            a decorator that wraps a function for recording.

        Raises:
            ValueError: neither or both output args given, duplicate
                output_name_list names, or a recurse name not declared.
            TypeError: a name is not a str, output_name_list is empty, or
                recurse_out_list is not a tuple/list of str.
        """
        # require exactly one of output_name / output_name_list
        if (output_name is None) == (output_name_list is None):
            raise ValueError(
                "provide exactly one of output_name or output_name_list")

        if output_name is not None:
            if not isinstance(output_name, str):
                raise TypeError("output_name must be a str")
        else:
            if (not isinstance(output_name_list, (tuple, list))
                    or len(output_name_list) == 0):
                raise TypeError(
                    "output_name_list must be a non-empty tuple/list")
            if not all(isinstance(n, str) for n in output_name_list):
                raise TypeError("output_name_list entries must be str")
            # only possible collision: a repeated output name
            if len(set(output_name_list)) != len(output_name_list):
                raise ValueError("output_name_list has duplicate names")

        # recurse_out_list must name declared outputs (so a typo cannot
        # silently recurse nothing); normalise None -> () for the record
        recurse = (tuple(recurse_out_list)
                   if recurse_out_list is not None else ())
        if (not isinstance(recurse, tuple)
                or not all(isinstance(n, str) for n in recurse)):
            raise TypeError(
                "recurse_out_list must be a tuple/list of str output names")
        declared = ({output_name} if output_name is not None
                    else set(output_name_list))
        unknown = [n for n in recurse if n not in declared]
        if unknown:
            raise ValueError(
                f"recurse_out_list names not declared outputs: {unknown}")

        ignore_names = tuple(ignore)
        if not all(isinstance(n, str) for n in ignore_names):
            raise TypeError("ignore must be a tuple/list of str param names")

        def decorator(fnc):
            """Wrap fnc so each successful call records under its args hash."""
            sig = inspect.signature(fnc)

            # a bound method's signature excludes self, so capture the receiver
            # from __self__ as the 'self' input -- no passthrough needed
            is_method = inspect.ismethod(fnc)

            # *args has no names to record, so reject it at decoration time;
            # **kwargs is fine (named values, spliced up a level below)
            var_kw = None
            for name, param in sig.parameters.items():
                if param.kind is inspect.Parameter.VAR_POSITIONAL:
                    raise TypeError(
                        f"{fnc.__name__} uses *{name}; recorder requires "
                        f"named parameters (no *args)"
                    )
                if param.kind is inspect.Parameter.VAR_KEYWORD:
                    var_kw = name

            @functools.wraps(fnc)
            def wrapped(*args, **kwargs):
                """Run fnc, then snapshot its inputs/outputs into a record."""
                # capture inputs (incl. defaults); serialised at record time
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                inputs = dict(bound.arguments)

                # bound-method receiver as the 'self' input (first, for
                # readability), from __self__ since the signature omits it
                if is_method:
                    inputs = {'self': fnc.__self__, **inputs}

                # splice **kwargs up a level: {'kwargs': {'x': 1}} -> {'x': 1};
                # sig.bind already routes named keywords, so no collision here
                if var_kw is not None:
                    inputs.update(inputs.pop(var_kw))

                # time only the wrapped call; an exception propagates, nothing
                # recorded (like joblib.Memory on a failed call)
                t0 = time.perf_counter()
                out = fnc(*args, **kwargs)
                time_sec = time.perf_counter() - t0

                # name the outputs; the shape checks raise without recording
                if output_name is not None:
                    outputs = {output_name: out}
                else:
                    if not isinstance(out, (tuple, list)):
                        raise TypeError(
                            f"{fnc.__name__} declared output_name_list "
                            f"but returned {type(out).__name__}; "
                            f"expected a tuple/list"
                        )
                    if len(out) != len(output_name_list):
                        raise ValueError(
                            f"{fnc.__name__} returned {len(out)} values "
                            f"but output_name_list has "
                            f"{len(output_name_list)} names"
                        )
                    outputs = dict(zip(output_name_list, out))

                # content-hash the link_types inputs/outputs from the live
                # values (before the snapshot): an input that is another call's
                # output shares its joblib.hash, so flatten_to_df links them.
                # Hashing only domain types avoids trivial-value edges.
                input_hashes = {n: joblib.hash(v) for n, v in inputs.items()
                                if isinstance(v, self.link_types)}
                output_hashes = {n: joblib.hash(v) for n, v in outputs.items()
                                 if isinstance(v, self.link_types)}

                # the declared identity of this call (see the module
                # docstring). A call that consumes a linked input without being
                # told its parent_uid cannot name its lineage, so it records no
                # recipe rather than a uid claiming to be a root.
                parent_uid = inputs.get('parent_uid')
                consumes_link = any(isinstance(v, self.link_types)
                                    for v in inputs.values())
                recipe_fields = {}
                if parent_uid is not None or not consumes_link:
                    recipe_kwargs = {
                        n: v for n, v in inputs.items()
                        if n not in ignore_names and n != 'parent_uid'
                        and not isinstance(v, self.link_types)}
                    recipe_fields = Recipe(
                        fnc.__qualname__, recipe_kwargs,
                        parents=(parent_uid,) if parent_uid else ()).as_dict()

                # snapshot to _cell form now, so the record keeps no live
                # reference to the (heavy) call values -- they can be collected
                # once the call returns. The DAG hashes above used the live
                # values, so nothing is lost.
                inputs = {n: _cell(v) for n, v in inputs.items()}
                outputs = {n: _cell(v) for n, v in outputs.items()}

                # key by joblib's args hash, matching the cache entry the same
                # call writes
                key = self._args_hash(fnc, args, kwargs, ignore_names)
                self._store(key, {
                    "hash": key,
                    "cache_key": key,
                    "function": fnc.__qualname__,
                    **recipe_fields,
                    **({"recurse": list(recurse)} if recurse else {}),
                    "inputs": inputs,
                    "outputs": outputs,
                    "input_hashes": input_hashes,
                    "output_hashes": output_hashes,
                    "time_sec": time_sec,
                })
                return out

            # stamp the filter onto the wrapper so a reader naming this call's
            # uid (recipe.recipe_for_call) drops exactly what was dropped here,
            # without the list being repeated at the call site
            wrapped._recipe_ignore = ignore_names
            return wrapped

        return decorator

    def _store(self, key, record) -> None:
        """Store a record by its hash key, overwriting and warning on repeat.

        A repeat key is the collision the module docstring describes; we keep
        the latest and warn so it is never silent. With a folder, it also
        mirrors to <hash>.json. When GLOW_RECORD_LOG is set (the AWS worker
        turns it on) it also prints one [record] line per write, to pair with
        the uploader's [upload] log for diagnosing lost records.
        """
        if key in self.records:
            warnings.warn(
                f"recorder overwriting record for hash {key} "
                f"({record['function']}); a prior call shared this args hash")
        self.records[key] = record
        # opt-in write log: one line per record as it is written, so a worker's
        # CloudWatch stream tells a record that was written-then-lost (a Spot
        # kill before its upload) from one never written. Silent for local runs.
        if os.environ.get('GLOW_RECORD_LOG'):
            print(f"[record] {record['function']} {key}", flush=True)
        if self.folder is not None:
            self._write(key, record)

    def _write(self, key, record) -> None:
        """Atomically mirror one record to folder/<hash>.json.

        Writes a temp file in the same dir then os.replace, so a reader never
        sees a half-written file; the hash names the file, so writers never
        contend.
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
        """Read every <hash>.json under folder into records, and return it.

        Merges the on-disk records into the in-memory map, so a reader
        reconstitutes what parallel writers produced. A no-op returning the
        current records when no folder is set.
        """
        if self.folder is None:
            return self.records
        for p in self.folder.glob('*.json'):
            rec = json.loads(p.read_text())
            self.records[rec['hash']] = rec
        return self.records

    def flatten_to_df(self, prefix_sep='.', leaf_keys=None):
        """Flatten the records into a leaf-per-row provenance DataFrame.

        Reads records as a DAG: B depends on A when B declares A's uid among
        its parents. A record written before the recipe fields is walked by the
        legacy edge instead (its input_hash equals an output_hash), so old and
        new records flatten together. Each leaf -- one no other feeds -- is
        one row carrying its own fields plus every ancestor's. A benchmark leaf
        is usually a score step over the fit it scored and the setup that built
        the data, so one row holds a whole trial: swept axes (setup), recipe
        and timing (fit), result (score).

        Each record contributes hash/function/time_sec and one
        in.<name> / out.<name> per input/output (via _cell), namespaced by its
        short function name (fit.time_sec). A name repeated in a row is
        suffixed #2/#3; the leaf goes first then ancestors shallowest-first, so
        prefixes are deterministic. A record with no link_types hashes is its
        own leaf. An output in the recurse list is expanded by _recurse_cell
        into per-key-path columns instead (see __call__).

        Operates on the in-memory records; call load() first to fold in other
        writers' files. leaf_keys restricts rows to chosen leaves (still walked
        to their ancestors), e.g. one CONFIG cache's leaves; missing
        keys are skipped. None uses every DAG leaf.

        Args:
            prefix_sep (str): separator between a prefix and its column name
                (the . in fit.time_sec).
            leaf_keys (iterable[str] | None): record keys to emit rows for;
                None walks every DAG leaf.

        Returns:
            a pandas DataFrame, one row per leaf; columns are the union across
            leaves, NaN where a leaf lacks an ancestor's column.
        """
        import pandas as pd

        records = self.records

        # uid -> its record, for the declared edges below
        key_of_uid = {rec['uid']: key for key, rec in records.items()
                      if rec.get('uid')}

        # output content hash -> the record that produced it, for the legacy
        # edges. A collision (two calls emitting an equal value) keeps the last
        # writer; in practice the heavy domain objects are unique per trial.
        # Hashes are always joblib.hash digests, so the is-not-None guards
        # below are defensive.
        producer_of = {}
        for key, rec in records.items():
            for h in rec.get('output_hashes', {}).values():
                if h is not None:
                    producer_of[h] = key

        def parents_of(key):
            """Records that produced this record's inputs (no self-edge).

            The union of both edge kinds: the declared parents (naming their
            producers by uid, an edge that holds on any machine) and the legacy
            content-hash join. Taking both keeps a mixed record set linked -- a
            new leaf whose ancestor predates the recipe fields, or an old leaf
            under a rebuilt ancestor -- and neither kind can invent an edge the
            other would deny.
            """
            rec = records[key]
            ps = {key_of_uid[uid] for uid in (rec.get('parents') or ())
                  if uid in key_of_uid}
            ps |= {producer_of[h]
                   for h in rec.get('input_hashes', {}).values()
                   if producer_of.get(h) is not None}
            return ps - {key}

        if leaf_keys is not None:
            # caller-chosen leaves; skip keys with no record (stale/dead-end)
            leaves = [key for key in leaf_keys if key in records]
        else:
            # a leaf feeds no other record by either edge kind: no record names
            # its uid as a parent, and none of its output hashes is another's
            # input (the self-exclusion keeps an identity call a leaf, and a
            # None hash never disqualifies one). Both must hold, so a record
            # consumed only through the legacy edge is not mistaken for a leaf.
            claimed = {uid for rec in records.values()
                       for uid in (rec.get('parents') or ())}
            consumers_of = defaultdict(set)
            for key, rec in records.items():
                for h in rec.get('input_hashes', {}).values():
                    if h is not None:
                        consumers_of[h].add(key)
            leaves = [
                key for key, rec in records.items()
                if rec.get('uid') not in claimed
                and all(not (consumers_of[h] - {key})
                        for h in rec.get('output_hashes', {}).values()
                        if h is not None)]

        rows = []
        for leaf in leaves:
            depth = _ancestor_depths(leaf, parents_of)
            # leaf first (claims the un-suffixed prefix), then ancestors
            # shallowest-first (then function, hash) so prefixes are
            # deterministic; a repeated short name gets #2 / #3
            ordered = [leaf] + sorted(
                depth, key=lambda k: (depth[k], records[k]['function'], k))
            row = {}
            seen = {}
            for key in ordered:
                role = records[key]['function'].rsplit('.', 1)[-1]
                seen[role] = seen.get(role, 0) + 1
                prefix = role if seen[role] == 1 else f'{role}#{seen[role]}'
                row.update(
                    _flatten_record(records[key], prefix, sep=prefix_sep))
            rows.append(row)

        return pd.DataFrame(rows)
