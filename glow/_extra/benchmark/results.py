"""Slice the shared provenance frame into one tidy CSV per CONFIG cache.

The driver writes every leaf into one shared RECORDER (data.RECORDER):
RECORDER.flatten_to_df yields one row per leaf carrying the data -> (plant ->)
score chain that produced it (see recorder / driver). That frame is
config-agnostic -- every cell ever run, across every figure -- so this module
re-attaches the CONFIG catalogue: for one cache name it selects that cache's
leaves and walks each up to its ancestors, then writes name.csv.

Membership is recomputed from the current CONFIG at read time (not read off a
stored tag) by walking the recorded provenance DAG forward from the cache's
declared cells:

  - Anchor at the data records. A data cell's kwargs are pure (no upstream
    object), so its record key is joblib's args hash of the dispatched builder
    (_data_record_key), found with no experiment rebuilt; a cell never run has
    no record and drops out.
  - Step forward to the effect records. Every record already stores the
    link-type (Experiment) hashes of its inputs and outputs, so an effect
    record that consumed an anchor's exp is one forward edge away; keep those
    whose stored inputs match one of the cache's effect cells. A None effect
    cell is the null path -- no effect record, the leaf hangs off the clean
    exp -- so the anchor itself is the frontier.
  - Step forward to the leaves. Take the fnc records off the frontier that
    carry the cache's leaf function (a cache runs its whole fnc grid, and a
    shared cell's other caches use a different leaf function).

Deriving membership from the current CONFIG means editing a cache (adding,
changing or dropping swept cells) is reflected at once: a dropped cell stops
matching, with nothing stale to prune. A cell shared by several caches (the
sweeps' common baseline) is one record reached by each cache's walk, so it
lands in each. The walk reads only the records (never the experiment cache), so
it is cheap; its one fragility is that it traverses the recorded edges, so a
missing intermediate (effect) record drops the leaves below it.
"""
import inspect
from collections import defaultdict
from pathlib import Path

from . import config
from .data import (RECORDER, data_factory_hcp, data_factory_wgn,
                   effect_factory_single, effect_factory_split)
from .file import get_path_result
from .recorder import _cell

# dispatch tables mirroring data_factory / effect_factory (which dispatch on a
# cell's 'source' / 'kind' key to the recorded builder that keys the record).
_DATA_FACTORY = {'wgn': data_factory_wgn, 'hcp': data_factory_hcp}
_EFFECT_FACTORY = {'single': effect_factory_single,
                   'split': effect_factory_split}


def _raw(fnc):
    """Return the undecorated function under a memoised+recorded callable.

    Peels joblib.Memory's .func then the recorder wrapper (inspect.unwrap), so
    its signature / qualname / args hash match what the recorder filed the
    record under (see recorder.Recorder._args_hash).
    """
    return inspect.unwrap(getattr(fnc, 'func', fnc))


def _data_record_key(kwargs_data) -> str:
    """Return the record key of the data_factory call for one data cell.

    Dispatches on the cell's source like data_factory, then computes joblib's
    args hash of the dispatched builder over the cell's remaining kwargs -- the
    same key the recorder filed the build under -- so the data record is found
    with nothing rebuilt (the extenter is content-hashed, so a freshly built
    one matches the recorded call).

    Args:
        kwargs_data (dict): one data_factory cell, e.g. {'source': 'wgn', ...}.

    Returns:
        str: the args-hash record key (present in the records iff the cell ran).
    """
    kwargs = {k: v for k, v in kwargs_data.items() if k != 'source'}
    return RECORDER._args_hash(_raw(_DATA_FACTORY[kwargs_data['source']]),
                               (), kwargs)


def _expected_inputs(fnc, kwargs, drop=()) -> dict:
    """Return the celled inputs a recorded call to fnc(**kwargs) would store.

    Binds kwargs to fnc's signature and applies its defaults (as the recorder
    does at record time), then cells each value the same way, so the result
    compares equal to the matching record's stored inputs. drop names inputs to
    omit -- the driver-supplied / link-type args (exp) a config cell omits.

    Args:
        fnc (Callable): the recorded builder whose signature to bind against.
        kwargs (dict): the config cell's kwargs (dispatch keys already removed).
        drop (tuple[str]): input names to leave out of the comparison.

    Returns:
        dict: {input name: celled value} for the cell's own (non-dropped) args.
    """
    bound = inspect.signature(_raw(fnc)).bind_partial(**kwargs)
    bound.apply_defaults()
    return {k: _cell(v) for k, v in bound.arguments.items() if k not in drop}


def _inputs_match(record, expected) -> bool:
    """True if the record's inputs equal expected on every expected key."""
    inputs = record.get('inputs', {})
    return all(inputs.get(k) == v for k, v in expected.items())


def config_leaf_keys(name: str) -> list:
    """Return the record keys of CONFIG cache name's leaves (forward walk).

    Walks the recorded provenance DAG forward from the cache's declared cells:
    anchor at the data records (_data_record_key), step to the effect records
    whose stored inputs match a cache effect cell (or, for a None cell, take the
    data record as the frontier -- the null path has no effect record), then
    take the fnc records off that frontier with the cache's leaf function.
    Membership is thus derived from the current CONFIG, so a dropped / changed
    cell simply stops matching (see the module docstring).

    Reads the in-memory records; call RECORDER.load() first to fold in what a
    parallel run left on disk.

    Args:
        name (str): a CONFIG cache name (e.g. 'sweep_llr').

    Returns:
        list[str]: the leaf record keys for this cache (empty if none ran).
    """
    kwargs_data_list, kwargs_effect_list, _, fnc = config.CONFIG[name]
    records = RECORDER.records

    # forward edges: an output link-hash -> the records consuming it as input
    consumers_of = defaultdict(set)
    for key, rec in records.items():
        for h in rec.get('input_hashes', {}).values():
            if h is not None:
                consumers_of[h].add(key)

    def children_of(key):
        """The records consuming any of this record's link outputs (no self)."""
        kids = set()
        for h in records[key].get('output_hashes', {}).values():
            if h is not None:
                kids |= consumers_of.get(h, set())
        kids.discard(key)
        return kids

    # anchor at the data records (a cell never run has no record, so drops out)
    anchors = {k for k in map(_data_record_key, kwargs_data_list)
               if k in records}

    # effect frontier: the leaves' parents. A None cell is the null path (no
    # effect record; the leaf hangs off the clean exp), so the anchor is the
    # frontier; a planted cell's parents are the effect records off an anchor
    # whose stored inputs match the cell.
    parents = set()
    for kwargs_effect in kwargs_effect_list:
        if kwargs_effect is None:
            parents |= anchors
            continue
        kind = kwargs_effect.get('kind', 'single')
        expected = _expected_inputs(
            _EFFECT_FACTORY[kind],
            {k: v for k, v in kwargs_effect.items() if k != 'kind'},
            drop=('exp',))
        for a in anchors:
            parents |= {c for c in children_of(a)
                        if _inputs_match(records[c], expected)}

    # leaves: the fnc records off the frontier. A cache runs its whole fnc grid,
    # and a cache sharing a (data, effect) cell runs a different leaf function
    # (run_ana vs run_segment / run_stat / ...), so the recorded function name
    # alone selects this cache's leaves; the recipe within the grid is recovered
    # downstream (see benchmark.plot).
    fnc_name = _raw(fnc).__qualname__
    leaves = set()
    for p in parents:
        for c in children_of(p):
            if records[c]['function'] == fnc_name:
                leaves.add(c)
    return list(leaves)


def config_results_df(name: str):
    """Walk the shared provenance frame up from one cache's leaves.

    Selects the cache's leaves (config_leaf_keys) and flattens each into a row
    carrying its data -> (plant ->) score chain (flatten_to_df seeded from those
    leaves). A row a sweep shares with another sweep is selected by both, since
    the walk reaches the shared record from either cache. Loads the recorder
    first to fold in any records a parallel run / other workers left on disk.
    Empty when no cell of this cache has run.

    Args:
        name (str): a CONFIG cache name.

    Returns:
        pandas.DataFrame: this cache's rows (empty when none have run).
    """
    RECORDER.load()
    return RECORDER.flatten_to_df(leaf_keys=config_leaf_keys(name))


def write_config_csvs(out_dir=None, names=None) -> dict:
    """Write one name.csv per CONFIG cache with leaves on disk.

    Loads the shared recorder once, then for each cache selects its leaves
    (config_leaf_keys) and flattens them into the per-figure results frame the
    plots consume, writing each non-empty slice to out_dir/<name>.csv. A cache
    with no leaves yet is skipped (no empty file).

    Args:
        out_dir (str | Path | None): destination directory; defaults to
            glow's per-user results dir (get_path_result).
        names (iterable[str] | None): cache names to export; None exports
            every CONFIG entry.

    Returns:
        dict[str, Path]: the {cache name: written csv path} for caches that
            had rows (those skipped for being empty are omitted).
    """
    out_dir = Path(out_dir) if out_dir is not None else get_path_result()
    out_dir.mkdir(parents=True, exist_ok=True)

    RECORDER.load()

    written = {}
    for name in (names if names is not None else config.CONFIG):
        sub = RECORDER.flatten_to_df(leaf_keys=config_leaf_keys(name))
        if sub.empty:
            continue
        path = out_dir / f'{name}.csv'
        sub.to_csv(path, index=False)
        written[name] = path
    return written
