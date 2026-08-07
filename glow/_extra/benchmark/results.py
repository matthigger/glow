"""Recover which recorded leaves and which finished cells are a cache's.

The driver writes every leaf into one shared RECORDER (data.RECORDER):
RECORDER.flatten_to_df yields one row per leaf carrying the data -> (plant ->)
score chain that produced it (see recorder / driver). Those records are
config-agnostic -- every cell ever run, across every figure -- so this module
re-attaches the CONFIG catalogue: for one cache name it names that cache's
leaves (config_leaf_keys, the rows make_csv exports) and its planted cells not
yet fully recorded (incomplete_cell_indices, the local and AWS rerun skip).

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

from . import config
from .data import (DATA_FACTORY, EFFECT_FACTORY, RECORDER, data_recipe,
                   effect_recipe)
from .recipe import declared_ignore, recipe_for_call
from .recorder import _cell

# the builders a cell's 'source' / 'kind' key selects, shared with the runner
# (data.py) rather than mirrored here, so a reader and a runner cannot disagree
# about which builder a cell means.
_DATA_FACTORY = DATA_FACTORY
_EFFECT_FACTORY = EFFECT_FACTORY


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


def _optional_inputs(fnc, kwargs, drop=()) -> set:
    """Return the expected-input names that come from fnc's defaults.

    A cell specifies some of a builder's parameters and leaves the rest at their
    defaults. Only the specified ones must appear in a record: a parameter added
    to the builder since a record was written is absent from it, and its absence
    means the default (see _inputs_match).

    Args:
        fnc (Callable): the recorded builder.
        kwargs (dict): the config cell's kwargs (dispatch keys removed).
        drop (tuple[str]): input names left out of the comparison.

    Returns:
        set[str]: expected names the cell did not specify.
    """
    return set(_expected_inputs(fnc, kwargs, drop=drop)) - set(kwargs)


def _inputs_match(record, expected, optional=()) -> bool:
    """True if the record's inputs equal expected on every expected key.

    A name in optional may be missing from the record -- it comes from a builder
    default, and a record written before that parameter existed stored nothing
    for it, so requiring it would drop every older record of that builder. A
    name the record does carry must still match.

    Args:
        record (dict): the stored record.
        expected (dict): {input name: celled value} to match (_expected_inputs).
        optional (iterable[str]): expected names allowed to be absent
            (_optional_inputs).

    Returns:
        bool: True if the record matches on every required key.
    """
    inputs = record.get('inputs', {})
    for name, value in expected.items():
        if name not in inputs:
            if name in optional:
                continue
            return False
        if inputs[name] != value:
            return False
    return True


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
    kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc = \
        config.CONFIG[name]
    records = RECORDER.records

    # the declared path: name every leaf uid this cache's cells produce and
    # take the records filed under them. Independent of any array's bytes, so a
    # leaf computed on another machine is found here.
    want = set()
    for kwargs_data in kwargs_data_list:
        for kwargs_effect in kwargs_effect_list:
            want |= set(cell_leaf_uids(kwargs_data, kwargs_effect,
                                       kwargs_fnc_list, fnc))
    declared = {key for key, rec in records.items()
                if rec.get('uid') in want}

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
        cell = {k: v for k, v in kwargs_effect.items() if k != 'kind'}
        expected = _expected_inputs(_EFFECT_FACTORY[kind], cell, drop=('exp',))
        optional = _optional_inputs(_EFFECT_FACTORY[kind], cell, drop=('exp',))
        for a in anchors:
            parents |= {c for c in children_of(a)
                        if _inputs_match(records[c], expected, optional)}

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
    return list(declared | leaves)


def planted_cells(name: str) -> list:
    """Return cache name's (kwargs_data, kwargs_effect) cells in grid order.

    One cell is a data cell crossed with a single effect cell -- the unit the
    AWS driver submits and skips (an instance builds the data once, plants that
    one effect, and runs the whole fnc grid on it; see glow._extra.aws). The
    cross is data-major, effect-minor, both grids in CONFIG order, so the flat
    list is reproducible across runs. A None effect cell (the null path) rides
    through unchanged.

    Args:
        name (str): a CONFIG cache name.

    Returns:
        list[tuple[dict, dict | None]]: (kwargs_data, kwargs_effect) pairs,
            one per (data cell, effect cell) of the cache.
    """
    kwargs_data_list, kwargs_effect_list, _, _ = config.CONFIG[name]
    return [(kwargs_data, kwargs_effect)
            for kwargs_data in kwargs_data_list
            for kwargs_effect in kwargs_effect_list]


def cell_parent_uid(kwargs_data, kwargs_effect) -> str:
    """Return the uid of the Experiment one planted cell's leaves measure.

    Named from the cell's kwargs alone -- nothing is built and no record is
    read. A None effect cell is the null path, whose parent is the clean exp
    itself.

    Args:
        kwargs_data (dict): one data_factory cell.
        kwargs_effect (dict | None): one effect_factory cell, or None.

    Returns:
        uid (str): the parent uid the cell's leaves are passed.
    """
    uid_data = data_recipe(kwargs_data).uid
    if kwargs_effect is None:
        return uid_data
    return effect_recipe(kwargs_effect, uid_data).uid


def cell_leaf_uids(kwargs_data, kwargs_effect, kwargs_fnc_list, fnc) -> list:
    """Return the uids one cell's leaves are filed under, in grid order.

    The whole point of declared identity: a cell's leaf ids are a pure function
    of the CONFIG, so what a run will produce (or has produced) can be named
    without building an Experiment, reading a record, or comparing any array.

    Args:
        kwargs_data (dict): one data_factory cell.
        kwargs_effect (dict | None): one effect_factory cell, or None.
        kwargs_fnc_list (list[dict]): the fnc-kwargs grid.
        fnc (Callable): the leaf measurement (memoised + recorded).

    Returns:
        list[str]: one leaf uid per fnc-kwargs cell.
    """
    parent = cell_parent_uid(kwargs_data, kwargs_effect)
    return [recipe_for_call(fnc, kwargs, parents=(parent,)).uid
            for kwargs in kwargs_fnc_list]


def get_cell_complete(kwargs_fnc_list, fnc):
    """Build the predicate deciding whether one planted cell is finished.

    The records-side source of truth behind every rerun skip, local (driver)
    and AWS alike: a cell is a data cell crossed with one effect
    (planted_cells), and it is complete when every leaf that effect builds --
    one per fnc-kwargs cell -- is in the local records.

    A cell's leaf uids are named straight from the CONFIG (cell_leaf_uids), so
    completeness is a set-membership test against the recorded uids: nothing is
    rebuilt, no ancestor record has to be found, and the answer does not depend
    on any array's bytes. That is what lets a leaf computed elsewhere -- an AWS
    worker whose Experiment differs in its last bits, as two CPUs' will --
    count for the cell it belongs to.

    A cell whose uids are not all present falls back to the legacy walk (anchor
    at the data record, match the effect record off it by its stored inputs,
    then require one recorded leaf per fnc-kwargs cell), so cells recorded
    before the recipe fields still read complete and are not rerun. Both
    indexes are built once here and closed over, so the predicate is cheap per
    cell. Call RECORDER.load() first to fold in what other writers left on
    disk.

    Either way it errs safe: a cell that cannot be shown complete is rerun,
    never wrongly skipped.

    Args:
        kwargs_fnc_list (list[dict]): the fnc-kwargs grid; a cell counts as
            complete only with one recorded leaf per entry.
        fnc (Callable): the leaf measurement (memoised + recorded); names the
            cell's leaf uids, and its recorded name selects them on the legacy
            path.

    Returns:
        cell_complete (Callable): cell_complete(kwargs_data, kwargs_effect)
            -> bool, True when the cell's whole leaf set is recorded. A None
            kwargs_effect is the null path, whose parent is the data record.
    """
    records = RECORDER.records
    recorded_uids = {rec['uid'] for rec in records.values() if rec.get('uid')}

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

    fnc_name = _raw(fnc).__qualname__
    # one input fingerprint per fnc-kwargs cell, over the names that identify a
    # leaf. The ignore list is dropped: those are driver-supplied (exp, its
    # mask companion) or execution knobs (fit_params, label), and a record does
    # store them, but as provenance rather than identity -- so a cell that
    # named none of them when its record was written, or ran under different
    # ones, still has to match. Read off the decorator, as recipe_for_call
    # does, so the two stay in step.
    drop = ('exp', 'mask_target_list', *declared_ignore(fnc))
    fnc_expected = [(_expected_inputs(fnc, kwargs, drop=drop),
                     _optional_inputs(fnc, kwargs, drop=drop))
                    for kwargs in kwargs_fnc_list]

    def cell_complete(kwargs_data, kwargs_effect) -> bool:
        """True if every fnc-kwargs leaf of this cell is recorded."""
        want = cell_leaf_uids(kwargs_data, kwargs_effect, kwargs_fnc_list, fnc)
        if want and recorded_uids.issuperset(want):
            return True
        anchor = _data_record_key(kwargs_data)
        if anchor not in records:
            return False
        # the effect frontier: the anchor itself on the null path, else the
        # effect record(s) off it whose stored inputs match the effect cell
        if kwargs_effect is None:
            frontier = {anchor}
        else:
            kind = kwargs_effect.get('kind', 'single')
            cell = {k: v for k, v in kwargs_effect.items() if k != 'kind'}
            builder = _EFFECT_FACTORY[kind]
            expected = _expected_inputs(builder, cell, drop=('exp',))
            optional = _optional_inputs(builder, cell, drop=('exp',))
            frontier = {c for c in children_of(anchor)
                        if _inputs_match(records[c], expected, optional)}
            if not frontier:
                return False
        leaves = [records[c] for p in frontier for c in children_of(p)
                  if records[c]['function'] == fnc_name]
        return all(any(_inputs_match(rec, expected, optional)
                       for rec in leaves)
                   for expected, optional in fnc_expected)

    return cell_complete


def incomplete_cell_indices(name: str, kwargs_fnc_list=None) -> list:
    """Return the planted cells of cache name not fully recorded on disk.

    The AWS driver's rerun skip: it submits only these indices, so a rerun
    fills the gaps without recomputing finished work. Indices are positions in
    planted_cells, in grid order. See get_cell_complete for what counts as
    complete (and why it errs safe), and call RECORDER.load() first to fold in
    what other writers left on disk.

    kwargs_fnc_list narrows the leaf grid completeness is judged against -- the
    grid the caller will actually run (e.g. one method's recipes,
    grid.filter_ana_list), so a cell holding those leaves is skipped whatever
    the cache's other recipes are missing.

    Args:
        name (str): a CONFIG cache name.
        kwargs_fnc_list (list[dict] | None): the leaf grid to require; None
            (default) is the cache's own full grid.

    Returns:
        list[int]: the planted-cell indices still to run (empty when every cell
            of the cache is already complete on disk).
    """
    _, _, config_fnc_list, fnc = config.CONFIG[name]
    if kwargs_fnc_list is None:
        kwargs_fnc_list = config_fnc_list
    cell_complete = get_cell_complete(kwargs_fnc_list, fnc)
    return [i for i, (kwargs_data, kwargs_effect)
            in enumerate(planted_cells(name))
            if not cell_complete(kwargs_data, kwargs_effect)]


def stat_cell_df():
    """Return the vba_stat cache's run_stat leaves, keyed by planted cell.

    The stat bake-off is fit by AWS workers that ship the run_stat leaf back
    without its data_factory / effect_factory ancestors, so the forward DAG
    walk (config_leaf_keys) can attach neither source nor effect_llr and drops
    those leaves -- the vba_stat CSV holds only the locally-run subset. The
    stat comparison is within a planted cell (the five stats fit on one
    experiment), so this reads the run_stat records directly and tags each with
    its planted-exp link hash -- the cell id every variant of a cell shares --
    sidestepping the missing ancestors. Source / effect_llr are not recovered
    (they live in the absent ancestors); the comparison does not need them.

    Returns:
        pandas.DataFrame: one row per run_stat leaf, columns cell (the planted
            exp hash), ana (recipe repr), stat_name, and the score confusion
            counts num_vox / tp / fp / tn / fn (NaN where a leaf lacks them).
    """
    import pandas as pd

    RECORDER.load()
    rows = []
    for key, rec in RECORDER.records.items():
        if rec.get('function') != 'run_stat':
            continue
        inp = rec.get('inputs', {})
        score = (rec.get('outputs') or {}).get('score') or {}
        target = score.get('target') or {}
        rows.append({
            'cell': rec.get('input_hashes', {}).get('exp', key),
            'ana': inp.get('ana'),
            'stat_name': inp.get('stat_name'),
            'num_vox': score.get('num_vox'),
            'tp': target.get('tp'), 'fp': target.get('fp'),
            'tn': target.get('tn'), 'fn': target.get('fn'),
        })
    return pd.DataFrame(rows)
