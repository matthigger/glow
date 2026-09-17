"""Recover which recorded leaves and which finished cells are a cache's.

The records the driver writes are config-agnostic -- every cell ever run,
across every figure -- so this module re-attaches the CONFIG catalogue: for one
cache name it names that cache's leaves (config_leaf_keys, the rows make_csv
exports) and its planted cells not yet fully recorded
(incomplete_cell_indices, the rerun skip).

Membership is recomputed from the current CONFIG at read time (not read off a
stored tag) by naming what the cache's cells produce and taking the records
filed under those names: a cell's uid chain is a pure function of its declared
kwargs (cell_leaf_uids), so a leaf can be named without building an
Experiment, reading an upstream record, or comparing any array. A cell never
run simply has no record and drops out.

Deriving membership from the current CONFIG means an edited cache is reflected
at once, with nothing stale to prune, and a cell shared by several caches lands
in each. Reading only uids means a leaf computed on another machine counts for
its cell, and a missing intermediate record hides nothing below it.
"""
from . import config
from .cell import exp_effect_recipe
from .recipe import recipe_for_call
from .store import RECORDER


def config_leaf_keys(name: str) -> list:
    """Return the record keys of CONFIG cache name's leaves.

    Names every leaf uid this cache's cells produce (cell_leaf_uids) and takes
    the records filed under them, so membership is derived from the current
    CONFIG and independent of any array's bytes: a dropped or changed cell
    stops matching, and a leaf computed on another machine is found here.

    Reads the in-memory records; call RECORDER.load() first to fold in what a
    parallel run left on disk.

    Args:
        name (str): a CONFIG cache name (e.g. 'sweep_llr').

    Returns:
        list[str]: the leaf record keys for this cache (empty if none ran).
    """
    kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc = \
        config.CONFIG[name]

    want = set()
    for kwargs_data in kwargs_data_list:
        for kwargs_effect in kwargs_effect_list:
            want |= set(cell_leaf_uids(kwargs_data, kwargs_effect,
                                       kwargs_fnc_list, fnc))
    return [key for key, rec in RECORDER.records.items()
            if rec.get('uid') in want]


def planted_cells(name: str) -> list:
    """Return cache name's (kwargs_data, kwargs_effect) cells in grid order.

    One cell is a data cell crossed with a single effect cell -- the unit a
    rerun skips: the data is built once, that one effect is planted, and the
    whole fnc grid is measured on it. The cross is data-major, effect-minor,
    both grids in CONFIG order, so the flat list is reproducible across runs. A
    None effect cell (the null path) rides through unchanged.

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
    """Return the uid of the cell one set of leaves measures.

    Named from the cell's kwargs alone -- nothing is realized and no record
    is read.

    Args:
        kwargs_data (dict): the data half of a cell.
        kwargs_effect (dict | None): the effect half, or None for the null
            path.

    Returns:
        uid (str): the parent uid the cell's leaves are passed.
    """
    return exp_effect_recipe(kwargs_data, kwargs_effect).uid


def cell_leaf_uids(kwargs_data, kwargs_effect, kwargs_fnc_list, fnc) -> list:
    """Return the uids one cell's leaves are filed under, in grid order.

    The whole point of declared identity: a cell's leaf ids are a pure function
    of the CONFIG, so what a run will produce (or has produced) can be named
    without building an Experiment, reading a record, or comparing any array.

    Args:
        kwargs_data (dict): the data half of a cell.
        kwargs_effect (dict | None): the effect half, or None.
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

    The records-side source of truth behind the rerun skip (driver): a cell is
    a data cell crossed with one effect, complete when every leaf that effect
    builds -- one per fnc-kwargs cell -- is in the records.

    A cell's leaf uids come straight from the CONFIG (cell_leaf_uids), so
    completeness is set membership against the recorded uids: nothing is
    rebuilt and the answer does not depend on any array's bytes, which is what
    lets a leaf computed on another machine count for its cell. The recorded
    uid index is built once and closed over; call RECORDER.load() first to
    fold in what other writers left on disk.

    It errs safe: a cell that cannot be shown complete is rerun.

    Args:
        kwargs_fnc_list (list[dict]): the fnc-kwargs grid; a cell counts as
            complete only with one recorded leaf per entry.
        fnc (Callable): the leaf measurement (memoised + recorded), which
            names the cell's leaf uids.

    Returns:
        cell_complete (Callable): cell_complete(kwargs_data, kwargs_effect)
            -> bool, True when the cell's whole leaf set is recorded. A None
            kwargs_effect is the null path, whose parent is the clean exp.
    """
    recorded_uids = {rec['uid'] for rec in RECORDER.records.values()
                     if rec.get('uid')}

    def cell_complete(kwargs_data, kwargs_effect) -> bool:
        """True if every fnc-kwargs leaf of this cell is recorded."""
        want = cell_leaf_uids(kwargs_data, kwargs_effect, kwargs_fnc_list,
                              fnc)
        return bool(want) and recorded_uids.issuperset(want)

    return cell_complete


def get_leaf_todo(kwargs_fnc_list, fnc):
    """Build the predicate naming which of one cell's leaves are unrecorded.

    The finer grain of the rerun skip: get_cell_complete answers whether a
    whole cell is done, this answers which of its leaves are not. A cell short
    one leaf need only run that leaf -- its siblings are already measured, and
    where only their records were merged (a fan-out ships records, not the
    joblib cache) rerunning them is a cold recompute rather than a cache hit.

    Membership is declared uids alone (cell_leaf_uids), so it errs safe: a leaf
    whose uid is not in the records -- including one recorded before the recipe
    fields existed -- reads as still to run. Call RECORDER.load() first to fold
    in what other writers left on disk.

    Args:
        kwargs_fnc_list (list[dict]): the fnc-kwargs grid.
        fnc (Callable): the leaf measurement (memoised + recorded).

    Returns:
        leaf_todo (Callable): leaf_todo(kwargs_data, kwargs_effect) ->
            list[dict], that cell's fnc-kwargs cells still to run, in grid
            order.
    """
    recorded_uids = {rec['uid'] for rec in RECORDER.records.values()
                     if rec.get('uid')}

    def leaf_todo(kwargs_data, kwargs_effect) -> list:
        """This cell's fnc-kwargs cells that are not already recorded."""
        uid_list = cell_leaf_uids(kwargs_data, kwargs_effect, kwargs_fnc_list,
                                  fnc)
        return [kwargs for kwargs, uid in zip(kwargs_fnc_list, uid_list)
                if not uid or uid not in recorded_uids]

    return leaf_todo


def incomplete_cell_indices(name: str, kwargs_fnc_list=None) -> list:
    """Return the planted cells of cache name not fully recorded on disk.

    The cache-level view of the rerun skip: what a sweep of this cache has
    left to do, so progress can be read off without running it. Indices are
    positions in planted_cells, in grid order. See get_cell_complete for what
    counts as
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


def stat_cell_df(name: str = None):
    """Return a run_stat cache's leaves, keyed by planted cell.

    The stat comparison is within a planted cell (the five stats fit on one
    experiment), so this reads the run_stat records directly and groups them
    by the parent uid each declares -- the cell id every variant of a cell
    shares. Source / effect_llr are not recovered (they live in the parent
    record); the comparison does not need them.

    Args:
        name (str | None): a CONFIG cache name to restrict to. Required
            whenever more than one run_stat cache has run: an unsmoothed
            recipe reprs the same in each of them, so only the cell it was fit
            on tells the caches apart, and that is what config_leaf_keys
            resolves. None reads every run_stat record.

    Returns:
        pandas.DataFrame: one row per run_stat leaf, columns cell (the planted
            exp uid), ana (recipe repr), stat_name, and the score confusion
            counts num_vox / tp / fp / tn / fn (NaN where a leaf lacks them).
    """
    import pandas as pd

    RECORDER.load()
    keep = None if name is None else set(config_leaf_keys(name))
    rows = []
    for key, rec in RECORDER.records.items():
        if rec.get('function') != 'run_stat':
            continue
        if keep is not None and key not in keep:
            continue
        inp = rec.get('inputs', {})
        score = (rec.get('outputs') or {}).get('score') or {}
        target = score.get('target') or {}
        rows.append({
            'cell': (rec.get('parents') or [key])[0],
            'ana': inp.get('ana'),
            'stat_name': inp.get('stat_name'),
            'num_vox': score.get('num_vox'),
            'tp': target.get('tp'), 'fp': target.get('fp'),
            'tn': target.get('tn'), 'fn': target.get('fn'),
        })
    return pd.DataFrame(rows)
