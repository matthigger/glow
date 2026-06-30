"""Enumerate a CONFIG cache's data cells as the unit of AWS work.

resolve_cells turns a cache name into the cache's ordered data cells plus its
shared effect / fnc grids and leaf fnc -- the run bundle the driver pickles and
ships to S3 (see glow._extra.aws.driver). Which sources a cache spans is a
CONFIG property (its data grid), not a driver knob; resolve_cells runs whatever
the cache declares. It is driver-side only:
a worker runs whatever bundle it is handed and never calls this, so the two
cannot disagree on the cell list. The enumeration is still deterministic --
itertools.product over fixed ranges, seeded RNG for the HCP feature subset (see
glow._extra.benchmark.config) -- which keeps cell ordering reproducible across
runs, so warm-cache hits line up.

A unit is one data cell (a data_factory kwargs dict), not a single function
call: a worker runs the cell's whole effect x analysis subtree serially in
one process (build the clean exp once, plant each effect, fit each recipe),
exactly the per-data-cell task the local parallel driver distributes (see
glow._extra.benchmark.driver). Splitting finer would rebuild the exp per
child; this keeps the single-owner build the benchmark layer is written for.
"""

from glow._extra.benchmark.config import CONFIG


def resolve_cells(name: str):
    """Return the data cells of CONFIG cache name plus its shared grids / fnc.

    Reads CONFIG[name] (the (data, effect, fnc-kwargs, fnc) four-tuple) and
    returns its data grid in full -- whatever sources the cache declares, since
    sources are a CONFIG property, not a driver knob. The data grid is a
    concrete, deterministically ordered list (see the module docstring), so the
    returned cell list is reproducible across runs -- the driver ships it in
    the run bundle (see glow._extra.aws.driver).

    Args:
        name (str): a CONFIG cache name (e.g. 'sweep_llr').

    Returns:
        data_cells (list[dict]): the data_factory kwargs dicts, in CONFIG order.
        kwargs_effect_list (list[dict | None]): the cache's effect grid.
        kwargs_fnc_list (list[dict]): the cache's leaf-fnc kwargs grid.
        fnc (Callable): the cache's leaf measurement (e.g. run_ana).

    Raises:
        KeyError: name is not a CONFIG cache.
    """
    kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc = CONFIG[name]
    return list(kwargs_data_list), kwargs_effect_list, kwargs_fnc_list, fnc
