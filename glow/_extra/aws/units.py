"""Enumerate a CONFIG cache's data cells as the unit of AWS work.

The driver and the worker must agree, with no shared state, on what array
child i runs. They both call resolve_cells(name, sources), which is a pure
function of the benchmark CONFIG catalogue -- itertools.product over fixed
ranges, seeded RNG for the HCP feature subset (see
glow._extra.benchmark.config) -- so the ordered cell list is identical in the
submitting process and in every worker. Array index i then maps to the same
data cell on both sides without the driver shipping anything per cell.

A unit is one data cell (a data_factory kwargs dict), not a single function
call: the worker runs the cell's whole effect x analysis subtree serially in
one process (build the clean exp once, plant each effect, fit each recipe),
exactly the per-data-cell task the local parallel driver distributes (see
glow._extra.benchmark.driver). Splitting finer would rebuild the exp per
child; this keeps the single-owner build the benchmark layer is written for.
"""

from glow._extra.benchmark.config import CONFIG


def resolve_cells(name: str, sources=('wgn',)):
    """Return the data cells of CONFIG cache name whose source is selected.

    Reads CONFIG[name] (the (data, effect, fnc-kwargs, fnc) four-tuple) and
    filters its data grid to the cells whose 'source' is in sources, leaving
    the effect / fnc grids and the leaf fnc untouched. The data grid is a
    concrete, deterministically ordered list (see the module docstring), so
    the returned cell list is reproducible -- index i is the same cell here
    and on the worker.

    Args:
        name (str): a CONFIG cache name (e.g. 'sweep_llr').
        sources (tuple[str]): the data sources to keep ('wgn' and/or 'hcp').
            Defaults to WGN only, the first AWS milestone (HCP needs its
            per-feature data staged to S3 first).

    Returns:
        data_cells (list[dict]): the selected data_factory kwargs dicts, in
            their CONFIG order.
        kwargs_effect_list (list[dict | None]): the cache's effect grid.
        kwargs_fnc_list (list[dict]): the cache's leaf-fnc kwargs grid.
        fnc (Callable): the cache's leaf measurement (e.g. run_ana).

    Raises:
        KeyError: name is not a CONFIG cache.
    """
    kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc = CONFIG[name]
    data_cells = [c for c in kwargs_data_list if c.get('source') in sources]
    return data_cells, kwargs_effect_list, kwargs_fnc_list, fnc
