"""Enumerate a CONFIG cache's data cells as the unit of AWS work.

resolve_cells turns a cache name + sources into the cache's ordered data cells
plus its shared effect / fnc grids and leaf fnc -- the run bundle the driver
pickles and ships to S3 (see glow._extra.aws.driver). It is driver-side only:
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


def resolve_cells(name: str, sources=('wgn',)):
    """Return the data cells of CONFIG cache name whose source is selected.

    Reads CONFIG[name] (the (data, effect, fnc-kwargs, fnc) four-tuple) and
    filters its data grid to the cells whose 'source' is in sources, leaving
    the effect / fnc grids and the leaf fnc untouched. The data grid is a
    concrete, deterministically ordered list (see the module docstring), so
    the returned cell list is reproducible across runs -- the driver ships it
    in the run bundle (see glow._extra.aws.driver).

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
