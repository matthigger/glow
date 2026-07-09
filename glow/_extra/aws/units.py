"""Enumerate a CONFIG cache's planted cells as the unit of AWS work.

resolve_cells turns a cache name into the cache's ordered planted cells plus
its shared fnc grid and leaf fnc -- the run bundle the driver pickles and ships
to S3 (see glow._extra.aws.driver). Which sources a cache spans is a CONFIG
property (its data grid), not a driver knob; resolve_cells runs whatever the
cache declares. It is driver-side only: a worker runs whatever bundle it is
handed and never calls this, so the two cannot disagree on the cell list. The
enumeration is deterministic -- itertools.product over fixed ranges, seeded RNG
for the HCP feature subset (see glow._extra.benchmark.config) -- which keeps
cell ordering reproducible across runs, so warm-cache hits line up.

A unit is one planted cell (a data cell crossed with one effect cell; see
results.planted_cells), not a whole data cell: a worker builds that cell's
clean exp once, plants the single effect, and fits every recipe in the fnc
grid on it. The data build repeats across the effect siblings that now live on
separate workers (cheap, and it keeps warm-resume and OOM retry per-effect),
rather than one worker owning a data cell's whole effect x analysis subtree.
"""

from glow._extra.benchmark.config import CONFIG
from glow._extra.benchmark.results import planted_cells


def resolve_cells(name: str):
    """Return the planted cells of CONFIG cache name plus its fnc grid / fnc.

    Crosses CONFIG[name]'s data grid with its effect grid into the flat planted
    cells (results.planted_cells) -- whatever sources the cache declares, since
    sources are a CONFIG property, not a driver knob. The cross is
    deterministically ordered (see the module docstring), so the returned cell
    list is reproducible across runs -- the driver ships it in the run bundle
    (see glow._extra.aws.driver).

    Args:
        name (str): a CONFIG cache name (e.g. 'sweep_llr').

    Returns:
        cells (list[tuple[dict, dict | None]]): (kwargs_data, kwargs_effect)
            pairs, in CONFIG order.
        kwargs_fnc_list (list[dict]): the cache's leaf-fnc kwargs grid.
        fnc (Callable): the cache's leaf measurement (e.g. run_ana).

    Raises:
        KeyError: name is not a CONFIG cache.
    """
    _, _, kwargs_fnc_list, fnc = CONFIG[name]
    return planted_cells(name), kwargs_fnc_list, fnc
