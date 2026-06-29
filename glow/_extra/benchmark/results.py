"""Slice the shared provenance frame into one tidy CSV per CONFIG cache.

The driver writes every run_ana leaf into one shared RECORDER
(data.RECORDER): RECORDER.flatten_to_df yields one row per leaf, each
carrying the data -> (plant ->) score chain that produced it (see
glow._extra.benchmark.recorder / driver). That frame is config-agnostic --
it holds every cell ever run, across every figure -- so this module
re-attaches the CONFIG catalogue: for one cache name it selects that
cache's leaves and walks each up to its ancestors, then writes name.csv.

Membership is read straight off the records: the driver, run under
with RECORDER.collecting(name):, tags each leaf's record with the cache it
ran under (the configs list). So this module neither re-walks the grid nor
rebuilds an experiment -- it just reads the tags and walks backwards
through the recorded provenance DAG. A leaf whose ancestor records are
missing (a dead-end -- a build never recorded, or a partial run) still
yields its row, the absent ancestor columns NaN; a cell whose run_ana
never ran has no leaf and is simply absent. Nothing is computed to assemble
results.

The tag, not a single stamped column, is what carries membership because
the sweeps share cells. The catalogue anchors each structural sweep on the
same baseline cell (the grid-midpoint effect_llr at the default extent,
b=1, num_img=100), so the moderate baseline run is the very same run_ana
call -- one cache id, one cached result, one record -- in sweep_llr,
sweep_b, sweep_extent and (WGN) sweep_nimg. That one record therefore
carries several cache names in its configs list and is selected by each,
which a single which-cache field could not express (the last writer would
clobber the rest). Tagging above the cache (so a cache hit on a shared cell
still records the later cache; see recorder.tag_call) is what makes the
list accumulate every cache that asked for the cell, with no grid recompute
at read time.
"""
from pathlib import Path

from .config import CONFIG
from .data import RECORDER
from .file import get_path_result

# the record list field the driver tags each leaf with (one entry per CONFIG
# cache it was run under; see recorder.Recorder.tag_call / driver).
CONFIG_TAG_FIELD = 'configs'


def config_leaf_keys(name: str) -> list:
    """Return record keys of leaves tagged as in CONFIG cache name.

    The driver tags each run_ana leaf with the cache it ran under
    (RECORDER.collecting -> the record's configs list), so membership is read
    straight off the records -- no grid is re-walked, no experiment rebuilt.
    A leaf shared by several caches (the sweeps' common baseline cell)
    carries every one of their names, so it is selected by each. Reads the
    in-memory records; call RECORDER.load() first to fold in what a parallel
    run left on disk.

    Args:
        name (str): a CONFIG cache name (e.g. 'sweep_llr').

    Returns:
        list[str]: the run_ana record keys tagged name (empty if none ran).
    """
    return [key for key, rec in RECORDER.records.items()
            if name in rec.get(CONFIG_TAG_FIELD, [])]


def config_results_df(name: str):
    """Walk the shared provenance frame up from one cache's tagged leaves.

    Selects the leaves tagged name (config_leaf_keys) and flattens each into
    a row carrying its data -> (plant ->) score chain (flatten_to_df seeded
    from those leaves). A row a sweep shares with another sweep is selected
    by both, since the shared record carries both names. Loads the recorder
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
    """Write one name.csv per CONFIG cache with tagged leaves on disk.

    Loads the shared recorder once, then for each cache selects its tagged
    leaves and flattens them into the per-figure results frame the plots
    consume, writing each non-empty slice to out_dir/<name>.csv. A cache
    with no tagged leaves yet is skipped (no empty file).

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
    for name in (names if names is not None else CONFIG):
        sub = RECORDER.flatten_to_df(leaf_keys=config_leaf_keys(name))
        if sub.empty:
            continue
        path = out_dir / f'{name}.csv'
        sub.to_csv(path, index=False)
        written[name] = path
    return written
