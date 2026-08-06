"""Aggregate the shared provenance records into one tidy CSV per CONFIG cache.

Aggregation is its own step, separate from running: a sweep (local or AWS)
only fills the shared records, and this module turns them into the per-figure
CSVs. Run it whenever the records or the catalogue change:

    python -m glow._extra.benchmark.make_csv                 # every cache
    python -m glow._extra.benchmark.make_csv 'sweep_*'       # glob
    python -m glow._extra.benchmark.make_csv null --out-dir /tmp/csv

Cache membership is recomputed from the current CONFIG at read time by walking
the recorded provenance DAG forward from each cache's declared cells
(results.config_leaf_keys), so a config edit is reflected on the next call with
nothing stale to prune. Reading the records is cheap, so re-exporting is cheap:
plot refreshes a cache's CSV on its way to the figures (write_config_csv).
"""
import argparse
from pathlib import Path

from . import config
from .data import RECORDER
from .file import get_path_result
from .results import config_leaf_keys


def _prepare_dir(out_dir) -> Path:
    """Return the CSV destination directory, created if missing.

    Args:
        out_dir (str | Path | None): destination; None is glow's per-user
            results dir (file.get_path_result).

    Returns:
        out_dir (Path): the existing destination directory.
    """
    out_dir = Path(out_dir) if out_dir is not None else get_path_result()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _dump(name: str, out_dir):
    """Flatten one cache's leaves to out_dir/name.csv and return the frame.

    Reads the records already in memory (the caller loads the recorder once),
    so a whole-catalogue export re-reads the on-disk records once rather than
    once per cache. A cache with no leaves yet writes no file.

    Args:
        name (str): a CONFIG cache name.
        out_dir (Path): an existing destination directory.

    Returns:
        pandas.DataFrame: the cache's rows (empty when no cell of it has run).
    """
    df = RECORDER.flatten_to_df(leaf_keys=config_leaf_keys(name))
    if not df.empty:
        df.to_csv(out_dir / f'{name}.csv', index=False)
    return df


def config_results_df(name: str):
    """Walk the shared provenance frame up from one cache's leaves.

    Selects the cache's leaves (results.config_leaf_keys) and flattens each
    into a row carrying its data -> (plant ->) score chain (flatten_to_df
    seeded from those leaves). A row a sweep shares with another sweep is
    selected by both, since the walk reaches the shared record from either
    cache. Loads the recorder first to fold in any records a parallel run /
    other workers left on disk. Empty when no cell of this cache has run.

    Args:
        name (str): a CONFIG cache name.

    Returns:
        pandas.DataFrame: this cache's rows (empty when none have run).
    """
    RECORDER.load()
    return RECORDER.flatten_to_df(leaf_keys=config_leaf_keys(name))


def write_config_csv(name: str, out_dir=None):
    """Write one cache's CSV from the records and return the frame written.

    The read-and-refresh entry point: a consumer that wants a cache's frame
    (plot) gets it and leaves the CSV up to date in passing, so the exported
    file never lags the figures drawn from the same records. Nothing is written
    for a cache with no leaves yet (no empty file).

    Args:
        name (str): a CONFIG cache name.
        out_dir (str | Path | None): destination directory; None is glow's
            per-user results dir (file.get_path_result).

    Returns:
        pandas.DataFrame: the cache's rows (empty when no cell of it has run).
    """
    RECORDER.load()
    return _dump(name, _prepare_dir(out_dir))


def write_config_csvs(out_dir=None, names=None) -> dict:
    """Write one name.csv per CONFIG cache with leaves on disk.

    Loads the shared recorder once, then for each cache selects its leaves
    (results.config_leaf_keys) and flattens them into the per-figure results
    frame the plots consume, writing each non-empty slice to
    out_dir/<name>.csv. A cache with no leaves yet is skipped (no empty file).

    Args:
        out_dir (str | Path | None): destination directory; None is glow's
            per-user results dir (file.get_path_result).
        names (iterable[str] | None): cache names to export; None exports
            every CONFIG entry.

    Returns:
        dict[str, Path]: the {cache name: written csv path} for caches that
            had rows (those skipped for being empty are omitted).
    """
    out_dir = _prepare_dir(out_dir)
    RECORDER.load()

    written = {}
    for name in (names if names is not None else config.CONFIG):
        if not _dump(name, out_dir).empty:
            written[name] = out_dir / f'{name}.csv'
    return written


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the make_csv CLI arguments.

    Args:
        argv (list | None): argument list to parse; None reads sys.argv.

    Returns:
        the parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description='Export the per-CONFIG benchmark CSVs from the records.')
    parser.add_argument('names', nargs='*',
                        help='cache names or fnmatch patterns (default: all)')
    parser.add_argument('--out-dir', default=None,
                        help='CSV destination (default: glow per-user results '
                             'dir)')
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Export the selected caches' CSVs and print what was written.

    Args:
        argv (list | None): argument list to parse; None reads sys.argv.
    """
    # the sweep CLI's resolver, imported here so importing this module for its
    # library API pulls in no CLI
    from .__main__ import resolve_names

    args = parse_args(argv)
    written = write_config_csvs(out_dir=args.out_dir,
                                names=resolve_names(args.names))
    if not written:
        print('no CSVs written (no selected cache has records on disk yet)')
        return
    print('wrote per-config CSVs:')
    for name, path in written.items():
        print(f'  {name}: {path}')


if __name__ == '__main__':
    main()
