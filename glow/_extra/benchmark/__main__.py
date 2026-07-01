"""CLI entry point for the paper benchmarks.

Running python -m glow._extra.benchmark resolves cache names from
config.CONFIG and drives each selected one through driver.drive, wrapped in
RECORDER.collecting(name) so every leaf is tagged with the cache it belongs
to (see driver / results). After the sweep it writes one <name>.csv per
cache from the shared provenance records (results.write_config_csvs).

--csv-only skips the sweep and rebuilds those CSVs from the records already
on disk -- the path to take after editing config.py / results.py when the
records are still good and no new experiments are needed. (It reads the
configs tags baked into the records at run time; it does not re-tag
membership, which only a real run does.)

--aws runs the sweep on AWS Batch instead of locally: it hands the same
resolved cache names to glow._extra.aws.drive_aws, which submits each cache's
data cells as a Batch array job and writes the same per-config CSVs from the
shared records when they drain (see glow._extra.aws). This is the one CLI for a
benchmark run, local or cloud; provisioning the AWS resources is a separate
concern (python -m glow._extra.aws -- bootstrap / setup / status / ...).

Usage:
    python -m glow._extra.benchmark                    # everything (local)
    python -m glow._extra.benchmark 'sweep_*'          # glob
    python -m glow._extra.benchmark -j 4 sweep_llr     # parallel local
    python -m glow._extra.benchmark --aws sweep_llr    # run on AWS Batch
    python -m glow._extra.benchmark --csv-only         # rebuild CSVs only
    python -m glow._extra.benchmark --list             # list cache names
"""
import argparse
from fnmatch import fnmatch


def resolve_names(patterns) -> list:
    """Expand literal cache names / fnmatch patterns into CONFIG names.

    Empty input means "everything". An unknown literal name raises, and a
    pattern that matches nothing also raises, so a typo surfaces rather than
    silently selecting no caches.

    Args:
        patterns (list): literal CONFIG names and/or fnmatch patterns; empty
            selects every catalogue entry.

    Returns:
        out (list): the resolved cache names, de-duplicated and in first-seen
            order.

    Raises:
        ValueError: a literal name is unknown, or a pattern matches nothing.
    """
    # imported lazily so --help / --list construct nothing they don't need
    from .config import CONFIG

    if not patterns:
        return list(CONFIG)

    out, seen = [], set()
    for pattern in patterns:
        if pattern in CONFIG:
            matches = [pattern]
        else:
            matches = [k for k in CONFIG if fnmatch(k, pattern)]
            if not matches:
                raise ValueError(f'no cache names match: {pattern}')
        for k in matches:
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def _report_csvs(written: dict) -> None:
    """Print the written per-config CSV paths, or a note that there were none.

    Args:
        written (dict): {cache name: csv path}, as write_config_csvs returns.
    """
    if not written:
        print('\nno CSVs written (no selected cache has records on disk yet)')
        return
    print('\nwrote per-config CSVs:')
    for name, path in written.items():
        print(f'  {name}: {path}')


def run(names=None, n_jobs: int = 1, verbose: bool = True,
        write_csv: bool = True, csv_only: bool = False, out_dir=None,
        aws: bool = False, aws_config_path=None) -> dict:
    """Drive the selected CONFIG caches, then write their per-config CSVs.

    For each resolved cache name, runs drive(*CONFIG[name], n_jobs=n_jobs)
    inside RECORDER.collecting(name) so every leaf is tagged with the cache it
    belongs to, then writes one <name>.csv per cache from the shared records
    (results.write_config_csvs). csv_only skips the sweep and just rebuilds
    those CSVs from the records already on disk -- the path to take after
    editing config.py / results.py when no new experiments are needed.

    aws runs the sweep on AWS Batch instead of locally: the resolved names are
    handed to glow._extra.aws.drive_aws, which submits each cache's cells as a
    Batch array job and writes the same CSVs from the shared records when they
    drain. The local HCP dataset is not loaded in this mode (the workers own
    the data), and n_jobs does not apply (the Batch array is the parallelism).

    The HCP reference dataset is ensured once up front (idempotent / cached)
    for the HCP-backed caches in a local run, except under csv_only / aws,
    which run no local experiments.

    Args:
        names (list | None): cache names or fnmatch patterns; None selects all.
        n_jobs (int): joblib worker count for a local sweep (1 = serial; -1 =
            all cores). Ignored when csv_only or aws is set.
        verbose (bool): print per-cache headers, the drive() progress bar,
            and the CSV-written summary.
        write_csv (bool): write the per-config CSVs after the sweep. Implied
            (and forced) when csv_only is set.
        csv_only (bool): skip the sweep; rebuild the CSVs from the records on
            disk. The HCP data is not loaded in this mode.
        out_dir (str | pathlib.Path | None): CSV destination; None is glow's
            per-user results dir (file.get_path_result).
        aws (bool): run the sweep on AWS Batch (drive_aws) rather than locally.
        aws_config_path (str | None): AWSConfig JSON path for aws; None uses
            the per-user default (config.AWSConfig.from_file).

    Returns:
        written (dict): {cache name: csv path} for the caches that had rows
            (empty when write_csv is False, or when no selected cache has run).
    """
    from .results import write_config_csvs

    resolved = resolve_names(names or [])

    if csv_only:
        written = write_config_csvs(out_dir=out_dir, names=resolved)
        if verbose:
            _report_csvs(written)
        return written

    if aws:
        from glow._extra.aws import AWSConfig, drive_aws
        aws_config = (AWSConfig.from_file(aws_config_path) if aws_config_path
                      else AWSConfig.from_file())
        return drive_aws(resolved, aws_config, write_csv=write_csv,
                         verbose=verbose, out_dir=out_dir)

    from .config import CONFIG
    from .data import RECORDER
    from .driver import drive
    from .hcp import ensure_hcp_data

    # most caches build on the HCP reference dataset; ensure it once (cached)
    ensure_hcp_data()

    for name in resolved:
        if verbose:
            print(f'\n=== {name} ({RECORDER.folder}) ===')
        # wrap drive so the cache name tags each leaf's record (driver reads
        # the active collecting() grouping); results then slices CSVs by tag
        with RECORDER.collecting(name):
            drive(*CONFIG[name], n_jobs=n_jobs, verbose=verbose)

    written = (write_config_csvs(out_dir=out_dir, names=resolved)
               if write_csv else {})
    if verbose:
        _report_csvs(written)
    return written


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the benchmark CLI arguments.

    Args:
        argv (list | None): argument list to parse; None reads sys.argv.

    Returns:
        the parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description='Run the paper benchmarks (local or AWS Batch).',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='*',
                        help='cache names or fnmatch patterns (default: all)')
    parser.add_argument('-j', '--n-jobs', type=int, default=1,
                        help='joblib worker count (1=serial; -1=all cores)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='suppress per-cache headers and the CSV summary')
    parser.add_argument('--out-dir', default=None,
                        help='CSV destination (default: glow per-user results '
                             'dir)')
    parser.add_argument('--aws', action='store_true',
                        help='run the sweep on AWS Batch (glow._extra.aws)')
    parser.add_argument('--aws-config', default=None,
                        help='AWSConfig JSON for --aws (default: per-user)')
    parser.add_argument('--list', action='store_true', dest='list_names',
                        help='print the catalogue cache names and exit')
    csv_group = parser.add_mutually_exclusive_group()
    csv_group.add_argument('--no-csv', action='store_true',
                           help='run the sweep but skip writing the CSVs')
    csv_group.add_argument('--csv-only', action='store_true',
                           help='skip the sweep; rebuild the CSVs from the '
                                'records on disk')
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Parse the CLI arguments and run the benchmark.

    Args:
        argv (list | None): argument list to parse; None reads sys.argv.
    """
    args = parse_args(argv)

    if args.list_names:
        from .config import CONFIG
        for name in CONFIG:
            print(name)
        return

    run(names=args.names, n_jobs=args.n_jobs, verbose=not args.quiet,
        write_csv=not args.no_csv, csv_only=args.csv_only,
        out_dir=args.out_dir, aws=args.aws,
        aws_config_path=args.aws_config)


if __name__ == '__main__':
    main()
