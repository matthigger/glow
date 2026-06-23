"""CLI entry point for paper benchmarks.

Running python -m glow.benchmark.paper resolves cache labels from
config.CACHE_BY_LABEL and drives each selected entry through
driver_local; with --aws every selected cache is handed to
driver_aws_multi at once, which uploads and submits them all up front
and polls them concurrently. Each catalogue entry is a
(TrialCache, run_fnc) pair; the run_fnc is already bound to its
analysis recipe (see paper/config.py), so the CLI doesn't need to know
whether a cache is a run_ana or run_mancova job.

Usage:
    python -m glow.benchmark.paper                       # everything (local)
    python -m glow.benchmark.paper null_*                # glob
    python -m glow.benchmark.paper -j 4 vba_hcp_famd     # parallel local
    python -m glow.benchmark.paper --aws 'sweep_*'       # AWS Batch
"""
import argparse
from fnmatch import fnmatch

from .driver import driver_paper


def resolve_labels(patterns) -> list:
    """Expand a list of literal labels / fnmatch patterns into entries.

    Empty input means "everything". Unknown literal labels raise; an
    fnmatch pattern that matches nothing also raises (so typos surface).

    Args:
        patterns (list): literal cache labels and/or fnmatch patterns;
            empty selects every catalogue entry

    Returns:
        out (list): (label, (cache, run_fnc)) pairs, de-duplicated and in
            first-seen order

    Raises:
        ValueError: a literal label is unknown, or a pattern matches nothing
    """
    # imported lazily: building the catalogue constructs every TrialCache
    # (touching the results dir on disk), which --help must not require
    from .config import CACHE_BY_LABEL

    if not patterns:
        return list(CACHE_BY_LABEL.items())

    out, seen = [], set()
    for pattern in patterns:
        if pattern in CACHE_BY_LABEL:
            matches = [pattern]
        else:
            matches = [k for k in CACHE_BY_LABEL if fnmatch(k, pattern)]
            if not matches:
                raise ValueError(f'no cache labels match: {pattern}')
        for k in matches:
            if k not in seen:
                seen.add(k)
                out.append((k, CACHE_BY_LABEL[k]))
    return out


def run(labels=None, n_jobs: int = 1, verbose: bool = True,
        aws: bool = False, aws_config_path: str | None = None) -> None:
    """Run every selected cache through driver_local (or driver_aws).

    Args:
        labels (list | None): cache labels or fnmatch patterns; None selects all
        n_jobs (int): parallel worker count (local driver only)
        verbose (bool): print per-cache headers and progress
        aws (bool): dispatch to AWS Batch via glow.aws.driver_aws instead of local
        aws_config_path (str | None): path to the AWSConfig JSON (aws only);
            None loads the default user-config location and, when verbose,
            announces where it loaded the config from
    """
    entries = resolve_labels(labels or [])

    if aws:
        from glow.aws import AWSConfig, driver_aws_multi
        from glow.aws.config import DEFAULT_CONFIG_PATH
        if aws_config_path is None:
            aws_config_path = DEFAULT_CONFIG_PATH
            if verbose:
                print(f'loading AWS config from default location: '
                      f'{aws_config_path}')
        aws_cfg = AWSConfig.from_file(aws_config_path)
        jobs = [(label, cache, run_fnc) for label, (cache, run_fnc) in entries]
        driver_aws_multi(jobs, aws_cfg, verbose=verbose)
        return

    for label, (cache, run_fnc) in entries:
        if verbose:
            print(f'\n=== {label} ({cache.recorder.folder}) ===')
        driver_paper(cache, run_fnc, n_jobs=n_jobs, verbose=verbose)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the paper-benchmark CLI arguments.

    Args:
        argv (list | None): argument list to parse; None reads sys.argv

    Returns:
        the parsed argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Run paper benchmarks.')
    parser.add_argument('labels', nargs='*',
                        help='cache labels or fnmatch patterns (default: all)')
    parser.add_argument('-j', '--n-jobs', type=int, default=1,
                        help='parallel worker count (default: 1, local only)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='suppress per-cache headers and the tqdm bar')
    parser.add_argument('--aws', action='store_true',
                        help='run on AWS Batch via glow.aws.driver_aws')
    from glow.aws.config import DEFAULT_CONFIG_PATH
    parser.add_argument('--aws-config', default=None,
                        help=f'path to AWSConfig JSON (default: '
                             f'{DEFAULT_CONFIG_PATH})')
    return parser.parse_args(argv)


if __name__ == '__main__':
    # prompt / load hcp data if need be
    from glow.benchmark.hcp import ensure_hcp_data
    ensure_hcp_data()

    args = parse_args()
    run(labels=args.labels, n_jobs=args.n_jobs, verbose=not args.quiet,
        aws=args.aws, aws_config_path=args.aws_config)
