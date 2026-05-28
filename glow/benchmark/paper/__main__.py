"""CLI entry point for paper benchmarks.

Resolves cache labels from ``config.CACHE_BY_LABEL`` and drives each
selected entry through ``driver_local`` (or ``driver_aws`` with
``--aws``).  Each catalogue entry is a ``(TrialCache, run_fnc)``
pair; the ``run_fnc`` is already bound to its analysis recipe (see
``paper/config.py``), so the CLI doesn't need to know whether a cache
is a ``run_ana`` or ``run_mancova`` job.

Usage:
    python -m glow.benchmark.paper                       # everything (local)
    python -m glow.benchmark.paper null_*                # glob
    python -m glow.benchmark.paper -j 4 vba_hcp_famd     # parallel local
    python -m glow.benchmark.paper --aws 'sweep_*'       # AWS Batch
"""
import argparse
from fnmatch import fnmatch

from glow.benchmark.driver import driver_local

from .config import CACHE_BY_LABEL


def resolve_labels(patterns):
    """Expand a list of literal labels / fnmatch patterns into entries.

    Empty input means "everything".  Unknown literal labels raise; an
    fnmatch pattern that matches nothing also raises (so typos surface).
    """
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


def run(labels=None, n_jobs=1, verbose=True,
        aws=False, aws_config_path='.glow_aws_config'):
    """Run every selected cache through ``driver_local`` (or ``driver_aws``)."""
    aws_cfg = None
    driver_aws = None
    if aws:
        from glow.aws import AWSConfig, driver_aws as _driver_aws
        aws_cfg = AWSConfig.from_file(aws_config_path)
        driver_aws = _driver_aws

    for label, (cache, run_fnc) in resolve_labels(labels or []):
        if verbose:
            print(f'\n=== {label} ({cache.folder}) ===')
        if aws:
            driver_aws(cache, run_fnc, aws_cfg, verbose=verbose)
        else:
            driver_local(cache, run_fnc, n_jobs=n_jobs, verbose=verbose)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run paper benchmarks.')
    parser.add_argument('labels', nargs='*',
                        help='cache labels or fnmatch patterns (default: all)')
    parser.add_argument('-j', '--n-jobs', type=int, default=1,
                        help='parallel worker count (default: 1, local only)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='suppress per-cache headers and the tqdm bar')
    parser.add_argument('--aws', action='store_true',
                        help='run on AWS Batch via glow.aws.driver_aws')
    parser.add_argument('--aws-config', default='.glow_aws_config',
                        help='path to AWSConfig JSON (default: .glow_aws_config)')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    run(labels=args.labels, n_jobs=args.n_jobs, verbose=not args.quiet,
        aws=args.aws, aws_config_path=args.aws_config)
