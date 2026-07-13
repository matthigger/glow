"""Archive the local joblib cache + records and clear the shared AWS S3 state.

Run it as:

    python -m glow._extra.benchmark.mv_cache [--no-aws] [--config PATH]

Cache-invalidation housekeeping: when a recorded spec changes, old joblib
entries key under dead hashes and just accrue on disk (see the benchmark
recorder / driver). This moves the current cache and its sibling records
(file.get_path_cache / get_path_records) into a dated archive under an "old"
folder and leaves fresh empty dirs in their place, so the next run starts cold
while the archived state stays recoverable. Cache and records move together
under one shared stamp so they stay in lock step (a record is keyed by the same
args hash joblib files its result under). It then deletes the shared S3 cache/
and records/ prefixes (glow._extra.aws.infra) so AWS workers stop warm-resuming
from a stale cache and stop building results off stale records too.

The dirs are commonly symlinks into Dropbox (see the storage layout); the move
follows them, archiving and recreating the real directory so each symlink stays
valid.
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

from .file import get_path_cache, get_path_records


def _free_stamp(items, base: str) -> str:
    """Return base, or a time-suffixed variant no item's archive dest uses yet.

    Keeps cache and records on one shared stamp: if either old/<label>-<base>
    already exists (a second archive the same day), both get the same
    seconds-resolution suffix so their archive names still match.

    Args:
        items (list): (label, real_dir) pairs to check.
        base (str): the preferred stamp (YYYY-MM-DD).

    Returns:
        stamp (str): base if free everywhere, else base with a -HHMMSS suffix.
    """
    def taken(stamp: str) -> bool:
        return any((real.parent / 'old' / f'{label}-{stamp}').exists()
                   for label, real in items)

    if not taken(base):
        return base
    return datetime.now().strftime('%Y-%m-%d-%H%M%S')


def _archive_one(real: Path, label: str, stamp: str) -> Optional[Path]:
    """Move one resolved dir into <parent>/old/<label>-<stamp>, then recreate.

    Args:
        real (Path): the resolved (symlink-followed) source directory.
        label (str): archive name stem, 'cache' or 'records'.
        stamp (str): the shared date stamp.

    Returns:
        dest (Path | None): the archive dir, or None if real was already empty.
    """
    if not any(real.iterdir()):
        print(f'[mv_cache] {label} already empty: {real}')
        return None
    dest = real.parent / 'old' / f'{label}-{stamp}'
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(real), str(dest))
    real.mkdir(parents=True, exist_ok=True)
    print(f'[mv_cache] archived {label} -> {dest}')
    return dest


def archive_caches(
        stamp: Optional[str] = None) -> Tuple[Optional[Path], Optional[Path]]:
    """Move the local cache and records into a dated old/ archive, in lock step.

    Resolves get_path_cache / get_path_records to their real directories
    (commonly symlinks into Dropbox), moves each to <parent>/old/<label>-<stamp>
    under one shared stamp, and remakes empty dirs so the next run writes into a
    clean tree and the symlinks, if any, stay valid.

    Args:
        stamp (str | None): archive date stamp; defaults to today (YYYY-MM-DD),
            bumped with a time suffix if that day's archive already exists.

    Returns:
        (cache_dest, records_dest): the two archive dirs, each None if that
            source was already empty (nothing moved).
    """
    cache_real = get_path_cache().resolve()
    records_real = get_path_records().resolve()
    stamp = _free_stamp([('cache', cache_real), ('records', records_real)],
                        stamp or datetime.now().strftime('%Y-%m-%d'))
    cache_dest = _archive_one(cache_real, 'cache', stamp)
    records_dest = _archive_one(records_real, 'records', stamp)
    return cache_dest, records_dest


def clear_aws_state(config_path: Optional[str] = None) -> None:
    """Delete the shared S3 cache/ and records/ prefixes, best effort.

    Routes through the infra clear_storage path (cache + records prefixes, in
    lock step with the local archive) so AWS workers stop warm-resuming from a
    stale synced cache and stop rebuilding results off stale records. runs/ (the
    per-run manifests) is left alone. A missing AWS config or any boto3 error is
    reported and swallowed: the local archive is the primary action, and AWS may
    not be configured on every machine.

    Args:
        config_path (str | None): AWSConfig JSON path; defaults to the per-user
            config location (aws.config.DEFAULT_CONFIG_PATH).
    """
    from glow._extra.aws import infra
    from glow._extra.aws.config import DEFAULT_CONFIG_PATH, AWSConfig

    path = config_path or DEFAULT_CONFIG_PATH
    try:
        cfg = AWSConfig.from_file(path)
    except FileNotFoundError:
        print(f'[mv_cache] no AWS config at {path}; skipping S3 clear')
        return

    args = SimpleNamespace(runs=False, records=True, cache=True, yes=True)
    # Swallow any boto3 failure (absent credentials, network) so a machine
    # without AWS still completes the local archive.
    try:
        infra.cmd_clear_storage(args, cfg)
    except Exception as exc:
        print(f'[mv_cache] S3 clear failed ({exc!r}); skipping')


def main(argv=None) -> None:
    """Archive the local cache + records and (unless --no-aws) clear S3."""
    parser = argparse.ArgumentParser(
        prog='python -m glow._extra.benchmark.mv_cache',
        description='Archive the local joblib cache and records to a dated '
                    'old/ folder and clear the shared AWS S3 cache + records.')
    parser.add_argument('--no-aws', action='store_true',
                        help='archive the local state only; leave S3 alone')
    parser.add_argument('--config', default=None,
                        help='AWSConfig JSON path (default: per-user config)')
    args = parser.parse_args(argv)

    archive_caches()
    if not args.no_aws:
        clear_aws_state(args.config)


if __name__ == '__main__':
    main()
