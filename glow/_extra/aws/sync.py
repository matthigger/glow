"""What benchmark state to mirror to S3, as (local_dir, key_prefix) pairs.

s3.py is the generic file mover; this is the policy: which of the benchmark's
content-addressed trees to ship, and where each lands under the run's S3
prefix. The driver and worker both build their sync pairs here so the layout
agrees.

The pairs are keyed by stable, machine-independent S3 prefixes:

  - records     -> {prefix}/records           (the per-hash <hash>.json files)
  - a func cache -> {prefix}/cache/{func_id}   (one joblib function's entries)

A joblib cache entry lives at store_backend.location/func_id/<args-hash>/, and
func_id (e.g. 'glow/_extra/benchmark/run/run_ana') is the same on every
machine while the location is not -- so func_id is the cross-machine key, and
each side joins it onto its own local cache root.

WGN policy (the first milestone): sync the records (tiny; they carry the
provenance DAG the CSVs are built from) and the run_ana cache. run_ana is the
~450 s compute and caches as a KB score dict, so shipping it gives Spot-resume
and cross-run dedup for almost no traffic; the heavy WGN exp caches
(data_factory_wgn / effect_factory) are deliberately left out -- they rebuild
deterministically from a seed on the worker, cheaper than shipping tens of MB.
HCP will add the exp-cache pairs (an expensive nifti load to rebuild) and the
per-feature data staging in a later milestone.
"""

from pathlib import Path
from typing import List, Tuple

from glow._extra.benchmark.file import get_path_records
from glow._extra.benchmark.run import run_ana

from .config import s3_key

Pair = Tuple[Path, str]


def _func_cache_pair(memorized_fnc, prefix: str) -> Pair:
    """The (local func cache dir, S3 key prefix) for one joblib MemorizedFunc.

    Args:
        memorized_fnc: a joblib MemorizedFunc (a MEMORY.cache-decorated fn).
        prefix (str): the run's s3_prefix.

    Returns:
        (local_dir, key_prefix): the function's local cache directory and the
            cross-machine S3 prefix it mirrors to (under {prefix}/cache).
    """
    func_id = memorized_fnc.func_id
    local_dir = Path(memorized_fnc.store_backend.location) / func_id
    return local_dir, s3_key(prefix, 'cache', func_id)


def _records_pair(prefix: str) -> Pair:
    """The (local records dir, S3 key prefix) pair for the per-hash records."""
    return get_path_records(), s3_key(prefix, 'records')


def records_pair(prefix: str) -> Pair:
    """The records sync pair (the driver pulls it to build the CSVs)."""
    return _records_pair(prefix)


def wgn_sync_pairs(prefix: str) -> List[Pair]:
    """The (local_dir, key_prefix) pairs a WGN worker syncs both ways.

    Records plus the run_ana cache (see the module docstring). The worker
    pulls these at start (warm resume), pushes them while it computes, and
    flushes on exit; the driver pulls the records to assemble the CSVs.

    Args:
        prefix (str): the run's s3_prefix.

    Returns:
        list[Pair]: the directories to mirror, each with its S3 key prefix.
    """
    return [_records_pair(prefix), _func_cache_pair(run_ana, prefix)]
