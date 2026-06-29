"""AWS Batch worker: run one benchmark data cell on the shared S3 state.

Invoked inside the container (the image ENTRYPOINT) as

    python -m glow._extra.aws.worker s3://bucket/.../runs/<id>/manifest.json

The manifest is a tiny JSON describing the run -- the CONFIG cache name, the
data sources to keep, the cell indices this array job is running, and the S3
prefix / region for the shared state. The worker:

  1. reads AWS_BATCH_JOB_ARRAY_INDEX, looks up its cell index in the manifest,
     and rebuilds that data cell from CONFIG (resolve_cells -- a pure function
     of the catalogue, so the cell matches what the driver enumerated; nothing
     per-cell is shipped, and no run function is pickled);
  2. pulls the shared records + run_ana cache from S3 so any fit a prior
     attempt or another run already computed is a cache hit (warm resume);
  3. runs the cell's whole effect x analysis subtree (_run_data_cell -- build
     the clean exp once, plant each effect, fit each recipe), while a
     background thread ships finished records / cache entries up every minute;
  4. flushes the uploader on exit.

Exits non-zero on any exception so AWS Batch marks the child FAILED -- the
driver then retries it (resuming from the synced cache) and escalates an
OOM-killed cell to the next memory tier (see glow._extra.aws.driver).
"""

import json
import os
import signal
import sys

import boto3

from . import s3, sync
from .units import resolve_cells


def _load_manifest(s3_client, bucket: str, key: str) -> dict:
    """Download and parse the run manifest JSON from S3.

    Returns:
        the manifest dict: {config_name, sources, cell_indices, s3_prefix}.
    """
    body = s3_client.get_object(Bucket=bucket, Key=key)['Body'].read()
    return json.loads(body)


def main(manifest_uri: str) -> None:
    """Run this array child's data cell and ship its results to S3.

    Args:
        manifest_uri (str): s3:// URI of the run's manifest.json.
    """
    bucket, manifest_key = s3.parse_s3_uri(manifest_uri)
    manifest = _load_manifest(boto3.client('s3'), bucket, manifest_key)

    config_name = manifest['config_name']
    sources = tuple(manifest['sources'])
    prefix = manifest['s3_prefix']
    region = manifest.get('region')
    s3_client = boto3.client('s3', region_name=region) if region \
        else boto3.client('s3')

    idx = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', '0'))
    cell_idx = manifest['cell_indices'][idx]

    data_cells, kwargs_effect_list, kwargs_fnc_list, fnc = resolve_cells(
        config_name, sources)
    kwargs_data = data_cells[cell_idx]
    print(f'[worker] {config_name} array_idx={idx} cell_idx={cell_idx} '
          f'source={kwargs_data.get("source")}', flush=True)

    # warm the local state: anything a prior attempt / run already computed is
    # then a cache hit (the records carry it; run_ana hits skip the ~450 s fit)
    pairs = sync.wgn_sync_pairs(prefix)
    for local_dir, key_prefix in pairs:
        s3.download_prefix(s3_client, bucket, key_prefix, local_dir)

    # run the cell while a background thread ships finished work up; the GIL is
    # released inside numpy, so sweeps proceed even mid-fit (see s3 module)
    uploader = s3.BackgroundUploader(s3_client, bucket, pairs).start()

    # on a Spot reclaim Batch sends SIGTERM ~2 min ahead; turn it into a clean
    # exit so the finally below flushes before the process dies
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    try:
        # imported here so --help / a missing cell errors before the heavy
        # benchmark import chain (numpy / glow) is paid
        from glow._extra.benchmark.driver import _run_data_cell
        _run_data_cell(kwargs_data, kwargs_effect_list, kwargs_fnc_list, fnc,
                       config_name=config_name)
    finally:
        n = uploader.stop()
        print(f'[worker] final flush uploaded {n} file(s)', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python -m glow._extra.aws.worker <manifest_s3_uri>',
              file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
