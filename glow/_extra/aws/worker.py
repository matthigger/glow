"""AWS Batch worker: run one benchmark planted cell on the shared S3 state.

Invoked inside the container (the image ENTRYPOINT) as

    python -m glow._extra.aws.worker s3://bucket/.../runs/<id>/manifest.json

The manifest is a tiny JSON pointing at the run bundle -- the bundle's S3 key,
the shared prefix / region for the shared state, and the cache label. The
worker:

  1. downloads + unpickles the run bundle the driver shipped (the resolved
     planted cells + shared fnc grid + leaf-fnc reference + cache label) and
     takes its AWS_BATCH_JOB_ARRAY_INDEX-th cell -- a (kwargs_data,
     kwargs_effect) pair -- with nothing looked up in CONFIG, so the cell list
     cannot drift from what the driver shipped (a config edit ships at submit
     time; only a glow code change still needs an image rebuild);
  2. if the cell is HCP, pulls just its features from the staged npy bundle
     (mask / affine / meta + its hcp_feats arrays) into hcp.bundle_dir(), so
     data_factory_hcp builds from the bundle with no niftis and no DUA prompt
     (WGN cells skip this; see glow._extra.aws.sync.hcp_bundle_keys);
  3. restores this cell's Spot-resume checkpoint if a prior attempt left one
     (glow._extra.aws.sync.checkpoint_dirs), so recipes it already finished are
     cache hits;
  4. runs the cell (_run_data_cell on the one effect -- build the clean exp
     once, plant that effect, fit each recipe) start to finish, building
     everything else it needs locally, while a background thread ships each
     finished record up every minute;
  5. on a Spot reclaim (SIGTERM ~2 min ahead) tars its partial progress to the
     checkpoint object; on clean completion drops the checkpoint -- best
     effort, since a failed drop costs one orphan object and must not fail a
     cell whose records are already shipped -- and flushes the uploader.

The worker pulls no shared cache: it runs one whole cell, so nothing another
worker computed can help it, and it uploads only the records its cell produces
(the driver pulls those home; see glow._extra.aws.sync). Its one
resume path is the per-cell checkpoint (steps 3 and 5), which restores just
this cell's partial progress, so a long cell reclaimed mid-run continues rather
than recomputing from cold. Exits non-zero on any exception so AWS Batch marks
the child FAILED -- the driver retries it and escalates an OOM-killed cell to
the next memory tier (see glow._extra.aws.driver).
"""

import json
import os
import pickle
import signal
import sys

import boto3
import joblib

from . import s3, sync
from .bundle import fnc_from_ref


def _load_manifest(s3_client, bucket: str, key: str) -> dict:
    """Download and parse the run manifest JSON from S3.

    Returns:
        the manifest dict: {config_name, bundle_key, s3_prefix, region,
        n_cells}.
    """
    body = s3_client.get_object(Bucket=bucket, Key=key)['Body'].read()
    return json.loads(body)


def main(manifest_uri: str) -> None:
    """Run this array child's data cell and ship its results to S3.

    Args:
        manifest_uri (str): s3:// URI of the run's manifest.json.
    """
    # turn on the recorder's per-record write log (silent locally); paired with
    # s3._log_upload's [upload] line it shows, in this child's CloudWatch stream,
    # every record written vs shipped -- so a Spot-killed attempt reveals what
    # it lost (see glow._extra.benchmark.recorder._store).
    os.environ['GLOW_RECORD_LOG'] = '1'

    bucket, manifest_key = s3.parse_s3_uri(manifest_uri)
    manifest = _load_manifest(boto3.client('s3'), bucket, manifest_key)

    prefix = manifest['s3_prefix']
    region = manifest.get('region')
    s3_client = boto3.client('s3', region_name=region) if region \
        else boto3.client('s3')

    # download + unpickle the run bundle the driver shipped; the worker runs
    # whatever it is handed, with no CONFIG lookup of its own (unpickling the
    # fnc / Analysis objects resolves their classes from the installed glow).
    # pickle is safe here: the bundle is written by this operator's own driver
    # into the operator's private S3 bucket -- a closed trust boundary, not
    # untrusted input.
    body = s3_client.get_object(
        Bucket=bucket, Key=manifest['bundle_key'])['Body'].read()
    cells, kwargs_fnc_list, fnc_ref, config_name = pickle.loads(body)
    # fnc ships as an import reference; resolving it here binds it to this
    # worker's local MEMORY / RECORDER, see bundle module
    fnc = fnc_from_ref(fnc_ref)

    idx = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', '0'))
    kwargs_data, kwargs_effect = cells[idx]
    print(f'[worker] {config_name} array_idx={idx} '
          f'source={kwargs_data.get("source")}', flush=True)

    # an HCP cell builds from the staged npy bundle, so pull just the files it
    # needs first (the shared mask / affine / meta + its hcp_feats arrays) --
    # then data_factory_hcp builds from the bundle with no niftis and no DUA
    # prompt. Pulled one way; never produced here, so never pushed.
    if kwargs_data.get('source') == 'hcp':
        bundle_pairs = sync.hcp_bundle_keys(prefix, kwargs_data['hcp_feats'])
        n_hcp = s3.download_each(s3_client, bucket, bundle_pairs)
        print(f'[worker] HCP bundle: {n_hcp} file(s) pulled for feats '
              f'{list(kwargs_data["hcp_feats"])}', flush=True)

    # per-cell Spot-resume checkpoint: a reclaimed attempt tars its partial
    # progress to one S3 object (on the SIGTERM below) and the retry restores it
    # here, so only the recipes the reclaim cut short recompute. Keyed by the
    # cell's content hash so it survives a driver tier-escalation resubmit (a
    # fresh array index for the same cell).
    cell_hash = joblib.hash((kwargs_data, kwargs_effect))
    ckpt_key = sync.checkpoint_key(prefix, cell_hash)
    ckpt_dirs = sync.checkpoint_dirs(fnc)
    if s3.read_checkpoint(s3_client, bucket, ckpt_key, ckpt_dirs):
        print('[worker] resumed from Spot checkpoint', flush=True)

    # ship only the records this cell produces (the driver pulls them home);
    # the worker pulls no shared state, since one whole cell runs here
    # and no other worker's cache helps it. A background thread sweeps every
    # minute; the GIL is released inside numpy, so sweeps proceed even mid-fit.
    pairs = [sync.records_pair(prefix)]
    uploader = s3.BackgroundUploader(s3_client, bucket, pairs).start()

    # on a Spot reclaim Batch sends SIGTERM ~2 min ahead; ship the finished
    # records and tar this cell's partial progress to the checkpoint, then exit
    # 143 so Batch retries it and the retry resumes from that checkpoint.
    def _on_spot_reclaim(*_):
        try:
            uploader.flush()
            s3.write_checkpoint(s3_client, bucket, ckpt_key, ckpt_dirs)
            print('[worker] Spot checkpoint written', flush=True)
        except Exception as exc:
            print(f'[worker] Spot checkpoint failed: {exc}', flush=True)
        sys.exit(143)

    signal.signal(signal.SIGTERM, _on_spot_reclaim)

    try:
        # imported here so --help / a missing cell errors before the heavy
        # benchmark import chain (numpy / glow) is paid
        from glow._extra.benchmark.driver import _run_data_cell
        _run_data_cell(kwargs_data, [kwargs_effect], kwargs_fnc_list, fnc)
        # cell finished cleanly: its records are shipped, so the checkpoint is
        # obsolete -- drop it rather than leave an orphan for the retry path.
        # A failure here costs only that orphan, so it must not propagate: the
        # cell's work is done and shipped, and raising would exit non-zero and
        # hand Batch a finished cell to retry.
        try:
            s3.delete_checkpoint(s3_client, bucket, ckpt_key)
        except Exception as exc:
            print(f'[worker] checkpoint cleanup failed: {exc}', flush=True)
    finally:
        n = uploader.stop()
        print(f'[worker] final flush uploaded {n} file(s)', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python -m glow._extra.aws.worker <manifest_s3_uri>',
              file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
