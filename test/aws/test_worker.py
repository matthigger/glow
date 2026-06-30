"""worker.main: maps the array index to the right cell of the run bundle.

The heavy parts (S3 sync, the actual fit) are stubbed; the test pins the one
piece of worker logic that must be right -- AWS_BATCH_JOB_ARRAY_INDEX picking
the correct data cell out of the bundle the driver pickled and shipped.
"""

import json
import pickle
from unittest.mock import patch

import joblib

from glow._extra.aws import sync, worker
from glow._extra.aws.bundle import fnc_to_ref
from glow._extra.aws.units import resolve_cells
from test.aws.fakes import FakeS3


def _put_bundle(fake, bucket, prefix, run_id, bundle):
    """Pickle bundle to S3 and write a manifest pointing at it; return URI."""
    bundle_key = f'{prefix}/runs/{run_id}/bundle.pkl'
    fake.put_object(Bucket=bucket, Key=bundle_key, Body=pickle.dumps(bundle))
    manifest = {'config_name': bundle[-1], 'bundle_key': bundle_key,
                's3_prefix': prefix, 'region': 'us-east-1',
                'n_cells': len(bundle[0])}
    manifest_key = f'{prefix}/runs/{run_id}/manifest.json'
    fake.put_object(Bucket=bucket, Key=manifest_key,
                    Body=json.dumps(manifest).encode())
    return f's3://{bucket}/{manifest_key}'


def test_array_index_selects_cell(monkeypatch):
    bucket, prefix = 'bkt', 'glow'
    data_cells, eff, fnc_kw, fnc = resolve_cells('sweep_llr')
    # the driver ships a subset (here cells 5, 6, 7); child i runs the i-th
    bundle = ([data_cells[5], data_cells[6], data_cells[7]], eff, fnc_kw,
              fnc_to_ref(fnc), 'sweep_llr')
    fake = FakeS3()
    uri = _put_bundle(fake, bucket, prefix, 'run-x', bundle)

    # stub the sync (no real dirs / threads) and the fit (record its cell)
    monkeypatch.setattr('glow._extra.aws.sync.sync_pairs', lambda p: [])
    seen = {}
    monkeypatch.setattr('glow._extra.benchmark.driver._run_data_cell',
                        lambda kd, ke, kf, fnc, config_name=None:
                        seen.update(kwargs_data=kd, config_name=config_name))
    monkeypatch.setenv('AWS_BATCH_JOB_ARRAY_INDEX', '2')

    with patch('glow._extra.aws.worker.boto3.client', lambda *a, **k: fake):
        worker.main(uri)

    # array index 2 -> the 3rd shipped cell is the 7th WGN cell of sweep_llr.
    # Compare by joblib.hash (the cache key): the cells have no value __eq__,
    # and after the pickle round-trip the worker holds a fresh object -- what
    # must match a local run is its hash, not its identity.
    assert joblib.hash(seen['kwargs_data']) == joblib.hash(data_cells[7])
    assert seen['config_name'] == 'sweep_llr'


def test_hcp_cell_pulls_only_its_features(monkeypatch):
    # an HCP cell pulls just the bundle files for its hcp_feats (+ shared
    # mask/affine/meta), via download_each -- not the whole panel
    bucket, prefix = 'bkt', 'glow'
    all_cells, eff, fnc_kw, fnc = resolve_cells('smoke')
    hcp_cells = [c for c in all_cells if c['source'] == 'hcp']
    bundle = ([hcp_cells[0]], eff, fnc_kw, fnc_to_ref(fnc), 'smoke')
    fake = FakeS3()
    uri = _put_bundle(fake, bucket, prefix, 'run-h', bundle)

    monkeypatch.setattr('glow._extra.aws.sync.sync_pairs', lambda p: [])
    monkeypatch.setattr('glow._extra.benchmark.driver._run_data_cell',
                        lambda *a, **k: None)
    pulled = {}
    monkeypatch.setattr('glow._extra.aws.s3.download_each',
                        lambda c, b, pairs: pulled.update(pairs=list(pairs)))
    monkeypatch.setenv('AWS_BATCH_JOB_ARRAY_INDEX', '0')

    with patch('glow._extra.aws.worker.boto3.client', lambda *a, **k: fake):
        worker.main(uri)

    feats = hcp_cells[0]['hcp_feats']
    assert pulled['pairs'] == sync.hcp_bundle_keys(prefix, feats)
    # mask + affine + meta + one file per feature, nothing more
    assert len(pulled['pairs']) == 3 + len(feats)
