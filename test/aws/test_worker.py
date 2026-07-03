"""worker.main: array-index cell selection and the real _run_data_cell call.

test_array_index_selects_cell pins the index logic with the fit stubbed --
AWS_BATCH_JOB_ARRAY_INDEX must pick the right cell of the bundle the driver
shipped. test_runs_real_data_cell then drives the real _run_data_cell end to
end (only S3 sync faked), so any drift between the worker's call and the driver
signature fails in-process rather than silently on a Batch worker. A stubbed
fit hides that drift: the stub's own signature drifts with the caller, so the
real function is the only faithful guard. test_hcp_cell_pulls_only_its_features
checks an HCP cell pulls just its feature bundle, not the whole panel.
"""

import json
import pickle
from unittest.mock import patch

import joblib
import pytest

from glow._extra.aws import sync, worker
from glow._extra.aws.bundle import fnc_to_ref
from glow._extra.aws.units import resolve_cells
from glow._extra.benchmark import data
from test.aws.fakes import FakeS3

# the real end-to-end test ships this as its leaf fnc (by import reference, so
# it must be a top-level function); each call appends here for the assertions
_FNC_CALLS = []


def _spy_fnc(exp, mask_target_list, **kwargs):
    """Record one leaf call and return a trivial score (end-to-end test fnc)."""
    _FNC_CALLS.append({'mask_target_list': list(mask_target_list),
                       'kwargs': kwargs})
    return {'ok': True}


@pytest.fixture(autouse=True)
def _records_to_tmp(monkeypatch, tmp_path):
    """Send the recorder's per-hash files to a tmp dir, not the real one."""
    monkeypatch.setattr(data.RECORDER, 'folder', tmp_path)


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
    data_cells, eff, fnc_kw, fnc = resolve_cells('sweep_llr_b1')
    # the driver ships a subset (here cells 5, 6, 7); child i runs the i-th
    bundle = ([data_cells[5], data_cells[6], data_cells[7]], eff, fnc_kw,
              fnc_to_ref(fnc), 'sweep_llr_b1')
    fake = FakeS3()
    uri = _put_bundle(fake, bucket, prefix, 'run-x', bundle)

    # stub the sync (no real dirs / threads) and the fit (record its cell)
    monkeypatch.setattr('glow._extra.aws.sync.sync_pairs', lambda p: [])
    seen = {}
    monkeypatch.setattr('glow._extra.benchmark.driver._run_data_cell',
                        lambda kd, ke, kf, fnc: seen.update(kwargs_data=kd))
    monkeypatch.setenv('AWS_BATCH_JOB_ARRAY_INDEX', '2')

    with patch('glow._extra.aws.worker.boto3.client', lambda *a, **k: fake):
        worker.main(uri)

    # array index 2 -> the 3rd shipped cell is the 7th WGN cell of sweep_llr_b1.
    # Compare by joblib.hash (the cache key): the cells have no value __eq__,
    # and after the pickle round-trip the worker holds a fresh object -- what
    # must match a local run is its hash, not its identity.
    assert joblib.hash(seen['kwargs_data']) == joblib.hash(data_cells[7])


def test_runs_real_data_cell(monkeypatch):
    # drive the real _run_data_cell (unstubbed) on one WGN null cell so the
    # worker -> driver call binds against the live signature; a spy fnc stands
    # in for the leaf so no permutation fit is paid. Only S3 sync is faked.
    _FNC_CALLS.clear()
    bucket, prefix = 'bkt', 'glow'
    data_cells, *_ = resolve_cells('null')
    bundle = ([data_cells[0]], [None], [{}], fnc_to_ref(_spy_fnc), 'null')
    fake = FakeS3()
    uri = _put_bundle(fake, bucket, prefix, 'run-r', bundle)

    monkeypatch.setattr('glow._extra.aws.sync.sync_pairs', lambda p: [])
    monkeypatch.setenv('AWS_BATCH_JOB_ARRAY_INDEX', '0')

    with patch('glow._extra.aws.worker.boto3.client', lambda *a, **k: fake):
        worker.main(uri)

    # the real _run_data_cell reached the leaf once, on an empty (null) target
    assert len(_FNC_CALLS) == 1
    assert _FNC_CALLS[0] == {'mask_target_list': [], 'kwargs': {}}


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
