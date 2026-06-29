"""worker.main: maps the array index to the right cell and runs it.

The heavy parts (S3 sync, the actual fit) are stubbed; the test pins the one
piece of worker logic that must be right -- AWS_BATCH_JOB_ARRAY_INDEX picking
the correct data cell out of the same enumeration the driver used.
"""

import json
from unittest.mock import patch

from glow._extra.aws import sync, worker
from glow._extra.aws.units import resolve_cells
from test.aws.fakes import FakeS3


def test_array_index_selects_cell(monkeypatch):
    bucket, prefix = 'bkt', 'glow'
    manifest = {'config_name': 'sweep_llr', 'sources': ['wgn'],
                'cell_indices': [5, 6, 7], 's3_prefix': prefix,
                'region': 'us-east-1'}
    fake = FakeS3()
    manifest_key = f'{prefix}/runs/run-x/manifest.json'
    fake.put_object(Bucket=bucket, Key=manifest_key,
                    Body=json.dumps(manifest).encode())

    # stub the sync (no real dirs / threads) and the fit (record its cell)
    monkeypatch.setattr('glow._extra.aws.sync.sync_pairs', lambda p: [])
    seen = {}
    monkeypatch.setattr('glow._extra.benchmark.driver._run_data_cell',
                        lambda kd, ke, kf, fnc, config_name=None:
                        seen.update(kwargs_data=kd, config_name=config_name))
    monkeypatch.setenv('AWS_BATCH_JOB_ARRAY_INDEX', '2')

    with patch('glow._extra.aws.worker.boto3.client', lambda *a, **k: fake):
        worker.main(f's3://{bucket}/{manifest_key}')

    # array index 2 -> cell_indices[2] == 7 -> the 7th WGN cell of sweep_llr
    data_cells, *_ = resolve_cells('sweep_llr', ('wgn',))
    assert seen['kwargs_data'] == data_cells[7]
    assert seen['config_name'] == 'sweep_llr'


def test_hcp_cell_pulls_only_its_features(monkeypatch):
    # an HCP cell pulls just the bundle files for its hcp_feats (+ shared
    # mask/affine/meta), via download_each -- not the whole panel
    bucket, prefix = 'bkt', 'glow'
    hcp_cells, *_ = resolve_cells('smoke', ('hcp',))
    manifest = {'config_name': 'smoke', 'sources': ['hcp'],
                'cell_indices': [0], 's3_prefix': prefix, 'region': None}
    fake = FakeS3()
    manifest_key = f'{prefix}/runs/run-h/manifest.json'
    fake.put_object(Bucket=bucket, Key=manifest_key,
                    Body=json.dumps(manifest).encode())

    monkeypatch.setattr('glow._extra.aws.sync.sync_pairs', lambda p: [])
    monkeypatch.setattr('glow._extra.benchmark.driver._run_data_cell',
                        lambda *a, **k: None)
    pulled = {}
    monkeypatch.setattr('glow._extra.aws.s3.download_each',
                        lambda c, b, pairs: pulled.update(pairs=list(pairs)))
    monkeypatch.setenv('AWS_BATCH_JOB_ARRAY_INDEX', '0')

    with patch('glow._extra.aws.worker.boto3.client', lambda *a, **k: fake):
        worker.main(f's3://{bucket}/{manifest_key}')

    feats = hcp_cells[0]['hcp_feats']
    assert pulled['pairs'] == sync.hcp_bundle_keys(prefix, feats)
    # mask + affine + meta + one file per feature, nothing more
    assert len(pulled['pairs']) == 3 + len(feats)
