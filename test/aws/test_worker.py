"""worker.main: roundtrip via the in-memory FakeS3 stub."""

import os
from unittest.mock import patch

import cloudpickle
import pandas as pd
import pytest

from glow.aws import worker
from test.aws.test_datasource import FakeS3


def _run_fnc(*, x, y):
    return {'sum': x + y}


def _run_fnc_df(*, x):
    return pd.DataFrame([{'sq': x * x}, {'sq': x * x + 1}])


def _seed(fake, *, bucket, manifest_key, trial_hash, run_fnc, trial):
    """Populate manifest.pkl + job.pkl for one array index."""
    manifest = [trial_hash]
    fake.store[(bucket, manifest_key)] = cloudpickle.dumps(manifest)

    job_key = f'{manifest_key.rsplit("/", 2)[0]}/{trial_hash}/job.pkl'
    fake.store[(bucket, job_key)] = cloudpickle.dumps((run_fnc, trial))
    return job_key


def test_worker_dict_result_roundtrip():
    fake = FakeS3()
    bucket = 'b'
    manifest_key = 'pre/jobs/run-abc/manifest.pkl'
    job_key = _seed(fake, bucket=bucket, manifest_key=manifest_key,
                    trial_hash='deadbeef', run_fnc=_run_fnc,
                    trial={'x': 2, 'y': 5})

    with patch('glow.aws.worker.boto3.client', return_value=fake), \
         patch.dict(os.environ, {'AWS_BATCH_JOB_ARRAY_INDEX': '0'}):
        worker.main(f's3://{bucket}/{manifest_key}')

    result_key = job_key.rsplit('/', 1)[0] + '/result.pkl'
    assert cloudpickle.loads(fake.store[(bucket, result_key)]) == {'sum': 7}


def test_worker_df_result_roundtrip():
    fake = FakeS3()
    bucket = 'b'
    manifest_key = 'pre/jobs/run-xyz/manifest.pkl'
    job_key = _seed(fake, bucket=bucket, manifest_key=manifest_key,
                    trial_hash='cafebabe', run_fnc=_run_fnc_df,
                    trial={'x': 3})

    with patch('glow.aws.worker.boto3.client', return_value=fake), \
         patch.dict(os.environ, {'AWS_BATCH_JOB_ARRAY_INDEX': '0'}):
        worker.main(f's3://{bucket}/{manifest_key}')

    result_key = job_key.rsplit('/', 1)[0] + '/result.pkl'
    df = cloudpickle.loads(fake.store[(bucket, result_key)])
    assert isinstance(df, pd.DataFrame)
    assert df['sq'].tolist() == [9, 10]


def test_worker_picks_correct_array_index():
    fake = FakeS3()
    bucket = 'b'
    manifest_key = 'pre/jobs/run-multi/manifest.pkl'
    fake.store[(bucket, manifest_key)] = cloudpickle.dumps(
        ['hash_a', 'hash_b', 'hash_c'])

    for h, val in [('hash_a', 10), ('hash_b', 20), ('hash_c', 30)]:
        job_key = f'pre/jobs/{h}/job.pkl'
        fake.store[(bucket, job_key)] = cloudpickle.dumps(
            (_run_fnc, {'x': val, 'y': 1}))

    with patch('glow.aws.worker.boto3.client', return_value=fake), \
         patch.dict(os.environ, {'AWS_BATCH_JOB_ARRAY_INDEX': '1'}):
        worker.main(f's3://{bucket}/{manifest_key}')

    result_key = 'pre/jobs/hash_b/result.pkl'
    assert cloudpickle.loads(fake.store[(bucket, result_key)]) == {'sum': 21}


def test_worker_default_index_zero_when_env_missing():
    fake = FakeS3()
    bucket = 'b'
    manifest_key = 'pre/jobs/run-single/manifest.pkl'
    _seed(fake, bucket=bucket, manifest_key=manifest_key,
          trial_hash='lone', run_fnc=_run_fnc, trial={'x': 1, 'y': 1})

    with patch('glow.aws.worker.boto3.client', return_value=fake), \
         patch.dict(os.environ, {}, clear=False):
        os.environ.pop('AWS_BATCH_JOB_ARRAY_INDEX', None)
        worker.main(f's3://{bucket}/{manifest_key}')

    assert (bucket, 'pre/jobs/lone/result.pkl') in fake.store


def test_worker_propagates_run_fnc_exception():
    fake = FakeS3()
    bucket = 'b'
    manifest_key = 'pre/jobs/run-bad/manifest.pkl'

    def _boom(**kw):
        raise RuntimeError('worker should crash')

    _seed(fake, bucket=bucket, manifest_key=manifest_key,
          trial_hash='bad', run_fnc=_boom, trial={})

    with patch('glow.aws.worker.boto3.client', return_value=fake), \
         patch.dict(os.environ, {'AWS_BATCH_JOB_ARRAY_INDEX': '0'}):
        with pytest.raises(RuntimeError, match='worker should crash'):
            worker.main(f's3://{bucket}/{manifest_key}')

    assert (bucket, 'pre/jobs/bad/result.pkl') not in fake.store
