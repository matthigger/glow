"""worker.main: roundtrip via the in-memory FakeS3 stub."""

import os
from unittest.mock import patch

import cloudpickle
import pandas as pd
import pytest

from glow._extra.aws import worker
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


def _check_dict(result):
    assert result == {'sum': 7}


def _check_df(result):
    assert isinstance(result, pd.DataFrame)
    assert result['sq'].tolist() == [9, 10]


@pytest.mark.parametrize('trial_hash, run_fnc, trial, check', [
    # worker cloudpickles whatever run_fnc returns — no type branching
    ('deadbeef', _run_fnc, {'x': 2, 'y': 5}, _check_dict),
    ('cafebabe', _run_fnc_df, {'x': 3}, _check_df),
])
def test_worker_result_roundtrip(trial_hash, run_fnc, trial, check):
    fake = FakeS3()
    bucket = 'b'
    manifest_key = f'pre/jobs/run-{trial_hash}/manifest.pkl'
    job_key = _seed(fake, bucket=bucket, manifest_key=manifest_key,
                    trial_hash=trial_hash, run_fnc=run_fnc, trial=trial)

    with patch('glow._extra.aws.worker.boto3.client', return_value=fake), \
         patch.dict(os.environ, {'AWS_BATCH_JOB_ARRAY_INDEX': '0'}):
        worker.main(f's3://{bucket}/{manifest_key}')

    result_key = job_key.rsplit('/', 1)[0] + '/result.pkl'
    check(cloudpickle.loads(fake.store[(bucket, result_key)]))


@pytest.mark.parametrize('env_index, expected_hash, expected_sum', [
    # AWS_BATCH_JOB_ARRAY_INDEX selects the manifest entry to run
    ('1', 'hash_b', 21),
    # env missing -> worker defaults to index 0 (the single-job case)
    (None, 'hash_a', 11),
])
def test_worker_picks_correct_array_index(env_index, expected_hash,
                                          expected_sum):
    fake = FakeS3()
    bucket = 'b'
    manifest_key = 'pre/jobs/run-multi/manifest.pkl'
    fake.store[(bucket, manifest_key)] = cloudpickle.dumps(
        ['hash_a', 'hash_b', 'hash_c'])

    for h, val in [('hash_a', 10), ('hash_b', 20), ('hash_c', 30)]:
        job_key = f'pre/jobs/{h}/job.pkl'
        fake.store[(bucket, job_key)] = cloudpickle.dumps(
            (_run_fnc, {'x': val, 'y': 1}))

    with patch('glow._extra.aws.worker.boto3.client', return_value=fake), \
         patch.dict(os.environ, {}, clear=False):
        if env_index is None:
            os.environ.pop('AWS_BATCH_JOB_ARRAY_INDEX', None)
        else:
            os.environ['AWS_BATCH_JOB_ARRAY_INDEX'] = env_index
        worker.main(f's3://{bucket}/{manifest_key}')

    result_key = f'pre/jobs/{expected_hash}/result.pkl'
    assert cloudpickle.loads(fake.store[(bucket, result_key)]) \
        == {'sum': expected_sum}


def test_worker_propagates_run_fnc_exception():
    fake = FakeS3()
    bucket = 'b'
    manifest_key = 'pre/jobs/run-bad/manifest.pkl'

    def _boom(**kw):
        raise RuntimeError('worker should crash')

    _seed(fake, bucket=bucket, manifest_key=manifest_key,
          trial_hash='bad', run_fnc=_boom, trial={})

    with patch('glow._extra.aws.worker.boto3.client', return_value=fake), \
         patch.dict(os.environ, {'AWS_BATCH_JOB_ARRAY_INDEX': '0'}):
        with pytest.raises(RuntimeError, match='worker should crash'):
            worker.main(f's3://{bucket}/{manifest_key}')

    assert (bucket, 'pre/jobs/bad/result.pkl') not in fake.store
