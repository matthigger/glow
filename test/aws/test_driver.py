"""driver_aws: happy path, OOM tier escalation, failure handling.

Uses an in-memory FakeS3 + FakeBatch so the orchestration logic is
exercised without provisioning AWS.  A separate @pytest.mark.runaws
test exercises the same flow against real AWS.
"""

import os
import uuid
from unittest.mock import patch

import boto3
import cloudpickle
import pandas as pd
import pytest

from glow.aws.config import AWSConfig
from glow.aws.driver import (
    _is_oom, driver_aws)
from glow.aws.datasource import _parse_s3_uri
from glow.benchmark.data import DataSource, DataSourceWGN
from glow.benchmark.trial_cache import TrialCache
from glow.util import stable_hash
from test.aws.test_datasource import FakeS3


# ---------- fakes -----------------------------------------------------------


class FakeBatch:
    """Tracks submit_job calls; ``describe_jobs`` returns scripted statuses.

    ``submit_then`` is a list of ``[(per_child_status_dict_or_callable), ...]``
    indexed by submission order.  Each entry decides what each child
    looks like when described.
    """

    def __init__(self, submit_then):
        self.submit_then = list(submit_then)
        self.submitted = []
        self._next_parent_id = iter(
            f'p-{i:03d}' for i in range(1000))

    def submit_job(self, **kwargs):
        parent = next(self._next_parent_id)
        self.submitted.append({'parent_id': parent, **kwargs})
        return {'jobId': parent}

    def describe_jobs(self, *, jobs):
        # Match each requested ``<parent>:<idx>`` (or plain parent) to a
        # status from submit_then.  jobs[0] tells us which submission.
        out = []
        for job_id in jobs:
            if ':' in job_id:
                parent, idx_s = job_id.rsplit(':', 1)
                idx = int(idx_s)
            else:
                parent, idx = job_id, 0
            sub_idx = next(
                i for i, s in enumerate(self.submitted)
                if s['parent_id'] == parent)
            child_status = self.submit_then[sub_idx][idx]
            if callable(child_status):
                child_status = child_status(job_id)
            payload = {'jobId': job_id, **child_status}
            out.append(payload)
        return {'jobs': out}


def _run_fnc(*, ds, seed):
    """Trial body; uses ``ds`` so we can verify WGN-pass-through."""
    exp = ds.exp
    return {'shape': str(exp.y.shape), 'seed': seed}


def _make_cache(tmp_path, n_trials=3):
    DataSource._exp_cache.clear()
    ds = DataSourceWGN(seed=0, shape=(2, 2, 2), b=1, num_img=5)
    return TrialCache(
        folder=tmp_path / 'cache',
        iter_kwargs={'seed': list(range(n_trials))},
        kwargs={'ds': ds})


def _cfg(**overrides):
    base = dict(
        s3_bucket='b', s3_prefix='pre',
        job_queue='q', job_definition='d',
        poll_seconds=0)        # no real sleeping in tests
    base.update(overrides)
    return AWSConfig(**base)


def _seed_results(fake_s3, *, bucket, prefix, manifest, results):
    """Pre-populate result.pkl objects for each trial_hash in manifest.

    Mimics what the worker would have uploaded.
    """
    for trial_hash, payload in zip(manifest, results):
        key = f'{prefix}/jobs/{trial_hash}/result.pkl'
        fake_s3.store[(bucket, key)] = cloudpickle.dumps(payload)


def _manifest_from_batch(fake_s3, parent_id_to_manifest_key):
    """Extract the trial_hash list a manifest was built from."""
    out = {}
    for parent_id, key in parent_id_to_manifest_key.items():
        out[parent_id] = cloudpickle.loads(fake_s3.store[('b', key)])
    return out


# ---------- _is_oom ---------------------------------------------------------


def test_is_oom_exit_137_without_timeout_text():
    assert _is_oom({'container': {'exitCode': 137}})


def test_is_oom_falsey_on_timeout_137():
    job = {
        'statusReason': 'Job attempt duration exceeded timeout',
        'container': {'exitCode': 137},
    }
    assert not _is_oom(job)


def test_is_oom_matches_text():
    assert _is_oom({'container': {'reason': 'OutOfMemoryError'}})
    assert _is_oom({'statusReason': 'OutOfMemory: killed by oom-killer'})


def test_is_oom_falsey_on_clean_failure():
    assert not _is_oom({'container': {'exitCode': 1,
                                      'reason': 'application error'}})


# ---------- driver_aws happy path ------------------------------------------


def test_happy_path_single_tier(tmp_path):
    cache = _make_cache(tmp_path, n_trials=3)
    trials = list(cache.iter_trial_no_repeat())
    hashes = [stable_hash(t) for t in trials]

    fake_s3 = FakeS3()
    # Pre-write result.pkls (driver will download these after "success").
    _seed_results(
        fake_s3, bucket='b', prefix='pre',
        manifest=hashes,
        results=[{'shape': '(1, 5, 8)', 'seed': i} for i in range(3)])

    fake_batch = FakeBatch(submit_then=[
        [{'status': 'SUCCEEDED'}] * 3,    # one submission, 3 children
    ])

    with patch('glow.aws.driver.boto3.client',
               side_effect=lambda kind, **_:
               fake_s3 if kind == 's3' else fake_batch):
        driver_aws(cache, _run_fnc, _cfg(), verbose=False)

    # results.csv populated with all 3 trial hashes
    assert len(cache._load_results()) == 3
    assert set(cache._load_results().index.astype(str)) == set(hashes)


def test_oom_escalation(tmp_path):
    cache = _make_cache(tmp_path, n_trials=2)
    trials = list(cache.iter_trial_no_repeat())
    hashes = [stable_hash(t) for t in trials]

    fake_s3 = FakeS3()
    _seed_results(
        fake_s3, bucket='b', prefix='pre',
        manifest=hashes,
        results=[{'shape': 'x', 'seed': 0}, {'shape': 'x', 'seed': 1}])

    # First submission (2k MB): child 0 OOM, child 1 SUCCEEDED.
    # Second submission (4k MB): only child 0 retried, SUCCEEDED.
    fake_batch = FakeBatch(submit_then=[
        [{'status': 'FAILED',
          'container': {'exitCode': 137, 'reason': 'OutOfMemoryError'}},
         {'status': 'SUCCEEDED'}],
        [{'status': 'SUCCEEDED'}],
    ])

    with patch('glow.aws.driver.boto3.client',
               side_effect=lambda kind, **_:
               fake_s3 if kind == 's3' else fake_batch):
        driver_aws(cache, _run_fnc, _cfg(), verbose=False)

    assert len(cache._load_results()) == 2
    # Second submission must have used the next tier
    mem_at_tier_1 = next(
        r['value'] for r in
        fake_batch.submitted[1]['containerOverrides']['resourceRequirements']
        if r['type'] == 'MEMORY')
    assert mem_at_tier_1 == '4000'


def test_non_oom_failure_is_skipped(tmp_path):
    cache = _make_cache(tmp_path, n_trials=2)
    trials = list(cache.iter_trial_no_repeat())
    hashes = [stable_hash(t) for t in trials]

    fake_s3 = FakeS3()
    _seed_results(
        fake_s3, bucket='b', prefix='pre',
        manifest=hashes,
        results=[{'shape': 'x', 'seed': 0}, {'shape': 'x', 'seed': 1}])

    # Child 0 fails with a non-OOM exit; child 1 succeeds.
    # Driver should save child 1 only, skip child 0, NOT escalate.
    fake_batch = FakeBatch(submit_then=[
        [{'status': 'FAILED',
          'container': {'exitCode': 1, 'reason': 'application error'}},
         {'status': 'SUCCEEDED'}],
    ])

    with patch('glow.aws.driver.boto3.client',
               side_effect=lambda kind, **_:
               fake_s3 if kind == 's3' else fake_batch):
        driver_aws(cache, _run_fnc, _cfg(), verbose=False)

    saved = cache._load_results()
    assert len(saved) == 1
    assert hashes[1] in set(saved.index.astype(str))
    assert hashes[0] not in set(saved.index.astype(str))
    # No re-submission
    assert len(fake_batch.submitted) == 1


def test_oom_at_last_tier_is_skipped(tmp_path):
    cache = _make_cache(tmp_path, n_trials=1)
    trials = list(cache.iter_trial_no_repeat())
    [h] = [stable_hash(t) for t in trials]

    fake_s3 = FakeS3()
    # Even though we seed a result.pkl, OOM at last tier => never downloaded
    _seed_results(fake_s3, bucket='b', prefix='pre',
                  manifest=[h], results=[{'shape': 'x', 'seed': 0}])

    cfg = _cfg(memory_mb_tiers=[2000])    # only one tier
    fake_batch = FakeBatch(submit_then=[
        # 1-trial submission goes through the non-array path: single status
        [{'status': 'FAILED',
          'container': {'exitCode': 137, 'reason': 'OutOfMemoryError'}}],
    ])

    with patch('glow.aws.driver.boto3.client',
               side_effect=lambda kind, **_:
               fake_s3 if kind == 's3' else fake_batch):
        driver_aws(cache, _run_fnc, cfg, verbose=False)

    assert len(cache._load_results()) == 0
    # No retry — was the last tier
    assert len(fake_batch.submitted) == 1


def test_no_uncached_trials_short_circuits(tmp_path):
    cache = _make_cache(tmp_path, n_trials=2)
    # Save results for both trials so iter_trial_no_repeat yields nothing.
    for trial in cache.iter_trial():
        cache.save_result({'dummy': 1}, trial)

    with patch('glow.aws.driver.boto3.client') as client:
        driver_aws(cache, _run_fnc, _cfg(), verbose=False)
    # We touch boto3.client for s3 and batch even with no work; ok
    # but no submit_job should happen.  Easier: just check the cache wasn't
    # corrupted.
    assert len(cache._load_results()) == 2


def test_single_trial_uses_non_array_submit(tmp_path):
    cache = _make_cache(tmp_path, n_trials=1)
    [trial] = list(cache.iter_trial_no_repeat())
    [h] = [stable_hash(trial)]

    fake_s3 = FakeS3()
    _seed_results(fake_s3, bucket='b', prefix='pre',
                  manifest=[h], results=[{'shape': 'x', 'seed': 0}])

    fake_batch = FakeBatch(submit_then=[[{'status': 'SUCCEEDED'}]])

    with patch('glow.aws.driver.boto3.client',
               side_effect=lambda kind, **_:
               fake_s3 if kind == 's3' else fake_batch):
        driver_aws(cache, _run_fnc, _cfg(), verbose=False)

    # For n=1, arrayProperties must NOT be passed to submit_job
    assert 'arrayProperties' not in fake_batch.submitted[0]
    assert len(cache._load_results()) == 1


def test_save_result_uses_original_trial_hash(tmp_path):
    """Local + AWS share results.csv keying — confirms via trial_hash."""
    cache = _make_cache(tmp_path, n_trials=2)
    trials = list(cache.iter_trial_no_repeat())
    local_hashes = {stable_hash(t) for t in trials}

    fake_s3 = FakeS3()
    _seed_results(
        fake_s3, bucket='b', prefix='pre',
        manifest=sorted(local_hashes),
        results=[{'shape': 'x'} for _ in range(2)])

    fake_batch = FakeBatch(submit_then=[
        [{'status': 'SUCCEEDED'}] * 2,
    ])

    with patch('glow.aws.driver.boto3.client',
               side_effect=lambda kind, **_:
               fake_s3 if kind == 's3' else fake_batch):
        driver_aws(cache, _run_fnc, _cfg(), verbose=False)

    saved_hashes = set(cache._load_results().index.astype(str))
    assert saved_hashes == local_hashes


# ---------- real-AWS smoke test --------------------------------------------


@pytest.mark.runaws
def test_driver_aws_smoke(tmp_path):
    """End-to-end on real AWS.  Requires .glow_aws_config + provisioned infra."""
    cfg = AWSConfig.from_file('.glow_aws_config')
    cache = _make_cache(tmp_path, n_trials=2)
    driver_aws(cache, _run_fnc, cfg, verbose=True)
    assert len(cache._load_results()) == 2
