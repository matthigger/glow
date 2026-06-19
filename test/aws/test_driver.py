"""driver_aws: happy path, OOM tier escalation, failure handling.

Uses an in-memory FakeS3 + FakeBatch so the orchestration logic is
exercised without provisioning AWS.  A separate @pytest.mark.runaws
test exercises the same flow against real AWS.
"""

import sys
from unittest.mock import patch

import cloudpickle
import pytest

from glow.aws.config import AWSConfig
from glow.aws.driver import (
    _Attempt, _inflight_postfix, _is_oom, driver_aws, driver_aws_multi)
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

    Worked example — one OOM-then-success scenario across two memory tiers::

        FakeBatch(submit_then=[
            # submission 0 (tier 0, e.g. 2000 MB): child 0 OOMs, child 1 ok
            [{'status': 'FAILED',
              'container': {'exitCode': 137, 'reason': 'OutOfMemoryError'}},
             {'status': 'SUCCEEDED'}],
            # submission 1 (tier 1, e.g. 4000 MB): only the OOM child retried
            [{'status': 'SUCCEEDED'}],
        ])

    The driver makes one ``submit_job`` per tier attempt, so ``submit_then[i]``
    is the scripted outcome of the i-th tier the driver escalates through.
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


# ---------- _is_oom ---------------------------------------------------------


@pytest.mark.parametrize('job, expected', [
    # exit 137 with no timeout phrasing -> OOM
    ({'container': {'exitCode': 137}}, True),
    # exit 137 but reason names a timeout -> NOT OOM (don't escalate timeouts)
    ({'statusReason': 'Job attempt duration exceeded timeout',
      'container': {'exitCode': 137}}, False),
    # text match on the reason -> OOM
    ({'container': {'reason': 'OutOfMemoryError'}}, True),
    # clean non-OOM failure (exit 1) -> NOT OOM
    ({'container': {'exitCode': 1, 'reason': 'application error'}}, False),
])
def test_is_oom(job, expected):
    assert _is_oom(job) is expected


# ---------- _inflight_postfix -----------------------------------------------


def test_inflight_postfix_counts_active_states_in_order():
    """Non-terminal children are tallied per state, in lifecycle order."""
    attempt = _Attempt(label='x', manifest=['h0', 'h1', 'h2', 'h3'],
                       parent_id='p-000', is_array=True)
    statuses = {
        'p-000:0': {'status': 'RUNNING'},
        'p-000:1': {'status': 'RUNNABLE'},
        'p-000:2': {'status': 'RUNNING'},
        # h3 not yet seen by describe_jobs -> counted as SUBMITTED.
    }
    assert _inflight_postfix(attempt, statuses) == \
        'SUBMITTED=1 RUNNABLE=1 RUNNING=2'


def test_inflight_postfix_empty_when_all_terminal():
    """Once every child is terminal the postfix clears (bar counts them)."""
    attempt = _Attempt(label='x', manifest=['h0', 'h1'],
                       parent_id='p-000', is_array=True)
    statuses = {
        'p-000:0': {'status': 'SUCCEEDED'},
        'p-000:1': {'status': 'FAILED'},
    }
    assert _inflight_postfix(attempt, statuses) == ''


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

    # results.csv populated with all 3 trial hashes.  The saved index must
    # equal the LOCAL stable_hash values: local and AWS runs share results.csv
    # keying, so a trial run on AWS is later seen as cached locally.
    assert len(cache._load_results()) == 3
    assert set(cache._load_results().index.astype(str)) == set(hashes)


def test_success_deletes_s3_objects(tmp_path):
    """Each succeeded trial's job + result pickles are dropped from S3."""
    cache = _make_cache(tmp_path, n_trials=3)
    trials = list(cache.iter_trial_no_repeat())
    hashes = [stable_hash(t) for t in trials]

    fake_s3 = FakeS3()
    _seed_results(
        fake_s3, bucket='b', prefix='pre', manifest=hashes,
        results=[{'shape': 'x', 'seed': i} for i in range(3)])

    fake_batch = FakeBatch(submit_then=[[{'status': 'SUCCEEDED'}] * 3])

    with patch('glow.aws.driver.boto3.client',
               side_effect=lambda kind, **_:
               fake_s3 if kind == 's3' else fake_batch):
        driver_aws(cache, _run_fnc, _cfg(), verbose=False)

    # Both job.pkl and result.pkl are gone for every trial.
    for h in hashes:
        assert ('b', f'pre/jobs/{h}/job.pkl') not in fake_s3.store
        assert ('b', f'pre/jobs/{h}/result.pkl') not in fake_s3.store
    deleted = {key for kind, _, key in fake_s3.calls if kind == 'delete'}
    assert deleted == {f'pre/jobs/{h}/{name}'
                       for h in hashes for name in ('job.pkl', 'result.pkl')}


def test_failed_trial_objects_are_not_deleted(tmp_path):
    """A non-OOM failure leaves its S3 objects in place (no delete)."""
    cache = _make_cache(tmp_path, n_trials=2)
    trials = list(cache.iter_trial_no_repeat())
    hashes = [stable_hash(t) for t in trials]

    fake_s3 = FakeS3()
    _seed_results(
        fake_s3, bucket='b', prefix='pre', manifest=hashes,
        results=[{'shape': 'x', 'seed': 0}, {'shape': 'x', 'seed': 1}])

    # Child 0 fails (non-OOM), child 1 succeeds.
    fake_batch = FakeBatch(submit_then=[
        [{'status': 'FAILED',
          'container': {'exitCode': 1, 'reason': 'application error'}},
         {'status': 'SUCCEEDED'}],
    ])

    with patch('glow.aws.driver.boto3.client',
               side_effect=lambda kind, **_:
               fake_s3 if kind == 's3' else fake_batch):
        driver_aws(cache, _run_fnc, _cfg(), verbose=False)

    deleted = {key for kind, _, key in fake_s3.calls if kind == 'delete'}
    # Only the succeeded trial (child 1) is cleaned up.
    assert f'pre/jobs/{hashes[1]}/result.pkl' in deleted
    assert f'pre/jobs/{hashes[0]}/result.pkl' not in deleted
    assert f'pre/jobs/{hashes[0]}/job.pkl' not in deleted


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
    # The real short-circuit guarantee: with nothing uncached, the driver
    # returns before touching AWS at all (no s3/batch client, no submit_job).
    client.assert_not_called()
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


# ---------- driver_aws_multi ------------------------------------------------


def test_multi_cache_submits_all_before_polling(tmp_path):
    """Two caches submit one array job each in the same tier, then poll.

    The old per-cache loop drained one cache to completion before
    submitting the next; driver_aws_multi must instead have both array
    jobs in flight (two submissions, one tier) and save both caches.
    """
    DataSource._exp_cache.clear()
    ds = DataSourceWGN(seed=0, shape=(2, 2, 2), b=1, num_img=5)

    def _cache(name, seeds):
        return TrialCache(folder=tmp_path / name,
                          iter_kwargs={'seed': seeds}, kwargs={'ds': ds})

    cache_a = _cache('a', [0, 1])
    cache_b = _cache('b', [2, 3])
    hashes_a = [stable_hash(t) for t in cache_a.iter_trial_no_repeat()]
    hashes_b = [stable_hash(t) for t in cache_b.iter_trial_no_repeat()]

    fake_s3 = FakeS3()
    _seed_results(fake_s3, bucket='b', prefix='pre', manifest=hashes_a,
                  results=[{'seed': 0}, {'seed': 1}])
    _seed_results(fake_s3, bucket='b', prefix='pre', manifest=hashes_b,
                  results=[{'seed': 2}, {'seed': 3}])

    # One array submission per cache (submission order = job order), all ok.
    fake_batch = FakeBatch(submit_then=[
        [{'status': 'SUCCEEDED'}] * 2,
        [{'status': 'SUCCEEDED'}] * 2,
    ])

    with patch('glow.aws.driver.boto3.client',
               side_effect=lambda kind, **_:
               fake_s3 if kind == 's3' else fake_batch):
        driver_aws_multi(
            [('a', cache_a, _run_fnc), ('b', cache_b, _run_fnc)],
            _cfg(), verbose=False)

    assert len(cache_a._load_results()) == 2
    assert len(cache_b._load_results()) == 2
    # Both caches submitted in one tier as array jobs (not one drained first).
    assert len(fake_batch.submitted) == 2
    assert all('arrayProperties' in s for s in fake_batch.submitted)


def test_multi_cache_per_cache_oom_escalation(tmp_path):
    """OOM children escalate per cache while the other cache stays put."""
    DataSource._exp_cache.clear()
    ds = DataSourceWGN(seed=0, shape=(2, 2, 2), b=1, num_img=5)

    def _cache(name, seeds):
        return TrialCache(folder=tmp_path / name,
                          iter_kwargs={'seed': seeds}, kwargs={'ds': ds})

    cache_a = _cache('a', [0, 1])
    cache_b = _cache('b', [2, 3])
    hashes_a = [stable_hash(t) for t in cache_a.iter_trial_no_repeat()]
    hashes_b = [stable_hash(t) for t in cache_b.iter_trial_no_repeat()]

    fake_s3 = FakeS3()
    _seed_results(fake_s3, bucket='b', prefix='pre', manifest=hashes_a,
                  results=[{'seed': 0}, {'seed': 1}])
    _seed_results(fake_s3, bucket='b', prefix='pre', manifest=hashes_b,
                  results=[{'seed': 2}, {'seed': 3}])

    # Tier 0: cache a (sub 0) both ok; cache b (sub 1) child 0 OOMs.
    # Tier 1: only cache b's OOM child retried (sub 2), succeeds.
    fake_batch = FakeBatch(submit_then=[
        [{'status': 'SUCCEEDED'}] * 2,
        [{'status': 'FAILED',
          'container': {'exitCode': 137, 'reason': 'OutOfMemoryError'}},
         {'status': 'SUCCEEDED'}],
        [{'status': 'SUCCEEDED'}],
    ])

    with patch('glow.aws.driver.boto3.client',
               side_effect=lambda kind, **_:
               fake_s3 if kind == 's3' else fake_batch):
        driver_aws_multi(
            [('a', cache_a, _run_fnc), ('b', cache_b, _run_fnc)],
            _cfg(), verbose=False)

    assert len(cache_a._load_results()) == 2
    assert len(cache_b._load_results()) == 2
    # Three submissions: two at tier 0 (a, b) and only b retried at tier 1.
    assert len(fake_batch.submitted) == 3
    mem_at_tier_1 = next(
        r['value'] for r in
        fake_batch.submitted[2]['containerOverrides']['resourceRequirements']
        if r['type'] == 'MEMORY')
    assert mem_at_tier_1 == '4000'


# ---------- real-AWS smoke test --------------------------------------------


@pytest.mark.runaws
def test_driver_aws_smoke(tmp_path):
    """End-to-end on real AWS.  Requires a written AWSConfig + provisioned infra."""
    # _run_fnc lives in this `test` package, which is importable here but
    # NOT shipped into the worker image (the Dockerfile copies only glow/).
    # Without this, cloudpickle stores run_fnc by reference and the worker
    # dies on `import test` -> ModuleNotFoundError; register the module so
    # run_fnc is pickled by value and travels with the job.
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    cfg = AWSConfig.from_file()
    cache = _make_cache(tmp_path, n_trials=2)
    driver_aws(cache, _run_fnc, cfg, verbose=True)
    assert len(cache._load_results()) == 2
