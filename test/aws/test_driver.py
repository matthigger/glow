"""drive_aws: helpers, submit shape, happy path, OOM tier escalation.

An in-memory FakeS3 + FakeBatch drive the submit -> poll -> classify ->
escalate loop without provisioning AWS.
"""

import json
from unittest.mock import patch

from glow._extra.aws.config import AWSConfig
from glow._extra.aws.driver import (_Attempt, _classify, _inflight_postfix,
                                     _is_oom, _submit_job, drive_aws)
from test.aws.fakes import FakeBatch, FakeS3, client_factory

OOM = {'status': 'FAILED',
       'container': {'exitCode': 137, 'reason': 'OutOfMemoryError'}}
OK = {'status': 'SUCCEEDED'}
# a non-OOM crash. NB _is_oom matches 'oom'/'memory' as a substring (faithful
# to the old driver), so the reason must avoid them -- e.g. not 'boom'.
CRASH = {'status': 'FAILED',
         'container': {'exitCode': 1, 'reason': 'nonzero exit'}}


def _cfg():
    return AWSConfig(s3_bucket='bkt', job_queue='q', job_definition='d',
                     s3_prefix='glow', poll_seconds=0,
                     memory_mb_tiers=[4000, 8000])


# ---------- pure helpers ----------------------------------------------------


def test_attempt_child_ids_array_vs_single():
    arr = _Attempt(name='c', cell_indices=[3, 4], parent_id='p', is_array=True)
    assert arr.child_ids == ['p:0', 'p:1']
    one = _Attempt(name='c', cell_indices=[3], parent_id='p', is_array=False)
    assert one.child_ids == ['p']


def test_is_oom():
    assert _is_oom(OOM)
    assert _is_oom({'status': 'FAILED',
                    'statusReason': 'Task failed: OutOfMemory'})
    assert not _is_oom(CRASH)
    # a timeout also exits 137 but must not be treated as OOM
    assert not _is_oom({'status': 'FAILED',
                        'statusReason': 'duration exceeded timeout',
                        'container': {'exitCode': 137}})


def test_classify_partitions():
    statuses = [OK, OOM, CRASH]
    completed, oom, other = _classify(statuses, [10, 11, 12])
    assert completed == [10]
    assert oom == [11]
    assert [c for c, _ in other] == [12]


def test_inflight_postfix():
    a = _Attempt(name='c', cell_indices=[0, 1, 2], parent_id='p',
                 is_array=True)
    statuses = {'p:0': {'status': 'SUCCEEDED'}, 'p:1': {'status': 'RUNNING'},
                'p:2': {'status': 'RUNNABLE'}}
    # terminal child omitted; the rest tallied by state
    assert _inflight_postfix(a, statuses) == 'RUNNABLE=1 RUNNING=1'


def test_submit_job_array_vs_single():
    batch = FakeBatch(submit_then=[[OK]])
    cfg = _cfg()
    _submit_job(batch=batch, aws_config=cfg, run_id='r', manifest_uri='s3://x',
                n=2, mem_mb=4000)
    assert batch.submitted[-1]['arrayProperties'] == {'size': 2}
    _submit_job(batch=batch, aws_config=cfg, run_id='r', manifest_uri='s3://x',
                n=1, mem_mb=4000)
    assert 'arrayProperties' not in batch.submitted[-1]
    # vcpus + memory land in the container override
    reqs = {r['type']: r['value']
            for r in batch.submitted[-1]['containerOverrides'][
                'resourceRequirements']}
    assert reqs == {'VCPU': '1', 'MEMORY': '4000'}


# ---------- end-to-end orchestration ----------------------------------------


def _run(fake_s3, fake_batch, **kw):
    with patch('glow._extra.aws.driver.boto3.client',
               client_factory(fake_s3, fake_batch)):
        return drive_aws('sweep_llr', _cfg(), write_csv=False, verbose=False,
                         **kw)


def test_happy_path_submits_one_array_and_finishes():
    # sweep_llr has 15 WGN cells -> one array job of size 15, all succeed
    fake_s3, fake_batch = FakeS3(), FakeBatch(submit_then=[[OK] * 15])
    _run(fake_s3, fake_batch)
    assert len(fake_batch.submitted) == 1
    assert fake_batch.submitted[0]['arrayProperties'] == {'size': 15}
    # the manifest names the cache, sources, and this attempt's cell indices
    (_, manifest_key), = [k for k in fake_s3.store
                          if k[1].endswith('manifest.json')]
    manifest = json.loads(fake_s3.store[('bkt', manifest_key)])
    assert manifest['config_name'] == 'sweep_llr'
    assert manifest['sources'] == ['wgn']
    assert manifest['cell_indices'] == list(range(15))


def test_oom_escalates_to_next_tier():
    # cell 0 OOMs at tier 0, only it is retried at tier 1 (where it succeeds)
    tier0 = [OOM] + [OK] * 14
    tier1 = [OK]  # one cell -> single (non-array) job
    fake_s3 = FakeS3()
    fake_batch = FakeBatch(submit_then=[tier0, tier1])
    _run(fake_s3, fake_batch)
    assert len(fake_batch.submitted) == 2
    # tier 1 resubmits exactly the OOM cell, at the larger memory tier
    second = fake_batch.submitted[1]
    mem = {r['type']: r['value']
           for r in second['containerOverrides']['resourceRequirements']}
    assert mem['MEMORY'] == '8000'
    manifests = [json.loads(v) for (b, k), v in fake_s3.store.items()
                 if k.endswith('manifest.json')]
    # two manifests: the full tier-0 grid and the tier-1 retry of just cell 0
    assert sorted(m['cell_indices'] for m in manifests) == [[0], list(range(15))]


def test_permanent_failure_not_retried():
    # a non-OOM crash is permanent: no escalation, reported as a failure
    fake_s3 = FakeS3()
    fake_batch = FakeBatch(submit_then=[[CRASH] + [OK] * 14])
    _run(fake_s3, fake_batch)
    assert len(fake_batch.submitted) == 1  # no tier-1 resubmission


def test_write_csv_pulls_records_and_writes(monkeypatch):
    calls = {}

    def _stub(out_dir=None, names=None):
        calls['names'] = list(names)
        return {}
    monkeypatch.setattr('glow._extra.benchmark.results.write_config_csvs',
                        _stub)
    fake_s3, fake_batch = FakeS3(), FakeBatch(submit_then=[[OK] * 15])
    with patch('glow._extra.aws.driver.boto3.client',
               client_factory(fake_s3, fake_batch)):
        written = drive_aws('sweep_llr', _cfg(), write_csv=True, verbose=False)
    assert written == {}
    assert calls['names'] == ['sweep_llr']
