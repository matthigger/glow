"""drive_aws: helpers, submit shape, retry policy, happy path, OOM escalation.

An in-memory FakeS3 + FakeBatch drive the submit -> poll -> classify ->
escalate loop without provisioning AWS.
"""

import fnmatch
import json
import pickle
from unittest.mock import patch

import joblib
import pytest

from glow._extra.aws.config import AWSConfig
from glow._extra.aws.driver import (RETRY_EVALUATE_ON_EXIT, _Attempt,
                                     _classify, _inflight_postfix, _is_oom,
                                     _poll_attempts, _pull_finished,
                                     _submit_job, drive_aws)
from glow._extra.aws.units import resolve_cells
from glow._extra.benchmark import data
from test.aws.fakes import FakeBatch, FakeS3, client_factory


@pytest.fixture(autouse=True)
def _empty_records(monkeypatch, tmp_path):
    """Start from empty records so drive_aws's local-records skip finds
    nothing complete -- every cell submits, keeping the array sizes below
    deterministic regardless of what has actually been run locally."""
    monkeypatch.setattr(data.RECORDER, 'folder', tmp_path)
    data.RECORDER.records.clear()

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


def _batch_action(rules, *, exit_code=None, status_reason='', reason=''):
    """Mimic AWS Batch evaluateOnExit: first matching rule wins, else RETRY.

    A rule matches when every on* condition it lists matches (glob), so a
    rule needs a listed condition to fire; a job matching no rule is retried.
    """
    for rule in rules:
        conds = [k for k in ('onExitCode', 'onStatusReason', 'onReason')
                 if k in rule]
        matched = bool(conds)
        for k in conds:
            if k == 'onExitCode':
                matched = matched and exit_code is not None and \
                    fnmatch.fnmatch(str(exit_code), rule[k])
            elif k == 'onStatusReason':
                matched = matched and fnmatch.fnmatch(status_reason, rule[k])
            else:
                matched = matched and fnmatch.fnmatch(reason, rule[k])
        if matched:
            return rule['action']
    return 'RETRY'


def test_retry_evaluate_on_exit_policy():
    rules = RETRY_EVALUATE_ON_EXIT
    # Batch caps evaluateOnExit at 5; each rule needs an action + a condition
    assert len(rules) <= 5
    for r in rules:
        assert r['action'] in ('RETRY', 'EXIT')
        assert any(k in r for k in ('onExitCode', 'onStatusReason', 'onReason'))
    # OOM (137) / abort (134) exit so the driver escalates the memory tier
    # rather than Batch re-running the cell at the same, doomed, memory
    assert _batch_action(rules, exit_code=137, reason='OutOfMemoryError',
                         status_reason='Essential container in task exited') \
        == 'EXIT'
    assert _batch_action(rules, exit_code=134) == 'EXIT'
    # a wall-clock timeout also exits 137 -> exits, resurfaces on a rerun
    assert _batch_action(rules, exit_code=137,
                         status_reason='duration exceeded timeout') == 'EXIT'
    # Spot reclaim: a hard host loss and the graceful SIGTERM both retry in
    # place on a fresh box (the worker warm-resumes from the synced cache)
    assert _batch_action(
        rules, status_reason='Host EC2 (instance i-0abc) terminated.') \
        == 'RETRY'
    assert _batch_action(rules, exit_code=143) == 'RETRY'
    # any other failure is treated as transient and retried (bounded by
    # attempts)
    assert _batch_action(rules, exit_code=1,
                         status_reason='Essential container in task exited') \
        == 'RETRY'


def test_inflight_postfix():
    a = _Attempt(name='c', cell_indices=[0, 1, 2], parent_id='p',
                 is_array=True)
    statuses = {'p:0': {'status': 'SUCCEEDED'}, 'p:1': {'status': 'RUNNING'},
                'p:2': {'status': 'RUNNABLE'}}
    # terminal child omitted; the rest tallied by state
    assert _inflight_postfix(a, statuses) == 'RUNNABLE=1 RUNNING=1'


def test_pull_finished_incremental(tmp_path):
    # a record already on S3 is pulled once, then skipped (download_prefix is
    # incremental) -- so a repeated mid-run pull only moves new work
    fake_s3 = FakeS3()
    fake_s3.store[('bkt', 'glow/records/a.json')] = b'1'
    local_dir = tmp_path / 'records'
    pairs = [(local_dir, 'glow/records')]
    assert _pull_finished(fake_s3, 'bkt', pairs, verbose=False) == 1
    assert (local_dir / 'a.json').read_bytes() == b'1'
    assert _pull_finished(fake_s3, 'bkt', pairs, verbose=False) == 0


class _TransientBatch:
    """describe_jobs returns RUNNING on the first sweep, SUCCEEDED after.

    So _poll_attempts runs one full non-terminal sweep -- exercising the
    mid-run pull -- before the child goes terminal and the loop breaks.
    """

    def __init__(self):
        self._n = 0

    def describe_jobs(self, *, jobs):
        self._n += 1
        status = 'RUNNING' if self._n == 1 else 'SUCCEEDED'
        return {'jobs': [{'jobId': j, 'status': status} for j in jobs]}


def test_poll_attempts_pulls_mid_run(tmp_path):
    # with the gate open (interval 0), the first non-terminal sweep pulls the
    # record already on S3 down before the job finishes
    fake_s3 = FakeS3()
    fake_s3.store[('bkt', 'glow/records/abc.json')] = b'{}'
    local_dir = tmp_path / 'records'
    attempt = _Attempt(name='c', cell_indices=[0], parent_id='p-000',
                       is_array=False)
    statuses = _poll_attempts(
        batch=_TransientBatch(), attempts=[attempt], poll_seconds=0,
        verbose=False, s3_client=fake_s3, bucket='bkt',
        download_pairs=[(local_dir, 'glow/records')], download_interval=0)
    assert (local_dir / 'abc.json').read_bytes() == b'{}'
    assert statuses[0][0]['status'] == 'SUCCEEDED'


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


def test_submit_job_retry_strategy():
    batch = FakeBatch(submit_then=[[OK]])
    cfg = _cfg()
    _submit_job(batch=batch, aws_config=cfg, run_id='r', manifest_uri='s3://x',
                n=2, mem_mb=4000)
    rs = batch.submitted[-1]['retryStrategy']
    assert rs['attempts'] == cfg.retry_attempts
    assert rs['evaluateOnExit'] == RETRY_EVALUATE_ON_EXIT


# ---------- end-to-end orchestration ----------------------------------------

# sweep_llr's full CONFIG grid (both sources) -- the array size the driver
# submits; derived so it tracks config rather than a hard-coded count.
N_CELLS = len(resolve_cells('sweep_llr')[0])


def _run(fake_s3, fake_batch, **kw):
    with patch('glow._extra.aws.driver.boto3.client',
               client_factory(fake_s3, fake_batch)):
        return drive_aws('sweep_llr', _cfg(), write_csv=False, verbose=False,
                         **kw)


def test_happy_path_submits_one_array_and_finishes():
    # sweep_llr's full grid -> one array job of size N_CELLS, all succeed
    fake_s3, fake_batch = FakeS3(), FakeBatch(submit_then=[[OK] * N_CELLS])
    _run(fake_s3, fake_batch)
    assert len(fake_batch.submitted) == 1
    assert fake_batch.submitted[0]['arrayProperties'] == {'size': N_CELLS}
    # the manifest points at the pickled bundle; the bundle carries this
    # attempt's resolved cells + the shared grids + leaf fnc + cache label
    (_, manifest_key), = [k for k in fake_s3.store
                          if k[1].endswith('manifest.json')]
    manifest = json.loads(fake_s3.store[('bkt', manifest_key)])
    assert manifest['config_name'] == 'sweep_llr'
    assert manifest['n_cells'] == N_CELLS
    data_cells, *_, label = pickle.loads(
        fake_s3.store[('bkt', manifest['bundle_key'])])
    expected, *_ = resolve_cells('sweep_llr')
    assert label == 'sweep_llr'
    # cells have no value __eq__; the cache key (joblib.hash) is what matters
    assert joblib.hash(data_cells) == joblib.hash(expected)


def test_oom_escalates_to_next_tier():
    # cell 0 OOMs at tier 0, only it is retried at tier 1 (where it succeeds)
    tier0 = [OOM] + [OK] * (N_CELLS - 1)
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
    # two manifests: the full tier-0 grid and the tier-1 retry of one cell
    assert sorted(m['n_cells'] for m in manifests) == [1, N_CELLS]
    # the tier-1 retry ships exactly the OOM cell (cell 0)
    retry = next(m for m in manifests if m['n_cells'] == 1)
    cells, *_ = pickle.loads(fake_s3.store[('bkt', retry['bundle_key'])])
    expected, *_ = resolve_cells('sweep_llr')
    assert joblib.hash(cells) == joblib.hash([expected[0]])


def test_permanent_failure_not_retried():
    # a non-OOM crash is permanent: no escalation, reported as a failure
    fake_s3 = FakeS3()
    fake_batch = FakeBatch(submit_then=[[CRASH] + [OK] * (N_CELLS - 1)])
    _run(fake_s3, fake_batch)
    assert len(fake_batch.submitted) == 1  # no tier-1 resubmission


def test_write_csv_pulls_records_and_writes(monkeypatch):
    calls = {}

    def _stub(out_dir=None, names=None):
        calls['names'] = list(names)
        return {}
    monkeypatch.setattr('glow._extra.benchmark.results.write_config_csvs',
                        _stub)
    fake_s3, fake_batch = FakeS3(), FakeBatch(submit_then=[[OK] * N_CELLS])
    with patch('glow._extra.aws.driver.boto3.client',
               client_factory(fake_s3, fake_batch)):
        written = drive_aws('sweep_llr', _cfg(), write_csv=True, verbose=False)
    assert written == {}
    assert calls['names'] == ['sweep_llr']
