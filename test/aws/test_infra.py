"""infra CLI: interactive .glow_aws_config creation and `status` output."""

import types

import pytest

from glow.aws import infra
from glow.aws.config import AWSConfig


def _answers(*responses):
    """An input() stand-in replaying scripted terminal responses in order."""
    it = iter(responses)
    return lambda _prompt='': next(it)


@pytest.mark.parametrize('answer', ['y', 'Y', 'yes', ''])
def test_init_config_accepts_defaults(tmp_path, monkeypatch, answer):
    """Accepting the prompt (or a blank answer) writes the default names."""
    path = tmp_path / '.glow_aws_config'
    monkeypatch.setattr('builtins.input', _answers(answer))

    infra._init_config(str(path))

    cfg = AWSConfig.from_file(path)
    for key, default in infra.DEFAULT_CONFIG_NAMES.items():
        assert getattr(cfg, key) == default


def test_init_config_prompts_each_on_decline(tmp_path, monkeypatch):
    """Declining the defaults prompts per field; a blank keeps that default."""
    path = tmp_path / '.glow_aws_config'
    monkeypatch.setattr(
        'builtins.input', _answers('n', 'my-bucket', '', 'my-def'))

    infra._init_config(str(path))

    cfg = AWSConfig.from_file(path)
    assert cfg.s3_bucket == 'my-bucket'
    assert cfg.job_queue == infra.DEFAULT_CONFIG_NAMES['job_queue']
    assert cfg.job_definition == 'my-def'


@pytest.mark.parametrize('answer, expected', [
    ('', True), ('y', True), ('Y', True), ('yes', True),
    ('n', False), ('no', False), ('nope', False),
])
def test_prompt_yes(monkeypatch, answer, expected):
    monkeypatch.setattr('builtins.input', _answers(answer))
    assert infra._prompt_yes('ok?') is expected


# ---------- status helpers --------------------------------------------------


@pytest.mark.parametrize('seconds, expected', [
    (None, '—'), (0, '0s'), (5, '5s'), (74, '1m14s'), (3725, '1h02m'),
])
def test_fmt_dur(seconds, expected):
    assert infra._fmt_dur(seconds) == expected


@pytest.mark.parametrize('name, expected', [
    ('glow-run-1a2b3c4d', 'run-1a2b3c4d'), ('no-prefix', 'no-prefix'),
])
def test_run_id(name, expected):
    assert infra._run_id(name) == expected


def test_fmt_progress_array_and_single():
    """Array parents tally ✓/✗/⋯ from statusSummary; singles show status."""
    parent = {'arrayProperties': {'size': 8, 'statusSummary': {
        'SUCCEEDED': 5, 'FAILED': 2, 'RUNNING': 1}}}
    assert infra._fmt_progress(parent) == '5✓ 2✗ 1⋯'
    assert infra._fmt_progress({'status': 'RUNNING'}) == 'RUNNING'


@pytest.mark.parametrize('job, expected', [
    ({'container': {'exitCode': 137}}, 'OOM'),
    ({'container': {'exitCode': 134}}, 'OOM'),
    ({'statusReason': 'OutOfMemory: killed'}, 'OOM'),
    ({'container': {'exitCode': 137,
                    'reason': 'timeout: attempt duration exceeded'}}, 'timeout'),
    ({'container': {'exitCode': 1, 'reason': 'ValueError'}}, 'err'),
])
def test_failure_tag(job, expected):
    """Timeouts (which also exit 137) outrank the OOM exit-code heuristic."""
    assert infra._failure_tag(job) == expected


def test_timing():
    job = {'createdAt': 1000, 'startedAt': 4000, 'stoppedAt': 9000}
    assert infra._timing(job) == (3.0, 5.0)
    assert infra._timing({'createdAt': 1000}) == (None, None)


# ---------- cmd_status (mocked Batch / Logs) --------------------------------


class _FakePaginator:
    """list_jobs paginator: queue listing by status, or array children."""

    def __init__(self, client):
        self._client = client

    def paginate(self, **kw):
        if 'jobQueue' in kw:
            yield {'jobSummaryList': self._client.queue.get(kw['jobStatus'], [])}
        else:
            yield {'jobSummaryList':
                   self._client.children.get(kw['arrayJobId'], [])}


class _FakeBatch:
    def __init__(self, queue, children=None, describe=None):
        self.queue = queue
        self.children = children or {}
        self.describe = describe or {}

    def get_paginator(self, _op):
        return _FakePaginator(self)

    def describe_jobs(self, jobs):
        return {'jobs': [self.describe[j] for j in jobs if j in self.describe]}


class _FakeLogs:
    def get_log_events(self, **_kw):
        return {'events': [{'message': 'Traceback (most recent call last):'},
                           {'message': 'ValueError: bad shape'}]}


def _cfg():
    return AWSConfig(s3_bucket='b', s3_prefix='p', region='us-east-1',
                     job_queue='glow-job-queue', job_definition='glow-jobdef')


def test_cmd_status_expands_array_failures(monkeypatch, capsys):
    """A failed array parent expands into per-child exit codes + tags + tail."""
    parent = {'jobId': 'p1', 'jobName': 'glow-run-cafaf917',
              'createdAt': 1000, 'startedAt': 2000, 'stoppedAt': 136000,
              'arrayProperties': {'size': 8, 'statusSummary': {
                  'SUCCEEDED': 6, 'FAILED': 2, 'RUNNING': 0}}}
    child2 = {'jobId': 'p1:2', 'arrayProperties': {'index': 2},
              'container': {'exitCode': 137,
                            'reason': 'OutOfMemoryError: Container killed'}}
    child5 = {'jobId': 'p1:5', 'arrayProperties': {'index': 5},
              'container': {'exitCode': 1, 'reason': 'boom'}}
    empty = {s: [] for s in infra.ACTIVE_STATES + infra.TERMINAL_STATES}
    batch = _FakeBatch(
        queue={**empty, 'FAILED': [parent]},
        children={'p1': [child2, child5]},
        describe={'p1:2': {**child2, 'container': {
                      **child2['container'], 'logStreamName': 'glow/d/abc'}},
                  'p1:5': {**child5, 'container': {
                      **child5['container'], 'logStreamName': 'glow/d/def'}}})
    monkeypatch.setattr(
        infra.boto3, 'client',
        lambda svc, **_kw: _FakeLogs() if svc == 'logs' else batch)

    args = types.SimpleNamespace(label=None, limit=5, logs=True)
    infra.cmd_status(args, _cfg())
    out = capsys.readouterr().out

    assert 'run-cafaf917' in out
    assert '6✓ 2✗' in out                 # array progress from statusSummary
    assert 'child[2]' in out and 'OOM' in out
    assert 'child[5]' in out and 'exit=1' in out
    assert 'ValueError: bad shape' in out  # CloudWatch log tail
    assert 'tags: 1 OOM · 1 err' in out
    assert 'ran 2m14s' in out              # stoppedAt - startedAt


def test_cmd_status_active_progress_no_drill(monkeypatch, capsys):
    """Active runs show progress/timing; no failures means no drill calls."""
    running = {'jobId': 'p2', 'jobName': 'glow-run-aaaa1111',
               'status': 'RUNNING', 'createdAt': 1000, 'startedAt': 3000,
               'arrayProperties': {'size': 4, 'statusSummary': {
                   'SUCCEEDED': 1, 'FAILED': 0, 'RUNNING': 3}}}
    empty = {s: [] for s in infra.ACTIVE_STATES + infra.TERMINAL_STATES}
    batch = _FakeBatch(queue={**empty, 'RUNNING': [running]})
    monkeypatch.setattr(infra.boto3, 'client', lambda svc, **_kw: batch)

    args = types.SimpleNamespace(label=None, limit=5, logs=False)
    infra.cmd_status(args, _cfg())
    out = capsys.readouterr().out

    assert '[status] active (1 run):' in out
    assert 'run-aaaa1111' in out and '1✓ 0✗ 3⋯' in out
    assert '[status] failures' not in out
