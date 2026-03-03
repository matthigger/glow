"""Test monitor_jobs() and failure resubmission with mocked AWS Batch API.

Tests the monitor_jobs functionality without requiring AWS credentials
or actual job execution. All tests use jobs that are already complete
so monitor_jobs exits immediately after one iteration.
"""

from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest

try:
    from glow.aws.aws_batch import CloudConfig, AWSBatchRunner
except ImportError:
    pytest.skip("boto3 not available", allow_module_level=True)


def _make_runner(**overrides):
    """Create an AWSBatchRunner with sensible test defaults."""
    defaults = dict(
        s3_bucket='test-bucket',
        s3_prefix='test/prefix',
        job_queue='test-queue',
        job_definition='test-job-def',
        region='us-east-1',
        vcpus=2,
    )
    defaults.update(overrides)
    return AWSBatchRunner(CloudConfig(**defaults))


def test_monitor_jobs_basic():
    """Test monitor_jobs with mocked AWS Batch responses - all jobs complete immediately"""
    
    cloud_config = CloudConfig(
        s3_bucket='test-bucket',
        s3_prefix='test/prefix',
        job_queue='test-queue',
        job_definition='test-job-def',
        region='us-east-1',
        vcpus=2
    )
    
    runner = AWSBatchRunner(cloud_config)
    job_ids = ['job-1', 'job-2']
    
    # All jobs already complete - function exits immediately
    mock_jobs = [
        {
            'jobId': 'job-1',
            'jobName': 'test_job_1',
            'status': 'SUCCEEDED',
            'startedAt': datetime(2024, 1, 1, 10, 0, 0),
            'stoppedAt': datetime(2024, 1, 1, 10, 5, 0),
        },
        {
            'jobId': 'job-2',
            'jobName': 'test_job_2',
            'status': 'SUCCEEDED',
            'startedAt': datetime(2024, 1, 1, 10, 0, 0),
            'stoppedAt': datetime(2024, 1, 1, 10, 3, 0),
        },
    ]
    
    with patch.object(runner.batch, 'describe_jobs', return_value={'jobs': mock_jobs}):
        with patch('time.sleep'):  # Skip sleep
            with patch('glow.aws.aws_batch.tqdm') as mock_tqdm:
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar
                
                runner.monitor_jobs(job_ids, poll_interval=0.01, job_info_map={})
                
                # Verify tqdm was used
                assert mock_tqdm.called
                assert mock_pbar.close.called  # Should close when done


def test_monitor_jobs_timestamp_handling():
    """Test that monitor_jobs handles both datetime objects and Unix timestamps"""
    
    cloud_config = CloudConfig(
        s3_bucket='test-bucket',
        s3_prefix='test/prefix',
        job_queue='test-queue',
        job_definition='test-job-def',
        region='us-east-1',
        vcpus=2
    )
    
    runner = AWSBatchRunner(cloud_config)
    job_ids = ['job-1']
    
    # Test with Unix timestamp (int) - milliseconds since epoch
    unix_timestamp_ms = int(datetime(2024, 1, 1, 10, 0, 0).timestamp() * 1000)
    
    # Job already complete so function exits immediately
    mock_jobs = [{
        'jobId': 'job-1',
        'jobName': 'test_job_1',
        'status': 'SUCCEEDED',
        'startedAt': unix_timestamp_ms,
        'stoppedAt': unix_timestamp_ms + 300000,  # 5 minutes later
    }]
    
    with patch.object(runner.batch, 'describe_jobs', return_value={'jobs': mock_jobs}):
        with patch('time.sleep'):
            with patch('glow.aws.aws_batch.tqdm') as mock_tqdm:
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar
                
                # Should not raise an error
                runner.monitor_jobs(job_ids, poll_interval=0.01, job_info_map={})
                assert mock_tqdm.called


def test_monitor_jobs_vcpu_hours_calculation():
    """Test that vCPU-hours are calculated correctly"""
    
    cloud_config = CloudConfig(
        s3_bucket='test-bucket',
        s3_prefix='test/prefix',
        job_queue='test-queue',
        job_definition='test-job-def',
        region='us-east-1',
        vcpus=4  # 4 vCPUs per job
    )
    
    runner = AWSBatchRunner(cloud_config)
    job_ids = ['job-1']
    
    # Job already complete - 1 hour runtime
    start_time = datetime(2024, 1, 1, 10, 0, 0)
    stop_time = datetime(2024, 1, 1, 11, 0, 0)
    
    mock_jobs = [{
        'jobId': 'job-1',
        'jobName': 'test_job_1',
        'status': 'SUCCEEDED',
        'startedAt': start_time,
        'stoppedAt': stop_time,
    }]
    
    with patch.object(runner.batch, 'describe_jobs', return_value={'jobs': mock_jobs}):
        with patch('time.sleep'):
            with patch('glow.aws.aws_batch.tqdm') as mock_tqdm:
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar
                
                # Capture printed output
                with patch('builtins.print') as mock_print:
                    runner.monitor_jobs(job_ids, poll_interval=0.01, job_info_map={})
                    
                    # Check that vCPU-hours was printed (should be 4.0 for 1 hour with 4 vCPUs)
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    vcpu_printed = any('vCPU-hours' in str(call) for call in print_calls)
                    assert vcpu_printed, "vCPU-hours should be printed in status updates"


# ---------------------------------------------------------------------------
# Spot termination detection
# ---------------------------------------------------------------------------

class TestIsSpotTermination:
    """Test _is_spot_termination() static method."""

    def test_detects_spot_termination(self):
        job = {'statusReason': 'Host EC2 (instance i-abc123) terminated.'}
        assert AWSBatchRunner._is_spot_termination(job) is True

    def test_rejects_oom(self):
        job = {
            'statusReason': 'OutOfMemoryError',
            'container': {'exitCode': 137},
        }
        assert AWSBatchRunner._is_spot_termination(job) is False

    def test_rejects_generic_failure(self):
        job = {'statusReason': 'Essential container in task exited'}
        assert AWSBatchRunner._is_spot_termination(job) is False

    def test_handles_none_reason(self):
        job = {'statusReason': None}
        assert AWSBatchRunner._is_spot_termination(job) is False

    def test_handles_missing_reason(self):
        job = {}
        assert AWSBatchRunner._is_spot_termination(job) is False


# ---------------------------------------------------------------------------
# Spot resubmission logic
# ---------------------------------------------------------------------------

class TestResubmitSpotTermination:
    """Test _resubmit_failed_jobs() for spot-terminated jobs."""

    def test_resubmit_spot_same_memory(self):
        """Spot resubmission should use the same memory tier, not the next."""
        runner = _make_runner()
        runner.batch.submit_job = MagicMock(
            return_value={'jobId': 'new-job-1'})

        failed = [{
            'jobId': 'job-1',
            'jobName': 'exp_00',
            'statusReason': 'Host EC2 (instance i-abc) terminated.',
            'container': {'command': ['python', 'worker.py']},
        }]

        resubmitted, reasons = runner._resubmit_failed_jobs(failed, {})

        assert len(resubmitted) == 1
        assert reasons == {'job-1': 'spot'}

        # verify same memory (first tier, since no prior OOM escalation)
        call_kwargs = runner.batch.submit_job.call_args
        resources = call_kwargs.kwargs['containerOverrides']['resourceRequirements']
        mem_value = next(r['value'] for r in resources if r['type'] == 'MEMORY')
        assert mem_value == '2000', 'spot retry should keep the same memory tier'

    def test_spot_retry_limit(self):
        """Jobs should not be resubmitted beyond max_spot_retries."""
        runner = _make_runner(max_spot_retries=2)
        runner.batch.submit_job = MagicMock(
            return_value={'jobId': 'new-job'})

        base_job = {
            'jobName': 'exp_00',
            'statusReason': 'Host EC2 (instance i-abc) terminated.',
            'container': {'command': ['python', 'worker.py']},
        }

        # exhaust retry budget
        for i in range(2):
            failed = [{**base_job, 'jobId': f'job-{i}'}]
            runner._resubmit_failed_jobs(failed, {})

        # third attempt should be skipped
        failed = [{**base_job, 'jobId': 'job-2'}]
        resubmitted, reasons = runner._resubmit_failed_jobs(failed, {})
        assert len(resubmitted) == 0
        assert len(reasons) == 0

    def test_spot_propagates_job_info(self):
        """job_info_map should be copied for the new job ID."""
        runner = _make_runner()
        runner.batch.submit_job = MagicMock(
            return_value={'jobId': 'new-job'})

        info_map = {'job-1': {'run_id': 'r1', 'exp_idx': 0}}
        failed = [{
            'jobId': 'job-1',
            'jobName': 'exp_00',
            'statusReason': 'Host EC2 (instance i-abc) terminated.',
            'container': {'command': ['python', 'worker.py']},
        }]

        runner._resubmit_failed_jobs(failed, info_map)
        assert 'new-job' in info_map
        assert info_map['new-job'] == info_map['job-1']


class TestMonitorWaitsForRetries:
    """Verify monitor_jobs does not exit before resubmitted jobs finish."""

    def test_monitor_continues_after_all_jobs_fail_and_resubmit(self):
        """Regression: all jobs OOM on first poll, retries succeed on second."""
        runner = _make_runner()
        call_count = [0]

        def mock_submit(**kw):
            call_count[0] += 1
            return {'jobId': f'retry-{call_count[0]}'}

        runner.batch.submit_job = MagicMock(side_effect=mock_submit)

        poll_num = [0]

        def describe_side_effect(jobs):
            poll_num[0] += 1
            result = []
            for jid in jobs:
                if jid.startswith('retry-'):
                    if poll_num[0] >= 2:
                        result.append({
                            'jobId': jid,
                            'jobName': f'name_{jid}',
                            'status': 'SUCCEEDED',
                            'startedAt': datetime(2024, 1, 1, 10, 2),
                            'stoppedAt': datetime(2024, 1, 1, 10, 5),
                        })
                    else:
                        result.append({
                            'jobId': jid,
                            'jobName': f'name_{jid}',
                            'status': 'RUNNING',
                            'startedAt': datetime(2024, 1, 1, 10, 2),
                        })
                else:
                    result.append({
                        'jobId': jid,
                        'jobName': f'name_{jid}',
                        'status': 'FAILED',
                        'statusReason': 'OutOfMemoryError',
                        'container': {
                            'exitCode': 137,
                            'command': ['python', 'worker.py'],
                        },
                        'startedAt': datetime(2024, 1, 1, 10, 0),
                        'stoppedAt': datetime(2024, 1, 1, 10, 1),
                    })
            return {'jobs': result}

        runner.batch.describe_jobs = MagicMock(side_effect=describe_side_effect)

        with patch('time.sleep'):
            with patch('glow.aws.aws_batch.tqdm') as mock_tqdm:
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar
                with patch('builtins.print') as mock_print:
                    runner.monitor_jobs(
                        ['job-1', 'job-2'],
                        poll_interval=0.01,
                        job_info_map={})

        assert poll_num[0] >= 2, (
            'monitor exited after first poll without waiting for retries')

        printed = ' '.join(str(c) for c in mock_print.call_args_list)
        assert 'succeeded: 2' in printed


class TestResubmitMixed:
    """Test _resubmit_failed_jobs with a mix of OOM and spot failures."""

    def test_oom_and_spot_together(self):
        runner = _make_runner()
        call_count = [0]

        def mock_submit(**kw):
            call_count[0] += 1
            return {'jobId': f'new-{call_count[0]}'}

        runner.batch.submit_job = MagicMock(side_effect=mock_submit)

        failed = [
            {
                'jobId': 'oom-job',
                'jobName': 'exp_oom',
                'statusReason': 'OutOfMemoryError',
                'container': {'exitCode': 137,
                              'command': ['python', 'worker.py']},
            },
            {
                'jobId': 'spot-job',
                'jobName': 'exp_spot',
                'statusReason': 'Host EC2 (instance i-xyz) terminated.',
                'container': {'command': ['python', 'worker.py']},
            },
        ]

        resubmitted, reasons = runner._resubmit_failed_jobs(failed, {})

        assert len(resubmitted) == 2
        assert reasons['oom-job'] == 'oom'
        assert reasons['spot-job'] == 'spot'

    def test_monitor_prints_correct_messages(self):
        """monitor_jobs should print distinct messages for OOM vs spot."""
        runner = _make_runner()

        mock_jobs = [
            {
                'jobId': 'oom-job',
                'jobName': 'exp_oom',
                'status': 'FAILED',
                'statusReason': 'OutOfMemoryError',
                'container': {'exitCode': 137,
                              'command': ['python', 'worker.py']},
                'startedAt': datetime(2024, 1, 1, 10, 0),
                'stoppedAt': datetime(2024, 1, 1, 10, 1),
            },
            {
                'jobId': 'spot-job',
                'jobName': 'exp_spot',
                'status': 'FAILED',
                'statusReason': 'Host EC2 (instance i-xyz) terminated.',
                'container': {'command': ['python', 'worker.py']},
                'startedAt': datetime(2024, 1, 1, 10, 0),
                'stoppedAt': datetime(2024, 1, 1, 10, 1),
            },
        ]

        call_count = [0]

        def mock_submit(**kw):
            call_count[0] += 1
            return {'jobId': f'retry-{call_count[0]}'}

        runner.batch.submit_job = MagicMock(side_effect=mock_submit)

        # first call returns failures, second returns the retries as succeeded
        def describe_side_effect(jobs):
            ids = set(jobs)
            result = []
            for j in mock_jobs:
                if j['jobId'] in ids:
                    result.append(j)
            # any retry jobs are immediately succeeded
            for jid in ids:
                if jid.startswith('retry-'):
                    result.append({
                        'jobId': jid,
                        'jobName': f'retry_{jid}',
                        'status': 'SUCCEEDED',
                        'startedAt': datetime(2024, 1, 1, 10, 2),
                        'stoppedAt': datetime(2024, 1, 1, 10, 3),
                    })
            return {'jobs': result}

        runner.batch.describe_jobs = MagicMock(side_effect=describe_side_effect)

        with patch('time.sleep'):
            with patch('glow.aws.aws_batch.tqdm') as mock_tqdm:
                mock_pbar = MagicMock()
                mock_tqdm.return_value = mock_pbar
                with patch('builtins.print') as mock_print:
                    runner.monitor_jobs(
                        ['oom-job', 'spot-job'],
                        poll_interval=0.01,
                        job_info_map={})

                    printed = ' '.join(
                        str(c) for c in mock_print.call_args_list)
                    assert 'OOM' in printed
                    assert 'SPOT' in printed
                    assert 'instance reclaimed' in printed


# ---------------------------------------------------------------------------
# OOM detection
# ---------------------------------------------------------------------------

class TestIsOomFailure:
    """Test _is_oom_failure() static method."""

    def test_exit_code_137(self):
        assert AWSBatchRunner._is_oom_failure({'container': {'exitCode': 137}})

    def test_exit_code_134(self):
        assert AWSBatchRunner._is_oom_failure({'container': {'exitCode': 134}})

    def test_oom_in_status_reason(self):
        assert AWSBatchRunner._is_oom_failure({
            'statusReason': 'OutOfMemoryError: Container killed',
            'container': {}})

    def test_oom_in_container_reason(self):
        assert AWSBatchRunner._is_oom_failure({
            'container': {'reason': 'OOM: killed process'}})

    def test_memory_in_container_reason(self):
        assert AWSBatchRunner._is_oom_failure({
            'container': {'reason': 'Insufficient memory'}})

    def test_memory_in_status_reason(self):
        assert AWSBatchRunner._is_oom_failure({
            'statusReason': 'Container ran out of memory',
            'container': {}})

    def test_normal_exit_code_1(self):
        assert not AWSBatchRunner._is_oom_failure({
            'statusReason': 'Essential container exited',
            'container': {'exitCode': 1, 'reason': 'task failed'}})

    def test_exit_code_0(self):
        assert not AWSBatchRunner._is_oom_failure({
            'container': {'exitCode': 0}})

    def test_empty_dict(self):
        assert not AWSBatchRunner._is_oom_failure({})

    def test_container_none(self):
        assert not AWSBatchRunner._is_oom_failure({'container': None})

    def test_status_reason_none(self):
        assert not AWSBatchRunner._is_oom_failure({
            'statusReason': None, 'container': {'exitCode': 2}})


# ---------------------------------------------------------------------------
# _base_job_name
# ---------------------------------------------------------------------------

class TestBaseJobName:

    def test_strips_retry1(self):
        assert AWSBatchRunner._base_job_name('glow_abc_exp000001_retry1') == 'glow_abc_exp000001'

    def test_strips_retry2(self):
        assert AWSBatchRunner._base_job_name('glow_abc_exp000001_retry2') == 'glow_abc_exp000001'

    def test_no_retry_unchanged(self):
        assert AWSBatchRunner._base_job_name('glow_abc_exp000001') == 'glow_abc_exp000001'

    def test_parenthesized_name_unchanged(self):
        assert AWSBatchRunner._base_job_name('job_with_parens(perm)') == 'job_with_parens(perm)'


# ---------------------------------------------------------------------------
# OOM resubmission logic
# ---------------------------------------------------------------------------

class TestResubmitOom:
    """Test _resubmit_failed_jobs() for OOM failures."""

    def test_no_memory_tiers(self):
        runner = _make_runner(oom_memory_mb_tiers=[])
        resubmitted, reasons = runner._resubmit_failed_jobs(
            [{'jobId': 'j1', 'jobName': 'exp_00',
              'container': {'exitCode': 137, 'command': ['--test']}}], {})
        assert resubmitted == []
        assert len(reasons) == 0

    def test_non_oom_skipped(self):
        runner = _make_runner()
        runner.batch = MagicMock()
        resubmitted, reasons = runner._resubmit_failed_jobs(
            [{'jobId': 'j1', 'jobName': 'exp_00',
              'statusReason': 'Essential container exited',
              'container': {'exitCode': 1, 'reason': 'task failed',
                            'command': ['--test']}}], {})
        assert resubmitted == []
        runner.batch.submit_job.assert_not_called()

    def test_oom_resubmits_with_next_tier(self):
        runner = _make_runner()
        runner.batch = MagicMock()
        runner.batch.submit_job.return_value = {'jobId': 'new-j1'}

        info_map = {'j1': {'run_id': 'r1', 'exp_idx': 0}}
        resubmitted, reasons = runner._resubmit_failed_jobs(
            [{'jobId': 'j1', 'jobName': 'exp_00',
              'container': {'exitCode': 137,
                            'command': ['--s3-bucket', 'b']}}],
            info_map)

        assert resubmitted == ['new-j1']
        assert reasons == {'j1': 'oom'}
        call_kw = runner.batch.submit_job.call_args.kwargs
        assert call_kw['jobName'] == 'exp_00_retry1'
        resources = call_kw['containerOverrides']['resourceRequirements']
        mem = next(r['value'] for r in resources if r['type'] == 'MEMORY')
        assert mem == '4000'
        assert 'new-j1' in info_map

    def test_missing_command_skipped(self):
        runner = _make_runner()
        runner.batch = MagicMock()
        resubmitted, _ = runner._resubmit_failed_jobs(
            [{'jobId': 'j1', 'jobName': 'exp_00',
              'container': {'exitCode': 137}}], {})
        assert resubmitted == []
        runner.batch.submit_job.assert_not_called()

    def test_mixed_oom_and_non_oom(self):
        runner = _make_runner()
        runner.batch = MagicMock()
        runner.batch.submit_job.return_value = {'jobId': 'new-j4'}

        failed = [
            {'jobId': 'j4', 'jobName': 'exp_04',
             'container': {'exitCode': 137, 'command': ['--test']}},
            {'jobId': 'j5', 'jobName': 'exp_05',
             'statusReason': 'timeout',
             'container': {'exitCode': 1, 'command': ['--test']}},
        ]
        resubmitted, reasons = runner._resubmit_failed_jobs(failed, {})
        assert len(resubmitted) == 1
        assert reasons.get('j4') == 'oom'

    def test_tier_escalation(self):
        """2 GB -> 4 GB -> 8 GB -> 16 GB -> exhausted."""
        runner = _make_runner()
        runner.batch = MagicMock()

        expected_tiers = [
            ('j1', 'exp_06', 1, '4000'),
            ('r1', 'exp_06_retry1', 2, '8000'),
            ('r2', 'exp_06_retry2', 3, '16000'),
        ]
        for job_id, job_name, expected_idx, expected_mem in expected_tiers:
            runner.batch.submit_job.return_value = {'jobId': f'new-{job_id}'}
            runner._resubmit_failed_jobs(
                [{'jobId': job_id, 'jobName': job_name,
                  'container': {'exitCode': 137, 'command': ['--test']}}], {})
            assert runner._job_memory_tier_index.get('exp_06') == expected_idx
            mem = next(
                r['value']
                for r in runner.batch.submit_job.call_args.kwargs[
                    'containerOverrides']['resourceRequirements']
                if r['type'] == 'MEMORY')
            assert mem == expected_mem

        # fourth attempt: exhausted
        runner.batch.submit_job.reset_mock()
        resubmitted, reasons = runner._resubmit_failed_jobs(
            [{'jobId': 'r3', 'jobName': 'exp_06_retry3',
              'container': {'exitCode': 137, 'command': ['--test']}}], {})
        assert resubmitted == []
        runner.batch.submit_job.assert_not_called()
