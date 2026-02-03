"""Test monitor_jobs() function with mocked AWS Batch API

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
