"""Tests for glow.aws.download (deferred result download CLI)."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from glow.aws.download import save_job_info, download_run, JOBS_DIR


@pytest.fixture(autouse=True)
def tmp_jobs_dir(tmp_path, monkeypatch):
    """Redirect JOBS_DIR to a temp directory for all tests."""
    jobs = tmp_path / 'jobs'
    jobs.mkdir()
    monkeypatch.setattr('glow.aws.download.JOBS_DIR', jobs)
    return jobs


def _make_cloud_config():
    return MagicMock(
        s3_bucket='test-bucket',
        s3_prefix='test-prefix',
        region='us-east-1',
    )


class TestSaveJobInfo:

    def test_saves_json(self, tmp_jobs_dir):
        info = [{
            'run_id': 'label_abc123',
            'label': 'label',
            'folder': '/tmp/results/label',
            'job_ids': ['j1', 'j2'],
            'runner': MagicMock(),
        }]
        paths = save_job_info(info, _make_cloud_config())
        assert len(paths) == 1
        data = json.loads(paths[0].read_text())
        assert data['run_id'] == 'label_abc123'
        assert data['s3_bucket'] == 'test-bucket'
        assert len(data['job_ids']) == 2

    def test_skips_cached(self, tmp_jobs_dir):
        info = [{'run_id': None, 'label': 'x', 'folder': '/tmp',
                 'job_ids': [], 'runner': None}]
        paths = save_job_info(info, _make_cloud_config())
        assert paths == []

    def test_multiple_runs(self, tmp_jobs_dir):
        info = [
            {'run_id': 'a_1', 'label': 'a', 'folder': '/tmp/a',
             'job_ids': ['j1'], 'runner': MagicMock()},
            {'run_id': 'b_2', 'label': 'b', 'folder': '/tmp/b',
             'job_ids': ['j2'], 'runner': MagicMock()},
        ]
        paths = save_job_info(info, _make_cloud_config())
        assert len(paths) == 2


class TestDownloadRun:

    def test_missing_run_id(self):
        with pytest.raises(FileNotFoundError, match='no_such_run'):
            download_run('no_such_run')

    def test_downloads_and_removes_json(self, tmp_jobs_dir):
        # create a saved run
        data = {
            'run_id': 'test_run_1',
            'label': 'test',
            'folder': '/tmp/results/test',
            'job_ids': ['j1'],
            's3_bucket': 'b',
            's3_prefix': 'p',
            'region': 'us-east-1',
        }
        path = tmp_jobs_dir / 'test_run_1.json'
        path.write_text(json.dumps(data))

        with patch('glow.aws.aws_batch.AWSBatchRunner') as MockRunner:
            mock_instance = MockRunner.return_value
            mock_instance.download_experiment_results = MagicMock()
            download_run('test_run_1')
            mock_instance.download_experiment_results.assert_called_once_with(
                'test_run_1', Path('/tmp/results/test'))

        # json removed after download
        assert not path.exists()
