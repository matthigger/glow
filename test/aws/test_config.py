"""AWSConfig: dataclass roundtrip + JSON file IO."""

import json

import pytest

from glow.aws.config import AWSConfig


def _make():
    return AWSConfig(
        s3_bucket='glow-experiments-test',
        s3_prefix='unit',
        job_queue='glow-q',
        job_definition='glow-def')


def test_defaults():
    cfg = _make()
    assert cfg.region == 'us-east-1'
    assert cfg.vcpus == 1
    assert cfg.memory_mb_tiers == [2000, 4000, 8000]
    assert cfg.max_concurrent == 4000
    assert cfg.timeout_minutes == 60
    assert cfg.retry_attempts == 3
    assert cfg.poll_seconds == 10


def test_dict_roundtrip():
    cfg = _make()
    assert AWSConfig.from_dict(cfg.to_dict()) == cfg


def test_file_roundtrip(tmp_path):
    cfg = _make()
    path = tmp_path / 'cfg.json'
    cfg.to_file(path)
    assert AWSConfig.from_file(path) == cfg
    # readable as plain JSON
    assert json.loads(path.read_text())['s3_bucket'] == 'glow-experiments-test'


def test_unknown_key_rejected():
    bad = _make().to_dict()
    bad['mystery_field'] = 1
    with pytest.raises(ValueError, match='unknown AWSConfig keys'):
        AWSConfig.from_dict(bad)


def test_tiers_distinct_per_instance():
    """default_factory must not share a list across instances."""
    a = _make()
    b = _make()
    a.memory_mb_tiers.append(16000)
    assert b.memory_mb_tiers == [2000, 4000, 8000]
