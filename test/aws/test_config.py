"""AWSConfig JSON round-trip + s3_key prefix handling."""

import pytest

from glow._extra.aws.config import AWSConfig, s3_key


def test_round_trip(tmp_path):
    cfg = AWSConfig(s3_bucket='b', job_queue='q', job_definition='d',
                    s3_prefix='glow', region='us-west-2', vcpus=2)
    path = tmp_path / 'aws_config.json'
    cfg.to_file(str(path))
    back = AWSConfig.from_file(str(path))
    assert back == cfg


def test_defaults():
    cfg = AWSConfig(s3_bucket='b', job_queue='q', job_definition='d')
    assert cfg.s3_prefix == ''
    assert cfg.memory_mb_tiers == [4000, 8000, 16000]
    # a single data cell runs its whole subtree serially, so the per-attempt
    # ceiling is generous (see config.py)
    assert cfg.timeout_minutes >= 600


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match='unknown AWSConfig keys'):
        AWSConfig.from_dict({'s3_bucket': 'b', 'job_queue': 'q',
                             'job_definition': 'd', 'bogus': 1})


def test_s3_key_tolerates_empty_prefix():
    assert s3_key('', 'records', 'x.json') == 'records/x.json'
    assert s3_key('glow', 'records', 'x.json') == 'glow/records/x.json'
