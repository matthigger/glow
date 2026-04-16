"""Helpers shared by AWS-gated tests.

Lean: only what test_aws_equivalence.py needs. No debug printouts.
"""
import configparser
import os
from pathlib import Path


def load_aws_config():
    """Load AWS config from repo-root .glow_aws_config or environment.

    Returns:
        dict with s3_bucket, job_queue, job_definition, region,
        account_id — or None if no source is configured.
    """
    # repo root is two parents up from this file (src/test/_aws_helpers.py)
    config_file = Path(__file__).parent.parent / '.glow_aws_config'

    if config_file.exists():
        parser = configparser.ConfigParser()
        parser.read(config_file)
        if 'aws' in parser:
            return {
                's3_bucket': parser['aws'].get('s3_bucket'),
                'job_queue': parser['aws'].get('job_queue'),
                'job_definition': parser['aws'].get('job_definition'),
                'region': parser['aws'].get('region', 'us-east-1'),
                'account_id': parser['aws'].get('account_id'),
            }

    # env-var fallback
    s3_bucket = os.getenv('GLOW_S3_BUCKET')
    job_queue = os.getenv('GLOW_JOB_QUEUE')
    job_definition = os.getenv('GLOW_JOB_DEFINITION')

    if s3_bucket or job_queue or job_definition:
        return {
            's3_bucket': s3_bucket,
            'job_queue': job_queue,
            'job_definition': job_definition,
            'region': os.getenv('AWS_REGION', 'us-east-1'),
            'account_id': None,
        }

    return None
