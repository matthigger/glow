"""AWS Batch worker entrypoint.

Invoked inside the container as::

    python -m glow.aws.worker s3://bucket/prefix/jobs/<run_id>/manifest.pkl

Reads ``AWS_BATCH_JOB_ARRAY_INDEX`` to pick its slot in the manifest,
downloads the matching ``job.pkl``, runs ``run_fnc(**trial)`` (where
``trial['ds']`` is either a ``DataSourceS3`` that downloads its exp,
or a deterministic ``DataSource`` rebuilt locally from its seed), and
uploads ``result.pkl`` next to the job pickle.

Non-zero exit on any exception so AWS Batch marks the child FAILED.
"""

import os
import sys

import boto3
import cloudpickle

from glow.aws.datasource import _parse_s3_uri


def _result_key_for(job_key: str) -> str:
    """``.../jobs/<trial_hash>/job.pkl`` → ``.../jobs/<trial_hash>/result.pkl``."""
    return job_key.rsplit('/', 1)[0] + '/result.pkl'


def _job_key_for(manifest_key: str, trial_hash: str) -> str:
    """``.../jobs/<run_id>/manifest.pkl`` → ``.../jobs/<trial_hash>/job.pkl``."""
    return f'{manifest_key.rsplit("/", 2)[0]}/{trial_hash}/job.pkl'


def main(manifest_uri: str) -> None:
    bucket, manifest_key = _parse_s3_uri(manifest_uri)
    s3 = boto3.client('s3')

    manifest = cloudpickle.loads(
        s3.get_object(Bucket=bucket, Key=manifest_key)['Body'].read())

    idx = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', '0'))
    trial_hash = manifest[idx]
    print(f'[worker] manifest={manifest_uri} idx={idx} '
          f'trial_hash={trial_hash}', flush=True)

    job_key = _job_key_for(manifest_key, trial_hash)
    run_fnc, trial = cloudpickle.loads(
        s3.get_object(Bucket=bucket, Key=job_key)['Body'].read())

    result = run_fnc(**trial)

    result_key = _result_key_for(job_key)
    s3.put_object(Bucket=bucket, Key=result_key,
                  Body=cloudpickle.dumps(result))
    print(f'[worker] uploaded s3://{bucket}/{result_key}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python -m glow.aws.worker <manifest_s3_uri>',
              file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
