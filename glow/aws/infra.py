"""Idempotent AWS Batch infrastructure CLI.

Subcommands::

    python -m glow.aws.infra setup     [--image-tag IMG] [--config PATH]
    python -m glow.aws.infra teardown  [--yes] [--delete-bucket] [--config PATH]
    python -m glow.aws.infra status    [--label LABEL]              [--config PATH]
    python -m glow.aws.infra clean     [--jobs] [--datasource] [--yes] [--config PATH]
    python -m glow.aws.infra pause                                 [--config PATH]
    python -m glow.aws.infra resume                                [--config PATH]

Reads ``AWSConfig`` from ``.glow_aws_config`` (or ``--config``) for
bucket / queue / job-definition / region names; the rest of the
provisioning detail (instance types, allocation strategy, VPC lookup)
lives in this file.

Assumes the IAM bootstrap from the README is done:
``ecsInstanceRole``, ``aws-ec2-spot-fleet-tagging-role``, and the
Batch service-linked role.  The IAM user running this only needs S3 /
Batch / ECR permissions (no IAM create), so role provisioning stays a
one-time admin step rather than something this CLI does.
"""

import argparse
import json
import sys
import time
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError

from glow.aws.config import AWSConfig, DEFAULT_CONFIG_PATH


# ---------- compute-environment constants -----------------------------------
COMPUTE_ENV_NAME = 'glow-compute-env-spot'
ALLOC_STRATEGY = 'SPOT_PRICE_CAPACITY_OPTIMIZED'

# Instance pool: 6/7 series c/m/r families, large → 12xlarge.  Picked
# under SPOT_PRICE_CAPACITY_OPTIMIZED so Batch picks the cheapest pool
# with available Spot capacity.  Carried verbatim from the old setup
# (bench_instance_types.py 2026-04-28 sweep: 5-series 2x slower per core,
# dropped from the list).
INSTANCE_TYPES = [
    'c6i.large', 'c6i.xlarge', 'c6i.2xlarge', 'c6i.4xlarge',
    'c6i.8xlarge', 'c6i.12xlarge',
    'c6a.large', 'c6a.xlarge', 'c6a.2xlarge', 'c6a.4xlarge',
    'c6a.8xlarge', 'c6a.12xlarge',
    'c7i.large', 'c7i.xlarge', 'c7i.2xlarge', 'c7i.4xlarge',
    'c7i.8xlarge', 'c7i.12xlarge',
    'c7a.large', 'c7a.xlarge', 'c7a.2xlarge', 'c7a.4xlarge',
    'c7a.8xlarge', 'c7a.12xlarge',
    'm6i.large', 'm6i.xlarge', 'm6i.2xlarge', 'm6i.4xlarge', 'm6i.8xlarge',
    'm6a.large', 'm6a.xlarge', 'm6a.2xlarge', 'm6a.4xlarge', 'm6a.8xlarge',
    'm7i.large', 'm7i.xlarge', 'm7i.2xlarge', 'm7i.4xlarge', 'm7i.8xlarge',
    'm7a.large', 'm7a.xlarge', 'm7a.2xlarge', 'm7a.4xlarge', 'm7a.8xlarge',
    'r6i.large', 'r6i.xlarge', 'r6i.2xlarge', 'r6i.4xlarge',
    'r6a.large', 'r6a.xlarge', 'r6a.2xlarge', 'r6a.4xlarge',
    'r7i.large', 'r7i.xlarge', 'r7i.2xlarge', 'r7i.4xlarge',
    'r7a.large', 'r7a.xlarge', 'r7a.2xlarge', 'r7a.4xlarge',
]

# IAM role names — assumed to already exist (see README).
ECS_INSTANCE_ROLE = 'ecsInstanceRole'
SPOT_FLEET_ROLE = 'aws-ec2-spot-fleet-tagging-role'

# ECR repository for the worker image.
ECR_REPO_NAME = 'glow-worker'


# ---------- setup -----------------------------------------------------------


def cmd_setup(args, cfg: AWSConfig):
    """Provision S3 + ECR + Batch resources (idempotent)."""
    region = cfg.region
    account_id = _account_id()

    print(f'[setup] region={region} account={account_id}')
    _setup_s3_bucket(cfg)

    image_uri = _resolve_image_uri(args, region, account_id)
    _setup_job_definition(cfg, image_uri=image_uri)
    ce_arn = _setup_compute_environment(cfg, account_id=account_id)
    _setup_job_queue(cfg, ce_arn=ce_arn)
    print('[setup] done.')


def _setup_s3_bucket(cfg: AWSConfig):
    s3 = boto3.client('s3', region_name=cfg.region)
    try:
        s3.head_bucket(Bucket=cfg.s3_bucket)
        print(f'  ✓ S3 bucket {cfg.s3_bucket} exists')
        return
    except ClientError as e:
        if e.response['Error']['Code'] not in ('404', '403', 'NoSuchBucket'):
            raise

    kwargs = {'Bucket': cfg.s3_bucket}
    if cfg.region != 'us-east-1':
        kwargs['CreateBucketConfiguration'] = {
            'LocationConstraint': cfg.region}
    s3.create_bucket(**kwargs)
    print(f'  ✓ created S3 bucket {cfg.s3_bucket}')


def _resolve_image_uri(args, region: str, account_id: str) -> str:
    """Return the ECR image URI Batch should run.

    With ``--image-tag <tag>``, the local image is pushed to ECR
    under that tag.  Without it, an existing ``glow-worker:latest`` in
    ECR is expected — looked up via ``describe_images``.
    """
    if args.image_tag:
        return _push_image_to_ecr(args.image_tag, region, account_id)

    ecr = boto3.client('ecr', region_name=region)
    try:
        ecr.describe_images(repositoryName=ECR_REPO_NAME,
                            imageIds=[{'imageTag': 'latest'}])
    except ClientError as e:
        msg = (f'No image in ECR {ECR_REPO_NAME}:latest. '
               f'Rerun with --image-tag <local_tag>, or push manually.')
        raise SystemExit(msg) from e
    uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/{ECR_REPO_NAME}:latest'
    print(f'  ✓ using existing ECR image {uri}')
    return uri


def _push_image_to_ecr(local_tag: str, region: str, account_id: str) -> str:
    """Tag + push a local Docker image to ECR.  Creates the repo if needed.

    Uses the ``docker`` CLI via ``subprocess`` rather than the Docker
    SDK so it works in any environment that already builds the image.
    """
    import shutil
    import subprocess

    if shutil.which('docker') is None:
        raise SystemExit('docker not found in PATH — install Docker first')

    ecr = boto3.client('ecr', region_name=region)
    try:
        ecr.describe_repositories(repositoryNames=[ECR_REPO_NAME])
    except ClientError as e:
        if e.response['Error']['Code'] == 'RepositoryNotFoundException':
            ecr.create_repository(repositoryName=ECR_REPO_NAME)
            print(f'  ✓ created ECR repo {ECR_REPO_NAME}')
        else:
            raise

    ecr_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/{ECR_REPO_NAME}:latest'
    token = ecr.get_authorization_token()['authorizationData'][0]
    import base64
    user, pw = base64.b64decode(token['authorizationToken']).decode().split(':', 1)
    proxy = token['proxyEndpoint']
    subprocess.run(
        ['docker', 'login', '-u', user, '--password-stdin', proxy],
        input=pw, text=True, check=True)
    subprocess.run(['docker', 'tag', local_tag, ecr_uri], check=True)
    subprocess.run(['docker', 'push', ecr_uri], check=True)
    print(f'  ✓ pushed {local_tag} → {ecr_uri}')
    return ecr_uri


def _setup_job_definition(cfg: AWSConfig, *, image_uri: str):
    batch = boto3.client('batch', region_name=cfg.region)
    account_id = _account_id()
    # ECS task roles — referenced; created out-of-band per README.
    exec_role = f'arn:aws:iam::{account_id}:role/GlowEcsTaskExecutionRole'
    task_role = f'arn:aws:iam::{account_id}:role/GlowEcsTaskRole'

    container = {
        'image': image_uri,
        'jobRoleArn': task_role,
        'executionRoleArn': exec_role,
        'resourceRequirements': [
            {'type': 'VCPU', 'value': str(cfg.vcpus)},
            {'type': 'MEMORY', 'value': str(cfg.memory_mb_tiers[0])},
        ],
        # driver overrides command per-submission
        'command': ['python', '-m', 'glow.aws.worker'],
        'environment': [
            {'name': 'AWS_DEFAULT_REGION', 'value': cfg.region},
        ],
    }
    response = batch.register_job_definition(
        jobDefinitionName=cfg.job_definition,
        type='container',
        platformCapabilities=['EC2'],
        containerProperties=container,
        retryStrategy={'attempts': cfg.retry_attempts},
        timeout={'attemptDurationSeconds': cfg.timeout_minutes * 60},
    )
    print(f'  ✓ registered job def {cfg.job_definition} '
          f'(revision {response["revision"]})')


def _setup_compute_environment(cfg: AWSConfig, *, account_id: str) -> str:
    batch = boto3.client('batch', region_name=cfg.region)
    ec2 = boto3.client('ec2', region_name=cfg.region)

    existing = batch.describe_compute_environments(
        computeEnvironments=[COMPUTE_ENV_NAME])['computeEnvironments']
    if existing:
        arn = existing[0]['computeEnvironmentArn']
        # Update maxvCpus to track config; everything else is immutable.
        try:
            batch.update_compute_environment(
                computeEnvironment=COMPUTE_ENV_NAME,
                computeResources={'maxvCpus': cfg.max_concurrent})
            print(f'  ✓ compute env {COMPUTE_ENV_NAME} exists '
                  f'(maxvCpus → {cfg.max_concurrent})')
        except ClientError as e:
            print(f'  ⚠ compute env update failed: {e}')
        return arn

    # Default VPC: first default subnet + default SG.
    subnets = ec2.describe_subnets(
        Filters=[{'Name': 'default-for-az', 'Values': ['true']}])['Subnets']
    if not subnets:
        raise SystemExit(
            'no default subnets found; pass --vpc-subnet / --vpc-sg manually')
    vpc_id = subnets[0]['VpcId']
    sg = ec2.describe_security_groups(
        Filters=[
            {'Name': 'vpc-id', 'Values': [vpc_id]},
            {'Name': 'group-name', 'Values': ['default']},
        ])['SecurityGroups'][0]

    service_role = (
        f'arn:aws:iam::{account_id}:role/aws-service-role/batch.amazonaws.com/'
        f'AWSServiceRoleForBatch')
    instance_profile = (
        f'arn:aws:iam::{account_id}:instance-profile/{ECS_INSTANCE_ROLE}')
    spot_fleet = (
        f'arn:aws:iam::{account_id}:role/{SPOT_FLEET_ROLE}')

    response = batch.create_compute_environment(
        computeEnvironmentName=COMPUTE_ENV_NAME,
        type='MANAGED',
        state='ENABLED',
        serviceRole=service_role,
        computeResources={
            'type': 'SPOT',
            'allocationStrategy': ALLOC_STRATEGY,
            'minvCpus': 0,
            'maxvCpus': cfg.max_concurrent,
            'desiredvCpus': 0,
            'instanceTypes': INSTANCE_TYPES,
            'subnets': [s['SubnetId'] for s in subnets],
            'securityGroupIds': [sg['GroupId']],
            'instanceRole': instance_profile,
            'spotIamFleetRole': spot_fleet,
        },
    )
    arn = response['computeEnvironmentArn']
    print(f'  ✓ created compute env {COMPUTE_ENV_NAME}')
    _wait_for_ce_valid(batch)
    return arn


def _wait_for_ce_valid(batch, timeout_s: int = 300):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ces = batch.describe_compute_environments(
            computeEnvironments=[COMPUTE_ENV_NAME])['computeEnvironments']
        if ces and ces[0]['status'] == 'VALID':
            return
        time.sleep(5)
    raise SystemExit(f'compute env did not reach VALID within {timeout_s}s')


def _setup_job_queue(cfg: AWSConfig, *, ce_arn: str):
    batch = boto3.client('batch', region_name=cfg.region)
    existing = batch.describe_job_queues(
        jobQueues=[cfg.job_queue])['jobQueues']
    if existing:
        print(f'  ✓ job queue {cfg.job_queue} exists')
        return
    batch.create_job_queue(
        jobQueueName=cfg.job_queue,
        state='ENABLED',
        priority=1,
        computeEnvironmentOrder=[
            {'order': 1, 'computeEnvironment': ce_arn}],
    )
    print(f'  ✓ created job queue {cfg.job_queue}')


# ---------- teardown --------------------------------------------------------


def cmd_teardown(args, cfg: AWSConfig):
    if not args.yes:
        print('--yes required to actually delete resources.')
        return
    print('[teardown] disabling + deleting Batch resources')

    batch = boto3.client('batch', region_name=cfg.region)
    _teardown_queue(batch, cfg)
    _teardown_compute_env(batch)
    _teardown_job_definition(batch, cfg)

    if args.delete_bucket:
        _teardown_bucket(cfg)
    print('[teardown] done.')


def _teardown_queue(batch, cfg: AWSConfig):
    queues = batch.describe_job_queues(
        jobQueues=[cfg.job_queue])['jobQueues']
    if not queues:
        return
    print(f'  disabling job queue {cfg.job_queue}')
    batch.update_job_queue(jobQueue=cfg.job_queue, state='DISABLED')
    _wait_for_queue_state(batch, cfg.job_queue, 'DISABLED')
    batch.delete_job_queue(jobQueue=cfg.job_queue)
    print(f'  ✓ deleted job queue {cfg.job_queue}')


def _wait_for_queue_state(batch, name: str, state: str, timeout_s=180):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        qs = batch.describe_job_queues(jobQueues=[name])['jobQueues']
        if not qs:
            return
        if qs[0]['state'] == state and qs[0]['status'] == 'VALID':
            return
        time.sleep(3)


def _teardown_compute_env(batch):
    ces = batch.describe_compute_environments(
        computeEnvironments=[COMPUTE_ENV_NAME])['computeEnvironments']
    if not ces:
        return
    print(f'  disabling compute env {COMPUTE_ENV_NAME}')
    batch.update_compute_environment(
        computeEnvironment=COMPUTE_ENV_NAME, state='DISABLED')
    deadline = time.time() + 300
    while time.time() < deadline:
        ces = batch.describe_compute_environments(
            computeEnvironments=[COMPUTE_ENV_NAME])['computeEnvironments']
        if (ces and ces[0]['state'] == 'DISABLED'
                and ces[0]['status'] == 'VALID'):
            break
        time.sleep(5)
    batch.delete_compute_environment(computeEnvironment=COMPUTE_ENV_NAME)
    print(f'  ✓ deleted compute env {COMPUTE_ENV_NAME}')


def _teardown_job_definition(batch, cfg: AWSConfig):
    defs = batch.describe_job_definitions(
        jobDefinitionName=cfg.job_definition,
        status='ACTIVE')['jobDefinitions']
    for d in defs:
        batch.deregister_job_definition(
            jobDefinition=f'{d["jobDefinitionName"]}:{d["revision"]}')
    if defs:
        print(f'  ✓ deregistered {len(defs)} revision(s) of {cfg.job_definition}')


def _teardown_bucket(cfg: AWSConfig):
    s3 = boto3.client('s3', region_name=cfg.region)
    print(f'  emptying + deleting bucket {cfg.s3_bucket}')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=cfg.s3_bucket):
        for obj in page.get('Contents', []):
            s3.delete_object(Bucket=cfg.s3_bucket, Key=obj['Key'])
    s3.delete_bucket(Bucket=cfg.s3_bucket)
    print(f'  ✓ deleted bucket {cfg.s3_bucket}')


# ---------- status ----------------------------------------------------------


def cmd_status(args, cfg: AWSConfig):
    batch = boto3.client('batch', region_name=cfg.region)
    states = ('SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING',
              'SUCCEEDED', 'FAILED')
    print(f'[status] queue={cfg.job_queue}')
    for state in states:
        jobs = _list_jobs(batch, queue=cfg.job_queue, state=state,
                          label=args.label)
        print(f'  {state:10s} {len(jobs):>6d}')

    failed = _list_jobs(batch, queue=cfg.job_queue, state='FAILED',
                        label=args.label)
    if failed:
        print('\n[status] recent failures:')
        for job in failed[:5]:
            print(f'  {job["jobName"]:30s} {job.get("statusReason", "")}')


def _list_jobs(batch, *, queue: str, state: str,
               label: Optional[str] = None) -> List[dict]:
    out: List[dict] = []
    paginator = batch.get_paginator('list_jobs')
    for page in paginator.paginate(jobQueue=queue, jobStatus=state):
        for job in page.get('jobSummaryList', []):
            if label is None or job['jobName'].startswith(f'glow-{label}'):
                out.append(job)
    return out


# ---------- clean -----------------------------------------------------------


def cmd_clean(args, cfg: AWSConfig):
    if not (args.jobs or args.datasource):
        print('pick at least one of --jobs / --datasource')
        return
    if not args.yes:
        print('--yes required to actually delete S3 prefixes')
        return
    s3 = boto3.client('s3', region_name=cfg.region)
    if args.jobs:
        _delete_prefix(s3, cfg.s3_bucket, f'{cfg.s3_prefix}/jobs/')
    if args.datasource:
        _delete_prefix(s3, cfg.s3_bucket, f'{cfg.s3_prefix}/datasource/')


def _delete_prefix(s3, bucket: str, prefix: str):
    paginator = s3.get_paginator('list_objects_v2')
    total = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys = [{'Key': obj['Key']} for obj in page.get('Contents', [])]
        if not keys:
            continue
        # delete_objects max 1000 per call
        for i in range(0, len(keys), 1000):
            s3.delete_objects(Bucket=bucket,
                              Delete={'Objects': keys[i:i + 1000]})
        total += len(keys)
    print(f'  ✓ deleted {total} objects under s3://{bucket}/{prefix}')


# ---------- pause / resume --------------------------------------------------


def cmd_pause(args, cfg: AWSConfig):
    boto3.client('batch', region_name=cfg.region).update_job_queue(
        jobQueue=cfg.job_queue, state='DISABLED')
    print(f'[pause] {cfg.job_queue} → DISABLED '
          '(in-flight children keep running; SUBMITTED stays pending)')


def cmd_resume(args, cfg: AWSConfig):
    boto3.client('batch', region_name=cfg.region).update_job_queue(
        jobQueue=cfg.job_queue, state='ENABLED')
    print(f'[resume] {cfg.job_queue} → ENABLED')


# ---------- helpers ---------------------------------------------------------


def _account_id() -> str:
    return boto3.client('sts').get_caller_identity()['Account']


# ---------- CLI -------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m glow.aws.infra',
        description='Idempotent AWS Batch provisioning for glow.aws.')
    p.add_argument('--config', default=DEFAULT_CONFIG_PATH,
                   help=f'path to AWSConfig JSON (default: {DEFAULT_CONFIG_PATH})')
    subs = p.add_subparsers(dest='cmd', required=True)

    sp = subs.add_parser('setup', help='provision S3/ECR/Batch resources')
    sp.add_argument('--image-tag', default=None,
                    help='local docker image tag to push to ECR; '
                         'omit if already pushed')
    sp.set_defaults(func=cmd_setup)

    sp = subs.add_parser('teardown', help='delete Batch resources')
    sp.add_argument('--yes', action='store_true',
                    help='required to actually delete anything')
    sp.add_argument('--delete-bucket', action='store_true',
                    help='also empty + delete the S3 bucket')
    sp.set_defaults(func=cmd_teardown)

    sp = subs.add_parser('status', help='show job counts per state')
    sp.add_argument('--label', default=None,
                    help='only count jobs whose name starts glow-<label>')
    sp.set_defaults(func=cmd_status)

    sp = subs.add_parser('clean', help='delete S3 prefixes')
    sp.add_argument('--jobs', action='store_true',
                    help='delete jobs/ (job.pkl, result.pkl, manifest.pkl)')
    sp.add_argument('--datasource', action='store_true',
                    help='delete datasource/ (uploaded HCP exps)')
    sp.add_argument('--yes', action='store_true')
    sp.set_defaults(func=cmd_clean)

    sp = subs.add_parser('pause', help='disable job queue dispatch')
    sp.set_defaults(func=cmd_pause)

    sp = subs.add_parser('resume', help='re-enable job queue dispatch')
    sp.set_defaults(func=cmd_resume)

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    cfg = AWSConfig.from_file(args.config)
    args.func(args, cfg)


if __name__ == '__main__':
    main()
