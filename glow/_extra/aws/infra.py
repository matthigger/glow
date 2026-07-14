"""Idempotent AWS Batch infrastructure CLI.

Subcommands:

    python -m glow._extra.aws.infra bootstrap [--config PATH]
    python -m glow._extra.aws.infra setup     [--build] [--image-tag IMG] [--config PATH]
    python -m glow._extra.aws.infra teardown  [--yes] [--delete-bucket] [--config PATH]
    python -m glow._extra.aws.infra status    [--label LABEL]              [--config PATH]
    python -m glow._extra.aws.infra clear_storage [--runs|--records|--cache] [--yes]
    python -m glow._extra.aws.infra clear_jobs    [--label LABEL] [--yes]      [--config PATH]
    python -m glow._extra.aws.infra stage_hcp                             [--config PATH]
    python -m glow._extra.aws.infra pull                                  [--config PATH]
    python -m glow._extra.aws.infra pause                                 [--config PATH]
    python -m glow._extra.aws.infra resume                                [--config PATH]

Reads AWSConfig from the default user-config path (or --config) for
bucket, queue, job-definition, and region names; the rest of the provisioning detail
(instance types, allocation strategy, VPC lookup) lives in this file.

bootstrap creates the account-wide IAM roles setup depends on (the Batch
service-linked role, ecsInstanceRole plus instance profile, the
spot-fleet role, and the two ECS task roles). It needs IAM-admin
credentials and only has to run once per account; setup and the daily
commands need only S3, Batch, and ECR permissions.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

from glow._extra.aws.config import AWSConfig, DEFAULT_CONFIG_PATH, s3_key


# ---------- compute-environment constants -----------------------------------
COMPUTE_ENV_NAME = 'glow-compute-env-spot'
ALLOC_STRATEGY = 'SPOT_PRICE_CAPACITY_OPTIMIZED'

# Instance pool: 6/7 series c/m/r families, large → 12xlarge.  Picked
# under SPOT_PRICE_CAPACITY_OPTIMIZED so Batch picks the cheapest pool
# with available Spot capacity.  5-series omitted: a 2026-04-28 benchmark
# clocked them ~2x slower per core.
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

# IAM role names created by bootstrap, consumed by setup.
ECS_INSTANCE_ROLE = 'ecsInstanceRole'
SPOT_FLEET_ROLE = 'aws-ec2-spot-fleet-tagging-role'
ECS_TASK_EXECUTION_ROLE = 'GlowEcsTaskExecutionRole'
ECS_TASK_ROLE = 'GlowEcsTaskRole'

# ECR repository for the worker image.
ECR_REPO_NAME = 'glow-worker'

# Default local tag built/pushed by `setup --build`.
DEFAULT_IMAGE_TAG = 'glow-worker:latest'


# ---------- bootstrap (one-time IAM) ----------------------------------------


def cmd_bootstrap(args, cfg: AWSConfig) -> None:
    """Create the account-wide IAM roles setup depends on (idempotent).

    Needs IAM-admin credentials; only has to run once per account. Every
    call here tolerates pre-existing roles, so rerunning is safe.

    Args:
        args: parsed argparse Namespace (unused; kept for the CLI
            dispatch signature).
        cfg (AWSConfig): supplies the S3 bucket scoped into the task
            role's inline policy.
    """
    iam = boto3.client('iam')
    print('[bootstrap] creating IAM roles (idempotent)')

    _create_service_linked_role(iam, 'batch.amazonaws.com')

    # EC2 instances that Batch launches need this role + a like-named
    # instance profile wrapping it.
    _create_role(
        iam, ECS_INSTANCE_ROLE, _trust('ec2.amazonaws.com'),
        managed=['service-role/AmazonEC2ContainerServiceforEC2Role'])
    _create_instance_profile(iam, ECS_INSTANCE_ROLE)

    # Lets Batch tag the Spot fleet it requests.
    _create_role(
        iam, SPOT_FLEET_ROLE, _trust('spotfleet.amazonaws.com'),
        managed=['service-role/AmazonEC2SpotFleetTaggingRole'])

    # ECS pulls the image / writes logs under the execution role; the worker
    # container reads/writes S3 under the task role.
    _create_role(
        iam, ECS_TASK_EXECUTION_ROLE, _trust('ecs-tasks.amazonaws.com'),
        managed=['service-role/AmazonECSTaskExecutionRolePolicy'])
    _create_role(
        iam, ECS_TASK_ROLE, _trust('ecs-tasks.amazonaws.com'),
        inline={'GlowS3Access': _s3_policy(cfg.s3_bucket)})

    print('[bootstrap] done. now run: '
          'python -m glow._extra.aws.infra setup --build')


def _trust(service: str) -> dict:
    """Build an assume-role trust policy for an AWS service principal.

    Args:
        service (str): the service principal, e.g. ec2.amazonaws.com.

    Returns:
        policy (dict): the trust-policy document.
    """
    return {
        'Version': '2012-10-17',
        'Statement': [{
            'Effect': 'Allow',
            'Principal': {'Service': service},
            'Action': 'sts:AssumeRole',
        }],
    }


def _s3_policy(bucket: str) -> dict:
    """Build an inline policy granting the worker get/put/list on its bucket.

    Args:
        bucket (str): the S3 bucket name to scope the policy to.

    Returns:
        policy (dict): the inline-policy document.
    """
    return {
        'Version': '2012-10-17',
        'Statement': [{
            'Effect': 'Allow',
            'Action': ['s3:GetObject', 's3:PutObject', 's3:ListBucket'],
            'Resource': [
                f'arn:aws:s3:::{bucket}',
                f'arn:aws:s3:::{bucket}/*',
            ],
        }],
    }


def _create_service_linked_role(iam, service: str) -> None:
    """Create a service-linked role, tolerating one that already exists.

    Args:
        iam: boto3 IAM client.
        service (str): the AWS service principal, e.g. batch.amazonaws.com.
    """
    try:
        iam.create_service_linked_role(AWSServiceName=service)
        print(f'  ✓ created service-linked role for {service}')
    except ClientError as e:
        # An existing role surfaces as InvalidInput here; that is the
        # success path on rerun.
        already = ('InvalidInput', 'EntityAlreadyExists')
        if e.response['Error']['Code'] in already:
            print(f'  ✓ service-linked role for {service} exists')
        else:
            raise


def _create_role(iam, name: str, trust: dict, *,
                 managed: Optional[List[str]] = None,
                 inline: Optional[dict] = None) -> None:
    """Create an IAM role and attach its policies (idempotent).

    A pre-existing role is tolerated; policy attachment then runs over it
    so reruns converge on the same attachments.

    Args:
        iam: boto3 IAM client.
        name (str): the role name.
        trust (dict): the assume-role trust-policy document.
        managed (list | None): AWS-managed policy ARN suffixes to attach.
        inline (dict | None): {policy_name: policy_document} inline policies.
    """
    try:
        iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=json.dumps(trust))
        print(f'  ✓ created role {name}')
    except ClientError as e:
        if e.response['Error']['Code'] != 'EntityAlreadyExists':
            raise
        print(f'  ✓ role {name} exists')

    for arn_suffix in (managed or []):
        iam.attach_role_policy(
            RoleName=name,
            PolicyArn=f'arn:aws:iam::aws:policy/{arn_suffix}')
    for policy_name, doc in (inline or {}).items():
        iam.put_role_policy(
            RoleName=name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(doc))


def _create_instance_profile(iam, name: str) -> None:
    """Create an instance profile and attach the like-named role (idempotent).

    Args:
        iam: boto3 IAM client.
        name (str): the profile name, also used as the attached role name.
    """
    try:
        iam.create_instance_profile(InstanceProfileName=name)
        print(f'  ✓ created instance profile {name}')
    except ClientError as e:
        if e.response['Error']['Code'] != 'EntityAlreadyExists':
            raise
        print(f'  ✓ instance profile {name} exists')
    # add_role_to_instance_profile errors if the role is already attached;
    # tolerate that so rerun stays idempotent.
    try:
        iam.add_role_to_instance_profile(
            InstanceProfileName=name, RoleName=name)
    except ClientError as e:
        if e.response['Error']['Code'] != 'LimitExceeded':
            raise


# ---------- setup -----------------------------------------------------------


def cmd_setup(args, cfg: AWSConfig) -> None:
    """Provision the S3, ECR, and Batch resources (idempotent).

    The worker image reaches ECR one of three ways, by flag:
        --build       build it from the Dockerfile here, then push (the
                      one-step (re)deploy after a worker-code change);
                      tags DEFAULT_IMAGE_TAG unless --image-tag overrides.
        --image-tag T push the local image already tagged T instead of
                      building it.
        neither       reuse the DEFAULT_IMAGE_TAG image already in ECR
                      (see _resolve_image_uri).

    Args:
        args: parsed argparse Namespace; reads args.build and args.image_tag.
        cfg (AWSConfig): bucket, queue, definition, region, and resource
            sizing.
    """
    region = cfg.region
    account_id = _account_id()

    print(f'[setup] region={region} account={account_id}')

    if args.build:
        args.image_tag = args.image_tag or DEFAULT_IMAGE_TAG
        _build_image(args.image_tag)

    _setup_s3_bucket(cfg)

    image_uri = _resolve_image_uri(args, region, account_id)
    _setup_job_definition(cfg, image_uri=image_uri, account_id=account_id)
    ce_arn = _setup_compute_environment(cfg, account_id=account_id)
    _setup_job_queue(cfg, ce_arn=ce_arn)
    print('[setup] done.')


def _setup_s3_bucket(cfg: AWSConfig) -> None:
    """Create the configured S3 bucket if it does not already exist.

    Args:
        cfg (AWSConfig): supplies the bucket name and region.
    """
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

    With --image-tag <tag>, the local image is pushed to ECR under that
    tag. Without it, an existing glow-worker:latest in ECR is expected,
    looked up via describe_images.

    Args:
        args: parsed argparse Namespace; reads args.image_tag.
        region (str): the AWS region.
        account_id (str): the AWS account id, for the ECR URI.

    Returns:
        image_uri (str): the resolved ECR image URI.
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


def _build_image(tag: str) -> None:
    """Build the worker Docker image from the repo's Dockerfile.

    Runs docker build with the repo root as the build context (so the
    Dockerfile's COPY glow/ ./glow/ captures the working-tree source)
    regardless of the caller's cwd.

    Args:
        tag (str): local image tag to build, e.g. glow-worker:latest.
    """
    import shutil
    import subprocess

    if shutil.which('docker') is None:
        raise SystemExit('docker not found in PATH — install Docker first')

    repo_root = Path(__file__).resolve().parents[3]
    dockerfile = repo_root / 'glow' / '_extra' / 'aws' / 'Dockerfile'
    print(f'[setup] docker build -t {tag} (context {repo_root})')
    subprocess.run(
        ['docker', 'build', '-t', tag, '-f', str(dockerfile), str(repo_root)],
        check=True)


def _push_image_to_ecr(local_tag: str, region: str, account_id: str) -> str:
    """Tag and push a local Docker image to ECR, creating the repo if needed.

    Uses the docker CLI via subprocess rather than the Docker SDK so it
    works in any environment that already builds the image.

    Args:
        local_tag (str): the local Docker image tag to push.
        region (str): the AWS region.
        account_id (str): the AWS account id, for the ECR URI.

    Returns:
        ecr_uri (str): the pushed image's ECR URI.
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


def _setup_job_definition(cfg: AWSConfig, *, image_uri: str,
                          account_id: str) -> None:
    """Register a Batch job definition pointing at the worker image.

    Args:
        cfg (AWSConfig): supplies definition name, region, vcpus, memory
            tiers, retry attempts, and timeout.
        image_uri (str): the ECR image URI the container runs.
        account_id (str): the AWS account id, for the task-role ARNs.
    """
    batch = boto3.client('batch', region_name=cfg.region)
    # ECS task roles referenced here are created by bootstrap.
    exec_role = f'arn:aws:iam::{account_id}:role/GlowEcsTaskExecutionRole'
    task_role = f'arn:aws:iam::{account_id}:role/GlowEcsTaskRole'

    # No command: the image ENTRYPOINT is `python -m glow._extra.aws.worker`, and
    # the driver overrides command per-submission to pass the manifest URI
    # as its argv.
    container = {
        'image': image_uri,
        'jobRoleArn': task_role,
        'executionRoleArn': exec_role,
        'resourceRequirements': [
            {'type': 'VCPU', 'value': str(cfg.vcpus)},
            {'type': 'MEMORY', 'value': str(cfg.memory_mb_tiers[0])},
        ],
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


def _default_subnets(ec2) -> list:
    """Return the default subnet of every availability zone, AZ-sorted.

    One default subnet per AZ in the account's default VPC, ordered by AZ so
    the compute environment spans every zone deterministically. Spot capacity
    and reclaim risk are per-AZ pools, so spreading over all zones lets a
    capacity-aware allocation strategy re-place a reclaimed instance in a
    healthy zone instead of relaunching into the same crunched one.

    Args:
        ec2: boto3 EC2 client.

    Returns:
        subnets (list[dict]): describe_subnets records, one per AZ, sorted by
            AvailabilityZone.
    """
    subnets = ec2.describe_subnets(
        Filters=[{'Name': 'default-for-az', 'Values': ['true']}])['Subnets']
    if not subnets:
        raise SystemExit(
            'no default subnets found; pass --vpc-subnet / --vpc-sg manually')
    return sorted(subnets, key=lambda s: s['AvailabilityZone'])


def _setup_compute_environment(cfg: AWSConfig, *, account_id: str) -> str:
    """Create or reconcile the Spot compute environment (idempotent).

    A new environment spans every default subnet of the account's default VPC
    -- one per availability zone -- so Batch spreads Spot instances across all
    zones and a reclaim in one zone re-lands in another (see _default_subnets).
    On rerun an existing environment is updated in place: maxvCpus always, and
    its subnet set whenever it drifts from the current default subnets. Growing
    a single-AZ environment to all zones is an in-place infrastructure update
    (the SPOT_PRICE_CAPACITY_OPTIMIZED strategy supports it); it only replaces
    instances, and idle at desiredvCpus 0 there are none to replace.

    Args:
        cfg (AWSConfig): supplies region and max_concurrent.
        account_id (str): the AWS account id, for the service / instance
            / spot-fleet role ARNs.

    Returns:
        ce_arn (str): the compute environment's ARN.
    """
    batch = boto3.client('batch', region_name=cfg.region)
    ec2 = boto3.client('ec2', region_name=cfg.region)

    subnets = _default_subnets(ec2)
    subnet_ids = [s['SubnetId'] for s in subnets]
    n_az = len({s['AvailabilityZone'] for s in subnets})

    existing = batch.describe_compute_environments(
        computeEnvironments=[COMPUTE_ENV_NAME])['computeEnvironments']
    if existing:
        arn = existing[0]['computeEnvironmentArn']
        update = {'maxvCpus': cfg.max_concurrent}
        live = existing[0].get('computeResources', {}).get('subnets', [])
        if set(live) != set(subnet_ids):
            update['subnets'] = subnet_ids
        try:
            batch.update_compute_environment(
                computeEnvironment=COMPUTE_ENV_NAME, computeResources=update)
            azs = (f', subnets → {len(subnet_ids)} across {n_az} AZ(s)'
                   if 'subnets' in update else '')
            print(f'  ✓ compute env {COMPUTE_ENV_NAME} exists '
                  f'(maxvCpus → {cfg.max_concurrent}{azs})')
        except ClientError as e:
            print(f'  ⚠ compute env update failed: {e}')
        return arn

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
            'subnets': subnet_ids,
            'securityGroupIds': [sg['GroupId']],
            'instanceRole': instance_profile,
            'spotIamFleetRole': spot_fleet,
        },
    )
    arn = response['computeEnvironmentArn']
    print(f'  ✓ created compute env {COMPUTE_ENV_NAME} '
          f'({len(subnet_ids)} subnets across {n_az} AZ(s))')
    _wait_for_ce_valid(batch)
    return arn


def _wait_for_ce_valid(batch, timeout_s: int = 300) -> None:
    """Block until the compute environment reaches VALID, else raise.

    Args:
        batch: boto3 Batch client.
        timeout_s (int): seconds to wait before raising SystemExit.

    Raises:
        SystemExit: if VALID is not reached within timeout_s.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ces = batch.describe_compute_environments(
            computeEnvironments=[COMPUTE_ENV_NAME])['computeEnvironments']
        if ces and ces[0]['status'] == 'VALID':
            return
        time.sleep(5)
    raise SystemExit(f'compute env did not reach VALID within {timeout_s}s')


def _setup_job_queue(cfg: AWSConfig, *, ce_arn: str) -> None:
    """Create the job queue bound to the compute environment (idempotent).

    Args:
        cfg (AWSConfig): supplies the queue name and region.
        ce_arn (str): the compute environment ARN to bind the queue to.
    """
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


def cmd_teardown(args, cfg: AWSConfig) -> None:
    """Disable and delete the Batch resources (requires --yes).

    Args:
        args: parsed argparse Namespace; reads args.yes and
            args.delete_bucket.
        cfg (AWSConfig): supplies the queue, definition, bucket, region.
    """
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


def _teardown_queue(batch, cfg: AWSConfig) -> None:
    """Disable and delete the job queue, waiting for it to drain.

    Args:
        batch: boto3 Batch client.
        cfg (AWSConfig): supplies the queue name.
    """
    queues = batch.describe_job_queues(
        jobQueues=[cfg.job_queue])['jobQueues']
    if not queues:
        return
    print(f'  disabling job queue {cfg.job_queue}')
    batch.update_job_queue(jobQueue=cfg.job_queue, state='DISABLED')
    _wait_for_queue_state(batch, cfg.job_queue, 'DISABLED')
    batch.delete_job_queue(jobQueue=cfg.job_queue)
    print(f'  ✓ deleted job queue {cfg.job_queue}')


def _wait_for_queue_state(batch, name: str, state: str,
                          timeout_s: int = 180) -> None:
    """Block until the queue reaches state (and VALID) or disappears.

    Returns on timeout rather than raising, so teardown proceeds even if
    the queue is slow to settle.

    Args:
        batch: boto3 Batch client.
        name (str): the job-queue name.
        state (str): the target state, e.g. DISABLED.
        timeout_s (int): seconds to wait before giving up.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        qs = batch.describe_job_queues(jobQueues=[name])['jobQueues']
        if not qs:
            return
        if qs[0]['state'] == state and qs[0]['status'] == 'VALID':
            return
        time.sleep(3)


def _teardown_compute_env(batch) -> None:
    """Disable the compute environment, wait for it to settle, then delete.

    Args:
        batch: boto3 Batch client.
    """
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


def _teardown_job_definition(batch, cfg: AWSConfig) -> None:
    """Deregister every ACTIVE revision of the job definition.

    Args:
        batch: boto3 Batch client.
        cfg (AWSConfig): supplies the job-definition name.
    """
    defs = batch.describe_job_definitions(
        jobDefinitionName=cfg.job_definition,
        status='ACTIVE')['jobDefinitions']
    for d in defs:
        batch.deregister_job_definition(
            jobDefinition=f'{d["jobDefinitionName"]}:{d["revision"]}')
    if defs:
        print(f'  ✓ deregistered {len(defs)} revision(s) of {cfg.job_definition}')


def _teardown_bucket(cfg: AWSConfig) -> None:
    """Empty and delete the S3 bucket.

    Args:
        cfg (AWSConfig): supplies the bucket name and region.
    """
    s3 = boto3.client('s3', region_name=cfg.region)
    print(f'  emptying + deleting bucket {cfg.s3_bucket}')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=cfg.s3_bucket):
        for obj in page.get('Contents', []):
            s3.delete_object(Bucket=cfg.s3_bucket, Key=obj['Key'])
    s3.delete_bucket(Bucket=cfg.s3_bucket)
    print(f'  ✓ deleted bucket {cfg.s3_bucket}')


# ---------- status ----------------------------------------------------------


# Non-terminal Batch job states, in lifecycle order: a job moves down this
# list before reaching the terminal SUCCEEDED / FAILED.  cancel acts on
# exactly these; status also reports the two terminal states.
ACTIVE_STATES = ('SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING')
TERMINAL_STATES = ('SUCCEEDED', 'FAILED')

# Default awslogs group AWS Batch writes container stdout/stderr to; the
# per-attempt stream name lives in describe_jobs container.logStreamName.
LOG_GROUP = '/aws/batch/job'


def cmd_status(args, cfg: AWSConfig) -> None:
    """Print job counts, per-run array progress, and expanded failures.

    Each run is one Batch (array) parent named glow-run-<hex>.  The job
    listing is taken once per state; the array parents' statusSummary and
    timestamps come along for free, so active-run progress and timing add
    no API calls.  Failures drill into the failed children for exit codes
    and an OOM/error tag, and with --logs tail each child's CloudWatch
    stream.

    Args:
        args: parsed argparse Namespace; reads args.label, args.limit,
            args.logs.
        cfg (AWSConfig): supplies the queue name and region.
    """
    batch = boto3.client('batch', region_name=cfg.region)
    print(f'[status] queue={cfg.job_queue}')

    by_state: Dict[str, List[dict]] = {
        state: _list_jobs(batch, queue=cfg.job_queue, state=state,
                          label=args.label)
        for state in ACTIVE_STATES + TERMINAL_STATES}

    _print_state_counts(by_state)
    _print_active_runs(by_state)
    _print_failures(batch, cfg, by_state['FAILED'],
                    limit=args.limit, logs=args.logs)


def _print_state_counts(by_state: Dict[str, List[dict]]) -> None:
    """Print the job count for each terminal or non-empty active state.

    Args:
        by_state (dict): state name -> list of job summaries.
    """
    print('\n[status] jobs by state:')
    zero = []
    for state in ACTIVE_STATES + TERMINAL_STATES:
        n = len(by_state[state])
        if n or state in TERMINAL_STATES:
            print(f'  {state:10s} {n:>6d}')
        else:
            zero.append(state)
    if zero:
        print(f'  (0: {" ".join(zero)})')


def _print_active_runs(by_state: Dict[str, List[dict]]) -> None:
    """Print in-flight runs with array-child progress and timing.

    Args:
        by_state (dict): state name -> list of job summaries.
    """
    active = [job for state in ACTIVE_STATES for job in by_state[state]]
    if not active:
        return
    print(f'\n[status] active ({_n_runs(active)}):')
    for job in active:
        wait, run = _timing(job)
        timing = (f'ran {_fmt_dur(run)}' if run is not None
                  else f'waited {_fmt_dur(wait)}')
        print(f'  {_run_id(job["jobName"]):16s} {_kind(job):9s} '
              f'{job.get("status", "?"):9s} {_fmt_progress(job):14s} {timing}')


def _list_jobs(batch, *, queue: str, state: str,
               label: Optional[str] = None) -> List[dict]:
    """List jobs in one queue and state, optionally filtered by label.

    Args:
        batch: boto3 Batch client.
        queue (str): the job-queue name.
        state (str): the job status to list, e.g. RUNNING.
        label (str | None): if given, keep only jobs whose name starts
            glow-<label>.

    Returns:
        out (list): job-summary dicts matching the filters.
    """
    out: List[dict] = []
    paginator = batch.get_paginator('list_jobs')
    for page in paginator.paginate(jobQueue=queue, jobStatus=state):
        for job in page.get('jobSummaryList', []):
            if label is None or job['jobName'].startswith(f'glow-{label}'):
                out.append(job)
    return out


# ---------- status helpers --------------------------------------------------


def _print_failures(batch, cfg: AWSConfig, failed: List[dict], *,
                    limit: int, logs: bool) -> None:
    """Expand failed runs: per-child exit code, OOM/error tag, log tail.

    For an array parent the failed children are listed (their summaries
    already carry container.exitCode / reason); single jobs are their own
    "child".  With logs=True each child's CloudWatch stream is tailed.
    A tag tally closes the section.

    Args:
        batch: boto3 Batch client.
        cfg (AWSConfig): supplies the region for the logs client.
        failed (list): FAILED parent/single job summaries from the queue.
        limit (int): max number of failed runs to expand.
        logs (bool): if True, tail each failed child's log stream.
    """
    if not failed:
        return
    shown = failed[:limit]
    hidden = len(failed) - len(shown)
    suffix = f', showing {len(shown)}' if hidden else ''
    print(f'\n[status] failures ({_n_runs(failed)}{suffix}):')

    logs_client = (boto3.client('logs', region_name=cfg.region)
                   if logs else None)
    tally: Dict[str, int] = {'OOM': 0, 'timeout': 0, 'err': 0}

    for job in shown:
        _, run = _timing(job)
        print(f'  {_run_id(job["jobName"]):16s} {_kind(job):9s} '
              f'{_fmt_progress(job):14s} ran {_fmt_dur(run)}')
        is_array = _array_size(job) is not None
        children = _failed_children(batch, job['jobId']) if is_array else [job]
        if logs_client is not None:
            _attach_log_streams(batch, children)
        for child in children:
            tally[_failure_tag(child)] += 1
            _print_child_failure(child, is_array=is_array,
                                 logs_client=logs_client)

    if hidden:
        print(f'  … and {hidden} more (raise --limit to expand)')
    summary = ' · '.join(f'{n} {tag}' for tag, n in tally.items() if n)
    if summary:
        print(f'  tags: {summary}')


def _print_child_failure(child: dict, *, is_array: bool,
                         logs_client) -> None:
    """Print one failed child's index, exit code, tag, and reason.

    Args:
        child (dict): a failed child (array) or the failed job (single).
        is_array (bool): True if child belongs to an array parent.
        logs_client: boto3 Logs client, or None to skip the log tail.
    """
    container = child.get('container') or {}
    exit_code = container.get('exitCode')
    exit_str = f'exit={exit_code}' if exit_code is not None else 'exit=?'
    reason = (container.get('reason') or child.get('statusReason') or '').strip()
    idx = (child.get('arrayProperties') or {}).get('index')
    label = f'child[{idx}]' if is_array and idx is not None else 'job'
    print(f'     {label:10s} {exit_str:9s} {_failure_tag(child):8s} '
          f'{reason[:80]}')
    if logs_client is not None:
        for line in _tail_log(logs_client, container.get('logStreamName')):
            print(f'        | {line}')


def _failed_children(batch, parent_id: str) -> List[dict]:
    """List an array parent's FAILED child summaries, in index order.

    The child summaries already include container.exitCode / reason and
    arrayProperties.index, so no describe_jobs call is needed unless log
    streams are wanted (see _attach_log_streams).

    Args:
        batch: boto3 Batch client.
        parent_id (str): the array parent job id.

    Returns:
        children (list): failed child job summaries, sorted by index.
    """
    out: List[dict] = []
    paginator = batch.get_paginator('list_jobs')
    for page in paginator.paginate(arrayJobId=parent_id, jobStatus='FAILED'):
        out.extend(page.get('jobSummaryList', []))
    out.sort(key=lambda j: (j.get('arrayProperties') or {}).get('index', 0))
    return out


def _attach_log_streams(batch, children: List[dict]) -> None:
    """Fill each child's container.logStreamName via describe_jobs in place.

    The list_jobs summary omits logStreamName, so describe the child ids
    (in 100-id chunks, the describe_jobs max) and copy the stream back.

    Args:
        batch: boto3 Batch client.
        children (list): child/single job summaries to enrich in place.
    """
    by_id = {c['jobId']: c for c in children}
    ids = list(by_id)
    for i in range(0, len(ids), 100):
        payload = batch.describe_jobs(jobs=ids[i:i + 100])
        for job in payload.get('jobs', []):
            stream = (job.get('container') or {}).get('logStreamName')
            if stream:
                by_id[job['jobId']].setdefault('container', {})
                by_id[job['jobId']]['container']['logStreamName'] = stream


def _tail_log(logs_client, stream: Optional[str], n: int = 12) -> List[str]:
    """Return the last n message lines of a CloudWatch log stream.

    Args:
        logs_client: boto3 CloudWatch Logs client.
        stream (str | None): the log stream name; None/empty -> a note.
        n (int): number of trailing lines to return.

    Returns:
        lines (list): up to n messages (oldest first), or a single
            explanatory line if the stream is missing or unreadable.
    """
    if not stream:
        return ['(no log stream)']
    try:
        resp = logs_client.get_log_events(
            logGroupName=LOG_GROUP, logStreamName=stream,
            startFromHead=False, limit=n)
    except ClientError as exc:
        return [f'(log fetch failed: {exc.response["Error"]["Code"]})']
    return [e['message'] for e in resp.get('events', [])]


def _failure_tag(job: dict) -> str:
    """Tag a failed job OOM / timeout / err for at-a-glance triage.

    Mirrors driver._is_oom's ordering (timeout before OOM, since both
    exit 137) but kept local so the CLI avoids the driver's heavy imports.

    Args:
        job (dict): a failed job/child summary or describe payload.

    Returns:
        tag (str): 'OOM', 'timeout', or 'err'.
    """
    status_reason = (job.get('statusReason') or '').lower()
    container = job.get('container') or {}
    container_reason = (container.get('reason') or '').lower()
    texts = (status_reason, container_reason)
    if any('duration' in t and 'timeout' in t for t in texts):
        return 'timeout'
    if container.get('exitCode') in (137, 134):
        return 'OOM'
    if any('memory' in t for t in texts):
        return 'OOM'
    return 'err'


def _fmt_progress(job: dict) -> str:
    """Compact array-child tally from a parent's statusSummary.

    Uses arrayProperties.statusSummary (already in the list_jobs summary,
    so no extra API call): ✓ succeeded, ✗ failed, ⋯ still in flight.

    Args:
        job (dict): a Batch job summary.

    Returns:
        progress (str): e.g. '5✓ 3✗ 2⋯', or the bare status for a
            single (non-array) job.
    """
    summary = (job.get('arrayProperties') or {}).get('statusSummary') or {}
    if not summary:
        return job.get('status', '?')
    done = summary.get('SUCCEEDED', 0)
    failed = summary.get('FAILED', 0)
    inflight = sum(v for k, v in summary.items()
                   if k not in TERMINAL_STATES)
    parts = [f'{done}✓', f'{failed}✗']
    if inflight:
        parts.append(f'{inflight}⋯')
    return ' '.join(parts)


def _timing(job: dict) -> Tuple[Optional[float], Optional[float]]:
    """Return (queue_wait_s, run_s) from a summary's epoch-ms stamps.

    Args:
        job (dict): a Batch job summary; createdAt / startedAt / stoppedAt
            are epoch milliseconds and any may be absent.

    Returns:
        queue_wait_s (float | None): seconds from submit to start.
        run_s (float | None): seconds from start to stop.
    """
    created = job.get('createdAt')
    started = job.get('startedAt')
    stopped = job.get('stoppedAt')
    wait = (started - created) / 1000 if created and started else None
    run = (stopped - started) / 1000 if started and stopped else None
    return wait, run


def _fmt_dur(seconds: Optional[float]) -> str:
    """Format a duration in seconds as compact h/m/s, or '—' if unknown."""
    if seconds is None:
        return '—'
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m{seconds % 60:02d}s'
    return f'{seconds // 3600}h{(seconds % 3600) // 60:02d}m'


def _kind(job: dict) -> str:
    """Describe a job as 'array×N' (array parent) or 'single'."""
    size = _array_size(job)
    return f'array×{size}' if size else 'single'


def _array_size(job: dict) -> Optional[int]:
    """Return the array size for an array parent, else None for a single job."""
    return (job.get('arrayProperties') or {}).get('size')


def _run_id(job_name: str) -> str:
    """Strip the 'glow-' prefix: 'glow-run-1a2b3c4d' -> 'run-1a2b3c4d'."""
    prefix = 'glow-'
    return job_name[len(prefix):] if job_name.startswith(prefix) else job_name


def _n_runs(jobs: List[dict]) -> str:
    """Pluralize a run count: '1 run' / '3 runs'."""
    return f'{len(jobs)} run' + ('' if len(jobs) == 1 else 's')


# ---------- clear_storage ---------------------------------------------------


def cmd_clear_storage(args, cfg: AWSConfig) -> None:
    """Delete the runs/, records/, and/or cache/ S3 prefixes (requires --yes).

    The shared state the driver / workers use (see glow._extra.aws.sync):
    runs/ holds the per-run manifests, records/ the per-hash records, cache/
    the synced joblib cache. Clearing cache/ + records/ forces a cold rerun;
    clearing runs/ just drops stale manifests.

    Args:
        args: parsed argparse Namespace; reads args.runs, args.records,
            args.cache, and args.yes.
        cfg (AWSConfig): supplies the bucket, prefix, and region.
    """
    chosen = [name for name, flag in
              (('runs', args.runs), ('records', args.records),
               ('cache', args.cache)) if flag]
    if not chosen:
        print('pick at least one of --runs / --records / --cache')
        return
    if not args.yes:
        print('--yes required to actually delete S3 prefixes')
        return
    s3 = boto3.client('s3', region_name=cfg.region)
    for name in chosen:
        _delete_prefix(s3, cfg.s3_bucket, s3_key(cfg.s3_prefix, name) + '/')


def _delete_prefix(s3, bucket: str, prefix: str) -> None:
    """Delete every object under an S3 prefix, in batches.

    Args:
        s3: boto3 S3 client.
        bucket (str): the S3 bucket name.
        prefix (str): the key prefix to clear.
    """
    paginator = s3.get_paginator('list_objects_v2')
    total = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys = [{'Key': obj['Key']} for obj in page.get('Contents', [])]
        if not keys:
            continue
        # delete_objects accepts at most 1000 keys per call.
        for i in range(0, len(keys), 1000):
            s3.delete_objects(Bucket=bucket,
                              Delete={'Objects': keys[i:i + 1000]})
        total += len(keys)
    print(f'  ✓ deleted {total} objects under s3://{bucket}/{prefix}')


# ---------- stage_hcp -------------------------------------------------------


def cmd_stage_hcp(args, cfg: AWSConfig) -> None:
    """Build the HCP npy bundle locally and upload it to S3 for workers.

    One-time staging, idempotent (a rerun skips objects already there):
    ensure_hcp_bundle converts the niftis to the per-feature bundle (mask /
    affine / meta + one float32 array per feature; see hcp.py) on first use,
    then the whole bundle is mirrored to {s3_prefix}/hcp_bundle. An HCP Batch
    worker pulls only the files its cell needs (see
    glow._extra.aws.sync.hcp_bundle_keys / worker), and data_factory_hcp builds
    from them with no niftis and no DUA prompt. The conversion needs the niftis
    present, so run it on a box with the data (any local HCP run downloads +
    extracts the Zenodo archive).

    Args:
        args: parsed argparse Namespace (unused; kept for CLI dispatch).
        cfg (AWSConfig): supplies the bucket, prefix, and region.
    """
    from glow._extra.aws import s3, sync
    from glow._extra.benchmark import hcp

    # convert niftis -> bundle if not already built (needs the niftis locally)
    hcp.ensure_hcp_bundle()
    local_dir, key_prefix = sync.hcp_bundle_pair(cfg.s3_prefix)
    s3_client = boto3.client('s3', region_name=cfg.region)
    print(f'[stage_hcp] uploading {local_dir} -> '
          f's3://{cfg.s3_bucket}/{key_prefix}')
    n = s3.upload_dir(s3_client, cfg.s3_bucket, local_dir, key_prefix)
    print(f'[stage_hcp] uploaded {n} new file(s) '
          '(existing objects skipped)')


# ---------- pull ------------------------------------------------------------


def cmd_pull(args, cfg: AWSConfig) -> None:
    """Download the shared records and run_ana cache from S3 to local.

    The pull side of the sync the workers push while they compute (see
    glow._extra.aws.sync.sync_pairs): the per-hash records/ (the provenance
    the CSVs are built from) and the run_ana cache (the expensive score-dict
    compute). drive_aws pulls the records at the end of a run to write the
    CSVs; this exposes the same pull on its own, so an interrupted or
    cancelled run's partial results are recoverable without waiting for a
    fresh run to drain. Non-destructive: existing local files are skipped
    (s3.download_prefix), so nothing local is overwritten.

    Args:
        args: parsed argparse Namespace (unused; kept for CLI dispatch).
        cfg (AWSConfig): supplies the bucket, prefix, and region.
    """
    from glow._extra.aws import s3, sync

    s3_client = boto3.client('s3', region_name=cfg.region)
    for local_dir, key_prefix in sync.sync_pairs(cfg.s3_prefix):
        print(f'[pull] s3://{cfg.s3_bucket}/{key_prefix} -> {local_dir}')
        n = s3.download_prefix(s3_client, cfg.s3_bucket, key_prefix, local_dir)
        print(f'[pull] downloaded {n} new file(s) (existing skipped)')


# ---------- clear_jobs ------------------------------------------------------


def cmd_clear_jobs(args, cfg: AWSConfig) -> None:
    """Terminate every active job in the queue (requires --yes).

    One terminate_job pass clears the whole active lifecycle: Batch cancels
    jobs that have not yet reached STARTING and kills STARTING / RUNNING
    containers (which transition to FAILED).  SUCCEEDED / FAILED jobs are
    left untouched.  Unlike pause this does not disable the queue, so a
    later `benchmark --aws` run dispatches fresh children; pair with pause to
    also stop new dispatch.

    Args:
        args: parsed argparse Namespace; reads args.label and args.yes.
        cfg (AWSConfig): supplies the queue name and region.
    """
    if not args.yes:
        print('--yes required to actually terminate jobs')
        return
    batch = boto3.client('batch', region_name=cfg.region)
    total = 0
    for state in ACTIVE_STATES:
        for job in _list_jobs(batch, queue=cfg.job_queue, state=state,
                              label=args.label):
            batch.terminate_job(
                jobId=job['jobId'],
                reason='terminated via glow._extra.aws.infra clear_jobs')
            total += 1
    scope = f' for label {args.label}' if args.label else ''
    print(f'  ✓ terminated {total} active job(s) in {cfg.job_queue}{scope}')


# ---------- pause / resume --------------------------------------------------


def cmd_pause(args, cfg: AWSConfig) -> None:
    """Disable the job queue so it stops dispatching new jobs.

    Args:
        args: parsed argparse Namespace (unused; kept for CLI dispatch).
        cfg (AWSConfig): supplies the queue name and region.
    """
    boto3.client('batch', region_name=cfg.region).update_job_queue(
        jobQueue=cfg.job_queue, state='DISABLED')
    print(f'[pause] {cfg.job_queue} → DISABLED '
          '(in-flight children keep running; SUBMITTED stays pending)')


def cmd_resume(args, cfg: AWSConfig) -> None:
    """Re-enable the job queue so it resumes dispatching jobs.

    Args:
        args: parsed argparse Namespace (unused; kept for CLI dispatch).
        cfg (AWSConfig): supplies the queue name and region.
    """
    boto3.client('batch', region_name=cfg.region).update_job_queue(
        jobQueue=cfg.job_queue, state='ENABLED')
    print(f'[resume] {cfg.job_queue} → ENABLED')


# ---------- helpers ---------------------------------------------------------


def _account_id() -> str:
    """Return the caller's AWS account id."""
    return boto3.client('sts').get_caller_identity()['Account']


# ---------- CLI -------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the infra CLI.

    Returns:
        parser (argparse.ArgumentParser): the configured parser; each
            subcommand sets its handler via func.
    """
    p = argparse.ArgumentParser(
        prog='python -m glow._extra.aws.infra',
        description='Idempotent AWS Batch provisioning for glow._extra.aws.')
    p.add_argument('--config', default=DEFAULT_CONFIG_PATH,
                   help=f'path to AWSConfig JSON (default: {DEFAULT_CONFIG_PATH})')
    subs = p.add_subparsers(dest='cmd', required=True)

    sp = subs.add_parser('bootstrap',
                         help='create one-time IAM roles (needs IAM admin)')
    sp.set_defaults(func=cmd_bootstrap)

    sp = subs.add_parser(
        'setup',
        help='provision S3/ECR/Batch resources; pass --build to (re)deploy '
             'the worker image')
    sp.add_argument('--build', action='store_true',
                    help='build the worker image from the Dockerfile and push '
                         f'it to ECR (tagged {DEFAULT_IMAGE_TAG} unless '
                         '--image-tag is also given). This is the one-step '
                         'path to (re)deploy worker code after a change, and '
                         'needs no pre-built image.')
    sp.add_argument('--image-tag', default=None,
                    help='skip the build and push this already-built local '
                         'image tag to ECR instead. With neither --build nor '
                         f'--image-tag, setup reuses the {DEFAULT_IMAGE_TAG} '
                         'image already in ECR.')
    sp.set_defaults(func=cmd_setup)

    sp = subs.add_parser('teardown', help='delete Batch resources')
    sp.add_argument('--yes', action='store_true',
                    help='required to actually delete anything')
    sp.add_argument('--delete-bucket', action='store_true',
                    help='also empty + delete the S3 bucket')
    sp.set_defaults(func=cmd_teardown)

    sp = subs.add_parser('status',
                         help='job counts, per-run progress, and failures')
    sp.add_argument('--label', default=None,
                    help='only show jobs whose name starts glow-<label>')
    sp.add_argument('--limit', type=int, default=5,
                    help='max failed runs to expand (default: 5)')
    sp.add_argument('--logs', action='store_true',
                    help="tail each failed child's CloudWatch log stream")
    sp.set_defaults(func=cmd_status)

    sp = subs.add_parser('clear_storage', help='delete S3 prefixes')
    sp.add_argument('--runs', action='store_true',
                    help='delete runs/ (per-run manifests)')
    sp.add_argument('--records', action='store_true',
                    help='delete records/ (the per-hash provenance records)')
    sp.add_argument('--cache', action='store_true',
                    help='delete cache/ (the synced joblib cache)')
    sp.add_argument('--yes', action='store_true')
    sp.set_defaults(func=cmd_clear_storage)

    sp = subs.add_parser('clear_jobs', help='terminate active jobs')
    sp.add_argument('--label', default=None,
                    help='only cancel jobs whose name starts glow-<label>')
    sp.add_argument('--yes', action='store_true',
                    help='required to actually terminate anything')
    sp.set_defaults(func=cmd_clear_jobs)

    sp = subs.add_parser('stage_hcp',
                         help='upload the local HCP reference data to S3')
    sp.set_defaults(func=cmd_stage_hcp)

    sp = subs.add_parser('pull',
                         help='download shared records + run_ana cache from '
                              'S3 (recover an interrupted run)')
    sp.set_defaults(func=cmd_pull)

    sp = subs.add_parser('pause', help='disable job queue dispatch')
    sp.set_defaults(func=cmd_pause)

    sp = subs.add_parser('resume', help='re-enable job queue dispatch')
    sp.set_defaults(func=cmd_resume)

    return p


# Default resource names written into a fresh config by `bootstrap`;
# each is overridable at the prompt.  s3_bucket must end up
# globally unique, hence the prompt rather than a silent default.
DEFAULT_CONFIG_NAMES = {
    's3_bucket': 'glow-experiments',
    'job_queue': 'glow-job-queue',
    'job_definition': 'glow-job-definition',
}


def _prompt_yes(question: str) -> bool:
    """Ask a yes/no question at the terminal, defaulting to yes on blank."""
    return input(f'{question} [Y/n] ').strip().lower() in ('', 'y', 'yes')


def _init_config(path: str) -> None:
    """Write a fresh AWSConfig JSON at path, prompting for resource names.

    Called by `bootstrap` when no config exists yet.  Prints the default
    bucket / queue / job-definition names and offers to accept them all; on
    a 'no', prompts for each (a blank answer keeps that field's default).

    Args:
        path (str): where to write the AWSConfig JSON
    """
    print(f'[bootstrap] no config at {path}; creating one.')
    print('  default names:')
    for key, default in DEFAULT_CONFIG_NAMES.items():
        print(f'    {key} = {default}')

    names = dict(DEFAULT_CONFIG_NAMES)
    if not _prompt_yes('use these defaults?'):
        for key, default in DEFAULT_CONFIG_NAMES.items():
            names[key] = input(f'  {key} [{default}]: ').strip() or default

    AWSConfig(**names).to_file(path)
    print(f'[bootstrap] wrote {path}')


def main(argv=None):
    args = _build_parser().parse_args(argv)
    # bootstrap is the entry point on a fresh machine, so it creates the
    # config if absent; later commands expect it to already exist.
    if args.cmd == 'bootstrap' and not Path(args.config).exists():
        _init_config(args.config)
    cfg = AWSConfig.from_file(args.config)
    args.func(args, cfg)


if __name__ == '__main__':
    main()
