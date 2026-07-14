"""infra CLI: pull recovery + Spot compute-environment setup.

pull is the download counterpart of the workers' upload sync (sync.sync_pairs):
it pulls records/ and the run_ana cache back to their local dirs, so a
cancelled or interrupted run's partial results are recoverable. The test drives
the real s3.download_prefix against FakeS3, with sync_pairs redirected to a tmp
dir so no real user-data dir is touched.

The compute-environment tests drive _setup_compute_environment against small
batch / ec2 fakes to pin the AZ-spread contract: a fresh environment is created
across every default subnet, and an existing single-AZ environment is grown to
all zones in place (while an already-spread one is left untouched).
"""

from types import SimpleNamespace

from glow._extra.aws import infra, sync
from glow._extra.aws.config import AWSConfig

from .fakes import FakeS3, client_factory


def test_pull_downloads_sync_pairs(tmp_path, monkeypatch):
    bucket = 'glow-experiments'
    fake_s3 = FakeS3()
    fake_s3.store[(bucket, 'records/abc.json')] = b'{"hash": "abc"}'
    fake_s3.store[(bucket, 'cache/run_ana/def/output.pkl')] = b'blob'

    rec_dir = tmp_path / 'records'
    cache_dir = tmp_path / 'cache'
    monkeypatch.setattr(infra.boto3, 'client',
                        client_factory(fake_s3, None))
    monkeypatch.setattr(sync, 'sync_pairs',
                        lambda prefix: [(rec_dir, 'records'),
                                        (cache_dir, 'cache')])

    cfg = AWSConfig(s3_bucket=bucket, job_queue='q', job_definition='d')
    infra.cmd_pull(SimpleNamespace(), cfg)

    assert (rec_dir / 'abc.json').read_bytes() == b'{"hash": "abc"}'
    assert (cache_dir / 'run_ana/def/output.pkl').read_bytes() == b'blob'


def test_pull_is_registered_in_cli():
    args = infra._build_parser().parse_args(['pull'])
    assert args.func is infra.cmd_pull


# ---------- compute-environment AZ spread ----------------------------------

# three default subnets, deliberately out of AZ order, in one VPC
_SUBNETS = [
    {'SubnetId': 'subnet-c', 'AvailabilityZone': 'us-east-1c', 'VpcId': 'vpc-1'},
    {'SubnetId': 'subnet-a', 'AvailabilityZone': 'us-east-1a', 'VpcId': 'vpc-1'},
    {'SubnetId': 'subnet-b', 'AvailabilityZone': 'us-east-1b', 'VpcId': 'vpc-1'},
]
_SORTED_IDS = ['subnet-a', 'subnet-b', 'subnet-c']


class _FakeEC2:
    """Serves default subnets and the default security group."""

    def __init__(self, subnets, sg_id='sg-default'):
        self._subnets = subnets
        self._sg_id = sg_id

    def describe_subnets(self, *, Filters):
        return {'Subnets': self._subnets}

    def describe_security_groups(self, *, Filters):
        return {'SecurityGroups': [{'GroupId': self._sg_id}]}


class _FakeBatchCE:
    """Scripts one existing CE (or none) and records create/update calls."""

    def __init__(self, existing=None):
        self.existing = existing
        self.created = []
        self.updated = []

    def describe_compute_environments(self, *, computeEnvironments):
        envs = [] if self.existing is None else [self.existing]
        return {'computeEnvironments': envs}

    def create_compute_environment(self, **kwargs):
        self.created.append(kwargs)
        return {'computeEnvironmentArn': 'arn:ce'}

    def update_compute_environment(self, **kwargs):
        self.updated.append(kwargs)
        return {}


def _factory(batch, ec2):
    """boto3.client stand-in routing 'batch' / 'ec2' to the fakes."""
    def _client(service, **kwargs):
        if service == 'batch':
            return batch
        if service == 'ec2':
            return ec2
        raise ValueError(f'unexpected service {service!r}')
    return _client


def test_setup_ce_grows_single_az_to_all_zones(monkeypatch):
    ec2 = _FakeEC2(_SUBNETS)
    batch = _FakeBatchCE(existing={
        'computeEnvironmentArn': 'arn:ce',
        'computeResources': {'subnets': ['subnet-a']}})
    monkeypatch.setattr(infra.boto3, 'client', _factory(batch, ec2))

    cfg = AWSConfig(s3_bucket='b', job_queue='q', job_definition='d',
                    max_concurrent=4000)
    arn = infra._setup_compute_environment(cfg, account_id='123')

    assert arn == 'arn:ce'
    assert not batch.created
    cr = batch.updated[0]['computeResources']
    assert cr['subnets'] == _SORTED_IDS
    assert cr['maxvCpus'] == 4000


def test_setup_ce_leaves_already_spread_subnets_untouched(monkeypatch):
    ec2 = _FakeEC2(_SUBNETS)
    batch = _FakeBatchCE(existing={
        'computeEnvironmentArn': 'arn:ce',
        'computeResources': {'subnets': list(_SORTED_IDS)}})
    monkeypatch.setattr(infra.boto3, 'client', _factory(batch, ec2))

    cfg = AWSConfig(s3_bucket='b', job_queue='q', job_definition='d')
    infra._setup_compute_environment(cfg, account_id='123')

    # spanning every AZ already -> only maxvCpus, no infra-churning subnet swap
    assert 'subnets' not in batch.updated[0]['computeResources']


def test_setup_ce_creates_across_all_zones(monkeypatch):
    ec2 = _FakeEC2(_SUBNETS)
    batch = _FakeBatchCE(existing=None)
    monkeypatch.setattr(infra.boto3, 'client', _factory(batch, ec2))
    monkeypatch.setattr(infra, '_wait_for_ce_valid', lambda *a, **k: None)

    cfg = AWSConfig(s3_bucket='b', job_queue='q', job_definition='d')
    infra._setup_compute_environment(cfg, account_id='123')

    cr = batch.created[0]['computeResources']
    assert cr['subnets'] == _SORTED_IDS
    assert cr['type'] == 'SPOT'
