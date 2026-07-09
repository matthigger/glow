"""infra CLI: the pull subcommand recovers the worker-synced trees from S3.

pull is the download counterpart of the workers' upload sync (sync.sync_pairs):
it pulls records/ and the run_ana cache back to their local dirs, so a
cancelled or interrupted run's partial results are recoverable. The test drives
the real s3.download_prefix against FakeS3, with sync_pairs redirected to a tmp
dir so no real user-data dir is touched.
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
