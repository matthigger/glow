"""In-memory boto3 stubs so the AWS layer is exercised without real AWS.

FakeS3 backs the object store with a dict and supports the calls the s3 /
driver / worker modules make (head/get/put + upload_file/download_file +
list_objects_v2 paginator + delete_objects). FakeBatch records submit_job
calls and returns scripted per-child statuses from describe_jobs, so the
driver's submit -> poll -> classify -> OOM-escalate loop can be driven with a
small script.
"""

from pathlib import Path

from botocore.exceptions import ClientError


class FakeS3:
    """In-memory S3 stub: head/get/put, file up/download, list, delete."""

    def __init__(self):
        self.store = {}
        self.calls = []

    def put_object(self, *, Bucket, Key, Body):
        self.calls.append(('put', Bucket, Key))
        self.store[(Bucket, Key)] = Body if isinstance(Body, bytes) else \
            Body.encode()

    def get_object(self, *, Bucket, Key):
        self.calls.append(('get', Bucket, Key))
        data = self.store[(Bucket, Key)]

        class _Body:
            def __init__(self, d): self._d = d
            def read(self): return self._d

        return {'Body': _Body(data)}

    def head_object(self, *, Bucket, Key):
        self.calls.append(('head', Bucket, Key))
        if (Bucket, Key) not in self.store:
            raise ClientError(
                {'Error': {'Code': '404', 'Message': 'Not Found'}},
                'HeadObject')
        return {}

    def upload_file(self, filename, Bucket, Key):
        self.calls.append(('upload_file', Bucket, Key))
        self.store[(Bucket, Key)] = Path(filename).read_bytes()

    def download_file(self, Bucket, Key, filename):
        self.calls.append(('download_file', Bucket, Key))
        Path(filename).write_bytes(self.store[(Bucket, Key)])

    def delete_objects(self, *, Bucket, Delete):
        for obj in Delete['Objects']:
            self.calls.append(('delete', Bucket, obj['Key']))
            self.store.pop((Bucket, obj['Key']), None)
        return {}

    def get_paginator(self, name):
        assert name == 'list_objects_v2'
        store = self.store

        class _Paginator:
            def paginate(self, *, Bucket, Prefix):
                contents = [{'Key': k}
                            for (b, k) in store if b == Bucket
                            and k.startswith(Prefix)]
                yield {'Contents': contents}

        return _Paginator()


class FakeBatch:
    """Records submit_job; describe_jobs returns scripted per-child statuses.

    submit_then[i] is the outcome list for the i-th submit_job (the i-th array
    job the driver submits): a list of per-child status dicts, indexed by
    array child index. So an OOM-then-success across two tiers is

        FakeBatch(submit_then=[
            [{'status': 'FAILED',
              'container': {'exitCode': 137, 'reason': 'OutOfMemoryError'}},
             {'status': 'SUCCEEDED'}],          # tier 0: child 0 OOMs
            [{'status': 'SUCCEEDED'}],          # tier 1: OOM child retried
        ])
    """

    def __init__(self, submit_then):
        self.submit_then = list(submit_then)
        self.submitted = []
        self._ids = iter(f'p-{i:03d}' for i in range(10_000))

    def submit_job(self, **kwargs):
        parent = next(self._ids)
        self.submitted.append({'parent_id': parent, **kwargs})
        return {'jobId': parent}

    def describe_jobs(self, *, jobs):
        out = []
        for job_id in jobs:
            if ':' in job_id:
                parent, idx_s = job_id.rsplit(':', 1)
                idx = int(idx_s)
            else:
                parent, idx = job_id, 0
            sub_idx = next(i for i, s in enumerate(self.submitted)
                           if s['parent_id'] == parent)
            status = self.submit_then[sub_idx][idx]
            out.append({'jobId': job_id, **status})
        return {'jobs': out}


def client_factory(fake_s3, fake_batch):
    """Return a boto3.client stand-in routing 's3'/'batch' to the fakes."""
    def _client(service, **kwargs):
        if service == 's3':
            return fake_s3
        if service == 'batch':
            return fake_batch
        raise ValueError(f'unexpected service {service!r}')
    return _client
