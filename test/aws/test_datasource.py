"""DataSourceS3: upload, lazy download, HEAD dedup.

Unit tests use an in-memory fake S3 client.  A second @pytest.mark.runaws
test exercises the same flow against real S3 (skipped without --runaws).
"""

import os
import uuid
from unittest.mock import patch

import boto3
import numpy as np
import pytest
from botocore.exceptions import ClientError

from glow.aws.datasource import DataSourceS3, _parse_s3_uri
from glow.benchmark.data import DataSource, DataSourceWGN


class FakeS3:
    """In-memory S3 stub: head/get/put + call accounting."""

    def __init__(self):
        self.store = {}
        self.calls = []

    def put_object(self, *, Bucket, Key, Body):
        self.calls.append(('put', Bucket, Key))
        self.store[(Bucket, Key)] = Body

    def get_object(self, *, Bucket, Key):
        self.calls.append(('get', Bucket, Key))
        body = self.store[(Bucket, Key)]

        class _Body:
            def __init__(self, data): self._data = data
            def read(self): return self._data

        return {'Body': _Body(body)}

    def head_object(self, *, Bucket, Key):
        self.calls.append(('head', Bucket, Key))
        if (Bucket, Key) not in self.store:
            raise ClientError(
                {'Error': {'Code': '404', 'Message': 'Not Found'}},
                'HeadObject')
        return {}


def _wgn():
    DataSource._exp_cache.clear()
    return DataSourceWGN(seed=0, shape=(3, 3, 3), b=1, num_img=10)


def test_parse_s3_uri():
    assert _parse_s3_uri('s3://my-bucket/some/key.pkl') == (
        'my-bucket', 'some/key.pkl')
    with pytest.raises(ValueError, match='not an s3:// URI'):
        _parse_s3_uri('http://example.com/x')


def test_from_source_uploads_then_dedups():
    fake = FakeS3()
    ds = _wgn()

    w1 = DataSourceS3.from_source(
        ds, bucket='b', prefix='pre', s3=fake)
    head_then_put = [c[0] for c in fake.calls]
    assert head_then_put == ['head', 'put']

    fake.calls.clear()
    w2 = DataSourceS3.from_source(
        ds, bucket='b', prefix='pre', s3=fake)
    # dedup: head sees the object, no put
    assert [c[0] for c in fake.calls] == ['head']

    assert w1.s3_uri == w2.s3_uri
    assert w1.s3_uri.startswith('s3://b/pre/datasource/') \
        and w1.s3_uri.endswith('/exp.pkl')


def test_exp_roundtrips_via_fake_s3():
    fake = FakeS3()
    ds = _wgn()
    exp_local = ds.exp
    wrap = DataSourceS3.from_source(ds, bucket='b', prefix='pre', s3=fake)

    # Drop any local exp cache so the wrapper has to read from S3
    DataSourceS3._exp_cache.clear()
    with patch('glow.aws.datasource.boto3.client', return_value=fake):
        exp_remote = wrap.exp

    assert np.array_equal(exp_local.y, exp_remote.y)
    assert np.array_equal(exp_local.x, exp_remote.x)


def test_exp_cached_per_uri():
    """Second .exp access does not re-hit S3."""
    fake = FakeS3()
    ds = _wgn()
    wrap = DataSourceS3.from_source(ds, bucket='b', prefix='pre', s3=fake)

    DataSourceS3._exp_cache.clear()
    with patch('glow.aws.datasource.boto3.client', return_value=fake):
        a = wrap.exp
        b = wrap.exp

    assert a is b
    assert sum(1 for c in fake.calls if c[0] == 'get') == 1


@pytest.mark.parametrize('seed_a, seed_b, same_key', [
    # content-addressed key is derived from the inner value's identity, so
    # equal inner sources map to the same S3 key (even across separate fakes).
    (0, 0, True),
    (0, 1, False),
])
def test_different_inner_ds_different_key(seed_a, seed_b, same_key):
    fake1, fake2 = FakeS3(), FakeS3()
    DataSource._exp_cache.clear()
    a = DataSourceS3.from_source(
        DataSourceWGN(seed=seed_a, shape=(3, 3, 3), b=1, num_img=10),
        bucket='b', prefix='pre', s3=fake1)
    DataSource._exp_cache.clear()
    b = DataSourceS3.from_source(
        DataSourceWGN(seed=seed_b, shape=(3, 3, 3), b=1, num_img=10),
        bucket='b', prefix='pre', s3=fake2)
    if same_key:
        assert a.s3_uri == b.s3_uri
    else:
        assert a.s3_uri != b.s3_uri


# ---------- real-AWS test ---------------------------------------------------


@pytest.mark.runaws
def test_from_source_real_s3():
    """End-to-end with real S3.  Skipped without --runaws."""
    bucket = os.environ.get('GLOW_TEST_BUCKET')
    if not bucket:
        pytest.skip('set GLOW_TEST_BUCKET to run this test')
    prefix = f'test/{uuid.uuid4().hex[:8]}'
    s3 = boto3.client('s3')
    ds = _wgn()
    try:
        wrap = DataSourceS3.from_source(
            ds, bucket=bucket, prefix=prefix, s3=s3)
        DataSourceS3._exp_cache.clear()
        exp_remote = wrap.exp
        assert np.array_equal(ds.exp.y, exp_remote.y)
    finally:
        _, key = _parse_s3_uri(wrap.s3_uri)
        try:
            s3.delete_object(Bucket=bucket, Key=key)
        except Exception:
            pass
