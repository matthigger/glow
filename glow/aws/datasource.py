"""DataSourceS3: lazy S3-backed stand-in for a benchmark DataSource.

A wrapped exp is built locally once (whichever process called
from_source) and uploaded to a content-addressed S3 key derived from
value_id(inner_ds).  Workers receive only the URI and download on first
.exp access.

NOT a DataSource subclass: it doesn't carry a/a_nuisance/seed/extenter
and can't re-build from scratch — the on-disk exp IS the identity.  It
just satisfies the one part of the DataSource contract that run_fnc
actually uses: .exp.

The driver swaps this wrapper in for a real-data DataSource when building
a cache's S3-shipped twin (glow.aws.driver._to_s3_cache).  Its hash
therefore differs from the inner ds's; the twin cache's
TrialCache.trial_alias_map maps that swapped hash back to the original, so
save_result still writes the original trial's results.csv row.
"""

import urllib.parse
from typing import Tuple

import boto3
import cloudpickle
from botocore.exceptions import ClientError

from glow.aws.config import s3_key
from glow.util import HashBySlots, value_id


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Split an s3://bucket/key/path URI into (bucket, key)."""
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != 's3':
        raise ValueError(f'not an s3:// URI: {uri}')
    return parsed.netloc, parsed.path.lstrip('/')


class DataSourceS3(HashBySlots):
    """S3-backed exp source.  Exposes .exp like any DataSource.

    Attributes:
        s3_uri (str): s3://bucket/key of the uploaded exp pickle.
    """

    __slots__ = ('s3_uri',)

    # Cache the downloaded exp per URI so repeated .exp accesses in one
    # worker process don't re-download.  Mirrors DataSource._exp_cache.
    _exp_cache = {}

    def __init__(self, *, s3_uri: str):
        self.s3_uri = s3_uri

    @property
    def exp(self):
        """The Experiment for this source, downloaded from S3 and cached."""
        if self.s3_uri not in self._exp_cache:
            bucket, key = _parse_s3_uri(self.s3_uri)
            s3 = boto3.client('s3')
            body = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
            self._exp_cache[self.s3_uri] = cloudpickle.loads(body)
        return self._exp_cache[self.s3_uri]

    @classmethod
    def from_source(cls, inner_ds, *, bucket: str, prefix: str,
                    s3=None) -> 'DataSourceS3':
        """Build inner_ds.exp locally, upload to S3, return a wrapper.

        The S3 key is content-addressed:
        {prefix}/datasource/{value_id(inner_ds)}/exp.pkl.  head_object
        short-circuits the upload if the key already exists — important
        when several driver_aws runs (or several trials in one run)
        share the same source.

        Args:
            inner_ds (DataSource): source whose built exp is uploaded.
            bucket (str): destination S3 bucket.
            prefix (str): key prefix under the bucket.
            s3: boto3 S3 client; a default-region client if None.

        Returns:
            DataSourceS3: wrapper holding the uploaded exp's s3:// URI.
        """
        if s3 is None:
            s3 = boto3.client('s3')
        key = s3_key(prefix, 'datasource', value_id(inner_ds), 'exp.pkl')
        if not _object_exists(s3, bucket, key):
            body = cloudpickle.dumps(inner_ds.exp)
            s3.put_object(Bucket=bucket, Key=key, Body=body)
        return cls(s3_uri=f's3://{bucket}/{key}')


def _object_exists(s3, bucket: str, key: str) -> bool:
    """Whether an S3 object exists, via a head_object probe."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] in ('404', 'NoSuchKey', 'NotFound'):
            return False
        raise
