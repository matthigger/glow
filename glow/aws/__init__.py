"""AWS Batch driver for the benchmark TrialCache abstraction.

driver_aws(trial_cache, run_fnc, aws_config) is a drop-in replacement
for glow.benchmark.driver.driver_local: pulls uncached trials from
the cache, runs each as one child of an AWS Batch array job, and writes
results back through trial_cache.save_result.

DataSourceS3 wraps a benchmark DataSource whose .exp the worker cannot
rebuild (a real-data source loading files the image does not carry, e.g.
HCP).  Before submitting, the driver builds an S3-shipped twin of each
cache (driver._to_s3_cache): it uploads each such source's exp once and
swaps in a DataSourceS3 that fetches from S3 on demand, while leaving
deterministic DataSourceWGN sources to rebuild on the worker from seed.
The twin's TrialCache.trial_alias_map redirects the swapped trial hashes
back to the originals, so the AWS and local runs share one results.csv.

Public surface is populated as the build progresses; see __all__.
"""

from .config import AWSConfig
from .datasource import DataSourceS3
from .driver import driver_aws, driver_aws_multi

__all__ = ['AWSConfig', 'DataSourceS3', 'driver_aws', 'driver_aws_multi']
