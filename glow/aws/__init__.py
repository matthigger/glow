"""AWS Batch driver for the benchmark TrialCache abstraction.

driver_aws(trial_cache, run_fnc, aws_config) is a drop-in replacement
for glow.benchmark.driver.driver_local: pulls uncached trials from
the cache, runs each as one child of an AWS Batch array job, and writes
results back through trial_cache.save_result.

DataSourceS3 wraps a benchmark DataSource whose .exp is too expensive to
rebuild on the worker (e.g. HCP).  The driver builds the exp locally,
uploads it once per unique source, and the worker fetches from S3 on
demand.  Deterministic sources (DataSourceWGN) are left in the trial
dict and rebuilt on the worker from their seed.

Public surface is populated as the build progresses; see __all__.
"""

from .config import AWSConfig
from .datasource import DataSourceS3
from .driver import driver_aws

__all__ = ['AWSConfig', 'DataSourceS3', 'driver_aws']
