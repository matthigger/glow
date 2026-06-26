"""AWS Batch driver -- DISABLED pending the benchmark overhaul.

The AWS driver was built against the benchmark's TrialCache / DataSource
caching layer (glow._extra.benchmark.trial_cache, glow._extra.benchmark.data), which the
benchmark-overhaul branch is replacing with a joblib.Memory-backed approach.
Those modules have been removed, so the driver/datasource modules no longer
import. They are left on disk for reference but are not wired up here; the
public entry points raise NotImplementedError until they are rewritten
against the new cache layer.
"""


def _disabled(*args, **kwargs):
    raise NotImplementedError(
        'glow._extra.aws is disabled pending the benchmark overhaul: the AWS driver '
        'was built on the removed TrialCache / DataSource caching layer and '
        'must be rewritten against the new joblib.Memory-backed benchmark.')


driver_aws = driver_aws_multi = _disabled

__all__ = ['driver_aws', 'driver_aws_multi']
