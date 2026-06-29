"""Run the paper benchmarks on AWS Batch.

The AWS counterpart of the local benchmark sweep (glow._extra.benchmark): the
driver submits a CONFIG cache's data cells as a Batch array job (one child per
cell) and each worker rebuilds + runs its cell from CONFIG, writing its records
and run_ana cache to a shared S3 prefix; when the array drains, the records are
pulled down and the per-config CSVs written with the unchanged benchmark read
path. The work split needs no per-cell shipping -- the array index addresses a
cell, since the cell enumeration is a pure function of the catalogue.

The whole approach rests on the benchmark layer's content-addressed on-disk
state: the joblib cache and per-hash records are keyed by the call's args hash,
so the same call writes the same file on any machine. "Share the cache across
workers" therefore reduces to copying files to and from S3 (no live cache
backend, no locking); see the s3 / sync modules. A worker pulls the cheap,
high-value run_ana cache first, so a Spot-retried cell resumes from the fits it
already completed rather than from cold.

Modules:
    config   -- AWSConfig (bucket / queue / definition / memory tiers), JSON.
    units    -- enumerate a CONFIG cache's data cells (the unit of work).
    s3       -- generic S3 file mover + a background uploader.
    sync     -- which benchmark trees to mirror, as (local_dir, key) pairs.
    driver   -- drive_aws: submit, escalate OOM tiers, collect, write CSVs.
    worker   -- the Batch container entrypoint (run one cell).
    infra    -- idempotent boto3 provisioning CLI (bootstrap / setup / ...).

Running a sweep is the benchmark CLI's job
(``python -m glow._extra.benchmark --aws <names>`` calls drive_aws);
``python -m glow._extra.aws`` is the provisioning CLI alias (see __main__ /
infra).  boto3 is imported lazily (only the driver / worker / infra need it),
so importing this package for AWSConfig alone is cheap.
"""

from .config import AWSConfig

__all__ = ['AWSConfig', 'drive_aws']


def __getattr__(name):
    # lazily expose drive_aws so `import glow._extra.aws` does not pull boto3
    # (the driver's dependency) just to reach AWSConfig.
    if name == 'drive_aws':
        from .driver import drive_aws
        return drive_aws
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
