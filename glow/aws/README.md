# `glow.aws` — AWS Batch driver for the benchmark `TrialCache`

`driver_aws(trial_cache, run_fnc, aws_config)` is a drop-in replacement for
`glow.benchmark.driver.driver_local`. It bundles each uncached trial to S3,
submits one Batch array job, polls children, and writes results back through
`trial_cache.save_result` — using the **same** `trial_hash` as the local
driver, so AWS and local runs share one `results.csv`.

```python
from glow.benchmark.paper.config import CACHE_BY_LABEL
from glow.aws import AWSConfig, driver_aws

cfg = AWSConfig.from_file('.glow_aws_config')
cache, run_fnc = CACHE_BY_LABEL['sweep_extent_wgn_n10']
driver_aws(cache, run_fnc, cfg)
```

Or from the paper CLI:

```bash
python -m glow.benchmark.paper --aws sweep_extent_wgn_n10
python -m glow.benchmark.paper --aws 'sweep_*'        # glob
python -m glow.benchmark.paper --aws                  # everything
```

## How it works

```
TrialCache.iter_trial_no_repeat()
        │
        ├─ for each non-WGN ds: upload ds.exp once (content-addressed)
        │       └─ s3://<bucket>/<prefix>/datasource/<value_id(ds)>/exp.pkl
        │
        ├─ for each uncached trial:
        │       cloudpickle.dumps((run_fnc, worker_trial))
        │       └─ s3://<bucket>/<prefix>/jobs/<trial_hash>/job.pkl
        │
        ├─ manifest.pkl listing trial_hashes in array-index order
        │
        └─ Batch.submit_job(arrayProperties={size: N},
                            command=[python, -m, glow.aws.worker, manifest_uri])
                  │
                  ├─ worker reads AWS_BATCH_JOB_ARRAY_INDEX
                  ├─ loads manifest + job.pkl
                  ├─ run_fnc(**trial)
                  └─ uploads result.pkl
                          │
                          └─ driver polls; on SUCCEEDED downloads result.pkl
                             and calls trial_cache.save_result(result, ORIGINAL trial)
                             (NOT the worker-side ds-wrapped trial — so
                              trial_hash matches the local cache.)
```

**OOM resubmission**: when a child exits with code 137 + an "OutOfMemory"
statusReason, the driver resubmits failing trials at the next memory tier
(`memory_mb_tiers`, default `[2000, 4000, 8000]`).

**WGN sources skip upload**: `DataSourceWGN.exp` is reproducible from
`(seed, shape, b, num_img, extenter)` via `np.random.default_rng`, so workers
rebuild it locally. This saves S3 traffic and storage. HCP (and any future
non-deterministic source) **is** uploaded once per unique `ds`.

## One-time AWS setup

You do three things by hand; the CLI does the rest.

1. **Create an AWS account** (and set a billing alert). Sign in, open
   *IAM → Users*, create a user with **AdministratorAccess** and an access
   key. (Admin is needed only for the one-time `bootstrap` below; day-to-day
   work doesn't use it.)
2. **`aws configure`** with that access key and your region (default
   `us-east-1` — any region with c7i/c7a Spot capacity works).
3. **Install Docker** (used to build the worker image).

Then run the bootstrap, which creates every IAM role Batch needs — the
service-linked role, `ecsInstanceRole` + instance profile, the spot-fleet
role, and the two ECS task roles (the worker's S3 access is scoped to your
bucket). It's idempotent, so rerunning is safe:

```bash
python -m glow.aws.infra bootstrap
```

> Why a separate command? `bootstrap` is the only step that needs IAM-admin
> rights. After it runs, `setup` and the daily commands need only S3 / Batch
> / ECR permissions.

## Per-project setup (run by `glow.aws.infra`)

```bash
# Project-local config (cwd)
cat > .glow_aws_config <<'EOF'
{
  "s3_bucket": "glow-experiments",
  "job_queue": "glow-job-queue",
  "job_definition": "glow-job-definition"
}
EOF
# s3_prefix defaults to "" (objects land at the bucket root). Set it only
# to namespace multiple projects within one bucket, e.g. "s3_prefix": "paper2026".

# Build worker image + push to ECR + register Batch resources
docker build -t glow-worker:latest -f glow/aws/Dockerfile .
python -m glow.aws.infra setup --image-tag glow-worker:latest
```

Subsequent updates to the worker image (after code changes):

```bash
docker build -t glow-worker:latest -f glow/aws/Dockerfile .
python -m glow.aws.infra setup --image-tag glow-worker:latest   # re-pushes + updates job def
```

## Daily workflow

```bash
# Submit work
python -m glow.benchmark.paper --aws sweep_extent_wgn_n10
python -m glow.benchmark.paper --aws 'sweep_*'

# Monitor
python -m glow.aws.infra status
python -m glow.aws.infra status --label sweep_extent_wgn_n10

# Stop dispatching (in-flight children keep running)
python -m glow.aws.infra pause
python -m glow.aws.infra resume

# Clean S3 between runs
python -m glow.aws.infra clean --jobs --yes              # drop manifests/results
python -m glow.aws.infra clean --jobs --datasource --yes # also drop cached HCP exps

# Tear down Batch (keep bucket + cached results)
python -m glow.aws.infra teardown --yes
python -m glow.aws.infra teardown --yes --delete-bucket  # nuke everything
```

## Cost notes

- Allocation strategy is `SPOT_PRICE_CAPACITY_OPTIMIZED` over c/m/r 6/7 series
  (large → 12xlarge). Spot pricing fluctuates; check `Cost Explorer` after
  the first sizeable run.
- The driver doesn't estimate cost. If you need a per-run estimate, snapshot
  Cost Explorer before/after.
- Each `driver_aws` run uploads N `job.pkl` files (~KB each) + 1
  `manifest.pkl`. Result objects are downloaded and discarded from S3 only
  when `python -m glow.aws.infra clean --jobs --yes` is run.

## Failure handling

- **Spot reclamation**: AWS Batch's `retryStrategy.attempts=3` (configurable
  via `AWSConfig.retry_attempts`) handles this transparently — never visible
  to the driver.
- **OOM**: detected via `exitCode in (134, 137)` + `OutOfMemory`-ish text in
  `statusReason` or `container.reason`. Timeout kills also exit 137, so the
  detector checks for "duration" + "timeout" first.
- **Timeouts / crashes**: logged + skipped. The failed `trial_hash` shows up
  on the next `driver_aws` (or `driver_local`) run via
  `iter_trial_no_repeat`.
- **Last-tier OOM**: same — logged + skipped; user can bump `memory_mb_tiers`
  and rerun.

## Files

| File | Purpose |
|---|---|
| `config.py` | `AWSConfig` dataclass + JSON `from_file`/`to_file` |
| `datasource.py` | `DataSourceS3` — S3-backed stand-in for a `DataSource` |
| `driver.py` | `driver_aws` — orchestration + OOM tier escalation |
| `worker.py` | `python -m glow.aws.worker <manifest_uri>` |
| `infra.py` | `bootstrap` / `setup` / `teardown` / `status` / `clean` / `pause` / `resume` |
| `Dockerfile` | Worker container image |
