# `glow.aws` — run the benchmark on AWS Batch

Run the paper benchmark on AWS Batch instead of your laptop. `driver_aws` is a
drop-in for the local driver: it dispatches each uncached trial as one child of
a Batch array job and writes results back to the same `results.csv`, so AWS and
local runs are interchangeable.

```bash
python -m glow.benchmark.paper --aws sweep_extent_wgn_n10
python -m glow.benchmark.paper --aws 'sweep_*'   # fnmatch glob
python -m glow.benchmark.paper --aws             # all caches
```

Or call it directly:

```python
from glow.aws import AWSConfig, driver_aws
from glow.benchmark.paper.config import CACHE_BY_LABEL

cfg = AWSConfig.from_file()  # default: per-user config dir (platformdirs)
cache, run_fnc = CACHE_BY_LABEL['sweep_extent_wgn_n10']
driver_aws(cache, run_fnc, cfg)
```

## One-time setup

First, by hand:

1. **Create an AWS account** (and set a billing alert). In *IAM → Users*, create
   a user with **AdministratorAccess** and an access key. Admin is needed only
   for `bootstrap` below; nothing afterward uses it.
2. **`aws configure`** with that access key and your region (default
   `us-east-1` — any region with c7i/c7a Spot capacity works).
3. **Install Docker** (to build the worker image).

Then bootstrap the IAM roles Batch needs (service-linked role, instance
profile, spot-fleet role, the two ECS task roles). It's idempotent:

```bash
python -m glow.aws.infra bootstrap
```

`bootstrap` also writes the AWSConfig to your per-user config directory
(located via platformdirs, so the exact path is OS-specific; bootstrap
prints where it wrote): it prints default
bucket / queue / job-definition names and lets you accept or override each.
Pick a globally-unique bucket you own — the worker's S3 access is scoped to it.

Finally, build the worker image and register the Batch resources:

```bash
docker build -t glow-worker:latest -f glow/aws/Dockerfile .
python -m glow.aws.infra setup --image-tag glow-worker:latest
```

## Daily workflow

```bash
# Submit work
python -m glow.benchmark.paper --aws sweep_extent_wgn_n10
python -m glow.benchmark.paper --aws 'sweep_*'

# Monitor
python -m glow.aws.infra status
python -m glow.aws.infra status --label sweep_extent_wgn_n10

# Pause / resume dispatch (in-flight children keep running)
python -m glow.aws.infra pause
python -m glow.aws.infra resume

# Clear jobs — terminate active jobs (cancels queued, kills in-flight)
python -m glow.aws.infra clear_jobs --yes
python -m glow.aws.infra clear_jobs --label sweep_extent_wgn_n10 --yes

# Clear storage — delete S3 between runs
python -m glow.aws.infra clear_storage --jobs --yes               # drop manifests/results
python -m glow.aws.infra clear_storage --jobs --datasource --yes  # also drop cached HCP exps

# Tear down Batch (keep bucket + cached results)
python -m glow.aws.infra teardown --yes
python -m glow.aws.infra teardown --yes --delete-bucket   # nuke everything
```

## Pausing, resuming, and clearing jobs

`pause` / `resume` flip the Batch **job queue** between `DISABLED` and
`ENABLED`; `clear_jobs` acts on the **jobs** themselves. They solve different
problems:

- **`pause`** disables the queue, so Batch stops placing queued jobs onto
  instances. Jobs already `STARTING`/`RUNNING` run to completion, and jobs
  waiting in `SUBMITTED`/`PENDING`/`RUNNABLE` simply park — nothing is lost,
  they keep their place in line. You can still submit more work while paused;
  it queues up. Use this to stop launching new Spot instances (the main cost
  driver) without discarding in-flight progress. Note it does **not** stop
  instances that are already running — only new placements.
- **`resume`** re-enables the queue; parked jobs start placing onto instances
  again. `pause` then `resume` is lossless round-trip.
- **`clear_jobs`** terminates the jobs. Queued jobs are cancelled and any
  `STARTING`/`RUNNING` containers are killed (they end up `FAILED`); their
  trials reappear on the next run, like any other failure. `clear_jobs` does
  not touch the queue state, so a later `--aws` run dispatches fresh children —
  pair it with `pause` if you want everything to stop and stay stopped. Scope
  it to one sweep with `--label`.

So: `pause` to idle the fleet without losing work, `clear_jobs` to actually
kill jobs, `clear_storage` to wipe S3 results, `teardown` to delete the Batch
resources entirely.

## Failure handling

- **Spot reclamation** is retried automatically by Batch — never visible to the
  driver.
- **Out-of-memory** trials are resubmitted at the next memory tier
  (`AWSConfig.memory_mb_tiers`, default `[2000, 4000, 8000]`).
- **Timeouts, crashes, and last-tier OOM** are logged and skipped; the failed
  trials reappear on the next `--aws` (or local) run, since the cache only
  re-dispatches what's still missing.
