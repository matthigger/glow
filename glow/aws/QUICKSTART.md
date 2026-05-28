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

cfg = AWSConfig.from_file('.glow_aws_config')
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

`bootstrap` also writes `.glow_aws_config` in the cwd: it prints default
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

# Stop dispatching (in-flight children keep running)
python -m glow.aws.infra pause
python -m glow.aws.infra resume

# Clean S3 between runs
python -m glow.aws.infra clean --jobs --yes               # drop manifests/results
python -m glow.aws.infra clean --jobs --datasource --yes  # also drop cached HCP exps

# Tear down Batch (keep bucket + cached results)
python -m glow.aws.infra teardown --yes
python -m glow.aws.infra teardown --yes --delete-bucket   # nuke everything
```

## Failure handling

- **Spot reclamation** is retried automatically by Batch — never visible to the
  driver.
- **Out-of-memory** trials are resubmitted at the next memory tier
  (`AWSConfig.memory_mb_tiers`, default `[2000, 4000, 8000]`).
- **Timeouts, crashes, and last-tier OOM** are logged and skipped; the failed
  trials reappear on the next `--aws` (or local) run, since the cache only
  re-dispatches what's still missing.
