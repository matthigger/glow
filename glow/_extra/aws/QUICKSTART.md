# `glow._extra.aws` — run the benchmark on AWS Batch

Run the paper benchmark on AWS Batch instead of your laptop. It's the same
benchmark CLI with `--aws`: each CONFIG cache's data cells go out as one Batch
array job (one child per cell), each worker rebuilds its cell from CONFIG and
runs it, writing its records to a shared S3 prefix, and when the array drains
the records are pulled down and the per-config CSVs written with the unchanged
read path — so AWS and local runs produce the same artifacts.

```bash
python -m glow._extra.benchmark --aws sweep_llr     # one cache (WGN cells)
python -m glow._extra.benchmark --aws 'sweep_*'     # fnmatch glob
python -m glow._extra.benchmark --aws               # every cache
python -m glow._extra.benchmark --list              # list cache names
```

That is the only way to launch a run — the same CLI as a local sweep, just
with `--aws`. `python -m glow._extra.aws` is the separate provisioning CLI
(setup, status, teardown — below); it does not run sweeps.

## How the shared state works

There is no live S3 cache backend. The benchmark's per-hash `<hash>.json`
records are content-addressed (keyed by the call's args hash), so the same call
writes the same file on any machine. Collecting results across workers is
therefore just copying files down from S3 (no locking, no merge):

- A worker runs one whole cell and pulls no shared cache — nothing another
  worker computed helps it. A background thread ships each finished record up
  every minute, so results land at the driver as workers go, and a
  Spot-interrupted worker has already shipped the records it finished.
- Spot-resume is per-cell, not a shared cache: a reclaimed worker tars its own
  partial progress (its records + the leaf fnc's cache) to one checkpoint
  object, and the retry restores just that — so a long cell interrupted mid-run
  resumes instead of recomputing from cold.
- The heavy WGN/HCP exp caches are *not* synced — they rebuild deterministically
  on the worker (a seed draw, or a nifti load from the staged data), cheaper
  than shipping tens of MB.

A cache runs whatever sources its CONFIG data grid declares (sources are a
CONFIG property, not a CLI knob). WGN cells run out of the box (they rebuild
from a seed on the worker). A cache with HCP cells needs the HCP reference data
staged to S3 once, or those cells fail. `stage_hcp` converts the niftis to a
compact per-feature npy bundle (a brain mask + one float32 array per feature,
exactly the arrays `from_search` loads, so the experiment hashes identically —
see `hcp.py`) and uploads it; a worker then pulls only the features its cell
uses and `data_factory_hcp` builds from the bundle (no niftis, no DUA prompt):

```bash
# one-time: build the npy bundle from the local niftis + upload it to S3
# (run on a box with the HCP data; any local HCP run downloads + extracts it)
python -m glow._extra.aws stage_hcp

# then any cache with HCP cells runs like normal (smoke spans WGN + HCP)
python -m glow._extra.benchmark --aws smoke
```

The heavy HCP exp caches are still not synced -- each worker rebuilds its exp
by loading the staged niftis (cheaper than shipping them); only the records are
mirrored both ways.

## One-time setup

By hand:

1. **Create an AWS account** (and a billing alert). In *IAM → Users*, create a
   user with **AdministratorAccess** and an access key — needed only for
   `bootstrap`; nothing afterward uses it.
2. **`aws configure`** with that key and your region (default `us-east-1`).
3. **Install Docker** (to build the worker image).

Then bootstrap the IAM roles Batch needs (idempotent). It also writes the
AWSConfig to your per-user config dir, prompting for the bucket / queue /
job-definition names (pick a globally-unique bucket you own — the worker's S3
access is scoped to it):

```bash
python -m glow._extra.aws bootstrap
```

Build the worker image and register the Batch resources in one step:

```bash
python -m glow._extra.aws setup --build
```

Two-step form if you'd rather build the image yourself:

```bash
docker build -t glow-worker:latest -f glow/_extra/aws/Dockerfile .
python -m glow._extra.aws setup --image-tag glow-worker:latest
```

Re-run `setup --build` whenever you change worker-baked code (anything the
worker imports: `glow/_extra/aws/worker.py`, `glow/_extra/benchmark/*`,
`glow/analysis/*`, …) to redeploy it.

## Daily workflow

```bash
# submit work (the benchmark CLI with --aws is the one way to run)
python -m glow._extra.benchmark --aws sweep_llr
python -m glow._extra.benchmark --aws 'sweep_*'

# monitor
python -m glow._extra.aws status
python -m glow._extra.aws status --logs            # + failed children's logs

# pause / resume dispatch (in-flight children keep running)
python -m glow._extra.aws pause
python -m glow._extra.aws resume

# terminate active jobs (cancels queued, kills in-flight)
python -m glow._extra.aws clear_jobs --yes

# delete shared S3 state
python -m glow._extra.aws clear_storage --runs --yes       # stale manifests
python -m glow._extra.aws clear_storage --records --yes    # force a cold rerun

# tear down Batch (keep the bucket)
python -m glow._extra.aws teardown --yes
python -m glow._extra.aws teardown --yes --delete-bucket          # nuke everything
```

`pause`/`resume` flip the **job queue** (`DISABLED`/`ENABLED`) losslessly —
in-flight children finish, queued ones park. `clear_jobs` terminates the
**jobs** themselves (a `--label` scopes it to one cache's `glow-<name>` jobs).

## Failure handling

- **Spot reclamation** is retried automatically by Batch, and the retry resumes
  from the reclaimed worker's per-cell checkpoint (only the recipes the reclaim
  cut short recompute).
- **Out-of-memory** cells are resubmitted at the next memory tier
  (`AWSConfig.memory_mb_tiers`, default `[4000, 8000, 16000]`).
- **Timeouts, crashes, and last-tier OOM** are reported and skipped; the local
  records are the source of truth for what is done, so rerunning re-submits
  only the incomplete cells and fills the gaps without touching finished work.
