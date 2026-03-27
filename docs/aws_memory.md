# Memory Estimation for AWS Batch Workers

With the streaming synthesis pipeline, both permutation and synthesis
workers have the same memory footprint.  Memory scales with experiment
dimensions, not the number of permutations.

## Per-worker memory model

Memory is estimated by a Lasso regression fitted on actual peak-RSS
measurements across a grid of `(num_vox, b, num_img)` values.  The
fitted coefficients are stored in
[`memory_permutation.json`](../glow/aws/memory_permutation.json) and loaded at runtime by
`AWSBatchRunner.estimate_memory_mb`.

To re-run the profiling benchmark and refit the model:

    python -m glow.benchmark.memory

The current model (R² = 0.96) retains these terms:

    est_mb = 164
           + 0.0008 × num_vox
           - 2.48   × b
           - 0.14   × num_img
           + 0.0012 × num_vox × b
           + 0.0001 × num_vox × num_img
           + 0.087  × b × num_img

The dominant terms are `num_vox × b` and `num_vox` (Ward clustering
and stat-array overhead that scales with voxel count).

## Experiment-worker memory model

A **second** regression is used for the **experiment worker** (paper/config
path), which runs the full experiment (`config.run_fnc(config, **kwargs)`)
in one process — all permutations and analyses serially.  The same script
[`glow.benchmark.memory`](../glow/benchmark/memory.py) has an **experiment**
profile:

    python -m glow.benchmark.memory --profile experiment

This runs a grid of `(num_vox, b, num_img, n_perm)`, measures peak RSS for
full `run_ana` in a subprocess, fits a Lasso regression, and saves
[`memory_experiment.json`](../glow/aws/memory_experiment.json).
`AWSBatchRunner.estimate_experiment_memory_mb` loads that model when
submitting experiment jobs.

**HCP vs WGN:** For **HCP**, the worker keeps the full `exp_orig` in memory
(loading from S3 each run would add ~12 s per job).  We therefore request
the **sum** of (1) full-dataset size and (2) the experiment regression
prediction, snapped to the nearest tier.  For **WGN**, only the regression
is used (the data is small).  See [aws_job_types.md](aws_job_types.md) § 3.

## Maximum subjects per tier

Assuming a whole-brain mask volume of **1450 cm³** (1,450,000 mm³),
so `num_vox = 1,450,000 / resolution³`.

### b = 1 (single imaging feature)

| Resolution | Voxels      | 2 GB  | 4 GB  | 8 GB  | 16 GB  |
|------------|-------------|-------|-------|-------|--------|
| 2.00 mm    | 181,000     | 142   | 335   | 719   | 1,489  |
| 1.50 mm    | 430,000     | 40    | 121   | 283   | 607    |
| 1.25 mm    | 742,000     | 9     | 56    | 149   | 336    |
| 1.00 mm    | 1,450,000   | -     | 12    | 60    | 155    |
| 0.80 mm    | 2,830,000   | -     | -     | 14    | 63     |

### b = 2 (e.g. HCP fa + md)

| Resolution | Voxels      | 2 GB  | 4 GB  | 8 GB  | 16 GB  |
|------------|-------------|-------|-------|-------|--------|
| 2.00 mm    | 181,000     | 122   | 312   | 694   | 1,457  |
| 1.50 mm    | 430,000     | 20    | 101   | 262   | 585    |
| 1.25 mm    | 742,000     | -     | 36    | 129   | 316    |
| 1.00 mm    | 1,450,000   | -     | -     | 40    | 135    |
| 0.80 mm    | 2,830,000   | -     | -     | -     | 43     |

Each cell is the maximum `num_img` (subjects) that fits in that tier.
A dash means even a single subject does not fit.

## OOM tier escalation

Each job is submitted with the memory tier predicted by the model
above (snapped up to the nearest tier: **2 GB**, **4 GB**, **8 GB**,
or **16 GB**).  If the job still OOMs at that tier it is automatically
resubmitted at the next tier up.  If it exceeds the 16 GB ceiling, a
`MemoryError` is raised and the run is stopped.
