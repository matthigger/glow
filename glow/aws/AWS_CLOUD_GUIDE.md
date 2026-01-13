# AWS Cloud Computing Guide for GLOW

This guide shows how to run GLOW permutation analysis on AWS for massive speed and cost savings.

## Overview

**Problem:** Running 100+ permutations locally takes hours  
**Solution:** Run all permutations simultaneously on AWS Batch with spot instances

### Benefits
- ⚡ **100× faster**: All permutations run in parallel
- 💰 **~70% cheaper**: Spot instances vs on-demand  
- 🔄 **Fault-tolerant**: Automatic retries, idempotent operations
- 📊 **Cost control**: Built-in price limits and estimates

### Architecture

```
Local Machine                    AWS Cloud
    │                               │
    ├─ Upload experiment ──────────> S3 Bucket
    │                               │
    ├─ Submit jobs ─────────────────> AWS Batch
    │                               │
    │                           ┌───┴───┐
    │                           │  Job  │ (Perm 0)
    │                           │  Job  │ (Perm 1)
    │                           │  ...  │ (Perm N)
    │                           └───┬───┘
    │                               │
    ├─ Download results <────────── S3 Bucket
    │                               │
    └─ Complete analysis locally    │
```

## Quick Start

### 1. AWS Setup (One-Time)

```bash
# Install AWS CLI
pip install boto3 awscli

# Configure credentials
aws configure
# Enter your AWS Access Key ID
# Enter your AWS Secret Access Key
# Enter region: us-east-1
```

### 2. Create AWS Resources

See `AWS_SETUP.md` for detailed setup, or use the provided Terraform/CloudFormation templates.

**Minimum requirements:**
- S3 bucket for data storage
- AWS Batch compute environment (with spot instances)
- AWS Batch job queue
- AWS Batch job definition (Docker container)

### 3. Run Analysis on Cloud

```python
from glow.aws import CloudConfig
from glow.experiment import AnalysisGLOW
import glow

# Create experiment
exp = glow.experiment.Experiment.from_gauss(
    seed=0,
    shape=(100, 100),
    a=2,
    b=2,
    num_img=100
)

# Configure cloud
cloud_config = CloudConfig(
    s3_bucket='my-glow-bucket',
    s3_prefix='experiments/run1',
    job_queue='glow-job-queue',
    job_definition='glow-job-definition:1',
    region='us-east-1',
    max_concurrent_jobs=100,
    max_cost_per_hour=10.0,  # Safety limit
    use_spot=True  # 70% cost savings!
)

# Run analysis (automatically uses cloud)
ana = AnalysisGLOW(
    exp=exp,
    n_perm=100,
    n_perm_adj=50,
    n_perm_prune=200,
    alpha_fwer=0.05,
    alpha_prune=0.05,
    verbose=True,
    cloud_config=cloud_config  # This triggers cloud execution!
)

# Results are ready to use
print(f'Found {len(ana.effect_list)} effects')
```

## Cost Estimation

### Automatic Cost Estimates

Every job submission shows cost estimates:

```
AWS BATCH JOB SUBMISSION
============================================================
experiment: glow_abc12345
jobs to submit: 101
skipped (completed): 0

COST ESTIMATE:
  per job: $0.0023
  total: $0.23
  compute hours: 1.7
  using spot instances: True

RESOURCES PER JOB:
  vCPUs: 2
  memory: 4096 MB
  timeout: 60 min
============================================================
```

### Manual Cost Estimation

```python
from glow.aws import estimate_cost

# Estimate before running
cost = estimate_cost(
    n_jobs=100,
    runtime_minutes=5,
    memory_mb=4096,
    vcpus=2,
    use_spot=True
)

print(f"Estimated total cost: ${cost['total_cost']:.2f}")
print(f"Cost per job: ${cost['cost_per_job']:.4f}")
```

### Real-World Costs (2024 Pricing)

| Setup | Jobs | Runtime | Spot | Total Cost |
|-------|------|---------|------|------------|
| Small (100x100) | 100 | 2 min | Yes | **$0.15** |
| Medium (500x500) | 100 | 5 min | Yes | **$0.38** |
| Large (1000x1000) | 100 | 15 min | Yes | **$1.13** |
| Very Large (2000x2000) | 100 | 30 min | Yes | **$2.25** |

Compare to local execution:
- Local: 100 permutations × 5 min = 500 min = **8.3 hours**
- Cloud: 100 permutations in parallel = **5 minutes**

## Fault Tolerance

### Automatic Retries

AWS Batch automatically retries failed jobs:
- Spot instance interruptions → automatic retry
- Out of memory → automatic retry (if configured)
- Network errors → automatic retry

### Idempotent Operations

All operations are safe to re-run:
```python
# Run analysis
ana = AnalysisGLOW(exp, n_perm=100, cloud_config=config)

# Oops! Interrupted. Just run again:
ana = AnalysisGLOW(exp, n_perm=100, cloud_config=config)
# ✅ Automatically skips completed permutations
```

### Skip Completed Permutations

```python
# First run: processes all 100 permutations
ana1 = AnalysisGLOW(exp, n_perm=100, cloud_config=config)

# Second run: skips completed, only processes new ones
# (useful if interrupted or adding more permutations)
ana2 = AnalysisGLOW(exp, n_perm=200, cloud_config=config)
# ✅ Only processes permutations 101-200
```

## Advanced Usage

### Manual Control

For more control, use `AWSBatchRunner` directly:

```python
from glow.aws import AWSBatchRunner, CloudConfig

runner = AWSBatchRunner(cloud_config)

# 1. Upload experiment
experiment_id = 'my_experiment_1'
runner.upload_experiment(exp, ana_kwargs, experiment_id)

# 2. Submit jobs with dry run (cost estimate only)
submission = runner.submit_jobs(
    experiment_id=experiment_id,
    n_perm=100,
    dry_run=True  # Don't actually submit
)
print(f"Would cost: ${submission['cost_estimate']['total_cost']:.2f}")

# 3. Actually submit
submission = runner.submit_jobs(
    experiment_id=experiment_id,
    n_perm=100,
    dry_run=False
)

# 4. Monitor progress
runner.monitor_jobs(submission['job_ids'], poll_interval=30)

# 5. Download results
results = runner.download_results(
    experiment_id=experiment_id,
    n_perm=100,
    output_dir='./results'
)
```

### Cost Controls

```python
# Set hard limits
cloud_config = CloudConfig(
    ...
    max_concurrent_jobs=50,  # Never more than 50 at once
    max_cost_per_hour=5.0,   # Abort if estimated > $5/hr
    timeout_minutes=30,      # Kill jobs after 30 min
)

# Use on-demand (no interruptions, but 3× cost)
cloud_config.use_spot = False
```

### Monitoring from Terminal

```python
# Start analysis in background
import threading

def run_analysis():
    ana = AnalysisGLOW(exp, n_perm=100, cloud_config=config, verbose=False)

thread = threading.Thread(target=run_analysis)
thread.start()

# Monitor separately
from glow.aws import AWSBatchRunner
runner = AWSBatchRunner(cloud_config)
runner.monitor_jobs(job_ids, poll_interval=30)
```

## Troubleshooting

### "Job failed with exit code 1"

Check CloudWatch logs:
```bash
aws logs tail /aws/batch/job --follow
```

Common causes:
- Out of memory → increase `memory_mb`
- Timeout → increase `timeout_minutes`
- Missing dependencies → rebuild Docker image

### "Cost estimate too high"

Options:
1. Reduce `n_perm`
2. Reduce `memory_mb` or `vcpus`
3. Increase `max_cost_per_hour` limit
4. Use smaller experiments (reduce voxels)

### "Spot instance interrupted"

This is normal! AWS Batch automatically retries. The job will complete eventually.

To avoid interruptions entirely:
```python
cloud_config.use_spot = False  # 3× cost, but no interruptions
```

### "Results incomplete"

Check which permutations are missing:
```python
completed = runner.check_existing_results(experiment_id, n_perm=100)
missing = set(range(101)) - completed
print(f'Missing: {missing}')

# Re-run missing ones
submission = runner.submit_jobs(
    experiment_id=experiment_id,
    n_perm=100,
    skip_completed=True  # Only run missing
)
```

## Best Practices

### 1. Start Small

Test with small experiments first:
```python
# Test run: 5 permutations on small data
test_exp = glow.experiment.Experiment.from_gauss(seed=0, shape=(10,10), ...)
ana_test = AnalysisGLOW(test_exp, n_perm=5, cloud_config=config)
```

### 2. Use Dry Runs

Always check costs first:
```python
runner.submit_jobs(experiment_id, n_perm=100, dry_run=True)
```

### 3. Monitor Costs

Set up AWS Budgets to alert when costs exceed thresholds:
```bash
aws budgets create-budget --account-id YOUR_ACCOUNT_ID \
  --budget '{"BudgetName":"GLOW-Monthly","BudgetLimit":{"Amount":"50","Unit":"USD"},...}'
```

### 4. Clean Up

Delete old results to save storage costs:
```bash
aws s3 rm s3://my-glow-bucket/experiments/old-run --recursive
```

### 5. Use Appropriate Resources

Match resources to data size:

| Data Size | vCPUs | Memory | Typical Runtime |
|-----------|-------|--------|-----------------|
| < 10k voxels | 1 | 2 GB | 1-2 min |
| 10k-100k voxels | 2 | 4 GB | 2-5 min |
| 100k-1M voxels | 4 | 8 GB | 5-15 min |
| > 1M voxels | 8 | 16 GB | 15-30 min |

## Cost Comparison

### Local (32-core machine)

```
100 permutations, 5 min each
- Serial: 500 minutes = 8.3 hours
- Parallel (32 cores): 16 minutes
- Cost: $0 (but ties up your machine)
```

### AWS Cloud (spot instances)

```
100 permutations, 5 min each
- Parallel (100 jobs): 5 minutes
- Cost: $0.38
- Your machine is free!
```

**Winner:** Cloud is 3× faster and frees your machine for ~$0.40!

## Next Steps

1. Set up AWS resources (see `AWS_SETUP.md`)
2. Build Docker image (see `Dockerfile`)
3. Run test analysis
4. Scale to production workloads

## Support

- Issues: GitHub Issues
- Questions: GitHub Discussions
- Docs: This guide + `AWS_SETUP.md`
