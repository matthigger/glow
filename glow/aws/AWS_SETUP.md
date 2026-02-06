## AWS Setup (Minimal)

This guide only covers steps that are not already automated by `./glow/aws/setup_aws_batch.sh`.

### 1. Create AWS Account
- https://aws.amazon.com/
- Enable billing (credit card)

### 2. Create IAM User (Console)
- Create a user (e.g., `glow`)
- Attach policies:
  - `IAMFullAccess`
  - `AWSBatchFullAccess`
  - `AmazonS3FullAccess`
  - `AmazonEC2ContainerRegistryFullAccess`
- Create access keys (CLI) and save them

### 3. Install AWS CLI + Configure
```bash
pip install awscli boto3
aws configure
```

### 4. Ensure ECR Repository Exists
`deploy_docker.sh` assumes the repository exists:
```bash
aws ecr create-repository --repository-name glow-worker --region us-east-1
```

### 5. Build + Push Docker Image
```bash
cd /path/to/glow/src
./glow/aws/deploy_docker.sh
```

### 6. Create AWS Batch Resources (Idempotent)
```bash
./glow/aws/setup_aws_batch.sh
```

This creates:
- S3 bucket
- IAM roles for ECS/Batch
- Compute environment (spot, `BEST_FIT_PROGRESSIVE`) with mixed instance families:
  - **Compute-optimized** (c5/c5a/c6i/c6a/c7i/c7a, ~2 GB/vCPU) — for initial 2 GB jobs
  - **General-purpose** (m5/m5a/m6i/m6a/m7i/m7a, ~4 GB/vCPU) — for 4 GB OOM retries
  - **Memory-optimized** (r5/r5a/r6i/r6a/r7i/r7a, ~8 GB/vCPU) — for 8-16 GB OOM retries
- Job queue + job definition
- `.glow_aws_config` (saved in the project root)
  - **OOM retries**: Jobs that run out of memory are automatically resubmitted with more memory: 2 → 4 → 8 → 16 GB (see `CloudConfig.oom_memory_mb_tiers`). `BEST_FIT_PROGRESSIVE` ensures each retry lands on an instance family matching its memory/vCPU ratio, avoiding wasted vCPUs.

It also creates a monitoring policy for instance-type tracking and attempts to attach it to your current IAM user.

---

## Quick Smoke Test
```bash
python test/run_aws_test.py
```

## Cleanup
Use AWS Console or `./glow/aws/cleanup_aws.sh` to clear queues and S3 storage.
