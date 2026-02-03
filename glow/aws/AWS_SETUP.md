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
- Compute environment + job queue
- Job definition
- `.glow_aws_config`

It also creates a monitoring policy for instance-type tracking and attempts to attach it to your current IAM user.

---

## Quick Smoke Test
```bash
python test/run_aws_test.py
```

## Cleanup
Use AWS Console or `./glow/aws/cleanup_aws.sh` to clear queues and S3 storage.
