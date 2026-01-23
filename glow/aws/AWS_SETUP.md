# AWS Setup Guide for GLOW Cloud Computing

Quick setup guide to get GLOW running on AWS Batch.

#### 1. Create AWS Account
1. Go to https://aws.amazon.com/
2. Click "Create an AWS Account"
3. Follow the signup process
4. **Enable billing** (add credit card)

#### 2. Create IAM User (Do this in AWS Console)
1. Go to AWS Console → IAM → Users
2. Click "Create user"
3. Username: `glow` (or your preferred name)
4. Click "Next"
5. **Attach policies directly:**
   - ✓ `IAMFullAccess`
   - ✓ `AWSBatchFullAccess`
   - ✓ `AmazonS3FullAccess`
   - ✓ `AmazonEC2ContainerRegistryFullAccess`
6. Click "Create user"
7. Click on the user → Security credentials → "Create access key"
8. Choose "Command Line Interface (CLI)"
9. **Save your Access Key ID and Secret Access Key**

#### 3. Install AWS CLI
```bash
pip install awscli boto3

# Configure with your credentials
aws configure
# Enter your Access Key ID
# Enter your Secret Access Key
# Default region: us-east-1
# Default output format: json
```

#### 4. Create ECR Repository
```bash
# Create ECR repository
aws ecr create-repository --repository-name glow-worker --region us-east-1
```

#### 5: Build and Deploy Docker Image

```bash
cd /path/to/glow/src
./glow/aws/deploy_docker.sh
```

**What it does:**
- ✓ Builds Docker image (with validation)
- ✓ Verifies boto3 and worker module are installed
- ✓ Pushes to ECR
- ✓ Validates ECR image matches local build

#### 6: Setup AWS Infrastructure

```bash
# Run with defaults (1024 max concurrent jobs)
./glow/aws/setup_aws_batch.sh

# Or customize concurrency (edit glow/aws/setup_aws_batch.sh first):
# MAX_VCPUS=2048      # Max vCPUs for compute environment
# VCPUS_PER_JOB=1     # vCPUs per job
# MEMORY_PER_JOB=4096 # MB per job
```

**What it does:**
- ✓ Creates S3 bucket (idempotent)
- ✓ Creates all IAM roles (5 roles, idempotent)
- ✓ Gets VPC/subnet/security group info
- ✓ Creates Batch compute environment (configurable concurrency)
- ✓ Creates job queue
- ✓ Registers job definition
- ✓ Saves configuration to `.glow_aws_config`

**Configuration (default):**
- Max vCPUs: 2048
- vCPUs per job: 1  
- Max concurrent jobs: 2048

**Note:** The script is **fully idempotent** - you can run it multiple times safely. To customize, edit `MAX_VCPUS` or `VCPUS_PER_JOB` at the top of `glow/aws/setup_aws_batch.sh` and rerun.

---

## Next Steps

After setup, you're ready to run experiments:

```python
from glow.aws import CloudConfig
from glow.experiment import AnalysisGLOW
import glow

# Create experiment
exp = glow.experiment.Experiment.from_gauss(seed=0, shape=(100, 100), ...)

# Configure cloud (values from .glow_aws_config)
cloud_config = CloudConfig(
    s3_bucket='glow-experiments-...',  # from .glow_aws_config
    s3_prefix='experiments/run1',
    job_queue='glow-job-queue',
    job_definition='glow-job-definition:1',  # check latest revision
    region='us-east-1'
)

# Run analysis on cloud
ana = AnalysisGLOW(exp, n_perm=100, cloud_config=cloud_config)
```

## Cleanup

Disable or delete resources via AWS Console or use `./glow/aws/cleanup_aws.sh` to clear queues and S3 storage.
