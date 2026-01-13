# AWS Setup Guide for GLOW Cloud Computing

Complete guide to setting up AWS infrastructure for running GLOW analyses in the cloud.

---

## Quick Start (Recommended)

### Manual Prerequisites (Required - Do These First)

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

#### 4. Create ECR Repository and Build/Push Docker Image
```bash
# Create ECR repository
aws ecr create-repository --repository-name glow-worker --region us-east-1

# Get your account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com

# Build Docker image
cd /path/to/glow/src
docker build -t glow-worker .

# Tag image
docker tag glow-worker:latest \
    ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/glow-worker:latest

# Push to ECR
docker push ${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/glow-worker:latest
```

---

### Automated Setup (Run This Script)

After completing the manual prerequisites above, run the automated setup script:

```bash
# Make script executable
chmod +x setup_aws_batch.sh

# Run setup
./setup_aws_batch.sh

# Optional: specify custom S3 bucket name
GLOW_S3_BUCKET=my-custom-bucket ./setup_aws_batch.sh
```

**What the script does:**
- ✓ Creates S3 bucket
- ✓ Creates all IAM roles (5 roles)
- ✓ Gets VPC/subnet/security group info
- ✓ Creates Batch compute environment
- ✓ Creates job queue
- ✓ Registers job definition
- ✓ Outputs configuration for Python

**Time:** ~2-3 minutes

---

## Manual Step-by-Step Setup (Reference)

If you prefer to run commands manually or need to troubleshoot, follow these detailed steps:

### 1. Install AWS CLI

```bash
pip install awscli boto3

# Configure with your credentials
aws configure
```

### 2. Create S3 Bucket

```bash
# Create bucket
aws s3 mb s3://my-glow-experiments --region us-east-1

# Enable versioning (recommended)
aws s3api put-bucket-versioning \
    --bucket my-glow-experiments \
    --versioning-configuration Status=Enabled

# Set lifecycle policy (auto-delete old results after 30 days)
cat > lifecycle.json << 'EOF'
{
    "Rules": [{
        "ID": "DeleteOldResults",
        "Status": "Enabled",
        "Prefix": "experiments/",
        "Expiration": {"Days": 30}
    }]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
    --bucket my-glow-experiments \
    --lifecycle-configuration file://lifecycle.json
```

### 3. Build and Push Docker Image

```bash
# Create ECR repository
aws ecr create-repository --repository-name glow-worker --region us-east-1

# Get login command
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build image
cd /home/matt/Dropbox/glow/src
docker build -t glow-worker .

# Tag image
docker tag glow-worker:latest \
    <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/glow-worker:latest

# Push to ECR
docker push <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/glow-worker:latest
```

### 4. Create IAM Roles

#### Batch Service Role

```bash
# Create trust policy
cat > batch-trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "batch.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}
EOF

# Create role
aws iam create-role \
    --role-name GlowBatchServiceRole \
    --assume-role-policy-document file://batch-trust-policy.json

# Attach AWS managed policy
aws iam attach-role-policy \
    --role-name GlowBatchServiceRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole
```

#### ECS Task Execution Role

```bash
# Create trust policy
cat > ecs-trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "ecs-tasks.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}
EOF

# Create role
aws iam create-role \
    --role-name GlowEcsTaskExecutionRole \
    --assume-role-policy-document file://ecs-trust-policy.json

# Attach AWS managed policy
aws iam attach-role-policy \
    --role-name GlowEcsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

#### ECS Task Role (for S3 access)

```bash
# Create inline policy for S3 access
cat > s3-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Action": [
            "s3:GetObject",
            "s3:PutObject",
            "s3:ListBucket"
        ],
        "Resource": [
            "arn:aws:s3:::my-glow-experiments",
            "arn:aws:s3:::my-glow-experiments/*"
        ]
    }]
}
EOF

# Create role
aws iam create-role \
    --role-name GlowEcsTaskRole \
    --assume-role-policy-document file://ecs-trust-policy.json

# Attach S3 policy
aws iam put-role-policy \
    --role-name GlowEcsTaskRole \
    --policy-name GlowS3Access \
    --policy-document file://s3-policy.json
```

#### EC2 Instance Role (for Batch compute environment)

**Important:** This role is required for EC2-based compute environments.

```bash
# Create EC2 trust policy
cat > ec2-trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "ec2.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}
EOF

# Create role
aws iam create-role \
    --role-name ecsInstanceRole \
    --assume-role-policy-document file://ec2-trust-policy.json

# Attach AWS managed policy
aws iam attach-role-policy \
    --role-name ecsInstanceRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role

# Create instance profile
aws iam create-instance-profile \
    --instance-profile-name ecsInstanceRole

# Add role to instance profile
aws iam add-role-to-instance-profile \
    --instance-profile-name ecsInstanceRole \
    --role-name ecsInstanceRole

# Wait for IAM propagation
sleep 30
```

#### Spot Fleet Role (for spot instances)

```bash
# Create spot fleet trust policy
cat > spot-trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "spotfleet.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}
EOF

# Create role
aws iam create-role \
    --role-name aws-ec2-spot-fleet-tagging-role \
    --assume-role-policy-document file://spot-trust-policy.json

# Attach AWS managed policy
aws iam attach-role-policy \
    --role-name aws-ec2-spot-fleet-tagging-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole
```

### 5. Create Compute Environment

**Important:** The `--compute-resources` parameter must be valid JSON.

#### Option A: Automatic (Recommended)

This script automatically detects your VPC, subnet, and security group:

```bash
#!/bin/bash

# Get your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Get VPC and network info
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query "Vpcs[0].VpcId" --output text)
SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=default-for-az,Values=true" --query "Subnets[0].SubnetId" --output text)
SG_ID=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=default" --query "SecurityGroups[0].GroupId" --output text)

echo "Using Account ID: $ACCOUNT_ID"
echo "Using VPC: $VPC_ID"
echo "Using Subnet: $SUBNET_ID"
echo "Using Security Group: $SG_ID"

# Create compute environment (with spot instances)
aws batch create-compute-environment \
  --compute-environment-name glow-compute-env-spot \
  --type MANAGED \
  --state ENABLED \
  --service-role arn:aws:iam::$ACCOUNT_ID:role/GlowBatchServiceRole \
  --compute-resources "{
    \"type\": \"SPOT\",
    \"allocationStrategy\": \"SPOT_CAPACITY_OPTIMIZED\",
    \"minvCpus\": 0,
    \"maxvCpus\": 256,
    \"desiredvCpus\": 0,
    \"instanceTypes\": [\"optimal\"],
    \"subnets\": [\"$SUBNET_ID\"],
    \"securityGroupIds\": [\"$SG_ID\"],
    \"instanceRole\": \"arn:aws:iam::$ACCOUNT_ID:instance-profile/ecsInstanceRole\",
    \"spotIamFleetRole\": \"arn:aws:iam::$ACCOUNT_ID:role/aws-ec2-spot-fleet-tagging-role\"
  }"
```

#### Option B: Manual

First, get your network IDs:

```bash
# Get VPC ID
aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query "Vpcs[0].VpcId" --output text

# Get subnet ID
aws ec2 describe-subnets --filters "Name=default-for-az,Values=true" --query "Subnets[0].SubnetId" --output text

# Get security group ID (replace VPC_ID)
aws ec2 describe-security-groups --filters "Name=vpc-id,Values=VPC_ID" "Name=group-name,Values=default" --query "SecurityGroups[0].GroupId" --output text
```

Then create the compute environment (replace `SUBNET_ID`, `SG_ID`, and `ACCOUNT_ID`):

```bash
aws batch create-compute-environment \
  --compute-environment-name glow-compute-env-spot \
  --type MANAGED \
  --state ENABLED \
  --service-role arn:aws:iam::ACCOUNT_ID:role/GlowBatchServiceRole \
  --compute-resources '{
    "type": "SPOT",
    "allocationStrategy": "SPOT_CAPACITY_OPTIMIZED",
    "minvCpus": 0,
    "maxvCpus": 256,
    "desiredvCpus": 0,
    "instanceTypes": ["optimal"],
    "subnets": ["SUBNET_ID"],
    "securityGroupIds": ["SG_ID"],
    "instanceRole": "arn:aws:iam::ACCOUNT_ID:instance-profile/ecsInstanceRole",
    "spotIamFleetRole": "arn:aws:iam::ACCOUNT_ID:role/aws-ec2-spot-fleet-tagging-role"
  }'
```

**Note:** The compute resources parameter must be valid JSON with:
- Arrays for `instanceTypes`, `subnets`, `securityGroupIds`
- Proper quote escaping when using shell variables
- No trailing commas

### 6. Create Job Queue

```bash
aws batch create-job-queue \
    --job-queue-name glow-job-queue \
    --state ENABLED \
    --priority 1 \
    --compute-environment-order order=1,computeEnvironment=glow-compute-env-spot
```

### 7. Create Job Definition

```bash
aws batch register-job-definition \
    --job-definition-name glow-job-definition \
    --type container \
    --platform-capabilities FARGATE \
    --container-properties '{
        "image": "<YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/glow-worker:latest",
        "jobRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/GlowEcsTaskRole",
        "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/GlowEcsTaskExecutionRole",
        "resourceRequirements": [
            {"type": "VCPU", "value": "2"},
            {"type": "MEMORY", "value": "4096"}
        ],
        "fargatePlatformConfiguration": {
            "platformVersion": "LATEST"
        }
    }'
```

### 8. Test Setup

```bash
# Submit a test job
aws batch submit-job \
    --job-name test-glow-job \
    --job-queue glow-job-queue \
    --job-definition glow-job-definition
```

## Configuration Summary

After setup, you'll use these values in your `CloudConfig`:

```python
from glow.aws import CloudConfig

cloud_config = CloudConfig(
    s3_bucket='my-glow-experiments',          # From step 2
    s3_prefix='experiments/test',
    job_queue='glow-job-queue',               # From step 6
    job_definition='glow-job-definition:1',   # From step 7
    region='us-east-1',
    max_concurrent_jobs=100,
    max_cost_per_hour=10.0,
    use_spot=True,
    timeout_minutes=60,
    memory_mb=4096,
    vcpus=2
)
```

## Cost Optimization

### Use Spot Instances

Spot instances are ~70% cheaper than on-demand:

```python
cloud_config.use_spot = True  # Recommended!
```

### Set Appropriate Resources

Don't over-provision:

```python
# For small experiments (< 10k voxels)
cloud_config.vcpus = 1
cloud_config.memory_mb = 2048

# For medium experiments (10k-100k voxels)
cloud_config.vcpus = 2
cloud_config.memory_mb = 4096

# For large experiments (> 100k voxels)
cloud_config.vcpus = 4
cloud_config.memory_mb = 8192
```

### Set Budget Alerts

```bash
# Create budget
aws budgets create-budget \
    --account-id <ACCOUNT_ID> \
    --budget file://budget.json \
    --notifications-with-subscribers file://notifications.json
```

budget.json:
```json
{
    "BudgetName": "GLOW-Monthly",
    "BudgetLimit": {
        "Amount": "50",
        "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
}
```

## Monitoring

### CloudWatch Logs

View job logs:
```bash
aws logs tail /aws/batch/job --follow
```

### Check Job Status

```bash
aws batch describe-jobs --jobs <JOB_ID>
```

### List Running Jobs

```bash
aws batch list-jobs --job-queue glow-job-queue --job-status RUNNING
```

## Cleanup

When done, delete resources to avoid charges:

```bash
# Disable job queue
aws batch update-job-queue \
    --job-queue glow-job-queue \
    --state DISABLED

# Disable compute environment
aws batch update-compute-environment \
    --compute-environment glow-compute-env-spot \
    --state DISABLED

# Delete job queue (after jobs complete)
aws batch delete-job-queue --job-queue glow-job-queue

# Delete compute environment
aws batch delete-compute-environment \
    --compute-environment glow-compute-env-spot

# Delete S3 bucket contents
aws s3 rm s3://my-glow-experiments --recursive

# Delete S3 bucket
aws s3 rb s3://my-glow-experiments
```

## Troubleshooting

### Docker Build Issues

#### "FileNotFoundError: Readme path `/app/README.md` does not exist"

The Dockerfile needs to copy README.md. Verify line 15 of Dockerfile:
```dockerfile
COPY README.md /app/
```

#### "Package 'glow' requires a different Python: 3.10.x not in '>=3.12'"

Update Dockerfile line 2 to use Python 3.12:
```dockerfile
FROM python:3.12-slim
```

#### "no basic auth credentials" when pushing to ECR

Login to ECR first:
```bash
aws ecr get-login-password --region us-east-1 | \
    sudo docker login --username AWS --password-stdin \
    <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
```

#### "An image does not exist locally with the tag"

Tag the image before pushing:
```bash
docker tag glow-worker:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/glow-worker:latest
```

### IAM Permission Errors

#### "User is not authorized to perform: iam:CreateRole"

Add IAMFullAccess to your user:
```bash
aws iam attach-user-policy --user-name glow --policy-arn arn:aws:iam::aws:policy/IAMFullAccess
```

#### "User is not authorized to perform: batch:CreateComputeEnvironment"

Add AWSBatchFullAccess to your user:
```bash
aws iam attach-user-policy --user-name glow --policy-arn arn:aws:iam::aws:policy/AWSBatchFullAccess
```

### AWS Batch Issues

#### "Compute environment stuck in CREATING"

Check your VPC/subnet/security group settings:
```bash
aws batch describe-compute-environments \
    --compute-environments glow-compute-env-spot
```

#### "Job fails immediately"

Check IAM roles have correct permissions:
```bash
aws iam get-role --role-name GlowEcsTaskRole
aws iam get-role-policy --role-name GlowEcsTaskRole --policy-name GlowS3Access
```

#### "Cannot pull Docker image"

Check ECR permissions:
```bash
aws ecr get-login-password --region us-east-1
```

#### "Unknown options" when creating compute environment

The `--compute-resources` parameter must be valid JSON. See Step 5 for correct syntax.

## Advanced: Terraform Setup

For infrastructure as code, see `terraform/` directory (coming soon).

## Support

- AWS Batch docs: https://docs.aws.amazon.com/batch/
- Issues: GitHub Issues
