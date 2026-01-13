#!/bin/bash
set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGION=${AWS_REGION:-us-east-1}
S3_BUCKET=${GLOW_S3_BUCKET:-glow-experiments-$(date +%s)}
COMPUTE_ENV_NAME="glow-compute-env-spot"
JOB_QUEUE_NAME="glow-job-queue"
JOB_DEFINITION_NAME="glow-job-definition"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  GLOW AWS Batch Automated Setup${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Get AWS account ID
echo -e "${YELLOW}Getting AWS account information...${NC}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}✓ Account ID: ${ACCOUNT_ID}${NC}"
echo ""

# Check if Docker image exists in ECR
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/glow-worker"
echo -e "${YELLOW}Checking for Docker image in ECR...${NC}"
if aws ecr describe-images --repository-name glow-worker --region $REGION --image-ids imageTag=latest &>/dev/null; then
    echo -e "${GREEN}✓ Docker image found in ECR${NC}"
    DOCKER_IMAGE="${ECR_REPO}:latest"
else
    echo -e "${RED}✗ Docker image not found in ECR${NC}"
    echo -e "${YELLOW}Please build and push the Docker image first:${NC}"
    echo "  1. docker build -t glow-worker ."
    echo "  2. aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO"
    echo "  3. docker tag glow-worker:latest ${ECR_REPO}:latest"
    echo "  4. docker push ${ECR_REPO}:latest"
    exit 1
fi
echo ""

# ============================================================================
# STEP 1: Create S3 Bucket
# ============================================================================
echo -e "${BLUE}[1/7] Creating S3 bucket...${NC}"
if aws s3 ls "s3://${S3_BUCKET}" 2>/dev/null; then
    echo -e "${YELLOW}  Bucket ${S3_BUCKET} already exists${NC}"
else
    aws s3 mb "s3://${S3_BUCKET}" --region $REGION
    echo -e "${GREEN}✓ Created S3 bucket: ${S3_BUCKET}${NC}"
fi
echo ""

# ============================================================================
# STEP 2: Create IAM Roles
# ============================================================================
echo -e "${BLUE}[2/7] Creating IAM roles...${NC}"

# Batch Service Role
echo -e "${YELLOW}  Creating Batch service role...${NC}"
cat > /tmp/batch-trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "batch.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}
EOF

if aws iam get-role --role-name GlowBatchServiceRole &>/dev/null; then
    echo -e "${YELLOW}    GlowBatchServiceRole already exists${NC}"
else
    aws iam create-role \
        --role-name GlowBatchServiceRole \
        --assume-role-policy-document file:///tmp/batch-trust-policy.json \
        --description "AWS Batch service role for GLOW"
    aws iam attach-role-policy \
        --role-name GlowBatchServiceRole \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole
    echo -e "${GREEN}    ✓ Created GlowBatchServiceRole${NC}"
fi

# ECS Task Execution Role
echo -e "${YELLOW}  Creating ECS task execution role...${NC}"
cat > /tmp/ecs-trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "ecs-tasks.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}
EOF

if aws iam get-role --role-name GlowEcsTaskExecutionRole &>/dev/null; then
    echo -e "${YELLOW}    GlowEcsTaskExecutionRole already exists${NC}"
else
    aws iam create-role \
        --role-name GlowEcsTaskExecutionRole \
        --assume-role-policy-document file:///tmp/ecs-trust-policy.json \
        --description "ECS task execution role for GLOW"
    aws iam attach-role-policy \
        --role-name GlowEcsTaskExecutionRole \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
    echo -e "${GREEN}    ✓ Created GlowEcsTaskExecutionRole${NC}"
fi

# ECS Task Role (for S3 access)
echo -e "${YELLOW}  Creating ECS task role...${NC}"
cat > /tmp/s3-policy.json << EOF
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
            "arn:aws:s3:::${S3_BUCKET}",
            "arn:aws:s3:::${S3_BUCKET}/*"
        ]
    }]
}
EOF

if aws iam get-role --role-name GlowEcsTaskRole &>/dev/null; then
    echo -e "${YELLOW}    GlowEcsTaskRole already exists${NC}"
    # Update S3 policy
    aws iam put-role-policy \
        --role-name GlowEcsTaskRole \
        --policy-name GlowS3Access \
        --policy-document file:///tmp/s3-policy.json
else
    aws iam create-role \
        --role-name GlowEcsTaskRole \
        --assume-role-policy-document file:///tmp/ecs-trust-policy.json \
        --description "ECS task role for GLOW S3 access"
    aws iam put-role-policy \
        --role-name GlowEcsTaskRole \
        --policy-name GlowS3Access \
        --policy-document file:///tmp/s3-policy.json
    echo -e "${GREEN}    ✓ Created GlowEcsTaskRole${NC}"
fi

# EC2 Instance Role
echo -e "${YELLOW}  Creating EC2 instance role...${NC}"
cat > /tmp/ec2-trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "ec2.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}
EOF

if aws iam get-role --role-name ecsInstanceRole &>/dev/null; then
    echo -e "${YELLOW}    ecsInstanceRole already exists${NC}"
else
    aws iam create-role \
        --role-name ecsInstanceRole \
        --assume-role-policy-document file:///tmp/ec2-trust-policy.json \
        --description "EC2 instance role for ECS/Batch"
    aws iam attach-role-policy \
        --role-name ecsInstanceRole \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role
    echo -e "${GREEN}    ✓ Created ecsInstanceRole${NC}"
fi

# Create instance profile
if aws iam get-instance-profile --instance-profile-name ecsInstanceRole &>/dev/null; then
    echo -e "${YELLOW}    Instance profile already exists${NC}"
else
    aws iam create-instance-profile --instance-profile-name ecsInstanceRole
    aws iam add-role-to-instance-profile \
        --instance-profile-name ecsInstanceRole \
        --role-name ecsInstanceRole
    echo -e "${GREEN}    ✓ Created instance profile${NC}"
fi

# Spot Fleet Role
echo -e "${YELLOW}  Creating spot fleet role...${NC}"
cat > /tmp/spot-trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "spotfleet.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}
EOF

if aws iam get-role --role-name aws-ec2-spot-fleet-tagging-role &>/dev/null; then
    echo -e "${YELLOW}    aws-ec2-spot-fleet-tagging-role already exists${NC}"
else
    aws iam create-role \
        --role-name aws-ec2-spot-fleet-tagging-role \
        --assume-role-policy-document file:///tmp/spot-trust-policy.json \
        --description "Spot fleet role for EC2 spot instances"
    aws iam attach-role-policy \
        --role-name aws-ec2-spot-fleet-tagging-role \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole
    echo -e "${GREEN}    ✓ Created aws-ec2-spot-fleet-tagging-role${NC}"
fi

echo -e "${YELLOW}  Waiting 10 seconds for IAM propagation...${NC}"
sleep 10
echo -e "${GREEN}✓ All IAM roles created${NC}"
echo ""

# ============================================================================
# STEP 3: Get VPC Information
# ============================================================================
echo -e "${BLUE}[3/7] Getting VPC information...${NC}"
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query "Vpcs[0].VpcId" --output text --region $REGION)
SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=default-for-az,Values=true" --query "Subnets[0].SubnetId" --output text --region $REGION)
SG_ID=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=default" --query "SecurityGroups[0].GroupId" --output text --region $REGION)

echo -e "${GREEN}✓ VPC ID: ${VPC_ID}${NC}"
echo -e "${GREEN}✓ Subnet ID: ${SUBNET_ID}${NC}"
echo -e "${GREEN}✓ Security Group ID: ${SG_ID}${NC}"
echo ""

# ============================================================================
# STEP 4: Create Compute Environment
# ============================================================================
echo -e "${BLUE}[4/7] Creating compute environment...${NC}"

# Check if already exists
if aws batch describe-compute-environments --compute-environments $COMPUTE_ENV_NAME --region $REGION --query "computeEnvironments[0].computeEnvironmentName" --output text 2>/dev/null | grep -q "$COMPUTE_ENV_NAME"; then
    STATUS=$(aws batch describe-compute-environments --compute-environments $COMPUTE_ENV_NAME --region $REGION --query "computeEnvironments[0].status" --output text)
    echo -e "${YELLOW}  Compute environment already exists (status: $STATUS)${NC}"
    
    if [ "$STATUS" = "INVALID" ]; then
        echo -e "${YELLOW}  Deleting invalid compute environment...${NC}"
        aws batch update-compute-environment \
            --compute-environment $COMPUTE_ENV_NAME \
            --state DISABLED \
            --region $REGION || true
        sleep 10
        aws batch delete-compute-environment \
            --compute-environment $COMPUTE_ENV_NAME \
            --region $REGION || true
        echo -e "${YELLOW}  Waiting 30 seconds for deletion...${NC}"
        sleep 30
        STATUS=""
    fi
fi

if [ -z "$STATUS" ] || [ "$STATUS" = "INVALID" ]; then
    aws batch create-compute-environment \
        --compute-environment-name $COMPUTE_ENV_NAME \
        --type MANAGED \
        --state ENABLED \
        --region $REGION \
        --service-role "arn:aws:iam::${ACCOUNT_ID}:role/GlowBatchServiceRole" \
        --compute-resources "{
            \"type\": \"SPOT\",
            \"allocationStrategy\": \"SPOT_CAPACITY_OPTIMIZED\",
            \"minvCpus\": 0,
            \"maxvCpus\": 256,
            \"desiredvCpus\": 0,
            \"instanceTypes\": [\"optimal\"],
            \"subnets\": [\"${SUBNET_ID}\"],
            \"securityGroupIds\": [\"${SG_ID}\"],
            \"instanceRole\": \"arn:aws:iam::${ACCOUNT_ID}:instance-profile/ecsInstanceRole\",
            \"spotIamFleetRole\": \"arn:aws:iam::${ACCOUNT_ID}:role/aws-ec2-spot-fleet-tagging-role\"
        }"
    
    echo -e "${YELLOW}  Waiting for compute environment to become VALID...${NC}"
    for i in {1..60}; do
        STATUS=$(aws batch describe-compute-environments --compute-environments $COMPUTE_ENV_NAME --region $REGION --query "computeEnvironments[0].status" --output text)
        if [ "$STATUS" = "VALID" ]; then
            break
        fi
        if [ "$STATUS" = "INVALID" ]; then
            REASON=$(aws batch describe-compute-environments --compute-environments $COMPUTE_ENV_NAME --region $REGION --query "computeEnvironments[0].statusReason" --output text)
            echo -e "${RED}✗ Compute environment is INVALID: $REASON${NC}"
            exit 1
        fi
        echo -e "    Status: $STATUS (waiting... $i/60)"
        sleep 5
    done
    
    if [ "$STATUS" != "VALID" ]; then
        echo -e "${RED}✗ Timeout waiting for compute environment${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Compute environment created and VALID${NC}"
else
    echo -e "${GREEN}✓ Using existing compute environment${NC}"
fi
echo ""

# ============================================================================
# STEP 5: Create Job Queue
# ============================================================================
echo -e "${BLUE}[5/7] Creating job queue...${NC}"
if aws batch describe-job-queues --job-queues $JOB_QUEUE_NAME --region $REGION --query "jobQueues[0].jobQueueName" --output text 2>/dev/null | grep -q "$JOB_QUEUE_NAME"; then
    echo -e "${YELLOW}  Job queue already exists${NC}"
else
    aws batch create-job-queue \
        --job-queue-name $JOB_QUEUE_NAME \
        --state ENABLED \
        --priority 1 \
        --region $REGION \
        --compute-environment-order order=1,computeEnvironment=$COMPUTE_ENV_NAME
    echo -e "${GREEN}✓ Job queue created${NC}"
fi
echo ""

# ============================================================================
# STEP 6: Register Job Definition
# ============================================================================
echo -e "${BLUE}[6/7] Registering job definition...${NC}"
aws batch register-job-definition \
    --job-definition-name $JOB_DEFINITION_NAME \
    --type container \
    --region $REGION \
    --platform-capabilities EC2 \
    --container-properties "{
        \"image\": \"${DOCKER_IMAGE}\",
        \"jobRoleArn\": \"arn:aws:iam::${ACCOUNT_ID}:role/GlowEcsTaskRole\",
        \"executionRoleArn\": \"arn:aws:iam::${ACCOUNT_ID}:role/GlowEcsTaskExecutionRole\",
        \"vcpus\": 2,
        \"memory\": 4096
    }" > /dev/null

JOB_DEF_REVISION=$(aws batch describe-job-definitions \
    --job-definition-name $JOB_DEFINITION_NAME \
    --region $REGION \
    --query "jobDefinitions[-1].revision" \
    --output text)

echo -e "${GREEN}✓ Job definition registered: ${JOB_DEFINITION_NAME}:${JOB_DEF_REVISION}${NC}"
echo ""

# ============================================================================
# STEP 7: Summary
# ============================================================================
echo -e "${BLUE}[7/7] Setup Summary${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ AWS Batch setup complete!${NC}"
echo ""

# Save configuration to file
CONFIG_FILE=".glow_aws_config"
echo -e "${BLUE}Saving configuration to ${CONFIG_FILE}...${NC}"
cat > $CONFIG_FILE << CONFIGEOF
# GLOW AWS Configuration
# Auto-generated by setup_aws_batch.sh on $(date)

[aws]
s3_bucket = ${S3_BUCKET}
job_queue = ${JOB_QUEUE_NAME}
job_definition = ${JOB_DEFINITION_NAME}:${JOB_DEF_REVISION}
region = ${REGION}
account_id = ${ACCOUNT_ID}
CONFIGEOF
echo -e "${GREEN}✓ Configuration saved to ${CONFIG_FILE}${NC}"
echo ""

echo -e "${BLUE}Configuration for Python:${NC}"
echo ""
echo "from glow.cloud import CloudConfig"
echo ""
echo "cloud_config = CloudConfig("
echo "    s3_bucket='${S3_BUCKET}',"
echo "    s3_prefix='experiments/test',"
echo "    job_queue='${JOB_QUEUE_NAME}',"
echo "    job_definition='${JOB_DEFINITION_NAME}:${JOB_DEF_REVISION}',"
echo "    region='${REGION}'"
echo ")"
echo ""
echo -e "${BLUE}Optional: Set environment variables (not needed if using config file):${NC}"
echo ""
echo "export GLOW_S3_BUCKET='${S3_BUCKET}'"
echo "export GLOW_JOB_QUEUE='${JOB_QUEUE_NAME}'"
echo "export GLOW_JOB_DEFINITION='${JOB_DEFINITION_NAME}:${JOB_DEF_REVISION}'"
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"

# Cleanup temp files
rm -f /tmp/batch-trust-policy.json /tmp/ecs-trust-policy.json /tmp/s3-policy.json \
      /tmp/ec2-trust-policy.json /tmp/spot-trust-policy.json

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Test setup: cd test && python test_aws_cloud.py"
echo "  2. See AWS_CLOUD_GUIDE.md for usage examples"
echo ""
