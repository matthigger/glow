#!/bin/bash
set -e  # Exit on error

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION - Edit these to customize your setup
# ═══════════════════════════════════════════════════════════════
REGION=${AWS_REGION:-us-east-1}
S3_BUCKET=${GLOW_S3_BUCKET:-glow-experiments-$(date +%s)}
COMPUTE_ENV_NAME="glow-compute-env-spot"
JOB_QUEUE_NAME="glow-job-queue"
JOB_DEFINITION_NAME="glow-job-definition"

# concurrency configuration
MAX_VCPUS=2048              # max vCPUs for compute environment
VCPUS_PER_JOB=1             # vCPUs per job (1 = max concurrency)
MEMORY_PER_JOB=4096         # memory (MB) per job

# ═══════════════════════════════════════════════════════════════

# colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  GLOW AWS Batch Setup (Idempotent)${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Region: $REGION"
echo "  S3 Bucket: $S3_BUCKET"
echo "  Max vCPUs: $MAX_VCPUS"
echo "  vCPUs per job: $VCPUS_PER_JOB"
echo "  Memory per job: ${MEMORY_PER_JOB} MB"
echo "  Max concurrent jobs: $((MAX_VCPUS / VCPUS_PER_JOB))"
echo ""

# get AWS account ID
echo -e "${YELLOW}Getting AWS account information...${NC}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}✓ Account ID: ${ACCOUNT_ID}${NC}"
echo ""

# check if Docker image exists in ECR
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/glow-worker"
echo -e "${YELLOW}Checking for Docker image in ECR...${NC}"
if aws ecr describe-images --repository-name glow-worker --region $REGION --image-ids imageTag=latest &>/dev/null; then
    echo -e "${GREEN}✓ Docker image found in ECR${NC}"
    DOCKER_IMAGE="${ECR_REPO}:latest"
else
    echo -e "${RED}✗ Docker image not found in ECR${NC}"
    echo -e "${YELLOW}Run: ./glow/aws/deploy_docker.sh${NC}"
    exit 1
fi
echo ""

# ============================================================================
# STEP 1: Create S3 Bucket (idempotent)
# ============================================================================
echo -e "${BLUE}[1/7] Creating S3 bucket...${NC}"
if aws s3 ls "s3://${S3_BUCKET}" 2>/dev/null; then
    echo -e "${GREEN}✓ Bucket ${S3_BUCKET} already exists${NC}"
else
    aws s3 mb "s3://${S3_BUCKET}" --region $REGION
    echo -e "${GREEN}✓ Created S3 bucket: ${S3_BUCKET}${NC}"
fi
echo ""

# ============================================================================
# STEP 2: Create IAM Roles (idempotent)
# ============================================================================
echo -e "${BLUE}[2/7] Creating IAM roles...${NC}"

create_or_update_role() {
    local role_name=$1
    local trust_policy=$2
    local policy_arn=$3
    local description=$4
    
    if aws iam get-role --role-name $role_name &>/dev/null; then
        echo -e "${GREEN}  ✓ $role_name already exists${NC}"
        aws iam update-assume-role-policy \
            --role-name $role_name \
            --policy-document "$trust_policy" 2>/dev/null || true
    else
        aws iam create-role \
            --role-name $role_name \
            --assume-role-policy-document "$trust_policy" \
            --description "$description"
        if [ -n "$policy_arn" ]; then
            aws iam attach-role-policy \
                --role-name $role_name \
                --policy-arn $policy_arn
        fi
        echo -e "${GREEN}  ✓ Created $role_name${NC}"
    fi
}

# batch service role
BATCH_TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"batch.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
create_or_update_role "GlowBatchServiceRole" "$BATCH_TRUST_POLICY" \
    "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole" \
    "AWS Batch service role for GLOW"

# ecs task execution role
ECS_TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
create_or_update_role "GlowEcsTaskExecutionRole" "$ECS_TRUST_POLICY" \
    "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy" \
    "ECS task execution role for GLOW"

# ecs task role (for S3 access)
create_or_update_role "GlowEcsTaskRole" "$ECS_TRUST_POLICY" "" \
    "ECS task role for GLOW S3 access"

# update S3 policy
S3_POLICY="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::${S3_BUCKET}\",\"arn:aws:s3:::${S3_BUCKET}/*\"]}]}"
aws iam put-role-policy \
    --role-name GlowEcsTaskRole \
    --policy-name GlowS3Access \
    --policy-document "$S3_POLICY" 2>/dev/null || true

# ec2 instance role
EC2_TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
create_or_update_role "ecsInstanceRole" "$EC2_TRUST_POLICY" \
    "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role" \
    "EC2 instance role for ECS/Batch"

# create instance profile (idempotent)
if ! aws iam get-instance-profile --instance-profile-name ecsInstanceRole &>/dev/null; then
    aws iam create-instance-profile --instance-profile-name ecsInstanceRole
    aws iam add-role-to-instance-profile \
        --instance-profile-name ecsInstanceRole \
        --role-name ecsInstanceRole 2>/dev/null || true
    echo -e "${GREEN}  ✓ Created instance profile${NC}"
fi

# spot fleet role
SPOT_TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"spotfleet.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
create_or_update_role "aws-ec2-spot-fleet-tagging-role" "$SPOT_TRUST_POLICY" \
    "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole" \
    "Spot fleet role for EC2 spot instances"

echo -e "${YELLOW}  Waiting 10 seconds for IAM propagation...${NC}"
sleep 10
echo -e "${GREEN}✓ All IAM roles ready${NC}"
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
# STEP 4: Create or Update Compute Environment (idempotent)
# ============================================================================
echo -e "${BLUE}[4/7] Setting up compute environment...${NC}"

if aws batch describe-compute-environments --compute-environments $COMPUTE_ENV_NAME --region $REGION --query "computeEnvironments[0].computeEnvironmentName" --output text 2>/dev/null | grep -q "$COMPUTE_ENV_NAME"; then
    STATUS=$(aws batch describe-compute-environments --compute-environments $COMPUTE_ENV_NAME --region $REGION --query "computeEnvironments[0].status" --output text)
    CURRENT_MAX_VCPUS=$(aws batch describe-compute-environments --compute-environments $COMPUTE_ENV_NAME --region $REGION --query "computeEnvironments[0].computeResources.maxvCpus" --output text)
    
    echo -e "${YELLOW}  Compute environment exists (status: $STATUS, maxvCpus: $CURRENT_MAX_VCPUS)${NC}"
    
    if [ "$CURRENT_MAX_VCPUS" != "$MAX_VCPUS" ]; then
        echo -e "${YELLOW}  Updating maxvCpus from $CURRENT_MAX_VCPUS to $MAX_VCPUS...${NC}"
        aws batch update-compute-environment \
            --compute-environment $COMPUTE_ENV_NAME \
            --compute-resources "{\"maxvCpus\": $MAX_VCPUS}" \
            --region $REGION
        echo -e "${GREEN}  ✓ Updated compute environment${NC}"
        sleep 5
    else
        echo -e "${GREEN}  ✓ Compute environment already configured correctly${NC}"
    fi
else
    echo -e "${YELLOW}  Creating new compute environment...${NC}"
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
            \"maxvCpus\": $MAX_VCPUS,
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
        echo -e "    Status: $STATUS (waiting... $i/60)"
        sleep 5
    done
    echo -e "${GREEN}✓ Compute environment created${NC}"
fi
echo ""

# ============================================================================
# STEP 5: Create Job Queue (idempotent)
# ============================================================================
echo -e "${BLUE}[5/7] Setting up job queue...${NC}"
if aws batch describe-job-queues --job-queues $JOB_QUEUE_NAME --region $REGION --query "jobQueues[0].jobQueueName" --output text 2>/dev/null | grep -q "$JOB_QUEUE_NAME"; then
    echo -e "${GREEN}✓ Job queue already exists${NC}"
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
        \"resourceRequirements\": [
            {\"type\": \"VCPU\", \"value\": \"${VCPUS_PER_JOB}\"},
            {\"type\": \"MEMORY\", \"value\": \"${MEMORY_PER_JOB}\"}
        ]
    }" > /dev/null

# get the latest active revision (sorted by revision number)
JOB_DEF_REVISION=$(aws batch describe-job-definitions \
    --job-definition-name $JOB_DEFINITION_NAME \
    --status ACTIVE \
    --region $REGION \
    --query "jobDefinitions | sort_by(@, &revision) | [-1].revision" \
    --output text)

echo -e "${GREEN}✓ Job definition registered: ${JOB_DEFINITION_NAME}:${JOB_DEF_REVISION}${NC}"
echo ""

# ============================================================================
# STEP 7: Save Configuration
# ============================================================================
echo -e "${BLUE}[7/7] Saving configuration...${NC}"
CONFIG_FILE=".glow_aws_config"
cat > $CONFIG_FILE << CONFIGEOF
# GLOW AWS Configuration
# Generated by setup_aws_batch.sh on $(date)

[aws]
s3_bucket = ${S3_BUCKET}
job_queue = ${JOB_QUEUE_NAME}
job_definition = ${JOB_DEFINITION_NAME}:${JOB_DEF_REVISION}
region = ${REGION}
account_id = ${ACCOUNT_ID}
max_vcpus = ${MAX_VCPUS}
vcpus_per_job = ${VCPUS_PER_JOB}
max_concurrent_jobs = $((MAX_VCPUS / VCPUS_PER_JOB))
CONFIGEOF
echo -e "${GREEN}✓ Configuration saved to ${CONFIG_FILE}${NC}"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ AWS Batch setup complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Configuration Summary:${NC}"
echo "  S3 Bucket: ${S3_BUCKET}"
echo "  Job Queue: ${JOB_QUEUE_NAME}"
echo "  Job Definition: ${JOB_DEFINITION_NAME}:${JOB_DEF_REVISION}"
echo "  Max vCPUs: ${MAX_VCPUS}"
echo "  vCPUs per job: ${VCPUS_PER_JOB}"
echo "  Max concurrent jobs: $((MAX_VCPUS / VCPUS_PER_JOB))"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Test: python test/run_aws_test.py"
echo "  2. Deploy code changes: ./glow/aws/deploy_docker.sh"
echo "  3. Run experiments: python glow/benchmark/paper.py"
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
