#!/bin/bash
set -e # exit immediately if a command exits with a non-zero status

# --- Colors for output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  GLOW Docker Image Deployment and Validation${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# --- Configuration ---
REGION="us-east-1"
IMAGE_NAME="glow-worker"

# --- Get AWS account ID ---
echo -e "${YELLOW}Getting AWS account ID...${NC}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}"
echo -e "${GREEN}✓ Account ID: ${ACCOUNT_ID}${NC}"
echo -e "${GREEN}✓ ECR Repository: ${ECR_URI}${NC}"
echo ""

# --- Step 1: Build Docker image locally (with cache) ---
echo -e "${BLUE}[1/6] Building Docker image locally...${NC}"
# build from project root, but use Dockerfile in glow/aws/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
docker build -f glow/aws/Dockerfile -t ${IMAGE_NAME}:latest .
echo -e "${GREEN}✓ Docker image built: ${IMAGE_NAME}:latest${NC}"
echo ""

# --- Step 2: Get local image digest ---
echo -e "${BLUE}[2/6] Getting local image digest...${NC}"
LOCAL_DIGEST=$(docker inspect ${IMAGE_NAME}:latest --format='{{.Id}}' | cut -d: -f2 | cut -c1-12)
LOCAL_DIGEST_FULL=$(docker inspect ${IMAGE_NAME}:latest --format='{{.Id}}')
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo -e "${GREEN}✓ Local digest (short): ${LOCAL_DIGEST}${NC}"
echo -e "${GREEN}✓ Build time: ${BUILD_TIME}${NC}"
echo ""

# --- Step 3: Tag image for ECR (both :latest and :digest) ---
echo -e "${BLUE}[3/6] Tagging image for ECR...${NC}"
docker tag ${IMAGE_NAME}:latest ${ECR_URI}:latest
docker tag ${IMAGE_NAME}:latest ${ECR_URI}:${LOCAL_DIGEST}
echo -e "${GREEN}✓ Image tagged: ${ECR_URI}:latest${NC}"
echo -e "${GREEN}✓ Image tagged: ${ECR_URI}:${LOCAL_DIGEST} (digest-based)${NC}"
echo ""

# --- Step 4: Log into ECR ---
echo -e "${BLUE}[4/6] Logging into ECR...${NC}"
aws ecr get-login-password --region ${REGION} | \
    docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
echo -e "${GREEN}✓ Logged into ECR${NC}"
echo ""

# --- Step 5: Push both tags to ECR ---
echo -e "${BLUE}[5/6] Pushing images to ECR...${NC}"
echo -e "${YELLOW}  Pushing :latest tag...${NC}"
docker push ${ECR_URI}:latest
echo -e "${YELLOW}  Pushing :${LOCAL_DIGEST} tag...${NC}"
docker push ${ECR_URI}:${LOCAL_DIGEST}
echo -e "${GREEN}✓ Both images pushed to ECR!${NC}"
echo ""

# wait for ECR to process
echo -e "${YELLOW}  Waiting for ECR to update metadata...${NC}"
sleep 5
echo ""

# --- Step 6: Validate ECR image and create manifest ---
echo -e "${BLUE}[6/6] Validating deployment...${NC}"

ECR_IMAGE_DIGEST=$(aws ecr describe-images \
    --repository-name ${IMAGE_NAME} \
    --image-ids imageTag=latest \
    --query 'imageDetails[0].imageDigest' \
    --output text \
    --region ${REGION})

ECR_PUSHED_AT=$(aws ecr describe-images \
    --repository-name ${IMAGE_NAME} \
    --image-ids imageTag=latest \
    --query 'imageDetails[0].imagePushedAt' \
    --output text \
    --region ${REGION})

ECR_SIZE=$(aws ecr describe-images \
    --repository-name ${IMAGE_NAME} \
    --image-ids imageTag=latest \
    --query 'imageDetails[0].imageSizeInBytes' \
    --output text \
    --region ${REGION})

ECR_SIZE_MB=$(echo "scale=1; $ECR_SIZE / 1024 / 1024" | bc)

echo -e "${GREEN}  ✓ ECR Image Digest: ${ECR_IMAGE_DIGEST}${NC}"
echo -e "${GREEN}  ✓ ECR Pushed At: ${ECR_PUSHED_AT}${NC}"
echo -e "${GREEN}  ✓ ECR Image Size: ${ECR_SIZE_MB} MB${NC}"
echo -e "${GREEN}  ✓ Local Build Digest: ${LOCAL_DIGEST}${NC}"
echo ""

# verify job definition will use this image
JOB_DEFINITION_NAME="glow-job-definition"
echo -e "${YELLOW}  Checking AWS Batch Job Definition...${NC}"
JOB_DEF_IMAGE=$(aws batch describe-job-definitions \
    --job-definition-name ${JOB_DEFINITION_NAME} \
    --status ACTIVE \
    --query "jobDefinitions | sort_by(@, &revision)[-1].containerProperties.image" \
    --output text \
    --region ${REGION})

echo -e "${YELLOW}    Job Definition Image: ${JOB_DEF_IMAGE}${NC}"

if [ "${JOB_DEF_IMAGE}" == "${ECR_URI}:latest" ]; then
    echo -e "${GREEN}    ✓ Job Definition will pull the image we just pushed${NC}"
    echo -e "${GREEN}    ✓ ECR Digest: ${ECR_IMAGE_DIGEST}${NC}"
elif [[ "${JOB_DEF_IMAGE}" == *"${IMAGE_NAME}"* ]]; then
    echo -e "${YELLOW}    ⚠ Job Definition uses: ${JOB_DEF_IMAGE}${NC}"
    echo -e "${YELLOW}    ⚠ Expected: ${ECR_URI}:latest${NC}"
    echo -e "${YELLOW}    ⚠ Run: ./glow/aws/setup_aws_batch.sh to update${NC}"
else
    echo -e "${RED}    ✗ WARNING: Job Definition points to different image!${NC}"
    echo -e "${RED}    ✗ Current: ${JOB_DEF_IMAGE}${NC}"
    echo -e "${RED}    ✗ Expected: ${ECR_URI}:latest${NC}"
    exit 1
fi

echo -e "${YELLOW}  Creating version manifest...${NC}"
cat > .docker_version << EOF
# GLOW Docker Image Version Manifest
# Generated: ${BUILD_TIME}

BUILD_TIME=${BUILD_TIME}
LOCAL_DIGEST_SHORT=${LOCAL_DIGEST}
LOCAL_DIGEST_FULL=${LOCAL_DIGEST_FULL}
ECR_DIGEST=${ECR_IMAGE_DIGEST}
ECR_URI=${ECR_URI}
ECR_TAG_LATEST=${ECR_URI}:latest
ECR_TAG_DIGEST=${ECR_URI}:${LOCAL_DIGEST}
ECR_PUSHED_AT=${ECR_PUSHED_AT}
ECR_SIZE_MB=${ECR_SIZE_MB}
REGION=${REGION}
ACCOUNT_ID=${ACCOUNT_ID}
EOF
echo -e "${GREEN}✓ Version manifest saved to .docker_version${NC}"
echo ""

# --- Summary ---
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Deployment and Validation Complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Deployment Summary:${NC}"
echo "  Build Time: ${BUILD_TIME}"
echo "  Local Digest: ${LOCAL_DIGEST}"
echo "  ECR Digest: ${ECR_IMAGE_DIGEST}"
echo "  ECR URI: ${ECR_URI}:latest"
echo "  Size: ${ECR_SIZE_MB} MB"
echo ""
echo -e "${BLUE}Tagged Versions:${NC}"
echo "  ${ECR_URI}:latest         (always pulls newest)"
echo "  ${ECR_URI}:${LOCAL_DIGEST}  (pinned version)"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Run tests: python test/run_aws_test.py"
echo "  2. Check version: cat .docker_version"
echo "  3. Rollback if needed: use :${LOCAL_DIGEST} tag"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
