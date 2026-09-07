#!/bin/bash
# Build, push, and deploy the glow viewer to Google Cloud Run.
#
# What this does:
#   1. Verifies the bundles are baked and the image would boot
#      (check_space)
#   2. Ensures an Artifact Registry repo exists in the target region
#   3. Builds the image locally from the repo root, pushes it
#   4. Deploys it to Cloud Run and prints the public URL
#
# Building locally rather than with `gcloud run deploy --source` is
# deliberate on two counts: that flag only finds a Dockerfile at the
# root of the source dir and ours lives here, and a local build is the
# same artifact you can `docker run` to check before it ships.
#
# Usage:
#   glow/_extra/viewer/web/deploy_cloud_run.sh --project my-gcp-project
#   glow/_extra/viewer/web/deploy_cloud_run.sh --project P --region us-east1
#   glow/_extra/viewer/web/deploy_cloud_run.sh --project P --yes
#   glow/_extra/viewer/web/deploy_cloud_run.sh --project P --build-only
#   glow/_extra/viewer/web/deploy_cloud_run.sh --project P --python PY
#
# The interpreter:
#   check_space runs in $PYTHON (default python3), which must be the one
#   glow is installed into -- a bare python3 is often not.
#     PYTHON=~/venv_glow/bin/python deploy_cloud_run.sh --project P
#
# Auth:
#   `gcloud auth login` once, and a project with billing enabled. The
#   free tier (2M requests, 180k vCPU-s, 360k GiB-s per month) covers a
#   demo comfortably; a billing account is still required to exist.
#
# Cost shape: the service scales to zero, so an idle month bills for the
# image sitting in Artifact Registry and nothing else.

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PYTHON="${PYTHON:-python3}"
PROJECT="${GCP_PROJECT:-}"
REGION="${GCP_REGION:-us-central1}"
SERVICE="${CLOUD_RUN_SERVICE:-glow-viewer}"
AR_REPO="${AR_REPO:-glow}"
ASSUME_YES=false
BUILD_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --project)    PROJECT="$2"; shift 2 ;;
        --python)     PYTHON="$2"; shift 2 ;;
        --region)     REGION="$2"; shift 2 ;;
        --service)    SERVICE="$2"; shift 2 ;;
        --repo)       AR_REPO="$2"; shift 2 ;;
        --yes|-y)     ASSUME_YES=true; shift ;;
        --build-only) BUILD_ONLY=true; shift ;;
        -h|--help)
            awk 'NR==1{next} /^[^#]/{exit} {sub(/^# ?/,""); print}' "$0"
            exit 0
            ;;
        *) echo -e "${RED}✗ Unknown option: $1${NC}"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PICKLE_DIR="$SCRIPT_DIR/pickles"

if [ -z "$PROJECT" ]; then
    echo -e "${RED}✗ GCP project not set${NC}"
    echo -e "${YELLOW}Pass --project <id> or set GCP_PROJECT${NC}"
    exit 1
fi

IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/${SERVICE}:latest"

echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Cloud Run deploy${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo -e "  Source:  ${YELLOW}${PROJECT_ROOT}${NC}"
echo -e "  Project: ${YELLOW}${PROJECT}${NC}   Region: ${YELLOW}${REGION}${NC}"
echo -e "  Service: ${YELLOW}${SERVICE}${NC}"
echo -e "  Image:   ${YELLOW}${IMAGE}${NC}"
echo ""

# ---------------------------------------------------------------------
# 1. Bundles + boot check
# ---------------------------------------------------------------------
if [ ! -d "$PICKLE_DIR" ] || [ -z "$(ls -A "$PICKLE_DIR"/*.p.gz 2>/dev/null)" ]
then
    echo -e "${RED}✗ No bundles in ${PICKLE_DIR}${NC}"
    echo -e "${YELLOW}  run: python -m glow._extra.viewer.web.bake_demos${NC}"
    exit 1
fi
BUNDLES=$(ls "$PICKLE_DIR"/*.p.gz | wc -l)
BUNDLE_SIZE=$(du -sh "$PICKLE_DIR" | cut -f1)
echo -e "  Bundles: ${GREEN}${BUNDLES} files, ${BUNDLE_SIZE}${NC}"

echo -e "${BLUE}Checking the image would boot ...${NC}"
if ! "$PYTHON" -c 'import glow' 2>/dev/null; then
    echo -e "${RED}✗ '${PYTHON}' cannot import glow${NC}"
    echo -e "${YELLOW}  set PYTHON=<interpreter glow is installed into>"
    echo -e "  or pass --python <path>${NC}"
    exit 1
fi
"$PYTHON" -m glow._extra.viewer.web.check_space

# ---------------------------------------------------------------------
# 2. Tooling
# ---------------------------------------------------------------------
for tool in gcloud docker; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo -e "${RED}✗ '${tool}' not found on PATH${NC}"; exit 1; }
done

if [ "$ASSUME_YES" = false ]; then
    echo ""
    read -r -p "  build and deploy? [y/N] " REPLY
    case "$REPLY" in y|Y|yes|YES) ;; *)
        echo -e "${RED}✗ Aborted${NC}"; exit 1 ;; esac
fi

# ---------------------------------------------------------------------
# 3. Build and push
# ---------------------------------------------------------------------
if [ "$BUILD_ONLY" = false ]; then
    echo ""
    echo -e "${BLUE}Ensuring Artifact Registry repo ...${NC}"
    gcloud artifacts repositories describe "$AR_REPO" \
        --project "$PROJECT" --location "$REGION" >/dev/null 2>&1 || \
    gcloud artifacts repositories create "$AR_REPO" \
        --project "$PROJECT" --location "$REGION" \
        --repository-format docker \
        --description "glow container images"

    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
fi

echo ""
echo -e "${BLUE}Building ${IMAGE} ...${NC}"
docker build -t "$IMAGE" \
    -f "$SCRIPT_DIR/Dockerfile" "$PROJECT_ROOT"

if [ "$BUILD_ONLY" = true ]; then
    echo ""
    echo -e "${GREEN}✓ Built (not pushed)${NC}"
    echo -e "  try it: ${YELLOW}docker run --rm -p 7860:7860 ${IMAGE}${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}Pushing ...${NC}"
docker push "$IMAGE"

# ---------------------------------------------------------------------
# 4. Deploy
# ---------------------------------------------------------------------
# --max-instances 1: the mount table lives in the worker process, so two
#   instances would each hold their own. A direct /view/<key>/ hit
#   self-mounts (see server.view), so a second instance is correct but
#   pays a redirect and a reload; one instance also bounds the bill.
# --min-instances 0: scale to zero, which is what keeps this free. The
#   cost is a cold start on the first hit after an idle period.
# --cpu-boost: cuts that cold start, and is free.
# --memory 2Gi: the server's own footprint plus the LRU's live viewers,
#   with room for the full-brain bundle, which is by far the largest.
echo ""
echo -e "${BLUE}Deploying to Cloud Run ...${NC}"
gcloud run deploy "$SERVICE" \
    --project "$PROJECT" \
    --region "$REGION" \
    --image "$IMAGE" \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --cpu-boost \
    --min-instances 0 \
    --max-instances 1 \
    --concurrency 16 \
    --timeout 300

URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" \
      --region "$REGION" --format 'value(status.url)')
echo ""
echo -e "${GREEN}✓ Deployed${NC}"
echo -e "  ${URL}"
