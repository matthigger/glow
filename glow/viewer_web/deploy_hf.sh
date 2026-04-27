#!/bin/bash
# Push the glow viewer web demo to a HuggingFace Space.
#
# What this does:
#   1. Verifies pickles are baked (offers to bake if not)
#   2. Bootstraps a Space checkout at $SPACE_DIR (clones on first run)
#   3. Rsyncs the deployable subset into the Space checkout:
#        glow/                       <-  src/glow/
#        Dockerfile                  <-  src/glow/viewer_web/Dockerfile
#        README.md                   <-  src/glow/viewer_web/README.md
#        .dockerignore               <-  src/glow/viewer_web/.dockerignore
#   4. Writes a Space-specific .gitignore (does NOT exclude pickles)
#   5. Commits and pushes
#
# Usage:
#   glow/viewer_web/deploy_hf.sh                    # interactive
#   glow/viewer_web/deploy_hf.sh --user myname      # set HF username
#   glow/viewer_web/deploy_hf.sh --yes              # skip confirmation
#   glow/viewer_web/deploy_hf.sh --space my-space   # custom Space name
#
# Auth:
#   Either run `hf auth login` once (from huggingface_hub; older docs say
#   `huggingface-cli login`, which is now deprecated), or have a git
#   credential helper that supplies your HF token at the password prompt
#   for huggingface.co.

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# defaults / env
SPACE_NAME="${HF_SPACE:-glow-viewer}"
HF_USER="${HF_USERNAME:-}"
SPACE_DIR="${SPACE_DIR:-$HOME/hf-glow-viewer}"
ASSUME_YES=false

# parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --user)        HF_USER="$2"; shift 2 ;;
        --space)       SPACE_NAME="$2"; shift 2 ;;
        --space-dir)   SPACE_DIR="$2"; shift 2 ;;
        --yes|-y)      ASSUME_YES=true; shift ;;
        -h|--help)
            awk 'NR==1{next} /^[^#]/{exit} {sub(/^# ?/,""); print}' "$0"
            exit 0
            ;;
        *)
            echo -e "${RED}✗ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# resolve project root from this script's location
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PICKLE_DIR="$SCRIPT_DIR/pickles"

if [ -z "$HF_USER" ]; then
    echo -e "${RED}✗ HF username not set${NC}"
    echo -e "${YELLOW}Pass --user <name> or set HF_USERNAME in your shell${NC}"
    exit 1
fi

SPACE_URL="https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  HF Space deploy${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "  Source:    ${YELLOW}${PROJECT_ROOT}${NC}"
echo -e "  Space URL: ${YELLOW}${SPACE_URL}${NC}"
echo -e "  Checkout:  ${YELLOW}${SPACE_DIR}${NC}"
echo ""

# ---------------------------------------------------------------------------
# 1. Pickles must exist
# ---------------------------------------------------------------------------
if [ ! -d "$PICKLE_DIR" ] || [ -z "$(ls -A "$PICKLE_DIR" 2>/dev/null)" ]; then
    echo -e "${YELLOW}No pickles found in ${PICKLE_DIR}${NC}"
    if [ "$ASSUME_YES" = true ]; then
        REPLY=y
    else
        read -r -p "  bake them now? [y/N] " REPLY
    fi
    case "$REPLY" in
        y|Y|yes|YES)
            python3 -m glow.viewer_web.bake_demos
            ;;
        *)
            echo -e "${RED}✗ Aborting -- run python -m glow.viewer_web.bake_demos first${NC}"
            exit 1
            ;;
    esac
fi

PICKLE_COUNT=$(ls "$PICKLE_DIR"/*.p.gz 2>/dev/null | wc -l)
PICKLE_SIZE=$(du -sh "$PICKLE_DIR" | cut -f1)
echo -e "  Pickles:   ${GREEN}${PICKLE_COUNT} files, ${PICKLE_SIZE}${NC}"

# ---------------------------------------------------------------------------
# 2. Space checkout
# ---------------------------------------------------------------------------
if [ ! -d "$SPACE_DIR/.git" ]; then
    echo ""
    echo -e "${YELLOW}Space checkout not found at ${SPACE_DIR}${NC}"
    echo -e "  Will clone: ${SPACE_URL}"
    if [ "$ASSUME_YES" = false ]; then
        read -r -p "  proceed? [y/N] " REPLY
        case "$REPLY" in
            y|Y|yes|YES) ;;
            *) echo -e "${RED}✗ Aborted${NC}"; exit 1 ;;
        esac
    fi
    git clone "$SPACE_URL" "$SPACE_DIR"
else
    echo -e "  Pulling latest from Space ..."
    git -C "$SPACE_DIR" pull --ff-only
fi

# ---------------------------------------------------------------------------
# 3. Sync deployable subset
# ---------------------------------------------------------------------------
echo ""
echo -e "${BLUE}Syncing files ...${NC}"

# clear stale package code (preserves git metadata)
rm -rf "$SPACE_DIR/glow"

# rsync the glow package, excluding subpackages and dev cruft the
# runtime viewer never imports (top-level glow/__init__.py only pulls
# analysis, effect, experiment, graph, mask).  Pickles are excluded
# because HF rejects binary files in git -- they are uploaded via
# `hf upload` (Xet) after the git push.
rsync -a \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache/' \
    --exclude '.ipynb_checkpoints/' \
    --exclude '*tmp*' \
    --exclude 'aws/' \
    --exclude 'benchmark/' \
    --exclude 'viewer_web/pickles/' \
    "$PROJECT_ROOT/glow/" "$SPACE_DIR/glow/"

# the source-tree gitignore would exclude pickles; remove it
rm -f "$SPACE_DIR/glow/viewer_web/.gitignore"

# top-level deploy files
cp "$SCRIPT_DIR/Dockerfile"     "$SPACE_DIR/Dockerfile"
cp "$SCRIPT_DIR/README.md"      "$SPACE_DIR/README.md"
cp "$SCRIPT_DIR/.dockerignore"  "$SPACE_DIR/.dockerignore"

# Space-specific .gitignore: pickles live outside git (uploaded via Xet)
cat > "$SPACE_DIR/.gitignore" <<'EOF'
__pycache__/
*.pyc
.pytest_cache/
glow/viewer_web/pickles/
EOF

# ---------------------------------------------------------------------------
# 4. Diff + commit + push
# ---------------------------------------------------------------------------
cd "$SPACE_DIR"

if git diff --quiet && git diff --staged --quiet && [ -z "$(git status --porcelain)" ]; then
    echo -e "${GREEN}✓ Space already up to date, nothing to push${NC}"
    exit 0
fi

git add -A
echo ""
echo -e "${BLUE}Pending changes:${NC}"
git status --short

if [ "$ASSUME_YES" = false ]; then
    echo ""
    read -r -p "  commit and push to HF? [y/N] " REPLY
    case "$REPLY" in
        y|Y|yes|YES) ;;
        *) echo -e "${RED}✗ Aborted (changes left in $SPACE_DIR)${NC}"; exit 1 ;;
    esac
fi

# include source SHA in commit message so deploys are traceable
SRC_SHA=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
SRC_DIRTY=""
if ! git -C "$PROJECT_ROOT" diff --quiet 2>/dev/null \
   || ! git -C "$PROJECT_ROOT" diff --staged --quiet 2>/dev/null; then
    SRC_DIRTY="-dirty"
fi
git commit -m "deploy ${SRC_SHA}${SRC_DIRTY}"

echo ""
echo -e "${BLUE}Pushing to ${SPACE_URL} ...${NC}"
git push

# ---------------------------------------------------------------------------
# 5. Upload pickles via hf upload (Xet)
# ---------------------------------------------------------------------------
# Pickles can't go through plain git -- HF rejects binary files in git and
# requires Xet for them.  `hf upload` handles Xet transparently.
if ! command -v hf >/dev/null 2>&1; then
    echo -e "${RED}✗ 'hf' CLI not found on PATH${NC}"
    echo -e "${YELLOW}  install with: pip install -U huggingface_hub${NC}"
    echo -e "${YELLOW}  then re-run this script (the git push already landed)${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}Uploading pickles via Xet ...${NC}"
hf upload "${HF_USER}/${SPACE_NAME}" \
    "${PICKLE_DIR}" "glow/viewer_web/pickles" \
    --repo-type space \
    --commit-message "upload pickles ${SRC_SHA}${SRC_DIRTY}"

echo ""
echo -e "${GREEN}✓ Done${NC}"
echo -e "  HF will rebuild automatically.  Watch progress:"
echo -e "  ${SPACE_URL}"
