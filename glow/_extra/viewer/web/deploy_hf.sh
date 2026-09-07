#!/bin/bash
# Push a static HuggingFace Space that links to the Cloud Run demo.
#
# The viewer itself is a Dash app and needs a live Python process, which
# only a Docker Space provides -- and those now require a paid HF plan.
# Static Spaces are free, so the Space is reduced to a landing page whose
# job is to send a reader to the deployed demo (see deploy_cloud_run.sh).
#
# What this does:
#   1. Bootstraps a Space checkout at $SPACE_DIR (clones on first run)
#   2. Writes README.md (the Space card) and index.html, with the demo
#      URL substituted in
#   3. Commits and pushes
#
# Usage:
#   glow/_extra/viewer/web/deploy_hf.sh --user NAME --url https://...
#   glow/_extra/viewer/web/deploy_hf.sh --user NAME --url URL --yes
#
# The URL is whatever deploy_cloud_run.sh printed, or:
#   gcloud run services describe glow-viewer --region REGION \
#       --format 'value(status.url)'
#
# Auth:
#   Either run `hf auth login` once (from huggingface_hub), or have a git
#   credential helper that supplies your HF token at the password prompt
#   for huggingface.co.

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SPACE_NAME="${HF_SPACE:-glow-viewer}"
HF_USER="${HF_USERNAME:-}"
SPACE_DIR="${SPACE_DIR:-$HOME/hf-glow-viewer}"
DEMO_URL="${CLOUD_RUN_URL:-}"
ASSUME_YES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --user)      HF_USER="$2"; shift 2 ;;
        --space)     SPACE_NAME="$2"; shift 2 ;;
        --space-dir) SPACE_DIR="$2"; shift 2 ;;
        --url)       DEMO_URL="$2"; shift 2 ;;
        --yes|-y)    ASSUME_YES=true; shift ;;
        -h|--help)
            awk 'NR==1{next} /^[^#]/{exit} {sub(/^# ?/,""); print}' "$0"
            exit 0
            ;;
        *) echo -e "${RED}✗ Unknown option: $1${NC}"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$HF_USER" ]; then
    echo -e "${RED}✗ HF username not set${NC}"
    echo -e "${YELLOW}Pass --user <name> or set HF_USERNAME${NC}"
    exit 1
fi
if [ -z "$DEMO_URL" ]; then
    echo -e "${RED}✗ Demo URL not set${NC}"
    echo -e "${YELLOW}Pass --url <cloud-run-url> or set CLOUD_RUN_URL${NC}"
    echo -e "${YELLOW}  the Space is only a link -- there is nothing to"
    echo -e "  publish without it${NC}"
    exit 1
fi
case "$DEMO_URL" in
    https://*) ;;
    *) echo -e "${RED}✗ Demo URL must be https://${NC}"; exit 1 ;;
esac

SPACE_URL="https://huggingface.co/spaces/${HF_USER}/${SPACE_NAME}"

echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  HF static Space deploy${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo -e "  Space URL: ${YELLOW}${SPACE_URL}${NC}"
echo -e "  Links to:  ${YELLOW}${DEMO_URL}${NC}"
echo -e "  Checkout:  ${YELLOW}${SPACE_DIR}${NC}"
echo ""

# ---------------------------------------------------------------------
# 1. Space checkout
# ---------------------------------------------------------------------
if [ ! -d "$SPACE_DIR/.git" ]; then
    echo -e "${YELLOW}Space checkout not found at ${SPACE_DIR}${NC}"
    echo -e "  Will clone: ${SPACE_URL}"
    if [ "$ASSUME_YES" = false ]; then
        read -r -p "  proceed? [y/N] " REPLY
        case "$REPLY" in y|Y|yes|YES) ;; *)
            echo -e "${RED}✗ Aborted${NC}"; exit 1 ;; esac
    fi
    git clone "$SPACE_URL" "$SPACE_DIR"
else
    echo -e "  Pulling latest from Space ..."
    git -C "$SPACE_DIR" pull --ff-only
fi

# ---------------------------------------------------------------------
# 2. Write the two files a static Space needs
# ---------------------------------------------------------------------
# Anything left from a previous Docker-Space deploy would still be served
# (and would still be a ~1 GB repo), so clear the tree first.
echo ""
echo -e "${BLUE}Writing Space files ...${NC}"
find "$SPACE_DIR" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +

# a Space is configured by YAML front matter on its README
cat "$SCRIPT_DIR/hf_space_card.md" > "$SPACE_DIR/README.md"
printf '\n# GLOW Viewer\n\nA landing page for the interactive viewer, which runs at\n%s\nand is deployed by glow/_extra/viewer/web/deploy_cloud_run.sh.\n' \
    "$DEMO_URL" >> "$SPACE_DIR/README.md"

# substitute the URL rather than committing it to the repo copy, so the
# template stays host-agnostic and a redeploy can retarget it
sed "s|__DEMO_URL__|${DEMO_URL}|g" "$SCRIPT_DIR/hf_index.html" \
    > "$SPACE_DIR/index.html"

if ! grep -q "$DEMO_URL" "$SPACE_DIR/index.html"; then
    echo -e "${RED}✗ URL substitution failed${NC}"
    exit 1
fi
echo -e "  ${GREEN}README.md + index.html${NC}"

# ---------------------------------------------------------------------
# 3. Commit + push
# ---------------------------------------------------------------------
cd "$SPACE_DIR"
if [ -z "$(git status --porcelain)" ]; then
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
    case "$REPLY" in y|Y|yes|YES) ;; *)
        echo -e "${RED}✗ Aborted (changes left in $SPACE_DIR)${NC}"
        exit 1 ;; esac
fi

git commit -m "link to the Cloud Run demo"
echo ""
echo -e "${BLUE}Pushing to ${SPACE_URL} ...${NC}"
git push

echo ""
echo -e "${GREEN}✓ Done${NC}"
echo -e "  ${SPACE_URL}"
