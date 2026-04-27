#!/bin/bash
# download experiment results from S3
#
# Usage:
#   glow/aws/download_results.sh                                 # download all runs (default)
#   glow/aws/download_results.sh --all --delete-s3
#   glow/aws/download_results.sh --run-id runtime_hcp_6b72d962
#   glow/aws/download_results.sh --label runtime_hcp --run-id runtime_hcp_6b72d962
#   glow/aws/download_results.sh --list --prefix glow-runtime-benchmark

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PREFIX="glow-paper-benchmarks"
RUN_ID=""
LABEL=""
LIST_MODE=false
ALL_MODE=false
KEEP_S3=""  # empty = unspecified, will prompt
FORCE=false
REGION="us-east-1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --label)
            LABEL="$2"
            shift 2
            ;;
        --list)
            LIST_MODE=true
            shift
            ;;
        --all)
            ALL_MODE=true
            shift
            ;;
        --keep-s3)
            KEEP_S3="true"
            shift
            ;;
        --delete-s3)
            KEEP_S3="false"
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Download experiment results from S3"
            echo ""
            echo "Options:"
            echo "  --prefix PREFIX    S3 prefix (default: glow-paper-benchmarks)"
            echo "  --run-id RUN_ID    Run ID to download results for"
            echo "  --all              Download every run under prefix (default if no --run-id)"
            echo "  --label LABEL      Config label (sets local output folder; single-run only)"
            echo "  --list             List available run IDs under prefix"
            echo "  --keep-s3          Don't delete from S3 after download"
            echo "  --delete-s3        Delete from S3 after download (skips prompt)"
            echo "  --force            Re-download runs that already have a .download_complete marker"
            echo "  --region REGION    AWS region (default: us-east-1)"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "If neither --keep-s3 nor --delete-s3 is given, the script prompts."
            echo ""
            echo "Examples:"
            echo "  $0                                          # interactive: download everything"
            echo "  $0 --list --prefix glow-runtime-benchmark"
            echo "  $0 --prefix glow-runtime-benchmark --run-id runtime_hcp_6b72d962"
            echo "  $0 --all --delete-s3"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# default to --all when neither --list nor --run-id is given
if [ "$LIST_MODE" = false ] && [ -z "$RUN_ID" ] && [ "$ALL_MODE" = false ]; then
    ALL_MODE=true
fi

if [ "$ALL_MODE" = true ] && [ -n "$RUN_ID" ]; then
    echo -e "${RED}✗ --all and --run-id are mutually exclusive${NC}"
    exit 1
fi

if [ "$ALL_MODE" = true ] && [ -n "$LABEL" ]; then
    echo -e "${RED}✗ --label cannot be combined with --all (output paths use run-id)${NC}"
    exit 1
fi

# load config for S3 bucket
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/.glow_aws_config"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Config not found: $CONFIG_FILE${NC}"
    echo -e "${YELLOW}Tip: Run glow/aws/setup_aws_batch.sh first${NC}"
    exit 1
fi

S3_BUCKET=$(python3 -c "
import configparser, sys
p = configparser.ConfigParser()
p.read('$CONFIG_FILE')
print(p['aws']['s3_bucket'])
")

if [ -z "$S3_BUCKET" ]; then
    echo -e "${RED}✗ Could not read s3_bucket from config${NC}"
    exit 1
fi

# --list mode: show available run IDs
if [ "$LIST_MODE" = true ]; then
    echo -e "${BLUE}Listing run IDs under s3://${S3_BUCKET}/${PREFIX}/${NC}"
    echo ""

    python3 << EOF
import boto3

s3 = boto3.client('s3', region_name='${REGION}')
bucket = '${S3_BUCKET}'
prefix = '${PREFIX}/'

paginator = s3.get_paginator('list_objects_v2')
run_ids = {}

for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
    for cp in page.get('CommonPrefixes', []):
        sub = cp['Prefix'][len(prefix):]
        if sub.endswith('/'):
            sub = sub[:-1]
        if not sub or sub == 'shared_exp_data':
            continue
        # count result files under this run
        result_prefix = f'{prefix}{sub}/results/'
        count = 0
        size = 0
        for rp in paginator.paginate(Bucket=bucket, Prefix=result_prefix):
            for obj in rp.get('Contents', []):
                count += 1
                size += obj['Size']
        if count > 0:
            run_ids[sub] = (count, size)

if not run_ids:
    print('  (no runs with results found)')
else:
    for rid, (count, size) in sorted(run_ids.items()):
        mb = size / (1024**2)
        print(f'  {rid:40s}  {count:4d} files  ({mb:.1f} MB)')
EOF
    exit 0
fi

# determine which run IDs to download
declare -a RUN_IDS_TO_DOWNLOAD
if [ "$ALL_MODE" = true ]; then
    echo -e "${BLUE}Discovering runs under s3://${S3_BUCKET}/${PREFIX}/${NC}"
    RUN_IDS_LIST=$(python3 << EOF
import boto3
s3 = boto3.client('s3', region_name='${REGION}')
bucket = '${S3_BUCKET}'
prefix = '${PREFIX}/'
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
    for cp in page.get('CommonPrefixes', []):
        sub = cp['Prefix'][len(prefix):].rstrip('/')
        if not sub or sub == 'shared_exp_data':
            continue
        result_prefix = f'{prefix}{sub}/results/'
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=result_prefix, MaxKeys=1)
        if resp.get('KeyCount', 0) > 0:
            print(sub)
EOF
)
    if [ -z "$RUN_IDS_LIST" ]; then
        echo -e "${YELLOW}(no runs with results found)${NC}"
        exit 0
    fi
    while IFS= read -r line; do
        [ -n "$line" ] && RUN_IDS_TO_DOWNLOAD+=("$line")
    done <<< "$RUN_IDS_LIST"
    echo -e "${GREEN}Found ${#RUN_IDS_TO_DOWNLOAD[@]} runs:${NC}"
    for rid in "${RUN_IDS_TO_DOWNLOAD[@]}"; do
        echo "  - $rid"
    done
    echo ""
else
    RUN_IDS_TO_DOWNLOAD=("$RUN_ID")
fi

# prompt for keep-s3 if not specified
if [ -z "$KEEP_S3" ]; then
    if [ "$ALL_MODE" = true ]; then
        echo -e "${YELLOW}Delete these ${#RUN_IDS_TO_DOWNLOAD[@]} runs from S3 after download?${NC}"
    else
        echo -e "${YELLOW}Delete this run from S3 after download?${NC}"
    fi
    echo -n "  [y]es delete / [n]o keep on S3: "
    read -r RESP
    case "$RESP" in
        y|Y|yes|YES)
            KEEP_S3="false"
            echo -e "  → ${YELLOW}will delete from S3 after download${NC}"
            ;;
        n|N|no|NO)
            KEEP_S3="true"
            echo -e "  → ${GREEN}keeping S3 copy${NC}"
            ;;
        *)
            echo -e "${RED}✗ Please answer y or n${NC}"
            exit 1
            ;;
    esac
    echo ""
fi

# download each run
for RID in "${RUN_IDS_TO_DOWNLOAD[@]}"; do
    # derive config label by stripping trailing _<8hex> from run-id
    if [ -n "$LABEL" ]; then
        DERIVED_LABEL="$LABEL"
    elif [[ "$RID" =~ ^(.+)_[a-f0-9]{8}$ ]]; then
        DERIVED_LABEL="${BASH_REMATCH[1]}"
    else
        DERIVED_LABEL="$RID"
    fi

    OUTPUT_DIR=$(python3 -c "
from platformdirs import user_data_dir
from pathlib import Path
print(Path(user_data_dir('glow', 'glow_author')) / 'results' / '${DERIVED_LABEL}')
")
    MARKER="$OUTPUT_DIR/.download_complete_${RID}"

    if [ "$FORCE" != "true" ] && [ -f "$MARKER" ]; then
        echo -e "${YELLOW}  skipping ${RID} (already complete at ${OUTPUT_DIR}; use --force to re-download)${NC}"
        echo ""
        continue
    fi

    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Download Results${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  Bucket:  ${YELLOW}${S3_BUCKET}${NC}"
    echo -e "  Prefix:  ${YELLOW}${PREFIX}${NC}"
    echo -e "  Run ID:  ${YELLOW}${RID}${NC}"
    echo -e "  Label:   ${YELLOW}${DERIVED_LABEL}${NC}"
    echo -e "  Output:  ${YELLOW}${OUTPUT_DIR}${NC}"
    echo -e "  Keep S3: ${YELLOW}${KEEP_S3}${NC}"
    echo ""

    python3 << EOF
import boto3
import io
import tarfile
from pathlib import Path

s3 = boto3.client('s3', region_name='${REGION}')
bucket = '${S3_BUCKET}'
result_prefix = '${PREFIX}/${RID}/results/'
output_dir = Path('${OUTPUT_DIR}')
marker = Path('${MARKER}')
keep_s3 = '${KEEP_S3}' == 'true'

output_dir.mkdir(parents=True, exist_ok=True)

paginator = s3.get_paginator('list_objects_v2')
n_archives = 0
n_files = 0
keys_to_delete = []

for page in paginator.paginate(Bucket=bucket, Prefix=result_prefix):
    for obj in page.get('Contents', []):
        s3_key = obj['Key']
        rel_path = s3_key[len(result_prefix):]
        if not rel_path:
            continue
        keys_to_delete.append(s3_key)

        body = s3.get_object(Bucket=bucket, Key=s3_key)['Body'].read()

        if rel_path.endswith('.tar.gz'):
            with tarfile.open(fileobj=io.BytesIO(body), mode='r:gz') as tar:
                tar.extractall(path=str(output_dir))
                n_files += len(tar.getmembers())
            n_archives += 1
        else:
            # legacy: loose result file, write directly under output_dir
            local_path = output_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(body)
            n_files += 1

        if (n_archives + n_files) % 10 == 0:
            print(f'  processed {n_archives} archives, {n_files} files...',
                  end='\r', flush=True)

if n_files == 0:
    print('  (no result files found)')
else:
    marker.touch()
    print(f'  \033[0;32m✓\033[0m extracted {n_files} files'
          f' from {n_archives} archives into {output_dir}')

if keys_to_delete and not keep_s3:
    for i in range(0, len(keys_to_delete), 1000):
        batch = keys_to_delete[i:i+1000]
        s3.delete_objects(
            Bucket=bucket,
            Delete={'Objects': [{'Key': k} for k in batch]}
        )
    print(f'  \033[0;32m✓\033[0m deleted {len(keys_to_delete)} S3 objects')
EOF
    echo ""
done

echo -e "${GREEN}✓ Done${NC}"
