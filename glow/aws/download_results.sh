#!/bin/bash
# download experiment results from S3
#
# Usage:
#   glow/aws/download_results.sh --prefix glow-runtime-benchmark --run-id runtime_hcp_6b72d962
#   glow/aws/download_results.sh --label runtime_hcp --run-id runtime_hcp_6b72d962
#   glow/aws/download_results.sh --list --prefix glow-runtime-benchmark
#   glow/aws/download_results.sh --list --prefix glow-paper-benchmarks

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
KEEP_S3=false
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
        --keep-s3)
            KEEP_S3=true
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
            echo "  --run-id RUN_ID   Run ID to download results for"
            echo "  --label LABEL      Config label (sets local output folder)"
            echo "  --list             List available run IDs under prefix"
            echo "  --keep-s3          Don't delete from S3 after download"
            echo "  --region REGION    AWS region (default: us-east-1)"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --list --prefix glow-runtime-benchmark"
            echo "  $0 --prefix glow-runtime-benchmark --run-id runtime_hcp_6b72d962"
            echo "  $0 --label runtime_hcp --run-id runtime_hcp_6b72d962"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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

# download mode: need --run-id
if [ -z "$RUN_ID" ]; then
    echo -e "${RED}✗ --run-id is required (use --list to find available runs)${NC}"
    exit 1
fi

# determine local output folder
if [ -n "$LABEL" ]; then
    OUTPUT_DIR=$(python3 -c "
from platformdirs import user_data_dir
from pathlib import Path
print(Path(user_data_dir('glow', 'glow_author')) / 'results' / '${LABEL}')
")
else
    OUTPUT_DIR=$(python3 -c "
from platformdirs import user_data_dir
from pathlib import Path
print(Path(user_data_dir('glow', 'glow_author')) / 'results' / '${RUN_ID}')
")
fi

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Download Results${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Bucket:  ${YELLOW}${S3_BUCKET}${NC}"
echo -e "  Prefix:  ${YELLOW}${PREFIX}${NC}"
echo -e "  Run ID:  ${YELLOW}${RUN_ID}${NC}"
echo -e "  Output:  ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "  Keep S3: ${YELLOW}${KEEP_S3}${NC}"
echo ""

python3 << EOF
import boto3
import os
from pathlib import Path

s3 = boto3.client('s3', region_name='${REGION}')
bucket = '${S3_BUCKET}'
result_prefix = '${PREFIX}/${RUN_ID}/results/'
output_dir = Path('${OUTPUT_DIR}')
keep_s3 = '${KEEP_S3}' == 'true'

output_dir.mkdir(parents=True, exist_ok=True)

paginator = s3.get_paginator('list_objects_v2')
downloaded = 0
keys_to_delete = []

for page in paginator.paginate(Bucket=bucket, Prefix=result_prefix):
    for obj in page.get('Contents', []):
        s3_key = obj['Key']
        rel_path = s3_key[len(result_prefix):]
        if not rel_path:
            continue
        local_path = output_dir / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, s3_key, str(local_path))
        downloaded += 1
        keys_to_delete.append(s3_key)
        if downloaded % 10 == 0:
            print(f'  downloaded {downloaded} files...', end='\r', flush=True)

if downloaded == 0:
    print('  (no result files found)')
else:
    print(f'  \033[0;32m✓\033[0m downloaded {downloaded} files to {output_dir}')

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
echo -e "${GREEN}✓ Done${NC}"
