#!/bin/bash
# cleanup AWS Batch queue and S3 storage for glow experiments

set -e

# colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # no color

# default values
JOB_QUEUE="glow-job-queue"
REGION="us-east-1"
FORCE=false
CLEAR_QUEUE=""
CLEAR_S3=""
S3_PREFIX=""
PROMPT_USER=true

# parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --queue)
            JOB_QUEUE="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --s3-prefix)
            S3_PREFIX="$2"
            shift 2
            ;;
        --queue-only)
            CLEAR_QUEUE=true
            CLEAR_S3=false
            PROMPT_USER=false
            shift
            ;;
        --s3-only)
            CLEAR_QUEUE=false
            CLEAR_S3=true
            PROMPT_USER=false
            shift
            ;;
        --both)
            CLEAR_QUEUE=true
            CLEAR_S3=true
            PROMPT_USER=false
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Cleanup AWS Batch queue and/or S3 storage for glow experiments"
            echo ""
            echo "Options:"
            echo "  --queue QUEUE      Job queue name (default: glow-job-queue)"
            echo "  --region REGION    AWS region (default: us-east-1)"
            echo "  --s3-prefix PREFIX S3 prefix to clean (default: from .glow_aws_config)"
            echo "  --queue-only       Only clear queue jobs"
            echo "  --s3-only          Only clear S3 storage"
            echo "  --both             Clear both queue and S3"
            echo "                     (if none specified, will prompt interactively)"
            echo "  --force            Skip confirmation prompt"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# prompt user if no action specified via flags
if [ "$PROMPT_USER" = true ]; then
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  AWS Glow Cleanup${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "What would you like to clean?"
    echo ""
    read -p "Clear all active jobs from queue? (yes/no): " -r
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        CLEAR_QUEUE=true
    else
        CLEAR_QUEUE=false
    fi
    echo ""
    read -p "Clear S3 storage for experiment results? (yes/no): " -r
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        CLEAR_S3=true
    else
        CLEAR_S3=false
    fi
    echo ""
    
    if [ "$CLEAR_QUEUE" = false ] && [ "$CLEAR_S3" = false ]; then
        echo -e "${YELLOW}Nothing selected. Exiting.${NC}"
        exit 0
    fi
fi

# determine header based on what we're cleaning
if [ "$CLEAR_QUEUE" = true ] && [ "$CLEAR_S3" = true ]; then
    HEADER="AWS Glow Cleanup (Queue + S3)"
elif [ "$CLEAR_S3" = true ]; then
    HEADER="AWS S3 Storage Cleanup"
else
    HEADER="AWS Batch Queue Cleanup"
fi

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  ${HEADER}${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Queue: ${YELLOW}${JOB_QUEUE}${NC}"
echo -e "Region: ${YELLOW}${REGION}${NC}"
if [ "$CLEAR_S3" = true ] || [ "$CLEAR_QUEUE" = true ]; then
    echo -e "Clear Queue: ${YELLOW}${CLEAR_QUEUE}${NC}"
    echo -e "Clear S3: ${YELLOW}${CLEAR_S3}${NC}"
fi
echo ""

# load AWS config for S3 if needed
S3_BUCKET=""
if [ "$CLEAR_S3" = true ]; then
    echo -e "${YELLOW}Loading AWS config...${NC}"
    
    # config file is stored in project root
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    CONFIG_FILE="$PROJECT_ROOT/.glow_aws_config"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}✗ AWS config file not found${NC}"
        echo -e "${RED}  Checked: $CONFIG_FILE${NC}"
        echo -e "${RED}  Cannot determine S3 bucket/prefix${NC}"
        echo ""
        echo -e "${YELLOW}Tip: Run glow/aws/setup_aws_batch.sh to create the config file${NC}"
        exit 1
    fi
    
    echo -e "  Using config: ${YELLOW}$CONFIG_FILE${NC}"
    
    # convert to absolute path for Python
    CONFIG_FILE_ABS="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
    
    S3_BUCKET=$(python3 << EOF
import configparser
import sys
import os
try:
    parser = configparser.ConfigParser()
    config_path = '${CONFIG_FILE_ABS}'
    if not os.path.exists(config_path):
        print(f'Config file not found: {config_path}', file=sys.stderr)
        sys.exit(1)
    parser.read(config_path)
    print(parser['aws']['s3_bucket'])
except Exception as e:
    print(f'Error reading config: {e}', file=sys.stderr)
    sys.exit(1)
EOF
)
    
    if [ -z "$S3_PREFIX" ]; then
        S3_PREFIX=$(python3 << EOF
import configparser
import sys
import os
try:
    parser = configparser.ConfigParser()
    config_path = '${CONFIG_FILE_ABS}'
    parser.read(config_path)
    # default to glow-paper-benchmarks if not in config
    print(parser['aws'].get('s3_prefix', 'glow-paper-benchmarks'))
except Exception as e:
    print(f'Error reading config: {e}', file=sys.stderr)
    sys.exit(1)
EOF
)
    fi
    
    echo -e "S3 Bucket: ${YELLOW}${S3_BUCKET}${NC}"
    echo -e "S3 Prefix: ${YELLOW}${S3_PREFIX}${NC}"
    echo ""
fi

# initialize flags
QUEUE_HAS_JOBS=false
S3_HAS_DATA=false

# clear queue jobs
if [ "$CLEAR_QUEUE" = true ]; then
    echo -e "${YELLOW}Checking job statuses in queue...${NC}"
    
    EXIT_CODE=0
    python3 << EOF || EXIT_CODE=$?
import boto3
import sys
from collections import Counter
    
try:
    batch = boto3.client('batch', region_name='${REGION}')
    
    # get all job statuses to show counts
    all_statuses = ['SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING']
    all_jobs = []
    status_counts = Counter()
    
    for status in all_statuses:
        try:
            next_token = None
            while True:
                kwargs = dict(
                    jobQueue='${JOB_QUEUE}',
                    jobStatus=status,
                    maxResults=100,
                )
                if next_token:
                    kwargs['nextToken'] = next_token
                response = batch.list_jobs(**kwargs)
                jobs = response['jobSummaryList']
                status_counts[status] += len(jobs)
                all_jobs.extend([(j['jobId'], j['jobName'], status) for j in jobs])
                next_token = response.get('nextToken')
                if not next_token:
                    break
        except Exception as e:
            print(f'  Warning: Error listing {status} jobs: {e}', file=sys.stderr)
    
    # show counts for all statuses
    print('  Job status counts:')
    total = 0
    for status in all_statuses:
        count = status_counts[status]
        total += count
        print(f'    {status}: {count}')
    print(f'  Total active jobs: {total}')
    
    jobs_to_cancel = all_jobs
    
    if not jobs_to_cancel:
        print('\n  ✓ No active jobs to cancel')
        sys.exit(2)
    
    with open('/tmp/glow_jobs_to_cancel.txt', 'w') as f:
        for job_id, _, status in jobs_to_cancel:
            f.write(f'{job_id}\t{status}\n')
    
    print(f'\n  Jobs to cancel: {len(jobs_to_cancel)}')
    cancel_counts = Counter(status for _, _, status in jobs_to_cancel)
    for status, count in sorted(cancel_counts.items()):
        print(f'    {status}: {count}')
    
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
EOF
    if [ $EXIT_CODE -eq 2 ]; then
        echo ""
        echo -e "${GREEN}✓ No active jobs to cancel${NC}"
        QUEUE_HAS_JOBS=false
    elif [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo -e "${RED}✗ Error checking queue status${NC}"
        exit 1
    else
        QUEUE_HAS_JOBS=true
    fi
fi

# check S3 storage
if [ "$CLEAR_S3" = true ]; then
    echo -e "${YELLOW}Checking S3 storage...${NC}"
    
    EXIT_CODE=0
    python3 << EOF || EXIT_CODE=$?
import boto3
import sys

try:
    s3 = boto3.client('s3', region_name='${REGION}')
    
    prefix = '${S3_PREFIX}/'
    bucket = '${S3_BUCKET}'
    
    # count objects
    paginator = s3.get_paginator('list_objects_v2')
    total_size = 0
    total_count = 0
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                total_size += obj['Size']
                total_count += 1
    
    if total_count == 0:
        print(f'  ✓ No objects found under s3://{bucket}/{prefix}')
        sys.exit(2)
    
    size_gb = total_size / (1024**3)
    size_mb = total_size / (1024**2)
    
    if size_gb >= 1:
        print(f'  Found {total_count} objects, {size_gb:.2f} GB')
    else:
        print(f'  Found {total_count} objects, {size_mb:.2f} MB')
    
    # save prefix for deletion
    with open('/tmp/glow_s3_prefix.txt', 'w') as f:
        f.write(f'{bucket}\n')
        f.write(f'{prefix}\n')
    
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
EOF
    if [ $EXIT_CODE -eq 2 ]; then
        echo ""
        echo -e "${GREEN}✓ S3 storage is already empty${NC}"
        S3_HAS_DATA=false
    elif [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo -e "${RED}✗ Error checking S3 storage${NC}"
        exit 1
    else
        S3_HAS_DATA=true
    fi
fi

echo ""

# confirmation prompt (unless --force)
if [ "$FORCE" = false ]; then
    if [ "$CLEAR_QUEUE" = true ] && [ "$QUEUE_HAS_JOBS" = true ]; then
        echo -e "${YELLOW}⚠  This will cancel all active jobs in ${JOB_QUEUE}${NC}"
    fi
    if [ "$CLEAR_S3" = true ] && [ "$S3_HAS_DATA" = true ]; then
        echo -e "${YELLOW}⚠  This will delete all data under s3://${S3_BUCKET}/${S3_PREFIX}/${NC}"
        echo -e "${YELLOW}⚠  This includes experiment results and cannot be undone!${NC}"
    fi
    if ([ "$CLEAR_QUEUE" = true ] && [ "$QUEUE_HAS_JOBS" = true ]) || ([ "$CLEAR_S3" = true ] && [ "$S3_HAS_DATA" = true ]); then
        echo ""
        read -p "Continue? (yes/no): " -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            echo -e "${RED}✗ Canceled by user${NC}"
            rm -f /tmp/glow_jobs_to_cancel.txt /tmp/glow_s3_prefix.txt
            exit 1
        fi
    fi
fi

# cancel jobs
if [ "$CLEAR_QUEUE" = true ] && [ "$QUEUE_HAS_JOBS" = true ]; then
    echo -e "${YELLOW}Canceling jobs...${NC}"
    
    python3 << EOF
import boto3
import sys
    
try:
    batch = boto3.client('batch', region_name='${REGION}')
    
    # read job IDs and statuses
    with open('/tmp/glow_jobs_to_cancel.txt', 'r') as f:
        jobs = []
        for line in f:
            line = line.strip()
            if line:
                job_id, status = line.split('\t')
                jobs.append((job_id, status))
    
    if not jobs:
        print('  No jobs to cancel')
        sys.exit(0)
    
    canceled = 0
    failed = 0
    reason = 'Queue cleanup via cleanup_aws.sh'
    
    for i, (job_id, status) in enumerate(jobs, 1):
        try:
            if status in ('SUBMITTED', 'PENDING'):
                batch.cancel_job(jobId=job_id, reason=reason)
            else:
                batch.terminate_job(jobId=job_id, reason=reason)
            canceled += 1
            
            if i % 10 == 0 or i == len(jobs):
                progress = (i / len(jobs)) * 100
                print(f'  Progress: {i}/{len(jobs)} ({progress:.0f}%)', end='\r')
                sys.stdout.flush()
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f'\n  ✗ Failed to cancel {job_id[:12]}...: {e}')
    
    print()  # newline after progress
    print(f'  ✓ Canceled: {canceled}')
    if failed > 0:
        print(f'  ✗ Failed: {failed}')
    
    # cleanup temp file
    import os
    os.remove('/tmp/glow_jobs_to_cancel.txt')
    
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        echo ""
        echo -e "${RED}✗ Queue cleanup failed${NC}"
        rm -f /tmp/glow_jobs_to_cancel.txt /tmp/glow_s3_prefix.txt
        exit 1
    fi
fi

# clear S3 storage
if [ "$CLEAR_S3" = true ] && [ "$S3_HAS_DATA" = true ]; then
    echo -e "${YELLOW}Deleting S3 objects...${NC}"
    
    python3 << EOF
import boto3
import sys

try:
    s3 = boto3.client('s3', region_name='${REGION}')
    
    # read bucket and prefix
    with open('/tmp/glow_s3_prefix.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        bucket = lines[0]
        prefix = lines[1]
    
    # delete all objects with this prefix
    paginator = s3.get_paginator('list_objects_v2')
    deleted = 0
    failed = 0
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            objects = [{'Key': obj['Key']} for obj in page['Contents']]
            
            if objects:
                try:
                    s3.delete_objects(
                        Bucket=bucket,
                        Delete={'Objects': objects}
                    )
                    deleted += len(objects)
                    
                    # show progress
                    print(f'  Progress: {deleted} objects deleted...', end='\r')
                    sys.stdout.flush()
                except Exception as e:
                    failed += len(objects)
                    print(f'\n  ✗ Failed to delete batch: {e}')
    
    print()  # newline after progress
    print(f'  ✓ Deleted: {deleted} objects')
    if failed > 0:
        print(f'  ✗ Failed: {failed}')
    
    # cleanup temp file
    import os
    os.remove('/tmp/glow_s3_prefix.txt')
    
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        echo ""
        echo -e "${RED}✗ S3 cleanup failed${NC}"
        rm -f /tmp/glow_s3_prefix.txt
        exit 1
    fi
fi

# final summary
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Cleanup complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""

if [ "$CLEAR_QUEUE" = true ]; then
    echo -e "${YELLOW}Note: Jobs may take a few seconds to fully terminate${NC}"
    echo -e "${YELLOW}Old FAILED jobs will remain in history (auto-purge after 7 days)${NC}"
fi

if [ "$CLEAR_S3" = true ]; then
    echo -e "${YELLOW}Note: S3 experiment results have been permanently deleted${NC}"
fi

echo ""
