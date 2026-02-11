#!/bin/bash
# Quick helper to view logs from a failed AWS Batch job

if [ -z "$1" ]; then
    echo "Usage: $0 <log-stream-name>"
    echo ""
    echo "Example:"
    echo "  $0 glow-job-definition/default/ba4e44fd23764819813add4f445ae225"
    echo ""
    echo "Or get the most recent failed job automatically:"
    echo "  ./view_job_logs.sh auto"
    exit 1
fi

if [ "$1" == "auto" ]; then
    echo "Finding most recent failed job..."
    JOB_ID=$(aws batch list-jobs --job-queue glow-job-queue --job-status FAILED --max-results 1 --query 'jobSummaryList[0].jobId' --output text)
    
    if [ "$JOB_ID" == "None" ] || [ -z "$JOB_ID" ]; then
        echo "No failed jobs found in queue."
        exit 1
    fi
    
    echo "Job ID: $JOB_ID"
    LOG_STREAM=$(aws batch describe-jobs --jobs $JOB_ID --query 'jobs[0].container.logStreamName' --output text)
    echo "Log Stream: $LOG_STREAM"
    echo ""
else
    LOG_STREAM="$1"
fi

echo "=== Last 30 lines of logs ==="
echo ""
aws logs get-log-events \
  --log-group-name /aws/batch/job \
  --log-stream-name "$LOG_STREAM" \
  --limit 100 \
  --query 'events[*].message' \
  --output text | tail -30

echo ""
echo "=== To see more logs ==="
echo "aws logs get-log-events --log-group-name /aws/batch/job --log-stream-name $LOG_STREAM --limit 200 --query 'events[*].message' --output text"
