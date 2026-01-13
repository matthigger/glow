# AWS Test Scripts

These scripts are **not** run by pytest automatically (they require AWS credentials and cost money).

## Scripts

### `run_aws_test.py`
Full AWS Batch integration test. Run directly:
```bash
python run_aws_test.py
```

Configure what runs by editing the flags at the top:
- `RUN_S3_TEST = True` - Test S3 upload only (free, fast)
- `RUN_FULL_TEST = True` - Full cloud execution with local comparison for validation (costs ~$0.05-0.15)

### `run_aws_diagnose.py`
Diagnose recent AWS Batch job failures:
```bash
python run_aws_diagnose.py
```

### `view_job_logs.sh`
View CloudWatch logs from a failed job:
```bash
./view_job_logs.sh auto  # Most recent failed job
./view_job_logs.sh <log-stream-name>  # Specific job
```

## Why not pytest?

These files are named `run_*.py` instead of `test_*.py` to prevent pytest from discovering and running them automatically. AWS tests:
- Require manual setup (credentials, infrastructure)
- Cost money to run
- Take several minutes
- Should be opt-in, not automatic

To include them in pytest, you would need to add proper `@pytest.mark.skipif` decorators or pytest markers.
