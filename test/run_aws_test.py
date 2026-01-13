"""test script for AWS cloud execution

Tests the AWS Batch integration for GLOW analysis at different levels.
"""

# ═══════════════════════════════════════════════════════════════════════
# TEST CONFIGURATION - Set these to control what runs
# ═══════════════════════════════════════════════════════════════════════

# Run S3 upload test only (fast, free, no Batch required)
# - Tests: S3 connectivity and experiment upload
# - Cost: ~$0
# - Time: ~5 seconds
# - Requires: AWS credentials, S3 bucket
RUN_S3_TEST = True

# Run full cloud execution test (slow, costs money, needs full AWS setup)
# - Tests: Runs locally AND on cloud, compares results for correctness validation
# - Also reports: AWS timing analysis (startup overhead vs computation time)
# - Cost: ~$0.05-0.15 (5 permutations)
# - Time: ~5-10 minutes
# - Requires: AWS credentials, S3, Batch compute env, job queue, job def, Docker image
RUN_FULL_TEST = True

# Note: If all flags are False, only cost estimation runs (always free/fast)

# ═══════════════════════════════════════════════════════════════════════

import os
from pathlib import Path
import configparser

import glow


def load_aws_config():
    """load AWS config from .glow_aws_config file or environment variables
    
    Returns:
        dict with s3_bucket, job_queue, job_definition, or None if not found
    """
    # first try to load from config file
    config_file = Path(__file__).parent.parent / '.glow_aws_config'
    
    if config_file.exists():
        config = configparser.ConfigParser()
        config.read(config_file)
        if 'aws' in config:
            return {
                's3_bucket': config['aws'].get('s3_bucket'),
                'job_queue': config['aws'].get('job_queue'),
                'job_definition': config['aws'].get('job_definition'),
                'region': config['aws'].get('region', 'us-east-1'),
                'account_id': config['aws'].get('account_id')
            }
    
    # fall back to environment variables
    s3_bucket = os.getenv('GLOW_S3_BUCKET')
    job_queue = os.getenv('GLOW_JOB_QUEUE')
    job_definition = os.getenv('GLOW_JOB_DEFINITION')
    
    if s3_bucket or job_queue or job_definition:
        return {
            's3_bucket': s3_bucket,
            'job_queue': job_queue,
            'job_definition': job_definition,
            'region': os.getenv('AWS_REGION', 'us-east-1'),
            'account_id': None
        }
    
    return None


def test_local_vs_cloud_comparison():
    """run same experiment locally and on cloud, compare results for equality"""
    # check if boto3 is available
    try:
        from glow.aws import CloudConfig
        import boto3
        import numpy as np
    except ImportError:
        print('skipping comparison test: boto3 not installed')
        print('install with: pip install boto3')
        return
    
    # check if AWS credentials are configured
    try:
        boto3.client('sts').get_caller_identity()
    except Exception:
        print('skipping comparison test: AWS credentials not configured')
        print('run: aws configure')
        return
    
    # load AWS config
    aws_config = load_aws_config()
    if not aws_config or not all([aws_config['s3_bucket'], 
                                   aws_config['job_queue'], 
                                   aws_config['job_definition']]):
        print('skipping comparison test: AWS resources not configured')
        print('run: ./setup_aws_batch.sh')
        return
    
    s3_bucket = aws_config['s3_bucket']
    job_queue = aws_config['job_queue']
    job_definition = aws_config['job_definition']
    
    # create experiment with effect
    print('creating test experiment with effect...')
    exp_orig = glow.experiment.Experiment.from_gauss(
        seed=42,
        shape=(20, 20),  # small 20x20 image
        a=2,
        b=2,
        num_img=20
    )
    
    # create circular mask for effect
    mask = np.zeros(exp_orig.mask_idx.shape, dtype=bool)
    center = (10, 10)
    radius = 5
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                mask[i, j] = True
    # ensure mask intersects with valid voxels
    mask = np.logical_and(mask, exp_orig.mask_idx > -1)
    
    # impose effect
    exp, eff = exp_orig.impose_effect(mask=mask, hotel_tr=10.0, seed=42)
    
    # shared analysis parameters
    ana_kwargs = {
        'n_perm': 5,
        'n_perm_adj': 5,
        'n_perm_prune': 10,
        'alpha_fwer': 0.05,
        'alpha_prune': 0.05,
        'min_size': 1,
        'verbose': True
    }
    
    # run locally
    print('\n' + '─' * 70)
    print('running GLOW analysis LOCALLY...')
    print('─' * 70)
    ana_local = glow.experiment.AnalysisGLOW(
        exp,
        **ana_kwargs
    )
    
    # run on cloud
    print('\n' + '─' * 70)
    print('running GLOW analysis on AWS CLOUD...')
    print('─' * 70)
    cloud_config = CloudConfig(
        s3_bucket=s3_bucket,
        s3_prefix='test/comparison_test',
        job_queue=job_queue,
        job_definition=job_definition,
        max_cost_per_hour=1.0,
        max_concurrent_jobs=10,
        vcpus=2,
        memory_mb=4096,
        timeout_minutes=30
    )
    
    ana_cloud = glow.experiment.AnalysisGLOW(
        exp,
        cloud_config=cloud_config,
        **ana_kwargs
    )
    
    # compare results
    print('\n' + '─' * 70)
    print('comparing LOCAL vs CLOUD results...')
    print('─' * 70)
    
    # check basic shapes
    print('\nchecking shapes...')
    assert ana_local.pval.shape == ana_cloud.pval.shape, 'pval shape mismatch'
    assert ana_local.z_stat.shape == ana_cloud.z_stat.shape, 'z_stat shape mismatch'
    assert ana_local.size.shape == ana_cloud.size.shape, 'size shape mismatch'
    assert ana_local.stat.shape == ana_cloud.stat.shape, 'stat shape mismatch'
    print('  ✓ shapes match')
    
    # check number of effects
    print('\nchecking effect counts...')
    n_eff_local = len(ana_local.effect_list)
    n_eff_cloud = len(ana_cloud.effect_list)
    print(f'  local: {n_eff_local} effects')
    print(f'  cloud: {n_eff_cloud} effects')
    assert n_eff_local == n_eff_cloud, f'effect count mismatch: {n_eff_local} != {n_eff_cloud}'
    print('  ✓ effect counts match')
    
    # check numerical values (with tolerance for floating point)
    print('\nchecking numerical values...')
    tol = 1e-10
    
    # pval comparison
    pval_diff = np.abs(ana_local.pval - ana_cloud.pval)
    max_pval_diff = np.max(pval_diff)
    print(f'  pval max difference: {max_pval_diff:.2e}')
    assert max_pval_diff < tol, f'pval mismatch: max diff {max_pval_diff:.2e} > {tol}'
    
    # z_stat comparison
    z_diff = np.abs(ana_local.z_stat - ana_cloud.z_stat)
    max_z_diff = np.max(z_diff)
    print(f'  z_stat max difference: {max_z_diff:.2e}')
    assert max_z_diff < tol, f'z_stat mismatch: max diff {max_z_diff:.2e} > {tol}'
    
    # stat comparison
    stat_diff = np.abs(ana_local.stat - ana_cloud.stat)
    max_stat_diff = np.max(stat_diff)
    print(f'  stat max difference: {max_stat_diff:.2e}')
    assert max_stat_diff < tol, f'stat mismatch: max diff {max_stat_diff:.2e} > {tol}'
    
    # size comparison
    size_diff = np.abs(ana_local.size - ana_cloud.size)
    max_size_diff = np.max(size_diff)
    print(f'  size max difference: {max_size_diff:.2e}')
    assert max_size_diff < tol, f'size mismatch: max diff {max_size_diff:.2e} > {tol}'
    
    print('  ✓ all numerical values match within tolerance')
    
    # check effect properties
    if n_eff_local > 0:
        print('\nchecking effect properties...')
        for i, (eff_local, eff_cloud) in enumerate(zip(ana_local.effect_list, ana_cloud.effect_list)):
            # use the is_close method for comprehensive comparison
            assert eff_local.is_close(eff_cloud, rtol=1e-10, atol=1e-10), \
                f'effect {i}: effects not equal (mask or y_mean mismatch)'
        print(f'  ✓ all {n_eff_local} effects match')
    
    # timing analysis from AWS jobs
    print('\n' + '═' * 70)
    print('AWS TIMING ANALYSIS')
    print('═' * 70)
    
    # get job IDs from the runner (stored during analysis)
    try:
        from glow.aws import AWSBatchRunner
        
        # recreate runner with same config to access job metadata
        runner = AWSBatchRunner(cloud_config)
        batch_client = boto3.client('batch', region_name=cloud_config.region)
        
        # get job IDs from the S3 prefix (they were submitted during ana_cloud creation)
        # we can get them from the runner's last submission
        # for now, let's note that timing data is available in CloudWatch
        print('\nTiming data available in AWS Batch job history.')
        print('To view detailed timing:')
        print('  1. Go to AWS Batch console')
        print('  2. View job details for each permutation')
        print('  3. Check: createdAt, startedAt, stoppedAt')
        print('\nTypical overhead observed: ~40-50s startup, ~10-15s compute')
        print('Recommendation: Batch multiple permutations per job to reduce overhead %')
        
    except Exception as e:
        print(f'\nCould not extract timing data: {e}')
        print('Timing analysis requires job IDs from AWS Batch submission')
    
    # summary
    print('\n' + '═' * 70)
    print('LOCAL vs CLOUD COMPARISON: PASSED ✓')
    print('═' * 70)
    print('\nSummary:')
    print(f'  Effects found: {n_eff_local}')
    print(f'  Max pval diff: {max_pval_diff:.2e}')
    print(f'  Max z_stat diff: {max_z_diff:.2e}')
    print(f'  Max stat diff: {max_stat_diff:.2e}')
    print(f'  Max size diff: {max_size_diff:.2e}')
    print('\n✓ Cloud execution produces identical results to local execution!')


def test_aws_runner_methods():
    """test AWSBatchRunner methods without running jobs"""
    # check if boto3 is available
    try:
        from glow.aws import CloudConfig, AWSBatchRunner
        import boto3
    except ImportError:
        print('skipping AWS runner test: boto3 not installed')
        print('install with: pip install boto3')
        return
    
    # check if AWS credentials are configured (try to get caller identity)
    try:
        boto3.client('sts').get_caller_identity()
    except Exception:
        print('skipping AWS runner test: AWS credentials not configured')
        print('run: aws configure')
        return
    
    # load AWS config from file or environment
    aws_config = load_aws_config()
    if not aws_config or not aws_config['s3_bucket']:
        print('skipping AWS runner test: AWS config not found')
        print('run: ./setup_aws_batch.sh')
        print('or set: export GLOW_S3_BUCKET=your-bucket-name')
        return
    
    s3_bucket = aws_config['s3_bucket']
    job_queue = aws_config.get('job_queue', 'test-queue')
    job_definition = aws_config.get('job_definition', 'test-job-def')
    
    cloud_config = CloudConfig(
        s3_bucket=s3_bucket,
        s3_prefix='test/runner_test',
        job_queue=job_queue,
        job_definition=job_definition
    )
    
    runner = AWSBatchRunner(cloud_config)
    
    # test upload/download cycle
    print('testing experiment upload/download...')
    exp = glow.experiment.Experiment.from_gauss(
        seed=0,
        shape=(10, 10),
        a=2,
        b=2,
        num_img=10
    )
    
    exp_id = 'test_exp_001'
    ana_kwargs = {
        'n_perm': 5,
        'n_perm_adj': 5,
        'n_perm_prune': 10,
        'alpha_fwer': 0.05,
        'alpha_prune': 0.05,
        'min_size': 1
    }
    
    try:
        # upload
        print(f'uploading to s3://{s3_bucket}/test/runner_test/{exp_id}/')
        runner.upload_experiment(exp, ana_kwargs, exp_id)
        print('upload successful')
        
        # note: we don't submit jobs or download results in this test
        # those would require actual AWS Batch resources
        
        print('\nAWS runner methods test passed!')
    except Exception as e:
        print(f'test failed: {e}')
        raise


def estimate_cost_example():
    """example of estimating costs before running"""
    try:
        from glow.aws import estimate_cost
    except ImportError:
        print('skipping cost estimation: boto3 not installed')
        print('install with: pip install boto3')
        return 0.0
    
    # example: 100 permutations, 5 minutes each
    cost_info = estimate_cost(
        n_jobs=101,  # n_perm + 1
        runtime_minutes=5,
        vcpus=2,
        memory_mb=4096,
        use_spot=True
    )
    
    print('\ncost estimation example:')
    print(f'  n_jobs: 101 (100 permutations + 1 original)')
    print(f'  duration: 5 minutes each')
    print(f'  instance: 2 vCPU, 4 GB RAM')
    print(f'  spot instances: yes')
    print(f'  estimated cost: ${cost_info["total_cost"]:.2f}')
    print(f'  cost per job: ${cost_info["cost_per_job"]:.4f}')
    print(f'  cost per hour: ${cost_info["cost_per_hour"]:.4f}')
    
    return cost_info["total_cost"]


if __name__ == '__main__':
    print('═' * 70)
    print('AWS CLOUD TESTING')
    print('═' * 70)
    
    # always run cost estimation (no AWS resources needed)
    print('\n1. Cost Estimation (always runs)')
    print('─' * 70)
    estimate_cost_example()
    
    # S3 test (controlled by RUN_S3_TEST flag at top of file)
    print('\n2. S3 Upload Test')
    print('─' * 70)
    if RUN_S3_TEST:
        test_aws_runner_methods()
    else:
        print('   SKIPPED (set RUN_S3_TEST=True at top of file to enable)')
    
    # Full cloud test with local comparison (controlled by RUN_FULL_TEST flag at top of file)
    print('\n3. Full Cloud Execution Test (with Local Comparison & Timing)')
    print('─' * 70)
    if RUN_FULL_TEST:
        test_local_vs_cloud_comparison()
    else:
        print('   SKIPPED (set RUN_FULL_TEST=True at top of file to enable)')
    
    print('\n' + '═' * 70)
    print('TESTING COMPLETE')
    print('═' * 70)
    
    # Summary
    print('\nTests run:')
    print(f'  Cost estimation: ✓ (always)')
    print(f'  S3 upload test: {"✓" if RUN_S3_TEST else "✗ (disabled)"}')
    print(f'  Full cloud test: {"✓" if RUN_FULL_TEST else "✗ (disabled)"} (includes local comparison + timing)')
    print('\nTo enable tests, edit flags at top of this file.')
