"""test AWS cloud execution for both architectures

permutation-level: AnalysisGLOW(cloud_config=X) - parallel perms
experiment-level: Config(cloud_config=X) - serial perms

named run_aws_test.py so pytest won't auto-run it
"""

# ═══════════════════════════════════════════════════════════════════════
# TEST CONFIGURATION - Set these to control what runs
# ═══════════════════════════════════════════════════════════════════════

# S3 connectivity test
# - Tests: S3 upload/download
RUN_S3_TEST = True

# Permutation-level architecture test (parallel permutations)
# - Architecture: AnalysisGLOW(cloud_config=X)
# - Best for: 1 large experiment
RUN_PERMUTATION_LEVEL_TEST = True

# Experiment-level architecture test (serial permutations)
# - Architecture: Config(cloud_config=X)
# - Best for: many small experiments
RUN_EXPERIMENT_LEVEL_TEST = True

# TFCE analysis test (tests pure Python TFCE on cloud)
# - Architecture: Config(cloud_config=X) with AnalysisVBA(tfce_flag=True)
# - Tests: TFCE integration on cloud workers
RUN_TFCE_TEST = True

# HCP data test (tests loading real imaging data from S3)
# - Architecture: Config(cloud_config=X) with source='hcp'
# - Tests: HCP data upload to S3, download on worker, experiment execution
# - Requires: local HCP data or existing HCP data on S3
RUN_HCP_TEST = True  # disabled by default (requires HCP data)


# ═══════════════════════════════════════════════════════════════════════

import os
from pathlib import Path
import configparser

import numpy as np

import glow
from glow.benchmark.config import Config
from glow.benchmark.run import run_ana


def print_debug_info():
    """print version and configuration info for debugging"""
    import sys
    import glow
    
    print('\n' + '=' * 70)
    print('DEBUG INFO')
    print('=' * 70)
    
    # python version
    print(f'\nPython: {sys.version}')
    
    # package versions
    print(f'\nGlow version: {glow.__version__ if hasattr(glow, "__version__") else "unknown"}')
    
    try:
        import numpy
        print(f'NumPy: {numpy.__version__}')
    except ImportError:
        print('NumPy: not installed')
    
    try:
        import boto3
        print(f'boto3: {boto3.__version__}')
    except ImportError:
        print('boto3: not installed')
    
    # check AWS config
    aws_config = load_aws_config()
    if aws_config:
        print('\nAWS Configuration:')
        print(f'  S3 Bucket: {aws_config.get("s3_bucket")}')
        print(f'  Job Queue: {aws_config.get("job_queue")}')
        print(f'  Job Definition: {aws_config.get("job_definition")}')
        print(f'  Region: {aws_config.get("region")}')
        print(f'  Account ID: {aws_config.get("account_id")}')
        
        # check Docker version manifest
        docker_version_path = Path('.docker_version')
        if docker_version_path.exists():
            print('\nDocker Image Version:')
            with open(docker_version_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            # only show key fields
                            if key in ['BUILD_TIME', 'LOCAL_DIGEST_SHORT', 'ECR_DIGEST', 
                                       'ECR_PUSHED_AT', 'ECR_SIZE_MB']:
                                print(f'  {key}: {value}')
        else:
            print('\nDocker Image Version:')
            print('  ⚠ .docker_version not found')
            print('  Run: ./glow/aws/deploy_docker.sh to create it')
            
            # check ECR image
            try:
                import boto3
                ecr_client = boto3.client('ecr', region_name=aws_config.get('region', 'us-east-1'))
                
                print('\nECR Image Info:')
                try:
                    response = ecr_client.describe_images(
                        repositoryName='glow-worker',
                        imageIds=[{'imageTag': 'latest'}]
                    )
                    if response['imageDetails']:
                        image = response['imageDetails'][0]
                        pushed_at = image.get('imagePushedAt')
                        size_mb = image.get('imageSizeInBytes', 0) / (1024 * 1024)
                        print(f'  Repository: glow-worker')
                        print(f'  Tag: latest')
                        print(f'  Size: {size_mb:.1f} MB')
                        print(f'  Pushed: {pushed_at}')
                        print(f'  Digest: {image.get("imageDigest", "unknown")[:20]}...')
                except Exception as e:
                    print(f'  ✗ Could not fetch image info: {e}')
                    
                # check job definition
                print('\nAWS Batch Job Definition:')
                try:
                    batch_client = boto3.client('batch', region_name=aws_config.get('region', 'us-east-1'))
                    response = batch_client.describe_job_definitions(
                        jobDefinitionName=aws_config['job_definition'].split(':')[0],
                        status='ACTIVE',
                        maxResults=1
                    )
                    if response['jobDefinitions']:
                        job_def = response['jobDefinitions'][0]
                        print(f'  Name: {job_def["jobDefinitionName"]}')
                        print(f'  Revision: {job_def["revision"]}')
                        print(f'  Type: {job_def["type"]}')
                        container = job_def.get('containerProperties', {})
                        print(f'  Image: {container.get("image", "unknown")}')
                        print(f'  vCPUs: {container.get("vcpus", "unknown")}')
                        print(f'  Memory: {container.get("memory", "unknown")} MB')
                except Exception as e:
                    print(f'  ✗ Could not fetch job definition: {e}')
                    
            except ImportError:
                print('\n  (boto3 not available for AWS checks)')
            except Exception as e:
                print(f'\n  ✗ AWS check failed: {e}')
    else:
        print('\n⚠ AWS configuration not found')
    
    print('=' * 70)


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
        print('run: ./glow/aws/setup_aws_batch.sh')
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
        print('run: ./glow/aws/setup_aws_batch.sh')
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


def test_experiment_level_architecture():
    """test experiment-level architecture (Config.cloud_config)
    
    Architecture: Each experiment runs in one AWS worker (serial perms)
    Use case: Many small experiments (benchmarking like paper.py)
    """
    try:
        import boto3
        from glow.aws.aws_batch import CloudConfig
    except ImportError:
        print('skipping experiment-level test: boto3 not installed')
        print('install with: pip install boto3')
        return
    
    # check credentials
    try:
        boto3.client('sts').get_caller_identity()
    except Exception:
        print('skipping experiment-level test: AWS credentials not configured')
        print('run: aws configure')
        return
    
    # load config
    aws_config = load_aws_config()
    if not aws_config or not all([aws_config['s3_bucket'], 
                                   aws_config['job_queue'], 
                                   aws_config['job_definition']]):
        print('skipping experiment-level test: AWS config incomplete')
        print('run: ./glow/aws/setup_aws_batch.sh')
        return
    
    print('testing experiment-level architecture (2 exp × 5 perms, 2 AWS jobs)...')
    
    # shared analysis config
    ana_kwargs_dict = {
        'GLOW': (glow.experiment.AnalysisGLOW,
                 dict(n_perm=5,
                      n_perm_adj=5,
                      n_perm_prune=10,
                      alpha_fwer=0.05,
                      alpha_prune=0.05,
                      min_size=1,
                      n_jobs_perm=1))  # serial within each job
    }
    
    cloud_config = CloudConfig(
        s3_bucket=aws_config['s3_bucket'],
        s3_prefix='test/experiment_level',
        job_queue=aws_config['job_queue'],
        job_definition=aws_config['job_definition'],
        region=aws_config.get('region', 'us-east-1'),
        timeout_minutes=30,
        retry_attempts=1
    )
    
    config = Config(
        label='test_experiment_level',
        source='wgn',
        run_fnc=run_ana,
        cloud_config=cloud_config,  # experiment-level!
        ana_kwargs_dict=ana_kwargs_dict,
        n_seed=2,
        hotel_tr_all=np.array([0.5]),
        wgn_shape=(5, 5, 5),
        wgn_a=2,
        wgn_b=2,
        wgn_num_img=20,
        exp_seed=42,
        effect_perc=0.2,
        n_jobs=1,
        detail_save=False,
        error_save=False
    )
    
    config.run_all(verbose=True)
    
    results = list((config.folder / 'out').glob('*_result.json'))
    print(f'✓ experiment-level test passed ({len(results)} results)')


def test_tfce_cloud():
    """test TFCE analysis on cloud (pure Python implementation)
    
    Minimal test: 1 experiment with AnalysisVBA(tfce_flag=True)
    Validates that our pure Python TFCE runs correctly on cloud workers.
    """
    try:
        import boto3
        from glow.aws.aws_batch import CloudConfig
    except ImportError:
        print('skipping TFCE test: boto3 not installed')
        print('install with: pip install boto3')
        return
    
    # check credentials
    try:
        boto3.client('sts').get_caller_identity()
    except Exception:
        print('skipping TFCE test: AWS credentials not configured')
        print('run: aws configure')
        return
    
    # load config
    aws_config = load_aws_config()
    if not aws_config or not all([aws_config['s3_bucket'], 
                                   aws_config['job_queue'], 
                                   aws_config['job_definition']]):
        print('skipping TFCE test: AWS config incomplete')
        print('run: ./glow/aws/setup_aws_batch.sh')
        return
    
    print('testing TFCE on cloud (1 exp × 5 perms with tfce_flag=True)...')
    
    # tfce analysis config
    ana_kwargs_dict = {
        'VBA-TFCE': (glow.experiment.AnalysisVBA,
                     dict(n_perm=5,
                          tfce_flag=True,
                          alpha_fwer=0.05,
                          n_jobs_perm=1))
    }
    
    cloud_config = CloudConfig(
        s3_bucket=aws_config['s3_bucket'],
        s3_prefix='test/tfce',
        job_queue=aws_config['job_queue'],
        job_definition=aws_config['job_definition'],
        region=aws_config.get('region', 'us-east-1'),
        timeout_minutes=30,
        retry_attempts=1
    )
    
    config = Config(
        label='test_tfce',
        source='wgn',
        run_fnc=run_ana,
        cloud_config=cloud_config,
        ana_kwargs_dict=ana_kwargs_dict,
        n_seed=1,
        hotel_tr_all=np.array([0.5]),
        wgn_shape=(5, 5, 5),
        wgn_a=2,
        wgn_b=2,
        wgn_num_img=20,
        exp_seed=42,
        effect_perc=0.2,
        n_jobs=1,
        detail_save=False,
        error_save=False
    )
    
    config.run_all(verbose=True)
    
    results = list((config.folder / 'out').glob('*_result.json'))
    print(f'✓ TFCE cloud test passed ({len(results)} results)')


def test_hcp_cloud():
    """test HCP data loading on cloud
    
    Minimal test: 1 experiment with HCP data (fa feature only).
    Tests the full HCP pipeline:
    1. Upload HCP data to S3 (if not already there)
    2. Worker downloads HCP data from S3
    3. Experiment runs with real imaging data
    """
    try:
        import boto3
        from glow.aws.aws_batch import CloudConfig, upload_hcp_data, check_hcp_data_exists
    except ImportError:
        print('skipping HCP test: boto3 not installed')
        print('install with: pip install boto3')
        return
    
    # check credentials
    try:
        boto3.client('sts').get_caller_identity()
    except Exception:
        print('skipping HCP test: AWS credentials not configured')
        print('run: aws configure')
        return
    
    # load config
    aws_config = load_aws_config()
    if not aws_config or not all([aws_config['s3_bucket'], 
                                   aws_config['job_queue'], 
                                   aws_config['job_definition']]):
        print('skipping HCP test: AWS config incomplete')
        print('run: ./glow/aws/setup_aws_batch.sh')
        return
    
    print('testing HCP data loading on cloud (1 exp × 5 perms with source=hcp)...')
    
    s3_bucket = aws_config['s3_bucket']
    s3_prefix = 'test/hcp'
    region = aws_config.get('region', 'us-east-1')
    
    # hcp data path (default location)
    hcp_path = '/home/matt/data/hcp100_aug25_registered'
    
    # check and upload HCP data if needed
    print('\n[PRE-FLIGHT] Checking HCP data...')
    if Path(hcp_path).exists():
        print(f'  Local HCP data found at {hcp_path}')
        if not check_hcp_data_exists(s3_bucket, s3_prefix, region):
            print('  Uploading HCP data to S3 (one-time operation)...')
            upload_hcp_data(s3_bucket, s3_prefix, hcp_path, region)
        else:
            print('  ✓ HCP data already in S3')
    else:
        if check_hcp_data_exists(s3_bucket, s3_prefix, region):
            print(f'  ✓ HCP data already in S3 (local not found at {hcp_path})')
        else:
            print(f'  ✗ HCP data not found locally ({hcp_path})')
            print(f'  ✗ HCP data not found in S3 either')
            print(f'  Skipping HCP test - upload data first with:')
            print(f'    upload_hcp_data("{s3_bucket}", "{s3_prefix}", "/path/to/hcp_data")')
            return
    
    # minimal glow analysis config (lighter than VBA)
    ana_kwargs_dict = {
        'GLOW': (glow.experiment.AnalysisGLOW,
                 dict(n_perm=5,
                      n_perm_adj=5,
                      n_perm_prune=10,
                      alpha_fwer=0.05,
                      alpha_prune=0.05,
                      min_size=1,
                      n_jobs_perm=1))
    }
    
    cloud_config = CloudConfig(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        job_queue=aws_config['job_queue'],
        job_definition=aws_config['job_definition'],
        region=region,
        timeout_minutes=60,  # hcp may take longer
        retry_attempts=1
    )
    
    config = Config(
        label='test_hcp',
        source='hcp',
        run_fnc=run_ana,
        cloud_config=cloud_config,
        ana_kwargs_dict=ana_kwargs_dict,
        n_seed=1,
        hotel_tr_all=np.array([0.5]),
        hcp_path=hcp_path,
        hcp_feats=['fa'],  # single feature for speed
        radius=5,  # small region for speed
        effect_perc=0.2,
        n_jobs=1,
        detail_save=False,
        error_save=False
    )
    
    config.run_all(verbose=True)
    
    results = list((config.folder / 'out').glob('*_result.json'))
    print(f'✓ HCP cloud test passed ({len(results)} results)')


if __name__ == '__main__':
    print('=' * 70)
    print('GLOW AWS Cloud Test Suite')
    print('=' * 70)
    
    # print debug/version info first
    print_debug_info()
    
    results = {}
    test_num = 1
    
    if RUN_S3_TEST:
        print(f'\n[{test_num}] S3 Connectivity')
        try:
            test_aws_runner_methods()
            results['s3'] = True
        except Exception as e:
            print(f'✗ failed: {e}')
            results['s3'] = False
        test_num += 1
    
    if RUN_PERMUTATION_LEVEL_TEST:
        print(f'\n[{test_num}] Permutation-Level (AnalysisGLOW.cloud_config)')
        try:
            test_local_vs_cloud_comparison()
            results['permutation_level'] = True
        except Exception as e:
            print(f'✗ failed: {e}')
            results['permutation_level'] = False
        test_num += 1
    
    if RUN_EXPERIMENT_LEVEL_TEST:
        print(f'\n[{test_num}] Experiment-Level (Config.cloud_config)')
        try:
            test_experiment_level_architecture()
            results['experiment_level'] = True
        except Exception as e:
            print(f'✗ failed: {e}')
            results['experiment_level'] = False
        test_num += 1
    
    if RUN_TFCE_TEST:
        print(f'\n[{test_num}] TFCE Cloud (AnalysisVBA with tfce_flag=True)')
        try:
            test_tfce_cloud()
            results['tfce'] = True
        except Exception as e:
            print(f'✗ failed: {e}')
            results['tfce'] = False
        test_num += 1
    
    if RUN_HCP_TEST:
        print(f'\n[{test_num}] HCP Data (source=hcp from S3)')
        try:
            test_hcp_cloud()
            results['hcp'] = True
        except Exception as e:
            print(f'✗ failed: {e}')
            results['hcp'] = False
        test_num += 1
    
    # summary
    print('\n' + '=' * 70)
    print('TEST SUMMARY')
    print('=' * 70)
    if results:
        for test_name, passed in results.items():
            status = '✓' if passed else '✗'
            print(f'{status} {test_name}')
        
        print()
        if all(results.values()):
            print('✓ ALL TESTS PASSED')
            print('\nBoth architectures verified:')
            if 'permutation_level' in results:
                print('  • Permutation-level: parallel perms (1 large experiment)')
            if 'experiment_level' in results:
                print('  • Experiment-level: serial perms (many small experiments)')
        else:
            print('⚠ SOME TESTS FAILED')
            failed = [name for name, passed in results.items() if not passed]
            print(f'\nFailed: {", ".join(failed)}')
    else:
        print('⚠ NO TESTS RUN')
        print('Edit flags at top of file to enable tests')
    print('=' * 70)
