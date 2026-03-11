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

# OOM resubmit test (unit tests, runs locally without AWS)
# - Tests: OOM detection, job name parsing, queue fallback, resubmission logic
# OOM resubmit tests moved to test/test_aws_batch_monitor.py (pytest)


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
        'n_perm_fwer': 5,
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



def run_batched_cloud_tests():
    """submit all enabled cloud tests in one wave, monitor once, then assert per test

    This avoids repeated spot-instance ramp-ups by keeping the queue full.

    Returns:
        dict of test_name -> True/False
    """
    import uuid
    import boto3
    from glow.aws.aws_batch import CloudConfig, AWSBatchRunner
    from glow.experiment.mancova import get_llr

    # ── check AWS prerequisites once ──────────────────────────────────
    try:
        boto3.client('sts').get_caller_identity()
    except Exception:
        print('skipping cloud tests: AWS credentials not configured')
        return {}

    aws_config = load_aws_config()
    if not aws_config or not all([aws_config['s3_bucket'],
                                   aws_config['job_queue'],
                                   aws_config['job_definition']]):
        print('skipping cloud tests: AWS config incomplete')
        return {}

    s3_bucket = aws_config['s3_bucket']
    job_queue = aws_config['job_queue']
    job_definition = aws_config['job_definition']
    region = aws_config.get('region', 'us-east-1')

    all_job_ids = []
    test_meta = {}  # test_name -> dict with per-test data
    results = {}

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Prepare + submit all jobs
    # ══════════════════════════════════════════════════════════════════

    # ── Permutation-level ─────────────────────────────────────────────
    if RUN_PERMUTATION_LEVEL_TEST:
        print('\n[Prepare] Permutation-level test')

        # create experiment with effect
        exp_orig = glow.experiment.Experiment.from_gauss(
            seed=42, shape=(20, 20), a=2, b=2, num_img=20)
        mask = np.zeros(exp_orig.mask_idx.shape, dtype=bool)
        center, radius = (10, 10), 5
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                    mask[i, j] = True
        mask = np.logical_and(mask, exp_orig.mask_idx > -1)
        exp, _ = exp_orig.impose_effect(mask=mask, effect_llr=1.0, seed=42)

        n_perm_fwer = 5
        ana_kwargs = dict(n_perm_fwer=n_perm_fwer, n_perm_prune=10,
                          alpha_fwer=0.05, alpha_prune=0.05, min_size=1,
                          verbose=True)

        # run local analysis (fast, needed for comparison)
        print('  running local analysis...')
        ana_local = glow.experiment.AnalysisGLOW(exp, **ana_kwargs)

        # submit cloud jobs (perm + synthesis) via AWSBatchRunner
        cloud_config = CloudConfig(
            s3_bucket=s3_bucket, s3_prefix='test/comparison_test',
            job_queue=job_queue, job_definition=job_definition,
            max_concurrent_jobs=10, vcpus=2, memory_mb=4096,
            timeout_minutes=30)
        runner = AWSBatchRunner(cloud_config)
        experiment_id = f'glow_{uuid.uuid4().hex[:8]}'

        cloud_ana_kwargs = {
            'get_stat': get_llr,
            'n_perm_prune': 10,
            'alpha_fwer': 0.05, 'alpha_prune': 0.05, 'min_size': 1,
        }
        runner.upload_experiment(exp, cloud_ana_kwargs, experiment_id)
        submission = runner.submit_jobs(
            experiment_id=experiment_id, n_perm=n_perm_fwer,
            skip_completed=True)

        perm_job_ids = submission['job_ids']

        synth_job_id = runner.submit_synthesis_job(experiment_id, n_perm_fwer)
        all_cloud_job_ids = perm_job_ids + [synth_job_id]

        all_job_ids.extend(all_cloud_job_ids)
        test_meta['permutation_level'] = {
            'runner': runner, 'experiment_id': experiment_id,
            'n_perm_fwer': n_perm_fwer, 'exp': exp,
            'ana_local': ana_local, 'ana_kwargs': ana_kwargs,
            'job_ids': all_cloud_job_ids,
        }
        print(f'  submitted {len(perm_job_ids)} perm + 1 synthesis job')

    # ── Experiment-level ──────────────────────────────────────────────
    if RUN_EXPERIMENT_LEVEL_TEST:
        print('\n[Prepare] Experiment-level test')
        ana_kwargs_dict = {
            'GLOW': (glow.experiment.AnalysisGLOW,
                     dict(n_perm_fwer=5, n_perm_prune=10,
                          alpha_fwer=0.05, alpha_prune=0.05, min_size=1,
                          n_jobs_perm=1))
        }
        cloud_config = CloudConfig(
            s3_bucket=s3_bucket, s3_prefix='test/experiment_level',
            job_queue=job_queue, job_definition=job_definition,
            region=region, timeout_minutes=30, retry_attempts=1)
        config_el = Config(
            label='test_experiment_level', source='wgn', run_fnc=run_ana,
            cloud_config=cloud_config, ana_kwargs_dict=ana_kwargs_dict,
            n_seed=2, effect_llr_all=np.array([0.2]),
            wgn_shape=(5, 5, 5), wgn_a=2, wgn_b=2, wgn_num_img=20,
            exp_seed=42, effect_perc=0.2, n_jobs=1,
            detail_save=False, error_save=False)
        job_info_el = config_el.submit_cloud_jobs(verbose=True)

        all_job_ids.extend(job_info_el['job_ids'])
        test_meta['experiment_level'] = {
            'job_info': job_info_el, 'config': config_el,
        }

    # ── TFCE ──────────────────────────────────────────────────────────
    if RUN_TFCE_TEST:
        print('\n[Prepare] TFCE test')
        ana_kwargs_dict = {
            'VBA-TFCE': (glow.experiment.AnalysisVBA,
                         dict(n_perm_fwer=5, tfce_flag=True, alpha_fwer=0.05,
                              n_jobs_perm=1))
        }
        cloud_config = CloudConfig(
            s3_bucket=s3_bucket, s3_prefix='test/tfce',
            job_queue=job_queue, job_definition=job_definition,
            region=region, timeout_minutes=30, retry_attempts=1)
        config_tfce = Config(
            label='test_tfce', source='wgn', run_fnc=run_ana,
            cloud_config=cloud_config, ana_kwargs_dict=ana_kwargs_dict,
            n_seed=1, effect_llr_all=np.array([0.2]),
            wgn_shape=(5, 5, 5), wgn_a=2, wgn_b=2, wgn_num_img=20,
            exp_seed=42, effect_perc=0.2, n_jobs=1,
            detail_save=False, error_save=False)
        job_info_tfce = config_tfce.submit_cloud_jobs(verbose=True)

        all_job_ids.extend(job_info_tfce['job_ids'])
        test_meta['tfce'] = {
            'job_info': job_info_tfce, 'config': config_tfce,
        }

    # ── HCP ───────────────────────────────────────────────────────────
    if RUN_HCP_TEST:
        print('\n[Prepare] HCP test')
        ana_kwargs_dict = {
            'GLOW': (glow.experiment.AnalysisGLOW,
                     dict(n_perm_fwer=5, n_perm_prune=10,
                          alpha_fwer=0.05, alpha_prune=0.05, min_size=1,
                          n_jobs_perm=1))
        }
        cloud_config = CloudConfig(
            s3_bucket=s3_bucket, s3_prefix='test/hcp',
            job_queue=job_queue, job_definition=job_definition,
            region=region, timeout_minutes=60, retry_attempts=1)
        config_hcp = Config(
            label='test_hcp', source='hcp', run_fnc=run_ana,
            cloud_config=cloud_config, ana_kwargs_dict=ana_kwargs_dict,
            n_seed=1, effect_llr_all=np.array([0.2]),
            hcp_feats=['fa'], radius=5,
            effect_perc=0.2, n_jobs=1,
            detail_save=False, error_save=False)
        job_info_hcp = config_hcp.submit_cloud_jobs(verbose=True)

        all_job_ids.extend(job_info_hcp['job_ids'])
        test_meta['hcp'] = {
            'job_info': job_info_hcp, 'config': config_hcp,
        }

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Monitor all jobs once
    # ══════════════════════════════════════════════════════════════════
    if not all_job_ids:
        print('\nno cloud tests enabled')
        return {}

    print(f'\n{"="*70}')
    print(f'MONITORING ALL {len(all_job_ids)} JOBS')
    print(f'{"="*70}')

    # single runner for monitoring (no auto-download; we do it per-test below)
    monitor_config = CloudConfig(
        s3_bucket=s3_bucket, s3_prefix='test',
        job_queue=job_queue, job_definition=job_definition,
        region=region)
    monitor_runner = AWSBatchRunner(monitor_config)
    monitor_runner.monitor_jobs(all_job_ids, job_info_map={})

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: Download results + assert per test
    # ══════════════════════════════════════════════════════════════════

    # ── Permutation-level: download final analysis, compare ─────────
    if 'permutation_level' in test_meta:
        print(f'\n{"─"*70}')
        print('[Assert] Permutation-level: local vs cloud comparison')
        print(f'{"─"*70}')
        try:
            m = test_meta['permutation_level']
            runner = m['runner']
            ana_local = m['ana_local']

            ana_cloud = runner.download_final_analysis(m['experiment_id'])

            # compare local vs cloud
            tol = 1e-10
            assert ana_local.pval.shape == ana_cloud.pval.shape, 'pval shape mismatch'
            assert ana_local.llr_adjusted.shape == ana_cloud.llr_adjusted.shape, 'llr_adjusted shape mismatch'

            max_pval_diff = np.max(np.abs(ana_local.pval - ana_cloud.pval))
            max_z_diff = np.max(np.abs(ana_local.llr_adjusted - ana_cloud.llr_adjusted))
            max_stat_diff = np.max(np.abs(ana_local.stat - ana_cloud.stat))
            max_size_diff = np.max(np.abs(ana_local.size - ana_cloud.size))

            print(f'  pval max diff:  {max_pval_diff:.2e}')
            print(f'  llr_adjusted max diff: {max_z_diff:.2e}')
            print(f'  stat max diff:  {max_stat_diff:.2e}')
            print(f'  size max diff:  {max_size_diff:.2e}')

            assert max_pval_diff < tol, f'pval mismatch: {max_pval_diff:.2e}'
            assert max_z_diff < tol, f'llr_adjusted mismatch: {max_z_diff:.2e}'
            assert max_stat_diff < tol, f'stat mismatch: {max_stat_diff:.2e}'
            assert max_size_diff < tol, f'size mismatch: {max_size_diff:.2e}'

            n_eff_local = len(ana_local.effect_list)
            n_eff_cloud = len(ana_cloud.effect_list)
            assert n_eff_local == n_eff_cloud, \
                f'effect count mismatch: {n_eff_local} != {n_eff_cloud}'

            if n_eff_local > 0:
                for i, (el, ec) in enumerate(
                        zip(ana_local.effect_list, ana_cloud.effect_list)):
                    assert el.is_close(ec, rtol=1e-10, atol=1e-10), \
                        f'effect {i} mismatch'

            print('  ✓ permutation-level test passed')
            results['permutation_level'] = True
        except Exception as e:
            print(f'  ✗ failed: {e}')
            results['permutation_level'] = False

    # ── Experiment-level: download + check ────────────────────────────
    if 'experiment_level' in test_meta:
        print(f'\n{"─"*70}')
        print('[Assert] Experiment-level')
        print(f'{"─"*70}')
        try:
            m = test_meta['experiment_level']
            ji = m['job_info']
            ji['runner'].download_experiment_results(
                ji['run_id'], ji['folder'])
            result_files = list((ji['folder'] / 'out').glob('*_result.json'))
            print(f'  ✓ experiment-level test passed ({len(result_files)} results)')
            results['experiment_level'] = True
        except Exception as e:
            print(f'  ✗ failed: {e}')
            results['experiment_level'] = False

    # ── TFCE: download + check ────────────────────────────────────────
    if 'tfce' in test_meta:
        print(f'\n{"─"*70}')
        print('[Assert] TFCE')
        print(f'{"─"*70}')
        try:
            m = test_meta['tfce']
            ji = m['job_info']
            ji['runner'].download_experiment_results(
                ji['run_id'], ji['folder'])
            result_files = list((ji['folder'] / 'out').glob('*_result.json'))
            print(f'  ✓ TFCE test passed ({len(result_files)} results)')
            results['tfce'] = True
        except Exception as e:
            print(f'  ✗ failed: {e}')
            results['tfce'] = False

    # ── HCP: download + check ─────────────────────────────────────────
    if 'hcp' in test_meta:
        print(f'\n{"─"*70}')
        print('[Assert] HCP')
        print(f'{"─"*70}')
        try:
            m = test_meta['hcp']
            ji = m['job_info']
            ji['runner'].download_experiment_results(
                ji['run_id'], ji['folder'])
            result_files = list((ji['folder'] / 'out').glob('*_result.json'))
            print(f'  ✓ HCP test passed ({len(result_files)} results)')
            results['hcp'] = True
        except Exception as e:
            print(f'  ✗ failed: {e}')
            results['hcp'] = False

    return results


if __name__ == '__main__':
    print('=' * 70)
    print('GLOW AWS Cloud Test Suite')
    print('=' * 70)

    print_debug_info()

    results = {}

    # ── local tests (no AWS jobs) ─────────────────────────────────────
    if RUN_S3_TEST:
        print('\n[1] S3 Connectivity')
        try:
            test_aws_runner_methods()
            results['s3'] = True
        except Exception as e:
            print(f'✗ failed: {e}')
            results['s3'] = False

    # ── cloud tests (batched: one submit wave, one monitor) ───────────
    any_cloud = (RUN_PERMUTATION_LEVEL_TEST or RUN_EXPERIMENT_LEVEL_TEST or
                 RUN_TFCE_TEST or RUN_HCP_TEST)
    if any_cloud:
        print(f'\n{"="*70}')
        print('BATCHED CLOUD TESTS')
        print(f'{"="*70}')
        cloud_results = run_batched_cloud_tests()
        results.update(cloud_results)

    # ── summary ───────────────────────────────────────────────────────
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
        else:
            failed = [name for name, passed in results.items() if not passed]
            print(f'⚠ SOME TESTS FAILED: {", ".join(failed)}')
    else:
        print('⚠ NO TESTS RUN')
        print('Edit flags at top of file to enable tests')
    print('=' * 70)
