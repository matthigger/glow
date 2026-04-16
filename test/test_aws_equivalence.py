"""AWS equivalence tests — gated by --runaws (see conftest.py).

Submits all cloud jobs in ONE wave, calls monitor_jobs ONCE, then asserts
per-test. This keeps the Batch queue full so spot instances don't ramp
up repeatedly — critical for cost control.

Tests:
    - test_permutation_level_equivalence: AnalysisGLOW(cloud_config=X)
      parallel perms; compare local vs cloud pval/stat/size/llr_adjusted
      and effect_list elementwise.
    - test_experiment_level: Config(cloud_config=X) serial perms; verify
      result files downloaded.
    - test_tfce: VBA+TFCE on cloud; verify result files.
    - test_hcp: HCP data on cloud; verify result files.
"""
import uuid

import numpy as np
import pytest

import glow
from glow.benchmark.config import Config
from glow.benchmark.runner import RunAna

from test._aws_helpers import load_aws_config


# ---------------------------------------------------------------------------
# Session fixture: submit all jobs, monitor once, return keyed results
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def batched_cloud_results():
    """Submit all four AWS tests in one wave, monitor once, return results.

    Returns:
        dict keyed by test name, each value is a dict of per-test metadata
        needed for assertions (local analysis, cloud analysis, result
        folders, etc.).
    """
    # Import inside the fixture so collection doesn't fail when boto3 is
    # absent and tests are skipped.
    try:
        import boto3
        from glow.aws.aws_batch import CloudConfig, AWSBatchRunner
        from glow.analysis.mancova import get_llr
    except ImportError as e:
        pytest.skip(f'AWS deps not available: {e}')

    # AWS credentials check
    try:
        boto3.client('sts').get_caller_identity()
    except Exception as e:
        pytest.skip(f'AWS credentials not configured: {e}')

    aws_config = load_aws_config()
    if not aws_config or not all([aws_config.get('s3_bucket'),
                                   aws_config.get('job_queue'),
                                   aws_config.get('job_definition')]):
        pytest.skip('AWS config incomplete (.glow_aws_config missing)')

    s3_bucket = aws_config['s3_bucket']
    job_queue = aws_config['job_queue']
    job_definition = aws_config['job_definition']
    region = aws_config.get('region', 'us-east-1')

    all_job_ids = []
    test_meta = {}

    # ------------------------------------------------------------------
    # PHASE 1: Prepare + submit all jobs
    # ------------------------------------------------------------------

    # --- Permutation-level ---
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
    n_perm_fwer_size_adjust = 25
    n_perm_total = n_perm_fwer + n_perm_fwer_size_adjust
    ana_kwargs = dict(n_perm_fwer=n_perm_fwer,
                      n_perm_fwer_size_adjust=n_perm_fwer_size_adjust,
                      alpha_fwer=0.05, min_size=1, verbose=True)

    # local reference (fast)
    ana_local = glow.analysis.AnalysisGLOW(exp, **ana_kwargs)

    perm_cfg = CloudConfig(
        s3_bucket=s3_bucket, s3_prefix='test/comparison_test',
        job_queue=job_queue, job_definition=job_definition,
        max_concurrent_jobs=10, vcpus=2, memory_mb=4096,
        timeout_minutes=30)
    perm_runner = AWSBatchRunner(perm_cfg)
    experiment_id = f'glow_{uuid.uuid4().hex[:8]}'
    cloud_ana_kwargs = {'get_stat': get_llr,
                        'n_perm_fwer_size_adjust': n_perm_fwer_size_adjust,
                        'alpha_fwer': 0.05, 'min_size': 1}
    perm_runner.upload_experiment(exp, cloud_ana_kwargs, experiment_id)
    submission = perm_runner.submit_jobs(
        experiment_id=experiment_id, n_perm=n_perm_total,
        skip_completed=True)
    synth_job_id = perm_runner.submit_synthesis_job(
        experiment_id, n_perm_total)
    perm_job_ids = submission['job_ids'] + [synth_job_id]
    all_job_ids.extend(perm_job_ids)
    test_meta['permutation_level'] = {
        'runner': perm_runner,
        'experiment_id': experiment_id,
        'ana_local': ana_local,
    }

    # --- Experiment-level ---
    el_ana_dict = {
        'GLOW': (glow.analysis.AnalysisGLOW,
                 dict(n_perm_fwer=5, alpha_fwer=0.05, min_size=1,
                      n_jobs_perm=1)),
    }
    el_cloud_cfg = CloudConfig(
        s3_bucket=s3_bucket, s3_prefix='test/experiment_level',
        job_queue=job_queue, job_definition=job_definition,
        region=region, timeout_minutes=30, retry_attempts=1)
    config_el = Config(
        label='test_experiment_level', source='wgn',
        runner=RunAna(el_ana_dict),
        cloud_config=el_cloud_cfg,
        n_seed=2, effect_llr_all=np.array([0.2]),
        wgn_shape=(5, 5, 5), wgn_a=2, wgn_b=2, wgn_num_img=20,
        exp_seed=42, effect_perc=0.2, n_jobs=1,
        detail_save=False, error_save=False)
    job_info_el = config_el.submit_cloud_jobs(verbose=True)
    all_job_ids.extend(job_info_el['job_ids'])
    test_meta['experiment_level'] = {'job_info': job_info_el}

    # --- TFCE ---
    tfce_ana_dict = {
        'VBA-TFCE': (glow.analysis.AnalysisVBA,
                     dict(n_perm_fwer=5, tfce_flag=True,
                          alpha_fwer=0.05, n_jobs_perm=1)),
    }
    tfce_cloud_cfg = CloudConfig(
        s3_bucket=s3_bucket, s3_prefix='test/tfce',
        job_queue=job_queue, job_definition=job_definition,
        region=region, timeout_minutes=30, retry_attempts=1)
    config_tfce = Config(
        label='test_tfce', source='wgn',
        runner=RunAna(tfce_ana_dict),
        cloud_config=tfce_cloud_cfg,
        n_seed=1, effect_llr_all=np.array([0.2]),
        wgn_shape=(5, 5, 5), wgn_a=2, wgn_b=2, wgn_num_img=20,
        exp_seed=42, effect_perc=0.2, n_jobs=1,
        detail_save=False, error_save=False)
    job_info_tfce = config_tfce.submit_cloud_jobs(verbose=True)
    all_job_ids.extend(job_info_tfce['job_ids'])
    test_meta['tfce'] = {'job_info': job_info_tfce}

    # --- HCP ---
    hcp_ana_dict = {
        'GLOW': (glow.analysis.AnalysisGLOW,
                 dict(n_perm_fwer=5, alpha_fwer=0.05, min_size=1,
                      n_jobs_perm=1)),
    }
    hcp_cloud_cfg = CloudConfig(
        s3_bucket=s3_bucket, s3_prefix='test/hcp',
        job_queue=job_queue, job_definition=job_definition,
        region=region, timeout_minutes=60, retry_attempts=1)
    config_hcp = Config(
        label='test_hcp', source='hcp',
        runner=RunAna(hcp_ana_dict),
        cloud_config=hcp_cloud_cfg,
        n_seed=1, effect_llr_all=np.array([0.2]),
        hcp_feats=['fa'], radius=5,
        effect_perc=0.2, n_jobs=1,
        detail_save=False, error_save=False)
    job_info_hcp = config_hcp.submit_cloud_jobs(verbose=True)
    all_job_ids.extend(job_info_hcp['job_ids'])
    test_meta['hcp'] = {'job_info': job_info_hcp}

    # ------------------------------------------------------------------
    # PHASE 2: Monitor ALL jobs in a single wave
    # ------------------------------------------------------------------
    monitor_cfg = CloudConfig(
        s3_bucket=s3_bucket, s3_prefix='test',
        job_queue=job_queue, job_definition=job_definition,
        region=region)
    monitor_runner = AWSBatchRunner(monitor_cfg)
    monitor_runner.monitor_jobs(all_job_ids, job_info_map={})

    # ------------------------------------------------------------------
    # PHASE 3: Download results (per-test assertions live in tests)
    # ------------------------------------------------------------------

    # Permutation-level: final analysis
    m = test_meta['permutation_level']
    m['ana_cloud'] = m['runner'].download_final_analysis(m['experiment_id'])

    # The three experiment-style tests: download result files to folders.
    # Cached runs may have already aggregated out/*.json into results.csv,
    # so accept either as evidence that the cloud pipeline produced results.
    for key in ('experiment_level', 'tfce', 'hcp'):
        ji = test_meta[key]['job_info']
        if ji['runner'] is not None:
            ji['runner'].download_experiment_results(ji['run_id'], ji['folder'])
        raw = list((ji['folder'] / 'out').glob('*_result.json'))
        aggregated = ji['folder'] / 'results.csv'
        test_meta[key]['result_files'] = raw
        test_meta[key]['has_results'] = bool(raw) or aggregated.exists()

    return test_meta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.aws
def test_permutation_level_equivalence(batched_cloud_results):
    """Cloud AnalysisGLOW result must match local to tol=1e-10."""
    m = batched_cloud_results['permutation_level']
    ana_local = m['ana_local']
    ana_cloud = m['ana_cloud']

    tol = 1e-10

    assert ana_local.pval.shape == ana_cloud.pval.shape, \
        'pval shape mismatch'
    assert ana_local.llr_adjusted_0.shape == ana_cloud.llr_adjusted_0.shape, \
        'llr_adjusted_0 shape mismatch'

    max_pval_diff = np.max(np.abs(ana_local.pval - ana_cloud.pval))
    max_z_diff = np.max(
        np.abs(ana_local.llr_adjusted_0 - ana_cloud.llr_adjusted_0))
    max_stat_diff = np.max(np.abs(ana_local.stat - ana_cloud.stat))
    max_size_diff = np.max(np.abs(ana_local.size - ana_cloud.size))

    assert max_pval_diff < tol, f'pval mismatch: {max_pval_diff:.2e}'
    assert max_z_diff < tol, f'llr_adjusted_0 mismatch: {max_z_diff:.2e}'
    assert max_stat_diff < tol, f'stat mismatch: {max_stat_diff:.2e}'
    assert max_size_diff < tol, f'size mismatch: {max_size_diff:.2e}'

    n_eff_local = len(ana_local.effect_list)
    n_eff_cloud = len(ana_cloud.effect_list)
    assert n_eff_local == n_eff_cloud, \
        f'effect count mismatch: {n_eff_local} != {n_eff_cloud}'

    for i, (el, ec) in enumerate(
            zip(ana_local.effect_list, ana_cloud.effect_list)):
        assert el.is_close(ec, rtol=1e-10, atol=1e-10), \
            f'effect {i} mismatch'


@pytest.mark.aws
def test_experiment_level(batched_cloud_results):
    """Experiment-level cloud run should produce result files."""
    assert batched_cloud_results['experiment_level']['has_results'], \
        'no results from experiment-level run (neither out/*.json nor results.csv)'


@pytest.mark.aws
def test_tfce(batched_cloud_results):
    """TFCE cloud run should produce result files."""
    assert batched_cloud_results['tfce']['has_results'], \
        'no results from TFCE run (neither out/*.json nor results.csv)'


@pytest.mark.aws
def test_hcp(batched_cloud_results):
    """HCP cloud run should produce result files."""
    assert batched_cloud_results['hcp']['has_results'], \
        'no results from HCP run (neither out/*.json nor results.csv)'
