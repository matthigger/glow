"""Cloud runtime benchmark: AnalysisGLOW at varying HCP voxel counts.

Submits 31 jobs to AWS Batch, each running a full AnalysisGLOW on HCP data
subsampled to a geometrically spaced voxel count (100 up to the full mask).
No effect is imposed -- this measures pure runtime.

Usage::

    python -m glow.benchmark.runtime
"""

import configparser
import json
import time
from pathlib import Path
from uuid import uuid4

import numpy as np

import glow
from glow.aws.aws_batch import CloudConfig
from glow.benchmark.config import Config

N_TARGETS = 31
MIN_VOXELS = 100

GLOW_PARAMS = dict(
    n_perm=250,
    n_perm_prune=100,
    min_size=1,
    alpha_prune=0.05,
    alpha_fwer=0.05,
    n_jobs_perm=1,
    prune_method='node',
)


def run_runtime(config, num_voxels, seed=0, **kwargs):
    """Measure AnalysisGLOW runtime at a given voxel count."""
    from glow.experiment.exper import ExperimentScaled

    total_vox = int((config.exp_orig.mask_idx > -1).sum())

    if num_voxels >= total_vox:
        exp = config.exp_orig
    else:
        extenter = glow.effect.ExtenterSphere(
            n_vox=int(num_voxels), connected=True)
        mask = extenter(
            mask_idx=config.exp_orig.mask_idx, seed=seed, contiguous=True)
        exp = config.exp_orig.apply_mask(mask)

    exp = ExperimentScaled.from_exp(exp)
    actual_voxels = exp.y.shape[2]

    ana_kwargs = config.ana_kwargs_dict['GLOW'][1]
    start = time.time()
    glow.experiment.AnalysisGLOW(exp=exp, **ana_kwargs)
    elapsed_sec = time.time() - start

    out_dir = config.folder / 'out'
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        'num_voxels': int(actual_voxels),
        'target_voxels': int(num_voxels),
        'elapsed_sec': elapsed_sec,
        'elapsed_min': elapsed_sec / 60,
        'seed': int(seed),
        'config_hash': config._config_hash(),
    }
    uid = str(uuid4())[:8]
    with open(out_dir / f'{uid}_result.json', 'w') as f:
        json.dump(result, f, indent=4, sort_keys=True)


def load_cloud_config():
    config_file = Path(__file__).parents[2] / '.glow_aws_config'
    parser = configparser.ConfigParser()
    parser.read(config_file)

    return CloudConfig(
        s3_bucket=parser['aws']['s3_bucket'],
        s3_prefix='glow-runtime-benchmark',
        job_queue=parser['aws']['job_queue'],
        job_definition=parser['aws']['job_definition'],
        region=parser['aws']['region'],
        vcpus=int(parser['aws'].get('vcpus_per_job', 1)),
        timeout_minutes=360,
        retry_attempts=1,
    )


def main():
    cloud_config = load_cloud_config()

    ana_kwargs_dict = {
        'GLOW': (glow.experiment.AnalysisGLOW, GLOW_PARAMS),
    }

    config = Config(
        label='runtime_hcp',
        source='hcp',
        run_fnc=run_runtime,
        ana_kwargs_dict=ana_kwargs_dict,
        hcp_feats=['fa', 'md'],
        n_seed=1,
        n_jobs=1,
        detail_save=False,
        error_save=False,
    )

    print('Loading HCP data to determine voxel range...')
    config.prep_exp_orig()
    max_vox = int((config.exp_orig.mask_idx > -1).sum())
    print(f'  max voxels: {max_vox:,}')

    targets = np.geomspace(MIN_VOXELS, max_vox, N_TARGETS).round().astype(int)
    targets = np.unique(targets)
    print(f'  {len(targets)} voxel targets: {targets[0]:,} .. {targets[-1]:,}')

    config.iter_params = {
        'num_voxels': targets.tolist(),
        'seed': [0],
    }

    config.cloud_config = cloud_config

    job_info = config.submit_cloud_jobs(verbose=True)
    if job_info['job_ids']:
        config.wait_and_download_results(job_info, verbose=True)
    else:
        print('No jobs to run (all cached).')


if __name__ == '__main__':
    main()
