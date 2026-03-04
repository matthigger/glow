"""Cloud runtime benchmark: AnalysisGLOW at varying HCP voxel counts.

Runs in permutation mode on AWS Batch.  For each voxel count, n_perm + 1
jobs are launched (one per permutation plus a synthesis worker that polls
S3 and runs ``_finalize_analysis``).  Each permutation worker records its
own wall-clock time; the synthesis worker collects these and records its
own timing, attaching both to the final analysis object.

Usage::

    python -m glow.benchmark.runtime
    python -m glow.benchmark.runtime --min-voxels 500 --max-voxels 5000 --n-steps 5
"""

import argparse
import configparser
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from platformdirs import user_data_dir

import glow
from glow.aws.aws_batch import AWSBatchRunner, CloudConfig
from glow.experiment.mancova import get_llr

def _ana_kwargs(n_perm):
    return dict(
        get_stat=get_llr,
        n_perm_prune=n_perm,
        alpha_fwer=0.05,
        alpha_prune=0.05,
        min_size=1,
        prune_method='node',
    )


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


def prep_experiments(exp_orig, targets):
    """Subsample exp_orig at each target voxel count, return list of (exp, actual_voxels)."""
    max_vox = int((exp_orig.mask_idx > -1).sum())
    experiments = []
    for target in targets:
        target = int(target)
        if target >= max_vox:
            exp = exp_orig
        else:
            extenter = glow.effect.ExtenterSphere(n_vox=target, connected=True)
            mask = extenter(mask_idx=exp_orig.mask_idx, seed=0, contiguous=True)
            exp = exp_orig.apply_mask(mask)
        exp = glow.experiment.ExperimentScaled.from_exp(exp)
        experiments.append((exp, int(exp.y.shape[2]), target))
    return experiments


def _save_experiment_result(runner, experiment_id, meta, out_dir, n_perm):
    """Download final analysis for one experiment and write result JSON."""
    ana = runner.download_final_analysis(experiment_id)

    perm_elapsed = [t for t in getattr(ana, 'perm_elapsed_sec', [])
                    if t is not None]
    synth_elapsed = getattr(ana, 'synthesis_elapsed_sec', None)

    wall_sec = ((max(perm_elapsed) + (synth_elapsed or 0))
                if perm_elapsed else None)

    result = {
        'num_voxels': meta['actual_voxels'],
        'target_voxels': meta['target_voxels'],
        'n_perm': n_perm,
        'perm_elapsed_sec': perm_elapsed,
        'perm_median_sec': float(np.median(perm_elapsed)) if perm_elapsed else None,
        'perm_max_sec': float(max(perm_elapsed)) if perm_elapsed else None,
        'synthesis_elapsed_sec': synth_elapsed,
        'elapsed_sec': wall_sec,
        'elapsed_min': wall_sec / 60 if wall_sec is not None else None,
    }

    uid = uuid4().hex[:8]
    with open(out_dir / f'{uid}_result.json', 'w') as f:
        json.dump(result, f, indent=4, sort_keys=True)

    med = result['perm_median_sec']
    syn = synth_elapsed
    med_str = f'{med:.1f}s' if med is not None else 'n/a'
    syn_str = f'{syn:.1f}s' if syn is not None else 'n/a'
    print(f'\n  ✓ {meta["actual_voxels"]:>6,} voxels: '
          f'median perm {med_str}, synthesis {syn_str}')


def parse_args():
    p = argparse.ArgumentParser(
        description='Cloud runtime benchmark for AnalysisGLOW vs voxel count')
    p.add_argument('--min-voxels', type=int, default=1000,
                   help='smallest voxel count (default: 1000)')
    p.add_argument('--max-voxels', type=int, default=None,
                   help='largest voxel count (default: all voxels in HCP mask)')
    p.add_argument('--n-steps', type=int, default=10,
                   help='number of log-spaced voxel counts from min to max '
                        '(default: 10)')
    p.add_argument('--n-perm', type=int, default=100,
                   help='number of permutations per experiment (default: 100)')
    return p.parse_args()


def main():
    args = parse_args()
    cloud_config = load_cloud_config()
    runner = AWSBatchRunner(cloud_config)

    print('Loading HCP data...')
    from glow.benchmark.hcp_data import get_hcp_path
    path = get_hcp_path()
    exp_orig = glow.experiment.ExperimentImageOnly.from_search(
        folder=path,
        sbj_regex=r'[\d]{6}',
        img_glob_dict={'fa': '*_fa.nii.gz', 'md': '*_md.nii.gz'})
    exp_orig = exp_orig.sample_x(a=2, seed=0, add_bias=True)

    max_vox = int((exp_orig.mask_idx > -1).sum())
    print(f'  max voxels in HCP mask: {max_vox:,}')

    max_vox_target = min(args.max_voxels, max_vox) if args.max_voxels else max_vox
    targets = np.geomspace(args.min_voxels, max_vox_target,
                           args.n_steps).round().astype(int)
    targets = np.unique(targets)
    n_perm = args.n_perm
    print(f'  {len(targets)} voxel targets: {targets[0]:,} .. {targets[-1]:,}')
    print(f'  n_perm: {n_perm}')

    base = Path(user_data_dir('glow', 'glow_author'))
    out_dir = base / 'results' / 'runtime_hcp' / 'out'
    out_dir.mkdir(parents=True, exist_ok=True)

    ana_kwargs = _ana_kwargs(n_perm)

    print('\nPreparing experiments...')
    exps = prep_experiments(exp_orig, targets)

    all_job_ids = []
    exp_meta = {}
    job_info_map = {}

    print(f'\nUploading & submitting ({n_perm} perm + 1 synthesis per experiment)...')
    for exp, actual_vox, target_vox in exps:
        experiment_id = f'runtime_{actual_vox}v_{uuid4().hex[:8]}'

        runner.upload_experiment(exp, ana_kwargs, experiment_id)
        submission = runner.submit_jobs(
            experiment_id=experiment_id, n_perm=n_perm, skip_completed=True)
        synth_job_id = runner.submit_synthesis_job(experiment_id, n_perm)

        perm_ids = submission['job_ids']
        all_job_ids.extend(perm_ids)
        all_job_ids.append(synth_job_id)

        meta = {
            'target_voxels': target_vox,
            'actual_voxels': actual_vox,
            'n_perm_jobs': len(perm_ids),
        }
        exp_meta[experiment_id] = meta

        job_info_map[synth_job_id] = {
            'on_complete': lambda _job, eid=experiment_id, m=meta: (
                _save_experiment_result(runner, eid, m, out_dir, n_perm)),
        }

        print(f'  {actual_vox:>6,} voxels: {len(perm_ids)} perm + 1 synthesis')

    print(f'\nTotal jobs: {len(all_job_ids)}')

    runner.monitor_jobs(all_job_ids, job_info_map=job_info_map)

    print(f'\nResults saved to: {out_dir}')


if __name__ == '__main__':
    main()
