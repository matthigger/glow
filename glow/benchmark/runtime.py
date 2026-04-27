"""Cloud runtime benchmarks and timeout estimation.

Three profiling modes (selected by ``--profile``):

  permutation (default)
      Per-permutation timing of AnalysisGLOW at varying HCP voxel counts.
      For each voxel count, n_perm + 1 jobs are launched (one per
      permutation plus a synthesis worker).  Each permutation worker
      records its own wall-clock time.

  experiment
      Full ``run_ana`` timing on WGN across a grid of (num_vox, b,
      num_img, n_perm) for GLOW, VBA, and VBA-TFCE independently.
      Fits per-analysis-type Lasso regression models and saves them as
      JSON files used by :func:`estimate_timeout_minutes`.

  spinup
      Measures cold-start latency: submits a no-op job to AWS Batch
      and records the time from SUBMITTED to RUNNING.  Warns if the
      queue has active jobs (warm instances bias the measurement).

Usage::

    python -m glow.benchmark.runtime                         # permutation (default)
    python -m glow.benchmark.runtime --profile experiment    # fit runtime models
    python -m glow.benchmark.runtime --profile spinup        # cold-start timing
    python -m glow.benchmark.runtime --profile spinup --n 5  # repeated measurements
"""

import argparse
import configparser
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from uuid import uuid4

import numpy as np
from platformdirs import user_data_dir

import glow
from glow.aws.aws_batch import AWSBatchRunner, CloudConfig
from glow.analysis.mancova import get_llr


# ---------------------------------------------------------------------------
# Runtime model paths and constants
# ---------------------------------------------------------------------------

RUNTIME_MODEL_DIR = Path(__file__).resolve().parent.parent / 'aws'
RUNTIME_MODEL_PATHS = {
    'GLOW': RUNTIME_MODEL_DIR / 'runtime_glow.json',
    'VBA': RUNTIME_MODEL_DIR / 'runtime_vba.json',
    'VBA-TFCE': RUNTIME_MODEL_DIR / 'runtime_vba_tfce.json',
}
RUNTIME_FEATURE_COLS = ['num_vox', 'b', 'num_img', 'n_perm']


# ---------------------------------------------------------------------------
# Load / predict runtime models
# ---------------------------------------------------------------------------

def load_runtime_model(analysis_type):
    """Load a fitted runtime model for *analysis_type*, or ``None``.

    *analysis_type* is one of ``'GLOW'``, ``'VBA'``, ``'VBA-TFCE'``.
    """
    from glow.benchmark.memory import load_model
    path = RUNTIME_MODEL_PATHS.get(analysis_type)
    if path is None:
        return None
    return load_model(path)


def predict_runtime_sec(model, num_vox, b, num_img, n_perm):
    """Predict runtime in seconds from a fitted runtime model dict."""
    from glow.benchmark.memory import _apply_poly_model
    return _apply_poly_model(
        model, RUNTIME_FEATURE_COLS,
        num_vox=num_vox, b=b, num_img=num_img, n_perm=n_perm,
    )


def _get_analysis_type(Ana, ana_kw):
    """Map (AnalysisClass, kwargs) to a runtime model key."""
    name = Ana.__name__
    if name == 'AnalysisGLOW':
        return 'GLOW'
    if name == 'AnalysisVBA':
        return 'VBA-TFCE' if ana_kw.get('tfce_flag', False) else 'VBA'
    if name == 'AnalysisCET':
        return 'VBA'
    return None


def _get_total_perms(Ana, ana_kw):
    """Total permutations for an analysis (including size-adjust for GLOW)."""
    n = ana_kw.get('n_perm_fwer', 100)
    if Ana.__name__ == 'AnalysisGLOW':
        n += ana_kw.get('n_perm_fwer_size_adjust', 0)
    return n


def _get_config_dimensions(config):
    """Extract ``(num_vox, b, num_img)`` from a Config."""
    if config.source == 'wgn':
        num_vox = config.crop_n_vox
        if num_vox is None:
            num_vox = int(np.prod(config.wgn_shape))
        return num_vox, config.wgn_b, config.wgn_num_img

    num_vox = config.crop_n_vox or 50_000
    b = len(config.hcp_feats)
    if config.exp_orig is not None:
        num_img = config.exp_orig.y.shape[1]
    else:
        num_img = 100
    return num_vox, b, num_img


_DEFAULT_SAFETY_FACTOR = 2.5


def _get_model_safety_factor(model):
    """Pull the per-model calibrated safety factor, or fall back to 2.5."""
    if model is None:
        return _DEFAULT_SAFETY_FACTOR
    return float(model.get('safety_factor', _DEFAULT_SAFETY_FACTOR))


def estimate_timeout_minutes(config, safety_factor=None):
    """Estimate per-job timeout for *config*'s experiment jobs.

    Returns ``(timeout_minutes, estimated_minutes, is_upper_bound)`` or
    ``None`` if no fitted runtime models are available.

    When ``safety_factor`` is None, the per-model calibrated 99.99%-quantile
    factor (written by ``fit_poly_lasso``) is used; otherwise the caller's
    value applies uniformly.
    """
    from glow.benchmark.runner import (RunAna, RunSegment, RunPruneCompare,
                                       RunMancovaGlow, RunMancovaVba)
    from glow.analysis.mancova import stat_dict

    num_vox, b, num_img = _get_config_dimensions(config)
    total_sec = 0.0
    is_upper_bound = False
    runner = config.runner
    factor_used = None

    def _apply_model(model):
        nonlocal factor_used
        f = (safety_factor if safety_factor is not None
             else _get_model_safety_factor(model))
        factor_used = max(factor_used or 0.0, f)

    if isinstance(runner, RunAna):
        for _label, (Ana, ana_kw) in runner.ana_kwargs_dict.items():
            atype = _get_analysis_type(Ana, ana_kw)
            if atype is None:
                return None
            model = load_runtime_model(atype)
            if model is None:
                return None
            n_perm = _get_total_perms(Ana, ana_kw)
            total_sec += max(0.0, predict_runtime_sec(
                model, num_vox, b, num_img, n_perm))
            _apply_model(model)

    elif isinstance(runner, RunPruneCompare):
        Ana, ana_kw = next(iter(runner.iter_ana_kwargs()))[1]
        atype = _get_analysis_type(Ana, ana_kw)
        model = load_runtime_model(atype) if atype else None
        if model is None:
            return None
        n_perm = _get_total_perms(Ana, ana_kw)
        total_sec = max(0.0, predict_runtime_sec(
            model, num_vox, b, num_img, n_perm))
        _apply_model(model)

    elif isinstance(runner, RunMancovaGlow):
        Ana, ana_kw = next(iter(runner.iter_ana_kwargs()))[1]
        model = load_runtime_model('GLOW')
        if model is None:
            return None
        n_perm = _get_total_perms(Ana, ana_kw)
        # runtime model is for 1 stat; mancova evaluates all stats per walk
        total_sec = max(0.0, predict_runtime_sec(
            model, num_vox, b, num_img, n_perm)) * len(stat_dict)
        is_upper_bound = True
        _apply_model(model)

    elif isinstance(runner, RunMancovaVba):
        Ana, ana_kw = next(iter(runner.iter_ana_kwargs()))[1]
        model = load_runtime_model('VBA-TFCE')
        if model is None:
            return None
        n_perm = _get_total_perms(Ana, ana_kw)
        total_sec = max(0.0, predict_runtime_sec(
            model, num_vox, b, num_img, n_perm)) * len(stat_dict)
        is_upper_bound = True
        _apply_model(model)

    elif isinstance(runner, RunSegment):
        # RunSegment does Ward's clustering, no permutations. No dedicated
        # runtime model; we use GLOW @ 100 perms as a loose upper bound.
        model = load_runtime_model('GLOW')
        if model is None:
            return None
        total_sec = max(0.0, predict_runtime_sec(
            model, num_vox, b, num_img, 100))
        is_upper_bound = True
        _apply_model(model)

    else:
        return None

    factor = factor_used if factor_used is not None else _DEFAULT_SAFETY_FACTOR
    estimated_min = total_sec / 60.0
    timeout_min = max(15.0, estimated_min * factor)
    return timeout_min, estimated_min, is_upper_bound


# ---------------------------------------------------------------------------
# Permutation benchmark helpers (existing)
# ---------------------------------------------------------------------------

def _ana_kwargs(n_perm):
    return dict(
        get_stat=get_llr,
        alpha_fwer=0.05,
        min_size=1,
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


def prep_experiments(exp_orig, targets, effect_llr=0, effect_perc=0.2):
    """Subsample exp_orig at each target voxel count, optionally impose effect."""
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

        if effect_llr > 0:
            n_eff = max(1, int(exp.y.shape[2] * effect_perc))
            eff_ext = glow.effect.ExtenterSphere(n_vox=n_eff)
            exp, _effect = exp.impose_effect(
                effect_llr=effect_llr, extenter=eff_ext, seed=0)

        experiments.append((exp, int(exp.y.shape[2]), target))
    return experiments


def _get_spinup_times(batch_client, job_ids):
    """Query describe_jobs and return per-job spinup seconds.

    Spinup = startedAt - createdAt (both are epoch-ms from AWS).
    Returns a list of floats (one per job that has both timestamps).
    """
    spinups = []
    # describe_jobs accepts up to 100 IDs per call
    for i in range(0, len(job_ids), 100):
        chunk = job_ids[i:i + 100]
        resp = batch_client.describe_jobs(jobs=chunk)
        for job in resp.get('jobs', []):
            created = job.get('createdAt')
            started = job.get('startedAt')
            if created is not None and started is not None:
                spinups.append((started - created) / 1000.0)
    return spinups


def _save_experiment_result(runner, experiment_id, meta, out_dir, n_perm,
                            effect_llr=0, effect_perc=0):
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
        'effect_llr': effect_llr,
        'effect_perc': effect_perc,
        'perm_elapsed_sec': perm_elapsed,
        'perm_median_sec': float(np.median(perm_elapsed)) if perm_elapsed else None,
        'perm_max_sec': float(max(perm_elapsed)) if perm_elapsed else None,
        'synthesis_elapsed_sec': synth_elapsed,
        'elapsed_sec': wall_sec,
        'elapsed_min': wall_sec / 60 if wall_sec is not None else None,
    }

    # spinup times are added in a post-hoc pass (see main_permutation)

    uid = uuid4().hex[:8]
    with open(out_dir / f'{uid}_result.json', 'w') as f:
        json.dump(result, f, indent=4, sort_keys=True)

    med = result['perm_median_sec']
    syn = synth_elapsed
    med_str = f'{med:.1f}s' if med is not None else 'n/a'
    syn_str = f'{syn:.1f}s' if syn is not None else 'n/a'
    print(f'\n  ✓ {meta["actual_voxels"]:>6,} voxels: '
          f'median perm {med_str}, synthesis {syn_str}')


def main_permutation(args):
    """Run permutation-mode runtime benchmark on HCP (existing behaviour)."""
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
    effect_llr = args.effect_llr
    effect_perc = args.effect_perc
    print(f'  {len(targets)} voxel targets: {targets[0]:,} .. {targets[-1]:,}')
    print(f'  n_perm: {n_perm}')
    if effect_llr > 0:
        print(f'  effect: LLR={effect_llr}, {effect_perc:.0%} of voxels')
    else:
        print('  effect: null (no imposed effect)')

    base = Path(user_data_dir('glow', 'glow_author'))
    out_dir = base / 'results' / 'runtime' / 'permutation' / 'out'

    existing = list(out_dir.glob('*_result.json')) if out_dir.exists() else []
    if existing:
        resp = input(f'\n  {len(existing)} previous result files in {out_dir}\n'
                     f'  Delete them? [y/N] ').strip().lower()
        if resp == 'y':
            shutil.rmtree(out_dir)
            print(f'  Cleared {len(existing)} files.')

    out_dir.mkdir(parents=True, exist_ok=True)

    ana_kwargs = _ana_kwargs(n_perm)

    print('\nPreparing experiments...')
    exps = prep_experiments(exp_orig, targets,
                            effect_llr=effect_llr, effect_perc=effect_perc)

    all_job_ids = []
    exp_job_ids = {}  # experiment_id -> list of perm job IDs
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
        exp_job_ids[experiment_id] = perm_ids

        meta = {
            'target_voxels': target_vox,
            'actual_voxels': actual_vox,
            'n_perm_jobs': len(perm_ids),
        }
        exp_meta[experiment_id] = meta

        job_info_map[synth_job_id] = {
            'on_complete': lambda _job, eid=experiment_id, m=meta: (
                _save_experiment_result(runner, eid, m, out_dir, n_perm,
                                        effect_llr, effect_perc)),
        }

        print(f'  {actual_vox:>6,} voxels: {len(perm_ids)} perm + 1 synthesis')

    print(f'\nTotal jobs: {len(all_job_ids)}')

    runner.monitor_jobs(all_job_ids, job_info_map=job_info_map)

    # collect spinup times from AWS timestamps and patch into result JSONs
    print('\nCollecting spinup times from AWS timestamps...')
    # build experiment_id -> result file mapping (by matching num_voxels)
    result_files = {
        json.load(open(p))['num_voxels']: p
        for p in out_dir.glob('*_result.json')
    }

    for experiment_id, perm_ids in exp_job_ids.items():
        spinups = _get_spinup_times(runner.batch, perm_ids)
        if not spinups:
            continue
        vox = exp_meta[experiment_id]['actual_voxels']
        path = result_files.get(vox)
        if path is None:
            continue
        with open(path) as f:
            result = json.load(f)
        result['spinup_sec'] = spinups
        result['spinup_median_sec'] = float(np.median(spinups))
        with open(path, 'w') as f:
            json.dump(result, f, indent=4, sort_keys=True)
        print(f'  {vox:>6,} voxels: median spinup {np.median(spinups):.1f}s '
              f'(n={len(spinups)})')

    print(f'\nResults saved to: {out_dir}')


# ---------------------------------------------------------------------------
# Experiment runtime profiling (run on AWS)
# ---------------------------------------------------------------------------

RUNTIME_EXPERIMENT_DIR = (
    Path(user_data_dir('glow', 'glow_author')) / 'results' / 'runtime' / 'experiment'
)


def _build_runtime_profile_configs(cloud_config=None):
    """Build Config objects for the runtime profiling grid."""
    from glow.benchmark.config import Config
    from glow.benchmark.runner import RunAna

    if cloud_config is not None:
        # reduced grid for cloud: cap voxels at 30k and drop 1050-perm
        # to avoid OOM (previous runs hit 16 GB ceiling at 50k x 1050).
        # 5 * 2 * 3 * 3 = 90 grid points (x3 analysis types = 270 jobs),
        # still plenty for Lasso.  The polynomial model + 2.5x safety
        # factor in estimate_timeout_minutes covers extrapolation.
        vox_targets = np.geomspace(500, 30_000, 5).round().astype(int).tolist()
        n_perm_values = [100, 500]
    else:
        # reduced grid for local: drop expensive 50k-vox and 1050-perm
        # configs.  The polynomial model extrapolates; the 2.5x safety
        # factor in estimate_timeout_minutes covers the gap.
        vox_targets = np.geomspace(500, 20_000, 4).round().astype(int).tolist()
        n_perm_values = [100, 500]
    b_values = [1, 4] if cloud_config is None else [1, 2, 4]
    img_values = [25, 100] if cloud_config is None else [25, 50, 100]

    max_side = 40  # 40^3 = 64000, larger than any vox target

    common = dict(
        source='wgn',
        n_seed=1,
        effect_llr_all=np.array([0.05]),
        effect_perc=0.2,
        wgn_shape=(max_side, max_side, max_side),
        wgn_a=2,
        n_jobs=1,
        detail_save=False,
        error_save=False,
        cloud_config=cloud_config,
        result_dir=RUNTIME_EXPERIMENT_DIR,
    )

    configs = []
    for n_perm, b, num_img, vox in product(
            n_perm_values, b_values, img_values, vox_targets):
        n_sa = min(50, n_perm // 4)
        n_fwer_glow = n_perm - n_sa

        configs.append(Config(
            label=f'rtprof_glow_{vox}v_{b}b_{num_img}i_{n_perm}p',
            runner=RunAna({'GLOW': (
                glow.analysis.AnalysisGLOW,
                dict(n_perm_fwer=n_fwer_glow,
                     n_perm_fwer_size_adjust=n_sa,
                     alpha_fwer=0.05, min_size=1),
            )}),
            wgn_b=b, wgn_num_img=num_img, crop_n_vox=vox,
            **common,
        ))

        configs.append(Config(
            label=f'rtprof_vba_{vox}v_{b}b_{num_img}i_{n_perm}p',
            runner=RunAna({'VBA': (
                glow.analysis.AnalysisVBA,
                dict(n_perm_fwer=n_perm, tfce_flag=False,
                     alpha_fwer=0.05),
            )}),
            wgn_b=b, wgn_num_img=num_img, crop_n_vox=vox,
            **common,
        ))

        configs.append(Config(
            label=f'rtprof_vba_tfce_{vox}v_{b}b_{num_img}i_{n_perm}p',
            runner=RunAna({'VBA-TFCE': (
                glow.analysis.AnalysisVBA,
                dict(n_perm_fwer=n_perm, tfce_flag=True,
                     alpha_fwer=0.05),
            )}),
            wgn_b=b, wgn_num_img=num_img, crop_n_vox=vox,
            **common,
        ))

    return configs


def _collect_runtime_results(configs):
    """Collect timing data from completed profiling results."""
    from glow.benchmark.file import load_update_all
    import pandas as pd

    rows = []
    for config in configs:
        df, _folder, _ = load_update_all(
            config.label, verbose=False, result_dir=RUNTIME_EXPERIMENT_DIR)
        if df.empty:
            continue
        _label, (Ana, ana_kw) = next(iter(config.runner.iter_ana_kwargs()))
        n_perm = _get_total_perms(Ana, ana_kw)
        for _, row in df.iterrows():
            rows.append({
                'num_vox': int(row.get('vox_total', 0)),
                'b': config.wgn_b,
                'num_img': config.wgn_num_img,
                'n_perm': n_perm,
                'analysis_type': str(row['label']),
                'time_sec': float(row['time_sec']),
            })

    return pd.DataFrame(rows)


def _fit_runtime_models(df):
    """Fit per-analysis-type Lasso regression models on timing data."""
    from glow.benchmark.memory import fit_poly_lasso

    models = {}
    for analysis_type, path in RUNTIME_MODEL_PATHS.items():
        df_type = df[df['analysis_type'] == analysis_type]
        if df_type.empty:
            print(f'  no data for {analysis_type}, skipping')
            continue
        print(f'\n  Fitting {analysis_type} model ({len(df_type)} points)...')
        models[analysis_type] = fit_poly_lasso(
            df_type,
            feature_cols=RUNTIME_FEATURE_COLS,
            target_col='time_sec',
            model_path=path,
            log_target=True,
        )
    return models


def _run_one_profile(config, kwargs):
    """Run one profiling experiment locally, catching errors."""
    try:
        config.runner.run(config=config, **kwargs)
    except Exception as e:
        print(f'  ✗ {config.label}: {e}')


def _run_profile_local(configs, n_jobs=-1):
    """Run profiling configs locally in parallel."""
    import joblib
    from joblib import Parallel, delayed

    # Cache exp_orig by WGN parameters (486 configs share only 9 unique
    # experiments).  Also cache the expensive _hash() call.
    print('  Checking cache...')
    exp_cache = {}
    to_run = []
    for config in configs:
        config.prep_folder()

        cache_key = (config.wgn_shape, config.wgn_a, config.wgn_b,
                     config.wgn_num_img, config.exp_seed)
        if cache_key not in exp_cache:
            config.prep_exp_orig()
            h = config.exp_orig._hash()
            config.exp_orig._hash = lambda _h=h: _h
            exp_cache[cache_key] = config.exp_orig
        else:
            config.exp_orig = exp_cache[cache_key]

        config.save_config(config.folder / 'config.yaml')
        kwargs_list = list(config.iter_kwargs())
        uncached = config._filter_uncached(kwargs_list, verbose=False)
        for _, kwargs, _missing in uncached:
            to_run.append((config, kwargs))
        # Clear exp_orig to reduce pickle size for multiprocessing;
        # workers recreate it from WGN parameters.
        config.exp_orig = None

    n_cached = len(configs) - len(to_run)
    if n_cached:
        print(f'  {n_cached} cached, {len(to_run)} to run')
    if not to_run:
        print('  all experiments cached')
        return

    print(f'  Running {len(to_run)} experiments locally (n_jobs={n_jobs}, '
          f'1 BLAS thread per worker)...')
    # Pin BLAS to 1 thread per worker so each profile sample describes
    # per-process-1-thread runtime — the regime that matches AWS Batch
    # workers and the parallel-local runner.  Without this, workers fight
    # for the same threadpool and the fitted model conflates BLAS contention
    # with workload size.
    with joblib.parallel_config(inner_max_num_threads=1):
        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_run_one_profile)(config, kwargs)
            for config, kwargs in to_run
        )


def main_experiment_profile(cloud=False, n_jobs=-1):
    """Run experiment-mode runtime profiling and fit models."""
    if cloud:
        cloud_config = load_cloud_config()
        cloud_config.timeout_minutes = 360
        cloud_config.retry_attempts = 1
    else:
        cloud_config = None

    configs = _build_runtime_profile_configs(cloud_config)
    n_grid = len(configs) // 3
    mode = 'cloud' if cloud else 'local'
    print(f'Runtime profiling: {len(configs)} experiment jobs ({mode})')
    print(f'  ({n_grid} grid points x 3 analysis types)')

    if cloud:
        from glow.benchmark.paper import (submit_all_jobs, build_job_info_map,
                                           monitor_all_jobs,
                                           download_remaining_results)

        all_job_info = submit_all_jobs(configs)
        all_job_ids, job_info_map = build_job_info_map(all_job_info)
        monitor_all_jobs(all_job_info, all_job_ids, job_info_map)
        download_remaining_results(all_job_info)
    else:
        _run_profile_local(configs, n_jobs=n_jobs)

    print('\n' + '=' * 60)
    print('Fitting runtime models...')
    print('=' * 60)

    df = _collect_runtime_results(configs)
    if df.empty:
        print('  no results collected')
        return

    print(f'  collected {len(df)} timing measurements')
    models = _fit_runtime_models(df)

    if models:
        print('\n  Sample predictions:')
        for atype in ['GLOW', 'VBA', 'VBA-TFCE']:
            model = load_runtime_model(atype)
            if model:
                pred = predict_runtime_sec(model, 50000, 2, 100, 1050)
                print(f'    {atype}: (50k vox, b=2, 100 img, '
                      f'1050 perm) -> {pred:.0f}s ({pred / 60:.1f} min)')


# ---------------------------------------------------------------------------
# spinup profiling
# ---------------------------------------------------------------------------

def _check_active_jobs(batch, job_queue):
    """Return count of non-terminal jobs in the queue."""
    active = 0
    for status in ('SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING'):
        response = batch.list_jobs(jobQueue=job_queue, jobStatus=status)
        active += len(response.get('jobSummaryList', []))
    return active


def _measure_spinup(cloud_config, quiet=False):
    """Submit a no-op job and return a dict of state transition times."""
    import boto3

    batch = boto3.client('batch', region_name=cloud_config.region)

    # pass --help so the worker's argparser exits 0 immediately after
    # image pull + container boot + Python startup + glow imports
    response = batch.submit_job(
        jobName='glow_spinup_probe',
        jobQueue=cloud_config.job_queue,
        jobDefinition=cloud_config.job_definition,
        containerOverrides={'command': ['--help']},
        retryStrategy={'attempts': 1},
        timeout={'attemptDurationSeconds': 300},
    )
    job_id = response['jobId']
    t_submit = time.monotonic()

    if not quiet:
        print(f'  job {job_id} submitted')

    # poll until terminal state, recording first-seen times
    seen = {}
    while True:
        resp = batch.describe_jobs(jobs=[job_id])
        job = resp['jobs'][0]
        status = job['status']

        if status not in seen:
            elapsed = time.monotonic() - t_submit
            seen[status] = elapsed
            if not quiet:
                print(f'  {status:>10s}  +{elapsed:6.1f}s')

        if status in ('SUCCEEDED', 'FAILED'):
            break

        time.sleep(2)

    # extract AWS-side epoch timestamps
    aws_times = {}
    for key in ('createdAt', 'startedAt', 'stoppedAt'):
        ms = job.get(key)
        if ms is not None:
            aws_times[key] = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

    result = {
        'job_id': job_id,
        'status': status,
        'seen': seen,
        'aws_times': aws_times,
    }

    # spinup = creation to RUNNING (AWS-side timestamps preferred)
    if 'createdAt' in aws_times and 'startedAt' in aws_times:
        spinup = (aws_times['startedAt'] - aws_times['createdAt']).total_seconds()
        result['spinup_sec'] = spinup
    elif 'RUNNING' in seen:
        result['spinup_sec'] = seen['RUNNING']

    return result


def main_spinup(n=1, quiet=False):
    """Measure cold-start spinup time for AWS Batch jobs."""
    import boto3

    cloud_config = load_cloud_config()
    cloud_config.timeout_minutes = 5
    cloud_config.retry_attempts = 1

    batch = boto3.client('batch', region_name=cloud_config.region)

    # warn if queue has active jobs (warm instances bias the measurement)
    active = _check_active_jobs(batch, cloud_config.job_queue)
    if active > 0:
        print(f'WARNING: {active} active job(s) in queue '
              f'"{cloud_config.job_queue}" — instances may be warm, '
              f'spinup times will be artificially low', file=sys.stderr)
        print(f'  Wait for queue to drain for a cold-start measurement.',
              file=sys.stderr)

    spinups = []
    for i in range(n):
        if n > 1 and not quiet:
            print(f'\n--- measurement {i + 1}/{n} ---')
        result = _measure_spinup(cloud_config, quiet=quiet)

        if result['status'] == 'FAILED':
            print(f'  job FAILED (job_id={result["job_id"]})', file=sys.stderr)
            continue

        sec = result.get('spinup_sec')
        if sec is not None:
            spinups.append(sec)
            if not quiet:
                print(f'  spinup: {sec:.1f}s')

        # wait for queue to drain between measurements
        if i < n - 1:
            if not quiet:
                print('  waiting for queue to drain...')
            while _check_active_jobs(batch, cloud_config.job_queue) > 0:
                time.sleep(10)

    if not spinups:
        print('no successful measurements', file=sys.stderr)
        sys.exit(1)

    if n > 1:
        import statistics
        print(f'\n=== {len(spinups)} measurements ===')
        print(f'  mean:   {statistics.mean(spinups):.1f}s')
        print(f'  median: {statistics.median(spinups):.1f}s')
        print(f'  min:    {min(spinups):.1f}s')
        print(f'  max:    {max(spinups):.1f}s')
        if len(spinups) > 1:
            print(f'  stdev:  {statistics.stdev(spinups):.1f}s')
    else:
        print(f'spinup: {spinups[0]:.1f}s')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--profile', choices=['permutation', 'experiment', 'spinup'],
        default='permutation',
        help='Profile mode (default: permutation)',
    )
    p.add_argument('--cloud', action='store_true',
                   help='Run experiment profiling on AWS (default: local)')
    p.add_argument('--n-jobs', type=int, default=-1,
                   help='Parallel jobs for local profiling (-1 = all cores)')
    p.add_argument('--min-voxels', type=int, default=1000)
    p.add_argument('--max-voxels', type=int, default=None)
    p.add_argument('--n-steps', type=int, default=10)
    p.add_argument('--n-perm', type=int, default=100)
    p.add_argument('--effect-llr', type=float, default=0.1)
    p.add_argument('--effect-perc', type=float, default=0.2)
    p.add_argument('--n', type=int, default=1,
                   help='Number of spinup measurements (default: 1)')
    p.add_argument('--quiet', action='store_true',
                   help='Only print summary (spinup mode)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.profile == 'experiment':
        main_experiment_profile(cloud=args.cloud, n_jobs=args.n_jobs)
    elif args.profile == 'spinup':
        main_spinup(n=args.n, quiet=args.quiet)
    else:
        main_permutation(args)


if __name__ == '__main__':
    main()
