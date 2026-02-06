#!/usr/bin/env python3
"""Profile memory usage for local worker-style permutation runs."""

import csv
import gc
import multiprocessing
import threading
import time
from pathlib import Path
import tempfile
import shutil

import numpy as np
import psutil
import tracemalloc
from scipy.ndimage import label, generate_binary_structure
import cloudpickle as pickle

import glow
from glow.benchmark.config import Config
from glow.aws.worker import get_array_info, process_permutation


def build_config(temp_folder):
    ana_kwargs_dict = {
        'GLOW': (glow.experiment.AnalysisGLOW, {
            'n_perm': 1,
            'n_perm_adj': 1,
            'n_perm_prune': 1,
            'min_size': 1,
            'alpha_prune': 0.05,
            'alpha_fwer': 0.05,
            'n_jobs_perm': 1
        })
    }

    config = Config(
        label='mem_profile_hcp',
        source='hcp',
        run_fnc=None,
        ana_kwargs_dict=ana_kwargs_dict,
        wgn_shape=(1, 1, 1),
        wgn_a=2,
        wgn_b=2,
        wgn_num_img=1,
        exp_seed=0,
        n_seed=1,
        hotel_tr_all=np.array([0.1]),
        effect_perc=0.2,
        n_jobs=1,
        detail_save=False,
        error_save=False
    )

    config.folder = temp_folder
    config.cloud_config = None
    return config


class RssSampler(threading.Thread):
    def __init__(self, poll_interval=0.05):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self._process = psutil.Process()
        self._poll_interval = poll_interval
        self.max_rss_mb = 0.0

    def run(self):
        while not self._stop_event.is_set():
            rss_mb = self._process.memory_info().rss / (1024 * 1024)
            if rss_mb > self.max_rss_mb:
                self.max_rss_mb = rss_mb
            time.sleep(self._poll_interval)

    def stop(self):
        self._stop_event.set()


def run_perm_subprocess(payload_path):
    ctx = multiprocessing.get_context('spawn')
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=run_perm_worker, args=(payload_path, child_conn))
    proc.start()
    result = parent_conn.recv()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f'worker exited with code {proc.exitcode}')
    if 'error' in result:
        raise RuntimeError(result['error'])
    return result


def run_perm_worker(payload_path, conn):
    try:
        with open(payload_path, 'rb') as f:
            payload = pickle.load(f)
        exp = payload['exp']
        ana_kwargs = payload['ana_kwargs']

        process = psutil.Process()
        rss_before_mb = process.memory_info().rss / (1024 * 1024)
        sampler = RssSampler()
        sampler.start()

        tracemalloc.start()
        start_time = time.time()
        process_permutation(exp, ana_kwargs, perm_idx=0)
        elapsed_sec = time.time() - start_time
        _, tracemalloc_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        sampler.stop()
        sampler.join(timeout=1.0)
        rss_after_mb = process.memory_info().rss / (1024 * 1024)
        peak_rss_mb = max(rss_before_mb, rss_after_mb, sampler.max_rss_mb)
        peak_tracemalloc_mb = tracemalloc_peak / (1024 * 1024)

        conn.send({
            'rss_before_mb': rss_before_mb,
            'rss_after_mb': rss_after_mb,
            'peak_rss_mb': peak_rss_mb,
            'peak_tracemalloc_mb': peak_tracemalloc_mb,
            'elapsed_sec': elapsed_sec
        })
    except Exception as e:
        conn.send({'error': str(e)})
    finally:
        conn.close()


def select_hcp_exp(config, target_vox, seed):
    mask_idx = config.exp_orig.mask_idx
    total_vox = int(np.sum(mask_idx > -1))
    if target_vox >= total_vox:
        raise ValueError('target_vox exceeds total voxels in HCP mask')

    extenter = glow.effect.ExtenterSphere(n_vox=target_vox, connected=True)
    mask = extenter(mask_idx=mask_idx, seed=seed, contiguous=True)
    count = int(mask.sum())
    exp = config.exp_orig.apply_mask(mask)
    return exp, None, count, total_vox


def get_max_component_vox(config):
    mask = config.exp_orig.mask_idx > -1
    structure = generate_binary_structure(3, 1) if (mask.ndim == 3) else generate_binary_structure(2, 1)
    comp_idx, _ = label(mask, structure=structure)
    comp_sizes = np.bincount(comp_idx.ravel())
    comp_sizes[0] = 0
    return int(comp_sizes.max())


def measure_memory(target_vox):
    temp_folder = Path(tempfile.mkdtemp(prefix='glow_mem_profile_'))
    try:
        config = build_config(temp_folder)
        config.prep_exp_orig()
        ana_kwargs = config.ana_kwargs_dict['GLOW'][1]
        exp, _, num_vox, total_vox = select_hcp_exp(config, target_vox, seed=0)

        input_arrays = []
        input_arrays.extend(get_array_info(exp, 'exp'))
        input_arrays.extend(get_array_info(ana_kwargs, 'ana_kwargs'))
        total_input_mb = sum(arr['size_mb'] for arr in input_arrays)

        payload_dir = Path(tempfile.mkdtemp(prefix='glow_mem_payload_'))
        payload_path = payload_dir / 'payload.pkl'
        with open(payload_path, 'wb') as f:
            pickle.dump({'exp': exp, 'ana_kwargs': ana_kwargs}, f)

        result = run_perm_subprocess(payload_path)
        shutil.rmtree(payload_dir, ignore_errors=True)

        data_mb = exp.y.nbytes / (1024 * 1024)
        del exp
        gc.collect()
        return {
            'num_vox': num_vox,
            'data_mb': data_mb,
            'input_array_mb': total_input_mb,
            'rss_before_mb': result['rss_before_mb'],
            'rss_after_mb': result['rss_after_mb'],
            'peak_rss_mb': result['peak_rss_mb'],
            'peak_tracemalloc_mb': result['peak_tracemalloc_mb'],
            'elapsed_sec': result['elapsed_sec']
        }
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)


def write_csv(samples, csv_path):
    fieldnames = [
        'num_vox',
        'data_mb',
        'input_array_mb',
        'rss_before_mb',
        'rss_after_mb',
        'peak_rss_mb',
        'peak_tracemalloc_mb',
        'elapsed_sec'
    ]
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                'num_vox': sample['num_vox'],
                'data_mb': f'{sample["data_mb"]:.6f}',
                'input_array_mb': f'{sample["input_array_mb"]:.6f}',
                'rss_before_mb': f'{sample["rss_before_mb"]:.6f}',
                'rss_after_mb': f'{sample["rss_after_mb"]:.6f}',
                'peak_rss_mb': f'{sample["peak_rss_mb"]:.6f}',
                'peak_tracemalloc_mb': f'{sample["peak_tracemalloc_mb"]:.6f}',
                'elapsed_sec': f'{sample["elapsed_sec"]:.6f}'
            })


if __name__ == '__main__':
    temp_folder = Path(tempfile.mkdtemp(prefix='glow_mem_profile_'))
    try:
        config = build_config(temp_folder)
        config.prep_exp_orig()
        max_comp_vox = get_max_component_vox(config)
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)

    voxel_targets = np.linspace(1, max_comp_vox, 25)
    voxel_targets = np.unique(voxel_targets.round().astype(int))
    n_jobs = 1
    results_dir = Path.home() / '.local' / 'share' / 'glow' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / 'one_perm_mem.csv'

    print('=' * 60)
    print('Worker Permutation Memory Profiling')
    print('=' * 60)
    print(f'voxels: {", ".join(str(v) for v in voxel_targets)}')
    print(f'n_jobs: {n_jobs}')
    print(f'csv_path: {csv_path}')
    print()

    samples = []
    for target_vox in voxel_targets:
        sample = measure_memory(int(target_vox))
        samples.append(sample)
        samples.sort(key=lambda s: s['num_vox'])
        write_csv(samples, csv_path)
        print(f'  voxels: {sample["num_vox"]:,}')
        print(f'  data_mb: {sample["data_mb"]:.1f}')
        print(f'  input_array_mb: {sample["input_array_mb"]:.1f}')
        print(f'  rss_before_mb: {sample["rss_before_mb"]:.1f}')
        print(f'  rss_after_mb: {sample["rss_after_mb"]:.1f}')
        print(f'  peak_rss_mb: {sample["peak_rss_mb"]:.1f}')
        print(f'  peak_tracemalloc_mb: {sample["peak_tracemalloc_mb"]:.1f}')
        print(f'  elapsed_sec: {sample["elapsed_sec"]:.2f}')
        print()

    print(f'csv saved: {csv_path}')
    print('run one_perm_plot.py to generate plots')
