#!/usr/bin/env python3
"""Profile compute time for imposing effects on HCP data."""

import csv
import time
from pathlib import Path
import tempfile
import shutil

import numpy as np
from scipy.ndimage import label, generate_binary_structure

import glow
from glow.benchmark.config import Config


def build_config(temp_folder):
    config = Config(
        label='effect_cpu_hcp',
        source='hcp',
        run_fnc=None,
        ana_kwargs_dict=None,
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


def select_hcp_exp(config, target_vox, seed):
    mask_idx = config.exp_orig.mask_idx
    total_vox = int(np.sum(mask_idx > -1))
    if target_vox >= total_vox:
        raise ValueError('target_vox exceeds total voxels in HCP mask')

    extenter = glow.effect.ExtenterSphere(n_vox=target_vox, connected=True)
    mask = extenter(mask_idx=mask_idx, seed=seed, contiguous=True)
    count = int(mask.sum())
    exp = config.exp_orig.apply_mask(mask)
    return exp, count, total_vox


def get_max_component_vox(config):
    mask = config.exp_orig.mask_idx > -1
    structure = generate_binary_structure(3, 1) if (mask.ndim == 3) else generate_binary_structure(2, 1)
    comp_idx, _ = label(mask, structure=structure)
    comp_sizes = np.bincount(comp_idx.ravel())
    comp_sizes[0] = 0
    return int(comp_sizes.max())


def measure_effect_time(target_vox):
    temp_folder = Path(tempfile.mkdtemp(prefix='glow_effect_cpu_'))
    try:
        config = build_config(temp_folder)
        config.prep_exp_orig()
        exp, num_vox, total_vox = select_hcp_exp(config, target_vox, seed=0)

        start_time = time.perf_counter()
        exp, _ = exp.impose_effect(
            extenter=glow.effect.ExtenterMinVar(n=int(num_vox * config.effect_perc)),
            seed=0,
            hotel_tr=0.1
        )
        elapsed_sec = time.perf_counter() - start_time

        return {
            'num_vox': num_vox,
            'elapsed_sec': elapsed_sec
        }
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)


def write_csv(samples, csv_path):
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'num_vox',
                'elapsed_sec'
            ]
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                'num_vox': sample['num_vox'],
                'elapsed_sec': f'{sample["elapsed_sec"]:.6f}'
            })


if __name__ == '__main__':
    temp_folder = Path(tempfile.mkdtemp(prefix='glow_effect_cpu_'))
    try:
        config = build_config(temp_folder)
        config.prep_exp_orig()
        max_comp_vox = get_max_component_vox(config)
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)

    max_comp_vox = 10000

    voxel_targets = np.linspace(1, max_comp_vox, 25)
    voxel_targets = np.unique(voxel_targets.round().astype(int))
    n_jobs = 1
    results_dir = Path.home() / '.local' / 'share' / 'glow' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / 'effect_cpu.csv'

    print('=' * 60)
    print('Effect Imposition CPU Profiling')
    print('=' * 60)
    print(f'voxels: {", ".join(str(v) for v in voxel_targets)}')
    print(f'n_jobs: {n_jobs}')
    print(f'csv_path: {csv_path}')
    print()

    samples = []
    for target_vox in voxel_targets:
        sample = measure_effect_time(int(target_vox))
        samples.append(sample)
        samples.sort(key=lambda s: s['num_vox'])
        write_csv(samples, csv_path)
        print(f'  voxels: {sample["num_vox"]:,}')
        print(f'  elapsed_sec: {sample["elapsed_sec"]:.3f}')
        print()

    print(f'csv saved: {csv_path}')
    print('run effect_cpu_plot.py to generate plots')
