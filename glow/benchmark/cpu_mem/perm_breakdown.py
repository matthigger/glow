#!/usr/bin/env python3
"""Profile cluster() vs iter_stat() breakdown within a single permutation.

Runs process_permutation with instrumented timing across a range of voxel
counts and produces a CSV + plot showing total runtime and the fraction
spent in cluster().
"""

import csv
import time
from pathlib import Path
import tempfile
import shutil

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import glow
from glow.benchmark.config import Config


def build_config(temp_folder):
    ana_kwargs_dict = {
        'GLOW': (glow.experiment.AnalysisGLOW, {
            'n_perm': 1,
            'n_perm_prune': 500,
            'min_size': 1,
            'alpha_prune': 0.05,
            'alpha_fwer': 0.05,
            'n_jobs_perm': 1
        })
    }

    config = Config(
        label='perm_breakdown',
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


def select_hcp_exp(config, target_vox, seed):
    mask_idx = config.exp_orig.mask_idx
    total_vox = int(np.sum(mask_idx > -1))
    if target_vox >= total_vox:
        raise ValueError('target_vox exceeds total voxels in HCP mask')

    extenter = glow.effect.ExtenterSphere(n_vox=target_vox, connected=True)
    mask = extenter(mask_idx=mask_idx, seed=seed, contiguous=True)
    count = int(mask.sum())
    exp = config.exp_orig.apply_mask(mask)
    return exp, count


def measure_breakdown(exp, ana_kwargs, perm_idx=0):
    """Run the phases of process_permutation with per-phase timing."""
    from glow.experiment.cluster import cluster
    from glow.experiment.mancova import get_llr

    get_stat = ana_kwargs.get('get_stat', get_llr)

    # phase 1: permute
    t0 = time.perf_counter()
    _exp = exp.permute(perm_idx)
    t1 = time.perf_counter()

    # phase 2: cluster
    children = cluster(exp=_exp)
    t2 = time.perf_counter()

    # phase 3: iter_stat + get_stat
    stat = []
    for reg_idx, size, e, h in glow.graph.iter_stat(
            exp=_exp, children=children, n_perm=None):
        stat_val = get_stat(e=e[:, :, 0], h=h[:, :, 0], n=size)
        stat.append(stat_val)
    t3 = time.perf_counter()

    permute_sec = t1 - t0
    cluster_sec = t2 - t1
    iter_stat_sec = t3 - t2
    total_sec = t3 - t0
    cluster_pct = 100.0 * cluster_sec / total_sec if total_sec > 0 else 0.0

    return {
        'permute_sec': permute_sec,
        'cluster_sec': cluster_sec,
        'iter_stat_sec': iter_stat_sec,
        'total_sec': total_sec,
        'cluster_pct': cluster_pct,
    }


def write_csv(samples, csv_path):
    fieldnames = [
        'num_vox', 'permute_sec', 'cluster_sec',
        'iter_stat_sec', 'total_sec', 'cluster_pct',
    ]
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow({
                'num_vox': s['num_vox'],
                'permute_sec': f'{s["permute_sec"]:.6f}',
                'cluster_sec': f'{s["cluster_sec"]:.6f}',
                'iter_stat_sec': f'{s["iter_stat_sec"]:.6f}',
                'total_sec': f'{s["total_sec"]:.6f}',
                'cluster_pct': f'{s["cluster_pct"]:.2f}',
            })


def plot_breakdown(samples, pdf_path):
    sns.set_theme(context='paper', style='whitegrid')

    voxels = [s['num_vox'] / 1000.0 for s in samples]
    total = [s['total_sec'] for s in samples]
    pct = [s['cluster_pct'] for s in samples]

    fig, ax_left = plt.subplots(figsize=(6, 4))
    ax_right = ax_left.twinx()

    line1, = ax_left.plot(
        voxels, total, marker='o', color='tab:blue', label='total runtime')
    line2, = ax_right.plot(
        voxels, pct, marker='s', color='tab:orange', label='% in cluster()')

    ax_left.set_xlabel('Number of voxels (thousands)')
    ax_left.set_ylabel('Total runtime (sec)', color='tab:blue')
    ax_left.tick_params(axis='y', labelcolor='tab:blue')
    ax_left.set_ylim(bottom=0)

    ax_right.set_ylabel('Time in cluster() (%)', color='tab:orange')
    ax_right.tick_params(axis='y', labelcolor='tab:orange')
    ax_right.set_ylim(0, 105)

    ax_left.legend(
        [line1, line2], [line1.get_label(), line2.get_label()],
        loc='center left', framealpha=0.95,
        facecolor='white', edgecolor='white')

    ax_left.set_title('Permutation Phase Breakdown')
    fig.tight_layout()
    fig.savefig(pdf_path)
    print(f'plot saved: {pdf_path}')
    plt.close(fig)


if __name__ == '__main__':
    temp_folder = Path(tempfile.mkdtemp(prefix='glow_perm_breakdown_'))
    try:
        config = build_config(temp_folder)
        config.prep_exp_orig()
    finally:
        shutil.rmtree(temp_folder, ignore_errors=True)

    voxel_targets = np.linspace(50, 50000, 10).round().astype(int)
    ana_kwargs = config.ana_kwargs_dict['GLOW'][1]

    results_dir = Path.home() / '.local' / 'share' / 'glow' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / 'perm_breakdown.csv'
    pdf_path = results_dir / 'perm_breakdown.pdf'

    print('=' * 60)
    print('Permutation Phase Breakdown Profiling')
    print('=' * 60)
    print(f'voxels: {", ".join(str(v) for v in voxel_targets)}')
    print(f'csv: {csv_path}')
    print()

    samples = []
    for target_vox in voxel_targets:
        exp, num_vox = select_hcp_exp(config, int(target_vox), seed=0)
        timing = measure_breakdown(exp, ana_kwargs)
        timing['num_vox'] = num_vox
        samples.append(timing)
        samples.sort(key=lambda s: s['num_vox'])
        write_csv(samples, csv_path)

        print(f'  voxels: {num_vox:,}')
        print(f'  total:       {timing["total_sec"]:.3f}s')
        print(f'    permute:   {timing["permute_sec"]:.3f}s')
        print(f'    cluster:   {timing["cluster_sec"]:.3f}s')
        print(f'    iter_stat: {timing["iter_stat_sec"]:.3f}s')
        print(f'  cluster %:   {timing["cluster_pct"]:.1f}%')
        print()

    print(f'csv saved: {csv_path}')
    plot_breakdown(samples, pdf_path)
