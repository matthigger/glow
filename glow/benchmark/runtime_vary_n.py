"""Benchmark runtime vs number of subjects at fixed voxel count.

Runs AnalysisGLOW on WGN data with 10k voxels across log-spaced
subject counts. Supports local and cloud (AWS Batch) execution.

Usage::

    python -m glow.benchmark.runtime_vary_n              # local
    python -m glow.benchmark.runtime_vary_n --cloud      # AWS Batch
    python -m glow.benchmark.runtime_vary_n --plot-only  # plot existing results
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogLocator, ScalarFormatter
from platformdirs import user_data_dir

import glow
from glow.benchmark.config import Config
from glow.benchmark.runner import RunAna

LABEL = 'runtime_vary_n'
CROP_N_VOX = 10_000
N_PERM = 100
N_REPEATS = 3
NUM_IMG_VALUES = np.unique(
    np.logspace(np.log10(10), np.log10(1000), 10).astype(int))

_wgn_side = math.ceil(CROP_N_VOX ** (1 / 3))

ana_kwargs_dict = {
    'GLOW': (glow.analysis.AnalysisGLOW,
             dict(n_perm_fwer=N_PERM, min_vox=1, alpha_fwer=0.05)),
}


def _make_config(cloud_config=None):
    return Config(
        label=LABEL,
        source='wgn',
        runner=RunAna(ana_kwargs_dict),
        n_seed=N_REPEATS,
        effect_llr_all=np.array([0.0]),
        effect_perc=0.1,
        crop_n_vox=CROP_N_VOX,
        wgn_shape=(_wgn_side, _wgn_side, _wgn_side),
        wgn_a=2,
        wgn_b=2,
        wgn_num_img=100,  # overridden by iter_params
        n_jobs=1,
        detail_save=False,
        error_save=False,
        x_param='wgn_num_img',
        iter_params={
            'wgn_num_img': NUM_IMG_VALUES.tolist(),
            'seed': list(range(N_REPEATS)),
        },
        fixed_params={'effect_llr': 0.0},
        cloud_config=cloud_config,
    )


def _load_results():
    """Load result JSONs from the config's output folder."""
    from glow.benchmark.file import load_update_all

    df, _folder, _n_new = load_update_all(LABEL, verbose=False)
    if df.empty:
        return None

    # filter to GLOW results with time_sec
    df = df[df['label'] == 'GLOW'].copy()
    if 'time_sec' not in df.columns or df['time_sec'].isna().all():
        return None

    # aggregate over seeds
    agg = (df.groupby('wgn_num_img')['time_sec']
           .agg(['median', 'min', 'max', 'count']).reset_index())
    agg = agg.sort_values('wgn_num_img')
    return agg


def plot_results(pdf_path):
    agg = _load_results()
    if agg is None:
        print('no results found to plot')
        raise SystemExit(1)

    print(f'loaded results for {len(agg)} num_img values')

    sns.set_theme(context='paper', style='darkgrid', font_scale=1.1)

    num_img = agg['wgn_num_img'].values
    med_min = agg['median'].values / 60
    lo_min = agg['min'].values / 60
    hi_min = agg['max'].values / 60

    fig, ax = plt.subplots(figsize=(6, 4))
    palette = sns.color_palette('deep')

    ax.fill_between(num_img, lo_min, hi_min, alpha=0.18, color=palette[0])
    ax.plot(num_img, med_min, marker='o', markersize=6, linewidth=1.5,
            color=palette[0], zorder=3, label='median')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
    ax.grid(True, which='major', linewidth=2.0, color='white')
    ax.grid(True, which='minor', linewidth=1.0, color='white', alpha=0.7)

    ax.set_xlabel('Number of subjects')
    ax.set_ylabel('Runtime (minutes)')
    ax.set_title(f'GLOW Runtime vs Subjects ({CROP_N_VOX:,} voxels, '
                 f'{N_PERM} perms)', pad=10)

    # log-log fit
    log_x = np.log10(num_img.astype(float))
    log_y = np.log10(med_min)
    mask = np.isfinite(log_x) & np.isfinite(log_y)
    m, b = np.polyfit(log_x[mask], log_y[mask], 1)
    fit_x = np.logspace(np.log10(num_img.min()), np.log10(num_img.max()), 100)
    ax.plot(fit_x, 10**b * fit_x**m, 'k-', linewidth=1.2, zorder=4,
            label=f'fit: m={m:.2f}')
    ax.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(pdf_path.with_suffix('.png'), dpi=200)
    print(f'plot saved: {pdf_path}')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cloud', action='store_true',
                   help='Submit to AWS Batch instead of running locally')
    p.add_argument('--plot-only', action='store_true',
                   help='Skip running, just plot existing results')
    args = p.parse_args()

    out_dir = Path(user_data_dir('glow', 'glow_author')) / 'results' / 'runtime'
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / 'runtime_vary_n.pdf'

    if not args.plot_only:
        cloud_config = None
        if args.cloud:
            from glow.benchmark.paper import load_cloud_config
            cloud_config = load_cloud_config()

        config = _make_config(cloud_config=cloud_config)
        config.run_all(verbose=True)

    plot_results(pdf_path)


if __name__ == '__main__':
    main()
