"""Plot runtime benchmark results.

Two modes (selected by ``--mode``):

  permutation (default)
      Log-log plot of voxels vs minutes from the HCP permutation benchmark.

  experiment
      Per-analysis-type runtime scaling from the WGN profiling grid
      (``rtprof_*`` folders).  Shows runtime vs voxel count for each
      ``n_perm`` level, averaged over ``b`` and ``num_img``.

Usage::

    python -m glow.benchmark.runtime_plot                     # permutation
    python -m glow.benchmark.runtime_plot --mode experiment   # experiment
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogLocator, ScalarFormatter
from platformdirs import user_data_dir


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _log_axes(ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10),
                                          numticks=20))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10),
                                          numticks=20))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
    ax.grid(True, which='major', linewidth=2.0, color='white')
    ax.grid(True, which='minor', linewidth=1.0, color='white', alpha=0.7)


def _save(fig, pdf_path):
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(pdf_path.with_suffix('.png'), dpi=200)
    print(f'plot saved: {pdf_path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Permutation mode
# ---------------------------------------------------------------------------

def load_results(results_dir: Path):
    rows = []
    for p in sorted(results_dir.glob('out/*_result.json')):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def plot_runtime(rows, pdf_path: Path):
    sns.set_theme(context='paper', style='darkgrid', font_scale=1.1)

    has_perm = any(r.get('perm_elapsed_sec') for r in rows)

    voxels = np.array([r['num_voxels'] for r in rows])
    order = np.argsort(voxels)
    voxels = voxels[order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = sns.color_palette('deep')

    if has_perm:
        perm_med = np.array([
            float(np.median(r['perm_elapsed_sec'])) / 60
            if r.get('perm_elapsed_sec') else r.get('elapsed_min', 0)
            for r in rows])[order]
        perm_lo = np.array([
            float(np.min(r['perm_elapsed_sec'])) / 60
            if r.get('perm_elapsed_sec') else perm_med[i]
            for i, r in enumerate(rows)])[order]
        perm_hi = np.array([
            float(np.max(r['perm_elapsed_sec'])) / 60
            if r.get('perm_elapsed_sec') else perm_med[i]
            for i, r in enumerate(rows)])[order]

        ax.fill_between(voxels, perm_lo, perm_hi, alpha=0.18,
                         color=palette[0], label='perm min-max')
        ax.plot(voxels, perm_med, marker='o', markersize=6, linewidth=1.5,
                color=palette[0], zorder=3, label='perm median')

        synth = np.array([
            (r.get('synthesis_elapsed_sec') or 0) / 60
            for r in rows])[order]
        if synth.max() > 0:
            ax.plot(voxels, synth, marker='s', markersize=4, linewidth=1.0,
                    color=palette[1], zorder=3, label='synthesis',
                    linestyle='--')

        wall = perm_hi + synth
        ax.plot(voxels, wall, marker='^', markersize=5, linewidth=1.0,
                color=palette[2], zorder=3, label='wall clock',
                linestyle=':')
        ax.legend(loc='upper left', fontsize=8)
    else:
        minutes = np.array([r['elapsed_min'] for r in rows])[order]
        ax.plot(voxels, minutes, marker='o', markersize=6, linewidth=1.5,
                color=palette[0], zorder=3)

    _log_axes(ax)
    ax.set_xlabel('Number of voxels')
    ax.set_ylabel('Runtime (minutes)')

    n_perm = rows[0].get('n_perm')
    title = 'AnalysisGLOW Runtime vs Voxel Count (HCP)'
    if n_perm:
        title += f' [n_perm={n_perm}]'
    ax.set_title(title, pad=10)

    _save(fig, pdf_path)


def main_permutation():
    base = Path(user_data_dir('glow', 'glow_author'))
    results_dir = base / 'results' / 'runtime' / 'permutation'

    if not results_dir.exists():
        print(f'results directory not found: {results_dir}')
        raise SystemExit(1)

    rows = load_results(results_dir)
    if not rows:
        print(f'no result files in {results_dir / "out"}')
        raise SystemExit(1)

    print(f'loaded {len(rows)} results')
    pdf_path = results_dir / 'runtime.pdf'
    plot_runtime(rows, pdf_path)


# ---------------------------------------------------------------------------
# Experiment mode
# ---------------------------------------------------------------------------

_RTPROF_RE = re.compile(
    r'^rtprof_(?P<type>.+?)_(?P<vox>\d+)v_(?P<b>\d+)b_(?P<img>\d+)i_(?P<perm>\d+)p$'
)


def _collect_experiment_results():
    """Scan ``rtprof_*`` result folders and build a DataFrame."""
    base = Path(user_data_dir('glow', 'glow_author')) / 'results' / 'runtime' / 'experiment'
    rows = []
    for d in sorted(base.iterdir()):
        m = _RTPROF_RE.match(d.name)
        if m is None:
            continue
        out = d / 'out'
        if not out.exists():
            continue
        for p in out.glob('*_result.json'):
            with open(p) as f:
                r = json.load(f)
            rows.append({
                'analysis_type': str(r.get('label', m.group('type'))),
                'num_vox': int(r.get('vox_total', m.group('vox'))),
                'b': int(m.group('b')),
                'num_img': int(m.group('img')),
                'n_perm': int(m.group('perm')),
                'time_sec': float(r['time_sec']),
            })

    return pd.DataFrame(rows)


def plot_experiment(df, pdf_path: Path):
    """Runtime vs voxel count, one panel per analysis type, lines by n_perm."""
    sns.set_theme(context='paper', style='darkgrid', font_scale=1.1)

    analysis_types = sorted(df['analysis_type'].unique())
    n_types = len(analysis_types)

    fig, axes = plt.subplots(1, n_types, figsize=(5.5 * n_types, 4.5),
                             squeeze=False)
    axes = axes.ravel()
    palette = sns.color_palette('deep', n_colors=df['n_perm'].nunique())
    perm_vals = sorted(df['n_perm'].unique())
    perm_colors = dict(zip(perm_vals, palette))
    markers = ['o', 's', '^', 'D', 'v']

    for ax, atype in zip(axes, analysis_types):
        sub = df[df['analysis_type'] == atype]
        for i, n_perm in enumerate(perm_vals):
            sp = sub[sub['n_perm'] == n_perm]
            if sp.empty:
                continue
            agg = (sp.groupby('num_vox')['time_sec']
                   .agg(['mean', 'std']).reset_index())
            agg = agg.sort_values('num_vox')

            vox = agg['num_vox'].values
            mean_min = agg['mean'].values / 60.0
            std_min = agg['std'].values / 60.0

            mk = markers[i % len(markers)]
            ax.plot(vox, mean_min, marker=mk, markersize=5, linewidth=1.3,
                    color=perm_colors[n_perm], zorder=3,
                    label=f'n_perm={n_perm}')
            ax.fill_between(vox,
                            np.maximum(mean_min - std_min, 1e-4),
                            mean_min + std_min,
                            alpha=0.15, color=perm_colors[n_perm])

        _log_axes(ax)
        ax.set_xlabel('Number of voxels')
        ax.set_ylabel('Runtime (minutes)')
        ax.set_title(atype, pad=10)
        ax.legend(fontsize=7, loc='upper left')

    _save(fig, pdf_path)


def main_experiment():
    df = _collect_experiment_results()
    if df.empty:
        print('no experiment-mode results found (rtprof_* folders)')
        raise SystemExit(1)

    types_found = df['analysis_type'].unique()
    print(f'loaded {len(df)} measurements  '
          f'({", ".join(sorted(types_found))})')

    base = Path(user_data_dir('glow', 'glow_author')) / 'results' / 'runtime' / 'experiment'
    pdf_path = base / 'runtime_experiment.pdf'
    plot_experiment(df, pdf_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--mode', choices=['permutation', 'experiment'],
                   default='permutation',
                   help='Which results to plot (default: permutation)')
    args = p.parse_args()

    if args.mode == 'experiment':
        main_experiment()
    else:
        main_permutation()


if __name__ == '__main__':
    main()
