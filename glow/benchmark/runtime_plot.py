"""Plot runtime benchmark results (log-log: voxels vs minutes).

Reads result JSON files produced by ``runtime.py`` and produces a PDF.

Usage::

    python -m glow.benchmark.runtime_plot
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from platformdirs import user_data_dir


def load_results(results_dir: Path):
    rows = []
    for p in sorted(results_dir.glob('out/*_result.json')):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def plot_runtime(rows, pdf_path: Path):
    sns.set_theme(context='paper', style='darkgrid', font_scale=1.1)

    voxels = np.array([r['num_voxels'] for r in rows])
    minutes = np.array([r['elapsed_min'] for r in rows])

    order = np.argsort(voxels)
    voxels = voxels[order]
    minutes = minutes[order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(voxels, minutes, marker='o', markersize=6, linewidth=1.5,
            color=sns.color_palette('deep')[0], zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of voxels')
    ax.set_ylabel('Runtime (minutes)')
    ax.set_title('AnalysisGLOW Runtime vs Voxel Count (HCP)', pad=10)

    from matplotlib.ticker import LogLocator, ScalarFormatter
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
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(pdf_path.with_suffix('.png'), dpi=200)
    print(f'plot saved: {pdf_path}')
    plt.close(fig)


def main():
    base = Path(user_data_dir('glow', 'glow_author'))
    results_dir = base / 'results' / 'runtime_hcp'

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


if __name__ == '__main__':
    main()
