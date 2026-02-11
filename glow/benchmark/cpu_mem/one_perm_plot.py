#!/usr/bin/env python3
"""Plot one-permutation CPU and memory profiling results."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def read_csv(csv_path):
    with csv_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_cpu(ax, rows, color='tab:blue'):
    voxels = [float(row['num_vox']) / 1000.0 for row in rows]
    times = [float(row['elapsed_sec']) for row in rows]
    ax.plot(voxels, times, marker='o', color=color, label='cpu time')
    ax.set_ylabel('Compute time (sec)', color=color)
    ax.tick_params(axis='y', labelcolor=color)


def plot_mem(ax, rows, color='tab:orange'):
    voxels = [float(row['num_vox']) / 1000.0 for row in rows]
    peak_rss = [float(row['peak_rss_mb']) / 1024.0 for row in rows]
    ax.plot(voxels, peak_rss, marker='o', color=color, label='peak memory')
    ax.set_ylabel('Peak Memory (GB)', color=color)
    ax.tick_params(axis='y', labelcolor=color)


if __name__ == '__main__':
    sns.set_theme(context='paper', style='whitegrid')
    results_dir = Path.home() / '.local' / 'share' / 'glow' / 'results'
    cpu_csv = results_dir / 'one_perm_cpu.csv'
    mem_csv = results_dir / 'one_perm_mem.csv'
    combined_pdf = results_dir / 'one_perm_cpu_mem.pdf'

    cpu_rows = read_csv(cpu_csv) if cpu_csv.exists() else []
    mem_rows = read_csv(mem_csv) if mem_csv.exists() else []

    if not cpu_rows and not mem_rows:
        if cpu_csv.exists():
            print(f'cpu csv empty: {cpu_csv}')
        else:
            print(f'cpu csv not found: {cpu_csv}')
        if mem_csv.exists():
            print(f'mem csv empty: {mem_csv}')
        else:
            print(f'mem csv not found: {mem_csv}')
        raise SystemExit(0)

    fig, ax_left = plt.subplots(figsize=(6, 4))
    ax_right = ax_left.twinx()

    lines = []
    labels = []

    if cpu_rows:
        plot_cpu(ax_left, cpu_rows, color='tab:blue')
        lines.extend(ax_left.get_lines())
        labels.extend([line.get_label() for line in ax_left.get_lines()])

    if mem_rows:
        plot_mem(ax_right, mem_rows, color='tab:orange')
        lines.extend(ax_right.get_lines())
        labels.extend([line.get_label() for line in ax_right.get_lines()])

    ax_left.set_xlabel('Number of voxels (thousands)')
    ax_left.set_title('Running One Outermost Permutation')
    if cpu_rows:
        max_cpu = max(float(row['elapsed_sec']) for row in cpu_rows)
        ax_left.set_ylim(bottom=0, top=max_cpu * 1.05)
    else:
        ax_left.set_ylim(bottom=0)
    if mem_rows:
        max_mem = max(float(row['peak_rss_mb']) for row in mem_rows) / 1024.0
        ax_right.set_ylim(bottom=0, top=max_mem * 1.05)
    else:
        ax_right.set_ylim(bottom=0)

    ax_left.set_axisbelow(True)
    ax_left.grid(True, alpha=0.3, zorder=0)
    if lines:
        ax_left.legend(
            lines,
            labels,
            loc='upper left',
            framealpha=0.95,
            facecolor='white',
            edgecolor='white'
        )

    fig.tight_layout()
    fig.savefig(combined_pdf)
    print(f'combined plot saved: {combined_pdf}')
