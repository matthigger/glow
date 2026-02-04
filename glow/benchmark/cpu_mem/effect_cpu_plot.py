#!/usr/bin/env python3
"""Plot effect CPU profiling results."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def read_csv(csv_path):
    with csv_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_effect(ax, rows):
    voxels = [float(row['num_vox']) / 1000.0 for row in rows]
    times = [float(row['elapsed_sec']) for row in rows]
    ax.plot(voxels, times, marker='o', color='tab:green')
    ax.set_xlabel('Number of voxels (thousands)')
    ax.set_ylabel('Compute time (sec)')
    ax.set_title('Time to Generate an Effect')
    ax.grid(True, alpha=0.3)


if __name__ == '__main__':
    sns.set_theme(context='paper', style='whitegrid')
    results_dir = Path.home() / '.local' / 'share' / 'glow' / 'results'
    effect_csv = results_dir / 'effect_cpu.csv'
    effect_pdf = results_dir / 'effect_cpu.pdf'

    effect_rows = read_csv(effect_csv) if effect_csv.exists() else []
    if effect_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_effect(ax, effect_rows)
        fig.tight_layout()
        fig.savefig(effect_pdf)
        print(f'effect plot saved: {effect_pdf}')
    elif effect_csv.exists():
        print(f'effect csv empty: {effect_csv}')
    else:
        print(f'effect csv not found: {effect_csv}')
