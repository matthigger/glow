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
from matplotlib.ticker import FuncFormatter, LogLocator, ScalarFormatter
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
                         color=palette[0], label='Permutation min–max')
        ax.plot(voxels, perm_med, marker='o', markersize=6, linewidth=1.5,
                color=palette[0], zorder=3, label='Permutation median')

        synth = np.array([
            (r.get('synthesis_elapsed_sec') or 0) / 60
            for r in rows])[order]
        if synth.max() > 0:
            ax.plot(voxels, synth, marker='s', markersize=4, linewidth=1.0,
                    color=palette[1], zorder=3, label='Synthesis',
                    linestyle='--')

        wall = perm_hi + synth
        ax.plot(voxels, wall, marker='^', markersize=5, linewidth=1.0,
                color=palette[2], zorder=3, label='Wall Clock',
                linestyle=':')
    else:
        minutes = np.array([r['elapsed_min'] for r in rows])[order]
        ax.plot(voxels, minutes, marker='o', markersize=6, linewidth=1.5,
                color=palette[0], zorder=3)

    # -- Log-log fits --
    def _loglog_fit(x, y):
        lx, ly = np.log10(x), np.log10(y)
        mask = np.isfinite(lx) & np.isfinite(ly)
        m, b = np.polyfit(lx[mask], ly[mask], 1)
        return m, 10**b

    fit_lines = []
    if has_perm:
        m, a = _loglog_fit(voxels, perm_med)
        fit_lines.append(('Permutation', a, m))
        if synth.max() > 0:
            m, a = _loglog_fit(voxels, synth)
            fit_lines.append(('Synthesis', a, m))
        m, a = _loglog_fit(voxels, wall)
        fit_lines.append(('Wall Clock', a, m))
    else:
        m, a = _loglog_fit(voxels, minutes)
        fit_lines.append(('Runtime', a, m))

    txt_path = pdf_path.with_suffix('.txt')
    header = (
        'Log-log best fit:  minutes = a · voxels^m\n'
        '  a = intercept (predicted minutes at 1 voxel)\n'
        '  m = scaling exponent (slope in log-log space;\n'
        '      m=1 → linear, m=2 → quadratic, etc.)\n'
    )
    with open(txt_path, 'w') as f:
        f.write(header + '\n')
        print(header)
        for name, a, m in fit_lines:
            line = (f'{name:15s}  a = {a:.4e}  m = {m:.4f}'
                    f'  (e.g. {10_000:,} vox → {a * 10_000**m:.2f} min,'
                    f' {100_000:,} vox → {a * 100_000**m:.2f} min)')
            print(line)
            f.write(line + '\n')
        # Whole-brain cost/time estimates per fit
        cost_per_min = 0.02 / 60
        brain_mm3 = 1_300_000
        resolutions = {
            '2 mm³':    int(brain_mm3 / 2**3),
            '1.25 mm³ (HCP)': int(brain_mm3 / 1.25**3),
            '1 mm³':    brain_mm3,
        }
        sep = '-' * 72
        for name, fa, fm in fit_lines:
            table_hdr = (f'\nWhole-brain estimates ({name}):\n{sep}\n'
                         f'{"Resolution":>20s}  {"Voxels":>10s}'
                         f'  {"Minutes":>10s}  {"Cost ($)":>10s}\n'
                         f'{sep}')
            print(table_hdr)
            f.write(table_hdr + '\n')
            for res_label, nvox in resolutions.items():
                mins = fa * nvox**fm
                cost = mins * cost_per_min
                row = (f'{res_label:>20s}  {nvox:>10,}'
                       f'  {mins:>10.4g}  {cost:>10.4g}')
                print(row)
                f.write(row + '\n')
            print(sep)
            f.write(sep + '\n')

        # Per-permutation summary table (rows = metric, cols = resolution)
        n_perm = rows[0].get('n_perm', 1)
        res_labels = list(resolutions.keys())
        res_voxels = [resolutions[r] for r in res_labels]

        # use permutation fit for per-perm estimates
        perm_fit = [(n, a, m) for n, a, m in fit_lines if n == 'Permutation']
        if perm_fit:
            _, pa, pm = perm_fit[0]
        else:
            _, pa, pm = fit_lines[0]

        col_w = 18
        sep2 = '-' * (22 + col_w * len(res_labels))
        hdr = f'\nPer-permutation estimates (from Permutation fit, n_perm={n_perm}):\n{sep2}\n'
        hdr += f'{"":>20s}'
        for rl in res_labels:
            hdr += f'  {rl:>{col_w - 2}s}'
        hdr += f'\n{sep2}'
        print(hdr)
        f.write(hdr + '\n')

        for metric, fmt in [('min / perm', lambda mins, _nv: f'{mins / n_perm:#.4g}'),
                            ('$ / perm',   lambda mins, _nv: f'{mins / n_perm * cost_per_min:#.4g}')]:
            row = f'{metric:>20s}'
            for nvox in res_voxels:
                mins = pa * nvox**pm
                row += f'  {fmt(mins, nvox):>{col_w - 2}s}'
            print(row)
            f.write(row + '\n')
        print(sep2)
        f.write(sep2 + '\n')

    print(f'\nfit summary saved: {txt_path}')

    # -- Brain volume vertical lines --
    brain_mm3 = 1_300_000  # ~1.4 million mm³
    brain_voxels = {
        'brain at 2 mm³':    int(brain_mm3 / 2**3),
        'brain at 1.25 mm³ (HCP)': int(brain_mm3 / 1.25**3),
        'brain at 1 mm³':    brain_mm3,
    }

    _log_axes(ax)

    def _voxel_fmt(x, _pos):
        exp = np.log10(x)
        if abs(exp - round(exp)) < 0.01:
            exp = int(round(exp))
            if exp == 0:
                return '1 voxel'
            return f'$10^{exp}$ voxels'
        return ''

    ax.xaxis.set_major_formatter(FuncFormatter(_voxel_fmt))
    ax.set_xlabel('')
    ax.set_ylabel('Runtime (minutes)')
    ax.set_xlim(right=1_600_000)

    for label, nvox in brain_voxels.items():
        ax.axvline(nvox, color='black', linestyle='--', linewidth=1.0,
                   alpha=0.6, zorder=2)
        ax.text(nvox, ax.get_ylim()[0] * 1.3, f' {label}',
                rotation=90, va='bottom', ha='right', fontsize=7,
                color='black')

    # -- Secondary cost axis (right) --
    cost_per_min = 0.02 / 60  # $0.02/hour
    ax_cost = ax.secondary_yaxis('right',
                                  functions=(lambda y: y * cost_per_min,
                                             lambda c: c / cost_per_min))
    ax_cost.set_ylabel('Cost ($)')

    ax.legend(loc='upper left', fontsize=7)

    ax.set_title('GLOW: Single Permutation Runtime vs Voxel Count', pad=10)

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
