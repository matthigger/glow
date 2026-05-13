"""Plot alpha_FWER sweep results.

For each (label, cft_pval) slice in the sweep output, write a PDF
showing dice / sens / spec vs alpha_fwer.  Bold line is the mean over
seeds at each (effect_llr, alpha_fwer); shaded band is the ``--ci``
percentile interval.  One coloured line per effect_llr.

Usage::

    python -m glow.benchmark.alpha_sweep_plot <config_label> \
        [--ci 90] [--out-dir <path>]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glow.benchmark.alpha_sweep import SWEEP_SUFFIX
from glow.benchmark.file import load_update_all

_METRICS = ('dice', 'sens', 'spec')


def _slug(cft_pval):
    if cft_pval is None or (isinstance(cft_pval, float) and np.isnan(cft_pval)):
        return ''
    return f'_cft{cft_pval:.4g}'


def _plot_label_slice(df, label, cft_pval, ci, out_path):
    """One PDF: dice/sens/spec vs alpha_fwer, line per effect_llr."""
    df = df.copy()
    df['alpha_fwer'] = pd.to_numeric(df['alpha_fwer'], errors='coerce')
    df['effect_llr'] = pd.to_numeric(df['effect_llr'], errors='coerce')
    for m in _METRICS:
        df[m] = pd.to_numeric(df[m], errors='coerce')
    df = df.dropna(subset=['alpha_fwer', 'effect_llr'])
    if df.empty:
        print(f'  {label}{_slug(cft_pval)}: empty after numeric coercion — skip')
        return

    llrs = sorted(df['effect_llr'].unique())
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=min(llrs), vmax=max(llrs)) if len(llrs) > 1 \
        else plt.Normalize(vmin=llrs[0] - 1, vmax=llrs[0] + 1)
    color_for = {llr: cmap(norm(llr)) for llr in llrs}

    lower_q = (100 - ci) / 2
    upper_q = 100 - lower_q

    fig, axes = plt.subplots(1, len(_METRICS), figsize=(14, 4.5), sharex=True)
    title_suffix = f' (cft_pval={cft_pval:.4g})' if cft_pval else ''
    fig.suptitle(f'{label}{title_suffix}: post-hoc alpha_FWER sweep '
                 f'(mean ± {ci}% band over seeds)', fontsize=11)

    for j, metric in enumerate(_METRICS):
        ax = axes[j]
        for llr in llrs:
            sub = df[df['effect_llr'] == llr]
            g = (sub.groupby('alpha_fwer')[metric]
                 .agg(['mean',
                       lambda s: np.percentile(s, lower_q),
                       lambda s: np.percentile(s, upper_q)])
                 .reset_index()
                 .sort_values('alpha_fwer'))
            g.columns = ['alpha_fwer', 'mean', 'q_low', 'q_high']
            c = color_for[llr]
            ax.plot(g['alpha_fwer'], g['mean'],
                    color=c, lw=2.0, label=f'{llr:.3g}')
            ax.fill_between(g['alpha_fwer'], g['q_low'], g['q_high'],
                            color=c, alpha=0.18)

        ax.set_xscale('log')
        ax.set_xlabel(r'$\alpha_{FWER}$')
        ax.set_ylabel(metric)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.set_title(metric)

    # one shared colorbar for effect_llr, on the right
    if len(llrs) > 1:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label('effect_llr')
    else:
        axes[0].legend(title='effect_llr', frameon=False, fontsize=8)

    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


def plot_sweep(config_label, ci=90, out_dir=None):
    sweep_label = f'{config_label}{SWEEP_SUFFIX}'
    df, folder, _ = load_update_all(sweep_label, verbose=False)
    if df.empty:
        raise SystemExit(
            f'no sweep results for "{config_label}".  '
            f'Run: python -m glow.benchmark.alpha_sweep {config_label}')

    out_dir = folder if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # build the slice key: (label, cft_pval).  NaN cft_pval -> None.
    df = df.copy()
    df['_cft_key'] = df['cft_pval'].where(df['cft_pval'].notna(), None)

    for (label, cft_key), sub in df.groupby(['label', '_cft_key'],
                                            dropna=False):
        cft_val = None if cft_key is None else float(cft_key)
        out_path = out_dir / f'{label}{_slug(cft_val)}.pdf'
        _plot_label_slice(sub, label, cft_val, ci, out_path)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('config_label')
    p.add_argument('--ci', type=float, default=90,
                   help='percentile band width (default: %(default)s)')
    p.add_argument('--out-dir', type=str, default=None,
                   help='output directory (default: <sweep folder>)')
    return p.parse_args()


if __name__ == '__main__':
    from pathlib import Path
    args = _parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else None
    plot_sweep(args.config_label, ci=args.ci, out_dir=out_dir)
