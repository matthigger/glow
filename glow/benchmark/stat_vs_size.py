"""Diagnostic: how do MANCOVA statistics vary with region size under H0?

For each permutation tree, every region gives a (size, stat) pair under the
null hypothesis (permuted data has no true effect).  This script:

1. Generates pure-WGN data and loads HCP data -- no effect.
2. Clusters each permutation and computes all 5 MANCOVA stats per region.
3. Fits several candidate regression models per stat and auto-selects the
   best by R².
4. Produces diagnostic plots: raw scatter with best fit, and residuals
   with binned empirical variance.

Usage::

    python -m glow.benchmark.stat_vs_size
    python -m glow.benchmark.stat_vs_size --n_perm 100
    python -m glow.benchmark.stat_vs_size --out /tmp/diag
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import glow.graph
from glow.experiment.cluster import cluster
from glow.experiment.exper import Experiment, ExperimentScaled
from glow.experiment.mancova import (
    get_hotel_tr,
    get_llr,
    get_neg_wilks,
    get_pillai,
    get_roys_root,
)

STAT_FUNCS = {
    'llr': get_llr,
    'hotel_tr': get_hotel_tr,
    'pillai': get_pillai,
    'neg_wilks': get_neg_wilks,
    'roys_root': get_roys_root,
}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect_all_stats(exp, children):
    """Compute all MANCOVA stats for every region in one tree walk."""
    records = []
    num_vox = exp.y.shape[2]

    for reg_idx, size, e, h in glow.graph.iter_stat(exp=exp, children=children):
        _e = e[:, :, 0]
        _h = h[:, :, 0]
        rec = {'reg_idx': reg_idx}
        for name, fn in STAT_FUNCS.items():
            try:
                rec[name] = fn(e=_e, h=_h, n=size)
            except np.linalg.LinAlgError:
                rec[name] = np.nan
        records.append(rec)

    sizes = glow.graph.node_sum(
        x=np.ones(num_vox, dtype=int), children=children)
    for rec in records:
        rec['size'] = int(sizes[rec['reg_idx']])

    return records


def collect_null_data(exp_base, n_perm, label='', verbose=True):
    """Run clustering + stat computation across permutations."""
    if not isinstance(exp_base, ExperimentScaled):
        exp_base = ExperimentScaled.from_exp(exp_base)

    b, num_img, num_vox = exp_base.y.shape
    if verbose:
        print(f'\n[{label}] b={b}, num_img={num_img}, '
              f'num_vox={num_vox}, n_perm={n_perm}')

    all_records = []
    t0 = time.time()
    for perm_idx in tqdm(range(n_perm), desc=f'{label} permutations',
                         disable=not verbose):
        _exp = exp_base.permute(perm_idx + 1)
        children = cluster(exp=_exp)
        recs = _collect_all_stats(_exp, children)
        for r in recs:
            r['perm_idx'] = perm_idx
        all_records.extend(recs)
        del _exp

    elapsed = time.time() - t0
    if verbose:
        print(f'  {len(all_records)} region-records in {elapsed:.1f}s')

    df = pd.DataFrame(all_records)
    df['source'] = label
    return df


# ---------------------------------------------------------------------------
# Experiment factories
# ---------------------------------------------------------------------------

def make_wgn_experiment():
    """Create a pure-WGN experiment matching benchmark params (no effect)."""
    return Experiment.from_gauss(
        b=2, num_img=100, shape=(16, 16, 16), seed=0, a=2)


def make_hcp_experiment():
    """Load HCP data with fa+md, radius-8 sphere mask (no effect)."""
    from glow.benchmark.hcp_data import get_hcp_path
    from glow.experiment.exper import ExperimentImageOnly
    import glow.effect

    path = get_hcp_path()
    img_glob_dict = {'fa': '*_fa.nii.gz', 'md': '*_md.nii.gz'}
    exp_img = ExperimentImageOnly.from_search(
        folder=path, sbj_regex=r'[\d]{6}', img_glob_dict=img_glob_dict)
    exp = exp_img.sample_x(a=2, seed=0, add_bias=True)

    extenter = glow.effect.ExtenterSphere(radius=8)
    mask = extenter(mask_idx=exp.mask_idx, seed=0, contiguous=True)
    exp = exp.apply_mask(mask)
    return exp


# ---------------------------------------------------------------------------
# Regression models
# ---------------------------------------------------------------------------

# Each model is defined by:
#   name:       human-readable label
#   requires_positive:  whether stat > 0 is needed
#   transform:  (size, stat) -> (X, y) for OLS
#   predict:    (beta, size_array) -> predicted stat in original space
#   residual:   (beta, size, stat) -> residual (in the fitted space)
#   equation:   (beta) -> human-readable string

MODELS = {
    'linear': dict(
        requires_positive=False,
        transform=lambda s, y: (
            np.column_stack([np.ones_like(s), s]), y),
        predict=lambda b, sz: b[0] + b[1] * sz,
        residual=lambda b, s, y: y - (b[0] + b[1] * s),
        equation=lambda b: f'stat = {b[0]:+.4f} {b[1]:+.6f}*size',
    ),
    'linear_no_intercept': dict(
        requires_positive=False,
        transform=lambda s, y: (s[:, np.newaxis], y),
        predict=lambda b, sz: b[0] * sz,
        residual=lambda b, s, y: y - b[0] * s,
        equation=lambda b: f'stat = {b[0]:.6f}*size',
    ),
    'power_law': dict(
        requires_positive=True,
        transform=lambda s, y: (
            np.column_stack([np.ones_like(s), np.log(s)]), np.log(y)),
        predict=lambda b, sz: np.exp(b[0] + b[1] * np.log(sz)),
        residual=lambda b, s, y: np.log(y) - (b[0] + b[1] * np.log(s)),
        equation=lambda b: f'ln(stat) = {b[0]:+.4f} {b[1]:+.4f}*ln(size)',
    ),
    'sqrt': dict(
        requires_positive=False,
        transform=lambda s, y: (
            np.column_stack([np.ones_like(s), np.sqrt(s)]), y),
        predict=lambda b, sz: b[0] + b[1] * np.sqrt(sz),
        residual=lambda b, s, y: y - (b[0] + b[1] * np.sqrt(s)),
        equation=lambda b: f'stat = {b[0]:+.4f} {b[1]:+.4f}*sqrt(size)',
    ),
    'log_linear': dict(
        requires_positive=True,
        transform=lambda s, y: (
            np.column_stack([np.ones_like(s), s]), np.log(y)),
        predict=lambda b, sz: np.exp(b[0] + b[1] * sz),
        residual=lambda b, s, y: np.log(y) - (b[0] + b[1] * s),
        equation=lambda b: f'ln(stat) = {b[0]:+.4f} {b[1]:+.6f}*size',
    ),
}


def _fit_model(model_name, s, y):
    """Fit one model and return (beta, R², residuals).

    Returns None if the model is inapplicable (e.g. requires positive
    stat values but some are <= 0).
    """
    spec = MODELS[model_name]
    valid = np.isfinite(y) & (s > 0) & np.isfinite(s)
    if spec['requires_positive']:
        valid &= (y > 0)
    if valid.sum() < 10:
        return None

    s_v, y_v = s[valid], y[valid]
    X, y_t = spec['transform'](s_v, y_v)
    beta, _, _, _ = np.linalg.lstsq(X, y_t, rcond=None)

    pred = X @ beta
    ss_res = np.sum((y_t - pred) ** 2)
    ss_tot = np.sum((y_t - y_t.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    resid_all = np.full_like(y, np.nan)
    resid_all[valid] = spec['residual'](beta, s_v, y_v)

    return dict(beta=beta, r2=r2, resid=resid_all, valid=valid,
                n_valid=int(valid.sum()))


def fit_all_models(s, y):
    """Fit every candidate model, return dict of results + best model name."""
    results = {}
    for name in MODELS:
        res = _fit_model(name, s, y)
        if res is not None:
            results[name] = res
    if not results:
        return {}, None
    best = max(results, key=lambda k: results[k]['r2'])
    return results, best


# ---------------------------------------------------------------------------
# Binned statistics
# ---------------------------------------------------------------------------

N_BINS = 15


def compute_binned_stats(sizes, values, n_bins=N_BINS):
    """Compute empirical mean and std in log-spaced size bins."""
    valid = np.isfinite(values) & (sizes > 0) & np.isfinite(sizes)
    s, v = sizes[valid], values[valid]

    if len(s) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    edges = np.logspace(np.log10(s.min()), np.log10(s.max()), n_bins + 1)
    centers, means, stds, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (s >= lo) & (s < hi)
        n = mask.sum()
        if n < 5:
            continue
        centers.append(np.sqrt(lo * hi))
        means.append(v[mask].mean())
        stds.append(v[mask].std())
        counts.append(n)

    return (np.array(centers), np.array(means),
            np.array(stds), np.array(counts))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_MODEL_COLORS = {
    'linear': 'orange',
    'linear_no_intercept': 'green',
    'power_law': 'purple',
    'sqrt': 'brown',
    'log_linear': 'teal',
}


def plot_stat(df, stat_col, source_label, ax_scatter, ax_resid):
    """Plot one stat for one source: scatter with all fits + best residuals.

    Top panel shows scatter with every applicable model's fit line.
    Bottom panel shows residuals for the best model only.
    """
    s = df['size'].values.astype(float)
    y = df[stat_col].values.astype(float)

    results, best_name = fit_all_models(s, y)
    if best_name is None:
        ax_scatter.set_title(f'{source_label}: {stat_col} (insufficient data)')
        return

    # subsample for scatter readability
    best = results[best_name]
    valid = best['valid']
    s_v, y_v = s[valid], y[valid]
    rng = np.random.default_rng(42)
    n_plot = min(50_000, len(s_v))
    idx = rng.choice(len(s_v), n_plot, replace=False) if len(s_v) > n_plot \
        else np.arange(len(s_v))

    # --- top panel: scatter + all model fit lines ---
    ax_scatter.scatter(s_v[idx], y_v[idx], s=1, alpha=0.05, rasterized=True,
                       color='steelblue')

    bc, bm, bs, _ = compute_binned_stats(s_v, y_v, n_bins=N_BINS)
    if len(bc) > 0:
        ax_scatter.fill_between(bc, bm - bs, bm + bs,
                                color='red', alpha=0.12)
        ax_scatter.plot(bc, bm, 'o-', color='red', markersize=3, linewidth=1,
                        label='binned mean')

    use_log_axes = all(
        results[n].get('r2', 0) < results.get('power_law', {}).get('r2', -1)
        or n == 'power_law'
        for n in results if MODELS[n]['requires_positive']
    ) and 'power_law' in results and np.all(y_v > 0)

    sz_line_lin = np.linspace(max(1, s_v.min()), s_v.max(), 300)
    sz_line_log = np.logspace(np.log10(max(1, s_v.min())),
                              np.log10(s_v.max()), 300)

    for name, res in sorted(results.items(), key=lambda kv: kv[1]['r2']):
        spec = MODELS[name]
        sz = sz_line_log if use_log_axes else sz_line_lin
        y_fit = spec['predict'](res['beta'], sz)
        is_best = (name == best_name)
        lw = 2.5 if is_best else 1.2
        ls = '-' if is_best else '--'
        color = _MODEL_COLORS.get(name, 'gray')
        label = f'{name} R²={res["r2"]:.4f}'
        if is_best:
            label += ' *'
        ax_scatter.plot(sz, y_fit, color=color, linewidth=lw, linestyle=ls,
                        label=label, alpha=0.9 if is_best else 0.6)

    if use_log_axes:
        ax_scatter.set_xscale('log')
        ax_scatter.set_yscale('log')

    ax_scatter.set_xlabel('region size (voxels)')
    ax_scatter.set_ylabel(stat_col)
    ax_scatter.set_title(f'{source_label}: {stat_col}')
    ax_scatter.legend(fontsize=6, loc='upper left')

    # --- bottom panel: residuals of best model ---
    resid = best['resid']
    resid_v = resid[valid]

    ax_resid.scatter(s_v[idx], resid_v[idx], s=1, alpha=0.05, rasterized=True,
                     color='steelblue')

    bc_r, bm_r, bs_r, bn_r = compute_binned_stats(s_v, resid_v, n_bins=N_BINS)
    if len(bc_r) > 0:
        ax_resid.fill_between(bc_r, bm_r - bs_r, bm_r + bs_r,
                              color='red', alpha=0.15,
                              label='binned mean +/- 1 std')
        ax_resid.plot(bc_r, bm_r, 'o-', color='red', markersize=3,
                      linewidth=1, label='binned mean')
        for c, n in zip(bc_r, bn_r):
            ax_resid.annotate(f'n={int(n)}', xy=(c, bm_r[list(bc_r).index(c)]),
                              fontsize=5, color='gray', ha='center',
                              xytext=(0, 8), textcoords='offset points')

    ax_resid.axhline(0, color='black', linewidth=0.5)
    ax_resid.set_xscale('log')
    ax_resid.set_xlabel('region size (voxels)')
    is_log_resid = MODELS[best_name]['requires_positive']
    ax_resid.set_ylabel('log-space residual' if is_log_resid else 'residual')
    ax_resid.set_title(
        f'{source_label}: {stat_col} residuals (best={best_name})')
    ax_resid.legend(fontsize=7, loc='upper right')


def make_diagnostic_plots(df_dict, out_dir):
    """Produce all diagnostic figures and summary table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stat_names = list(STAT_FUNCS.keys())
    sources = list(df_dict.keys())
    n_src = len(sources)

    for stat_col in stat_names:
        fig, axes = plt.subplots(2, n_src, figsize=(7 * n_src, 10),
                                 squeeze=False)
        for j, src in enumerate(sources):
            df = df_dict[src]
            plot_stat(df, stat_col, src,
                      ax_scatter=axes[0, j], ax_resid=axes[1, j])
        fig.tight_layout()
        fig.savefig(out_dir / f'stat_vs_size_{stat_col}.png', dpi=150)
        plt.close(fig)
        print(f'  saved stat_vs_size_{stat_col}.png')

    # --- summary table: all models x all stats ---
    rows = []
    for src in sources:
        df = df_dict[src]
        for stat_col in stat_names:
            s = df['size'].values.astype(float)
            y = df[stat_col].values.astype(float)
            results, best_name = fit_all_models(s, y)
            for model_name, res in results.items():
                resid = res['resid']
                valid = res['valid']
                resid_v = resid[valid]
                bc, bm, bs, bn = compute_binned_stats(
                    s[valid], resid_v, n_bins=N_BINS)
                rows.append({
                    'source': src,
                    'stat': stat_col,
                    'model': model_name,
                    'equation': MODELS[model_name]['equation'](res['beta']),
                    'R2': res['r2'],
                    'best': model_name == best_name,
                    'n': res['n_valid'],
                    'resid_std': resid_v.std(),
                    'resid_std_min_bin': bs.min() if len(bs) else np.nan,
                    'resid_std_max_bin': bs.max() if len(bs) else np.nan,
                    'resid_std_ratio': (bs.max() / bs.min()
                                        if len(bs) and bs.min() > 0
                                        else np.nan),
                })

    summary = pd.DataFrame(rows)
    summary_path = out_dir / 'fit_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f'\n  saved {summary_path}')

    # pretty-print: best model per (source, stat)
    best_df = summary[summary['best']].copy()
    print('\n' + '=' * 100)
    print('BEST MODEL PER STAT')
    print('=' * 100)
    for _, row in best_df.iterrows():
        print(f"  {row['source']:4s}  {row['stat']:12s}  "
              f"best={row['model']:22s}  "
              f"R²={row['R2']:.4f}  "
              f"σ={row['resid_std']:.4f}  "
              f"σ_bin=[{row['resid_std_min_bin']:.4f}, "
              f"{row['resid_std_max_bin']:.4f}]  "
              f"ratio={row['resid_std_ratio']:.2f}")

    # runner-up comparison
    print('\n' + '-' * 100)
    print('ALL MODELS (sorted by R² within each source+stat)')
    print('-' * 100)
    for (src, stat), grp in summary.groupby(['source', 'stat']):
        grp_sorted = grp.sort_values('R2', ascending=False)
        for i, (_, row) in enumerate(grp_sorted.iterrows()):
            marker = '>>>' if row['best'] else '   '
            print(f"  {marker} {row['source']:4s}  {row['stat']:12s}  "
                  f"{row['model']:22s}  R²={row['R2']:.4f}  "
                  f"{row['equation']}")
        print()

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic: MANCOVA stat vs region size under H0')
    parser.add_argument('--n_perm', type=int, default=50,
                        help='permutations per data source (default: 50)')
    parser.add_argument('--out', type=str, default=None,
                        help='output directory for plots (default: auto)')
    args = parser.parse_args()

    if args.out is None:
        from glow.benchmark.config import path_result
        out_dir = path_result / 'stat_vs_size'
    else:
        out_dir = Path(args.out)

    df_dict = {}

    print('Preparing WGN experiment ...')
    exp_wgn = make_wgn_experiment()
    df_wgn = collect_null_data(exp_wgn, n_perm=args.n_perm, label='WGN')
    df_dict['WGN'] = df_wgn

    print('\nPreparing HCP experiment ...')
    try:
        exp_hcp = make_hcp_experiment()
        df_hcp = collect_null_data(exp_hcp, n_perm=args.n_perm, label='HCP')
        df_dict['HCP'] = df_hcp
    except Exception as exc:
        print(f'  HCP loading failed: {exc}')
        print('  (continuing with WGN only)')

    print(f'\nGenerating diagnostic plots in {out_dir} ...')
    make_diagnostic_plots(df_dict, out_dir)
    print('Done.')


if __name__ == '__main__':
    main()
