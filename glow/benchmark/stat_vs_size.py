"""Diagnostic: how do MANCOVA statistics vary with region size under H0?

For each permutation tree, every region gives a (size, stat) pair under the
null hypothesis (permuted data has no true effect).  This script:

1. Generates pure-WGN data and loads HCP data -- no effect.
2. Clusters each permutation and computes requested MANCOVA stats per region.
3. Fits several candidate regression models per stat and auto-selects the
   best by R².
4. Produces diagnostic plots: raw scatter with best fit, and residuals
   with binned empirical variance.

Usage::

    python -m glow.benchmark.stat_vs_size
    python -m glow.benchmark.stat_vs_size --n_perm 100
    python -m glow.benchmark.stat_vs_size --stats llr hotel_tr pillai
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
from glow.experiment.mancova import stat_dict

DEFAULT_STATS = list(stat_dict.keys())


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect_all_stats(exp, children, stat_funcs):
    """Compute requested MANCOVA stats for every region in one tree walk."""
    records = []
    num_vox = exp.y.shape[2]

    for reg_idx, size, e, h in glow.graph.iter_stat(exp=exp, children=children):
        _e = e[:, :, 0]
        _h = h[:, :, 0]
        rec = {'reg_idx': reg_idx}
        for name, fn in stat_funcs.items():
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


def collect_null_data(exp_base, n_perm, stat_funcs, label='', verbose=True):
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
        recs = _collect_all_stats(_exp, children, stat_funcs)
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

NUM_VOX = 10_000


def _crop_to_n_vox(exp, n_vox=NUM_VOX, seed=0):
    """Crop an experiment to exactly n_vox contiguous voxels."""
    import glow.effect
    if exp.y.shape[2] <= n_vox:
        return exp
    extenter = glow.effect.ExtenterSphere(n_vox=n_vox)
    mask = extenter(mask_idx=exp.mask_idx, seed=seed, contiguous=True)
    return exp.apply_mask(mask)


def make_wgn_experiment():
    """Create a pure-WGN experiment matching benchmark params (no effect)."""
    exp = Experiment.from_gauss(
        b=2, num_img=100, shape=(50, 50, 50), seed=0, a=2)
    return _crop_to_n_vox(exp)


def make_hcp_experiment():
    """Load HCP data with fa+md, cropped to NUM_VOX (no effect)."""
    from glow.benchmark.hcp_data import get_hcp_path
    from glow.experiment.exper import ExperimentImageOnly

    path = get_hcp_path()
    img_glob_dict = {'fa': '*_fa.nii.gz', 'md': '*_md.nii.gz'}
    exp_img = ExperimentImageOnly.from_search(
        folder=path, sbj_regex=r'[\d]{6}', img_glob_dict=img_glob_dict)
    exp = exp_img.sample_x(a=2, seed=0, add_bias=True)
    return _crop_to_n_vox(exp)


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
    'log_size': dict(
        requires_positive=False,
        transform=lambda s, y: (
            np.column_stack([np.ones_like(s), np.log(s)]), y),
        predict=lambda b, sz: b[0] + b[1] * np.log(sz),
        residual=lambda b, s, y: y - (b[0] + b[1] * np.log(s)),
        equation=lambda b: f'stat = {b[0]:+.4f} {b[1]:+.4f}*ln(size)',
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
    """Fit one size-weighted model and return (beta, R², residuals).

    Each observation is weighted by its region size so that larger regions
    receive proportionally more influence on the fit.

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

    # WLS: weight each observation by region size
    w = np.sqrt(s_v)[:, None]
    Xw = X * w
    yw = y_t * w.ravel()
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

    pred = X @ beta
    wt = s_v
    ss_res = np.sum(wt * (y_t - pred) ** 2)
    y_mean = np.average(y_t, weights=wt)
    ss_tot = np.sum(wt * (y_t - y_mean) ** 2)
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
    'linear': '#e69f00',
    'power_law': '#9467bd',
    'sqrt': '#8c564b',
    'log_size': '#1f77b4',
    'log_linear': '#17becf',
}

_STAT_LABELS = {
    'llr': 'Log-Likelihood Ratio',
    'neg_wilks': r"$-$Wilks' $\Lambda$",
    'hotel_tr': 'Hotelling Trace',
    'pillai': "Pillai's Trace",
    'roys_root': "Roy's Largest Root",
}

_MODEL_LABELS = {
    'linear': 'Linear',
    'power_law': 'Power law',
    'sqrt': 'Square root',
    'log_size': 'Logarithmic',
    'log_linear': 'Log-linear',
}

_MODEL_PARAMETRIC = {
    'linear': r'$\mathrm{LLR} = a_0 + a_1 |r|$',
    'power_law': r'$\ln\mathrm{LLR} = a_0 + a_1 \ln|r|$',
    'sqrt': r'$\mathrm{LLR} = a_0 + a_1 \sqrt{|r|}$',
    'log_size': r'$\mathrm{LLR} = a_0 + a_1 \ln|r|$',
    'log_linear': r'$\ln\mathrm{LLR} = a_0 + a_1 |r|$',
}

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def make_diagnostic_plots(df_dict, out_dir):
    """Produce 2x2 per-model figures, summary table, and LaTeX table.

    For each model that was fit, a 2x2 PDF is created:
      columns = data sources (WGN, HCP, ...),
      rows    = fit (top) / residuals (bottom).
    X-axis is always log-scaled.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stat_names = [c for c in list(stat_dict.keys())
                  if any(c in df.columns for df in df_dict.values())]
    sources = list(df_dict.keys())

    # --- 2x2 figures: one per (stat, model) ---
    for stat_col in stat_names:
        all_results = {}
        for src in sources:
            df = df_dict[src]
            s = df['size'].values.astype(float)
            y = df[stat_col].values.astype(float)
            results, best_name = fit_all_models(s, y)
            all_results[src] = (results, best_name, s, y)

        model_names = sorted({m for r, _, _, _ in all_results.values()
                              for m in r})
        for model_name in model_names:
            _make_2x2_figure(df_dict, sources, stat_col, model_name,
                             all_results, out_dir)

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

    # --- LaTeX table ---
    latex_path = out_dir / 'fit_summary.tex'
    _write_latex_table(summary, latex_path)
    print(f'  saved {latex_path}')

    # --- best model JSON (average R² across sources) ---
    import json
    from glow.experiment.analysis import _BEST_MODELS_PATH

    best_by_stat = {}
    for stat_col in stat_names:
        stat_rows = summary[summary['stat'] == stat_col]
        avg_r2 = stat_rows.groupby('model')['R2'].mean()
        best_by_stat[stat_col] = avg_r2.idxmax()

    json_path = out_dir / 'best_models.json'
    json_path.write_text(json.dumps(best_by_stat, indent=2) + '\n')
    print(f'  saved {json_path}')

    _update_repo_json(best_by_stat, _BEST_MODELS_PATH)

    print('\n' + '=' * 60)
    print('BEST MODEL PER STAT (mean R² across sources)')
    print('=' * 60)
    for stat_col in stat_names:
        stat_rows = summary[summary['stat'] == stat_col]
        avg_r2 = stat_rows.groupby('model')['R2'].mean()
        best_model = avg_r2.idxmax()
        best_r2 = avg_r2.max()
        model_label = _MODEL_LABELS.get(best_model, best_model)
        print(f'  {stat_col:12s}  {model_label:15s}  mean R²={best_r2:.4f}')

    return summary


def _update_repo_json(best_by_stat, repo_path):
    """Write best_models.json into the repo, prompting if it already exists."""
    import json
    if repo_path.exists():
        answer = input(f'\n  {repo_path} already exists. Overwrite? [y/N] ')
        if answer.strip().lower() != 'y':
            print('  skipped repo update')
            return
    repo_path.write_text(json.dumps(best_by_stat, indent=2) + '\n')
    print(f'  saved {repo_path}')


def _equation_to_latex(eq_str):
    """Convert plain-text equation string to LaTeX math."""
    s = eq_str
    s = s.replace('sqrt(size)', r'\sqrt{|r|}')
    s = s.replace('ln(stat)', r'\ln\mathrm{LLR}')
    s = s.replace('ln(size)', r'\ln|r|')
    s = s.replace('*', r'\,')
    s = s.replace('size', '|r|')
    s = s.replace('stat', r'\mathrm{LLR}')
    return s


def _write_latex_table(summary, path):
    """Write one LaTeX table per source, all in the same file."""
    all_lines = []
    sources = summary['source'].unique()
    for src in sorted(sources):
        src_df = summary[summary['source'] == src].copy()
        label_suffix = src.lower().replace(' ', '_')
        lines = [
            r'\begin{table}[H]',
            r'\centering',
            rf'\caption{{Size-weighted regression models for {src}'
            r' under $H_0$, sorted by $R^2$.}',
            rf'\label{{tab:stat_vs_size_{label_suffix}}}',
            r'\begin{tabular}{lcc}',
            r'\toprule',
            r'Model & $R^2$ & Fitted equation \\',
            r'\midrule',
        ]
        for stat_col in src_df['stat'].unique():
            grp = src_df[src_df['stat'] == stat_col]
            grp_sorted = grp.sort_values('R2', ascending=False)
            for _, row in grp_sorted.iterrows():
                model_tex = _MODEL_LABELS.get(row['model'], row['model'])
                eq_tex = _equation_to_latex(row['equation'])
                r2_str = f"{row['R2']:.4f}"
                if row['best']:
                    model_tex = r'\textbf{' + model_tex + '}'
                    r2_str = r'\textbf{' + r2_str + '}'
                lines.append(
                    f"  {model_tex} & {r2_str} & ${eq_tex}$ \\\\")
        lines += [
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ]
        all_lines.extend(lines)
        all_lines.append('')

    path.write_text('\n'.join(all_lines) + '\n')


def _make_2x2_figure(df_dict, sources, stat_col, model_name,
                     all_results, out_dir):
    """2x2 figure for one model.  Columns = sources, rows = fit / residuals.

    X-axis is log-scaled throughout.
    """
    stat_label = _STAT_LABELS.get(stat_col, stat_col)
    model_label = _MODEL_LABELS.get(model_name, model_name)
    spec = MODELS[model_name]

    n_src = min(len(sources), 2)
    fig, axes = plt.subplots(2, n_src, figsize=(5 * n_src, 8),
                             squeeze=False, constrained_layout=True)

    for j, src in enumerate(sources[:n_src]):
        results, best_name, s, y = all_results[src]
        if model_name not in results:
            continue
        res = results[model_name]

        valid = res['valid']
        s_v, y_v = s[valid], y[valid]
        rng = np.random.default_rng(42)
        n_plot = min(50_000, len(s_v))
        idx = (rng.choice(len(s_v), n_plot, replace=False)
               if len(s_v) > n_plot else np.arange(len(s_v)))

        ax_fit = axes[0, j]
        ax_res = axes[1, j]

        # fit panel
        ax_fit.scatter(s_v[idx], y_v[idx], s=1, rasterized=True,
                       color='0.6', label='Observed')
        bc, bm, bs, _ = compute_binned_stats(s_v, y_v, n_bins=N_BINS)
        if len(bc) > 0:
            ax_fit.fill_between(bc, bm - bs, bm + bs,
                                color='red', alpha=0.10)
            ax_fit.plot(bc, bm, 'o-', color='red', markersize=4,
                        linewidth=1.2, label='Binned mean', zorder=5)

        sz_line = np.linspace(max(1, s_v.min()), s_v.max(), 300)
        y_fit = spec['predict'](res['beta'], sz_line)
        ax_fit.plot(sz_line, y_fit, color='black', linewidth=2.5, zorder=4,
                    label=f'{model_label}  $R^2$={res["r2"]:.4f}')
        ax_fit.set_xscale('log')
        ax_fit.set_xlabel(r'Region size $|r|$ (voxels)')
        ax_fit.set_ylabel(stat_label)
        ax_fit.set_title(f'{src}')
        ax_fit.legend(loc='upper left', framealpha=0.9)

        # residual panel
        resid_v = res['resid'][valid]
        ax_res.scatter(s_v[idx], resid_v[idx], s=1, rasterized=True,
                       color='0.6', label='Observed')
        bc_r, bm_r, bs_r, _ = compute_binned_stats(
            s_v, resid_v, n_bins=N_BINS)
        if len(bc_r) > 0:
            ax_res.fill_between(bc_r, bm_r - bs_r, bm_r + bs_r,
                                color='red', alpha=0.12,
                                label=r'Binned mean $\pm$ 1 s.d.')
            ax_res.plot(bc_r, bm_r, 'o-', color='red', markersize=4,
                        linewidth=1.2, zorder=5)
        ax_res.axhline(0, color='black', linewidth=0.5)
        ax_res.set_xscale('log')
        ax_res.set_xlabel(r'Region size $|r|$ (voxels)')
        is_log_resid = spec['requires_positive']
        ax_res.set_ylabel('Log-space residual' if is_log_resid else 'Residual')
        ax_res.legend(loc='upper right', framealpha=0.9)

    parametric = _MODEL_PARAMETRIC.get(model_name, model_label)
    fig.suptitle(f'{model_label} Model: {parametric}', fontsize=13)
    fname = f'stat_vs_size_{stat_col}_{model_name}.pdf'
    fig.savefig(out_dir / fname)
    plt.close(fig)
    print(f'  saved {fname}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic: MANCOVA stat vs region size under H0')
    parser.add_argument('--n_perm', type=int, default=50,
                        help='permutations per data source (default: 50)')
    parser.add_argument('--stats', nargs='+', default=DEFAULT_STATS,
                        choices=list(stat_dict.keys()),
                        help=f'statistics to compute (default: {DEFAULT_STATS})')
    parser.add_argument('--out', type=str, default=None,
                        help='output directory for plots (default: auto)')
    args = parser.parse_args()

    stat_funcs = {k: stat_dict[k] for k in args.stats}

    if args.out is None:
        from glow.benchmark.config import path_result
        out_dir = path_result / 'stat_vs_size'
    else:
        out_dir = Path(args.out)

    df_dict = {}

    print('Preparing WGN experiment ...')
    exp_wgn = make_wgn_experiment()
    df_wgn = collect_null_data(exp_wgn, n_perm=args.n_perm,
                               stat_funcs=stat_funcs, label='WGN')
    df_dict['WGN'] = df_wgn

    print('\nPreparing HCP experiment ...')
    try:
        exp_hcp = make_hcp_experiment()
        df_hcp = collect_null_data(exp_hcp, n_perm=args.n_perm,
                                   stat_funcs=stat_funcs, label='HCP')
        df_dict['HCP'] = df_hcp
    except Exception as exc:
        print(f'  HCP loading failed: {exc}')
        print('  (continuing with WGN only)')

    print(f'\nGenerating diagnostic plots in {out_dir} ...')
    make_diagnostic_plots(df_dict, out_dir)
    print('Done.')


if __name__ == '__main__':
    main()
