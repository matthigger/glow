"""Diagnostic: fit GAM to MANCOVA stat vs region size under H0.

Fits a size-weighted GAM in log-log space for each statistic and data
source, produces scatter plots with the fitted curve, and reports R².

Usage::

    python -m glow.benchmark.stat_vs_size
    python -m glow.benchmark.stat_vs_size --n_perm 100
    python -m glow.benchmark.stat_vs_size --stats llr hotel_tr
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from tqdm import tqdm

import glow.graph
from glow.analysis import AnalysisGLOW
from glow.analysis.cluster import cluster
from glow.analysis.mancova import stat_dict
from glow.experiment.exper import Experiment, ExperimentScaled

DEFAULT_STATS = list(stat_dict.keys())

# ---------------------------------------------------------------------------
# Data collection (reused from stat_vs_size_old.py)
# ---------------------------------------------------------------------------

NUM_VOX = 10_000


def _crop_to_n_vox(exp, n_vox=NUM_VOX, seed=0):
    import glow.effect
    if exp.y.shape[2] <= n_vox:
        return exp
    extenter = glow.effect.ExtenterSphere(n_vox=n_vox)
    mask = extenter(mask_idx=exp.mask_idx, seed=seed, contiguous=True)
    return exp.apply_mask(mask)


def make_wgn_experiment():
    exp = Experiment.from_gauss(
        b=2, num_img=100, shape=(50, 50, 50), seed=0, a=2)
    return _crop_to_n_vox(exp)


def make_hcp_experiment(seed=0):
    from glow.benchmark.hcp_data import get_hcp_path
    from glow.experiment.exper import ExperimentImageOnly
    path = get_hcp_path()
    img_glob_dict = {'fa': '*_fa.nii.gz', 'md': '*_md.nii.gz'}
    exp_img = ExperimentImageOnly.from_search(
        folder=path, sbj_regex=r'[\d]{6}', img_glob_dict=img_glob_dict)
    exp = exp_img.sample_x(a=2, seed=0, add_bias=True)
    return _crop_to_n_vox(exp, seed=seed)


def _collect_all_stats(exp, children, stat_funcs):
    records = []
    num_vox = exp.y.shape[2]
    for reg_idx, size, e, h in glow.graph.iter_stat(exp=exp, children=children):
        _e, _h = e[:, :, 0], h[:, :, 0]
        rec = {'reg_idx': reg_idx}
        for name, fn in stat_funcs.items():
            try:
                rec[name] = fn(e=_e, h=_h, n=size)
            except np.linalg.LinAlgError:
                rec[name] = np.nan
        records.append(rec)

    sizes = glow.graph.node_sum(np.ones(num_vox, dtype=int), children)
    for rec in records:
        rec['size'] = int(sizes[rec['reg_idx']])
    return records


def collect_null_data(exp_base, n_perm, stat_funcs, label=''):
    if not isinstance(exp_base, ExperimentScaled):
        exp_base = ExperimentScaled.from_exp(exp_base)
    b, num_img, num_vox = exp_base.y.shape
    print(f'\n[{label}] b={b}, num_img={num_img}, '
          f'num_vox={num_vox}, n_perm={n_perm}')

    all_records = []
    for perm_idx in tqdm(range(n_perm), desc=f'{label} permutations'):
        _exp = exp_base.permute(perm_idx + 1)
        children = cluster(exp=_exp)
        recs = _collect_all_stats(_exp, children, stat_funcs)
        for r in recs:
            r['perm_idx'] = perm_idx
        all_records.extend(recs)

    df = pd.DataFrame(all_records)
    df['source'] = label
    return df


# ---------------------------------------------------------------------------
# GAM fitting in log-log space
# ---------------------------------------------------------------------------

_STAT_LABELS = {
    'llr': 'Log-Likelihood Ratio',
    'wilks': r"Wilks' $\Lambda$",
    'hotel_tr': 'Hotelling Trace',
    'pillai': "Pillai's Trace",
    'roys_root': "Roy's Largest Root",
}


def fit_gam(size, stat, n_splines=10):
    """Fit the two-stage size-adjustment GAM (mean + log-variance).

    Delegates to :py:meth:`AnalysisGLOW.fit_size_gam` so the diagnostic
    plot uses exactly the same μ_fn and σ_fn that the analysis pipeline
    will use at runtime.

    Returns:
        (fit, r2): GAMFitResult from AnalysisGLOW (with mu_fn / sigma_fn
        callables) and the R² of the mean stage.  Returns (None, None)
        when the data is too thin for the fit.
    """
    valid = np.isfinite(stat) & np.isfinite(size) & (size > 0)
    if valid.sum() < 50:
        return None, None
    fit = AnalysisGLOW.fit_size_gam(size[valid].astype(float),
                                     stat[valid].astype(float),
                                     n_splines=n_splines,
                                     score_method='z_score')
    if not fit.size_adjusted:
        return None, None
    return fit, fit.r2


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

N_BINS = 15


def _binned_stats(x, y):
    valid = np.isfinite(y) & np.isfinite(x)
    xv, yv = x[valid], y[valid]
    if len(xv) < 2:
        return np.array([]), np.array([]), np.array([])
    edges = np.linspace(xv.min(), xv.max(), N_BINS + 1)
    centers, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (xv >= lo) & (xv < hi)
        if mask.sum() < 5:
            continue
        centers.append((lo + hi) / 2)
        means.append(yv[mask].mean())
        stds.append(yv[mask].std())
    return np.array(centers), np.array(means), np.array(stds)


def plot_gam_fits(df_dict, stat_col, out_dir):
    """One figure per stat: columns = sources, scatter + GAM fit."""
    sources = list(df_dict.keys())
    n_src = len(sources)
    n_cols = min(n_src, 3)
    n_rows = int(np.ceil(n_src / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False, constrained_layout=True)

    stat_label = _STAT_LABELS.get(stat_col, stat_col)
    results = {}

    for j, src in enumerate(sources):
        ax = axes[j // n_cols, j % n_cols]
        df = df_dict[src]
        size = df['size'].values.astype(float)
        stat = df[stat_col].values.astype(float)

        valid = np.isfinite(stat) & np.isfinite(size) & (size > 0)
        if valid.sum() < 50:
            ax.set_title(f'{src} — insufficient data')
            results[src] = (None, None)
            continue

        log_s_v = np.log10(size[valid])
        y_v = stat[valid]

        # subsample for scatter
        rng = np.random.default_rng(42)
        n_plot = min(50_000, len(log_s_v))
        idx = (rng.choice(len(log_s_v), n_plot, replace=False)
               if len(log_s_v) > n_plot else np.arange(len(log_s_v)))
        ax.scatter(log_s_v[idx], y_v[idx], s=1, rasterized=True,
                   color='0.7', alpha=0.5)

        # binned means +/- 1 SD
        bc, bm, bs = _binned_stats(log_s_v, y_v)
        if len(bc) > 0:
            ax.fill_between(bc, bm - bs, bm + bs, color='red', alpha=0.10)
            ax.plot(bc, bm, 'o-', color='red', ms=4, lw=1.2, zorder=5,
                    label=r'Binned mean $\pm$ 1 SD')

        # fit two-stage GAM on raw data
        fit, r2 = fit_gam(size, stat)
        results[src] = (fit, r2)
        if fit is None:
            continue

        # GAM fit curve — span the range of bin centers
        if len(bc) > 1:
            gam_lo, gam_hi = bc[0], bc[-1]
        else:
            gam_lo, gam_hi = log_s_v.min(), log_s_v.max()
        log_line = np.linspace(gam_lo, gam_hi, 300)
        sizes_line = 10 ** log_line
        mu_line = fit.mu_fn(sizes_line)
        # σ̂ from sigma_fn carries the log-transform bias (~0.53× under-
        # estimate for Gaussian residuals; see studentize note).  The
        # bias is constant and cancels in the FWER pipeline, but for a
        # diagnostic plot we want σ̂ to visually match the empirical
        # binned ±1 SD, so apply the digamma correction:
        # E[log e²] = log σ² + ψ(1/2) - log(1/2) ≈ log σ² - 1.27.
        BIAS_LOG_VAR = 1.27
        sigma_line = fit.sigma_fn(sizes_line) * np.exp(0.5 * BIAS_LOG_VAR)

        ax.plot(log_line, mu_line,
                color='black', lw=2.5, zorder=5,
                label=fr'GAM $\hat{{\mu}}$  $R^2$={r2:.4f}')
        # GAM-estimated ±1σ̂: dashed envelope lines in a contrasting
        # colour so the band is visible behind the empirical SD shading.
        ax.plot(log_line, mu_line + sigma_line,
                color='C0', lw=1.6, ls='--', zorder=5,
                label=r'GAM $\hat{\mu} \pm \hat{\sigma}$')
        ax.plot(log_line, mu_line - sigma_line,
                color='C0', lw=1.6, ls='--', zorder=5)

        ax.set_xlabel(r'$\log_{10}\,|r|$')
        ax.set_ylabel(stat_label)
        ax.set_title(src)
        ax.legend(loc='upper left', framealpha=0.9)

    for k in range(n_src, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].set_visible(False)

    fname = f'stat_vs_size_{stat_col}_gam.pdf'
    fig.savefig(out_dir / fname)
    plt.close(fig)
    print(f'  saved {fname}')

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic: GAM fit of stat vs region size under H0')
    parser.add_argument('--n_perm', type=int, default=50)
    parser.add_argument('--stats', nargs='+', default=DEFAULT_STATS,
                        choices=DEFAULT_STATS)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--hcp_seeds', type=int, default=1,
                        help='Number of HCP locations to sample (each gets '
                             'a different random crop seed)')
    args = parser.parse_args()

    stat_funcs = {k: stat_dict[k] for k in args.stats}

    if args.out is None:
        from glow.benchmark.config import path_result
        out_dir = path_result / 'stat_vs_size'
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_dict = {}

    print('Preparing WGN experiment ...')
    exp_wgn = make_wgn_experiment()
    df_dict['WGN'] = collect_null_data(exp_wgn, n_perm=args.n_perm,
                                       stat_funcs=stat_funcs, label='WGN')

    for hcp_seed in range(args.hcp_seeds):
        label = 'HCP' if args.hcp_seeds == 1 else f'HCP_{hcp_seed}'
        print(f'\nPreparing {label} (crop seed={hcp_seed}) ...')
        try:
            exp_hcp = make_hcp_experiment(seed=hcp_seed)
            df_dict[label] = collect_null_data(
                exp_hcp, n_perm=args.n_perm,
                stat_funcs=stat_funcs, label=label)
        except (OSError, ValueError, AssertionError) as exc:
            print(f'  {label} loading failed: {exc}')

    print(f'\nFitting GAMs and plotting to {out_dir} ...\n')
    for stat_col in args.stats:
        results = plot_gam_fits(df_dict, stat_col, out_dir)
        for src, (fit, r2) in results.items():
            if r2 is not None:
                print(f'  {src:8s}  {stat_col:12s}  R²(mu)={r2:.4f}')
        print()

    print('Done.')


if __name__ == '__main__':
    main()
