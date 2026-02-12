"""Diagnostic: how do MANCOVA statistics vary with region size under H0?

For each permutation tree, every region gives a (size, stat) pair under the
null hypothesis (permuted data has no true effect).  This script:

1. Generates pure-WGN data and loads HCP data -- no effect.
2. Clusters each permutation and computes all four MANCOVA stats per region.
3. Fits a power-law mean model: ln(stat) = a + b*ln(size).
4. Produces diagnostic plots: raw scatter with fit, and log-space residuals
   with binned empirical variance to assess whether a variance adjustment
   is necessary.

Usage::

    python -m glow.experiment.stat_vs_size
    python -m glow.experiment.stat_vs_size --n_perm 100
    python -m glow.experiment.stat_vs_size --out /tmp/diag
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
    decompose,
    get_hotel_tr,
    get_neg_wilks,
    get_pillai,
    get_roys_root,
)

# all four stats we evaluate
STAT_FUNCS = {
    'hotel_tr': get_hotel_tr,
    'pillai': get_pillai,
    'neg_wilks': get_neg_wilks,
    'roys_root': get_roys_root,
}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect_all_stats(exp, children):
    """Compute all four MANCOVA stats for every region in one tree walk.

    Args:
        exp (ExperimentScaled): pre-processed experiment (single permutation)
        children (np.array): (num_node, 2) child index array

    Returns:
        records (list[dict]): one dict per region with keys
            reg_idx, size, hotel_tr, pillai, neg_wilks, roys_root
    """
    records = []
    num_vox = exp.y.shape[2]

    for reg_idx, e, h in glow.graph.iter_stat(exp=exp, children=children):
        # e, h have shape (b, b, 1) -- single permutation
        _e = e[:, :, 0]
        _h = h[:, :, 0]
        rec = {'reg_idx': reg_idx}
        for name, fn in STAT_FUNCS.items():
            try:
                rec[name] = fn(e=_e, h=_h)
            except np.linalg.LinAlgError:
                rec[name] = np.nan
        records.append(rec)

    # compute sizes via node_sum
    sizes = glow.graph.node_sum(
        x=np.ones(num_vox, dtype=int), children=children)
    for rec in records:
        rec['size'] = int(sizes[rec['reg_idx']])

    return records


def collect_null_data(exp_base, n_perm, label='', verbose=True):
    """Run clustering + stat computation across permutations.

    Args:
        exp_base (Experiment): experiment with design matrix (no effect).
        n_perm (int): number of permutations to run.
        label (str): data source label for display.
        verbose (bool): show progress.

    Returns:
        df (pd.DataFrame): columns = perm_idx, reg_idx, size, plus the
            four stat columns.
    """
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
        # use perm_idx >= 1 so data is always permuted (null)
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

    # apply radius=8 spherical mask (same as benchmark)
    extenter = glow.effect.ExtenterSphere(radius=8)
    mask = extenter(mask_idx=exp.mask_idx, seed=0, contiguous=True)
    exp = exp.apply_mask(mask)
    return exp


# ---------------------------------------------------------------------------
# Power-law fit
# ---------------------------------------------------------------------------

def _wls_fit(X, y, w):
    """Weighted least squares: minimise sum w_i (y_i - X_i @ beta)^2.

    Returns beta (coefficients).
    """
    sw = np.sqrt(w)[:, np.newaxis]
    Xw = X * sw
    yw = y * sw.ravel()
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    return beta


def fit_power_law(df, stat_col, weight_col='size'):
    """Fit power-law model: ln(stat) = a + b*ln(size) via WLS.

    Args:
        df (pd.DataFrame): must contain `stat_col`, 'size', weight_col
        stat_col (str): name of the stat column
        weight_col (str): column to use as regression weight

    Returns:
        beta (np.array): [a, b] coefficients
        log_r2 (float): weighted R² in log-space
    """
    s = df['size'].values.astype(float)
    y = df[stat_col].values.astype(float)
    w = df[weight_col].values.astype(float)

    valid = np.isfinite(y) & (y > 0) & (s > 0) & np.isfinite(w)
    s_v, y_v, w_v = s[valid], y[valid], w[valid]
    log_s, log_y = np.log(s_v), np.log(y_v)

    X = np.column_stack([np.ones(len(s_v)), log_s])
    beta = _wls_fit(X, log_y, w_v)

    # weighted R² in log-space
    pred = X @ beta
    ss_res = np.sum(w_v * (log_y - pred) ** 2)
    ss_tot = np.sum(w_v * (log_y - np.average(log_y, weights=w_v)) ** 2)
    log_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return beta, log_r2


# ---------------------------------------------------------------------------
# Binned statistics
# ---------------------------------------------------------------------------

def compute_binned_stats(sizes, values, n_bins=15):
    """Compute empirical mean and std in log-spaced size bins.

    Args:
        sizes (np.array): region sizes (positive)
        values (np.array): values to bin (same length as sizes)
        n_bins (int): number of bins

    Returns:
        bin_centers (np.array): geometric mean of each bin
        bin_means (np.array): mean of values in each bin
        bin_stds (np.array): std of values in each bin
        bin_counts (np.array): number of observations in each bin
    """
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

N_BINS = 15


def plot_stat(df, stat_col, source_label, ax_scatter, ax_resid):
    """Plot one stat for one source: scatter with power-law fit, residuals.

    Args:
        df (pd.DataFrame): region data
        stat_col (str): MANCOVA stat column name
        source_label (str): 'WGN' or 'HCP'
        ax_scatter: matplotlib axis for raw scatter + fit
        ax_resid: matplotlib axis for log-space residuals with binned variance
    """
    s = df['size'].values.astype(float)
    y = df[stat_col].values.astype(float)
    pos = np.isfinite(y) & (y > 0) & (s > 0)
    s_v, y_v = s[pos], y[pos]

    if len(s_v) < 10:
        ax_scatter.set_title(f'{source_label}: {stat_col} (insufficient data)')
        return

    log_s, log_y = np.log(s_v), np.log(y_v)

    # subsample for scatter readability
    rng = np.random.default_rng(42)
    n_plot = min(50_000, len(s_v))
    idx = rng.choice(len(s_v), n_plot, replace=False) if len(s_v) > n_plot \
        else np.arange(len(s_v))

    # --- top panel: raw scatter + power-law fit ---
    ax_scatter.scatter(s_v[idx], y_v[idx], s=1, alpha=0.05, rasterized=True,
                       color='steelblue')

    # binned mean ± std in raw space
    bc, bm, bs, bn = compute_binned_stats(s_v, y_v, n_bins=N_BINS)
    if len(bc) > 0:
        ax_scatter.fill_between(bc, bm - bs, bm + bs,
                                color='red', alpha=0.15, label='binned mean +/- 1 std')
        ax_scatter.plot(bc, bm, 'o-', color='red', markersize=3, linewidth=1,
                        label='binned mean')

    # power-law fit line
    beta, log_r2 = fit_power_law(df[pos.values if hasattr(pos, 'values') else pos],
                                 stat_col)
    sz_line = np.logspace(np.log10(max(1, s_v.min())),
                          np.log10(s_v.max()), 200)
    y_fit = np.exp(beta[0] + beta[1] * np.log(sz_line))
    ax_scatter.plot(sz_line, y_fit, color='orange', linewidth=2,
                    label=f'power law (log R²={log_r2:.3f})')

    ax_scatter.set_xscale('log')
    ax_scatter.set_yscale('log')
    ax_scatter.set_xlabel('region size (voxels)')
    ax_scatter.set_ylabel(stat_col)
    ax_scatter.set_title(f'{source_label}: {stat_col}')
    ax_scatter.legend(fontsize=7, loc='upper right')

    # --- bottom panel: log-space residuals with binned variance ---
    resid = log_y - (beta[0] + beta[1] * log_s)

    ax_resid.scatter(s_v[idx], resid[idx], s=1, alpha=0.05, rasterized=True,
                     color='steelblue')

    # binned stats on residuals
    bc_r, bm_r, bs_r, bn_r = compute_binned_stats(s_v, resid, n_bins=N_BINS)
    if len(bc_r) > 0:
        # shaded band: mean +/- 1 std
        ax_resid.fill_between(bc_r, bm_r - bs_r, bm_r + bs_r,
                              color='red', alpha=0.15,
                              label='binned mean +/- 1 std')
        ax_resid.plot(bc_r, bm_r, 'o-', color='red', markersize=3,
                      linewidth=1, label='binned mean')

        # annotate counts
        for c, n in zip(bc_r, bn_r):
            ax_resid.annotate(f'n={int(n)}', xy=(c, bm_r[list(bc_r).index(c)]),
                              fontsize=5, color='gray', ha='center',
                              xytext=(0, 8), textcoords='offset points')

    ax_resid.axhline(0, color='black', linewidth=0.5)
    ax_resid.set_xscale('log')
    ax_resid.set_xlabel('region size (voxels)')
    ax_resid.set_ylabel('log-space residual')
    ax_resid.set_title(f'{source_label}: {stat_col} residuals (power law)')
    ax_resid.legend(fontsize=7, loc='upper right')


def make_diagnostic_plots(df_dict, out_dir):
    """Produce all diagnostic figures.

    Args:
        df_dict (dict): source_label -> DataFrame
        out_dir (Path): directory to save figures
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stat_names = list(STAT_FUNCS.keys())
    sources = list(df_dict.keys())
    n_src = len(sources)

    # --- per-stat figure: columns = sources, rows = [scatter, residuals] ---
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

    # --- summary table ---
    rows = []
    for src in sources:
        df = df_dict[src]
        for stat_col in stat_names:
            s = df['size'].values.astype(float)
            y = df[stat_col].values.astype(float)
            pos = np.isfinite(y) & (y > 0) & (s > 0)
            if pos.sum() < 10:
                continue
            sub = df[pos]
            beta, log_r2 = fit_power_law(sub, stat_col)

            # binned residual std summary
            log_s = np.log(s[pos])
            log_y = np.log(y[pos])
            resid = log_y - (beta[0] + beta[1] * log_s)
            bc, bm, bs, bn = compute_binned_stats(
                s[pos], resid, n_bins=N_BINS)

            rows.append({
                'source': src,
                'stat': stat_col,
                'a (intercept)': beta[0],
                'b (slope)': beta[1],
                'log_R2': log_r2,
                'resid_std_overall': resid.std(),
                'resid_std_min_bin': bs.min() if len(bs) else np.nan,
                'resid_std_max_bin': bs.max() if len(bs) else np.nan,
                'resid_std_ratio': (bs.max() / bs.min()
                                    if len(bs) and bs.min() > 0
                                    else np.nan),
            })

    summary = pd.DataFrame(rows)
    summary_path = out_dir / 'power_law_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f'\n  saved {summary_path}')

    # pretty-print
    print('\n' + '=' * 80)
    print('POWER-LAW FIT SUMMARY')
    print('=' * 80)
    for _, row in summary.iterrows():
        print(f"  {row['source']:4s} {row['stat']:12s}  "
              f"ln(stat) = {row['a (intercept)']:+.4f} "
              f"{row['b (slope)']:+.4f}·ln(size)  "
              f"R²(log)={row['log_R2']:.4f}  "
              f"σ_resid={row['resid_std_overall']:.4f}  "
              f"σ_bin=[{row['resid_std_min_bin']:.4f}, "
              f"{row['resid_std_max_bin']:.4f}]  "
              f"ratio={row['resid_std_ratio']:.2f}")
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
        from platformdirs import user_data_dir
        out_dir = Path(user_data_dir('glow', 'glow_author')) / 'stat_vs_size'
    else:
        out_dir = Path(args.out)

    df_dict = {}

    # WGN
    print('Preparing WGN experiment ...')
    exp_wgn = make_wgn_experiment()
    df_wgn = collect_null_data(exp_wgn, n_perm=args.n_perm, label='WGN')
    df_dict['WGN'] = df_wgn

    # HCP
    print('\nPreparing HCP experiment ...')
    try:
        exp_hcp = make_hcp_experiment()
        df_hcp = collect_null_data(exp_hcp, n_perm=args.n_perm, label='HCP')
        df_dict['HCP'] = df_hcp
    except Exception as exc:
        print(f'  HCP loading failed: {exc}')
        print('  (continuing with WGN only)')

    # produce plots
    print(f'\nGenerating diagnostic plots in {out_dir} ...')
    make_diagnostic_plots(df_dict, out_dir)
    print('Done.')


if __name__ == '__main__':
    main()
