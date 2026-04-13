import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_cmap_dict(label_list):
    label_list = sorted(label_list)
    if 'GLOW' in label_list:
        # glow / vba analysis
        c_list = sns.hls_palette(n_colors=len(label_list), l=.6, s=.9)
    else:
        # mancova (or other) comparison
        c_list = sns.husl_palette(n_colors=len(label_list), h=.9)
    return dict(zip(label_list, c_list))


def plot_compute_time(df):
    labels_sorted = sorted(df['label'].unique().tolist())
    color_map = get_cmap_dict(labels_sorted)

    plt.title('Computation Time (per Experiment)')
    plt.xlabel('time (sec)')
    plt.ylabel('')
    sns.boxplot(data=df, x='time_sec', y='label', palette=color_map,
                hue='label')


def plot_calibration(df, alpha_max=0.20, n_pts=200, title=None):
    """Plot FWER calibration curve: nominal alpha vs empirical rejection rate.

    Requires a ``min_pval`` column (minimum FWER-corrected p-value per seed).
    Each method (label) gets its own curve; the diagonal is the reference.
    """
    if 'min_pval' not in df.columns:
        print('  (no min_pval column — skipping calibration plot)')
        return

    df2 = df.copy()
    df2['min_pval'] = pd.to_numeric(df2['min_pval'], errors='coerce')
    df2 = df2.dropna(subset=['min_pval'])
    if df2.empty:
        return

    labels_sorted = sorted(df2['label'].unique().tolist())
    color_map = get_cmap_dict(labels_sorted)

    alphas = np.linspace(0, alpha_max, n_pts)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, alpha_max], [0, alpha_max], ls='--', color='grey', lw=1,
            label='ideal')

    for label in labels_sorted:
        pvals = df2.loc[df2['label'] == label, 'min_pval'].values
        n = len(pvals)
        if n == 0:
            continue
        rates = np.array([(pvals <= a).mean() for a in alphas])

        ax.plot(alphas, rates, lw=2.5, color=color_map[label], label=label)

        # binomial 95% CI at nominal alpha = 0.05
        a05 = 0.05
        r05 = (pvals <= a05).mean()
        se = np.sqrt(r05 * (1 - r05) / n) if n > 1 else 0
        ax.errorbar(a05, r05, yerr=1.96 * se, fmt='o', ms=5,
                    color=color_map[label], capsize=3)

    ax.set_xlabel('nominal $\\alpha$')
    ax.set_ylabel('empirical rejection rate')
    ax.set_title(title if title else 'FWER Calibration (Null)')
    ax.legend(frameon=False)
    ax.set_xlim(0, alpha_max)
    ax.set_ylim(0, alpha_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


_METRIC_TITLES = {
    'dice': 'Dice',
    'sens': 'Sensitivity',
    'spec': 'Specificity',
    'pct_max_dice': r'Dice$(\hat{r})\;/\;\max_r$ Dice$(r)$',
}

_X_PARAM_LABELS = {
    'effect_llr': 'Effect LLR',
    'effect_perc': 'Effect Size (% of Volume)',
    'wgn_num_img': 'Number of Subjects',
    'wgn_b': 'Number of Features ($b$)',
}


def plot_x_vs_metrics(df, x_param='effect_llr', metrics=['dice', 'sens', 'spec'],
                      one_vs_rest=False, one_label='GLOW', alpha=.5, ci=90,
                      title=None, ylabel=None):
    # ensure numeric x + metrics (prevents lexicographic sorts)
    df2 = df.copy()
    df2[x_param] = pd.to_numeric(df2[x_param], errors='coerce')
    for m in metrics:
        df2[m] = pd.to_numeric(df2[m], errors='coerce')
    df2 = df2.dropna(subset=[x_param])

    # aggregate to unique (label, seed, x_param)
    df_agg = (
        df2.groupby(['label', 'seed', x_param], as_index=False)[metrics]
        .mean()
    )

    # build color map by sorted labels → tab10(0), tab10(1), ...
    labels_sorted = sorted(df_agg['label'].unique().tolist())
    color_map = get_cmap_dict(labels_sorted)

    nrows = 2 if one_vs_rest else 1
    fig, axes = plt.subplots(
        nrows, len(metrics),
        figsize=(14, 5.5 if one_vs_rest else 3.0),
        sharex='col'
    )
    if nrows == 1:
        axes = np.atleast_2d(axes)

    # percentiles for shading
    lower_q = (100 - ci) / 2
    upper_q = 100 - lower_q

    for j, metric in enumerate(metrics):
        ax_top = axes[0, j]

        # top: per-seed lines + bold mean + percentile shading
        for label, sub in df_agg.groupby('label'):
            color = color_map[label]

            # thin per-seed lines
            for seed, g in sub.groupby('seed', sort=False):
                g = g.sort_values(x_param)
                ax_top.plot(
                    g[x_param], g[metric],
                    lw=.5, alpha=alpha, color=color
                )

            # aggregate stats: mean + quantiles
            g_stats = (
                sub.groupby(x_param)[metric]
                .agg(['mean',
                      lambda s: np.percentile(s, lower_q),
                      lambda s: np.percentile(s, upper_q)])
                .reset_index()
                .sort_values(x_param)
            )
            g_stats.columns = [x_param, 'mean', 'q_low', 'q_high']

            # mean curve
            ax_top.plot(
                g_stats[x_param], g_stats['mean'],
                lw=3, color=color, label=label
            )

            # shaded band (only on top plots)
            ax_top.fill_between(
                g_stats[x_param], g_stats['q_low'], g_stats['q_high'],
                color=color, alpha=0.2
            )

        if j == 0:
            ax_top.legend(frameon=False)
        ax_top.set_title(title if title else _METRIC_TITLES.get(metric, metric))
        if metric == 'spec':
            ax_top.set_ylim(0, 1)
        ax_top.grid(True, alpha=alpha, linewidth=1.2)
        if nrows == 1:
            ax_top.set_xlabel(_X_PARAM_LABELS.get(x_param, x_param))

        # bottom: GLOW - best(other), no shading
        if nrows == 2:
            ax_bot = axes[1, j]

            pivot = (
                df_agg.pivot_table(
                    index=['seed', x_param], columns='label', values=metric
                )
                .reset_index()
            )

            others = [c for c in pivot.columns
                      if c not in {'seed', x_param, one_label}]
            if (one_label in pivot.columns) and len(others) > 0:
                valid = pivot[one_label].notna()
                if others:
                    valid &= pivot[others].notna().any(axis=1)
                pv = pivot.loc[valid].copy()
                if not pv.empty:
                    pv['best_other'] = pv[others].max(axis=1, skipna=True)
                    pv['diff'] = pv[one_label] - pv['best_other']

                    # per-seed thin lines (light grey for visibility)
                    for seed, g in pv.groupby('seed', sort=False):
                        g = g.sort_values(x_param)
                        ax_bot.plot(
                            g[x_param], g['diff'],
                            lw=.5, alpha=0.7, color='lightgrey'
                        )

                    # mean diff (bold black line, no shading)
                    diff_mean = (
                        pv.groupby(x_param)['diff']
                        .mean()
                        .reset_index()
                        .sort_values(x_param)
                    )
                    ax_bot.plot(
                        diff_mean[x_param], diff_mean['diff'],
                        lw=3, color='black'
                    )

                    ax_bot.set_ylabel(f'{one_label} - best vba')
                    ax_bot.axhline(0, lw=.5, color='black', alpha=alpha)
                    ax_bot.grid(True, alpha=alpha, linewidth=1.2)
                    ax_bot.set_xlabel(_X_PARAM_LABELS.get(x_param, x_param))
                else:
                    ax_bot.axis('off')
            else:
                ax_bot.axis('off')

    # log x-axis if strictly positive
    xmin = df_agg[x_param].min()
    if pd.notnull(xmin) and xmin > 0:
        for r in range(nrows):
            for ax in axes[r]:
                ax.set_xscale('log')

    axes[0, 0].set_ylabel(ylabel if ylabel else 'score')
    plt.tight_layout()

if __name__ == '__main__':
    import shutil
    import glow.benchmark
    from glow.benchmark.paper_config import CONFIG_BY_LABEL

    force_replot = True

    path_result = glow.benchmark.get_path_result()
    latest = path_result / '_latest'
    latest.mkdir(exist_ok=True)

    for label, config in CONFIG_BY_LABEL.items():
        df, folder, n_new = glow.benchmark.load_update_all(label,
                                                           verbose=False)
        if df.empty:
            print(f'skipping {label}: no data')
            continue

        # filter to current config (ignore stale results from old configs)
        if 'config_hash' in df.columns:
            df = df[df['config_hash'] == config._config_hash()]
            if df.empty:
                print(f'skipping {label}: no data for current config')
                continue

        x_param = getattr(config, 'x_param', 'effect_llr')
        is_null = label.startswith('null_')

        # null configs: calibration plot instead of score curves
        if is_null:
            path = folder / 'calibration.pdf'
            if force_replot or n_new or not path.exists():
                print(f'creating: {path}')
                cal_title = 'WGN' if 'wgn' in label else 'HCP'
                plot_calibration(df, title=cal_title)
                plt.gcf().savefig(path, bbox_inches='tight')
                plt.close('all')
            else:
                print(f'skipping: {path} (already exists, no new data)')

            if path.exists():
                dest = latest / f'{label}.pdf'
                shutil.copy2(path, dest)
                print(f'  -> {dest}')
            continue

        # generate score plot
        path = folder / 'score.pdf'
        if force_replot or n_new or not path.exists():
            print(f'creating: {path}')
            labels_in_data = set(df['label'].unique())
            has_comparison = 'GLOW' in labels_in_data and len(labels_in_data) > 1
            plot_x_vs_metrics(df, x_param=x_param,
                              one_vs_rest=has_comparison)
            plt.gcf().savefig(path, bbox_inches='tight')
            plt.close('all')
        else:
            print(f'skipping: {path} (already exists, no new data)')

        # copy score.pdf to _latest/
        if path.exists():
            dest = latest / f'{label}.pdf'
            shutil.copy2(path, dest)
            print(f'  -> {dest}')

        # generate pct_max_dice plot (only GLOW variants)
        if 'pct_max_dice' in df.columns:
            path_mf = folder / 'max_dice.pdf'
            if force_replot or n_new or not path_mf.exists():
                print(f'creating: {path_mf}')
                df_glow = df[df['label'].str.contains('GLOW')]
                source_tag = label.rsplit('_', 1)[-1].upper()
                plot_x_vs_metrics(df_glow, x_param=x_param,
                                  metrics=['pct_max_dice'],
                                  one_vs_rest=False, title=source_tag,
                                  ylabel=r'Dice$(\hat{r})\;/\;\max_r$ Dice$(r)$')
                plt.gcf().savefig(path_mf, bbox_inches='tight')
                plt.close('all')
            else:
                print(f'skipping: {path_mf} (already exists, no new data)')

            if path_mf.exists():
                dest = latest / f'{label}_max_dice.pdf'
                shutil.copy2(path_mf, dest)
                print(f'  -> {dest}')