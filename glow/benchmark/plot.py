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

    plt.title('Computation time (per experiment)')
    plt.xlabel('time (sec)')
    plt.ylabel('')
    sns.boxplot(data=df, x='time_sec', y='label', palette=color_map,
                hue='label')


def plot_x_vs_metrics(df, x_param='hotel_tr', metrics=['f1', 'sens', 'spec'],
                      one_vs_rest=False, one_label='GLOW', alpha=.5):
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
    fig, axes = plt.subplots(nrows, 3,
                             figsize=(14, 5.5 if one_vs_rest else 3.0),
                             sharex='col')
    if nrows == 1:
        axes = np.atleast_2d(axes)

    for j, metric in enumerate(metrics):
        ax_top = axes[0, j]

        # top: per-seed lines + bold mean
        for label, sub in df_agg.groupby('label'):
            color = color_map[label]
            for seed, g in sub.groupby('seed', sort=False):
                g = g.sort_values(x_param)
                ax_top.plot(g[x_param], g[metric], lw=.5, alpha=alpha,
                            color=color)
            mean_curve = (
                sub.groupby(x_param, as_index=False)[metric]
                .mean()
                .sort_values(x_param)
            )
            ax_top.plot(mean_curve[x_param], mean_curve[metric], lw=3,
                        color=color, label=label)

        if j == 0:
            ax_top.legend(frameon=False)
        ax_top.set_title(metric)
        ax_top.grid(True, alpha=alpha, linewidth=1.2)
        if nrows == 1:
            ax_top.set_xlabel(x_param)

        # bottom: GLOW - best(other) if possible for this metric
        if nrows == 2:
            ax_bot = axes[1, j]

            # pivot to wide: columns per label
            pivot = (
                df_agg.pivot_table(index=['seed', x_param], columns='label',
                                   values=metric)
                .reset_index()
            )

            others = [c for c in pivot.columns if
                      c not in {'seed', x_param, one_label}]
            if (one_label in pivot.columns) and len(others) > 0:
                # require glow present and at least one other present on each row
                valid = pivot[one_label].notna()
                if others:
                    valid &= pivot[others].notna().any(axis=1)
                pv = pivot.loc[valid].copy()
                if not pv.empty:
                    pv['best_other'] = pv[others].max(axis=1, skipna=True)
                    pv['diff'] = pv[one_label] - pv['best_other']

                    # per-seed thin lines
                    for seed, g in pv.groupby('seed', sort=False):
                        g = g.sort_values(x_param)
                        ax_bot.plot(g[x_param], g['diff'], lw=.5, alpha=alpha,
                                    color='black')

                    # bold mean diff
                    diff_mean = (
                        pv.groupby(x_param, as_index=False)['diff']
                        .mean()
                        .sort_values(x_param)
                    )
                    ax_bot.plot(diff_mean[x_param], diff_mean['diff'], lw=3,
                                color='black')

                    ax_bot.set_ylabel(f'{one_label} - best vba')
                    ax_bot.axhline(0, lw=.5, color='black', alpha=alpha)
                    ax_bot.grid(True, alpha=alpha, linewidth=1.2)
                    ax_bot.set_xlabel(x_param)
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

    axes[0, 0].set_ylabel('score')
    plt.tight_layout()


if __name__ == '__main__':
    import glow.benchmark

    path_result = glow.benchmark.get_path_result()
    for label in path_result.iterdir():
        folder = path_result / label
        for timestamp in folder.iterdir():
            df, _folder, n_new = glow.benchmark.load_update_all(label,
                                                                timestamp.stem,
                                                                verbose=False)

            if df.empty:
                continue

            path = _folder / 'time.pdf'
            if n_new or not path.exists():
                print(f'creating: {path}')
                plot_compute_time(df)
                plt.gcf().savefig(path, bbox_inches='tight')
                plt.close(plt.gcf())

            path = _folder / 'score.pdf'
            if n_new or not path.exists():
                print(f'creating: {path}')
                plot_x_vs_metrics(df, one_vs_rest='vba' in label.stem)
                plt.gcf().savefig(path, bbox_inches='tight')