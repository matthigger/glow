from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


def extract(df):
    # extract
    pval_list = sorted(df['p_val'].unique())
    seed_list = sorted(df['seed'].unique())

    shape = len(seed_list), len(pval_list)
    score_dict = defaultdict(lambda: np.full(shape=shape, fill_value=np.nan))

    for _, row in df.iterrows():
        seed_idx = seed_list.index(row['seed'])
        pval_idx = pval_list.index(row['p_val'])

        for feat in ('f1', 'sens', 'spec'):
            score_dict[row['Analysis'], feat][seed_idx, pval_idx] = row[feat]

    return pval_list, seed_list, score_dict


def plot_f1_sens_spec(pval_list, score_dict):
    fig, ax = plt.subplots(2, 3)

    # plot top row
    style_dict = {'AnalysisTFCE': {'color': 'r'},
                  'AnalysisHGLM': {'color': 'b'}}
    style_single = {'linewidth': .5,
                    'zorder': 1,
                    'label': '_nolegend_'}
    style_mean = {'linewidth': 5,
                  'zorder': 2,
                  'label': '_nolegend_'}
    for _ax, feat in zip(ax[0, :], ('f1', 'sens', 'spec')):
        plt.sca(_ax)
        for method, kwargs in style_dict.items():
            plt.plot(pval_list, score_dict[method, feat].T, **kwargs,
                     **style_single)
            plt.plot(pval_list, np.nanmean(score_dict[method, feat], axis=0),
                     **kwargs, **style_mean)

        plt.xlabel('p_val')
        plt.ylabel(feat)
        plt.xscale('log')

    # plot bottom row
    for _ax, feat in zip(ax[1, :], ('f1', 'sens', 'spec')):
        plt.sca(_ax)
        x = score_dict['AnalysisHGLM', feat] - score_dict['AnalysisTFCE', feat]
        plt.axhline([0], linewidth=2, color='k')
        plt.plot(pval_list, x.T, color='k', **style_single)
        plt.plot(pval_list, np.nanmean(x, axis=0), color='k', **style_mean)

        plt.xlabel('p_val')
        plt.ylabel(f'{feat}: hglm - TFCE')
        plt.xscale('log')

    # add legend in last plot of top row
    plt.sca(ax[0, -1])
    del style_single['label']
    for method, kwargs in style_dict.items():
        plt.plot([], [], label=method[-4:], **kwargs, **style_single)
    plt.legend()

    return fig
