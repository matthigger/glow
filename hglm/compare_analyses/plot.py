from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


class Plotter:
    def __init__(self, df):
        # extract
        self.pval_list = sorted(df['p_val'].unique())
        self.seed_list = sorted(df['seed'].unique())
        self.rough_list = sorted(df['rough'].unique())

        shape = len(self.seed_list), len(self.pval_list), len(self.rough_list)
        self.score_dict = defaultdict(lambda: np.full(shape=shape,
                                                      fill_value=np.nan))

        for _, row in df.iterrows():
            seed_idx = self.seed_list.index(row['seed'])
            pval_idx = self.pval_list.index(row['p_val'])
            rough_idx = self.rough_list.index(row['rough'])

            for feat in ('f1', 'sens', 'spec', 'auc'):
                idx = seed_idx, pval_idx, rough_idx
                self.score_dict[row['Analysis'], feat][*idx] = row[feat]

    def plot_feats(self, pval=None, rough=None,
                   feat_list=('auc', 'f1', 'sens', 'spec')):
        assert not ((pval is None) and (rough is None)), \
            'pval or rough required'
        assert hasattr(pval, '__iter__') != hasattr(rough, '__iter__'), \
            'pval xor rough must have multiple values'

        if hasattr(pval, '__iter__'):
            # x axis is p-value
            x_feat = pval
            x_label = 'P-value'
            if rough is None:
                # average roughness of all observed
                score_dict = {k: np.nanmean(v, axis=2)
                              for k, v in self.score_dict.items()}
            else:
                # focus on given roughness
                idx = self.rough_list.index(rough)
                score_dict = {k: v[:, :, idx]
                              for k, v in self.score_dict.items()}
        else:
            # x axis is roughness
            x_feat = rough
            x_label = 'Roughness'
            if pval is None:
                # average pval of all observed
                score_dict = {k: np.nanmean(v, axis=1)
                              for k, v in self.score_dict.items()}
            else:
                # focus on given pval
                idx = self.pval_list.index(pval)
                score_dict = {k: v[:, idx, :]
                              for k, v in self.score_dict.items()}

        if len(feat_list) == 1:
            fig, ax = plt.subplots(1, 2)
            # reshape to allow for consistent indexing with multiple features
            ax = ax.reshape(2, 1)
        else:
            fig, ax = plt.subplots(2, len(feat_list))

        # plot top row
        style_dict = {'AnalysisTFCE': {'color': 'r'},
                      'AnalysisHGLM': {'color': 'b'}, }
        style_single = {'linewidth': .4,
                        'alpha': .5,
                        'zorder': 1,
                        'label': '_nolegend_'}
        style_mean = {'linewidth': 5,
                      'zorder': 2,
                      'label': '_nolegend_'}
        for _ax, feat in zip(ax[0, :], feat_list):
            plt.sca(_ax)
            for method, kwargs in style_dict.items():
                if (method, feat) not in score_dict.keys():
                    continue
                plt.plot(x_feat, score_dict[method, feat].T, **kwargs,
                         **style_single)
                plt.plot(x_feat,
                         np.nanmean(score_dict[method, feat], axis=0),
                         **kwargs, **style_mean)

            plt.xlabel(x_label)
            plt.ylabel(feat)
            plt.xscale('log')

        # plot bottom row
        for _ax, feat in zip(ax[1, :], feat_list):
            plt.sca(_ax)
            x = score_dict['AnalysisHGLM', feat] - score_dict[
                'AnalysisTFCE', feat]
            plt.axhline([0], linewidth=2, color='k')
            plt.plot(x_feat, x.T, color='k', **style_single)
            plt.plot(x_feat, np.nanmean(x, axis=0), color='k', **style_mean)

            plt.xlabel(x_label)
            plt.ylabel(f'{feat}: hglm - TFCE')
            plt.xscale('log')

        # add legend in last plot of top row
        plt.sca(ax[0, -1])
        del style_single['label']
        for method, kwargs in style_dict.items():
            plt.plot([], [], label=method[-4:], **kwargs, **style_single)
        plt.legend()

        return fig
