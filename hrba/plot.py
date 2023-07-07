from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hrba.graph import get_f1

sns.set(font_scale=1.3)


def scatter_size_vs_two(*, epoch, **kwargs):
    fig, ax = plt.subplots(1, 2)
    plt.sca(ax[0])
    llr_over_size = epoch.llr / epoch.size
    scatter_size_vs_stat(epoch=epoch, y_feat=llr_over_size, **kwargs, )
    thresh = np.percentile(llr_over_size.max(axis=1), [95])
    plt.axhline(thresh, color='r', linestyle='--')
    plt.sca(ax[1])
    scatter_size_vs_stat(epoch=epoch, y_feat=epoch.llr, **kwargs, )
    return fig


def scatter_size_vs_stat(epoch, mask=None, y_feat='llr', min_size=1):
    """ scatters size vs f_stat, colors by f1 score if mask is passed

     Args:
        epoch (Epoch):
        y_feat (np.array): y feature to plot (same size as epoch.size)
        mask (np.array): target mask
        min_size (int): smallest size to be plotted
     """

    # compute f1 score
    if mask is not None:
        f1 = get_f1(mask=mask,
                    mask_idx=epoch.exp.mask_idx,
                    children=epoch.child_dict[0])
    else:
        f1 = None

    x = epoch.size.astype(float)
    assert isinstance(y_feat, np.ndarray)
    assert y_feat.shape == x.shape
    y = copy(y_feat)

    if min_size > 1:
        b = epoch.size < min_size
        x[b] = np.nan
        y[b] = np.nan

    if mask is None:
        raise NotImplementedError
    else:
        plt.scatter(x[1:, :], y[1:, :], alpha=.02, color='k',
                    linewidth=0, label='region (permuted)')

        b = f1 > 0
        plt.scatter(x[0, :][b], y[0, :][b], c=f1.flatten()[b],
                    cmap='plasma', label='region w/ target', marker='s')
        cbar = plt.colorbar()
        cbar.set_label('F1 score', rotation=90)

    plt.xlabel('size')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
