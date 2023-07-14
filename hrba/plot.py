from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hrba.graph import get_f1

sns.set(font_scale=1.3)


def scatter_summary(*, epoch, **kwargs):
    fig, ax = plt.subplots(3, 1)

    plt.sca(ax[0])
    plt.suptitle('modelling LLR vs size (H0)')
    scatter_size_vs_stat(epoch=epoch, y_feat=epoch.llr[1:, :],
                         size=epoch.size[1:, :])
    size = epoch.size[1:, :].flatten()
    size = np.linspace(size.min(), size.max(), 101)
    mu, var = epoch.llr_model.predict(size)
    plt.plot(size, mu, color='b', label='mean')
    plt.fill_between(size, mu + var ** .5, mu - var ** .5, color='b', alpha=.3,
                     label='+/- 1 std')
    plt.legend()

    plt.sca(ax[1])
    plt.suptitle('Region Stat Signifigance')
    scatter_size_vs_stat(epoch=epoch, y_feat=epoch.llr, **kwargs, )
    plt.ylabel('llr')

    plt.sca(ax[2])
    plt.suptitle('Region Discovery')
    scatter_size_vs_stat(epoch=epoch, y_feat=epoch.z_stat, **kwargs, )
    thresh = np.percentile(epoch.z_stat.max(axis=1), [95])
    plt.axhline(thresh, color='r', linestyle='--')
    plt.ylabel('z-stat')
    plt.yscale('linear')

    return fig


def scatter_size_vs_stat(epoch, y_feat, size=None, mask=None, min_size=1):
    """ scatters size vs f_stat, colors by f1 score if mask is passed

     Args:
        epoch (Epoch):
        y_feat (np.array): y feature to plot (same size as size)
        size (np.array): size of each region to plot (defaults to epoch.size)
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

    if size is None:
        size = epoch.size.astype(float)

    assert isinstance(y_feat, np.ndarray)
    assert y_feat.shape == size.shape
    y = copy(y_feat)

    if min_size > 1:
        b = epoch.size < min_size
        size[b] = np.nan
        y[b] = np.nan

    if mask is None:
        plt.scatter(size, y, alpha=.02, color='k',
                    linewidth=0, label='region')
    else:
        plt.scatter(size[1:, :], y[1:, :], alpha=.02, color='k',
                    linewidth=0, label='region (permuted)')

        b = f1 > 0
        plt.scatter(size[0, :][b], y[0, :][b], c=f1.flatten()[b],
                    cmap='plasma', label='region w/ target', marker='s')
        cbar = plt.colorbar()
        cbar.set_label('F1 score', rotation=90)

    plt.xlabel('size')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
