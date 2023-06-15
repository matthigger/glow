import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hrba.graph import get_f1

sns.set(font_scale=1.3)


def scatter_size_vs_two(*args, **kwargs):
    fig, ax = plt.subplots(1, 2)
    plt.sca(ax[0])
    scatter_size_vs_stat(*args, **kwargs, y_feat='f_stat', f_trend=True)
    plt.sca(ax[1])
    scatter_size_vs_stat(*args, **kwargs, y_feat='z_stat', f_trend=False)
    return fig


def scatter_size_vs_stat(epoch, mask=None, y_feat='f_stat', f_trend=False):
    """ scatters size vs f_stat, colors by f1 score if mask is passed

     Args:
        epoch (Epoch):
        mask (np.array): target mask
        y_feat (str): 'f_stat' or 'z_stat'
        f_trend (bool): toggles graphing of f trends (for f to z adjustment)
     """

    # compute f1 score
    if mask is not None:
        f1 = get_f1(mask=mask,
                    mask_idx=epoch.exp.mask_idx,
                    children=epoch.child_dict[0])
    else:
        f1 = None

    x = epoch.size
    if y_feat == 'f_stat':
        y = epoch.f_stat
    elif y_feat == 'z_stat':
        y = epoch.z_stat
    else:
        raise AttributeError(f'unrecognized y_feat: {y_feat}')

    plt.scatter(x[1:, :], y[1:, :], alpha=.02, color='k',
                linewidth=0, label='region (permuted)')

    if f_trend:
        assert y_feat == 'f_stat', 'f_trend only valid on f_stat scatter'

        # plot trend of f used in computing z stats
        _x = np.log10(np.linspace(x.min(), x.max(), 101)).reshape(-1, 1)
        f_mean = 10 ** epoch.model_f_mu.predict(_x)
        _x = 10 ** _x.flatten()
        plt.plot(_x, f_mean, linewidth=2, color='g', label='f_mean(size)')

    plt.scatter(x[0, :], y[0, :], c=f1.flatten(),
                cmap='plasma', label='region', marker='s')

    if mask is not None:
        cbar = plt.colorbar()
        cbar.set_label('F1 score', rotation=90)
    plt.xlabel('size')
    plt.ylabel(y_feat)
    plt.xscale('log')
    if y_feat == 'f_stat':
        plt.yscale('log')
    plt.legend()
