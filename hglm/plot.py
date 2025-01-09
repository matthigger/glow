from copy import copy

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import hglm.graph

sns.set(font_scale=1.3)

import plotly.express as px
import plotly.graph_objects as go


def make_gif(file_out, n_list=30, fps=10, mask_idx=None, min_n=1,
             **kwargs):
    """

    Args:
        file_out (str): output file
        n_list (int or iterable): defines how many unique regions are in each time
            slice in output gif.  if integer n, then n logarithmically spaced
            regions are chosen from 1 to the number of unique items in
            mask_idx.  alternatively user can pass an iterable which gives
            number of regions explicitly (must be decreasing)
        fps (int): frames per second
        mask_idx (np.array): 2d mask index, may be a slice of the full 3d mask
            index, we'll ignore any merge events outside of this slice
        **kwargs: see image_iter
    """
    # prep n
    if isinstance(n_list, int):
        n_list = np.geomspace((mask_idx > -1).sum(), min_n, n_list).astype(int)

    n_list = np.array(sorted(n_list, reverse=True)).astype(int)

    # build a list of images
    list_image = list()
    idx = 0
    for image, _, color_dict in image_iter(mask_idx=mask_idx, **kwargs):
        if len(color_dict) == n_list[idx]:
            # image of right number of regions found, store it
            list_image.append(copy(image))
            idx += 1

        if idx >= len(n_list):
            # all images found, quit
            break

    # pillow uses duration
    duration = 1 / fps * 1000

    imageio.mimsave(file_out, list_image, duration=duration, loop=0)


def image_iter(children, mask_idx, num_vox):
    """ yields array per step in hierarchical cluster (region has same color)

    the first image has a unique color per voxel, the final image has the
    same color for all voxels.  intermediate images show the regions which
    are formed

    Args:
        children (np.array): (num_reg - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_)
        mask_idx (np.array): 2d mask index, may be a slice of the full 3d mask
            index, we'll ignore any merge events outside of this slice
        num_vox (int): number of voxels in analysis (this fnc may not assume
            mask_idx contains all of them, may only be a slice of a 3d object)

    Yields:
        image (np.array): first 2 dims are same shape as mask_idx,
            third dimension is 3 (rgb color).  image[i, j, :] is the uint8 rgb
            color
        mask_idx_current (np.array): same shape as mask_idx.  contains the
            mask of every region in the current image
        color_dict (dict): keys are reg_idx (in current image) and values are
            rgb np.array colors.  (they're randomly chosen)
    """

    def sample_color(reg):
        rng = np.random.default_rng(seed=reg)
        return rng.integers(low=0, high=255, size=3)

    assert mask_idx.ndim == 2, 'only 2d images supported'

    # initialize image
    color_dict = {idx: sample_color(idx) for idx in mask_idx[mask_idx > -1]}
    mask_idx_current = copy(mask_idx)
    image = np.zeros(shape=(*mask_idx.shape, 3), dtype=np.uint8)
    for reg_idx, color in color_dict.items():
        for i, j in zip(*np.where(mask_idx_current == reg_idx)):
            image[i, j, :] = color

    yield image, mask_idx_current, color_dict

    for idx, c in enumerate(iter(children)):
        # order c so that c[0] is an active region (has some ancestor,
        # or self which is in mask_idx), c[1] may or may not have this
        # may or may not be in mask_idx
        if c[0] not in color_dict and c[1] not in color_dict:
            # neither color is in mask, skip it entirely
            continue
        elif c[0] in color_dict and c[1] in color_dict:
            # choose lower region index to keep color (arbitrary choice)
            c = sorted(c)
        elif c[0] not in color_dict and c[1] in color_dict:
            # swap order so that output c[0] is an active region
            c = c[1], c[0]
        else:
            # condition: c[0] in color_dict and c[1] not in color_dict
            # already in proper order here, loop body just documents this fact
            pass

        # update color_dict (c0's color is propogated to parent)
        reg_idx = idx + num_vox
        color = color_dict[c[0]]
        color_dict[reg_idx] = color
        del color_dict[c[0]]
        if c[1] in color_dict:
            del color_dict[c[1]]

        # update image (c1's color is replaced with c0 / reg_idx's color)
        for i, j in zip(*np.where(mask_idx_current == c[1])):
            image[i, j, :] = color

        # update mask_idx_current
        mask = tuple(mask_idx_current == _c for _c in c)
        mask_idx_current[mask[0]] = reg_idx
        mask_idx_current[mask[1]] = reg_idx

        if mask[1].any():
            # only pass a new image if its changed
            yield image, mask_idx_current, color_dict


def prep_df(ana_hglm, mask_target=None):
    df_list = list()
    for perm_idx, (fstat, z, size) in enumerate(zip(ana_hglm.fstat[:, 0, :],
                                                    ana_hglm.z_stat,
                                                    ana_hglm.size)):
        children = ana_hglm.child_dict[perm_idx]
        d = {'region idx': np.arange(size.size),
             'F-stat': fstat,
             'Z-stat': z,
             'size (voxels)': size,
             'permutation': perm_idx,
             'discovered': np.zeros(z.shape, dtype=bool),
             'F-std-h0': ana_hglm.fstat[perm_idx, 1:, :].std(axis=0),
             'F-mu-h0': ana_hglm.fstat[perm_idx, 1:, :].mean(axis=0)}

        if not perm_idx:
            # add stats specific to unpermuted data
            d['p-val (FWER control)'] = ana_hglm.p_val

            # compute f1 (dice) score with mask_target
            if mask_target is not None:
                d['dice'] = hglm.graph.get_f1(children=ana_hglm.child_dict[0],
                                              mask_idx=ana_hglm.exp.mask_idx,
                                              mask=mask_target)
                miss, hits = hglm.graph.get_miss_hits(children=children,
                                                      mask_idx=ana_hglm.exp.mask_idx,
                                                      mask=mask_target)
                d['False-Pos (voxels)'] = miss
                d['True-Pos (voxels)'] = hits

            # mark any regions as discovered
            for effect in ana_hglm.effect_list:
                d['discovered'][effect.reg_idx] = True
        df_list.append(pd.DataFrame(d))

    df = pd.concat(df_list)

    return df


def scatter_size_vs_stat(analysis, y_feat, mask=None, min_size=1):
    """ scatters size vs f_stat, colors by f1 score if mask is passed

     Args:
        analysis (Analysis):
        y_feat (np.array): y feature to plot (same size as size)
        mask (np.array): target mask
        min_size (int): smallest size to be plotted
     """

    # compute f1 score
    if mask is not None:
        f1 = hglm.graph.get_f1(mask=mask,
                               mask_idx=analysis.exp.mask_idx,
                               children=analysis.child_dict[0])
    else:
        f1 = None

    size = analysis.size.astype(float)

    assert isinstance(y_feat, np.ndarray)
    assert y_feat.shape[1] == size.shape[1]
    size = size[:y_feat.shape[0], :]
    y = copy(y_feat)

    if min_size > 1:
        b = analysis.size < min_size
        size[b] = np.nan
        y[b] = np.nan

    if mask is None:
        plt.scatter(size, y, alpha=1, color='k',
                    linewidth=0, label='region')
    else:
        plt.scatter(size[1:, :], y[1:, :], alpha=1, color='k',
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


def scatter_plotly(ana_hglm, mask_target=None, x_feat='size (voxels)',
                   y_feat='Z-stat', color_feat='dice',
                   plot_permute=True, plot_tree=True, log_x=True, log_y=True):
    """

    todo:
    - doesn't work without mask
    - push unpermuted nodes back (and make them grey)
    - line widths on tree a bit too thick
    - max number of scatter points (enforce a minimum size)
    """

    df = prep_df(ana_hglm, mask_target=mask_target)
    df['hover_name'] = df['region idx'].map(lambda x: f'Region {x:.0f}')
    idx0 = np.where((df['permutation'] != 0).values)[0]
    idx1 = list(df.columns).index('hover_name')
    df.iloc[idx0, idx1] = 'Permuted Region'

    # trim to unpermuted
    _df = df[df['permutation'] == 0].sort_values('region idx')
    x = _df[x_feat].values
    y = _df[y_feat].values

    tree_x = list()
    tree_y = list()
    num_vox = ana_hglm.exp.y.shape[2]
    for idx, child in enumerate(ana_hglm.child_dict[0]):
        par = idx + num_vox
        for c in child:
            # plotly will only draw this line segment (not connected to next)
            # given Nones
            tree_x += [x[c], x[par], None]
            tree_y += [y[c], y[par], None]

    hover_data = {'size (voxels)': ':.0f',
                  'Z-stat': ':.3e',
                  'F-stat': ':.3e',
                  'F-std-h0': ':.3e',
                  'F-mu-h0': ':.3e'}
    hover_data_permuted = {'permutation': ':.0f',
                           'region idx': ':.0f'}
    hover_data_unpermuted = {'p-val (FWER control)': ':.2e',
                             'dice': ':.3f',
                             'False-Pos (voxels)': ':.0f',
                             'True-Pos (voxels)': ':.0f'}

    fig_list = list()

    b = df['permutation'] == 0
    if plot_permute:
        # plot permuted regions
        fig_list.append(px.scatter(df[~b],
                                   x=x_feat,
                                   y=y_feat,
                                   hover_name='hover_name',
                                   hover_data=hover_data | hover_data_permuted))

    if plot_tree:
        # tree connecting unpermuted data points to each other (hierarchical
        # segmentation)
        fig_list.append(px.line(x=tree_x,
                                y=tree_y,
                                color_discrete_sequence=('grey',)))

    # plot unpermuted regions
    fig_list.append(px.scatter(df[b],
                               x=x_feat,
                               y=y_feat,
                               color=color_feat,
                               hover_name='hover_name',
                               hover_data=hover_data | hover_data_unpermuted))
    fig_list[-1].update_traces(marker=dict(size=15))

    data = fig_list[0].data
    for fig in fig_list[1:]:
        data += fig.data
    fig = go.Figure(data=data)
    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    fig.update_layout(height=600,
                      xaxis_title=x_feat,
                      yaxis_title=y_feat,
                      coloraxis_colorbar=dict(title=color_feat),
                      hoverlabel=dict(bgcolor='white'))
    return fig, df


# tmp
if __name__ == '__main__':
    from compare_analyses.data import load_update_all, load

    p_val, seed = 0.25, 15
    folder = '/home/matt/Dropbox/pnl_hglm/results/exp_24Apr22-1533'
    df = load_update_all(folder)
    ana_hglm, effect = load(folder=folder, df=df, p_val=p_val, seed=seed,
                            Analysis='AnalysisHGLM')
    prep_df(ana_hglm, mask_target=effect.mask)
