from copy import copy

import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from hglm.graph import get_f1

sns.set(font_scale=1.3)


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

    imageio.mimsave(file_out, list_image, fps=fps, loop=0)


def image_iter(children, mask_idx, num_vox):
    """
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
        f1 = get_f1(mask=mask,
                    mask_idx=analysis.exp.mask_idx,
                    children=analysis.child_dict[0])
    else:
        f1 = None

    size = analysis.size.astype(float)

    assert isinstance(y_feat, np.ndarray)
    assert y_feat.shape == size.shape
    y = copy(y_feat)

    if min_size > 1:
        b = analysis.size < min_size
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
