from copy import copy

import numpy as np
import pandas as pd

import glow.graph


def image_iter(children, mask_idx, num_vox):
    """yield a colour-labelled array at each step of the hierarchy.

    the first image has a unique color per voxel, the final image has one
    color per tree root.  intermediate images show the regions which
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
    color_dict = {int(idx): sample_color(idx) for idx in mask_idx[mask_idx > -1]}
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
        reg_idx = int(idx + num_vox)
        color = color_dict[int(c[0])]
        color_dict[reg_idx] = color
        del color_dict[int(c[0])]
        if int(c[1]) in color_dict:
            del color_dict[int(c[1])]

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


def prep_df(ana_glow, mask_target=None):
    df_list = list()
    _adj = ana_glow.llr_z_0
    for perm_idx, (llr, adj, size) in enumerate(zip(ana_glow.stat,
                                                    _adj,
                                                    ana_glow.size)):
        children = ana_glow.children
        d = {'region idx': np.arange(size.size),
             'llr': llr,
             'llr_z': adj,
             'size (voxels)': size,
             'permutation': perm_idx,
             'discovered': np.zeros(adj.shape, dtype=bool)}

        if not perm_idx:
            # add stats specific to unpermuted data
            d['p-val (FWER control)'] = ana_glow.pval

            # compute Dice score with mask_target
            if mask_target is not None:
                d['dice'], d['sens'], d['spec'] = \
                    glow.graph.get_dice_sens_spec(
                        children=ana_glow.children,
                        mask_idx=ana_glow.exp.mask_idx,
                        mask=mask_target)

                miss, hits = glow.graph.get_miss_hits(children=children,
                                                      mask_idx=ana_glow.exp.mask_idx,
                                                      mask=mask_target)
                d['False-Pos (voxels)'] = miss
                d['True-Pos (voxels)'] = hits

            # mark any regions as discovered
            for effect in ana_glow.effect_list:
                d['discovered'][effect.reg_idx] = True
        df_list.append(pd.DataFrame(d))

    df = pd.concat(df_list)

    return df
