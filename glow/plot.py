"""Colour-labelled images visualising a Ward hierarchy as regions merge."""
from copy import copy

import numpy as np


def image_iter(children, mask_idx, num_vox: int):
    """Yield a colour-labelled image at each step of the Ward hierarchy.

    The first image has a unique colour per voxel; the final image has one
    colour per tree root. Intermediate images show the regions as they form.

    Args:
        children (np.array): (num_reg - 1, 2) int child index pairs, one row
            per merge (equivalent to sklearn.cluster.Ward.children_)
        mask_idx (np.array): (X, Y) int 2d mask index, may be a slice of the
            full 3d mask index; merge events outside this slice are ignored
        num_vox (int): number of voxels in the analysis. This function may not
            assume mask_idx contains all of them; mask_idx may be a slice of a
            larger 3d object.

    Yields:
        image (np.array): (X, Y, 3) uint8 rgb image, first two dims matching
            mask_idx. image[i, j, :] is the rgb colour of voxel (i, j).
        mask_idx_current (np.array): (X, Y) int, same shape as mask_idx,
            holding the region label of every voxel in the current image
        color_dict (dict): maps reg_idx (int, in the current image) to its
            (3,) uint8 rgb np.array colour (randomly chosen)
    """

    def sample_color(reg):
        rng = np.random.default_rng(seed=reg)
        return rng.integers(low=0, high=255, size=3)

    assert mask_idx.ndim == 2, 'only 2d images supported'

    color_dict = {int(idx): sample_color(idx) for idx in mask_idx[mask_idx > -1]}
    mask_idx_current = copy(mask_idx)
    image = np.zeros(shape=(*mask_idx.shape, 3), dtype=np.uint8)
    for reg_idx, color in color_dict.items():
        for i, j in zip(*np.where(mask_idx_current == reg_idx)):
            image[i, j, :] = color

    yield image, mask_idx_current, color_dict

    for idx, c in enumerate(iter(children)):
        # reorder c so that c[0] is an active region (one already coloured in
        # the current image); c[1] may or may not be active. this keeps a
        # single colour flowing up to each merged parent.
        if c[0] not in color_dict and c[1] not in color_dict:
            # neither child is in the mask, so this merge is invisible here
            continue
        elif c[0] in color_dict and c[1] in color_dict:
            # arbitrary tie-break: keep the lower region index's colour
            c = sorted(c)
        elif c[0] not in color_dict and c[1] in color_dict:
            # swap so c[0] is the active region
            c = c[1], c[0]
        else:
            # remaining case: c[0] active, c[1] not, already in proper order
            pass

        # c0's colour is propagated to the parent region
        reg_idx = int(idx + num_vox)
        color = color_dict[int(c[0])]
        color_dict[reg_idx] = color
        del color_dict[int(c[0])]
        if int(c[1]) in color_dict:
            del color_dict[int(c[1])]

        # c1's colour is replaced by c0 / reg_idx's colour
        for i, j in zip(*np.where(mask_idx_current == c[1])):
            image[i, j, :] = color

        mask = tuple(mask_idx_current == _c for _c in c)
        mask_idx_current[mask[0]] = reg_idx
        mask_idx_current[mask[1]] = reg_idx

        # only yield a new image when it actually changed
        if mask[1].any():
            yield image, mask_idx_current, color_dict
