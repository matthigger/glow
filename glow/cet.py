from _bisect import bisect_left

import numpy as np
from sklearn.metrics import f1_score

from glow.mask import get_neighbor_offsets, iter_neighbor


def optimize_cluster_thresh(stat, mask_true, mask_idx, alpha_fwer=.05,
                            **kwargs):
    """ using the cluster threshold which achieves max f1, get mask_est

    note: this shouldn't ever be used in practice (no access to ground
    truth!) but it's a useful upper bound of cluster extent thresholding for
    analysis

    Args:
        stat (np.array): (n_perm + 1, num_vox) higher val = more significant.
            stat[0, :] corresponds to unpermuted values
        mask_true (np.array): same shape as mask_idx, True where effect exists
            and False elsewehre
        mask_idx (np.array): -1 where voxels not to be analyzed, all other
            voxels get a unique integer (voxel index)
        alpha_fwer (float): FWER bound

    Returns:
        mask_est (np.array): optmized clutser thresholding mask
        f1 (float): f1 score achieved
    """
    mask_active = mask_idx > -1
    val_size_tup_list = list()
    for _val in stat:
        # shape into image
        img = np.zeros(mask_idx.shape)
        img[mask_active] = _val

        # compute thresh & max_size per region
        thresh, max_size = get_maxsize_per_thresh(img=img,
                                                  mask_active=mask_active,
                                                  **kwargs)
        val_size_tup_list.append((thresh, max_size))

    # align
    val_all, max_size_all = align_curves(val_size_tup_list)

    # nans imply all voxels are above threshold
    max_size_all[np.isnan(max_size_all)] = mask_active.sum()

    # for every thresh, get the min_size_thresh to ensure FWER < alpha_fwer
    min_size_thresh = np.percentile(max_size_all,
                                    100 * (1 - alpha_fwer),
                                    axis=0)

    # iterate through unpermuted data
    best_f1 = 0
    best_mask_est = np.zeros(mask_idx.shape, dtype=bool)
    img = np.zeros(mask_idx.shape)
    img[mask_active] = stat[0, :]
    for ijk, label_map, reg_size in iter_label_map_thresh(img=img,
                                                          mask_active=mask_active,
                                                          **kwargs):
        # lookup FWER threshold on size
        thresh = img[*ijk]
        size_thresh_idx = bisect_left(val_all, thresh)
        _min_size_thresh = min_size_thresh[size_thresh_idx]

        # delete regions which are too small
        label_map = label_map.copy()
        for reg, size in reg_size.items():
            if size <= _min_size_thresh:
                # delete this region
                label_map[label_map == reg] = 0

        # compute / store f1 score
        mask_est = label_map > 0
        f1 = f1_score(y_true=mask_true[mask_active],
                      y_pred=mask_est[mask_active])
        if f1 > best_f1:
            best_f1 = f1
            best_mask_est = mask_est

    return best_mask_est, best_f1


def iter_label_map_thresh(img, conn=None, offset=None, mask_active=None,
                          **kwargs):
    """ yields label_maps generated from applying every threshold to img

    Args:
        img (np.array): input image
        conn: see get_neighbor_offsets
        offset: see get_neighbor_offsets
        mask_active (np.array): boolean, same shape as img.  defines which
            voxels are included

    Yields:
        ijk (tuple): indexing of voxel just added
        label_map (np.array): label_map, 0 outside of regions (not meeting
            or exceeding thresh).  otherwise has region index
        reg_size (dict): keys are region index, values are size of region
    """
    # get offset
    assert (offset is None) != (conn is None), 'offset xor conn required'
    if offset is None:
        offset = get_neighbor_offsets(conn, **kwargs)

    if mask_active is None:
        mask_active = np.ones(img.shape)

    # build list of tuples (thresh, ijk) for each voxel (sorted big to small)
    thresh_ijk_list = list()
    for ijk in np.argwhere(mask_active):
        ijk = tuple(ijk)
        thresh_ijk_list.append((img[*ijk], ijk))
    thresh_ijk_list = sorted(thresh_ijk_list, reverse=True)

    label_map = np.zeros(img.shape)
    next_reg = 1
    reg_size = dict()
    for thresh, ijk in thresh_ijk_list:
        # get index of all neighboring regions
        neigh_set = set(iter_neighbor(label_map, ijk, offset=offset,
                                      mask_active=mask_active)) - {0}

        if neigh_set:
            # ijk is adjacent to existing region
            neigh_list = list(neigh_set)
            reg = neigh_list[0]
            for _reg in neigh_list[1:]:
                # connect the "new" neighbors which this voxel links
                label_map[label_map == _reg] = reg
                reg_size[reg] += reg_size[_reg]
                del reg_size[_reg]

            # mark new voxel
            label_map[*ijk] = reg
            reg_size[reg] += 1

        else:
            # new singleton region
            reg = next_reg
            next_reg += 1

            label_map[*ijk] = reg
            reg_size[reg] = 1

        yield ijk, label_map, reg_size


def get_maxsize_per_thresh(*, img, **kwargs):
    """ finds max size of cluster for every threshold in input image

    Args:
        (see iter_label_map_thresh)

    Returns:
        thresh (np.array): threshold
        max_size (np.array): max size of cluster at each cooresponding thresh
    """
    thresh = list()
    max_size = list()
    for ijk, label_map, reg_size in iter_label_map_thresh(img=img,
                                                          **kwargs):
        # record max_size & thresh (if a new max_size was reached)
        reg = label_map[*ijk]
        if not max_size or (reg_size[reg] > max_size[-1]):
            max_size.append(reg_size[reg])
            thresh.append(img[*ijk])

    return np.array(thresh), np.array(max_size)


def align_curves(curves):
    """ align a set of (x, y) curves onto a common x grid (chatGPT)

    Args:
        curves (list of tuple): A list of (x, y) pairs where
            - x (array_like): 1D array of x-values (must be sorted).
            - y (array_like): 1D array of y-values, same length as x.

    Returns:
        x (ndarray): Sorted union of all distinct x values.
        y (ndarray): Array of shape (n_curves, len(x_common)).
                     Each row is a curve with values forward-filled
                     to the right, and NaN before the first observation.
    """
    # build union of x’s
    x = np.unique(np.concatenate([np.asarray(x) for x, _ in curves]))
    n_curves = len(curves)
    y = np.full((n_curves, len(x)), np.nan)

    for i, (_x, _y) in enumerate(curves):
        _x = np.asarray(_x)
        _y = np.asarray(_y)
        idxs = np.searchsorted(x, _x)
        y[i, idxs] = _y

        # forward fill *only after* the first observed x
        mask = np.isnan(y[i])
        if not mask.all():
            first_known = np.where(~mask)[0][0]
            for j in range(first_known + 1, len(x)):
                if mask[j]:
                    y[i, j] = y[i, j - 1]

    return x, y
