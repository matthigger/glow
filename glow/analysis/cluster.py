from enum import StrEnum

import numpy as np
from scipy.ndimage import label, generate_binary_structure
from sklearn.feature_extraction.image import grid_to_graph

from glow.mask import bbox_crop
from .mancova import decompose
from .ward import ward_tree


class ClusterMode(StrEnum):
    """Ward projection mode.

    NAIVE: cluster raw y (no projection).
    GLM_ERROR: project onto full design space (q0 + q1).
    FOCUS: project onto contrast-of-interest subspace (q1) only.
    """
    NAIVE = 'Naive'
    GLM_ERROR = 'GLM Error'
    FOCUS = 'Focus'


def cluster(exp, mode=ClusterMode.FOCUS):
    """hierarchical segmentation via Ward's method (6-connectivity in 3d).

    Supports non-contiguous masks: each connected component is clustered
    independently and results are concatenated into a forest.

    Args:
        exp (Experiment): experiment providing y and mask_idx
        mode (ClusterMode): which Y projection to cluster on.

    Returns:
        children (np.array): (num_internal, 2) child index pairs.
            For a contiguous mask, num_internal = num_vox - 1.
            For k components, num_internal = num_vox - k.
    """
    assert exp.mask_idx.ndim in (2, 3), 'mask must be 2d or 3d'
    mode = ClusterMode(mode)

    if mode is ClusterMode.NAIVE:
        y = exp.y
    elif mode is ClusterMode.GLM_ERROR:
        q, r = np.linalg.qr(exp.x.T)
        q = q.T
        y = np.einsum('bnr,na->bar', exp.y, q.T, optimize=True)
    else:
        _, q1, _ = decompose(exp.x, exp.contrast)
        y = np.einsum('bnr,na->bar', exp.y, q1.T, optimize=True)

    num_vox = y.shape[2]
    y = y.reshape((-1, num_vox))

    mask = exp.mask_idx >= 0
    mask_bb, bb_slices = bbox_crop(mask)
    structure = generate_binary_structure(mask.ndim, 1)
    labeled, num_components = label(mask_bb, structure=structure)

    if num_components == 1:
        connectivity = grid_to_graph(*mask_bb.shape, mask=mask_bb)
        children = ward_tree(X=y.T, connectivity=connectivity)[0]
        return children

    # global indices of active voxels, ordered by raster scan of the bbox
    global_idx = exp.mask_idx[bb_slices][mask_bb]

    # per-component clustering → forest
    all_children = []
    internal_offset = 0

    for c in range(1, num_components + 1):
        comp_mask_bb = labeled == c
        comp_bb, _ = bbox_crop(comp_mask_bb)
        comp_select = labeled[mask_bb] == c
        comp_global_idx = global_idx[comp_select]
        n_c = comp_global_idx.size

        if n_c < 2:
            continue

        local_X = y[:, comp_global_idx].T
        local_conn = grid_to_graph(*comp_bb.shape, mask=comp_bb)
        local_children = ward_tree(X=local_X, connectivity=local_conn)[0]

        remapped = np.empty_like(local_children)
        for col in range(2):
            is_leaf = local_children[:, col] < n_c
            remapped[is_leaf, col] = comp_global_idx[
                local_children[is_leaf, col]]
            int_local = local_children[~is_leaf, col] - n_c
            remapped[~is_leaf, col] = num_vox + internal_offset + int_local

        all_children.append(remapped)
        internal_offset += n_c - 1

    children = np.concatenate(all_children, axis=0)
    return children
