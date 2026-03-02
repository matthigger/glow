import numpy as np
from scipy.ndimage import label, generate_binary_structure
from sklearn.cluster import ward_tree
from sklearn.feature_extraction.image import grid_to_graph


def count_components(mask_idx):
    """Number of connected components in the mask.

    Uses 6-connectivity for 3D, 4-connectivity for 2D (matching
    grid_to_graph defaults and Ward's clustering connectivity).

    Args:
        mask_idx (np.array): -1 outside mask, sequential int inside

    Returns:
        k (int): number of connected components
    """
    mask = mask_idx >= 0
    structure = generate_binary_structure(mask.ndim, 1)
    _, k = label(mask, structure=structure)
    return k


def cluster(exp, mode='ward-glm'):
    """hierarchical segmentation via Ward's method (6-connectivity in 3d).

    Supports non-contiguous masks: each connected component is clustered
    independently and results are concatenated into a forest.

    Args:
        exp (Experiment): experiment providing y and mask_idx
        mode (str): 'ward-naive' (pooled covariance) or 'ward-glm'
            (residual after projecting onto x)

    Returns:
        children (np.array): (num_internal, 2) child index pairs.
            For a contiguous mask, num_internal = num_vox - 1.
            For k components, num_internal = num_vox - k.
    """
    assert exp.mask_idx.ndim in (2, 3), 'mask must be 2d or 3d'
    assert mode in ('ward-naive', 'ward-glm'), 'mode not recognized'

    if mode == 'ward-naive':
        y = exp.y
    else:
        q, r = np.linalg.qr(exp.x.T)
        q = q.T
        y = np.einsum('bnr,na->bar', exp.y, q.T, optimize=True)

    num_vox = y.shape[2]
    y = y.reshape((-1, num_vox))

    mask = exp.mask_idx >= 0
    structure = generate_binary_structure(mask.ndim, 1)
    labeled, num_components = label(mask, structure=structure)

    if num_components == 1:
        connectivity = grid_to_graph(*mask.shape, mask=mask)
        children = ward_tree(X=y.T, connectivity=connectivity)[0]
        return children

    # per-component clustering → forest
    all_children = []
    internal_offset = 0

    for c in range(1, num_components + 1):
        comp_mask = labeled == c
        comp_global_idx = exp.mask_idx[comp_mask]
        n_c = comp_global_idx.size

        if n_c < 2:
            continue

        local_X = y[:, comp_global_idx].T
        local_conn = grid_to_graph(*comp_mask.shape, mask=comp_mask)
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
