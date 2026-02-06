import numpy as np
from scipy.ndimage import label, generate_binary_structure
from sklearn.cluster import ward_tree
from sklearn.feature_extraction import grid_to_graph


def cluster(exp, mode='ward-glm'):
    """hierarchical segmentation via Ward's method (6-connectivity in 3d).

    Args:
        exp (Experiment): experiment providing y and mask_idx
        mode (str): 'ward-naive' (pooled covariance) or 'ward-glm'
            (residual after projecting onto x)

    Returns:
        children (np.array): (num_node, 2) child index pairs
    """
    # get connectivity (ensures only neighboring voxels joined)
    # grid_to_graph uses 6-connectivity (face neighbors) for 3D by default
    assert exp.mask_idx.ndim in (2, 3), 'mask must be 2d or 3d'
    assert mode in ('ward-naive', 'ward-glm'), 'mode not recognized'

    if mode == 'ward-naive':
        y = exp.y
    else:
        # compute qr decomposition
        q, r = np.linalg.qr(exp.x.T)
        q = q.T

        # map y into span of x
        y = np.einsum('bnr,na->bar', exp.y, q.T, optimize=True)

    # reshape to vector
    num_vox = y.shape[2]
    y = y.reshape((-1, num_vox))

    # build connectivity using 6-connectivity (face neighbors) for 3D
    # grid_to_graph uses 6-connectivity by default for 3D grids
    mask = exp.mask_idx >= 0
    connectivity = grid_to_graph(*mask.shape, mask=mask)

    # ensure contiguous input (https://github.com/matthigger/glow/issues/3)
    # Use 6-connectivity for consistency with clustering
    structure = generate_binary_structure(3, 1) if (mask.ndim == 3) else generate_binary_structure(2, 1)
    _, num_regions = label(mask, structure=structure)
    if num_regions > 1:
        raise NotImplementedError('non-contiguous inputs currently '
                                  'unsupported')

    # ward's clustering
    children = ward_tree(X=y.T, connectivity=connectivity)[0]

    return children
