import numpy as np
from scipy.ndimage import label
from sklearn.cluster import ward_tree
from sklearn.feature_extraction import grid_to_graph


def cluster(exp, mode='ward-proj'):
    """ hierarchical segmentation of image

    Args:
        exp (Experiment):
        mode (str): 'ward', 'proj'
            'ward-full': reduces image-pooled spatial covariance
            'ward-proj': removes error

    Returns:
        children (np.array): (num_reg, 2) each col are index of child
            regions
    """
    # get connectivity (ensures only neighboring voxels joined)
    assert exp.mask_idx.ndim in (2, 3), 'mask must be 2d or 3d'
    assert mode in ('ward-full', 'ward-proj'), 'mode not recognized'

    if mode == 'ward-full':
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

    # build connectivity
    mask = exp.mask_idx >= 0
    connectivity = grid_to_graph(*mask.shape, mask=mask)

    # ensure contiguous input (https://github.com/matthigger/hglm/issues/3)
    _, num_regions = label(mask)
    if num_regions > 1:
        raise NotImplementedError('non-contiguous inputs currently '
                                  'unsupported')

    # ward's clustering
    children = ward_tree(X=y.T, connectivity=connectivity)[0]

    return children
