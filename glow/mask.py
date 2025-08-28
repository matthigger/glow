import numpy as np
from sklearn.metrics import f1_score, recall_score, confusion_matrix

conn_dict = {6: np.array([[[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]],
                          [[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]],
                          [[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]]),
             18: np.array([[[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]],
                           [[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                           [[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]]]),
             26: np.array([[[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                           [[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]],
                           [[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]]])}


def get_mask_idx(mask):
    """ builds mask_idx from mask

    Args:
        mask (np.array): boolean, False where voxels not to be analyzed

    Returns:
        mask_idx (np.array): -1 where voxels not to be analyzed, all other
            voxels get a unique integer (voxel index)
    """
    mask = mask.astype(bool)
    mask_idx = np.full(mask.shape, fill_value=-1)
    num_vox = mask.sum()
    mask_idx[mask] = np.arange(num_vox)
    return mask_idx


def get_entropy(mask_idx):
    """ computes the entropy of a label map, ignoring labels < 0

    Args:
        mask_idx (np.array): -1 where voxels not to be analyzed, all other
            voxels get a unique integer (voxel index)

    Returns:
        entropy (float): entropy in bits
    """
    valid_labels = mask_idx[mask_idx >= 0]
    if valid_labels.size == 0:
        return 0.0

    value, count = np.unique(valid_labels, return_counts=True)
    prob = count / count.sum()
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)


def get_score(mask_pred, mask_target, mask_active=None):
    """ gets f1, sens, spec scores per analysis given ground truth effect
    """
    if mask_active is None:
        # no mask_active passed, assume all voxels were analyzed
        y_true = mask_target.flatten()
        y_pred = mask_pred.flatten()
    else:
        # discard inactive voxels (not analyzed)
        y_true = mask_target[mask_active]
        y_pred = mask_pred[mask_active]

    # compute scores (default to zero)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    sens = recall_score(y_true=y_true, y_pred=y_pred, zero_division=0)

    # specificity with safe division (defaults to 1)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    denom = tn + fp
    spec = 1 if denom == 0 else tn / denom

    return f1, sens, spec


def trim_zeros_2d(x, to_trim=0):
    """ trims rows / columns which are entirely zero

     (of course, spoils affine but useful to "zoom" in jupyter demos)

     """
    # rows and cols where there is at least one non-zero element
    non_zero_rows = np.any(x != to_trim, axis=1)
    non_zero_cols = np.any(x != to_trim, axis=0)

    # indices of the first and last True values
    row_start, row_end = np.where(non_zero_rows)[0][[0, -1]]
    col_start, col_end = np.where(non_zero_cols)[0][[0, -1]]

    return x[row_start:row_end + 1, col_start:col_end + 1]


def get_neighbor_offsets(conn, not_reflexive=True):
    """ get neighbor index offset from a connectivity mask

    expected usage:

    offset = get_neighbor_offsets(conn=26)

    # get a list of neighbor voxel coordinates of ijk
    ijk_list = [tuple(_ijk) for _ijk in ijk + offset]

    # pop them into the array as
    array[*ijk_list[0]]

    Args:
        conn (int or array_like): Either:
            - An integer key (e.g., 6, 18, or 26) referring to a predefined
              3D connectivity mask in `conn_dict`, or
            - An N-dimensional array (with odd size along each axis) where
              nonzero entries define neighbors relative to the center.
        not_reflexive (bool, optional): If True, the central voxel (offset = 0
            along all axes) is excluded from the returned offset. Default is False.

    Returns:
        numpy.ndarray: An array of shape (nconn, ndim) where:
            - `nconn` is the number of neighbors,
            - `ndim` is the dimensionality of the mask (`conn.ndim`).
          Each row is an offset vector relative to the center voxel.
    """
    if type(conn) is int:
        conn = conn_dict[conn]
    else:
        conn = np.array(conn)

    center = tuple(s // 2 for s in conn.shape)
    coords = np.argwhere(conn)
    offset = coords - center

    if not_reflexive:
        # ensure that voxel is not its own neighbor
        offset = offset[~np.all(offset == 0, axis=1)]

    return offset


def iter_neighbor(a, ijk, conn=None, offset=None, **kwargs):
    assert (offset is None) != (conn is None), 'offset xor conn required'

    if offset is None:
        offset = get_neighbor_offsets(conn, **kwargs)

    top = np.array(a.shape)
    btm = np.zeros(len(a.shape))

    for _offset in offset:
        _ijk = ijk + _offset
        if (_ijk < btm).any() or (_ijk >= top).any():
            # new _ijk is out of bounds
            continue
        yield a[*tuple(_ijk)]
