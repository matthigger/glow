"""Boolean-mask and label-map utilities: indexing, scoring, neighbours."""

import numpy as np
from sklearn.metrics import f1_score, recall_score, confusion_matrix

# structuring elements keyed by 3D connectivity (6-, 18-, 26-neighbour)
conn_dict = {6: np.array([[[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]],
                          [[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]],
                          [[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]]]),
             18: np.array([[[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]],
                           [[1, 1, 1],
                            [1, 0, 1],
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
    """Build a voxel-index array from a boolean mask.

    Args:
        mask (np.array): boolean, False where voxels are excluded

    Returns:
        mask_idx (np.array): int, same shape as mask. -1 for excluded
            voxels, otherwise a unique sequential index in 0..num_vox-1
    """
    mask = mask.astype(bool)
    mask_idx = np.full(mask.shape, fill_value=-1)
    num_vox = mask.sum()
    mask_idx[mask] = np.arange(num_vox)
    return mask_idx


def get_entropy(mask_idx) -> float:
    """Compute the entropy of a label map in bits, ignoring labels < 0.

    Args:
        mask_idx (np.array): integer label array (-1 for excluded voxels)

    Returns:
        entropy (float): Shannon entropy in bits
    """
    valid_labels = mask_idx[mask_idx >= 0]
    if valid_labels.size == 0:
        return 0.0

    value, count = np.unique(valid_labels, return_counts=True)
    prob = count / count.sum()
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)


def get_score(mask_pred, mask_target, mask_active=None) -> tuple:
    """Compute Dice, sensitivity, and specificity against ground truth.

    Args:
        mask_pred (np.array): boolean predicted support
        mask_target (np.array): boolean ground-truth support, same shape
            as mask_pred
        mask_active (np.array): boolean, the analyzed voxels. Voxels
            outside it are excluded from the comparison. Defaults to all
            voxels.

    Returns:
        dice (float): Dice / F1 overlap (0 when undefined)
        sens (float): sensitivity / recall
        spec (float): specificity (1 when undefined)
    """
    if mask_active is None:
        y_true = mask_target.flatten()
        y_pred = mask_pred.flatten()
    else:
        # restrict the comparison to the voxels that were analyzed
        y_true = mask_target[mask_active]
        y_pred = mask_pred[mask_active]

    dice = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    sens = recall_score(y_true=y_true, y_pred=y_pred, zero_division=0)

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    denom = tn + fp
    # specificity is undefined with no true negatives; report perfect (1)
    spec = 1 if denom == 0 else tn / denom

    return dice, sens, spec


def bbox_crop(arr, mask=None):
    """Crop an ndarray to the bounding box of non-zero entries.

    Args:
        arr (np.array): array of any dimensionality
        mask (np.array): boolean array selecting "active" cells. If None,
            active cells are those where arr != 0.

    Returns:
        cropped (np.array): tight slice of arr with no all-inactive border
        slices (tuple of slice): the slices applied, one per axis
    """
    if mask is None:
        mask = arr != 0
    coords = np.argwhere(mask)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    slices = tuple(slice(l, h) for l, h in zip(lo, hi))
    return arr[slices], slices


def get_neighbor_offsets(conn, not_reflexive: bool = True):
    """Compute neighbour index offsets from a connectivity mask.

    Args:
        conn (int or np.array): connectivity key (6, 18, 26) or an
            explicit structuring element
        not_reflexive (bool): exclude the zero-offset (self) entry

    Returns:
        offset (np.array): (n_neighbours, ndim) offset vectors
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


def iter_neighbor(a, ijk, conn=None, offset=None, mask_active=None, **kwargs):
    """Yield the value of a at each in-bounds neighbour of ijk.

    Exactly one of conn or offset must be given. Neighbours that fall
    outside a (or outside mask_active, when supplied) are skipped.

    Args:
        a (np.array): array to read neighbour values from
        ijk (np.array): (ndim,) integer index of the centre voxel
        conn (int or np.array): connectivity key / structuring element,
            passed to get_neighbor_offsets. XOR with offset.
        offset (np.array): (n_neighbours, ndim) precomputed offsets.
            XOR with conn.
        mask_active (np.array): boolean, same shape as a. Only voxels
            True here count as neighbours.
        **kwargs: forwarded to get_neighbor_offsets (e.g. not_reflexive).

    Yields:
        the value of a at each valid neighbour of ijk
    """
    assert (offset is None) != (conn is None), 'offset xor conn required'

    if offset is None:
        offset = get_neighbor_offsets(conn, **kwargs)

    top = np.array(a.shape)
    btm = np.zeros(len(a.shape))

    for _offset in offset:
        _ijk = ijk + _offset
        if (_ijk < btm).any() or (_ijk >= top).any():
            continue

        _ijk = tuple(_ijk)
        if (mask_active is not None) and not mask_active[*_ijk]:
            continue

        yield a[*_ijk]
