"""Boolean-mask and label-map utilities: indexing, scoring, neighbours."""

import numpy as np

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


def counts_from_tp_fp(tp, fp, n_pos, n_total) -> dict:
    """Complete the confusion counts from tp / fp and the totals.

    The two producers (confusion_counts, over masks, and
    glow.graph.confusion_counts_tree, over a Ward tree) each compute tp and
    fp their own way, then share this bookkeeping: fn is the unrecovered
    target (n_pos - tp) and tn is whatever analyzed voxel is left over.
    Pure arithmetic, so tp / fp may be scalars (one mask) or per-region
    arrays (a whole tree); the returned dict matches the input type.

    Args:
        tp: true-positive count(s)
        fp: false-positive count(s)
        n_pos: target-positive voxel count (tp + fn)
        n_total: analyzed voxel count (tp + fp + tn + fn)

    Returns:
        a dict {'tp', 'fp', 'tn', 'fn'} of the same scalar / array type
    """
    fn = n_pos - tp
    tn = n_total - tp - fp - fn
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def confusion_counts(mask_pred, mask_target, mask_active=None) -> dict:
    """Confusion counts of a predicted support against ground truth.

    These four counts are the canonical detection score: every overlap
    metric (Dice, sensitivity, PPV, specificity) is a function of them,
    derived on demand via stats_from_counts. We store the counts rather
    than the metrics so any metric can be recovered downstream.

    Args:
        mask_pred (np.array): boolean predicted support
        mask_target (np.array): boolean ground-truth support, same shape
            as mask_pred
        mask_active (np.array): boolean, the analyzed voxels. Voxels
            outside it are excluded from the comparison. Defaults to all
            voxels.

    Returns:
        a dict of int counts {'tp', 'fp', 'tn', 'fn'} over the analyzed
        voxels (true/false positive/negative)
    """
    pred = mask_pred.astype(bool)
    target = mask_target.astype(bool)
    if mask_active is not None:
        # restrict the comparison to the voxels that were analyzed
        pred = pred[mask_active]
        target = target[mask_active]

    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    return counts_from_tp_fp(tp, fp, n_pos=int(target.sum()),
                             n_total=pred.size)


def stats_from_counts(tp, fp, tn, fn) -> dict:
    """Derive Dice, sensitivity, PPV, and specificity from confusion counts.

    Each metric is an elementwise function of the four counts, so the
    inputs may be numpy arrays (per-region) or pandas Series (a results
    table's columns); the matching outputs are returned in a dict of the
    same type. Bare scalars are not supported -- the 0/0 guard below
    relies on boolean indexing -- so wrap a lone count set in an array.

    A ratio is undefined only when its denominator is zero, and every
    denominator here is a sum of the counts in its numerator, so an
    undefined ratio is always a literal 0/0. We fill those with 0,
    except specificity, which is conventionally 1 when there are no
    true-negative voxels to find.

    Args:
        tp, fp, tn, fn: array-like (numpy array or pandas Series) of
            matching shape holding the confusion counts

    Returns:
        a dict {'dice', 'sens', 'ppv', 'spec'} of the same array-like type
    """
    def ratio(num, denom, fill):
        with np.errstate(divide='ignore', invalid='ignore'):
            out = num / denom
        out[denom == 0] = fill
        return out

    return {
        'dice': ratio(2 * tp, 2 * tp + fp + fn, 0.0),
        'sens': ratio(tp, tp + fn, 0.0),
        'ppv': ratio(tp, tp + fp, 0.0),
        'spec': ratio(tn, tn + fp, 1.0),
    }


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
