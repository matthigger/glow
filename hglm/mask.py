import numpy as np
from sklearn.metrics import f1_score, recall_score, confusion_matrix


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

    # compute scores
    f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    sens = recall_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    spec = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])

    return f1, sens, spec
