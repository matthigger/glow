"""Freedman-Lane permutation utility."""

import numpy as np

from glow.analysis.mancova import decompose


def _perm_indices(seed: int, num_img: int):
    """Build the index array for one Freedman-Lane permutation under seed.

    Sole source of truth for the seed-to-perm mapping: every consumer
    that needs FL permutations under a given seed (the full FL matrix
    in get_freed_lane, the per-row build loops in batched perm-LLR
    backends) routes through here so the mapping stays consistent.

    perm[k] gives the original-image index that the FL-permuted data
    puts at position k.

    Seed 0 returns the identity, holding the reserved-0 convention
    Experiment.permute and get_freed_lane already state: draw 0 is the
    observed, unpermuted data. A caller walking base_seed + i from 0 thus
    gets the observed draw in row 0 and the null in rows 1: -- what the
    GLOW arms' fits and Analysis.z_score_stat assume of a draw matrix.

    Args:
        seed (int): permutation seed; 0 gives the identity
        num_img (int): number of images

    Returns:
        perm (np.array): (num_img,) int permutation index array
    """
    if not seed:
        return np.arange(num_img)
    rng = np.random.default_rng(seed)
    return np.argsort(rng.permutation(num_img))


def get_freed_lane(x, contrast, perm_idx: int):
    """Build the Freedman-Lane permutation matrix (textbook convention).

    Column-layout: Y_v* = P A Y_v + B Y_v where A = I - Q0Q0T,
    B = Q0Q0T.  I.e., compute residuals A Y_v, permute them by P, then
    add the original fitted part B Y_v back.

    Applied along glow's row-layout image axis as y_perm = y @ freed_lane,
    so the matrix returned is the column-layout transpose M.T = A P.T + B,
    which in index form is (I - Q0Q0T)[:, perm] + Q0Q0T.

    Args:
        x (np.array): (a, num_img) design matrix
        contrast (np.array): (a,) boolean, True for features of interest
        perm_idx (int): permutation seed (0 reserved for unpermuted data)

    Returns:
        freed_lane (np.array): (num_img, num_img) Freedman-Lane matrix
    """
    assert perm_idx, 'perm_idx = 0 reserved for unpermuted data'

    q = decompose(x, contrast)
    # (num_img, num_img) nuisance projector Q0Q0T
    q0 = q[0].T @ q[0]

    num_img = x.shape[1]
    perm = _perm_indices(perm_idx, num_img)

    # Match q0's dtype on np.eye so the subtraction doesn't promote a
    # float32 q0 to float64 (which would then propagate into the
    # permute einsum against y).
    return (np.eye(num_img, dtype=q0.dtype) - q0)[:, perm] + q0
