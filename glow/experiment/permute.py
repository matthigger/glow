"""Permutation utilities for Freedman-Lane and partition enumeration."""

import math
import warnings
from collections import Counter

import numpy as np

from glow.analysis.mancova import decompose


def get_freed_lane(x, contrast, perm_idx):
    """build Freedman-Lane permutation matrix (standard textbook convention).

    Column-layout: ``Y_v* = P A Y_v + B Y_v`` where A = I - Q0Q0T,
    B = Q0Q0T.  I.e., compute residuals A Y_v, permute them by P, then
    add the original fitted part B Y_v back.

    Applied along glow's row-layout image axis as ``y_perm = y @ freed_lane``,
    so the matrix returned is the column-layout transpose ``M.T = A P.T + B``,
    which in index form is ``(I - Q0Q0T)[:, perm] + Q0Q0T``.

    (Previous versions returned ``[perm, :]``, which corresponds to
    permuting Y_v BEFORE projection — statistically equivalent under H0
    by row-exchangeability, but doesn't expose the O(N) survivor-kernel
    gather used by ``glow.graph.compute_llr_inner_kernel``.)

    Args:
        x (np.array): (a, num_img) design matrix
        contrast (np.array): (a,) boolean, True for features of interest
        perm_idx (int): permutation seed (0 reserved for unpermuted data)

    Returns:
        freed_lane (np.array): (num_img, num_img) Freedman-Lane matrix
    """
    assert perm_idx, 'perm_idx = 0 reserved for unpermuted data'

    q = decompose(x, contrast)
    q0 = q[0].T @ q[0]  # (num_img, num_img) nuisance projector Q0Q0T

    num_img = x.shape[1]
    rng = np.random.default_rng(perm_idx)
    perm = np.argsort(rng.permutation(num_img))

    # Match q0's dtype on np.eye so the subtraction doesn't promote a
    # float32 q0 to float64 (which would then propagate into the
    # permute einsum against y).
    return (np.eye(num_img, dtype=q0.dtype) - q0)[:, perm] + q0


class NotEnoughPermutations(Warning):
    """issued when fewer permutations exist than were requested."""
    pass


def perms_at_least(partition, thresh):
    """check whether at least thresh permutations exist.

    Args:
        partition (iter): integer partition (e.g. [0, 0, 1, 1, 2])
        thresh (float): minimum required number of permutations

    Returns:
        enough (bool): True if at least thresh permutations exist
        n_perms (int or None): exact count (only computed when not enough)
    """
    if thresh <= 1:
        return True, None
    n = len(partition)
    counts = list(Counter(partition).values())

    total = 1
    remaining = n
    for c in counts[:-1]:
        total *= math.comb(remaining, c)
        if total >= thresh:
            return True, None
        remaining -= c
    return total >= thresh, total


def get_perm_iter(partition, n_perm, seed=0):
    """yield permutations of partition.

    automatically switches between random sampling (when the number of
    possible permutations exceeds n_perm) and exhaustive enumeration.

    Args:
        partition (iter): integer partition (e.g. [0, 0, 1, 2])
        n_perm (int): number of permutations requested (in addition to
            the original, unpermuted partition which is always yielded
            first)
        seed (int): random seed for sampling

    Yields:
        partition (np.array): permuted partition
    """
    partition = np.array(partition).astype(int)
    partition_init = partition
    yield partition_init

    rng = np.random.default_rng(seed)

    enough_perms, n_perm_poss = perms_at_least(partition, thresh=n_perm)

    if enough_perms:
        for _ in range(n_perm):
            yield partition[rng.permutation(partition.size)]
    else:
        warnings.warn(f'using only {n_perm_poss} of {n_perm} requested',
                      NotEnoughPermutations, stacklevel=2)
        for partition in get_perm_iter_all(partition):
            if not np.array_equal(partition, partition_init):
                yield partition


def get_perm_iter_all(partition):
    """enumerate every distinct permutation of partition via backtracking.

    Args:
        partition (iter): integer partition (e.g. [0, 0, 1, 2])

    Yields:
        partition (np.array): each distinct permutation
    """
    symbol_count = Counter(partition)
    n = len(partition)
    path = []
    symbol_list = sorted(symbol_count.keys())

    def backtrack():
        if len(path) == n:
            yield np.array(path)
            return
        for x in symbol_list:
            if symbol_count[x] > 0:
                symbol_count[x] -= 1
                path.append(x)
                yield from backtrack()
                path.pop()
                symbol_count[x] += 1

    yield from backtrack()
