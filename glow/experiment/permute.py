import math
import warnings
from collections import Counter

import numpy as np

from .mancova import decompose


def get_freed_lane(x, contrast, perm_idx):
    assert perm_idx, 'perm_idx = 0 reserved for unpermuted data'

    q = decompose(x, contrast)

    # build permutation matrix
    num_img = x.shape[1]
    rng = np.random.default_rng(perm_idx)
    p = rng.permutation(np.eye(num_img)).T

    # build freed_lane
    q0 = q[0].T @ q[0]
    return p @ (np.eye(num_img) - q0) + q0


class NotEnoughPermutations(Warning):
    pass


def perms_at_least(partition, thresh):
    """ computes number of permutations (if exceed n_perm, quits early)

    Returns:
        enough_perms (bool): True if there are at least thresh permutations
        num_perms (int): exact number of permutations (only computed if
            not enough permutations, else its None)
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
    """ gets permutations of partition, swaps between exhaustive or sampling

    in the case the n_perm < n_perm_possible then we'll save compute and do
    a better estimate to just go through all perms exhaustively
    (see get_perm_iter_all())

    Args:
        partition (iter): partition (e.g. [0, 0, 0, 1, 2])
        n_perm (int): number of permutations requested (in addition to
            giving the original permutation which is yielded first)
        seed (int): for RNG

    Yields:
        partition (iter): partition
    """
    # first iteration is always the original partition
    partition = np.array(partition).astype(int)
    partition_init = partition
    yield partition_init

    rng = np.random.default_rng(seed)

    enough_perms, n_perm_poss = perms_at_least(partition, thresh=n_perm)

    if enough_perms:
        # enough permutations exist, draw samples (repeats possible)
        for n_perm in range(n_perm):
            yield partition[rng.permutation(partition.size)]
    else:
        # use all available permutations, warn caller
        warnings.warn(f'using only {n_perm_poss} of {n_perm} requested',
                      NotEnoughPermutations, stacklevel=2)
        for partition in get_perm_iter_all(partition):
            if not np.array_equal(partition, partition_init):
                # avoid sending original partition again
                yield partition


def get_perm_iter_all(partition):
    """ backtracks through all permutations of partition possible

    Args:
        partition (iter): partition (e.g. [0, 0, 0, 1, 2])

    Yields:
        partition (iter): partition
    """
    symbol_count = Counter(partition)
    n = len(partition)
    path = list()
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
