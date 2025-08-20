import warnings
from collections import Counter
from itertools import chain
from math import gamma

import numpy as np


def get_perm_matrix(seed, num_img):
    """ gets (n x n) permutation matrix

    Returns:
        perm (np.array): (n, n) has exactly one 1 in each row and col
    """
    if seed == 0:
        # by convention, no permutation for seed=0
        return np.eye(num_img)

    rng = np.random.default_rng(seed)
    return rng.permutation(np.eye(num_img))


class Permuter:
    def __init__(self, x):
        # ensure input x is only from reduced model
        self.h = np.linalg.pinv(x) @ x

    def __call__(self, y, n_perm, perm_idx_min=0, keep_orig=False):
        """ applies freedman lane permutation testing to y

        note: n_perm includes the unpermuted data as first output
        note: if y input is 2d, assumed (b, num_img) y_perm output will
            have shape (b, num_img, n_perm)

        Args:
            y (np.array): (b, num_img, num_vox) image intensities
            n_perm (int): number of permutations
            perm_idx_min (int): smallest permutation index to use
            keep_orig (bool): if True, the first "perm_idx" output is the
                unpermuted data.

        Returns:
            y_perm (np.array): (b, num_img, num_vox, n_perm) permuted image
                intensities.  note: y_perm[..., 0] is equivilent to input y
        """
        # build freedman lane permutation matrices
        assert not (n_perm == 1 and keep_orig), 'invalid inputs, see doc'
        assert perm_idx_min > 0, 'invalid perm_idx_min (0 is for non-permuted)'
        perm_iter = range(perm_idx_min, perm_idx_min + n_perm - keep_orig)
        if keep_orig:
            # perm_idx = 0 is identity, see get_perm_matrix()
            perm_iter = chain([0, ], perm_iter)
        freed_lane = np.stack([self.get_freed_lane(idx) for idx in perm_iter],
                              axis=2)

        two_dim_input = y.ndim == 2
        if two_dim_input:
            # cast to 3d (temporarily)
            y = y[:, :, np.newaxis]

        y_perm = np.einsum('bnr,naz->barz', y, freed_lane, optimize=True)

        if two_dim_input:
            # return without 3rd dimension (input was originally two-dim)
            y_perm = y_perm[:, :, 0, :]

        return y_perm

    def get_freed_lane(self, perm_idx):
        # permute data residuals under reduced model (freedman lane)
        num_img = self.h.shape[1]
        p = get_perm_matrix(seed=perm_idx, num_img=num_img)
        return (np.eye(num_img) - self.h) @ p + self.h


class NotEnoughPermutations(Warning):
    pass


def get_n_perm_possible(partition):
    """ counts permutations possible (gamma faster than factorial)

    Args:
        partition (iter): partition (e.g. [0, 0, 0, 1, 2])
    """
    n = len(partition)
    counts = Counter(partition).values()
    return gamma(n + 1) / np.prod([gamma(c + 1) for c in counts])


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
    n_perm_poss = get_n_perm_possible(partition)
    if n_perm_poss >= n_perm:
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
