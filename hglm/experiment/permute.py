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
