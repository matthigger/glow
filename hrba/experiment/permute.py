from copy import copy

import numpy as np

from hrba.graph import iter_edge


def permute_eps_from_exp(*, exp, **kwargs):
    return permute_eps(x=exp.x, contrast=exp.contrast, y=exp.y, **kwargs)


def permute_eps(x, contrast, y, children, n_permute, negative_seed=False):
    """ gets error cov under permutations (freedman lane) for all reg in tree

    Args:
        x (np.array): (a, num_img) explanatory variables
        contrast (np.array): (a) True for each corresponding feature in x which
            is "of interest" (other x features form the reduced model in
            computing f statistic)
        y (np.array): (b, num_img, num_vox) image intensities
        children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
            sklearn.cluster.Ward.children_)
        n_permute (int): number of permutations
        negative_seed (bool): if True, seeds are non positive integers (0,
            -1, -2 ...) helpful option to avoid same permutation

    Returns:
        eps (np.array): (n_permute, num_reg, 2, b, b) error covariances.
            the 2 corresponds to reduced (0) and full (1) models
    """

    # compute freedman lane transform (y_perm = y @ ((I - H0)P + H0)
    b, num_img, num_vox = y.shape
    x = x[~contrast, :], x
    h = np.stack([np.linalg.pinv(_x) @ _x for _x in x], axis=2)
    freed_lane = np.empty(shape=(num_img, num_img, n_permute))
    i = np.eye(num_img)
    if negative_seed:
        # negatives not allowed as rng seeds, we overflow to positive
        seed_all = np.arange(0, -n_permute, -1).astype(np.uint)
    else:
        seed_all = range(n_permute)
    for idx, seed in enumerate(seed_all):
        p = get_perm_matrix(seed=seed, num_img=num_img)
        freed_lane[:, :, idx] = (i - h[:, :, 0]) @ p + h[:, :, 0]

    # compute size of each region
    num_reg = num_vox + children.shape[0]
    size = np.ones(num_reg, dtype=int)
    size[num_vox:] = 0
    for child, parent in iter_edge(children=children, num_vox=num_vox):
        size[parent] += size[child]

    # compute y_mean for all regions (not just single voxel regions)
    y_mean_all = np.zeros((b, num_img, num_reg))
    y_mean_all[:, :, :num_vox] = y
    for child, parent in iter_edge(children=children, num_vox=num_vox):
        # add child's contribution to mean y (not yet normalized)
        lam = size[child] / size[parent]
        y_mean_all[:, :, parent] += y_mean_all[:, :, child] * lam

    # compute middle of final term (freed_lane @ h @ freed_lane.T)
    # phpt has shape (num_img, num_img, n_perm, 2)
    phpt = np.einsum('abc,bdf,edc->aecf', freed_lane, h, freed_lane,
                     optimize='greedy')

    # compute last_term = y_mean @ h @ y_mean.T
    # last_term has shape (
    last_term = np.einsum('cfb,fgae,dgb->abecd', y_mean_all, phpt, y_mean_all,
                          optimize='greedy')

    # compute mean y outer product for all single voxel regions (under all
    # permutations)
    myo = np.zeros(shape=(n_permute, num_reg, b, b))
    myo[:, :num_vox, :, :] = \
        np.einsum('fcb,cda,eda,geb->abfg', y, freed_lane, freed_lane, y,
                  optimize='greedy')

    # compute mean y outer product for multi voxel regions
    for child, parent in iter_edge(children=children, num_vox=num_vox):
        lam = size[child] / size[parent]
        myo[:, parent, :, :] += myo[:, child, :, :] * lam

    # myo is constant across models
    last_term[:, :, 0, :, :] -= myo
    last_term[:, :, 1, :, :] -= myo

    return -last_term / num_img


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
