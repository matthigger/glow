import numpy as np

from .mancova import decompose, get_mancova


def permute_llr_partition(x, y, partition, n_perm=200):
    r""" llr under permutation test: n regions share same beta

    model0: all voxels follow same beta
    model1: each subset of partition has its own beta

    log p(model0) / p(model1) = - n * det(e) + \sum_i n_i det(e_i)

    where e is the error covariance (see get_mancova()) of all voxels,
    n is the total number of voxels and e_i and n_i are the values
    for subset i

    Args:
        x (np.array): (a, num_img) explanatory variables
        y (np.array): (b, num_img, num_vox) image intensities
        partition (np.array): (num_vox) region index of each voxel in y
            (agnostic to particular labels, its just the partitioning
            of voxels into subsets that we need here)
        n_perm (int): number of permutations to perform (in addition
            to the unpermuted data)

    Returns:
        llr (np.array): (n_perm + 1) log likelihood ratio.  llr[0] is
            the llr of the original data
    """
    # todo: count if too few permutations & warn as needed

    # decompose x into orthogonal spaces
    contrast = np.zeros(x.shape[0], dtype=bool)
    q_tup = decompose(x, contrast)

    num_vox = y.shape[2]

    def get_ll(y):
        """ computes log likelihood of model """
        e = get_mancova(q_tup=q_tup, y=y)[0]
        s, ll = np.linalg.slogdet(e)
        assert s != -1, 'error covariance not positive semi-definite'

        return ll

    labels = np.unique(partition)
    ll_whole = get_ll(y) * num_vox
    rng = np.random.default_rng(seed=0)
    llr_list = list()
    for perm_idx in range(n_perm):
        if perm_idx:
            # perm_idx == 0 is original data
            partition = rng.permutation(partition)

        # compute the ll sum
        term_sum = 0
        for label in labels:
            b = partition == label
            term_sum += get_ll(y[:, :, b]) * b.sum()

        llr_list.append(term_sum - ll_whole)

    return np.array(llr_list)
