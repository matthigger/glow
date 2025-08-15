from collections import defaultdict

import numpy as np

import hglm.graph
from .mancova import decompose, get_mancova


def permute_llr_partition(x, y, partition, n_perm=200):
    r""" llr under permutation test: n regions share same beta

    model0: all voxels follow same beta
    model1: each subset of partition has its own beta

    log p(model0) / p(model1) = - det(e) + \sum_i det(e_i)

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

    def get_ll(y):
        """ computes log likelihood of model """
        e = get_mancova(q_tup=q_tup, y=y)[0]
        s, ll = np.linalg.slogdet(e)
        assert s != -1, 'error covariance not positive semi-definite'

        return ll

    labels = np.unique(partition)
    ll_whole = get_ll(y)
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
            term_sum += get_ll(y[:, :, b])

        llr_list.append(term_sum - ll_whole)

    return np.array(llr_list)


def tailor(sig_reg_list, children, exp, alpha_tailor, n_perm, _pval_dict=None):
    """ attempts to tailor to regions with all, and only, one effect

    Args:
        sig_reg_list (list): list of regions declared significant
        children (np.array): (num_reg, 2) each col are index of child
            regions
        exp (Experiment): the source data to run experiment on
        alpha_tailor (float): threshold at which a prune event happens (a
            significant region and all its ancestors are discarded)
        n_perm (int): number of permutations to perform

    Returns:
        reg_out_list (list): index of prune regions
        pval_homo_dict (dict): keys are region index, values are the pval
            of that region being homogenous (small=hetero)
    """
    assert 1 / n_perm <= alpha_tailor, \
        'inconsistent n_perm_tailor & alpha_tailor: all merge no prune'

    sig_reg_list = list(np.sort(sig_reg_list))

    # build parent representation of graph
    num_vox = exp.y.shape[2]
    parent_all = hglm.graph.get_parent(children, num_leaf=num_vox)

    def get_sig_parent(reg_idx):
        """Yield parent nodes of reg_idx, stopping at -1"""
        while True:
            reg_idx = parent_all[reg_idx]
            if reg_idx == -1:
                return None
            elif reg_idx in sig_reg_list:
                return reg_idx

    # List of all significant immediate descendants of a significant node.
    # "Immediate" means the largest nested region directly below it:
    # e.g., if region 1 ⊂ region 2 ⊂ region 3, and all are significant,
    # then region 2 is considered a significant child of reg 3 — not reg 1
    sig_kid_dict = defaultdict(list)
    for reg_idx in sig_reg_list:
        _parent = get_sig_parent(reg_idx)
        if _parent is not None:
            sig_kid_dict[_parent].append(reg_idx)

    def get_pval(parent, kid_list):
        """ run homogeneity test (small pval = hetero) """
        # mask is zero for not included voxels or constituent region
        # index for corresponding voxels.  voxels in the parent but
        # not in any significant kid have value of parent
        mask = hglm.graph.get_mask(reg_idx=parent,
                                   mask_idx=exp.mask_idx,
                                   children=children) * parent
        for reg_idx in kid_list:
            # replace voxels in mask with constituent regions
            _mask = hglm.graph.get_mask(reg_idx=reg_idx,
                                        mask_idx=exp.mask_idx,
                                        children=children)
            mask[_mask] = reg_idx

        # permutation test: is the parent homogenous?
        b = mask.astype(bool)
        _b = exp.mask_idx[b]
        llr_list = permute_llr_partition(x=exp.x,
                                         y=exp.y[:, :, _b],
                                         partition=mask[b],
                                         n_perm=n_perm)
        return (llr_list[0] >= llr_list).mean()

    # start with leaf nodes & apply all necessary merge operations
    if _pval_dict is None:
        _pval_dict = dict()
    reg_out_set = set(sig_reg_list) - set(sig_kid_dict.keys())
    for parent, kid_list in sorted(sig_kid_dict.items()):
        if not reg_out_set.issuperset(kid_list):
            # some child deemed heterogeneous, parent pruned as a result
            continue

        if parent not in _pval_dict:
            _pval_dict[parent] = get_pval(parent, kid_list)

        if _pval_dict[parent] >= alpha_tailor:
            # region is homogenous: merge (remove kids, add parent)
            reg_out_set -= set(kid_list)
            reg_out_set.add(parent)

    return sorted(reg_out_set), _pval_dict
