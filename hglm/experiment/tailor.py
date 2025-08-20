import warnings
from collections import Counter
from math import gamma

import numpy as np

from .mancova import decompose, get_mancova
from ..graph import get_label_map, SCGraph, GRAPH_EXCLUDE


def get_llr(label_map, exp, _q_tup=None, _skip_homo=False):
    """ gets log likelihood of homogenous vs heterogenous models

    model0 (homo): all voxels follow same beta
    model1 (hetero): each subset has its own beta

    ll = log p(model0) / p(model1) = - det(e) + \sum_i det(e_i)

    Args:
        label_map (np.array): has same shape as exp.mask_idx, defines
            partitioning of the whole region into subsets.  -1 outside
            of analysis
        exp (Experiment):
        _q_tup (tup): see regress.decompose()
        _skip_homo (bool): when False (default), subtracts -det(e).  otherwise
            ignores this (its constant across perms in get_homo_pval,
            setting to False avoids some computation)

    Returns:
        ll (float): log likelihood ratio
    """

    def _get_ll_one_reg(y, **kwargs):
        """ computes log likelihood of model """
        e = get_mancova(y=y, **kwargs)[0]
        s, ll = np.linalg.slogdet(e)
        assert s != -1, 'error covariance not positive semi-definite'

        return ll

    if _q_tup is None:
        # decompose x into orthogonal spaces (full model)
        contrast = np.zeros(exp.x.shape[0], dtype=bool)
        _q_tup = decompose(exp.x, contrast)

    # trim to relevant portion of y
    mask = label_map > -1
    idx = exp.mask_idx[mask]
    y = exp.y[:, :, idx]

    # compute the ll sum
    if _skip_homo:
        ll = -_get_ll_one_reg(y, q_tup=_q_tup)
    else:
        ll = 0

    for label in np.unique(label_map):
        if label == -1:
            continue
        b = (label_map == label)[mask]
        ll += _get_ll_one_reg(y[:, :, b], q_tup=_q_tup)

    return ll


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


def get_homo_pval(label_map, exp, n_perm):
    """ run llr homogeneity test

    where e is the error covariance (see get_mancova()) of all voxels,
    and e_i are corresponding values for subset i

    Args:
        label_map (np.array): has same shape as exp.mask_idx, defines
            partitioning of the whole region into subsets.  -1 outside
            of analysis
        exp (Experiment):
        n_perm (int): number of permutations to perform (in addition
            to the unpermuted data)

    Returns:
        pval (float): percentage of permutations (including self) which are
            more likely homogenous (low value suggests hetero)
    """
    # todo: count if too few permutations & warn as needed

    # decompose x into orthogonal spaces (full model)
    contrast = np.zeros(exp.x.shape[0], dtype=bool)
    _q_tup = decompose(exp.x, contrast)

    # split into mask (defines volume) and partition (vox to region mapping)
    mask = label_map > -1
    partition = label_map[mask]

    # compute llr
    llr = list()
    perm_iter = get_perm_iter(partition=partition, n_perm=n_perm, seed=0)
    for perm_idx, partition in enumerate(perm_iter):
        label_map[mask] = partition
        llr.append(get_llr(label_map, exp, _q_tup=_q_tup, _skip_homo=True))

    # compute p_value
    llr = np.array(llr)
    return (llr[0] >= llr).mean()


def tailor(sig_reg_list, children, exp, n_perm=300, alpha_tailor=.05):
    """ attempts to tailor to regions with all, and only, one effect

    we discard the most heterogenous (and all ancestor) nodes until none are
    classified as heterogenous (i.e. each has pval > alpha_tailor).
    reg_out greedily selects the largest remaining regions such that the
    output is disjoint (merging all the homogenous effects)

    Args:
        sig_reg_list (list): list of regions declared significant
        children (np.array): (num_reg, 2) each col are index of child
            regions
        exp (Experiment): the source data to run experiment on
        n_perm (int): number of permutations to perform (homo test per region)

    Returns:
        reg_out (np.array): index of output regions
    """
    # pre-compute: decompose x into orthogonal spaces (full model)
    contrast = np.zeros(exp.x.shape[0], dtype=bool)
    _q_tup = decompose(exp.x, contrast)

    # get subgraph corresponding to significant regions
    subgraph = SCGraph.from_children(children,
                                     num_leaf=exp.y.shape[2],
                                     subset=sig_reg_list)

    # compute pval per region
    homo_pval_dict = dict()
    for par, kid_list in subgraph.children.items():
        if not kid_list:
            # "parent" is a leaf, no partitioning to test

            continue
        label_map = get_label_map(reg_idx_list=[par] + kid_list,
                                  mask_idx=exp.mask_idx,
                                  children=children)
        homo_pval_dict[par] = get_homo_pval(label_map, exp, n_perm=n_perm)

    # sort homo_pval_dict small pvals to large
    nodes_rm = set()
    for reg_idx in sorted(homo_pval_dict.keys(), key=homo_pval_dict.get):
        if homo_pval_dict[reg_idx] > alpha_tailor:
            break
        nodes_rm |= set(subgraph.iter_ancest(node=reg_idx, incl_self=True))
    subgraph.modify(nodes_rm=nodes_rm)

    # greedily choose disjoint & max size (i.e. in subgraph, parent not in subgraph)
    no_parent = subgraph.parent == GRAPH_EXCLUDE
    reg_out_list = np.where(subgraph.included & no_parent)[0]

    return reg_out_list, homo_pval_dict
