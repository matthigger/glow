import warnings

import numpy as np

from glow.experiment.permute import NotEnoughPermutations
from .mancova import decompose, get_mancova
from .permute import get_perm_iter
from ..graph import get_label_map, get_parent, iter_topo, SCGraph, GRAPH_EXCLUDE


def get_llr(label_map, exp, _q_tup=None, _skip_homo=False):
    r"""log-likelihood ratio of homogeneous vs heterogeneous models.

    model0 (homo): all voxels share the same beta.
    model1 (hetero): each subset has its own beta.

    Args:
        label_map (np.array): same shape as exp.mask_idx, partitioning
            voxels into subsets (-1 outside analysis)
        exp (Experiment): experiment data
        _q_tup: pre-computed QR decomposition (see decompose())
        _skip_homo (bool): skip the homogeneous term (constant across
            permutations in get_homo_pval)

    Returns:
        ll (float): log-likelihood ratio
    """

    def _get_ll_one_reg(y, **kwargs):
        """log-likelihood contribution of a single region."""
        e = get_mancova(y=y, **kwargs)[0]
        num_vox = y.shape[2]
        s, ll = np.linalg.slogdet(e / num_vox)
        assert s != -1, 'error covariance not positive semi-definite'

        return ll * num_vox

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


def get_homo_pval(label_map, exp, n_perm):
    """permutation p-value for the homogeneity test.

    a low p-value suggests the region is heterogeneous (contains
    more than one distinct effect).

    Args:
        label_map (np.array): same shape as exp.mask_idx, partitioning
            voxels into subsets (-1 outside analysis)
        exp (Experiment): experiment data
        n_perm (int): number of permutations (in addition to unpermuted)

    Returns:
        pval (float): fraction of permutations at least as extreme
    """
    # decompose x into orthogonal spaces (full model)
    contrast = np.zeros(exp.x.shape[0], dtype=bool)
    _q_tup = decompose(exp.x, contrast)

    # split into mask (defines volume) and partition (vox to region mapping)
    mask = label_map > -1
    partition = label_map[mask]

    # compute llr
    llr = list()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=NotEnoughPermutations)
        perm_iter = get_perm_iter(partition=partition, n_perm=n_perm, seed=0)
        for perm_idx, partition in enumerate(perm_iter):
            label_map[mask] = partition
            llr.append(get_llr(label_map, exp, _q_tup=_q_tup, _skip_homo=True))

    # compute p_value
    llr = np.array(llr)
    return (llr[0] >= llr).mean()


def prune(sig_reg_list, children, exp, n_perm=300, alpha_prune=.05):
    """prune significant regions to a disjoint, homogeneous set.

    discards heterogeneous nodes (and their ancestors) until every
    remaining node passes the homogeneity test, then greedily selects
    the largest disjoint regions.

    Args:
        sig_reg_list (list): regions declared significant
        children (np.array): (num_node, 2) child index pairs
        exp (Experiment): experiment data
        n_perm (int): permutations for the homogeneity test

    Returns:
        reg_out_list (np.array): indices of output regions
        homo_pval_dict (dict): reg_idx -> homogeneity p-value
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
        if homo_pval_dict[reg_idx] > alpha_prune:
            break
        nodes_rm |= set(subgraph.iter_ancest(node=reg_idx, incl_self=True))
    subgraph.modify(nodes_rm=nodes_rm)

    # greedily choose disjoint & max size (i.e. in subgraph, parent not in subgraph)
    no_parent = subgraph.parent == GRAPH_EXCLUDE
    reg_out_list = np.where(subgraph.included & no_parent)[0]

    return reg_out_list, homo_pval_dict


def prune_greedy(sig_reg_list, children, num_leaf, stat_adj):
    """Greedy pruning: select significant regions by largest adjusted stat.

    Iteratively picks the significant region with the highest adjusted
    statistic, adds it to the output, and removes all regions that
    share voxels with it (ancestors and descendants in the hierarchy).

    Args:
        sig_reg_list (list): region indices declared significant
        children (np.array): (num_leaf - 1, 2) child index pairs
        num_leaf (int): number of leaf nodes (voxels)
        stat_adj (np.array): (num_reg,) adjusted statistic for perm 0

    Returns:
        reg_out_list (list): indices of selected regions (sorted)
    """
    if not len(sig_reg_list):
        return []

    # build parent lookup and descendant sets
    parent = get_parent(children, num_leaf)

    # for each significant region, collect all descendants
    sig_set = set(sig_reg_list)
    desc = {}  # reg_idx -> set of descendants (including self)
    for reg in sig_set:
        desc[reg] = set(iter_topo(children=children, num_leaf=num_leaf,
                                  node_start=reg))

    # for each significant region, collect all ancestors
    anc = {}
    for reg in sig_set:
        ancestors = set()
        node = reg
        while True:
            p = parent[node]
            if p == -1:
                break
            ancestors.add(p)
            node = p
        anc[reg] = ancestors

    # sort significant regions by adjusted stat (descending)
    candidates = sorted(sig_set, key=lambda r: stat_adj[r], reverse=True)

    selected = []
    removed = set()
    for reg in candidates:
        if reg in removed:
            continue
        selected.append(reg)
        # remove all regions that share voxels (ancestors + descendants)
        overlap = (desc[reg] | anc[reg]) & sig_set
        removed |= overlap

    return sorted(selected)
