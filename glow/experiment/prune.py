import warnings

import numpy as np

from glow.experiment.permute import NotEnoughPermutations
from .mancova import decompose, get_mancova
from .permute import get_perm_iter
from ..graph import get_label_map, get_parent, iter_topo, SCGraph, GRAPH_EXCLUDE


def _loglik_from_cov(cov, n):
    """Gaussian profile log-likelihood from a covariance matrix.

    Returns -(n/2) * log|det(cov/n)|.  Additive constants (that
    cancel in all ratios) are omitted.

    Args:
        cov (np.array): (b, b) un-normalised covariance (e.g. E or E+H)
        n (int): number of voxels

    Returns:
        ll (float): profile log-likelihood (higher = better fit)
    """
    s, logdet = np.linalg.slogdet(cov / n)
    assert s != -1, 'covariance not positive semi-definite'
    return -0.5 * logdet * n


def _region_loglik(y, q_tup):
    """Gaussian profile log-likelihood for a single region.

    Returns -(n/2) * log|det(E/n)| where E is the MANCOVA error
    matrix.

    Args:
        y (np.array): (b, num_img, n_vox) imaging data for a region
        q_tup: pre-computed QR decomposition from decompose()

    Returns:
        ll (float): profile log-likelihood (higher = better fit)
    """
    e = get_mancova(y=y, q_tup=q_tup)[0]
    return _loglik_from_cov(e, y.shape[2])


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
        ll (float): log-likelihood ratio (in -2*LL scale)
    """
    if _q_tup is None:
        # decompose x into orthogonal spaces (full model)
        contrast = np.zeros(exp.x.shape[0], dtype=bool)
        _q_tup = decompose(exp.x, contrast)

    # trim to relevant portion of y
    mask = label_map > -1
    idx = exp.mask_idx[mask]
    y = exp.y[:, :, idx]

    # compute the ll sum (in -2*LL scale for backward compat)
    if _skip_homo:
        ll = 2 * _region_loglik(y, q_tup=_q_tup)
    else:
        ll = 0

    for label in np.unique(label_map):
        if label == -1:
            continue
        b = (label_map == label)[mask]
        ll -= 2 * _region_loglik(y[:, :, b], q_tup=_q_tup)

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


def _compute_region_ll(y_region, q_tup):
    """log-likelihood under the full and null MANCOVA models.

    Full model: regress on interest + nuisance features (error = E).
    Null model: regress on nuisance only (error = E + H).

    Args:
        y_region (np.array): (b, num_img, n_vox) imaging data for a region
        q_tup: pre-computed QR decomposition from decompose(x, contrast)

    Returns:
        ll_full (float): -(n/2) * log|det(E / n)|
        ll_null (float): -(n/2) * log|det((E + H) / n)|
    """
    e, h, _ = get_mancova(y=y_region, q_tup=q_tup)
    n = y_region.shape[2]
    return _loglik_from_cov(e, n), _loglik_from_cov(e + h, n)


def _build_vox_cache(sig_reg_list, children, num_vox):
    """cache voxel indices for each significant node.

    Args:
        sig_reg_list (list): significant region indices
        children (np.array): (num_vox - 1, 2) child index pairs
        num_vox (int): number of leaf nodes (voxels)

    Returns:
        vox_cache (dict): node -> np.array of leaf (voxel) indices
    """
    vox_cache = {}
    for node in sig_reg_list:
        vox_cache[node] = np.array(list(iter_topo(
            children=children, num_leaf=num_vox,
            node_start=node, only_leaf=True)))
    return vox_cache


def _compute_all_ll(sig_reg_list, y, q_tup, vox_cache):
    """compute LL_1 and LL_0 for every significant node.

    Args:
        sig_reg_list (list): significant region indices
        y (np.array): (b, num_img, num_vox) imaging data
        q_tup: pre-computed QR decomposition
        vox_cache (dict): node -> voxel index array

    Returns:
        ll_full (dict): node -> LL_1
        ll_null (dict): node -> LL_0
    """
    ll_full = {}
    ll_null = {}
    for node in sig_reg_list:
        y_region = y[:, :, vox_cache[node]]
        ll_full[node], ll_null[node] = _compute_region_ll(y_region, q_tup)
    return ll_full, ll_null


def _dp_solve(subgraph, sig_reg_list, ll_full, ll_null, lam):
    """bottom-up DP on the significant subtree.

    Finds the antichain E maximising sum_{i in E} [gain(i) - lambda].

    Args:
        subgraph (SCGraph): short-circuited significant subtree
        sig_reg_list (list): significant region indices (ascending order)
        ll_full (dict): node -> LL_1
        ll_null (dict): node -> LL_0
        lam (float): per-region penalty

    Returns:
        reg_out_list (list): sorted indices of selected effect regions
        dp_info (dict): diagnostic arrays (null_sum, gain, best per node)
    """
    null_sum = {}
    gain = {}
    best = {}
    chose_select = {}

    # process in ascending index order (valid topological order)
    for node in sorted(sig_reg_list):
        kids = subgraph.children.get(node, [])

        if not kids:
            # leaf of significant subtree
            null_sum[node] = ll_null[node]
        else:
            null_sum[node] = sum(null_sum[k] for k in kids)

        gain[node] = ll_full[node] - null_sum[node]
        g_net = gain[node] - lam

        if not kids:
            # leaf: select this region or leave null
            best[node] = max(g_net, 0.0)
            chose_select[node] = g_net > 0
        else:
            # internal: select this region or recurse into children
            split_val = sum(best[k] for k in kids)
            best[node] = max(g_net, split_val)
            chose_select[node] = g_net >= split_val

    # backtrack to recover E
    reg_out_list = []

    def _backtrack(node):
        if chose_select[node]:
            reg_out_list.append(node)
        else:
            for kid in subgraph.children.get(node, []):
                _backtrack(kid)

    # start from roots of significant subtree
    for node in sorted(sig_reg_list):
        if subgraph.parent[node] == GRAPH_EXCLUDE:
            _backtrack(node)

    dp_info = dict(null_sum=null_sum, gain=gain, best=best, lam=lam)
    return sorted(reg_out_list), dp_info


def prune_dp(sig_reg_list, children, exp, exp_eff=1, lam=None):
    """optimal pruning via DP with geometric prior.

    Finds the antichain E in the significant subtree that maximises
    the MAP objective under a geometric prior on K >= 0 effect regions:

        J(E) = sum_{i in E} [gain(i) - lambda]

    where gain(i) = LL_1(i) - null_sum(i) is the log-likelihood ratio
    of the full model (effect present) vs the null model (nuisance only)
    accumulated over the leaves of i's subtree.

    The per-region penalty lambda is derived from a geometric prior
    with expected number of effects exp_eff:

        lambda = log(1 + 1 / exp_eff)

    Args:
        sig_reg_list (list): regions declared significant (via FWER)
        children (np.array): (num_vox - 1, 2) child index pairs
        exp (Experiment): experiment data
        exp_eff (float): expected number of effect regions under the
            geometric prior (default 1).  higher values yield a
            smaller per-region penalty and thus more regions.
        lam (float or None): override per-region penalty.  if given,
            exp_eff is ignored.

    Returns:
        reg_out_list (list): sorted indices of effect regions
        dp_info (dict): diagnostics (null_sum, gain, best, lam)
    """
    if not sig_reg_list:
        return [], dict(null_sum={}, gain={}, best={}, lam=0.0,
                        ll_full={}, ll_null={},
                        subgraph_children={},
                        sig_reg_list=[])

    num_vox = exp.y.shape[2]

    # build significant subtree
    subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                     subset=sig_reg_list)

    # pre-compute QR decomposition (reused everywhere)
    q_tup = decompose(exp.x, exp.contrast)

    # cache voxel indices per significant node
    vox_cache = _build_vox_cache(sig_reg_list, children, num_vox)

    # compute LL_1 and LL_0 on real (unpermuted) data
    ll_full, ll_null = _compute_all_ll(sig_reg_list, exp.y, q_tup,
                                       vox_cache)

    # derive lambda from the geometric prior (if not provided)
    if lam is None:
        lam = np.log(1 + 1 / exp_eff)

    # run DP
    reg_out_list, dp_info = _dp_solve(subgraph, sig_reg_list,
                                      ll_full, ll_null, lam)

    # store intermediates for viewer re-computation
    dp_info['ll_full'] = ll_full
    dp_info['ll_null'] = ll_null
    dp_info['subgraph_children'] = dict(subgraph.children)
    dp_info['sig_reg_list'] = list(sig_reg_list)

    return reg_out_list, dp_info


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
