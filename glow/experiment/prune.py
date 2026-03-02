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
        children (np.array): (num_internal, 2) child index pairs
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


def _dp_solve(subgraph, sig_reg_list, gain_per_node, lam):
    """bottom-up DP on the significant subtree.

    Finds the antichain E maximising sum_{i in E} [gain(i) - lambda].

    gain(i) is the same-node log-likelihood ratio:
        gain(i) = LL_full(i) - LL_null(i)
    where both are computed on the same region so that the spatial
    covariance (sigma) cancels.

    Args:
        subgraph (SCGraph): short-circuited significant subtree
        sig_reg_list (list): significant region indices (ascending order)
        gain_per_node (dict): node -> same-node LLR
        lam (float): per-region penalty

    Returns:
        reg_out_list (list): sorted indices of selected effect regions
        dp_info (dict): diagnostic arrays (gain, best per node)
    """
    gain = {}
    best = {}
    chose_select = {}

    # process in ascending index order (valid topological order)
    for node in sorted(sig_reg_list):
        kids = subgraph.children.get(node, [])

        gain[node] = gain_per_node[node]
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

    dp_info = dict(gain=gain, best=best, lam=lam)
    return sorted(reg_out_list), dp_info


def _gains_from_ll(ll_full, ll_null, sig_reg_list):
    """same-node LLR gain for each significant node.

    gain(i) = LL_full(i) - LL_null(i).  Both are computed on the
    same region so that the spatial covariance (sigma) cancels.

    Returns:
        gain_per_node (dict): node -> same-node LLR
    """
    return {node: ll_full[node] - ll_null[node] for node in sig_reg_list}


def _calibrate_lambda(sig_reg_list, exp, q_tup, vox_cache,
                      n_perm, alpha):
    """calibrate lambda from Freedman-Lane permutations.

    For each permutation, computes the same-node LLR gain for every
    significant node and records the maximum gain.  Lambda is set to
    the (1 - alpha) quantile of these max-gains so that under H0
    even the single strongest false-positive region is unlikely to
    exceed the penalty.

    Args:
        sig_reg_list (list): significant region indices
        exp (Experiment): experiment data (unpermuted)
        q_tup: pre-computed QR decomposition
        vox_cache (dict): node -> voxel index array
        n_perm (int): number of permutations (excluding unpermuted)
        alpha (float): significance level for the quantile

    Returns:
        lam (float): calibrated per-region penalty
        max_gains (np.array): (n_perm,) max single-node gain per perm
    """
    max_gains = np.empty(n_perm)
    for p in range(n_perm):
        exp_perm = exp.permute(perm_idx=p + 1)  # 1-indexed (0 = unpermuted)
        ll_f_p, ll_n_p = _compute_all_ll(sig_reg_list, exp_perm.y,
                                         q_tup, vox_cache)
        gains_p = _gains_from_ll(ll_f_p, ll_n_p, sig_reg_list)
        max_gains[p] = max(gains_p.values()) if gains_p else 0.0

    lam = float(np.quantile(max_gains, 1.0 - alpha))
    return lam, max_gains


def prune_node(sig_reg_list, children, exp, n_perm=100, alpha=0.05,
               exp_eff=None, lam=None):
    """optimal pruning via DP with permutation-calibrated penalty.

    Finds the antichain E in the significant subtree that maximises:

        J(E) = sum_{i in E} [gain(i) - lambda]

    where gain(i) = LL_full(i) - LL_null(i) is the same-node LLR
    (full MANCOVA model vs null model, same region, so the spatial
    covariance sigma cancels).

    Lambda is calibrated from Freedman-Lane permutations: for each
    permutation the maximum same-node gain across all significant
    regions is recorded, and lambda is set to the (1 - alpha) quantile.
    This ensures that under H0 even the single strongest false-positive
    region is unlikely to exceed the penalty.

    If ``lam`` is provided it overrides the permutation calibration.
    If ``exp_eff`` is provided (and ``lam`` is None) it uses the
    analytic geometric-prior formula lambda = log(1 + 1/exp_eff)
    instead of permutations.

    Args:
        sig_reg_list (list): regions declared significant (via FWER)
        children (np.array): (num_internal, 2) child index pairs
        exp (Experiment): experiment data
        n_perm (int): number of calibration permutations (default 100)
        alpha (float): quantile level for calibration (default 0.05)
        exp_eff (float or None): expected number of effect regions
            (geometric-prior fallback; ignored when lam is given)
        lam (float or None): override per-region penalty

    Returns:
        reg_out_list (list): sorted indices of effect regions
        dp_info (dict): diagnostics (gain, best, lam, ...)
    """
    if not sig_reg_list:
        return [], dict(gain={}, best={}, lam=0.0,
                        gain_per_node={},
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

    # compute same-node gains on real (unpermuted) data
    ll_full, ll_null = _compute_all_ll(sig_reg_list, exp.y, q_tup,
                                       vox_cache)
    gain_per_node = _gains_from_ll(ll_full, ll_null, sig_reg_list)

    # determine lambda
    if lam is not None:
        pass  # explicit override
    elif exp_eff is not None:
        lam = np.log(1 + 1 / exp_eff)
    else:
        # permutation calibration
        lam, _ = _calibrate_lambda(sig_reg_list, exp, q_tup, vox_cache,
                                   n_perm=n_perm, alpha=alpha)

    # run DP
    reg_out_list, dp_info = _dp_solve(subgraph, sig_reg_list,
                                      gain_per_node, lam)

    # store intermediates for viewer re-computation
    dp_info['gain_per_node'] = gain_per_node
    dp_info['subgraph_children'] = dict(subgraph.children)
    dp_info['sig_reg_list'] = list(sig_reg_list)

    return reg_out_list, dp_info


def _cache_sufficient_stats(sig_reg_list, y, vox_cache):
    """cache (ysum, yout, size) per significant node.

    Args:
        sig_reg_list (list): significant region indices
        y (np.array): (b, num_img, num_vox) imaging data
        vox_cache (dict): node -> voxel index array

    Returns:
        stats (dict): node -> (ysum, yout, size)
            ysum: (b, num_img) sum of y across voxels
            yout: (b, b) sum of y_v @ y_v.T across voxels
            size: number of voxels
    """
    stats = {}
    for node in sig_reg_list:
        y_r = y[:, :, vox_cache[node]]
        n = y_r.shape[2]
        ysum = y_r.sum(axis=2)
        yr = y_r.reshape((y_r.shape[0], -1), order='F')
        yout = yr @ yr.T
        stats[node] = [ysum.copy(), yout.copy(), n]
    return stats


def _null_ll_from_stats(ysum, yout, n, q0):
    """null-model log-likelihood from sufficient statistics.

    Computes -(n/2) * log|det((E+H)/n)| where
    E + H = yout - ysum @ q0.T @ q0 @ ysum.T / n.

    Args:
        ysum (np.array): (b, num_img)
        yout (np.array): (b, b)
        n (int): number of voxels
        q0 (np.array): nuisance subspace from decompose()

    Returns:
        ll (float): null-model profile log-likelihood
    """
    e_plus_h = yout - ysum @ q0.T @ q0 @ ysum.T / n
    return _loglik_from_cov(e_plus_h, n)


def _leaf_depths(subgraph, sig_reg_list):
    """depth of each SCGraph leaf (1 = isolated root, deeper = more ancestors).

    Args:
        subgraph (SCGraph): significant subtree
        sig_reg_list (list): significant region indices

    Returns:
        depths (dict): leaf_node -> depth (int >= 1)
    """
    depths = {}

    def _walk(node, d):
        kids = subgraph.children.get(node, [])
        if not kids:
            depths[node] = d
        else:
            for k in kids:
                _walk(k, d + 1)

    for node in sorted(sig_reg_list):
        if subgraph.parent[node] == GRAPH_EXCLUDE:
            _walk(node, 1)
    return depths


def _depth_weights(subgraph, sig_reg_list):
    """per-node weight so every voxel contributes to D likelihood terms.

    Internal nodes get weight 1.  SCGraph leaves at depth d(l) get
    weight 1 + D - d(l) where D = max depth across all leaves.

    Args:
        subgraph (SCGraph): significant subtree
        sig_reg_list (list): significant region indices

    Returns:
        weights (dict): node -> float weight
        depths (dict): leaf_node -> depth
    """
    depths = _leaf_depths(subgraph, sig_reg_list)
    D = max(depths.values()) if depths else 1
    weights = {}
    for node in sig_reg_list:
        if node in depths:
            weights[node] = 1 + D - depths[node]
        else:
            weights[node] = 1
    return weights, depths


def _apply_effect(node, stats, q_tup, subgraph):
    """adjust sufficient stats after selecting node as an effect.

    Removes the Q1 projection of node's y_bar from:
      - the node itself and all descendants (all voxels shifted)
      - all ancestors (only the V_node voxels shifted)

    Args:
        node: selected effect node index
        stats (dict): node -> [ysum, yout, size] (mutated in-place)
        q_tup: (q0, q1, q2) from decompose()
        subgraph (SCGraph): significant subtree
    """
    ysum_i, yout_i, n_i = stats[node]
    f = (ysum_i / n_i) @ q_tup[1].T @ q_tup[1]

    # snapshot ysum_i before mutating (needed for ancestor adjustment)
    ysum_i_orig = ysum_i.copy()

    # node itself + descendants: all n_d voxels lie inside V_i
    for d in [node] + list(subgraph.iter_desc(node)):
        ysum_d, yout_d, n_d = stats[d]
        stats[d][1] = yout_d - ysum_d @ f.T - f @ ysum_d.T + n_d * (f @ f.T)
        stats[d][0] = ysum_d - n_d * f

    # ancestors: only the n_i voxels in V_i are shifted
    for a in subgraph.iter_ancest(node):
        ysum_a, yout_a, n_a = stats[a]
        stats[a][1] = yout_a - ysum_i_orig @ f.T - f @ ysum_i_orig.T + n_i * (f @ f.T)
        stats[a][0] = ysum_a - n_i * f


def _weighted_cost(sig_reg_list, stats, q0, weights):
    """weighted sum of null-model LL across all significant nodes."""
    return sum(
        weights[node] * _null_ll_from_stats(stats[node][0], stats[node][1],
                                            stats[node][2], q0)
        for node in sig_reg_list
    )


def prune_tree(sig_reg_list, children, exp):
    """Disjoint greedy pruning by tree-wide adjusted null-model LL.

    Each iteration evaluates every remaining candidate by temporarily
    removing its Q1 projection from itself, its descendants, and its
    ancestors, then computing the weighted sum of null-model LLs
    across all remaining nodes.  The candidate that maximises this
    tree-wide cost is selected; it and all its relatives (ancestors
    and descendants) are then discarded from the pool.

    After discarding, scores must be recomputed because disjoint
    candidates can share ancestors with the discarded set, changing
    the set of nodes in the sum.  However, no sufficient-statistic
    recomputation is needed: discarded nodes are the only ones whose
    stats would change, and remaining nodes' stats are untouched.

    Shallow SCGraph leaves are upweighted so every voxel contributes
    to the same number of likelihood terms.

    Args:
        sig_reg_list (list): regions declared significant (via FWER)
        children (np.array): (num_internal, 2) child index pairs
        exp (Experiment): experiment data

    Returns:
        reg_out_list (list): sorted indices of selected effect regions
        info (dict): diagnostics (gain_per_node, sig_reg_list,
            subgraph_children, cost_history, weights)
    """
    if not sig_reg_list:
        return [], dict(gain_per_node={}, sig_reg_list=[],
                        subgraph_children={}, cost_history=[],
                        weights={}, wt_gain_sum_passes=[])

    num_vox = exp.y.shape[2]

    subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                     subset=sig_reg_list)
    q_tup = decompose(exp.x, exp.contrast)
    vox_cache = _build_vox_cache(sig_reg_list, children, num_vox)
    stats = _cache_sufficient_stats(sig_reg_list, exp.y, vox_cache)
    weights, _ = _depth_weights(subgraph, sig_reg_list)

    ll_full, ll_null = _compute_all_ll(sig_reg_list, exp.y, q_tup,
                                       vox_cache)
    gain_per_node = _gains_from_ll(ll_full, ll_null, sig_reg_list)

    q0 = q_tup[0]

    # pre-compute weighted null LL once per node (stats never change)
    wll_null = {node: weights[node] * _null_ll_from_stats(
                    stats[node][0], stats[node][1], stats[node][2], q0)
                for node in sig_reg_list}

    remaining = set(sig_reg_list)
    baseline = sum(wll_null[j] for j in remaining)
    cost_history = [baseline]
    selected = []
    wt_gain_sum_passes = []  # list of dicts, one per greedy iteration

    while remaining:
        best_node = None
        best_cost = baseline
        pass_deltas = {}

        for node in remaining:
            affected = ([node]
                        + list(subgraph.iter_desc(node))
                        + list(subgraph.iter_ancest(node)))
            affected_remaining = [a for a in affected if a in remaining]
            backup = {a: (stats[a][0].copy(), stats[a][1].copy(),
                          stats[a][2])
                      for a in affected if a in stats}

            _apply_effect(node, stats, q_tup, subgraph)

            delta = sum(
                weights[a] * _null_ll_from_stats(
                    stats[a][0], stats[a][1], stats[a][2], q0)
                - wll_null[a]
                for a in affected_remaining)
            pass_deltas[node] = delta
            new_cost = baseline + delta

            if new_cost > best_cost:
                best_cost = new_cost
                best_node = node

            for a, (ys, yo, n) in backup.items():
                stats[a][0] = ys
                stats[a][1] = yo
                stats[a][2] = n

        wt_gain_sum_passes.append(pass_deltas)

        if best_node is None:
            break

        selected.append(best_node)
        cost_history.append(best_cost)

        to_discard = ({best_node}
                      | set(subgraph.iter_desc(best_node))
                      | set(subgraph.iter_ancest(best_node)))
        remaining -= to_discard
        baseline = sum(wll_null[j] for j in remaining)

    info = dict(
        gain_per_node=gain_per_node,
        sig_reg_list=list(sig_reg_list),
        subgraph_children=dict(subgraph.children),
        cost_history=cost_history,
        weights=weights,
        wt_gain_sum_passes=wt_gain_sum_passes,
    )
    return sorted(selected), info


def prune_tree_dp(sig_reg_list, children, exp, exp_eff):
    """DP-optimal pruning using tree-wide adjusted-LL gains.

    For each significant node i, computes delta_i: the change in the
    weighted sum of null-model log-likelihoods across the entire tree
    when node i's Q1 effect is subtracted from itself and all relatives.
    These tree-wide gains are then passed to the standard DP
    solver with a geometric-prior penalty lambda = log(1 + 1/exp_eff).

    The DP solution is approximately optimal for the full tree-wide cost
    because the cross-terms between disjoint effects at shared ancestors
    are empirically negligible (<0.2%).

    Args:
        sig_reg_list (list): regions declared significant (via FWER)
        children (np.array): (num_internal, 2) child index pairs
        exp (Experiment): experiment data
        exp_eff (float): expected number of effect regions (required);
            sets lambda = log(1 + 1/exp_eff)

    Returns:
        reg_out_list (list): sorted indices of selected effect regions
        info (dict): diagnostics (gain, best, lam, gain_per_node,
            tree_wide_gain, sig_reg_list, subgraph_children, weights)
    """
    if not sig_reg_list:
        return [], dict(gain={}, best={}, lam=0.0,
                        gain_per_node={}, tree_wide_gain={},
                        sig_reg_list=[], subgraph_children={},
                        weights={})

    num_vox = exp.y.shape[2]

    subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                     subset=sig_reg_list)
    q_tup = decompose(exp.x, exp.contrast)
    vox_cache = _build_vox_cache(sig_reg_list, children, num_vox)
    stats = _cache_sufficient_stats(sig_reg_list, exp.y, vox_cache)
    weights, _ = _depth_weights(subgraph, sig_reg_list)

    ll_full, ll_null = _compute_all_ll(sig_reg_list, exp.y, q_tup,
                                       vox_cache)
    gain_per_node = _gains_from_ll(ll_full, ll_null, sig_reg_list)

    q0 = q_tup[0]

    wll_null = {node: weights[node] * _null_ll_from_stats(
                    stats[node][0], stats[node][1], stats[node][2], q0)
                for node in sig_reg_list}

    sig_set = set(sig_reg_list)

    tree_wide_gain = {}
    for node in sig_reg_list:
        affected = ([node]
                    + list(subgraph.iter_desc(node))
                    + list(subgraph.iter_ancest(node)))
        affected_in = [a for a in affected if a in sig_set]
        backup = {a: (stats[a][0].copy(), stats[a][1].copy(), stats[a][2])
                  for a in affected if a in stats}

        _apply_effect(node, stats, q_tup, subgraph)

        delta = sum(
            weights[a] * _null_ll_from_stats(
                stats[a][0], stats[a][1], stats[a][2], q0)
            - wll_null[a]
            for a in affected_in)
        tree_wide_gain[node] = delta

        for a, (ys, yo, n) in backup.items():
            stats[a][0] = ys
            stats[a][1] = yo
            stats[a][2] = n

    lam = np.log(1 + 1 / exp_eff)

    reg_out_list, dp_info = _dp_solve(subgraph, sig_reg_list,
                                      tree_wide_gain, lam)

    dp_info['gain_per_node'] = gain_per_node
    dp_info['tree_wide_gain'] = tree_wide_gain
    dp_info['sig_reg_list'] = list(sig_reg_list)
    dp_info['subgraph_children'] = dict(subgraph.children)
    dp_info['weights'] = weights

    return reg_out_list, dp_info
