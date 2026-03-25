import numpy as np

from ..graph import get_parent, dp_antichain, SCGraph, iter_topo, GRAPH_EXCLUDE


def prune_greedy(sig_reg_list, children, stat):
    """Greedy LLR pruning: iteratively pick the highest-LLR significant
    region, remove all ancestors and descendants, repeat.

    Args:
        sig_reg_list (list[int]): regions declared significant (via FWER)
        children (np.array): (num_internal, 2) Ward child-index pairs
        stat (np.array): 1-D raw LLR per region

    Returns:
        selected (list[int]): sorted region indices (disjoint antichain)
        info (dict): diagnostic key ``sig_reg_list``
    """
    if not sig_reg_list:
        return [], dict(sig_reg_list=[])

    num_vox = len(stat) - children.shape[0]
    parent = get_parent(children, num_vox)

    def _ancestors(node):
        out = set()
        p = parent[node]
        while p != -1:
            out.add(p)
            p = parent[p]
        return out

    def _descendants(node):
        out = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n >= num_vox:
                c0, c1 = children[n - num_vox]
                out.add(c0)
                out.add(c1)
                stack.append(c0)
                stack.append(c1)
        return out

    candidates = sorted(sig_reg_list, key=lambda r: stat[r], reverse=True)
    removed = set()
    selected = []

    for reg in candidates:
        if reg in removed or stat[reg] <= 0:
            continue
        selected.append(reg)
        removed.add(reg)
        removed |= _ancestors(reg)
        removed |= _descendants(reg)

    return sorted(selected), dict(sig_reg_list=list(sig_reg_list))


def prune_dp(sig_reg_list, children, stat, lam=0.0, exp_n_eff=None):
    """DP antichain pruning: find the antichain maximising total gain.

    Builds a short-circuited subgraph of significant regions, then
    runs bottom-up DP to find the globally optimal antichain that
    maximises ``sum(stat[i] - lam)`` over all valid antichains.

    With ``lam=0`` (and ``exp_n_eff=None``), this selects the antichain
    that maximises total LLR — a strict improvement over the greedy
    approximation.

    Args:
        sig_reg_list (list[int]): regions declared significant (via FWER)
        children (np.array): (num_internal, 2) Ward child-index pairs
        stat (np.array): 1-D statistic per region (e.g. raw LLR)
        lam (float): per-region penalty.  Ignored when *exp_n_eff* is
            provided.
        exp_n_eff (float | None): expected number of effect regions under
            the geometric prior.  When given, overrides *lam* with
            ``log(1 + 1/exp_n_eff)``.

    Returns:
        selected (list[int]): sorted region indices (disjoint antichain)
        info (dict): diagnostic keys ``sig_reg_list``, ``best``, ``chose``
    """
    if not sig_reg_list:
        return [], dict(sig_reg_list=[], best={}, chose={})

    if exp_n_eff is not None:
        lam = np.log(1.0 + 1.0 / exp_n_eff)

    num_vox = len(stat) - children.shape[0]
    subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                     subset=sig_reg_list)

    selected, raw = dp_antichain(
        nodes=sorted(sig_reg_list),
        children_map=subgraph.children,
        gain=stat,
        lam=lam,
    )

    info = dict(sig_reg_list=list(sig_reg_list), **raw)
    return selected, info


# ---------------------------------------------------------------------------
# Helpers for prune_greedy_full_adjust
# ---------------------------------------------------------------------------

def _loglik_from_cov(cov, n):
    """Gaussian profile log-likelihood from a covariance matrix.

    Returns -(n/2) * log|det(cov/n)|.  Additive constants (that
    cancel in all ratios) are omitted.
    """
    s, logdet = np.linalg.slogdet(cov / n)
    assert s != -1, 'covariance not positive semi-definite'
    return -0.5 * logdet * n


def _null_ll_from_stats(ysum, yout, n, q0):
    """Null-model log-likelihood from sufficient statistics.

    Computes -(n/2) * log|det((E+H)/n)| where
    E + H = yout - ysum @ q0.T @ q0 @ ysum.T / n.
    """
    e_plus_h = yout - ysum @ q0.T @ q0 @ ysum.T / n
    return _loglik_from_cov(e_plus_h, n)


def _build_vox_cache(sig_reg_list, children, num_vox):
    """Cache voxel indices for each significant node."""
    vox_cache = {}
    for node in sig_reg_list:
        vox_cache[node] = np.array(list(iter_topo(
            children=children, num_leaf=num_vox,
            node_start=node, only_leaf=True)))
    return vox_cache


def _cache_sufficient_stats(sig_reg_list, y, vox_cache):
    """Cache (ysum, yout, size) per significant node.

    Returns:
        stats (dict): node -> [ysum, yout, size]
            ysum: (b, num_img) sum of y across voxels
            yout: (b, b) outer product sum
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


def _leaf_depths(subgraph, sig_reg_list):
    """Depth of each SCGraph leaf (1 = isolated root)."""
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
    """Per-node weight so every voxel contributes to the same number of
    likelihood terms.

    Internal nodes get weight 1.  SCGraph leaves at depth d(l) get
    weight 1 + D - d(l) where D = max depth across all leaves.
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
    """Adjust sufficient stats after selecting *node* as an effect.

    Removes the Q1 projection of node's y_bar from:
      - the node itself and all descendants (all voxels shifted)
      - all ancestors (only the V_node voxels shifted)
    """
    ysum_i, yout_i, n_i = stats[node]
    f = (ysum_i / n_i) @ q_tup[1].T @ q_tup[1]

    ysum_i_orig = ysum_i.copy()

    for d in [node] + list(subgraph.iter_desc(node)):
        ysum_d, yout_d, n_d = stats[d]
        stats[d][1] = yout_d - ysum_d @ f.T - f @ ysum_d.T + n_d * (f @ f.T)
        stats[d][0] = ysum_d - n_d * f

    for a in subgraph.iter_ancest(node):
        ysum_a, yout_a, n_a = stats[a]
        stats[a][1] = yout_a - ysum_i_orig @ f.T - f @ ysum_i_orig.T + n_i * (f @ f.T)
        stats[a][0] = ysum_a - n_i * f


# ---------------------------------------------------------------------------
# prune_greedy_full_adjust
# ---------------------------------------------------------------------------

def prune_greedy_full_adjust(sig_reg_list, children, exp):
    """Disjoint greedy pruning by tree-wide adjusted null-model LL.

    Each iteration evaluates every remaining candidate by temporarily
    removing its Q1 projection from itself, its descendants, and its
    ancestors, then computing the weighted sum of null-model LLs
    across all remaining nodes.  The candidate that maximises this
    tree-wide cost is selected; it and all its relatives (ancestors
    and descendants) are then discarded from the pool.

    Args:
        sig_reg_list (list[int]): regions declared significant (via FWER)
        children (np.array): (num_internal, 2) Ward child-index pairs
        exp: Experiment with ``.y``, ``.x``, ``.contrast``, ``.mask_idx``

    Returns:
        selected (list[int]): sorted region indices (disjoint antichain)
        info (dict): diagnostic keys ``sig_reg_list``,
            ``cost_history``, ``weights``
    """
    from .mancova import decompose

    if not sig_reg_list:
        return [], dict(sig_reg_list=[], cost_history=[], weights={})

    num_vox = exp.y.shape[2]

    subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                     subset=sig_reg_list)
    q_tup = decompose(exp.x, exp.contrast)
    vox_cache = _build_vox_cache(sig_reg_list, children, num_vox)
    stats = _cache_sufficient_stats(sig_reg_list, exp.y, vox_cache)
    weights, _ = _depth_weights(subgraph, sig_reg_list)

    q0 = q_tup[0]

    wll_null = {node: weights[node] * _null_ll_from_stats(
                    stats[node][0], stats[node][1], stats[node][2], q0)
                for node in sig_reg_list}

    remaining = set(sig_reg_list)
    baseline = sum(wll_null[j] for j in remaining)
    cost_history = [baseline]
    selected = []

    while remaining:
        best_node = None
        best_cost = baseline

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
            new_cost = baseline + delta

            if new_cost > best_cost:
                best_cost = new_cost
                best_node = node

            for a, (ys, yo, n) in backup.items():
                stats[a][0] = ys
                stats[a][1] = yo
                stats[a][2] = n

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
        sig_reg_list=list(sig_reg_list),
        cost_history=cost_history,
        weights=weights,
    )
    return sorted(selected), info
