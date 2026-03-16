import math
import warnings

import numpy as np
from scipy.stats import t as t_dist

from ..graph import SCGraph, dp_antichain, iter_topo
from .mancova import decompose, llr_from_ysum_yout


def prune(sig_reg_list, children, stat, sizes=None, exp=None,
          exp_eff=None, lam=None,
          n_perm_prune=25, alpha_prune=0.05,
          get_stat=None):
    """Optimal pruning via DP on the significant subtree.

    Three modes (checked in order):

    1. ``lam`` provided: constant penalty per region.
    2. ``exp_eff`` provided: geometric-prior penalty
       ``lam = log(1 + 1/exp_eff)``.
    3. Neither (default): permutation-based per-node penalty via
       :func:`prune_perm`.  Requires ``sizes`` and ``exp``.

    Args:
        sig_reg_list (list): regions declared significant (via FWER)
        children (np.array): (num_internal, 2) child index pairs
        stat (np.array): 1-D statistic per region (raw LLR for
            permutation mode; can be llr_adjusted for legacy modes)
        sizes (np.array or None): 1-D region sizes (required for
            permutation mode)
        exp: Experiment object (required for permutation mode)
        exp_eff (float or None): expected number of effect regions
            (geometric-prior penalty; ignored when lam is given)
        lam (float or None): explicit constant penalty override
        n_perm_prune (int): random re-partitions per node
            (permutation mode only)
        alpha_prune (float): family-wise prune error rate
            (permutation mode only)
        get_stat: unused, kept for interface compatibility

    Returns:
        reg_out_list (list): sorted indices of effect regions
        dp_info (dict): diagnostics
    """
    if not sig_reg_list:
        return [], dict(gain={}, best={}, lam=0.0,
                        subgraph_children={},
                        sig_reg_list=[])

    num_vox = len(stat) - children.shape[0]

    if lam is not None or exp_eff is not None:
        # legacy constant-lambda or geometric-prior mode
        subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                         subset=sig_reg_list)
        if lam is not None:
            pass
        elif exp_eff is not None:
            lam = np.log(1 + 1 / exp_eff)

        reg_out_list, dp_info = dp_antichain(
            nodes=sorted(sig_reg_list),
            children_map=subgraph.children,
            gain=stat,
            lam=lam,
        )
        dp_info['subgraph_children'] = dict(subgraph.children)
        dp_info['sig_reg_list'] = list(sig_reg_list)
        return reg_out_list, dp_info

    # permutation-based pruning (default)
    if sizes is None or exp is None:
        raise ValueError(
            'permutation pruning (default) requires sizes and exp. '
            'Pass exp_eff or lam to use a constant penalty instead.')

    return prune_perm(
        sig_reg_list=sig_reg_list,
        children=children,
        stat=stat,
        sizes=sizes,
        exp=exp,
        n_perm_prune=n_perm_prune,
        alpha_prune=alpha_prune,
    )


def prune_perm(sig_reg_list, children, stat, sizes, exp,
               n_perm_prune=25, alpha_prune=0.05):
    """Bottom-up DP with on-the-fly permutation penalties.

    For each SCGraph internal node, determines the antichain below it,
    estimates the overfitting bonus of that partition via random
    re-partitioning, fits a t-distribution to the deltas, and sets
    lambda as the Bonferroni-corrected quantile.

    Lambda is used locally for the parent-vs-children comparison and
    then discarded (not propagated upward).

    Args:
        sig_reg_list (list): significant region indices
        children (np.array): (num_internal, 2) Ward children array
        stat (np.array): (num_reg,) raw statistic per region
        sizes (np.array): (num_reg,) region sizes
        exp: Experiment with y, x, contrast
        n_perm_prune (int): random re-partitions per node
        alpha_prune (float): family-wise prune error rate

    Returns:
        selected (list[int]): sorted antichain region indices
        info (dict): diagnostics including prune_delta, prune_lambda,
            prune_pval arrays
    """
    num_vox = exp.y.shape[2]
    num_reg = num_vox + children.shape[0]

    subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                     subset=sig_reg_list)

    q_tup = decompose(x=exp.x, contrast=exp.contrast)
    y = exp.y

    # count internal SCGraph nodes for Bonferroni correction
    num_internal = sum(1 for n in sig_reg_list
                       if subgraph.children.get(n, []))
    alpha_per_node = alpha_prune / max(num_internal, 1)

    # per-node diagnostic arrays
    delta_arr = np.full(num_reg, np.nan)
    pval_arr = np.full(num_reg, np.nan)

    # bottom-up DP
    best = {}
    chose = {}
    antichain = {}
    _leaf_cache = {}

    def _get_leaves(node):
        if node not in _leaf_cache:
            _leaf_cache[node] = np.array(sorted(
                iter_topo(children=children, num_leaf=num_vox,
                          node_start=node, only_leaf=True)), dtype=int)
        return _leaf_cache[node]

    for node in sorted(sig_reg_list):
        kids = subgraph.children.get(node, [])

        if not kids:
            best[node] = float(stat[node])
            chose[node] = True
            antichain[node] = [node]
            continue

        # collect the resolved antichain below this node
        child_ac = []
        for kid in kids:
            child_ac.extend(antichain[kid])

        split_val = sum(float(stat[r]) for r in child_ac)

        # group sizes for permutation test
        ac_sizes = [int(sizes[r]) for r in child_ac]
        parent_size = int(sizes[node])
        leftover = parent_size - sum(ac_sizes)

        # check if enough distinct partitions exist
        min_group = min(ac_sizes) if ac_sizes else 0
        if (len(ac_sizes) == 1
                and leftover > 0
                and _n_partitions(parent_size, min(min_group, leftover))
                    < n_perm_prune):
            # too few distinct re-partitions; collapse to parent
            best[node] = float(stat[node])
            chose[node] = True
            antichain[node] = [node]
            delta_arr[node] = 0.0
            continue

        parent_voxels = _get_leaves(node)

        deltas = _compute_perm_deltas(
            parent_voxels=parent_voxels,
            ac_sizes=ac_sizes,
            leftover=leftover,
            y=y,
            q_tup=q_tup,
            parent_stat=float(stat[node]),
            n_perm=n_perm_prune,
            seed=node,
        )

        # fit t-distribution
        mu = float(np.mean(deltas))
        sigma = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0

        if sigma > 0:
            df = len(deltas) - 1
            lam_node = mu + sigma * t_dist.ppf(1 - alpha_per_node, df)
        else:
            lam_node = mu

        delta_arr[node] = lam_node

        # observed split p-value
        observed_delta = split_val - float(stat[node])
        if sigma > 0:
            pval_arr[node] = float(
                t_dist.cdf((float(stat[node]) - split_val - mu) / sigma, df))

        # local comparison: lambda helps parent, then is discarded
        keep_val = float(stat[node]) + lam_node
        if keep_val >= split_val:
            best[node] = float(stat[node])
            chose[node] = True
            antichain[node] = [node]
        else:
            best[node] = split_val
            chose[node] = False
            antichain[node] = list(child_ac)

    # collect final antichain from SCGraph roots
    all_children = set()
    for kids in subgraph.children.values():
        all_children.update(kids)
    roots = sorted(n for n in sig_reg_list if n not in all_children)

    selected = []
    for root in roots:
        selected.extend(antichain[root])
    selected = sorted(selected)

    # warn if any non-significant node was selected (should not happen)
    sig_set = set(sig_reg_list)
    for r in selected:
        if r not in sig_set:
            warnings.warn(
                f'prune_perm: non-significant node {r} selected in antichain',
                stacklevel=2)

    # compute accumulated lambda (diagnostic only)
    lambda_arr = np.full(num_reg, np.nan)
    for node in sorted(sig_reg_list):
        kids = subgraph.children.get(node, [])
        own_delta = delta_arr[node] if np.isfinite(delta_arr[node]) else 0.0
        child_sum = sum(
            lambda_arr[k] for k in kids if np.isfinite(lambda_arr[k]))
        lambda_arr[node] = own_delta + child_sum

    info = {
        'subgraph_children': dict(subgraph.children),
        'sig_reg_list': list(sig_reg_list),
        'best': best,
        'chose': chose,
        'lam': 0.0,
        'prune_delta': delta_arr,
        'prune_lambda': lambda_arr,
        'prune_pval': pval_arr,
    }
    return selected, info


def _n_partitions(n, k):
    """Number of ways to choose k items from n (C(n, k)).

    Returns math.inf when too large to matter.
    """
    try:
        return math.comb(n, k)
    except (ValueError, OverflowError):
        return math.inf


def _compute_perm_deltas(parent_voxels, ac_sizes, leftover,
                         y, q_tup, parent_stat, n_perm, seed=0):
    """Compute permutation deltas for one parent node.

    Randomly re-partitions *parent_voxels* into groups matching
    *ac_sizes* (significant groups that get full-model LLR) plus a
    leftover group (LLR = 0).  Returns an array of deltas:
    ``sum(group LLRs) - parent_stat``.

    Args:
        parent_voxels (np.array): 1-D voxel indices for the parent
        ac_sizes (list[int]): sizes of the antichain regions
        leftover (int): number of leftover (non-significant) voxels
        y (np.array): (b, num_img, num_vox) full imaging data
        q_tup: QR decomposition from decompose()
        parent_stat (float): parent's raw LLR
        n_perm (int): number of random re-partitions
        seed (int): RNG seed for reproducibility

    Returns:
        np.array: (n_perm,) delta values
    """
    n = len(parent_voxels)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_perm)

    for perm_idx in range(n_perm):
        perm = rng.permutation(n)
        total_llr = 0.0
        offset = 0

        for group_size in ac_sizes:
            group_vox = parent_voxels[perm[offset:offset + group_size]]
            y_group = y[:, :, group_vox]
            ysum = y_group.sum(axis=2)
            yout = np.einsum('bin,cin->bc', y_group, y_group)
            llr = llr_from_ysum_yout(ysum, yout, group_size, q_tup)
            total_llr += llr if np.isfinite(llr) else 0.0
            offset += group_size

        # leftover group gets LLR = 0 (not computed)
        deltas[perm_idx] = total_llr - parent_stat

    return deltas
