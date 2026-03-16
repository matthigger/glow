import math
import warnings

import numpy as np

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
    """Bottom-up DP with accumulated permutation penalties.

    For each SCGraph internal node with 2+ children, estimates the
    overfitting bonus of splitting via random re-partitioning:
    ``lambda(i) = max(deltas)``.  Lambdas accumulate bottom-up so that
    keeping a parent absorbs all split-penalties in its subtree::

        gain'(j) = LLR(j) + sum(lambda(i) for internal i in j's subtree)

    Then :func:`dp_antichain` runs on ``gain'`` with ``lam=0``.

    Single-child SCGraph nodes get no penalty (the DP decides on raw
    LLR alone).  Nodes too small to permute are forced as parents
    (their descendants are removed from the DP).

    Args:
        sig_reg_list (list): significant region indices
        children (np.array): (num_internal, 2) Ward children array
        stat (np.array): (num_reg,) raw statistic per region
        sizes (np.array): (num_reg,) region sizes
        exp: Experiment with y, x, contrast
        n_perm_prune (int): random re-partitions per node
        alpha_prune (float): unused (kept for interface compatibility)

    Returns:
        selected (list[int]): sorted antichain region indices
        info (dict): diagnostics
    """
    num_vox = exp.y.shape[2]
    num_reg = num_vox + children.shape[0]

    subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                     subset=sig_reg_list)

    q_tup = decompose(x=exp.x, contrast=exp.contrast)
    y = exp.y

    delta_arr = np.full(num_reg, np.nan)
    pval_homo_arr = np.full(num_reg, np.nan)
    compared_to = {}

    _leaf_cache = {}

    def _get_leaves(node):
        if node not in _leaf_cache:
            _leaf_cache[node] = np.array(sorted(
                iter_topo(children=children, num_leaf=num_vox,
                          node_start=node, only_leaf=True)), dtype=int)
        return _leaf_cache[node]

    # -- Phase 1: per-node lambda from permutations --------------------------
    lam_per_node = {}
    forced_parents = set()

    for node in sorted(sig_reg_list):
        kids = subgraph.children.get(node, [])
        if not kids:
            continue

        compared_to[node] = [int(r) for r in kids]

        if len(kids) == 1:
            lam_per_node[node] = 0.0
            delta_arr[node] = 0.0
            continue

        kid_sizes = [int(sizes[r]) for r in kids]
        parent_size = int(sizes[node])
        leftover = parent_size - sum(kid_sizes)

        min_group = min(kid_sizes)
        check_size = min(min_group, leftover) if leftover > 0 else min_group
        if (check_size > 0
                and _n_partitions(parent_size, check_size) < n_perm_prune):
            forced_parents.add(node)
            lam_per_node[node] = 0.0
            delta_arr[node] = 0.0
            continue

        parent_voxels = _get_leaves(node)

        deltas = _compute_perm_deltas(
            parent_voxels=parent_voxels,
            ac_sizes=kid_sizes,
            leftover=leftover,
            y=y,
            q_tup=q_tup,
            parent_stat=float(stat[node]),
            n_perm=n_perm_prune,
            seed=node,
        )

        lam_node = float(np.max(deltas))
        lam_per_node[node] = lam_node
        delta_arr[node] = lam_node

        # diagnostic: empirical p-value for observed split
        split_val = sum(float(stat[r]) for r in kids)
        if leftover > 0:
            ac_voxels = set()
            for r in kids:
                ac_voxels.update(_get_leaves(r).tolist())
            leftover_vox = np.array(
                [v for v in parent_voxels if v not in ac_voxels], dtype=int)
            y_left = y[:, :, leftover_vox]
            ysum_left = y_left.sum(axis=2)
            yout_left = np.einsum('bin,cin->bc', y_left, y_left)
            lo_llr = llr_from_ysum_yout(ysum_left, yout_left, leftover, q_tup)
            if np.isfinite(lo_llr):
                split_val += lo_llr

        observed_delta = split_val - float(stat[node])
        n_exceed = int(np.sum(deltas >= observed_delta))
        pval_homo_arr[node] = max(n_exceed, 1) / len(deltas)

    # -- Phase 2: remove forced-parent descendants from DP -------------------
    nodes_to_remove = set()
    for fp in forced_parents:
        for desc in subgraph.iter_desc(fp):
            nodes_to_remove.add(desc)

    dp_nodes = sorted(n for n in sig_reg_list if n not in nodes_to_remove)
    dp_node_set = set(dp_nodes)

    dp_children_map = {}
    for node in dp_nodes:
        kids = subgraph.children.get(node, [])
        dp_children_map[node] = [k for k in kids if k in dp_node_set]

    # -- Phase 3: accumulate lambdas bottom-up -------------------------------
    cum_lam = {}
    for node in dp_nodes:
        kids = dp_children_map.get(node, [])
        my_lam = lam_per_node.get(node, 0.0)
        cum_lam[node] = my_lam + sum(cum_lam.get(k, 0.0) for k in kids)

    # -- Phase 4: build penalised gain and run DP ----------------------------
    gain_prime = {}
    for node in dp_nodes:
        gain_prime[node] = float(stat[node]) + cum_lam.get(node, 0.0)

    selected, dp_raw = dp_antichain(
        nodes=dp_nodes,
        children_map=dp_children_map,
        gain=gain_prime,
        lam=0.0,
    )

    # -- Phase 5: diagnostics ------------------------------------------------
    sig_set = set(sig_reg_list)
    for r in selected:
        if r not in sig_set:
            warnings.warn(
                f'prune_perm: non-significant node {r} selected in antichain',
                stacklevel=2)

    chose = dp_raw.get('chose', {})

    kept_vs_children_arr = np.full(num_reg, np.nan)
    for node in dp_nodes:
        if dp_children_map.get(node, []):
            kept_vs_children_arr[node] = 1.0 if chose.get(node, True) else 0.0
    for node in forced_parents:
        kept_vs_children_arr[node] = 1.0

    kept_final_arr = np.full(num_reg, np.nan)
    selected_set = set(selected)
    for node in sig_reg_list:
        kept_final_arr[node] = 1.0 if node in selected_set else 0.0

    info = {
        'subgraph_children': dict(subgraph.children),
        'sig_reg_list': list(sig_reg_list),
        'best': dp_raw.get('best', {}),
        'chose': chose,
        'lam': 0.0,
        'prune_delta': delta_arr,
        'prune_pval_homo': pval_homo_arr,
        'prune_kept_vs_children': kept_vs_children_arr,
        'prune_kept_final': kept_final_arr,
        'prune_compared_to': compared_to,
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
    *ac_sizes* plus a leftover group.  All groups (including leftover)
    get their LLR computed so both sides of the comparison cover the
    full parent volume.  Returns an array of deltas:
    ``sum(all group LLRs) - parent_stat``.

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

        if leftover > 0:
            group_vox = parent_voxels[perm[offset:offset + leftover]]
            y_group = y[:, :, group_vox]
            ysum = y_group.sum(axis=2)
            yout = np.einsum('bin,cin->bc', y_group, y_group)
            llr = llr_from_ysum_yout(ysum, yout, leftover, q_tup)
            total_llr += llr if np.isfinite(llr) else 0.0

        deltas[perm_idx] = total_llr - parent_stat

    return deltas
