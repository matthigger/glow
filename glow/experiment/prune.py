import math
import warnings

import numpy as np

from ..graph import SCGraph, dp_antichain, iter_topo
from .mancova import decompose, llr_from_ysum_yout


def prune(sig_reg_list, children, stat, sizes=None, exp=None,
          exp_eff=3, lam=None, n_perm_prune=25,
          get_stat=None, **kwargs):
    """Optimal pruning via DP on the significant subtree.

    Two modes (checked in order):

    1. ``lam`` provided: constant penalty per region (legacy).
    2. Otherwise (default): hybrid permutation + geometric-prior
       penalty via :func:`prune_perm`.  Requires ``sizes`` and ``exp``.
       Each split must exceed `mu + (k-1)*lam_geom` where mu is the
       mean permutation delta (data-dependent overfitting correction)
       and lam_geom = log(1 + 1/exp_eff).

    Args:
        sig_reg_list (list): regions declared significant (via FWER)
        children (np.array): (num_internal, 2) child index pairs
        stat (np.array): 1-D statistic per region (raw LLR)
        sizes (np.array or None): 1-D region sizes (required for
            default mode)
        exp: Experiment object (required for default mode)
        exp_eff (float): expected number of effect regions for the
            geometric prior (default 3)
        lam (float or None): explicit constant penalty override
            (bypasses permutation test entirely)
        n_perm_prune (int): random re-partitions per node
            (default mode only)
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

    if lam is not None or sizes is None or exp is None:
        if lam is None:
            lam = np.log(1 + 1 / exp_eff)
        subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                         subset=sig_reg_list)
        reg_out_list, dp_info = dp_antichain(
            nodes=sorted(sig_reg_list),
            children_map=subgraph.children,
            gain=stat,
            lam=lam,
        )
        dp_info['subgraph_children'] = dict(subgraph.children)
        dp_info['sig_reg_list'] = list(sig_reg_list)
        return reg_out_list, dp_info

    return prune_perm(
        sig_reg_list=sig_reg_list,
        children=children,
        stat=stat,
        sizes=sizes,
        exp=exp,
        exp_eff=exp_eff,
        n_perm_prune=n_perm_prune,
    )


def prune_perm(sig_reg_list, children, stat, sizes, exp,
               exp_eff=3, n_perm_prune=25):
    """Bottom-up DP with hybrid permutation + geometric-prior penalty.

    At each internal SCGraph node the decision is local:

        split if  (observed_delta - mu) > (k - 1) * lam_geom

    where *mu* is the mean of the permutation-delta distribution
    (data-dependent overfitting correction), *k* is the number of
    regions in the resolved antichain below this node, and
    *lam_geom = ln(1 + 1/exp_eff)* is the per-region geometric-prior
    cost.

    Equivalently: keep if  stat[node] + mu + (k-1)*lam_geom >= split_val.

    Args:
        sig_reg_list (list): significant region indices
        children (np.array): (num_internal, 2) Ward children array
        stat (np.array): (num_reg,) raw statistic per region
        sizes (np.array): (num_reg,) region sizes
        exp: Experiment with y, x, contrast
        exp_eff (float): expected number of effect regions (default 3)
        n_perm_prune (int): random re-partitions per node (default 25)

    Returns:
        selected (list[int]): sorted antichain region indices
        info (dict): diagnostics
    """
    num_vox = exp.y.shape[2]
    num_reg = num_vox + children.shape[0]

    lam_geom = np.log(1 + 1 / exp_eff)

    subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                     subset=sig_reg_list)

    q_tup = decompose(x=exp.x, contrast=exp.contrast)
    y = exp.y

    delta_arr = np.full(num_reg, np.nan)
    pval_homo_arr = np.full(num_reg, np.nan)
    kept_vs_children_arr = np.full(num_reg, np.nan)
    compared_to = {}

    _leaf_cache = {}

    def _get_leaves(node):
        if node not in _leaf_cache:
            _leaf_cache[node] = np.array(sorted(
                iter_topo(children=children, num_leaf=num_vox,
                          node_start=node, only_leaf=True)), dtype=int)
        return _leaf_cache[node]

    def _leftover_llr(parent_voxels, child_ac, leftover):
        """Compute LLR for the leftover voxels not in any child."""
        if leftover <= 0:
            return 0.0
        ac_voxels = set()
        for r in child_ac:
            ac_voxels.update(_get_leaves(r).tolist())
        leftover_vox = np.array(
            [v for v in parent_voxels if v not in ac_voxels], dtype=int)
        y_left = y[:, :, leftover_vox]
        ysum = y_left.sum(axis=2)
        yout = np.einsum('bin,cin->bc', y_left, y_left)
        llr = llr_from_ysum_yout(ysum, yout, leftover, q_tup)
        return llr if np.isfinite(llr) else 0.0

    # bottom-up DP
    best = {}
    chose = {}
    antichain = {}

    for node in sorted(sig_reg_list):
        kids = subgraph.children.get(node, [])

        if not kids:
            best[node] = float(stat[node])
            chose[node] = True
            antichain[node] = [node]
            continue

        # resolved antichain below this node
        child_ac = []
        for kid in kids:
            child_ac.extend(antichain[kid])

        compared_to[node] = [int(r) for r in child_ac]
        k = len(child_ac)

        # observed split value (children + leftover)
        ac_sizes = [int(sizes[r]) for r in child_ac]
        parent_size = int(sizes[node])
        leftover = parent_size - sum(ac_sizes)

        parent_voxels = _get_leaves(node)
        split_val = (sum(float(stat[r]) for r in child_ac)
                     + _leftover_llr(parent_voxels, child_ac, leftover))

        # small-region guard: not enough distinct partitions
        min_group = min(ac_sizes) if ac_sizes else 0
        check_size = min(min_group, leftover) if leftover > 0 else min_group
        if (check_size > 0
                and _n_partitions(parent_size, check_size) < n_perm_prune):
            best[node] = float(stat[node])
            chose[node] = True
            antichain[node] = [node]
            delta_arr[node] = 0.0
            kept_vs_children_arr[node] = 1.0
            continue

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

        mu = float(np.mean(deltas))
        penalty = (k - 1) * lam_geom
        delta_arr[node] = mu

        observed_delta = split_val - float(stat[node])
        n_exceed = int(np.sum(deltas >= observed_delta))
        pval_homo_arr[node] = max(n_exceed, 1) / len(deltas)

        keep_val = float(stat[node]) + mu + penalty
        if keep_val >= split_val:
            best[node] = float(stat[node])
            chose[node] = True
            antichain[node] = [node]
            kept_vs_children_arr[node] = 1.0
        else:
            best[node] = split_val
            chose[node] = False
            antichain[node] = list(child_ac)
            kept_vs_children_arr[node] = 0.0

    # collect final antichain from SCGraph roots
    all_children = set()
    for kids in subgraph.children.values():
        all_children.update(kids)
    roots = sorted(n for n in sig_reg_list if n not in all_children)

    selected = []
    for root in roots:
        selected.extend(antichain[root])
    selected = sorted(selected)

    sig_set = set(sig_reg_list)
    for r in selected:
        if r not in sig_set:
            warnings.warn(
                f'prune_perm: non-significant node {r} selected',
                stacklevel=2)

    kept_final_arr = np.full(num_reg, np.nan)
    selected_set = set(selected)
    for node in sig_reg_list:
        kept_final_arr[node] = 1.0 if node in selected_set else 0.0

    info = {
        'subgraph_children': dict(subgraph.children),
        'sig_reg_list': list(sig_reg_list),
        'best': best,
        'chose': chose,
        'lam': lam_geom,
        'exp_eff': exp_eff,
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
