"""Prune significant Ward regions to a disjoint antichain.

Each rule turns the FWER-significant region set into a disjoint antichain
(no selected region an ancestor of another), differing in what it maximizes:

  - prune_greedy blooms the highest-LLR region and removes its tree
    relatives, repeating; fast, GLOW's default, but only locally optimal.
  - prune_dp takes the globally optimal max-total-LLR cut via dp_antichain,
    a bottom-up dynamic program over the significant-region subgraph.
  - prune_oracle takes the antichain of largest Dice against a known target
    support. It is handed the answer, so it is a ceiling rather than a
    method: the Dice this fit's significant set still has in it, whatever
    the ranking. Benchmark use only.
"""

import numpy as np

from ..graph import get_fp_tp, get_parent, SCGraph

# prune_oracle's Dinkelbach loop: a Dice gain this small is a fixed point.
# The iteration converges superlinearly, so the cap only guards against a
# pathological cycle -- a few rounds is the norm.
_DICE_TOL = 1e-9
_DICE_MAX_ITER = 32


def prune_greedy(sig_reg_list: list, children, stat) -> tuple:
    """Prune significant regions greedily by descending LLR.

    Iteratively picks the highest-LLR significant region, removes all of
    its ancestors and descendants, and repeats. The result is a disjoint
    antichain (no selected region is an ancestor of another).

    Args:
        sig_reg_list (list): int regions declared significant (via FWER)
        children (np.array): (num_internal, 2) Ward child-index pairs
        stat (np.array): (num_reg,) raw LLR per region

    Returns:
        selected (list): sorted int region indices (disjoint antichain)
        info (dict): diagnostic, with key sig_reg_list
    """
    if not sig_reg_list:
        return [], dict(sig_reg_list=[])

    num_vox = len(stat) - children.shape[0]
    parent = get_parent(children, num_vox)

    def _ancestors(node):
        """Return the set of strict ancestors of node in the tree."""
        out = set()
        p = parent[node]
        while p != -1:
            out.add(p)
            p = parent[p]
        return out

    def _descendants(node):
        """Return the set of strict descendants of node in the tree."""
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


def prune_dp(sig_reg_list: list, children, stat, lam: float = 0.0,
             exp_n_eff: float = None) -> tuple:
    """Prune significant regions to the exact max-total-LLR tree cut.

    Builds a short-circuited subgraph over just the significant regions and
    runs dp_antichain over it, selecting the disjoint antichain that
    maximizes sum(stat[r] - lam). With lam=0 (and exp_n_eff=None) this is
    the unpenalized max-likelihood cut -- globally optimal, unlike greedy,
    but prone to oversegment. A positive lam (or a geometric prior via
    exp_n_eff) penalizes each bloomed region, favoring fewer, larger ones.

    Args:
        sig_reg_list (list): int regions declared significant (via FWER)
        children (np.array): (num_internal, 2) Ward child-index pairs
        stat (np.array): (num_reg,) raw LLR per region
        lam (float): per-region penalty; ignored when exp_n_eff is given
        exp_n_eff (float | None): expected number of effect regions under a
            geometric prior; when given, overrides lam with log(1 + 1/n)

    Returns:
        selected (list): sorted int region indices (disjoint antichain)
        info (dict): diagnostic, keys sig_reg_list, best, chose
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

    return selected, dict(sig_reg_list=list(sig_reg_list), **raw)


def _dice(reg_list, tp, size, n_target: float) -> float:
    """Return the Dice of a disjoint region set against the target.

    The regions are disjoint (an antichain), so the union's counts are the
    sums of theirs: with T recovered target voxels and P selected voxels
    against a target of G, Dice = 2 tp / (2 tp + fp + fn) = 2 T / (P + G).

    Args:
        reg_list: selected region indices; an empty selection scores 0.
        tp (np.array): (num_reg,) target voxels per region
        size (np.array): (num_reg,) voxels per region
        n_target (float): voxels in the target support

    Returns:
        float: the Dice of the union of reg_list.
    """
    if len(reg_list) == 0:
        return 0.0
    idx = np.asarray(reg_list, dtype=int)
    denom = size[idx].sum() + n_target
    return 0.0 if denom <= 0 else float(2.0 * tp[idx].sum() / denom)


def prune_oracle(sig_reg_list: list, children, mask_target, mask_idx) -> tuple:
    """Prune significant regions to the antichain of largest Dice.

    The oracle rule: of every disjoint antichain of the significant set, the
    one whose union best matches a known target support. It is handed the
    ground truth, so it measures headroom -- what a perfect selector could
    still win off this fit -- rather than being a method one could run on
    real data (see the module docstring).

    Exact, not a search. Dice over a disjoint selection is 2T/(P + G), a
    ratio of two sums over the selected regions, so maximizing it is a
    linear-fractional program over the tree's antichains. Dinkelbach's
    iteration (Dinkelbach 1967) reduces it to a handful of max-total-gain
    antichains: at the current Dice d, the antichain maximizing
    sum(2*tp - d*size) either beats d -- and becomes the next d -- or proves
    d optimal. Each round is one exact dp_antichain over the same
    short-circuited significant subgraph prune_dp runs on, started from the
    best single region so every round climbs from a feasible point.

    Args:
        sig_reg_list (list): int regions declared significant (via FWER)
        children (np.array): (num_internal, 2) Ward child-index pairs
        mask_target (np.array): (X, Y, Z) boolean target support, the
            planted effect to match; voxels outside the analysis are
            ignored
        mask_idx (np.array): (X, Y, Z) int voxel-index array (-1 outside
            the analysis)

    Returns:
        selected (list): sorted int region indices (disjoint antichain)
        info (dict): diagnostic, keys sig_reg_list and dice (the attained
            Dice of the selection)
    """
    mask_active = mask_idx > -1
    mask_target = np.asarray(mask_target, dtype=bool) & mask_active
    n_target = float(mask_target.sum())
    if not sig_reg_list or n_target == 0:
        return [], dict(sig_reg_list=list(sig_reg_list), dice=0.0)

    fp, tp = get_fp_tp(mask_target, mask_idx, children)
    size = tp + fp

    num_vox = int(mask_active.sum())
    subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                     subset=list(sig_reg_list))
    nodes = sorted(int(reg) for reg in sig_reg_list)

    # the best single region: feasible, so the iteration only ever climbs
    idx = np.asarray(nodes, dtype=int)
    dice_reg = 2.0 * tp[idx] / (size[idx] + n_target)
    best = int(np.argmax(dice_reg))
    selected, dice = [int(idx[best])], float(dice_reg[best])

    for _ in range(_DICE_MAX_ITER):
        candidate, _ = dp_antichain(nodes=nodes,
                                    children_map=subgraph.children,
                                    gain=2.0 * tp - dice * size, lam=0.0)
        dice_candidate = _dice(candidate, tp, size, n_target)
        if dice_candidate <= dice + _DICE_TOL:
            break
        selected, dice = candidate, dice_candidate

    return sorted(selected), dict(sig_reg_list=list(sig_reg_list), dice=dice)


def dp_antichain(nodes, children_map, gain, lam=0.0):
    """Max-gain antichain of a tree via bottom-up dynamic programming.

    Maximizes sum_{i in B} (gain[i] - lam) over every antichain B of the
    tree given by children_map (a node set, none an ancestor of another).
    Each node either blooms whole -- contributing its net gain -- or defers
    to the best disjoint selection within its children, whichever sums
    higher; a node always has the option of contributing 0, so negative net
    gains are dropped. prune_dp calls this on the significant-region
    subgraph to find the exact max-total-LLR tree cut.

    Args:
        nodes (list): node indices in ascending (bottom-up) order, so each
            node's children are processed before the node itself
        children_map (dict): node -> list of its child nodes (within
            nodes); a node absent or mapped to [] is a leaf of this tree
        gain (np.array | dict): node -> gain value (e.g. raw LLR)
        lam (float | dict): per-region penalty; a scalar applied to every
            node, or a node -> penalty dict

    Returns:
        selected (list): sorted node indices forming the optimal antichain
        info (dict): diagnostics, keys best (node -> subtree optimum) and
            chose (node -> bloomed-whole bool)
    """
    _lam_is_dict = isinstance(lam, dict)
    best = {}
    chose = {}

    for node in nodes:
        kids = children_map.get(node, [])
        lam_node = lam[node] if _lam_is_dict else lam
        g_net = gain[node] - lam_node

        if not kids:
            best[node] = max(g_net, 0.0)
            chose[node] = g_net > 0
        else:
            split_val = sum(best[k] for k in kids)
            best[node] = max(g_net, split_val)
            chose[node] = g_net >= split_val

    all_children = set()
    for kids in children_map.values():
        all_children.update(kids)
    roots = sorted(n for n in nodes if n not in all_children)

    selected = []

    def _bt(node):
        if chose[node]:
            selected.append(node)
        else:
            for kid in children_map.get(node, []):
                _bt(kid)

    for root in roots:
        _bt(root)

    return sorted(selected), dict(best=best, chose=chose)
