"""Greedy pruning of significant Ward regions to a disjoint antichain."""

import numpy as np

from ..graph import get_parent


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
