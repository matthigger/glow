import numpy as np

from ..graph import get_parent


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
