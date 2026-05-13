import numpy as np

from ..graph import get_parent, dp_antichain, SCGraph


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
