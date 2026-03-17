import numpy as np

from ..graph import SCGraph, dp_antichain


def prune_greedy(sig_reg_list, children, stat):
    """Select the LLR-maximising antichain from the significant subtree.

    Builds the short-circuited subgraph of significant regions and runs
    a bottom-up DP (via :func:`dp_antichain` with ``lam=0``) that picks
    the antichain maximising total raw LLR.

    Args:
        sig_reg_list (list[int]): regions declared significant (via FWER)
        children (np.array): (num_internal, 2) Ward child-index pairs
        stat (np.array): 1-D raw LLR per region

    Returns:
        selected (list[int]): sorted region indices in the antichain
        dp_info (dict): diagnostic keys ``sig_reg_list``,
            ``subgraph_children``, ``best``, ``chose``
    """
    if not sig_reg_list:
        return [], dict(sig_reg_list=[], subgraph_children={},
                        best={}, chose={})

    num_vox = len(stat) - children.shape[0]
    subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                     subset=sig_reg_list)

    selected, raw = dp_antichain(
        nodes=sorted(sig_reg_list),
        children_map=subgraph.children,
        gain=stat,
        lam=0.0,
    )

    dp_info = {
        'sig_reg_list': list(sig_reg_list),
        'subgraph_children': dict(subgraph.children),
        'best': raw['best'],
        'chose': raw['chose'],
    }
    return selected, dp_info
