import numpy as np

from ..graph import SCGraph, dp_antichain


def prune(sig_reg_list, children, llr_adjusted, exp_eff=None, lam=None):
    """Optimal pruning via DP on the significant subtree.

    Finds the antichain E that maximises::

        J(E) = sum_{i in E} [llr_adjusted(i) - lam]

    where ``llr_adjusted`` is the size-adjusted LLR statistic already
    computed during the FWER step.

    If ``lam`` is provided it is used as the penalty.  If ``exp_eff``
    is provided (and ``lam`` is None) the analytic geometric-prior
    formula ``lam = log(1 + 1/exp_eff)`` is used.  Otherwise
    ``lam = 0`` (no additional penalty beyond the size adjustment
    already baked into ``llr_adjusted``).

    Args:
        sig_reg_list (list): regions declared significant (via FWER)
        children (np.array): (num_internal, 2) child index pairs
        llr_adjusted (np.array): 1-D size-adjusted LLR, one entry per
            region (leaves + internal nodes)
        exp_eff (float or None): expected number of effect regions
            (geometric-prior penalty; ignored when lam is given)
        lam (float or None): explicit penalty override

    Returns:
        reg_out_list (list): sorted indices of effect regions
        dp_info (dict): diagnostics (gain, best, lam, ...)
    """
    if not sig_reg_list:
        return [], dict(gain={}, best={}, lam=0.0,
                        subgraph_children={},
                        sig_reg_list=[])

    num_vox = len(llr_adjusted) - children.shape[0]

    subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                     subset=sig_reg_list)

    if lam is not None:
        pass
    elif exp_eff is not None:
        lam = np.log(1 + 1 / exp_eff)
    else:
        lam = 0.0

    reg_out_list, dp_info = dp_antichain(
        nodes=sorted(sig_reg_list),
        children_map=subgraph.children,
        gain=llr_adjusted,
        lam=lam,
    )

    dp_info['subgraph_children'] = dict(subgraph.children)
    dp_info['sig_reg_list'] = list(sig_reg_list)

    return reg_out_list, dp_info
