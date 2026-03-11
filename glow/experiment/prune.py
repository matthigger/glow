import numpy as np

from .mancova import decompose, loglik_from_cov, get_mancova
from ..graph import SCGraph, dp_antichain


def _region_loglik(y, q_tup):
    """Gaussian profile log-likelihood for a single region.

    Returns -(n/2) * log|det(E/n)| where E is the MANCOVA error
    matrix.

    Args:
        y (np.array): (b, num_img, n_vox) imaging data for a region
        q_tup: pre-computed QR decomposition from decompose()

    Returns:
        ll (float): profile log-likelihood (higher = better fit)
    """
    e = get_mancova(y=y, q_tup=q_tup)[0]
    return loglik_from_cov(e, y.shape[2])


def _compute_region_ll(y_region, q_tup):
    """log-likelihood under the full and null MANCOVA models.

    Full model: regress on interest + nuisance features (error = E).
    Null model: regress on nuisance only (error = E + H).

    Args:
        y_region (np.array): (b, num_img, n_vox) imaging data for a region
        q_tup: pre-computed QR decomposition from decompose(x, contrast)

    Returns:
        ll_full (float): -(n/2) * log|det(E / n)|
        ll_null (float): -(n/2) * log|det((E + H) / n)|
    """
    e, h, _ = get_mancova(y=y_region, q_tup=q_tup)
    n = y_region.shape[2]
    return loglik_from_cov(e, n), loglik_from_cov(e + h, n)


def _build_vox_cache(sig_reg_list, children, num_vox):
    """cache voxel indices for each significant node.

    Args:
        sig_reg_list (list): significant region indices
        children (np.array): (num_internal, 2) child index pairs
        num_vox (int): number of leaf nodes (voxels)

    Returns:
        vox_cache (dict): node -> np.array of leaf (voxel) indices
    """
    from ..graph import iter_topo
    vox_cache = {}
    for node in sig_reg_list:
        vox_cache[node] = np.array(list(iter_topo(
            children=children, num_leaf=num_vox,
            node_start=node, only_leaf=True)))
    return vox_cache


def _compute_all_ll(sig_reg_list, y, q_tup, vox_cache):
    """compute LL_1 and LL_0 for every significant node.

    Args:
        sig_reg_list (list): significant region indices
        y (np.array): (b, num_img, num_vox) imaging data
        q_tup: pre-computed QR decomposition
        vox_cache (dict): node -> voxel index array

    Returns:
        ll_full (dict): node -> LL_1
        ll_null (dict): node -> LL_0
    """
    ll_full = {}
    ll_null = {}
    for node in sig_reg_list:
        y_region = y[:, :, vox_cache[node]]
        ll_full[node], ll_null[node] = _compute_region_ll(y_region, q_tup)
    return ll_full, ll_null


def _dp_solve(subgraph, sig_reg_list, gain_per_node, lam):
    """bottom-up DP on the significant subtree.

    Thin wrapper around ``dp_antichain`` using the SCGraph's
    short-circuited children map.

    Args:
        subgraph (SCGraph): short-circuited significant subtree
        sig_reg_list (list): significant region indices (ascending order)
        gain_per_node (dict): node -> same-node LLR
        lam (float): per-region penalty

    Returns:
        reg_out_list (list): sorted indices of selected effect regions
        dp_info (dict): diagnostic arrays (gain, best per node)
    """
    return dp_antichain(
        nodes=sorted(sig_reg_list),
        children_map=subgraph.children,
        gain=gain_per_node,
        lam=lam,
    )


def _gains_from_ll(ll_full, ll_null, sig_reg_list):
    """same-node LLR gain for each significant node.

    gain(i) = LL_full(i) - LL_null(i).  Both are computed on the
    same region so that the spatial covariance (sigma) cancels.

    Returns:
        gain_per_node (dict): node -> same-node LLR
    """
    return {node: ll_full[node] - ll_null[node] for node in sig_reg_list}


def _calibrate_lambda(sig_reg_list, exp, q_tup, vox_cache,
                      n_perm, alpha):
    """calibrate lambda from Freedman-Lane permutations.

    For each permutation, computes the same-node LLR gain for every
    significant node and records the maximum gain.  Lambda is set to
    the (1 - alpha) quantile of these max-gains so that under H0
    even the single strongest false-positive region is unlikely to
    exceed the penalty.

    Also returns per-node H0 gain statistics (mean, std) for
    diagnosing size-dependent bias in the calibration.

    Args:
        sig_reg_list (list): significant region indices
        exp (Experiment): experiment data (unpermuted)
        q_tup: pre-computed QR decomposition
        vox_cache (dict): node -> voxel index array
        n_perm (int): number of permutations (excluding unpermuted)
        alpha (float): significance level for the quantile

    Returns:
        lam (float): calibrated per-region penalty
        max_gains (np.array): (n_perm,) max single-node gain per perm
        node_h0_mean (dict): node -> mean gain under H0
        node_h0_std (dict): node -> std of gain under H0
    """
    n_nodes = len(sig_reg_list)
    gain_matrix = np.empty((n_perm, n_nodes))
    for p in range(n_perm):
        exp_perm = exp.permute(perm_idx=p + 1)
        ll_f_p, ll_n_p = _compute_all_ll(sig_reg_list, exp_perm.y,
                                         q_tup, vox_cache)
        gains_p = _gains_from_ll(ll_f_p, ll_n_p, sig_reg_list)
        gain_matrix[p, :] = [gains_p[node] for node in sig_reg_list]

    max_gains = gain_matrix.max(axis=1)
    lam = float(np.quantile(max_gains, 1.0 - alpha))

    node_h0_mean = dict(zip(sig_reg_list, gain_matrix.mean(axis=0)))
    node_h0_std = dict(zip(sig_reg_list, gain_matrix.std(axis=0)))

    return lam, max_gains, node_h0_mean, node_h0_std


def prune(sig_reg_list, children, exp, n_perm=100, alpha=0.05,
          exp_eff=None, lam=None):
    """optimal pruning via DP with permutation-calibrated penalty.

    Finds the antichain E in the significant subtree that maximises:

        J(E) = sum_{i in E} [gain(i) - lambda]

    where gain(i) = LL_full(i) - LL_null(i) is the same-node LLR
    (full MANCOVA model vs null model, same region, so the spatial
    covariance sigma cancels).

    Lambda is calibrated from Freedman-Lane permutations: for each
    permutation the maximum same-node gain across all significant
    regions is recorded, and lambda is set to the (1 - alpha) quantile.
    This ensures that under H0 even the single strongest false-positive
    region is unlikely to exceed the penalty.

    If ``lam`` is provided it overrides the permutation calibration.
    If ``exp_eff`` is provided (and ``lam`` is None) it uses the
    analytic geometric-prior formula lambda = log(1 + 1/exp_eff)
    instead of permutations.

    Args:
        sig_reg_list (list): regions declared significant (via FWER)
        children (np.array): (num_internal, 2) child index pairs
        exp (Experiment): experiment data
        n_perm (int): number of calibration permutations (default 100)
        alpha (float): quantile level for calibration (default 0.05)
        exp_eff (float or None): expected number of effect regions
            (geometric-prior fallback; ignored when lam is given)
        lam (float or None): override per-region penalty

    Returns:
        reg_out_list (list): sorted indices of effect regions
        dp_info (dict): diagnostics (gain, best, lam, ...)
    """
    if not sig_reg_list:
        return [], dict(gain={}, best={}, lam=0.0,
                        gain_per_node={},
                        subgraph_children={},
                        sig_reg_list=[])

    num_vox = exp.y.shape[2]

    # build significant subtree
    subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                     subset=sig_reg_list)

    # pre-compute QR decomposition (reused everywhere)
    q_tup = decompose(exp.x, exp.contrast)

    # cache voxel indices per significant node
    vox_cache = _build_vox_cache(sig_reg_list, children, num_vox)

    # compute same-node gains on real (unpermuted) data
    ll_full, ll_null = _compute_all_ll(sig_reg_list, exp.y, q_tup,
                                       vox_cache)
    gain_per_node = _gains_from_ll(ll_full, ll_null, sig_reg_list)

    # determine lambda
    node_h0_mean, node_h0_std = {}, {}
    if lam is not None:
        pass  # explicit override
    elif exp_eff is not None:
        lam = np.log(1 + 1 / exp_eff)
    else:
        # permutation calibration
        lam, _, node_h0_mean, node_h0_std = _calibrate_lambda(
            sig_reg_list, exp, q_tup, vox_cache,
            n_perm=n_perm, alpha=alpha)

    # run DP
    reg_out_list, dp_info = _dp_solve(subgraph, sig_reg_list,
                                      gain_per_node, lam)

    # store intermediates for viewer re-computation
    dp_info['gain_per_node'] = gain_per_node
    dp_info['subgraph_children'] = dict(subgraph.children)
    dp_info['sig_reg_list'] = list(sig_reg_list)
    dp_info['node_gain_h0_mean'] = node_h0_mean
    dp_info['node_gain_h0_std'] = node_h0_std

    return reg_out_list, dp_info
