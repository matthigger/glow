"""Reference (slow) inner-FL permutation helper for tests.

Mirrors the buffer-path that used to live in
``AnalysisGLOW._process_permutation``: for each inner permutation,
runs the full ``compute_llr_batched`` pass and stores the per-region
LLR.  Returns the full ``(n_perm_inner, num_reg)`` buffer plus the
nanmean / nanstd summaries.

Used by ``test_analysis.test_race_vs_slow_mu_sigma`` to check the
race-based Welford accounting against an unambiguous reference.  Not
part of production — production uses the race in
``AnalysisGLOW._process_permutation`` and never materialises the full
buffer.
"""
import warnings

import numpy as np

import glow.graph


def compute_slow_inner(exp, children, q0, q1, layer, n_perm_inner,
                       base_seed, min_vox):
    """Run ``n_perm_inner`` inner FL perms against a fixed tree.

    Args:
        exp: source experiment (unpermuted at the inner level).
        children: Ward children for the outer-perm tree.
        q0, q1: pre-decomposed nuisance / contrast subspaces.
        layer: per-node tree depth from ``compute_tree_layers``.
        n_perm_inner: number of inner permutations to draw.
        base_seed: ``perm.permute`` seed base (production uses
            ``(perm_idx + 1) * 100_000``).
        min_vox: regions below this size are skipped in Phase 2 and
            return NaN; matches the production gate.

    Returns:
        dict with
            ``llr_inner`` (n_perm_inner, num_reg) float64,
            ``mu``, ``sigma`` (num_reg,) — nanmean / nanstd(ddof=1),
            ``n_per_reg`` (num_reg,) int64.
    """
    llr0, _ = glow.graph.compute_llr_batched(
        exp, children=children, q0=q0, q1=q1,
        min_size=min_vox, layer=layer)
    num_reg = llr0.shape[0]

    llr_inner = np.empty((n_perm_inner, num_reg), dtype=float)
    for i in range(n_perm_inner):
        _exp_i = exp.permute(base_seed + i)
        llr_i, _ = glow.graph.compute_llr_batched(
            _exp_i, children=children, q0=q0, q1=q1,
            min_size=min_vox, layer=layer)
        llr_inner[i, :] = llr_i

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu = np.nanmean(llr_inner, axis=0)
        sigma = np.nanstd(llr_inner, axis=0, ddof=1)

    n_per_reg = np.where(np.isfinite(mu), n_perm_inner, 0).astype(np.int64)
    return {'llr_inner': llr_inner, 'mu': mu, 'sigma': sigma,
            'n_per_reg': n_per_reg}
