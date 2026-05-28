"""Inner Freedman-Lane permutation drivers.

For one outer permutation: given the outer-perm tree (Ward children)
and an experiment, draw ``n_perm`` inner-perm LLR samples and reduce
them to per-region ``(mu, std)``.  Two production backends:

  - :func:`cpu_perm` -- the working horse.  Rides
    :func:`glow.graph.iter_llr_perm`: per-voxel sufficient statistics
    are computed once; per perm-chunk the only work is one big
    ``Q^T P r_v`` GEMM plus cumsum-and-diff aggregation on the DFS
    pre-order axis.  Each chunk of draws is folded into a
    Welford / Chan-parallel accumulator and dropped -- the full
    ``(n_perm, num_reg)`` draws matrix is never materialized.  Handles
    intercept-only and general-Q0 nuisance on the same code path --
    under intercept-only Q0 commutes with P, so the ``rho`` and ``X_v``
    terms come out numerically zero and the rest of the algorithm is
    unaffected.
  - :func:`cpu_reliable` -- trust anchor for tests.  Drives the
    per-region ``iter_mancova`` + ``get_llr`` path per draw -- an
    independent code path used to cross-validate ``cpu_perm``.

Both share a single keyword-only signature and return ``(mu, std)``::

    mu, std = inner_perm.cpu_perm(
        exp=exp, base_seed=base_seed, n_perm=n_perm,
        q0=q0, q1=q1, children=children, min_vox=min_vox)

``exp`` is whatever experiment the inner perms should operate on -- the
backends don't need to know whether it carries an outer permutation;
they just sample iid permutations from it.

``base_seed`` is the starting RNG seed: each draw ``i`` uses
``base_seed + i``.  The caller is responsible for choosing a
non-colliding base across outer perms (``_glow.py`` reserves a
100_000-wide block per outer perm).

:func:`cpu_reliable_full` returns the raw ``(n_perm, num_reg)`` LLR
draws matrix; tests that want to inspect individual draws call it
directly.  ``cpu_perm`` no longer has a ``_full`` variant -- callers
that need per-draw output materialize via
``np.vstack(list(glow.graph.iter_llr_perm(...)))``.
"""
import numpy as np

import glow.graph
from glow.analysis import mancova
from glow.experiment import permute


def _welford_moments(chunks, num_reg):
    """Reduce an iterator of (Pc, num_reg) NaN-aware LLR chunks to (mu, std).

    Per-region accumulators ``(n, mean, M2)`` updated via Chan's
    parallel-combine rule.  NaN cells are excluded from the count,
    matching the old ``nanmean / nanstd(ddof=1)`` semantics.  Returns
    NaN ``mu`` where no chunk had a valid sample, NaN ``std`` where
    fewer than 2 valid samples accumulated.

    Numerically stable across chunk boundaries: the naive single-pass
    ``(sumsq - sum^2/n)/(n-1)`` formula loses precision when var << mean^2;
    Chan-parallel updates avoid the cancellation by maintaining the
    centered ``M2`` directly.
    """
    n = np.zeros(num_reg, dtype=np.float64)
    mean = np.zeros(num_reg, dtype=np.float64)
    M2 = np.zeros(num_reg, dtype=np.float64)

    for chunk in chunks:
        valid = ~np.isnan(chunk)
        chunk_safe = np.where(valid, chunk, 0.0)
        n_b = valid.sum(axis=0).astype(np.float64)
        sum_b = chunk_safe.sum(axis=0)

        # Guarded divisions: when n_b == 0 the chunk contributes
        # nothing; mean_b can be anything (multiplied by 0 below).
        safe_nb = np.where(n_b > 0, n_b, 1.0)
        mean_b = sum_b / safe_nb
        dev = np.where(valid, chunk_safe - mean_b[None, :], 0.0)
        M2_b = (dev * dev).sum(axis=0)

        new_n = n + n_b
        safe_new_n = np.where(new_n > 0, new_n, 1.0)
        delta = mean_b - mean
        mean = mean + delta * (n_b / safe_new_n)
        M2 = M2 + M2_b + delta * delta * (n * n_b / safe_new_n)
        n = new_n

    mu = np.where(n > 0, mean, np.nan)
    safe_dof = np.where(n > 1, n - 1, 1.0)
    var = np.where(n > 1, M2 / safe_dof, np.nan)
    # ULP-level negatives can creep in; clamp before sqrt.
    var = np.where((~np.isnan(var)) & (var < 0), 0.0, var)
    std = np.sqrt(var)
    return mu, std


def cpu_perm(*, exp, base_seed, n_perm, q0, q1, children, min_vox):
    """Inner-perm ``(mu, std)`` via the unified streaming perm-LLR backend.

    Builds Phase-1 sufficient statistics once, then folds each
    perm-chunk of LLR draws into a Welford / Chan-parallel accumulator.
    Peak memory is Phase-1 state + per-chunk Phase-2 temporaries; the
    full ``(n_perm, num_reg)`` draws matrix is never materialized.

    Handles intercept-only and general-Q0 nuisance on the same code
    path.
    """
    num_vox = exp.y.shape[2]
    num_img = exp.y.shape[1]
    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=num_vox)
    perms = np.empty((n_perm, num_img), dtype=np.int64)
    for i in range(n_perm):
        perms[i] = permute._perm_indices(base_seed + i, num_img)
    num_reg = int(region_l.shape[0])
    chunks = glow.graph.iter_llr_perm(
        y=exp.y, q0=q0, q1=q1, perms=perms,
        leaf_ord=leaf_ord, region_l=region_l, region_h=region_h,
        min_size=min_vox)
    return _welford_moments(chunks, num_reg)


def cpu_reliable_full(*, exp, base_seed, n_perm,
                      q0, q1, children, min_vox):
    """Trust-anchor CPU path -- returns ``(n_perm, num_reg)`` draws.

    Thin wrapper over the per-region ``iter_mancova`` + ``get_llr``
    path: for each draw, FL-permute via ``exp.permute(base_seed + i)``,
    then iterate ``(reg_idx, size, e, h)`` per region and finish with
    ``get_llr(e, h, n=size)``.  No batching, no closed-form 2x2 slogdet,
    no Phase-1 hoisting -- an independent code path from
    ``compute_llr_batched`` for cross-validating the optimised backend.

    ``q0`` / ``q1`` are accepted for interface parity but unused:
    ``iter_mancova`` recomputes them internally via
    ``decompose(exp.x, exp.contrast)``, which is exactly the same
    ``(q0, q1)`` the caller would have passed in.
    """
    del q0, q1
    num_reg = exp.y.shape[2] + children.shape[0]
    draws = np.full((n_perm, num_reg), np.nan, dtype=np.float64)
    for i in range(n_perm):
        _exp = exp.permute(base_seed + i)
        for reg_idx, size, e, h in glow.graph.iter_mancova(
                _exp, children=children):
            if size < min_vox:
                continue
            draws[i, reg_idx] = mancova.get_llr(e, h, n=size)
    return draws


def cpu_reliable(*, exp, base_seed, n_perm, q0, q1, children, min_vox):
    """Trust-anchor ``(mu, std)``: ``cpu_reliable_full`` draws folded
    through the same Welford accumulator ``cpu_perm`` uses, so both
    backends share identical moment-reduction semantics."""
    draws = cpu_reliable_full(
        exp=exp, base_seed=base_seed, n_perm=n_perm,
        q0=q0, q1=q1, children=children, min_vox=min_vox)
    return _welford_moments([draws], draws.shape[1])
