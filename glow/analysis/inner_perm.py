"""Inner Freedman-Lane permutation drivers.

For one outer permutation: given the outer-perm tree (Ward children)
and an experiment, draw ``n_perm`` inner-perm LLR samples and reduce
them to per-region ``(mu, std)``.  Two production backends:

  - :func:`cpu_perm` -- the working horse.  Rides
    ``glow.graph.compute_llr_perm_full``: per-voxel sufficient
    statistics are computed once; per perm the only work is one big
    ``Q^T P r_v`` GEMM plus cumsum-and-diff aggregation on the DFS
    pre-order axis.  Handles intercept-only and general-Q0 nuisance on
    the same code path -- under intercept-only Q0 commutes with P, so
    the ``rho`` and ``X_v`` terms come out numerically zero and the
    rest of the algorithm is unaffected.
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

Each backend is a thin wrapper over its ``*_full`` variant, which
returns the raw ``(n_perm, num_reg)`` LLR draws matrix; tests that
want to inspect individual draws call those directly.
"""
import functools
import warnings

import numpy as np

import glow.graph
from glow.analysis import mancova


def moments_from_draws(fn):
    """Decorator: wraps a fn returning ``(n_perm, num_reg)`` draws into
    one returning ``(mu, std)``.  Phase-2 leaves draws NaN for
    size < min_vox regions; ``nanmean`` / ``nanstd`` ignore them.
    """
    @functools.wraps(fn)
    def wrapper(**kwargs):
        draws = fn(**kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mu = np.nanmean(draws, axis=0)
            std = np.nanstd(draws, axis=0, ddof=1)
        return mu, std
    return wrapper


def _build_perms(*, base_seed, n_perm, num_img):
    """One ``(n_perm, num_img)`` int array of FL permutations.

    Mirrors ``glow.experiment.permute.get_freed_lane`` so that the
    seed-to-perm mapping is identical to ``exp.permute(base_seed +
    i)``.  Each row is the index array such that
    ``y_perm[..., k] = y[..., perm[k]]``.
    """
    perms = np.empty((n_perm, num_img), dtype=np.int64)
    for i in range(n_perm):
        rng = np.random.default_rng(base_seed + i)
        perms[i] = np.argsort(rng.permutation(num_img))
    return perms


def cpu_perm_full(*, exp, base_seed, n_perm,
                  q0, q1, children, min_vox):
    """Inner-perm draws via the unified perm-LLR backend.

    Computes per-voxel sufficient statistics once and runs ``n_perm``
    Freedman-Lane permutations through one ``Q^T P r_v`` GEMM each.
    Compared to the prior implementations, Phase 1 of
    ``compute_llr_batched`` (Python tree walk over Ward internals) is
    hoisted out of the inner loop entirely.

    Handles intercept-only and general-Q0 nuisance on the same code
    path.  Returns ``(n_perm, num_reg)`` draws.
    """
    num_vox = exp.y.shape[2]
    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=num_vox)
    perms = _build_perms(base_seed=base_seed, n_perm=n_perm,
                         num_img=exp.y.shape[1])
    return glow.graph.compute_llr_perm_full(
        y=exp.y, q0=q0, q1=q1, perms=perms,
        leaf_ord=leaf_ord, region_l=region_l, region_h=region_h,
        min_size=min_vox)


def cpu_reliable_full(*, exp, base_seed, n_perm,
                      q0, q1, children, min_vox):
    """Trust-anchor CPU path -- returns ``(n_perm, num_reg)`` draws.

    Thin wrapper over the per-region ``iter_mancova`` + ``get_llr``
    path: for each draw, FL-permute via ``exp.permute(base_seed + i)``,
    then iterate ``(reg_idx, size, e, h)`` per region and finish with
    ``get_llr(e, h, n=size)``.  No batching, no closed-form 2x2 slogdet,
    no Phase-1 hoisting -- an independent code path from
    ``compute_llr_batched`` for cross-validating the optimised backends.

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


cpu_perm = moments_from_draws(cpu_perm_full)
cpu_reliable = moments_from_draws(cpu_reliable_full)
