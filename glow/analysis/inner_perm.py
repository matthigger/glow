"""Inner Freedman-Lane permutation drivers.

For one outer permutation: given the outer-perm tree (Ward children)
and an experiment, draw ``n_perm`` inner-perm LLR samples and reduce
them to per-region ``(mu, std)``.  Two production backends split on
whether Q0 commutes with permutations:

  - :func:`cpu_fast` -- intercept-only.  Q0 commutes with permutations,
    so Phase-1 (yout/ysum/T_u) hoists out of the inner loop and each
    draw is a row-permuted ``q1.T`` against per-region precomputes.
  - :func:`cpu_slow` -- general-Q0.  Full ``compute_llr_batched`` per
    draw, against a fresh ``exp.permute(base_seed + i)``.

A third backend, :func:`cpu_reliable`, drives the per-region
``iter_mancova`` + ``get_llr`` path per draw -- a different code path
from ``compute_llr_batched``, used as the trust anchor in tests of
the optimised backends above.

All three share a single keyword-only signature and return ``(mu,
std)``::

    run = inner_perm.cpu_fast if use_fast else inner_perm.cpu_slow
    mu, std = run(
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


def cpu_fast_full(*, exp, base_seed, n_perm,
                   q0, q1, children, min_vox):
    """Intercept-only CPU fast path -- returns ``(n_perm, num_reg)`` draws.

    Q0 commutes with the FL permutation so Phase-1 (``yout``, ``ysum``,
    ``T_u``) is permutation-invariant and hoists out of the inner loop.
    Each draw is then a row-permuted ``q1.T`` against the per-region
    precomputes, run through ``compute_llr_inner_fast``.
    """
    b, n_img, num_vox = exp.y.shape
    q0_proj = q0.T @ q0
    eye_n = np.eye(n_img, dtype=q0.dtype)

    _dtype = exp.y.dtype if exp.y.dtype == np.float32 else np.float64
    num_reg = num_vox + children.shape[0]
    ysum_u = np.empty((num_reg, b, n_img), dtype=_dtype)
    yout_u = np.empty((num_reg, b, b), dtype=_dtype)
    size_u = np.empty(num_reg, dtype=int)
    for reg_idx, sz, ys, yo in glow.graph.iter_size_ysum_yout(
            exp.y, children=children):
        ysum_u[reg_idx] = ys
        yout_u[reg_idx] = yo
        size_u[reg_idx] = sz
    _sz_3d = size_u.astype(_dtype)[:, None, None]
    _a0 = np.einsum('rbn,an->rba', ysum_u, q0, optimize=True)
    t_u = yout_u - np.einsum('rba,rca->rbc', _a0, _a0, optimize=True) / _sz_3d
    del _a0

    num_reg = ysum_u.shape[0]
    draws = np.empty((n_perm, num_reg), dtype=float)
    for i in range(n_perm):
        rng = np.random.default_rng(base_seed + i)
        perm = np.argsort(rng.permutation(n_img))
        freed_lane = (eye_n - q0_proj)[:, perm] + q0_proj
        q1_T_perm = (freed_lane @ q1.T).astype(_dtype, copy=False)
        llr_i, _ = glow.graph.compute_llr_inner_fast(
            t_u, ysum_u, size_u, q1_T_perm, min_size=min_vox)
        draws[i] = llr_i
    return draws


def cpu_slow_full(*, exp, base_seed, n_perm,
                   q0, q1, children, min_vox):
    """General-Q0 CPU slow path -- returns ``(n_perm, num_reg)`` draws.

    Full ``compute_llr_batched`` per draw, each against a fresh
    ``exp.permute(base_seed + i)``.
    """
    num_vox = exp.y.shape[2]
    num_reg = num_vox + children.shape[0]
    draws = np.empty((n_perm, num_reg), dtype=float)
    for i in range(n_perm):
        _exp_inner = exp.permute(base_seed + i)
        llr_i, _ = glow.graph.compute_llr_batched(
            _exp_inner, children=children, q0=q0, q1=q1,
            min_size=min_vox)
        draws[i] = llr_i
    return draws


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


cpu_fast = moments_from_draws(cpu_fast_full)
cpu_slow = moments_from_draws(cpu_slow_full)
cpu_reliable = moments_from_draws(cpu_reliable_full)
