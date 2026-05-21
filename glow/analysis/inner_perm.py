"""Inner Freedman-Lane permutation drivers.

For one outer permutation: given the outer-perm tree (Ward children)
and an experiment, draw ``n_perm`` inner-perm LLR samples and reduce
them to per-region ``(mu, sigma)``.  Four backends cover the
(cpu vs gpu) x (intercept-only fast vs general slow) grid:

  - :func:`cpu_fast` -- intercept-only.  Q0 commutes with permutations,
    so Phase-1 (yout/ysum/T_u) hoists out of the inner loop and each
    draw is a row-permuted ``q1.T`` against per-region precomputes.
  - :func:`cpu_slow` -- general-Q0.  Full ``compute_llr_batched`` per
    draw, against a fresh ``exp.permute(base_seed + i)``.
  - :func:`gpu_fast` -- intercept-only on GPU (single-batch CUDA Graph,
    closed-form 2x2 LLR).
  - :func:`gpu_slow` -- general-Q0 on GPU (multi-batch CUDA Graph with
    Chan-merged moments).

All four share a single keyword-only signature and return ``(mu,
sigma)``::

    run = inner_perm.gpu_fast if use_gpu and use_fast else ...
    mu, sigma = run(
        exp=exp, base_seed=base_seed, n_perm=n_perm,
        q0=q0, q1=q1, children=children, layer=layer, min_vox=min_vox)

``exp`` is whatever experiment the inner perms should operate on -- the
backends don't need to know whether it carries an outer permutation;
they just sample iid permutations from it.

``base_seed`` is the starting RNG seed: each draw ``i`` uses
``base_seed + i``.  The caller is responsible for choosing a
non-colliding base across outer perms (``_glow.py`` reserves a
100_000-wide block per outer perm).

CPU backends are thin wrappers over :func:`cpu_fast_full` /
:func:`cpu_slow_full`, which return the raw ``(n_perm, num_reg)`` LLR
draws matrix; tests that want to inspect individual draws call those
directly.  GPU backends reduce moments on-device, so the raw draws
matrix is never materialised on the host.
"""
import functools
import warnings

import numpy as np

import glow.graph


def moments_from_draws(fn):
    """Decorator: wraps a fn returning ``(n_perm, num_reg)`` draws into
    one returning ``(mu, sigma)``.  Phase-2 leaves draws NaN for
    size < min_vox regions; ``nanmean`` / ``nanstd`` ignore them.
    """
    @functools.wraps(fn)
    def wrapper(**kwargs):
        draws = fn(**kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mu = np.nanmean(draws, axis=0)
            sigma = np.nanstd(draws, axis=0, ddof=1)
        return mu, sigma
    return wrapper


def cpu_fast_full(*, exp, base_seed, n_perm,
                   q0, q1, children, layer, min_vox):
    """Intercept-only CPU fast path -- returns ``(n_perm, num_reg)`` draws.

    Q0 commutes with the FL permutation so Phase-1 (``yout``, ``ysum``,
    ``T_u``) is permutation-invariant and hoists out of the inner loop.
    Each draw is then a row-permuted ``q1.T`` against the per-region
    precomputes, run through ``compute_llr_inner_fast``.
    """
    n_img = exp.y.shape[1]
    q0_proj = q0.T @ q0
    eye_n = np.eye(n_img, dtype=q0.dtype)

    ysum_u, yout_u, size_u = glow.graph.compute_phase1(
        exp.y, children, layer=layer)
    _dtype = exp.y.dtype if exp.y.dtype == np.float32 else np.float64
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
                   q0, q1, children, layer, min_vox):
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
            min_size=min_vox, layer=layer)
        draws[i] = llr_i
    return draws


cpu_fast = moments_from_draws(cpu_fast_full)
cpu_slow = moments_from_draws(cpu_slow_full)


def gpu_fast(*, exp, base_seed, n_perm,
              q0, q1, children, layer, min_vox):
    """Intercept-only GPU path -- returns ``(mu, sigma)``.

    Closed-form 2x2 LLR, single CUDA Graph.
    """
    from glow.analysis.inner_perm_gpu import _run_intercept
    # ``perm_idx`` is only used by the GPU helper to derive a default
    # base_seed when base_seed is None; we always pass base_seed so
    # the value here is irrelevant.
    out = _run_intercept(
        exp, perm_idx=0,
        q0=q0, q1=q1, children=children, layer=layer,
        n_perm_inner=n_perm, min_vox=min_vox,
        base_seed=base_seed, device='cuda')
    return out['mu'], out['sigma']


def gpu_slow(*, exp, base_seed, n_perm,
              q0, q1, children, layer, min_vox):
    """General-Q0 GPU path -- returns ``(mu, sigma)``.

    Alpha/beta decomposition, multi-batch CUDA Graph with Chan-merged
    moments.
    """
    from glow.analysis.inner_perm_gpu import _run_general
    out = _run_general(
        exp, perm_idx=0,
        q0=q0, q1=q1, children=children, layer=layer,
        n_perm_inner=n_perm, min_vox=min_vox,
        base_seed=base_seed, device='cuda')
    return out['mu'], out['sigma']
