"""Inner Freedman-Lane permutation drivers.

For one outer permutation: given the outer-perm tree (Ward children)
and the source experiment, draw ``n_perm_inner`` inner-perm LLR samples
and reduce them to per-region ``(mu, sigma, n_per_reg)``.  Four backends
cover the (cpu vs gpu) x (intercept-only fast vs general slow) grid:

  - :func:`cpu_fast` -- intercept-only.  Q0 commutes with permutations,
    so Phase-1 (yout/ysum/T_u) hoists out of the inner loop and each
    draw is a row-permuted ``q1.T`` against per-region precomputes.
  - :func:`cpu_slow` -- general-Q0.  Full ``compute_llr_batched`` per
    draw, against a fresh ``exp.permute(base + i)``.
  - :func:`gpu_fast` -- intercept-only on GPU (single-batch CUDA Graph,
    closed-form 2x2 LLR).
  - :func:`gpu_slow` -- general-Q0 on GPU (multi-batch CUDA Graph with
    Chan-merged moments).

All four share a single keyword-only signature so the caller just picks
one and calls it::

    run = inner_perm.gpu_fast if use_gpu and use_fast else ...
    moments = run(exp=exp, exp_perm=exp_perm, perm_idx=perm_idx,
                  q0=q0, q1=q1, children=children, layer=layer,
                  n_perm_inner=n_perm_inner, min_vox=min_vox)

CPU backends read ``exp.y`` (and ignore ``exp_perm``); GPU backends
read ``exp_perm.y`` (and ignore ``exp``).  The caller is responsible
for outer-permuting via ``exp.permute(perm_idx)`` and passing the
result as ``exp_perm``.

CPU backends honour ``keep_draws=True`` to additionally return the raw
``(n_perm_inner, num_reg)`` LLR matrix; GPU backends raise on it
because moments are reduced on-device and the per-draw LLR is never
materialised on the host.
"""
import warnings

import numpy as np

import glow.graph


def cpu_fast(*, exp, exp_perm, perm_idx, q0, q1, children, layer,
              n_perm_inner, min_vox, keep_draws=False):
    """Intercept-only CPU fast path.

    Q0 commutes with the FL permutation so Phase-1 (``yout``, ``ysum``,
    ``T_u``) is permutation-invariant and hoists out of the inner loop.
    Each draw is then a row-permuted ``q1.T`` against the per-region
    precomputes, run through ``compute_llr_inner_fast``.
    """
    del exp_perm  # GPU backends read this; CPU paths reuse exp.y
    n_img = exp.y.shape[1]
    base = (perm_idx + 1) * 100_000
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
    draws = np.empty((n_perm_inner, num_reg), dtype=float)
    for i in range(n_perm_inner):
        rng = np.random.default_rng(base + i)
        perm = np.argsort(rng.permutation(n_img))
        freed_lane = (eye_n - q0_proj)[:, perm] + q0_proj
        q1_T_perm = (freed_lane @ q1.T).astype(_dtype, copy=False)
        llr_i, _ = glow.graph.compute_llr_inner_fast(
            t_u, ysum_u, size_u, q1_T_perm, min_size=min_vox)
        draws[i] = llr_i
    return _moments_from_draws(draws, keep_draws=keep_draws)


def cpu_slow(*, exp, exp_perm, perm_idx, q0, q1, children, layer,
              n_perm_inner, min_vox, keep_draws=False):
    """General-Q0 CPU slow path: full ``compute_llr_batched`` per draw."""
    del exp_perm
    base = (perm_idx + 1) * 100_000
    num_vox = exp.y.shape[2]
    num_reg = num_vox + children.shape[0]
    draws = np.empty((n_perm_inner, num_reg), dtype=float)
    for i in range(n_perm_inner):
        _exp_inner = exp.permute(base + i)
        llr_i, _ = glow.graph.compute_llr_batched(
            _exp_inner, children=children, q0=q0, q1=q1,
            min_size=min_vox, layer=layer)
        draws[i] = llr_i
    return _moments_from_draws(draws, keep_draws=keep_draws)


def gpu_fast(*, exp, exp_perm, perm_idx, q0, q1, children, layer,
              n_perm_inner, min_vox, keep_draws=False):
    """Intercept-only GPU path -- closed-form 2x2 LLR, single CUDA Graph."""
    del exp  # CPU backends read this; GPU paths consume exp_perm
    _reject_keep_draws_on_gpu(keep_draws)
    from glow.analysis.inner_perm_gpu import _run_intercept
    out = _run_intercept(
        exp_perm, perm_idx, q0=q0, q1=q1, children=children, layer=layer,
        n_perm_inner=n_perm_inner, min_vox=min_vox,
        base_seed=None, device='cuda')
    return {'mu': out['mu'], 'sigma': out['sigma'],
            'n_per_reg': out['n_per_reg']}


def gpu_slow(*, exp, exp_perm, perm_idx, q0, q1, children, layer,
              n_perm_inner, min_vox, keep_draws=False):
    """General-Q0 GPU path -- alpha/beta decomposition, Chan-merged moments."""
    del exp
    _reject_keep_draws_on_gpu(keep_draws)
    from glow.analysis.inner_perm_gpu import _run_general
    out = _run_general(
        exp_perm, perm_idx, q0=q0, q1=q1, children=children, layer=layer,
        n_perm_inner=n_perm_inner, min_vox=min_vox,
        base_seed=None, device='cuda')
    return {'mu': out['mu'], 'sigma': out['sigma'],
            'n_per_reg': out['n_per_reg']}


def _reject_keep_draws_on_gpu(keep_draws):
    if keep_draws:
        raise NotImplementedError(
            'keep_draws=True is not supported on GPU backends: '
            'moments are reduced on-device and the per-draw LLR '
            'matrix is never materialised on the host.')


def _moments_from_draws(draws, *, keep_draws):
    """Reduce ``(n_perm_inner, num_reg)`` draws to per-region moments.

    Phase-2 skips regions with ``size < min_vox`` so those columns stay
    NaN; ``nanmean`` / ``nanstd`` ignore them and ``n_per_reg`` records
    the finite count.
    """
    n_per_reg = np.isfinite(draws).sum(axis=0).astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu = np.nanmean(draws, axis=0)
        sigma = np.nanstd(draws, axis=0, ddof=1)
    out = {'mu': mu, 'sigma': sigma, 'n_per_reg': n_per_reg}
    if keep_draws:
        out['draws'] = draws
    return out
