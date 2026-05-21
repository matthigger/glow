"""Inner Freedman-Lane permutation drivers.

For one outer permutation: given the outer-perm tree (Ward children)
and the source experiment, draw ``n_perm_inner`` inner-perm LLR samples
and reduce them to per-region ``(mu, sigma, n_per_reg)``.  Four backends
cover the (cpu vs gpu) x (intercept-only fast vs general slow) grid:

  - ``cpu_fast``: intercept-only.  Q0 commutes with permutations, so
    Phase-1 (yout/ysum/T_u) hoists out of the inner loop and each draw
    is a row-permuted ``q1.T`` against per-region precomputes.
  - ``cpu_slow``: general-Q0.  Full ``compute_llr_batched`` per draw,
    against a fresh ``exp.permute(base + i)``.
  - ``gpu_fast``: intercept-only on GPU (single-batch CUDA Graph,
    closed-form 2x2 LLR).
  - ``gpu_slow``: general-Q0 on GPU (multi-batch CUDA Graph with
    Chan-merged moments).

All four return per-region moments in the same dict contract.  CPU
backends additionally support ``keep_draws=True`` for tests that want
to inspect the ``(n_perm_inner, num_reg)`` raw draws matrix; GPU paths
reduce on-device and raise on that flag.
"""
import warnings

import numpy as np

import glow.graph


def draw_llr_samples(
    *,
    exp,
    exp_perm,
    perm_idx,
    q0,
    q1,
    children,
    layer,
    n_perm_inner,
    min_vox,
    backend,
    keep_draws=False,
):
    """Run ``n_perm_inner`` inner FL perms; return per-region moments.

    Args:
        exp: unpermuted source Experiment.  CPU backends read
            ``exp.y``; GPU backends ignore it.
        exp_perm: outer-permuted Experiment (``exp.permute(perm_idx)``).
            GPU backends read ``exp_perm.y``; CPU backends ignore it
            (pass ``None`` to free the ~1 GB outer-perm copy).
        perm_idx: outer permutation index.  Used to seed inner perms
            via ``base = (perm_idx + 1) * 100_000``.
        q0, q1: contrast subspaces from ``decompose``.
        children: ``(num_internal, 2)`` Ward children for the outer-perm
            tree (built from ``exp_perm``).
        layer: ``(num_reg,)`` depth table for ``children``.
        n_perm_inner: inner perm count.  Must be >= 1.
        min_vox: regions with size < min_vox return NaN moments / 0 n.
        backend: one of ``'cpu_fast'``, ``'cpu_slow'``, ``'gpu_fast'``,
            ``'gpu_slow'``.
        keep_draws: if True, also return ``draws`` -- the raw
            ``(n_perm_inner, num_reg)`` LLR matrix.  CPU paths only;
            GPU paths raise ``NotImplementedError`` (they reduce
            moments on-device and never materialise draws on the host).

    Returns:
        dict with ``mu``, ``sigma``, ``n_per_reg`` (host arrays length
        num_reg).  With ``keep_draws=True``, also includes ``draws``.
    """
    assert n_perm_inner >= 1, 'n_perm_inner must be >= 1'

    if backend == 'cpu_fast':
        return _draw_cpu_fast(
            exp=exp, perm_idx=perm_idx, q0=q0, q1=q1,
            children=children, layer=layer,
            n_perm_inner=n_perm_inner, min_vox=min_vox,
            keep_draws=keep_draws)
    if backend == 'cpu_slow':
        return _draw_cpu_slow(
            exp=exp, perm_idx=perm_idx, q0=q0, q1=q1,
            children=children, layer=layer,
            n_perm_inner=n_perm_inner, min_vox=min_vox,
            keep_draws=keep_draws)
    if backend in ('gpu_fast', 'gpu_slow'):
        if keep_draws:
            raise NotImplementedError(
                'keep_draws=True is not supported on GPU backends: '
                'moments are reduced on-device and the per-draw LLR '
                'matrix is never materialised on the host.')
        runner = _draw_gpu_fast if backend == 'gpu_fast' else _draw_gpu_slow
        return runner(
            exp_perm=exp_perm, perm_idx=perm_idx, q0=q0, q1=q1,
            children=children, layer=layer,
            n_perm_inner=n_perm_inner, min_vox=min_vox)
    raise ValueError(
        f'backend must be one of cpu_fast, cpu_slow, gpu_fast, gpu_slow; '
        f'got {backend!r}')


def _draw_cpu_fast(*, exp, perm_idx, q0, q1, children, layer,
                    n_perm_inner, min_vox, keep_draws):
    """Intercept-only CPU fast path.

    Q0 commutes with the FL permutation so Phase-1 (``yout``, ``ysum``,
    ``T_u``) is permutation-invariant and hoists out of the inner loop.
    Each draw is then a row-permuted ``q1.T`` against the per-region
    precomputes, run through ``compute_llr_inner_fast``.
    """
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


def _draw_cpu_slow(*, exp, perm_idx, q0, q1, children, layer,
                    n_perm_inner, min_vox, keep_draws):
    """General-Q0 CPU slow path: full ``compute_llr_batched`` per draw."""
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


def _draw_gpu_fast(*, exp_perm, perm_idx, q0, q1, children, layer,
                    n_perm_inner, min_vox):
    """Intercept-only GPU path -- closed-form 2x2 LLR, single CUDA Graph."""
    from glow.analysis.inner_perm_gpu import _run_intercept
    out = _run_intercept(
        exp_perm, perm_idx, q0=q0, q1=q1, children=children, layer=layer,
        n_perm_inner=n_perm_inner, min_vox=min_vox,
        base_seed=None, device='cuda')
    return {'mu': out['mu'], 'sigma': out['sigma'],
            'n_per_reg': out['n_per_reg']}


def _draw_gpu_slow(*, exp_perm, perm_idx, q0, q1, children, layer,
                    n_perm_inner, min_vox):
    """General-Q0 GPU path -- alpha/beta decomposition, Chan-merged moments."""
    from glow.analysis.inner_perm_gpu import _run_general
    out = _run_general(
        exp_perm, perm_idx, q0=q0, q1=q1, children=children, layer=layer,
        n_perm_inner=n_perm_inner, min_vox=min_vox,
        base_seed=None, device='cuda')
    return {'mu': out['mu'], 'sigma': out['sigma'],
            'n_per_reg': out['n_per_reg']}


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
