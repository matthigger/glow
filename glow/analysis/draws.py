"""CPU Freedman-Lane draws: the fast path and the trust anchor.

Given a Ward tree and an experiment, draw n_perm Freedman-Lane samples
(Freedman & Lane 1983) and return the per-region LLR for each, as one
(n_perm, num_reg) matrix. Row 0 is the observed draw. For
AnalysisGLOWSplit that matrix is the whole hypothesis family -- its column
moments standardize the regions and its row maxima are the max-z null; for
AnalysisGLOW it is one outer perm's inner null, drawn once per tree.

Three CPU entry points over that one matrix, all agreeing on it:

  - cpu_reliable -- per draw and per region through iter_mancova + get_llr,
    unbatched. Slow (~35x cpu_batched) and obviously correct, so the
    optimised backends can be validated against it cell by cell.
  - cpu_batched -- the same matrix through glow.graph.iter_llr_perm: one
    GEMM plus cumsum-and-diff over the DFS pre-order voxel axis per
    perm-chunk.
  - cpu_summary -- cpu_batched's draws reduced to a DrawSummary while
    streaming, never holding the matrix. What both GLOW arms take on the
    CPU.

All three share a keyword-only signature and a seed-to-draw mapping:

    draws = draws.cpu_reliable(
        exp=exp, base_seed=0, n_perm=n_perm_fwer + 1,
        q0=q0, q1=q1, children=children, min_vox=min_vox)

Draw i uses seed base_seed + i, and seed 0 is the identity
(permute._perm_indices), so base_seed=0 puts the observed draw in row 0.

DrawSummary is the other half of the module: the five arrays a fit keeps of
a draw matrix, hence the contract a streaming backend must meet.
summarize_draws is that reduction over a materialized matrix.
"""
from dataclasses import dataclass

import numpy as np

import glow.graph
from glow.analysis import mancova
from glow.experiment import permute
from ._base import Analysis, Z_STD_FLOOR
from .fwer import max_over_active


def cpu_reliable(*, exp, base_seed: int, n_perm: int, q0, q1, children,
                 min_vox: int):
    """Compute the trust-anchor (n_perm, num_reg) LLR draws.

    For each draw, FL-permute via exp.permute(base_seed + i), then iterate
    (reg_idx, size, e, h) per region and finish with get_llr(e, h, n=size):
    an independent code path from the batched kernel, which is what makes
    it the anchor.

    q0 / q1 are accepted for interface parity but unused -- iter_mancova
    recomputes the same pair internally.

    Args:
        exp (Experiment): experiment to sample permutations from
        base_seed (int): draw i uses RNG seed base_seed + i
        n_perm (int): number of FL draws, counting the observed
        q0 (np.array): accepted for interface parity, unused
        q1 (np.array): accepted for interface parity, unused
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN

    Returns:
        draws (np.array): (n_perm, num_reg) per-draw LLR, NaN where a
            region is smaller than min_vox or could not be computed
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


def _iter_batched(*, exp, base_seed: int, n_perm: int, q0, q1, children,
                  min_vox: int, perm_chunk: int):
    """Open a batched-kernel chunk generator over the same draws.

    The seed-to-draw mapping handed to glow.graph.iter_llr_perm, in one
    place because cpu_batched and cpu_summary must agree draw for draw.

    Args:
        exp (Experiment): experiment to sample permutations from
        base_seed (int): draw i uses RNG seed base_seed + i
        n_perm (int): number of FL draws, counting the observed
        q0 (np.array): (a0, num_img) nuisance subspace
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN
        perm_chunk (int): draws per yielded chunk

    Returns:
        generator: yields (Pc, num_reg) float64 chunks in draw order
    """
    num_img, num_vox = exp.y.shape[1], exp.y.shape[2]
    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=num_vox)
    perms = np.empty((n_perm, num_img), dtype=np.int64)
    for i in range(n_perm):
        perms[i] = permute._perm_indices(base_seed + i, num_img)
    return glow.graph.iter_llr_perm(
        y=exp.y, q0=q0, q1=q1, perms=perms, leaf_ord=leaf_ord,
        region_l=region_l, region_h=region_h, min_size=min_vox,
        perm_chunk=perm_chunk)


def cpu_batched(*, exp, base_seed: int, n_perm: int, q0, q1, children,
                min_vox: int, perm_chunk: int = 8):
    """Compute the (n_perm, num_reg) LLR draws via the batched kernel.

    A drop-in for cpu_reliable -- same signature, seed-to-draw mapping and
    NaN convention -- riding glow.graph.iter_llr_perm instead of the
    per-region walk, and agreeing with it to fp64 round-off.

    Materializes the matrix, so peak memory is (n_perm, num_reg) float64. A
    fit wants the streaming cpu_summary; this is for callers that need the
    raw draws.

    Args:
        exp (Experiment): experiment to sample permutations from
        base_seed (int): draw i uses RNG seed base_seed + i
        n_perm (int): number of FL draws, counting the observed
        q0 (np.array): (a0, num_img) nuisance subspace
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN
        perm_chunk (int): draws per kernel chunk; a throughput knob only,
            since the chunks are stacked either way

    Returns:
        draws (np.array): (n_perm, num_reg) per-draw LLR, NaN where a
            region is smaller than min_vox or could not be computed
    """
    return np.vstack(list(_iter_batched(
        exp=exp, base_seed=base_seed, n_perm=n_perm, q0=q0, q1=q1,
        children=children, min_vox=min_vox, perm_chunk=perm_chunk)))


@dataclass(frozen=True, eq=False)
class DrawSummary:
    """Everything a GLOW fit keeps of its (n_perm, num_reg) draw matrix.

    Five arrays, all O(num_reg) or O(n_perm), against a matrix that is
    their product. Whether a backend reduces a materialized matrix
    (summarize_draws) or accumulates these while streaming it
    (draws_gpu.gpu_summarize) is invisible to the fit.

    Attributes:
        llr (np.array): (num_reg,) observed per-region LLR -- row 0 of the
            matrix, unstandardized, NaN off the comparison set
        mu (np.array): (num_reg,) per-region mean over every draw, the
            observed row included
        std (np.array): (num_reg,) per-region std (ddof=1) over every
            draw, as measured -- a degenerate column keeps its 0 or NaN
        z_obs (np.array): (num_reg,) observed z, row 0 standardized by
            (mu, std); MaxStatPerm.from_max's stat_obs
        max_stat (np.array): (n_perm,) max z per draw over the comparison
            set, in draw order, entry 0 the observed draw
    """

    llr: np.ndarray
    mu: np.ndarray
    std: np.ndarray
    z_obs: np.ndarray
    max_stat: np.ndarray


def summarize_draws(draws, *, reg_active):
    """Reduce a materialized draw matrix to a DrawSummary.

    Routes the standardization through Analysis.z_score_stat and the row
    maxima through fwer.max_over_active, so it holds no convention of its
    own -- named once so a streaming backend has something to equal.

    Args:
        draws (np.array): (n_perm, num_reg) per-draw LLR, row 0 observed
        reg_active (np.array): (num_reg,) boolean comparison set

    Returns:
        DrawSummary: see the class docstring
    """
    z, mu, std = Analysis.z_score_stat(draws)
    # copies, not row-0 views: the summary outlives draws, and a view would
    # hold the whole matrix alive for one row of it
    return DrawSummary(llr=np.array(draws[0]), mu=mu, std=std,
                       z_obs=np.array(z[0]),
                       max_stat=max_over_active(z, reg_active))


def _chan_combine(chunk, n, mean, m2):
    """Fold one (Pc, num_reg) draw-chunk into running per-region moments.

    The numpy twin of draws_gpu._chan_combine. Chan's parallel-combine rule
    (Chan, Golub & LeVeque 1979) because var << mean^2 here (LLR carries a
    0.5 * size prefactor), where a naive sum-of-squares pass loses the
    variance to cancellation. NaN cells leave the count, which makes the
    finalized moments nanmean / nanstd(ddof=1).

    Args:
        chunk (np.array): (Pc, num_reg) draws, NaN where invalid
        n (np.array): (num_reg,) running valid-sample count
        mean (np.array): (num_reg,) running mean
        m2 (np.array): (num_reg,) running sum of squared deviations

    Returns:
        n, mean, m2 (np.array): the updated (num_reg,) accumulators
    """
    valid = ~np.isnan(chunk)
    safe = np.where(valid, chunk, 0.0)
    n_b = valid.sum(axis=0).astype(np.float64)
    mean_b = safe.sum(axis=0) / np.where(n_b > 0, n_b, 1.0)
    dev = np.where(valid, safe - mean_b[None, :], 0.0)
    m2_b = (dev * dev).sum(axis=0)

    new_n = n + n_b
    safe_new_n = np.where(new_n > 0, new_n, 1.0)
    delta = mean_b - mean
    return (new_n,
            mean + delta * (n_b / safe_new_n),
            m2 + m2_b + delta * delta * (n * n_b / safe_new_n))


def _chan_moments(n, mean, m2):
    """Reduce running (n, mean, m2) to (mu, std).

    Matches np.nanmean / np.nanstd(ddof=1) as z_score_stat calls them: NaN
    mu where no valid sample accumulated, NaN std where fewer than two did,
    and ULP-level negative variance clamped to zero before the sqrt.

    Args:
        n (np.array): (num_reg,) valid-sample count
        mean (np.array): (num_reg,) running mean
        m2 (np.array): (num_reg,) running sum of squared deviations

    Returns:
        mu, std (np.array): (num_reg,) each
    """
    mu = np.where(n > 0, mean, np.nan)
    var = np.where(n > 1, m2 / np.where(n > 1, n - 1, 1.0), np.nan)
    return mu, np.where(n > 1, np.sqrt(np.maximum(var, 0.0)), np.nan)


def cpu_summary(*, exp, base_seed: int, n_perm: int, q0, q1, children,
                min_vox: int, reg_active, perm_chunk: int = 8):
    """Summarize the CPU draw matrix without ever forming it.

    Equivalent to summarize_draws(cpu_batched(...)) at O(num_reg + n_perm)
    memory instead of O(n_perm * num_reg); the CPU counterpart of
    draws_gpu.gpu_summarize.

    Two passes, because standardizing needs moments the first pass has not
    finished. Pass one folds each chunk into Chan accumulators for
    (mu, std) and reads the observed row off; pass two re-draws the same
    chunks for the row maxima. Re-drawing is exact -- a chunk is a
    deterministic function of its permutation indices -- and costs 2x the
    LLR work for the memory.

    Args:
        exp (Experiment): experiment to sample permutations from
        base_seed (int): draw i uses RNG seed base_seed + i; 0 puts the
            observed draw in row 0
        n_perm (int): number of FL draws, counting the observed
        q0 (np.array): (a0, num_img) nuisance subspace
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN
        reg_active (np.array): (num_reg,) boolean comparison set
        perm_chunk (int): draws per kernel chunk

    Returns:
        DrawSummary: see the class docstring
    """
    kwargs = dict(exp=exp, base_seed=base_seed, n_perm=n_perm, q0=q0,
                  q1=q1, children=children, min_vox=min_vox,
                  perm_chunk=perm_chunk)
    num_reg = exp.y.shape[2] + children.shape[0]

    # Pass 1: (mu, std). Row 0's raw LLR is unstandardized, so it owes
    # nothing to the moments and is read off here.
    n = np.zeros(num_reg, dtype=np.float64)
    mean = np.zeros(num_reg, dtype=np.float64)
    m2 = np.zeros(num_reg, dtype=np.float64)
    llr_obs = None
    for chunk in _iter_batched(**kwargs):
        if llr_obs is None:
            llr_obs = np.array(chunk[0])
        n, mean, m2 = _chan_combine(chunk, n, mean, m2)
    mu, std = _chan_moments(n, mean, m2)

    # Pass 2: z against those moments, then the two reductions a fit keeps.
    # Z_STD_FLOOR and max_over_active are shared with z_score_stat, so the
    # floor and the max-z convention each live in one place.
    denom = np.where(std > Z_STD_FLOOR, std, 1.0)
    max_stat = np.empty(0, dtype=np.float64)
    z_obs = None
    for chunk in _iter_batched(**kwargs):
        z = (chunk - mu[None, :]) / denom[None, :]
        if z_obs is None:
            z_obs = np.array(z[0])
        max_stat = np.concatenate(
            [max_stat, max_over_active(z, reg_active)])

    return DrawSummary(llr=llr_obs, mu=mu, std=std, z_obs=z_obs,
                       max_stat=max_stat)


# eq=False for the same reason MaxStatPerm gives: a generated __eq__ would
# compare array fields pairwise and raise on the ambiguous truth value.
