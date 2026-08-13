"""CPU Freedman-Lane draw matrix: the trust anchor.

Given a Ward tree and an experiment, draw n_perm Freedman-Lane samples
(Freedman & Lane 1983) and return the per-region LLR for each, as one
(n_perm, num_reg) matrix. That matrix is GLOW's whole hypothesis family:
its column moments standardize the regions, its row maxima are the max-z
null, and row 0 is the observed draw (see AnalysisGLOW).

cpu_reliable is deliberately the slow implementation -- per draw, per
region, through iter_mancova + get_llr with no batching and no hoisting.
It exists to be obviously correct, so the device backend
(glow.analysis.draws_gpu.gpu_perm) can be validated against it cell by
cell. The two share a keyword-only signature and a seed-to-draw mapping:

    draws = draws.cpu_reliable(
        exp=exp, base_seed=0, n_perm=n_perm_fwer + 1,
        q0=q0, q1=q1, children=children, min_vox=min_vox)

base_seed is the starting RNG seed: draw i uses base_seed + i. Seed 0 is
the identity (permute._perm_indices), so base_seed=0 puts the observed
draw in row 0 and the null in rows 1:.

For the batched CPU alternative -- one GEMM plus cumsum-and-diff over the
DFS pre-order voxel axis per perm-chunk, ~60x faster at full-brain
num_vox -- see glow.graph.iter_llr_perm, which yields chunks a caller can
stack or reduce as it likes.

DrawSummary is the other half of this module: the five arrays a fit keeps
of a draw matrix, and so the contract a streaming backend has to meet. The
matrix runs to ~16.7 GiB at 5001 draws and full-brain num_vox, which is why
the device path never forms one (draws_gpu.gpu_summarize). summarize_draws
is that same reduction taken over a materialized matrix -- the reference
the streaming one is held against.
"""
from dataclasses import dataclass

import numpy as np

import glow.graph
from glow.analysis import mancova
from ._base import Analysis
from .fwer import max_over_active


def cpu_reliable(*, exp, base_seed: int, n_perm: int, q0, q1, children,
                 min_vox: int):
    """Compute the trust-anchor (n_perm, num_reg) LLR draws.

    Thin wrapper over the per-region iter_mancova + get_llr path: for
    each draw, FL-permute via exp.permute(base_seed + i), then iterate
    (reg_idx, size, e, h) per region and finish with
    get_llr(e, h, n=size). No batching, no closed-form 2x2 slogdet, no
    Phase-1 hoisting -- an independent code path from
    compute_llr_batched for cross-validating the optimised backends.

    q0 / q1 are accepted for interface parity but unused: iter_mancova
    recomputes them internally via decompose(exp.x, exp.contrast), which
    is exactly the same (q0, q1) the caller would have passed in.

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


# eq=False for the same reason MaxStatPerm gives: a generated __eq__ would
# compare array fields pairwise and raise on the ambiguous truth value.
@dataclass(frozen=True, eq=False)
class DrawSummary:
    """Everything a GLOW fit keeps of its (n_perm, num_reg) draw matrix.

    Five arrays, all O(num_reg) or O(n_perm), against a matrix that is
    their product. Whether a backend materializes the matrix and reduces it
    (summarize_draws) or accumulates these while streaming it
    (draws_gpu.gpu_summarize) is an implementation choice the fit cannot
    see, which is what lets the two be swapped and compared.

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
    maxima through fwer.max_over_active, so this holds no convention of its
    own -- it is the composition a fit would otherwise write inline, named
    once so a streaming backend has something to be equal to.

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
