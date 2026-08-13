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
"""
import numpy as np

import glow.graph
from glow.analysis import mancova


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
