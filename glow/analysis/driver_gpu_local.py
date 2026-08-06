"""Pipelined local driver: CPU pool for Ward, one GPU for the inner perms.

AnalysisGLOW.fit parallelises whole outer permutations across processes,
which is right on CPU but wrong with a device backend: each worker would
build its own CUDA context (~300-400 MB apiece on an 8 GB card) and they
would serialise on the device anyway. This driver splits the phases
instead, so Ward keeps the whole CPU pool while exactly one stream owns
the GPU:

  Phase A (CPU pool, parallel) -- per outer perm k: permute, Ward cluster,
    observed LLR. Returns only (k, children, llr, size); at num_vox=25k
    that is ~400 KB per tree rather than a 19 MB copy of y.
  Phase B (GPU funnel, as-arrive) -- inner_perm_gpu.prep_shared once for
    the whole fit, then one gpu_perm_shared call per tree carrying just
    its outer-perm index. The float64 nuisance split, the transfer and the
    T_v contraction therefore happen once, not once per outer perm.
  Phase C (serial) -- FWER synthesis via AnalysisGLOW.finalize.

Seeds and the outer-perm reduction come from AnalysisGLOW itself
(_INNER_SEED_BLOCK, reduce_outer), so a fit here is comparable to
AnalysisGLOW.fit rather than merely similar -- see
test_driver_gpu_local.py, which holds the two to float round-off on z,
the FWER null, the p-values and the discovered effect list.

The device backend is a numerical variant, not a different estimator: it
draws the same permutations from the same seeds and reduces them through
the same Chan accumulator. Nothing here belongs in the recipe hash.
"""
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import glow.graph
from glow.experiment.exper import ExperimentScaled
from . import inner_perm_gpu
from ._glow import AnalysisGLOW, _INNER_SEED_BLOCK, reduce_outer
from .cluster import ClusterMode, cluster
from .mancova import decompose


def _phase_a(exp, k: int, *, q0, q1, cluster_mode):
    """Run one outer perm's CPU half: permute, Ward, observed LLR.

    Pure (no self, no device work) so joblib workers can run it. Mirrors
    the first half of AnalysisGLOW._run_outer.

    Returns:
        k (int): the outer-perm index, so an unordered generator still
            keys its results correctly
        children (np.array): (num_reg - num_vox, 2) this perm's Ward tree
        llr (np.array): (num_reg,) observed region LLR
        size (np.array): (num_reg,) region voxel count
    """
    _exp = exp.permute(k) if k else exp
    children = cluster(_exp, mode=cluster_mode)
    llr, size = glow.graph.compute_llr_batched(
        _exp, children=children, q0=q0, q1=q1)
    return k, children, llr, size


def driver_gpu_local(exp, *, n_perm_fwer: int, n_perm_inner: int = 250,
                     alpha_fwer: float = 0.05, min_vox: int = 1,
                     cluster_mode: ClusterMode = ClusterMode.FOCUS,
                     n_jobs_cpu: int = -1, perm_chunk: int = 16,
                     device: str = 'cuda', acc_dtype=np.float64,
                     verbose: bool = False) -> AnalysisGLOW:
    """Fit AnalysisGLOW with Ward on the CPU pool and inner perms on the GPU.

    Args:
        exp (Experiment): experiment to analyze; scaled here exactly as
            AnalysisGLOW.fit scales it (ExperimentScaled.from_exp)
        n_perm_fwer (int): outer FL permutations feeding the max-z null
        n_perm_inner (int): inner FL permutations per outer perm
        alpha_fwer (float): family-wise error rate
        min_vox (int): smallest region size admitted to the FWER set
        cluster_mode (ClusterMode): Ward projection mode
        n_jobs_cpu (int): joblib workers for Phase A. Each holds its own
            permuted copy of y, so cap this at full-brain scale (~1 GB per
            worker at num_vox=224619, b=6).
        perm_chunk (int): inner draws per device chunk; see
            inner_perm_gpu.gpu_perm for why 16 rather than bigger
        device (str): torch device string
        acc_dtype: device hot-loop dtype. Defaults to float64,
            which reproduces AnalysisGLOW.fit's p-values exactly at
            every b measured; float32 is ~2.3x faster but perturbs
            max_z_null by ~2e-3 relative, enough to flip a handful
            of p-values on real data.
        verbose (bool): progress bar and finalize prints

    Returns:
        AnalysisGLOW: fitted -- observed-tree attributes, max_z_null, pval
            and effect_list populated, as AnalysisGLOW.fit leaves them
    """
    exp = ExperimentScaled.from_exp(exp)
    q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)
    n_total = n_perm_fwer + 1

    if verbose:
        print(f'  [1/2] {n_total} outer perms '
              f'({exp.y.shape[2]} voxels, {n_perm_inner} inner, '
              f'n_jobs_cpu={n_jobs_cpu}, device={device}) ...')

    # Dispatch Phase A BEFORE touching CUDA: the pool's processes should be
    # up before this one initialises a device context. The generator is
    # unordered because each result carries its own k.
    phase_a = Parallel(n_jobs=n_jobs_cpu, return_as='generator_unordered')(
        delayed(_phase_a)(exp, k, q0=q0, q1=q1, cluster_mode=cluster_mode)
        for k in range(n_total))

    shared = inner_perm_gpu.prep_shared(
        exp, q0=q0, q1=q1, device=device, acc_dtype=acc_dtype)

    ana = AnalysisGLOW(
        n_perm_fwer=n_perm_fwer, n_perm_inner=n_perm_inner,
        alpha_fwer=alpha_fwer, min_vox=min_vox, cluster_mode=cluster_mode)
    ana.max_z_null = np.empty(n_total)

    for k, children, llr, size in tqdm(phase_a, total=n_total,
                                       desc='ward+gpu_perm',
                                       disable=not verbose):
        mu, std = inner_perm_gpu.gpu_perm_shared(
            shared, children=children,
            base_seed=(k + 1) * _INNER_SEED_BLOCK, n_perm=n_perm_inner,
            min_vox=min_vox, outer_perm=k, perm_chunk=perm_chunk)

        z, ana.max_z_null[k] = reduce_outer(llr, mu, std, size, min_vox)
        if k == 0:
            ana.children = children
            ana.size = size
            ana.llr = llr
            ana.mu = mu
            ana.std = std
            ana.z = z

    if verbose:
        print('  [2/2] FWER synthesis ...')
    ana.finalize(exp, verbose=verbose)
    return ana
