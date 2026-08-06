"""Device backend behind AnalysisGLOW.fit(gpu=...): CPU Ward, one GPU inner.

AnalysisGLOW.fit parallelises whole outer permutations across processes,
which is right on CPU but wrong with a device backend: each worker would
build its own CUDA context (~300-400 MB apiece on an 8 GB card) and they
would serialise on the device anyway. fit_gpu splits the phases instead,
so Ward keeps the whole CPU pool while exactly one stream owns the GPU:

  Phase A (CPU pool, parallel) -- per outer perm k: permute, Ward cluster,
    observed LLR. Returns only (k, children, llr, size); at num_vox=25k
    that is ~400 KB per tree rather than a 19 MB copy of y.
  Phase B (GPU funnel, as-arrive) -- inner_perm_gpu.prep_shared once for
    the whole fit, then one gpu_perm_shared call per tree carrying just
    its outer-perm index. The float64 nuisance split, the transfer and the
    T_v contraction therefore happen once, not once per outer perm.
  Phase C (serial) -- FWER synthesis via AnalysisGLOW.finalize.

Seeds and the outer-perm reduction come from AnalysisGLOW itself
(_INNER_SEED_BLOCK, reduce_outer), so a device fit is comparable to a CPU
fit rather than merely similar -- see test_fit_gpu.py, which holds the two
to float round-off on z, the FWER null, the p-values and the discovered
effect list.

The device is therefore a hardware choice, not an estimator choice, and
nothing here belongs in the recipe hash: the benchmark runs one cell on
CPU and the next on GPU and files both as the same artifact (see
glow._extra.benchmark.run.FIT_IGNORE). That holds at GpuConfig's default
float64; float32 is opt-in and never what gpu=True selects.
"""
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import glow.graph
from glow.experiment.exper import ExperimentScaled
from . import inner_perm_gpu
from ._base import resolve_n_jobs
from ._glow import _INNER_SEED_BLOCK, reduce_outer
from .cluster import cluster
from .mancova import decompose


@dataclass(frozen=True)
class GpuConfig:
    """Device-backend knobs for AnalysisGLOW.fit(gpu=...).

    Pass one in place of gpu=True to tune the backend; the defaults are
    what gpu=True and gpu='auto' select.

    Attributes:
        device (str): torch device string
        perm_chunk (int): inner draws per device chunk; see
            inner_perm_gpu.gpu_perm for why 16 rather than bigger
        acc_dtype (type): device hot-loop dtype. float64 reproduces a CPU
            fit's p-values exactly at every b measured; float32 is ~2.3x
            faster but perturbs max_z_null by ~2e-3 relative, enough to
            flip a handful of p-values on real data. Pass float32 only
            when timing the backend itself: it makes the fit a different
            artifact from the CPU one, which the benchmark's cache key
            does not model (see the module docstring).
    """

    device: str = 'cuda'
    perm_chunk: int = 16
    acc_dtype: type = np.float64


def resolve_gpu(gpu, *, name: str = 'fit'):
    """Normalise a fit(gpu=...) argument into a GpuConfig, or None for CPU.

    Args:
        gpu: False / None to stay on the CPU; True to require a device;
            'auto' to take one when visible and fall back to the CPU when
            not; a GpuConfig to require a device with tuned knobs.
        name (str): caller name, for the error message only.

    Returns:
        GpuConfig | None: the resolved knobs, None to run on the CPU.

    Raises:
        RuntimeError: a device was required (True or a GpuConfig) and none
            is visible.
        TypeError: gpu is none of the accepted forms.
    """
    if not gpu:
        return None
    if gpu == 'auto':
        return GpuConfig() if inner_perm_gpu.is_available() else None
    if gpu is True:
        gpu = GpuConfig()
    if not isinstance(gpu, GpuConfig):
        raise TypeError(f"gpu must be bool, 'auto' or GpuConfig, "
                        f'got {gpu!r}')
    if not inner_perm_gpu.is_available():
        raise RuntimeError(
            f'{name}(gpu=...) needs a visible CUDA device: torch is not '
            f"installed or torch.cuda.is_available() is False. Pass "
            f"gpu='auto' to fall back to the CPU instead.")
    return gpu


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


def fit_gpu(ana, exp, *, n_jobs: int = -1, gpu_config: GpuConfig = None,
            verbose: bool = False):
    """Fit ana with Ward on the CPU pool and the inner perms on one GPU.

    The backend AnalysisGLOW.fit dispatches to for a truthy gpu; it reads
    the recipe knobs off ana and populates it, so the returned object is
    the one fit was called on, left exactly as a CPU fit leaves it.

    Args:
        ana (AnalysisGLOW): the recipe to fit, populated in place.
        exp (Experiment): experiment to analyze; scaled here exactly as
            AnalysisGLOW.fit scales it (ExperimentScaled.from_exp).
        n_jobs (int): joblib workers for Phase A (the Ward pool, not the
            outer-perm split a CPU fit uses). Each holds its own permuted
            copy of y, so RAM caps this before cores do -- ~1 GB per
            worker at num_vox=224619, b=6.
        gpu_config (GpuConfig): device knobs; None takes the defaults.
        verbose (bool): progress bar and finalize prints.

    Returns:
        ana (AnalysisGLOW): fitted -- observed-tree attributes, max_z_null,
            pval and effect_list populated.
    """
    if gpu_config is None:
        gpu_config = GpuConfig()
    n_jobs = resolve_n_jobs(n_jobs)

    exp = ExperimentScaled.from_exp(exp)
    q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)
    n_total = ana.n_perm_fwer + 1

    if verbose:
        print(f'  [1/2] {n_total} outer perms '
              f'({exp.y.shape[2]} voxels, {ana.n_perm_inner} inner, '
              f'n_jobs={n_jobs}, device={gpu_config.device}) ...')

    # Dispatch Phase A BEFORE touching CUDA: the pool's processes should be
    # up before this one initialises a device context. The generator is
    # unordered because each result carries its own k.
    phase_a = Parallel(n_jobs=n_jobs, return_as='generator_unordered')(
        delayed(_phase_a)(exp, k, q0=q0, q1=q1,
                          cluster_mode=ana.cluster_mode)
        for k in range(n_total))

    shared = inner_perm_gpu.prep_shared(
        exp, q0=q0, q1=q1, device=gpu_config.device,
        acc_dtype=gpu_config.acc_dtype)

    ana.max_z_null = np.empty(n_total)

    for k, children, llr, size in tqdm(phase_a, total=n_total,
                                       desc='ward+gpu_perm',
                                       disable=not verbose):
        mu, std = inner_perm_gpu.gpu_perm_shared(
            shared, children=children,
            base_seed=(k + 1) * _INNER_SEED_BLOCK,
            n_perm=ana.n_perm_inner, min_vox=ana.min_vox, outer_perm=k,
            perm_chunk=gpu_config.perm_chunk)

        z, ana.max_z_null[k] = reduce_outer(llr, mu, std, size, ana.min_vox)
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
