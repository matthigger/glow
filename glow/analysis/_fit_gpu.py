"""Device backend for AnalysisGLOW.fit(gpu=...) -- offline, pending a port.

The backend this module used to hold was built for the pre-split
architecture: Ward on a CPU pool once per outer permutation, the inner
draws for each of those trees funnelled through one CUDA stream. That
shape no longer exists. AnalysisGLOW now builds ONE tree, on a held-out
segmentation fold, and flattens the permutations into two disjoint
streams over the test fold (see glow.analysis._glow), so there is no
per-outer-perm tree to pool and no nested inner loop to funnel.

Porting it is a small job rather than a rewrite -- inner_perm_gpu already
has the pieces (prep_shared once for the fit, prep_tree once for the one
tree, _chunk_llr per chunk). What it needs is to assemble those chunks
into the one (n_perm_fwer + 1, num_reg) matrix the CPU path builds, and
hand it to the same Analysis.z_score_stat / MaxStatPerm reduction, so the
two backends cannot drift.

It is not urgent. Flattening the nesting took a fit from
n_perm_fwer * n_perm_inner draws to n_perm_fwer + 1 -- at the benchmark's
500 x 250 that is ~126,000 down to 501, plus 501 Ward builds down to 1.
That is a larger factor than the device ever bought on the inner loop, so
the CPU path is no longer the bottleneck the device existed to relieve.
The current CPU draws come from cpu_reliable_full, the slow trust-anchor
backend; moving them to the batched one is the cheaper win to take
first.

GpuConfig and resolve_gpu stay because they are the device-detection
surface the package re-exports and the tests probe; only the fit itself
is gone.
"""
from dataclasses import dataclass

import numpy as np

from . import inner_perm_gpu


@dataclass(frozen=True)
class GpuConfig:
    """Device-backend knobs for AnalysisGLOW.fit(gpu=...).

    Retained for the pending port (see the module docstring); nothing
    consumes it today.

    Attributes:
        device (str): torch device string
        perm_chunk (int): draws per device chunk; see
            inner_perm_gpu.gpu_perm for why 16 rather than bigger
        acc_dtype (type): device hot-loop dtype. float64 reproduced a CPU
            fit's p-values exactly at every b measured; float32 was ~2.3x
            faster but perturbed fwer.max_stat by ~2e-3 relative, enough to
            flip a handful of p-values on real data.
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


def fit_gpu(ana, exp, **kwargs):
    """Refuse a device fit while the backend is being ported.

    A stub rather than a deletion so a stale caller gets told what
    happened instead of an ImportError.

    Raises:
        NotImplementedError: always.
    """
    raise NotImplementedError(
        'the AnalysisGLOW device backend is offline pending its port to '
        'the split architecture; fit on the CPU (see _fit_gpu.__doc__)')
