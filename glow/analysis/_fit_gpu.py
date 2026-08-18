"""Device backend for the GLOW arms' fit(gpu=...).

Two jobs: normalise the fit(gpu=...) argument into GpuConfig knobs, and
draw a (n_perm, num_reg) Freedman-Lane matrix on device. The reduction that
follows is the CPU path's own -- Analysis.z_score_stat then MaxStatPerm --
so the backends cannot drift in anything but the draws they hand it.

The draws come from draws_gpu.gpu_perm, against whichever tree and
experiment the caller fixed: AnalysisGLOWSplit's one fold-A tree over
n_perm_fwer + 1 draws, or one of AnalysisGLOW's per-perm trees over its
n_perm_inner + 1. Either way base_seed = 0, so row 0 is the observed draw
(permute._perm_indices reserves seed 0 for it), and the backend needs no
structure of its own beyond chunk sizing.

acc_dtype defaults to float64 because Analysis.fit documents a fit as
identical on either device, which is what keeps gpu out of RECORD_FIELDS
and out of the benchmark cache key. A GpuConfig can ask for float32, which
is faster but perturbs fwer.max_stat enough to flip a handful of p-values.

perm_chunk is sized from free device memory rather than fixed, since
_chunk_llr's peak allocation grows as perm_chunk * a0 * b^2 * num_vox (see
resolve_perm_chunk).
"""
from dataclasses import dataclass

import numpy as np

from . import draws_gpu


# Largest chunk worth taking: throughput is set by whether the per-chunk
# working set stays L2-resident, not by capacity.
_PERM_CHUNK_MAX = 16

# Per-draw device bytes, as a multiple of the dominant
# (perm_chunk, a0, b, b, num_vox) broadcast inside _chunk_llr's
# _gram_a_cross: that broadcast, its cumsum copy, alpha and the region
# readouts are live together.
_CHUNK_BYTES_PER_TERM = 7

# Share of free device memory the chunk loop may plan to occupy; the rest
# absorbs fragmentation and the region-space temporaries.
_DEVICE_MEM_FRACTION = 0.8


@dataclass(frozen=True)
class GpuConfig:
    """Device-backend knobs for a GLOW fit(gpu=...).

    Attributes:
        device (str): torch device string
        perm_chunk (int | None): draws per device chunk. None sizes it from
            free device memory (resolve_perm_chunk); an int pins it, which
            is what the chunk-invariance tests use.
        acc_dtype (type): device hot-loop dtype. float64 reproduces a CPU
            fit's p-values exactly; float32 is faster but perturbs
            fwer.max_stat enough to flip a handful of them.
    """

    device: str = 'cuda'
    perm_chunk: int | None = None
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
        return GpuConfig() if draws_gpu.is_available() else None
    if gpu is True:
        gpu = GpuConfig()
    if not isinstance(gpu, GpuConfig):
        raise TypeError(f"gpu must be bool, 'auto' or GpuConfig, "
                        f'got {gpu!r}')
    if not draws_gpu.is_available():
        raise RuntimeError(
            f'{name}(gpu=...) needs a visible CUDA device: torch is not '
            f"installed or torch.cuda.is_available() is False. Pass "
            f"gpu='auto' to fall back to the CPU instead.")
    return gpu


def describe_backend(gpu, config, *, cpu_anchor: bool = False) -> str:
    """Name the draw backend a fit resolved, and why it got that one.

    The three backends span orders of magnitude, and two ways of landing
    on a slow one are silent: gpu='auto' drops to the CPU without a word
    when no device is visible (a CPU-only torch wheel leaves nvidia-smi
    still listing the card), and cpu_anchor=True is easy to leave set.

    Args:
        gpu: the fit(gpu=...) argument, as the caller passed it.
        config (GpuConfig | None): what resolve_gpu made of it.
        cpu_anchor (bool): the fit(cpu_anchor=...) argument.

    Returns:
        text (str): the backend, then the reason in parentheses.
    """
    if config is not None:
        why = "gpu='auto' found one" if gpu == 'auto' else 'requested'
        return (f'device {config.device} ({why}, '
                f'{np.dtype(config.acc_dtype).name} accumulation)')
    if cpu_anchor:
        return ('CPU draws.cpu_reliable, the slow trust anchor '
                '(cpu_anchor=True; ~35x the batched path, and meant for '
                'holding it honest rather than for fitting)')
    fast = 'CPU draws.cpu_summary, the batched kernel streamed'
    if gpu == 'auto':
        return (f"{fast} (gpu='auto' found no device: "
                f'{draws_gpu.unavailable_reason()})')
    return (f'{fast} (no device asked for; pass gpu=True, or '
            f"gpu='auto' to take one only when visible)")


def resolve_perm_chunk(config: GpuConfig, *, b: int, num_img: int,
                       num_vox: int, a0: int) -> int:
    """Pick the largest chunk of draws whose working set fits on device.

    _chunk_llr's peak allocation is dominated by the
    (perm_chunk, a0, b, b, num_vox) broadcast inside _gram_a_cross, so the
    per-draw cost grows as a0 * b^2 * num_vox and one fixed figure cannot
    serve both benchmark and full-brain num_vox. Clamping down costs little
    where memory is tight, the voxel axis already saturating the device.

    The estimate is deliberately crude -- a factor on the dominant term
    (_CHUNK_BYTES_PER_TERM), the prep state that outlives every chunk, and
    a fraction of free memory held back for fragmentation -- since it only
    has to land in the right power of two.

    Args:
        config (GpuConfig): the resolved knobs
        b (int): imaging features
        num_img (int): images, which size the residual stack held in prep
        num_vox (int): voxels
        a0 (int): nuisance features

    Returns:
        perm_chunk (int): draws per device chunk, at least 1. Returns
            config.perm_chunk unchanged when it names an int.
    """
    if config.perm_chunk is not None:
        return int(config.perm_chunk)

    import torch

    itemsize = np.dtype(config.acc_dtype).itemsize
    num_reg = 2 * num_vox - 1

    # prep state, live for the whole loop: the residual stack U dominates,
    # the region-space float64 scans trail it
    prep_bytes = (b * num_img * num_vox * itemsize
                  + 2 * b * b * num_reg * np.dtype(np.float64).itemsize)
    per_draw = _CHUNK_BYTES_PER_TERM * a0 * b * b * num_vox * itemsize

    free, _ = torch.cuda.mem_get_info(torch.device(config.device))
    budget = _DEVICE_MEM_FRACTION * free - prep_bytes
    return int(min(_PERM_CHUNK_MAX, max(1, budget // max(per_draw, 1))))


def gpu_draws(config: GpuConfig, *, exp, base_seed: int, n_perm: int,
              q0, q1, children, min_vox: int):
    """Draw the matrix on device and return it whole.

    The device counterpart of draws.cpu_batched, sized and dtyped from the
    same GpuConfig gpu_summary uses. A GLOW fit takes this route under
    keep_stat, where the caller wants every cell; peak host memory is the
    whole (n_perm, num_reg) float64 matrix, so prefer gpu_summary wherever
    the moments, the observed row and the row maxima are enough.

    Args:
        config (GpuConfig): resolved device knobs
        exp (Experiment): experiment to draw permutations from
        base_seed (int): draw i uses seed base_seed + i; 0 puts the
            observed draw in row 0
        n_perm (int): number of FL draws, counting the observed
        q0 (np.array): (a0, num_img) nuisance subspace
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN

    Returns:
        draws (np.array): (n_perm, num_reg) per-draw LLR, row 0 observed
    """
    b, num_img, num_vox = exp.y.shape
    perm_chunk = resolve_perm_chunk(
        config, b=b, num_img=num_img, num_vox=num_vox, a0=q0.shape[0])
    return draws_gpu.gpu_perm(
        exp=exp, base_seed=base_seed, n_perm=n_perm, q0=q0, q1=q1,
        children=children, min_vox=min_vox, perm_chunk=perm_chunk,
        device=config.device, acc_dtype=config.acc_dtype)


def gpu_summary(config: GpuConfig, *, exp, base_seed: int, n_perm: int,
                q0, q1, children, min_vox: int, reg_active=None):
    """Summarize the draws on device, never materializing the matrix.

    The device counterpart of draws.cpu_summary -- same two-pass shape, same
    DrawSummary -- and a drop-in for it wherever a GLOW arm draws. Sizes
    perm_chunk first (see resolve_perm_chunk), then defers to
    draws_gpu.gpu_summarize.

    Args:
        config (GpuConfig): resolved device knobs
        exp (Experiment): experiment to draw permutations from
        base_seed (int): draw i uses seed base_seed + i; 0 puts the
            observed draw in row 0
        n_perm (int): number of FL draws, counting the observed
        q0 (np.array): (a0, num_img) nuisance subspace
        q1 (np.array): (a1, num_img) interest subspace
        children (np.array): (num_reg - num_vox, 2) Ward tree
        min_vox (int): regions smaller than this are left NaN
        reg_active (np.array): (num_reg,) boolean comparison set

    Returns:
        DrawSummary: see glow.analysis.draws.DrawSummary
    """
    b, num_img, num_vox = exp.y.shape
    perm_chunk = resolve_perm_chunk(
        config, b=b, num_img=num_img, num_vox=num_vox, a0=q0.shape[0])
    return draws_gpu.gpu_summarize(
        exp=exp, base_seed=base_seed, n_perm=n_perm, q0=q0, q1=q1,
        children=children, min_vox=min_vox, reg_active=reg_active,
        perm_chunk=perm_chunk, device=config.device,
        acc_dtype=config.acc_dtype)
