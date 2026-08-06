"""Threshold-Free Cluster Enhancement (TFCE).

Two backends compute TFCE(p) = sum_h e(h)^E * h^H (Smith & Nichols 2009):

    python  apply_tfce_img -- scipy connected components over n_steps
            height thresholds. Always available, and the reference.
    fsl     fslmaths -tfce -- found on $FSLDIR / $PATH. Roughly 4x faster
            on a permutation stack, because fslmaths treats a 4d input as
            independent 3d volumes (bit-identically to one call per
            volume), so one subprocess enhances every permutation.

resolve_backend picks fsl when the binary is present and the request is
one fslmaths can serve -- it hardcodes 100 height steps and reads 3d or
4d images only -- and python otherwise.

The two are close but not equal, and it comes down to one endpoint. FSL
loops "for (float curThr = 0; curThr < maxT + deltaT; curThr += deltaT)"
accumulating curThr in float32, and admits voxels strictly above curThr.
So its last step lands either just below maxT, and the peak voxel still
counts, or just above it and the peak drops out -- decided by the low
bits of maxT, near enough a coin flip. apply_tfce_img steps an exact
linspace(dh, max, n_steps) and admits h and above, so it always counts
that step. When FSL drops it the peak comes out 100^H / sum_k k^H lower,
3.0% at H=2. Replaying the loop in float32 reproduces fslmaths bit for
bit, so that is the whole of the difference; neither is more correct,
both being 100-step Riemann sums that disagree on one endpoint.

Little of it survives into a result, since FWER reads only the
per-permutation maximum, which usually sits in a cluster rather than on
the peak voxel: on 25k-voxel null data the max-stat null moves by a
median 6e-7 relative and 0.9% of voxels shift their p-value, each by one
permutation rank. So the backend is a speed knob, kept out of
RECORD_FIELDS so a cached fit does not re-key on whether its machine had
FSL -- interchangeable for discovery, not for reproducing a stored
p-value exactly.
"""

import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from shutil import which

import numpy as np
from scipy.ndimage import label, generate_binary_structure

from glow.mask import bbox_crop

# Default backend for every call that does not name one: 'auto' (prefer
# fsl when usable), 'fsl' (demand it, raise if unusable) or 'python'.
BACKEND = 'auto'

# fslmaths -tfce derives its step from the image, deltaT = max/100, so it
# serves n_steps=100 only. Another count would mean passing the
# undocumented -tfce_delta; nothing here needs one, so it goes to python.
FSL_N_STEPS = 100

# Cap on one fslmaths call's in-memory stack, which bounds both the
# temporary NIfTI and the float64 array built to hold it.
MAX_CHUNK_BYTES = 512 * 1024 ** 2

_INSTALL_URL = 'https://fsl.fmrib.ox.ac.uk/fsl/docs/install/index.html'


@lru_cache(maxsize=None)
def get_fsl_bin(name: str) -> str | None:
    """Locate an FSL binary on $FSLDIR or $PATH.

    Args:
        name (str): binary name, e.g. 'fslmaths'

    Returns:
        path (str or None): absolute path, or None if not installed
    """
    fsldir = os.environ.get('FSLDIR')
    if fsldir:
        for sub in ('share/fsl/bin', 'bin'):
            path = Path(fsldir) / sub / name
            if path.exists():
                return str(path)
    return which(name)


def has_fsl() -> bool:
    """Report whether the fslmaths binary is discoverable."""
    return get_fsl_bin('fslmaths') is not None


def resolve_backend(backend: str = None, *, ndim: int = 3,
                    n_steps: int = FSL_N_STEPS) -> str:
    """Choose the TFCE backend for one request.

    Args:
        backend (str or None): 'auto', 'fsl' or 'python'. None takes the
            module default, BACKEND.
        ndim (int): dimensionality of the images to enhance
        n_steps (int): requested height steps

    Returns:
        name (str): 'fsl' or 'python'

    Raises:
        ValueError: if backend is not a recognised name
        RuntimeError: if fsl is demanded but cannot serve the request
    """
    if backend is None:
        backend = BACKEND
    if backend not in ('auto', 'fsl', 'python'):
        raise ValueError(f"backend must be auto, fsl or python: {backend!r}")

    if ndim != 3:
        why = f'fslmaths reads 3d or 4d images only, got {ndim}d'
    elif n_steps != FSL_N_STEPS:
        why = f'fslmaths -tfce uses {FSL_N_STEPS} height steps, got {n_steps}'
    elif not has_fsl():
        why = ('fslmaths not found on $FSLDIR or $PATH '
               f'(install: {_INSTALL_URL})')
    else:
        why = None

    if backend == 'python':
        return 'python'
    if backend == 'fsl':
        if why:
            raise RuntimeError(f'fsl TFCE backend unavailable: {why}')
        return 'fsl'
    return 'python' if why else 'fsl'


def apply_tfce_img(x, H: float = 2.0, E: float = 0.5,
                   connectivity: int = None, n_steps: int = 100):
    """Apply TFCE to a 2d or 3d statistical image.

    Computes TFCE(p) = sum_h e(h)^E * h^H where e(h) is the cluster
    extent at threshold h (Smith & Nichols 2009).

    Args:
        x (np.array): 2d or 3d array of statistical values
        H (float): height exponent
        E (float): extent exponent
        connectivity (int or None): neighbourhood size.
            3d: 6 (faces, default), 18 (faces+edges), or 26 (full).
            2d: 4 (edges, default) or 8 (edges+corners).
            None selects face/edge connectivity for the input ndim.
        n_steps (int): number of threshold steps (100 matches FSL)

    Returns:
        tfce (np.array): TFCE-enhanced image, same shape as x
    """
    x = np.asarray(x)
    ndim = x.ndim
    assert ndim in (2, 3), f'expected 2d or 3d input, got {ndim}d'

    # connectivity structure for scipy.ndimage.label
    if connectivity is None:
        struct = generate_binary_structure(ndim, 1)
    elif ndim == 3:
        if connectivity == 6:
            struct = generate_binary_structure(3, 1)
        elif connectivity == 18:
            struct = generate_binary_structure(3, 2)
        else:
            struct = generate_binary_structure(3, 3)
    else:
        if connectivity == 4:
            struct = generate_binary_structure(2, 1)
        else:
            struct = generate_binary_structure(2, 2)

    # nothing exceeds threshold 0, so TFCE is identically zero
    img_max = x.max()
    if img_max <= 0:
        return np.zeros_like(x)

    # FSL uses fixed number of steps with dh = max / n_steps
    dh = img_max / n_steps
    thresholds = np.linspace(dh, img_max, n_steps)

    tfce = np.zeros_like(x)

    for h in thresholds:
        binary = x >= h
        if not binary.any():
            continue

        labeled, n_clusters = label(binary, structure=struct)
        cluster_sizes = np.bincount(labeled.ravel())
        extent_map = cluster_sizes[labeled]
        extent_map[labeled == 0] = 0

        # TFCE contribution: e^E * h^H (no dh factor, matching FSL)
        tfce += (extent_map ** E) * (h ** H)

    return tfce


def apply_tfce_stack_fsl(stack, H: float = 2.0, E: float = 0.5,
                         connectivity: int = None):
    """Apply TFCE to a stack of 3d images with one fslmaths call.

    fslmaths enhances each volume of a 4d input independently, so the
    whole stack costs a single subprocess plus one NIfTI round trip.
    Volumes whose maximum is non-positive are held out of the call and
    returned as zeros: TFCE is identically zero on them, and one such
    volume otherwise aborts the entire run ("tfce requires a positive
    deltaT input").

    Args:
        stack (np.array): (n_img, X, Y, Z) statistical images
        H (float): height exponent
        E (float): extent exponent
        connectivity (int or None): 6 (faces, default), 18 or 26

    Returns:
        tfce (np.array): (n_img, X, Y, Z) float64 enhanced images

    Raises:
        RuntimeError: if fslmaths is missing or exits non-zero
    """
    # lazy import: only the fsl backend pays for nibabel
    import nibabel as nib

    fslmaths = get_fsl_bin('fslmaths')
    if fslmaths is None:
        raise RuntimeError(f'fslmaths not found (install: {_INSTALL_URL})')

    stack = np.asarray(stack)
    assert stack.ndim == 4, f'expected (n_img, X, Y, Z), got {stack.shape}'

    tfce = np.zeros(stack.shape, dtype=np.float64)
    keep_idx = np.flatnonzero(np.nanmax(stack, axis=(1, 2, 3)) > 0)
    if keep_idx.size == 0:
        return tfce

    # fslmaths indexes volumes last; uncompressed NIFTI over NIFTI_GZ
    # saves roughly 20% of the round trip
    img_4d = np.moveaxis(stack[keep_idx], 0, -1)
    env = {**os.environ, 'FSLOUTPUTTYPE': 'NIFTI'}
    conn = 6 if connectivity is None else int(connectivity)

    with tempfile.TemporaryDirectory() as tmp_dir:
        f_in = Path(tmp_dir) / 'tfce_in.nii'
        f_out = Path(tmp_dir) / 'tfce_out.nii'
        nib.save(nib.Nifti1Image(img_4d.astype(np.float32), np.eye(4)),
                 str(f_in))
        proc = subprocess.run(
            [fslmaths, str(f_in), '-tfce', str(H), str(E), str(conn),
             str(f_out)],
            capture_output=True, env=env, text=True)
        if proc.returncode:
            raise RuntimeError(
                f'fslmaths -tfce failed ({proc.returncode}): '
                f'{proc.stderr.strip() or proc.stdout.strip()}')
        # reshape absorbs the squeeze fslmaths applies to a lone volume
        out = np.asarray(nib.load(str(f_out)).dataobj, dtype=np.float64)

    tfce[keep_idx] = np.moveaxis(out.reshape(img_4d.shape), -1, 0)
    return tfce


def get_mask_bb(mask_idx):
    """Crop a mask index to its bounding box and add a zero border.

    TFCE requires the image boundary to be zero, and cropping to the
    active bounding box keeps the enhanced grid as small as possible.

    Args:
        mask_idx (np.array): 2d or 3d index array, -1 for inactive voxels

    Returns:
        mask_bb (np.array): boolean, cropped and 1-voxel padded support.
            Its True entries are in the same order as a (num_vox,) stat
            vector.
    """
    if mask_idx.ndim != 2:
        # keep 2d; only promote higher-dim inputs to a full 3d grid
        mask_idx = np.atleast_3d(mask_idx)

    mask_idx_bb, _ = bbox_crop(mask_idx, mask=(mask_idx > -1))
    mask_idx_bb = np.pad(mask_idx_bb, pad_width=1, constant_values=-1)
    return mask_idx_bb > -1


def apply_tfce_stat(stat, mask_idx, H: float = 2.0, E: float = 0.5,
                    connectivity: int = None, n_steps: int = 100, *,
                    backend: str = None):
    """Apply TFCE to every image of a stat matrix, on either backend.

    Args:
        stat (np.array): (n_img, num_vox) statistical values, one row per
            image and one column per active voxel
        mask_idx (np.array): 2d or 3d index array, -1 for inactive voxels,
            otherwise the voxel index into a stat row
        H (float): height exponent
        E (float): extent exponent
        connectivity (int or None): see apply_tfce_img
        n_steps (int): number of threshold steps (python backend only)
        backend (str or None): 'auto', 'fsl' or 'python' (see
            resolve_backend); None takes the module default

    Returns:
        tfce (np.array): (n_img, num_vox) TFCE stats per active voxel
    """
    mask_bb = get_mask_bb(mask_idx)
    img = np.zeros((len(stat), *mask_bb.shape))
    img[:, mask_bb] = stat

    name = resolve_backend(backend, ndim=mask_bb.ndim, n_steps=n_steps)
    if name == 'fsl':
        tfce = apply_tfce_stack_fsl(img, H=H, E=E, connectivity=connectivity)
    else:
        tfce = np.stack([
            apply_tfce_img(_img, H=H, E=E, connectivity=connectivity,
                           n_steps=n_steps) for _img in img])
    return tfce[:, mask_bb]


def apply_tfce_x(x, mask_idx, *, backend: str = None):
    """Apply TFCE to a vector of stats given a 2d or 3d mask index.

    Crops to the bounding box of active voxels before running TFCE,
    then pads with a single-voxel border of zeros (TFCE requires the
    image boundary to be zero).

    Args:
        x (np.array): (num_vox,) statistical values, one per active voxel
        mask_idx (np.array): 2d or 3d index array, -1 for inactive voxels,
            otherwise the voxel index into x
        backend (str or None): 'auto', 'fsl' or 'python' (see
            resolve_backend); None takes the module default

    Returns:
        tfce (np.array): (num_vox,) TFCE stats, one per active voxel
    """
    return apply_tfce_stat(np.asarray(x)[None], mask_idx, backend=backend)[0]


def get_chunk_slices(n_img: int, n_vox_img: int, n_chunk: int) -> list:
    """Split n_img images into slices for the batched fsl backend.

    Aims for n_chunk equal slices (one per worker), then splits further
    if that would exceed MAX_CHUNK_BYTES of float64 image per call.

    Args:
        n_img (int): number of images to enhance
        n_vox_img (int): voxels in one padded bounding-box image
        n_chunk (int): target number of chunks, normally the worker count

    Returns:
        slice_list (list of slice): contiguous, covering range(n_img)
    """
    max_img = max(1, int(MAX_CHUNK_BYTES // max(1, n_vox_img * 8)))
    n_chunk = max(n_chunk, -(-n_img // max_img))
    edges = np.linspace(0, n_img, min(n_chunk, n_img) + 1).astype(int)
    return [slice(lo, hi) for lo, hi in zip(edges, edges[1:]) if hi > lo]
