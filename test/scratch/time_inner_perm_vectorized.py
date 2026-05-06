"""Time per-region vs batched LLR on a mandrill-scale merged graph.

Builds the mandrill demo experiment, runs n_perm_fwer outer perms to
generate a merged graph at realistic scale, then times one inner-perm
pass via:

  1) the legacy per-region path: iter_stat + get_llr (Python loop)
  2) the new batched path: compute_llr_batched (single numpy pass)

Reports microseconds per region for each path and the speedup factor.
"""

import os
import time

# keep BLAS single-threaded so timings reflect the algorithmic change
# rather than thread contention with numpy's internal parallelism.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

import glow.graph
from glow.analysis.cluster import cluster
from glow.analysis.mancova import decompose, get_llr
from glow.graph import compute_llr_batched, graph_merge, iter_stat


def _build_mandrill_exp():
    """Replicate the mandrill demo path from glow/viewer/__main__.py."""
    import pathlib
    from PIL import Image

    from glow.experiment.exper import Experiment
    from glow.mask import get_mask_idx

    data_dir = (pathlib.Path(__file__).resolve().parents[1] / 'data')
    img_path = data_dir / 'mandrill_small.png'
    img = np.array(Image.open(img_path)).astype(np.float64)
    h, w, _ = img.shape

    num_img = 12
    feat_indices = [0, 1, 2]
    b = len(feat_indices)
    num_vox = h * w
    pixel_flat = img.reshape(num_vox, 3).T

    rng = np.random.default_rng(0)
    y = np.empty((b, num_img, num_vox))
    for fi, ci in enumerate(feat_indices):
        base = pixel_flat[ci]
        for i in range(num_img):
            y[fi, i, :] = base + rng.normal(0, 15.0, num_vox)

    mask_idx = get_mask_idx(np.ones((h, w), dtype=bool))
    x = np.arange(num_img, dtype=float).reshape(1, -1)
    contrast = np.array([True])
    return Experiment(x=x, y=y, contrast=contrast, mask_idx=mask_idx,
                      add_bias=True)


def _build_merged_children(exp, n_perm_fwer):
    """Cluster on n_perm_fwer + 1 trees and merge into one graph."""
    num_vox = exp.y.shape[2]
    children_list = []
    for k in range(n_perm_fwer + 1):
        _exp = exp.permute(k) if k else exp
        children_list.append(cluster(exp=_exp, mode='q1'))
    _, merged_children, _ = graph_merge(n_common=num_vox,
                                        children_list=children_list)
    return merged_children


def _time_per_region(exp, children):
    """Legacy path: iter_stat + get_llr per region."""
    num_vox = exp.y.shape[2]
    num_reg = num_vox + children.shape[0]
    llr = np.full(num_reg, np.nan)
    t0 = time.perf_counter()
    for reg_idx, size, e, h in iter_stat(exp=exp, children=children):
        llr[reg_idx] = get_llr(e, h, n=size)
    elapsed = time.perf_counter() - t0
    return elapsed, llr


def _time_batched(exp, children, q0, q1):
    """New path: single batched call."""
    t0 = time.perf_counter()
    llr, _ = compute_llr_batched(exp, children=children, q0=q0, q1=q1)
    elapsed = time.perf_counter() - t0
    return elapsed, llr


def main():
    n_perm_fwer = 25
    print(f'building mandrill experiment ...')
    exp = _build_mandrill_exp()
    b, num_img, num_vox = exp.y.shape
    print(f'  shape: b={b} num_img={num_img} num_vox={num_vox}')

    print(f'clustering {n_perm_fwer + 1} trees + merging ...')
    merged_children = _build_merged_children(exp, n_perm_fwer)
    num_reg = num_vox + merged_children.shape[0]
    print(f'  merged graph: {num_reg} regions '
          f'({merged_children.shape[0]} internal)')

    # one FL'd experiment to time both paths against
    inner_seed = n_perm_fwer + 1
    _exp = exp.permute(inner_seed)
    q0, q1, _ = decompose(x=_exp.x, contrast=_exp.contrast)

    print('\ntiming per-region path (iter_stat + get_llr) ...')
    t_loop, llr_loop = _time_per_region(_exp, merged_children)
    us_per_reg_loop = 1e6 * t_loop / num_reg
    print(f'  per-region loop: {t_loop:.3f} s '
          f'({us_per_reg_loop:.2f} us / region)')

    print('timing batched path (compute_llr_batched) ...')
    t_batch, llr_batch = _time_batched(_exp, merged_children, q0, q1)
    us_per_reg_batch = 1e6 * t_batch / num_reg
    print(f'  batched:         {t_batch:.3f} s '
          f'({us_per_reg_batch:.2f} us / region)')

    # numerical agreement check
    valid = ~np.isnan(llr_loop) & ~np.isnan(llr_batch)
    max_abs = float(np.max(np.abs(llr_batch[valid] - llr_loop[valid])))
    max_rel = float(np.max(np.abs(
        (llr_batch[valid] - llr_loop[valid])
        / np.maximum(np.abs(llr_loop[valid]), 1e-12))))
    print(f'\nagreement (over {valid.sum()} valid regions):')
    print(f'  max abs diff: {max_abs:.3e}')
    print(f'  max rel diff: {max_rel:.3e}')

    speedup = t_loop / t_batch
    print(f'\nspeedup: {speedup:.1f}x')


if __name__ == '__main__':
    main()
