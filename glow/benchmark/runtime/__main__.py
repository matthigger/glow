"""Local runtime benchmark for ward_tree + inner-perm CPU backend.

Running python -m glow.benchmark.runtime sweeps HCP voxel counts at
--n-steps log-spaced points between --min-voxels and --max-voxels and,
for each, times three functions on the same (X, connectivity, exp,
children) inputs:

  - glow.analysis.ward.ward_tree
  - sklearn.cluster.ward_tree
  - glow.analysis.inner_perm.cpu_perm  (unified perm-LLR backend;
    handles intercept-only and general Q0 on the same code path)

Results are appended to a JSON file after every voxel count so you can
follow progress live (tail -f the output path, or just rerun this
script's plotting / summary).

Run:

    python -m glow.benchmark.runtime               # defaults (1k..600k, 20 steps)
    python -m glow.benchmark.runtime --n-perm 1    # quicker
"""
import os

# Pin BLAS / OpenMP / numba to a single thread so the timing reflects
# single-CPU work.  Env vars cover numba (read at JIT-compile time) and
# any BLAS that has not been loaded yet; threadpool_limits below clamps
# BLAS pools that are already live by the time we get here.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import json
import time
from pathlib import Path

from threadpoolctl import threadpool_limits

threadpool_limits(limits=1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from platformdirs import user_data_dir
from scipy.ndimage import generate_binary_structure, label
from sklearn.cluster import ward_tree as sklearn_ward_tree
from sklearn.feature_extraction.image import grid_to_graph

import glow
import glow.effect
import glow.experiment
from glow.analysis import inner_perm
from glow.analysis.cluster import cluster as glow_cluster
from glow.analysis.mancova import decompose
from glow.analysis.ward import ward_tree as glow_ward_tree
from glow.mask import bbox_crop


DEFAULT_OUTPUT = (Path(user_data_dir('glow', 'glow_author'))
                  / 'results' / 'runtime' / 'ward_inner_perm.json')


def load_hcp_exp():
    """Load the HCP image-only experiment used throughout this sweep."""
    from brainjar import hcp_ya_open
    path = hcp_ya_open.process()
    exp = glow.experiment.ExperimentImageOnly.from_search(
        folder=path,
        sbj_regex=r'[\d]{6}',
        img_glob_dict={'fa': '*_fa.nii.gz', 'md': '*_md.nii.gz'})
    return exp.sample_x(a=2, seed=0, add_bias=True)


def subsample(exp_orig, n_vox: int, seed: int = 0):
    """Subsample exp_orig to ~n_vox voxels via a single contiguous sphere.

    Args:
        exp_orig: the full experiment to subsample
        n_vox (int): target voxel count; returns exp_orig unchanged if it
            already has at most this many voxels
        seed (int): RNG seed for the sphere extenter

    Returns:
        the subsampled experiment (or exp_orig if no subsampling was needed)
    """
    max_vox = int((exp_orig.mask_idx > -1).sum())
    if n_vox >= max_vox:
        return exp_orig
    extenter = glow.effect.ExtenterSphere(n_vox=n_vox, connected=True)
    mask = extenter(mask_idx=exp_orig.mask_idx, seed=seed, contiguous=True)
    return exp_orig.apply_mask(mask)


def ward_inputs(exp):
    """Build (X, connectivity) for the largest connected component.

    Mirrors the per-component prep inside glow.analysis.cluster.cluster
    (FOCUS mode): project exp.y onto the interest subspace q1, then carve
    out the component's rows / grid graph. For a contiguous=True subsample
    the largest component is the whole mask.

    Args:
        exp: the experiment to derive ward inputs from

    Returns:
        X (np.array): (n_comp_vox, a1 * b) projected feature rows for the
            component, C-contiguous
        connectivity: the component's grid-graph sparse adjacency
    """
    _, q1, _ = decompose(exp.x, exp.contrast)
    y = np.einsum('bnr,na->bar', exp.y, q1.T, optimize=True)
    num_vox = y.shape[2]
    y = y.reshape((-1, num_vox))

    mask = exp.mask_idx >= 0
    mask_bb, bb_slices = bbox_crop(mask)
    structure = generate_binary_structure(mask.ndim, 1)
    labeled, num_components = label(mask_bb, structure=structure)
    global_idx = exp.mask_idx[bb_slices][mask_bb]

    best = None
    best_n = 0
    for c in range(1, num_components + 1):
        comp_select = labeled[mask_bb] == c
        comp_global_idx = global_idx[comp_select]
        n_c = comp_global_idx.size
        if n_c <= best_n:
            continue
        comp_mask_bb = labeled == c
        comp_bb, _ = bbox_crop(comp_mask_bb)
        local_X = np.ascontiguousarray(y[:, comp_global_idx].T)
        local_conn = grid_to_graph(*comp_bb.shape, mask=comp_bb)
        best = (local_X, local_conn)
        best_n = n_c
    return best


def time_call(fn, *args, **kwargs):
    """Run fn once and return (elapsed_seconds, result)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


def warm_up() -> None:
    """Trigger numba JIT compiles so the first measurement is honest."""
    rng = np.random.default_rng(0)
    side = 4
    n = side ** 3
    X = rng.standard_normal((n, 2))
    mask = np.ones((side, side, side), dtype=bool)
    conn = grid_to_graph(side, side, side, mask=mask)
    glow_ward_tree(X=X, connectivity=conn)
    sklearn_ward_tree(X=X, connectivity=conn)


def write_results(path: Path, results: list) -> None:
    """Atomic-ish overwrite of path so partial reads see a full list.

    Args:
        path (pathlib.Path): JSON output path
        results (list): result dicts to serialize
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    tmp.replace(path)


_PLOT_SERIES = [
    ('glow_ward_sec', 'glow.ward_tree', 'C0', 'o'),
    ('sklearn_ward_sec', 'sklearn.ward_tree', 'C1', 's'),
    ('cpu_perm_sec', 'cpu_perm', 'C2', '^'),
]


def write_plot(path: Path, results: list) -> None:
    """Save a log-log plot of num_vox vs. time for all backends.

    Args:
        path (pathlib.Path): output image path; the suffix sets the format
        results (list): result dicts as written by run_one
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(results, key=lambda r: r['num_vox'])
    nv = np.array([r['num_vox'] for r in rows])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for key, label, color, marker in _PLOT_SERIES:
        ys = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        mask = np.isfinite(ys)
        if not mask.any():
            continue
        ax.plot(nv[mask], ys[mask], marker=marker, color=color, label=label,
                lw=1.5, ms=5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('num_vox')
    ax.set_ylabel('time (sec)')
    ax.set_title('runtime vs. num_vox')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    tmp = path.with_suffix(path.suffix + '.tmp')
    fig.savefig(tmp, bbox_inches='tight', format=path.suffix.lstrip('.'))
    plt.close(fig)
    tmp.replace(path)


def run_one(exp, n_perm: int, base_seed: int) -> dict:
    """Time all backends on one exp and return a result dict.

    Args:
        exp: the (sub)sampled experiment to time
        n_perm (int): inner permutations for cpu_perm
        base_seed (int): base RNG seed for cpu_perm

    Returns:
        a result dict with sizes (num_vox, b, num_img, ward_n_samples) and
            per-backend timings (glow_ward_sec, sklearn_ward_sec, cpu_perm_sec)
    """
    actual_vox = int(exp.y.shape[2])
    row = {'num_vox': actual_vox,
           'b': int(exp.y.shape[0]),
           'num_img': int(exp.y.shape[1]),
           'n_perm': int(n_perm)}

    X, conn = ward_inputs(exp)
    row['ward_n_samples'] = int(X.shape[0])

    glow_sec, _ = time_call(
        glow_ward_tree, X=X, connectivity=conn)
    row['glow_ward_sec'] = glow_sec

    sklearn_sec, _ = time_call(
        sklearn_ward_tree, X=X, connectivity=conn)
    row['sklearn_ward_sec'] = sklearn_sec

    # children for inner-perm must match exp.y's voxel axis -- use the
    # full forest builder rather than the local component's children.
    children = glow_cluster(exp, mode='Focus')
    q0, q1, _ = decompose(exp.x, exp.contrast)

    perm_sec, _ = time_call(
        inner_perm.cpu_perm,
        exp=exp, base_seed=base_seed, n_perm=n_perm,
        q0=q0, q1=q1, children=children, min_vox=1)
    row['cpu_perm_sec'] = perm_sec

    return row


def parse_args() -> argparse.Namespace:
    """Parse the runtime-benchmark CLI arguments.

    Returns:
        the parsed argparse.Namespace
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--min-voxels', type=int, default=1_000)
    p.add_argument('--max-voxels', type=int, default=600_000)
    p.add_argument('--n-steps', type=int, default=20)
    p.add_argument('--n-perm', type=int, default=250,
                   help='inner permutations for cpu_perm')
    p.add_argument('--seed', type=int, default=0,
                   help='RNG seed for extenter + inner-perm base')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> None:
    """Run the voxel-count sweep, writing JSON + a plot after each step."""
    args = parse_args()
    print(f'Loading HCP data...')
    exp_orig = load_hcp_exp()
    max_vox = int((exp_orig.mask_idx > -1).sum())
    print(f'  HCP mask: {max_vox:,} voxels   b={exp_orig.y.shape[0]}   '
          f'num_img={exp_orig.y.shape[1]}')

    max_target = min(args.max_voxels, max_vox)
    targets = np.geomspace(args.min_voxels, max_target, args.n_steps)
    targets = np.unique(targets.round().astype(int))
    print(f'  {len(targets)} targets: {targets[0]:,} .. {targets[-1]:,}')
    print(f'  n_perm={args.n_perm}')
    print(f'  output: {args.output}')

    print('\nWarming up numba JIT...')
    warm_up()

    results = []
    for i, target in enumerate(targets, 1):
        print(f'\n[{i}/{len(targets)}] target={int(target):,} voxels')
        exp = subsample(exp_orig, int(target), seed=args.seed)

        row = run_one(exp, n_perm=args.n_perm, base_seed=args.seed)
        row['target_vox'] = int(target)
        results.append(row)

        write_results(args.output, results)
        plot_path = args.output.with_suffix('.pdf')
        write_plot(plot_path, results)

        print(f'  num_vox={row["num_vox"]:>7,}'
              f'   glow_ward={row["glow_ward_sec"]:8.2f}s'
              f'   sklearn_ward={row["sklearn_ward_sec"]:8.2f}s'
              f'   cpu_perm={row["cpu_perm_sec"]:8.2f}s')
        print(f'  wrote: {args.output}')
        print(f'  wrote: {plot_path}')

    print(f'\nDone. {len(results)} rows -> {args.output}')


if __name__ == '__main__':
    main()
