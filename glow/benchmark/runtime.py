"""Local runtime benchmark for ward_tree + inner-perm CPU backends.

Sweeps HCP voxel counts at ``--n-steps`` log-spaced points between
``--min-voxels`` and ``--max-voxels`` and, for each, times four
functions on the same (X, connectivity, exp, children) inputs:

  - ``glow.analysis.ward.ward_tree``
  - ``sklearn.cluster.ward_tree``
  - ``glow.analysis.inner_perm.cpu_fast``  (intercept-only Phase-1 hoist)
  - ``glow.analysis.inner_perm.cpu_slow``  (full ``compute_llr_batched``
    per draw; skipped above ``--max-slow-vox`` since it scales poorly)

Results are appended to a JSON file after every voxel count so you can
follow progress live (``tail -f`` the output path, or just rerun this
script's plotting / summary).

Run::

    python -m glow.benchmark.runtime               # defaults (1k..600k, 20 steps)
    python -m glow.benchmark.runtime --n-perm 1    # quicker
    python -m glow.benchmark.runtime --max-slow-vox 0   # skip cpu_slow entirely
"""
import argparse
import json
import time
from pathlib import Path

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


def subsample(exp_orig, n_vox, seed=0):
    """Subsample ``exp_orig`` to ~n_vox voxels via a single contiguous sphere."""
    max_vox = int((exp_orig.mask_idx > -1).sum())
    if n_vox >= max_vox:
        return exp_orig
    extenter = glow.effect.ExtenterSphere(n_vox=n_vox, connected=True)
    mask = extenter(mask_idx=exp_orig.mask_idx, seed=seed, contiguous=True)
    return exp_orig.apply_mask(mask)


def ward_inputs(exp):
    """Build ``(X, connectivity)`` for the largest connected component.

    Mirrors the per-component prep inside ``glow.analysis.cluster.cluster``
    (FOCUS mode): project ``exp.y`` onto the interest subspace ``q1``,
    then carve out the component's rows / grid graph.  For a
    ``contiguous=True`` subsample the largest component is the whole mask.
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
    """Run ``fn`` once and return ``(elapsed_seconds, result)``."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


def warm_up():
    """Trigger numba JIT compiles so the first measurement is honest."""
    rng = np.random.default_rng(0)
    side = 4
    n = side ** 3
    X = rng.standard_normal((n, 2))
    mask = np.ones((side, side, side), dtype=bool)
    conn = grid_to_graph(side, side, side, mask=mask)
    glow_ward_tree(X=X, connectivity=conn)
    sklearn_ward_tree(X=X, connectivity=conn)


def write_results(path, results):
    """Atomic-ish overwrite of ``path`` so partial reads see a full list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)
    tmp.replace(path)


def run_one(exp, n_perm, run_slow, base_seed):
    """Time all four backends on one ``exp``; returns a result dict."""
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

    fast_sec, _ = time_call(
        inner_perm.cpu_fast,
        exp=exp, base_seed=base_seed, n_perm=n_perm,
        q0=q0, q1=q1, children=children, min_vox=1)
    row['cpu_fast_sec'] = fast_sec

    if run_slow:
        slow_sec, _ = time_call(
            inner_perm.cpu_slow,
            exp=exp, base_seed=base_seed, n_perm=n_perm,
            q0=q0, q1=q1, children=children, min_vox=1)
        row['cpu_slow_sec'] = slow_sec
    else:
        row['cpu_slow_sec'] = None

    return row


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--min-voxels', type=int, default=1_000)
    p.add_argument('--max-voxels', type=int, default=600_000)
    p.add_argument('--n-steps', type=int, default=20)
    p.add_argument('--n-perm', type=int, default=5,
                   help='inner permutations for cpu_fast / cpu_slow')
    p.add_argument('--max-slow-vox', type=int, default=100_000,
                   help='skip cpu_slow above this voxel count (0 = always skip)')
    p.add_argument('--seed', type=int, default=0,
                   help='RNG seed for extenter + inner-perm base')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main():
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
    print(f'  n_perm={args.n_perm}   max_slow_vox={args.max_slow_vox:,}')
    print(f'  output: {args.output}')

    print('\nWarming up numba JIT...')
    warm_up()

    results = []
    for i, target in enumerate(targets, 1):
        run_slow = args.max_slow_vox > 0 and int(target) <= args.max_slow_vox
        print(f'\n[{i}/{len(targets)}] target={int(target):,} voxels'
              f'   cpu_slow={"yes" if run_slow else "skip"}')
        exp = subsample(exp_orig, int(target), seed=args.seed)

        row = run_one(exp, n_perm=args.n_perm, run_slow=run_slow,
                      base_seed=args.seed)
        row['target_vox'] = int(target)
        results.append(row)

        write_results(args.output, results)

        slow_str = (f'{row["cpu_slow_sec"]:8.2f}s'
                    if row['cpu_slow_sec'] is not None else '   (skip)')
        print(f'  num_vox={row["num_vox"]:>7,}'
              f'   glow_ward={row["glow_ward_sec"]:8.2f}s'
              f'   sklearn_ward={row["sklearn_ward_sec"]:8.2f}s'
              f'   cpu_fast={row["cpu_fast_sec"]:8.2f}s'
              f'   cpu_slow={slow_str}')
        print(f'  wrote: {args.output}')

    print(f'\nDone. {len(results)} rows -> {args.output}')


if __name__ == '__main__':
    main()
