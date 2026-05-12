"""Benchmark glow.analysis.ward.ward_tree vs sklearn.cluster.ward_tree.

Run with:
    ~/venv_glow/bin/python test/scratch/bench_ward.py

Reports wall-clock for both implementations on grid-constrained Ward at
sizes typical of GLOW's per-permutation hot path.  The first call to the
Numba impl pays a one-time JIT cost; we time the second call.
"""
import time

import numpy as np
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import ward_tree as sk_ward_tree

from glow.analysis.ward import ward_tree as our_ward_tree


def time_call(fn, n_warmup=1, n_reps=3):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def bench_one(shape, a, seed=0):
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    X = rng.standard_normal((n, a)).astype(np.float64)
    conn = grid_to_graph(*shape)

    t_ours = time_call(lambda: our_ward_tree(X, conn))
    t_sk = time_call(lambda: sk_ward_tree(X=X, connectivity=conn))
    return n, t_sk, t_ours, t_sk / t_ours


def main():
    print(f'{"shape":<15} {"n_vox":<8} {"a":<5} {"sklearn":<12} {"ours":<12} {"speedup":<10}')
    print('-' * 70)
    cases = [
        ((10, 10, 10), 8),
        ((15, 15, 15), 8),
        ((20, 20, 20), 8),
        ((20, 20, 20), 30),
        ((30, 30, 30), 8),
        ((30, 30, 30), 30),
    ]
    for shape, a in cases:
        n, t_sk, t_ours, sp = bench_one(shape, a)
        print(f'{str(shape):<15} {n:<8} {a:<5} {t_sk*1000:<12.1f} '
              f'{t_ours*1000:<12.1f} {sp:<10.2f}')


if __name__ == '__main__':
    main()
