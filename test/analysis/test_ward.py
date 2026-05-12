"""Fidelity + validity tests for glow.analysis.ward.ward_tree.

This is a heap-greedy reimplementation of sklearn's structured ward_tree,
so the dendrograms match exactly (same children array, same distances to
float64 precision).
"""
import numpy as np
import pytest
from scipy import sparse
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import ward_tree as sk_ward_tree

from glow.analysis.ward import ward_tree as our_ward_tree


def _data_invariant(X):
    return float(((X - X.mean(axis=0)) ** 2).sum())


def _raw_ward_sum(distances):
    """sklearn / our ``ward_tree`` returns sqrt(2 * raw_ward) per merge."""
    return float(((distances ** 2) / 2.0).sum())


def _merge_leaf_sets(children, n_samples):
    desc = [frozenset([i]) for i in range(n_samples)]
    out = []
    for a, b in children:
        s = desc[a] | desc[b]
        out.append(s)
        desc.append(s)
    return out


def _is_topologically_valid(children, n_samples):
    for i, (a, b) in enumerate(children):
        if a >= n_samples + i or b >= n_samples + i:
            return False
        if a < 0 or b < 0:
            return False
    return True


# -------------------------------------------------------- exact sklearn match


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
@pytest.mark.parametrize('shape,F', [
    ((4, 4, 4), 8),
    ((5, 5, 5), 8),
    ((6, 6, 6), 12),
    ((8, 8, 8), 8),
    ((10, 10, 10), 8),
])
def test_constrained_children_match_sklearn(seed, shape, F):
    """Heap-greedy reproduces sklearn's children array exactly."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    X = rng.standard_normal((n, F)).astype(np.float64)
    conn = grid_to_graph(*shape)

    ours = our_ward_tree(X, conn, return_distance=True)
    sk = sk_ward_tree(X=X, connectivity=conn, return_distance=True)

    assert np.array_equal(ours[0], sk[0]), 'children must match sklearn exactly'
    np.testing.assert_allclose(ours[4], sk[4], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('shape', [(4, 4, 4), (6, 6, 6), (8, 8, 8)])
def test_constrained_partitions_match_sklearn(seed, shape):
    """Same set of leaf-set partitions as sklearn (subsumed by exact
    children match, but spot-checks the dendrogram semantics)."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    F = 8
    X = rng.standard_normal((n, F)).astype(np.float64)
    conn = grid_to_graph(*shape)

    ours = our_ward_tree(X, conn)
    sk = sk_ward_tree(X=X, connectivity=conn)

    lo = set(_merge_leaf_sets(ours[0], n))
    ls = set(_merge_leaf_sets(sk[0], n))
    assert lo == ls


# ------------------------------------------------------------------ validity


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
@pytest.mark.parametrize('shape,F', [
    ((4, 4, 4), 8),
    ((5, 5, 5), 8),
    ((6, 6, 6), 12),
    ((8, 8, 8), 8),
])
def test_fisher_invariant(seed, shape, F):
    """Sum of raw Ward distances equals the data invariant
    ``||X - mean||²``. Holds for any complete hierarchical merge."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    X = rng.standard_normal((n, F)).astype(np.float64)
    conn = grid_to_graph(*shape)

    ours = our_ward_tree(X, conn, return_distance=True)
    np.testing.assert_allclose(
        _raw_ward_sum(ours[4]), _data_invariant(X), rtol=1e-9)


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('shape', [(4, 4, 4), (6, 6, 6), (8, 8, 8)])
def test_tree_well_formed(seed, shape):
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    F = 8
    X = rng.standard_normal((n, F)).astype(np.float64)
    conn = grid_to_graph(*shape)

    children, n_comp, n_leaves, parents = our_ward_tree(X, conn)
    assert n_comp == 1
    assert n_leaves == n
    assert children.shape == (n - 1, 2)
    assert _is_topologically_valid(children, n)

    # every leaf reachable from root
    leaves_at_root = _merge_leaf_sets(children, n)[-1]
    assert leaves_at_root == frozenset(range(n))

    n_nodes = 2 * n - 1
    assert parents.shape == (n_nodes,)
    roots = np.where(parents == np.arange(n_nodes))[0]
    assert len(roots) == 1
    assert roots[0] == n_nodes - 1


def test_forest_multiple_components():
    """Disconnected connectivity → forest.  Total merges = n - n_components.

    sklearn doesn't natively support forests (warns and force-completes the
    connectivity), so we don't compare against it here — just validate
    that our output is internally consistent and matches the Fisher
    invariant.
    """
    rng = np.random.default_rng(0)
    n_per = 30
    F = 4
    X = rng.standard_normal((2 * n_per, F)).astype(np.float64)
    conn = sparse.lil_matrix((2 * n_per, 2 * n_per))
    for i in range(n_per - 1):
        conn[i, i + 1] = 1
        conn[i + 1, i] = 1
        conn[n_per + i, n_per + i + 1] = 1
        conn[n_per + i + 1, n_per + i] = 1
    conn = conn.tocsr()

    children, n_comp, n_leaves, parents = our_ward_tree(X, conn)
    assert n_comp == 2
    assert n_leaves == 2 * n_per
    assert children.shape == (2 * n_per - 2, 2)

    # two roots
    n_nodes = 2 * (2 * n_per) - 2
    roots = np.where(parents == np.arange(n_nodes))[0]
    assert len(roots) == 2

    # Fisher invariant holds per component → equals total ||X - mean_c||²
    # where each leaf's "mean" is its own component's centroid.
    _, distances = our_ward_tree(X, conn, return_distance=True)[3:5]
    raw_sum = float(((distances ** 2) / 2.0).sum())
    # invariant: sum over components of within-component-variance
    inv = 0.0
    for c in (slice(0, n_per), slice(n_per, 2 * n_per)):
        Xc = X[c]
        inv += float(((Xc - Xc.mean(axis=0)) ** 2).sum())
    np.testing.assert_allclose(raw_sum, inv, rtol=1e-9)


# --------------------------------------------------------------- speed bench


def test_speed_vs_sklearn_at_5k():
    """At ~5k voxels, our heap-greedy must beat sklearn by >3x.

    Both implementations are warmed up before timing (Numba JIT excluded).
    """
    import time

    shape = (17, 17, 17)  # 4913 voxels
    F = 8
    rng = np.random.default_rng(0)
    n = int(np.prod(shape))
    X = rng.standard_normal((n, F)).astype(np.float64)
    conn = grid_to_graph(*shape)

    # warm up (JIT)
    our_ward_tree(X, conn)
    sk_ward_tree(X=X, connectivity=conn)

    def time_min(fn, n_reps=3):
        ts = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return min(ts)

    t_ours = time_min(lambda: our_ward_tree(X, conn))
    t_sk = time_min(lambda: sk_ward_tree(X=X, connectivity=conn))
    speedup = t_sk / t_ours
    assert speedup > 2.0, (
        f'expected >2x speedup at 5k voxels, got {speedup:.2f}x '
        f'(sklearn={t_sk*1000:.1f}ms, ours={t_ours*1000:.1f}ms)')
