"""Connectivity-constrained Ward clustering via Mullner's NN-array algorithm.

Faster drop-in for sklearn.cluster.ward_tree(X, connectivity=...) (Mullner
2011). Produces the same dendrogram as sklearn (same children array,
distances matching to float64 ULP).

Algorithm: each cluster c tracks its current nearest neighbour nn[c] and
that distance nn_d[c]. A global min-heap is keyed by (nn_d[c], c). The next
merge is always the global min over alive clusters' NN distances, which is
the same edge sklearn's heap-greedy picks.

Compared to sklearn's single-global-edge-heap approach, this dramatically
cuts heap pressure: instead of pushing every edge (and accumulating ~95%
stale entries when endpoints merge away), we push at most one entry per
cluster per NN change. On HCP-scale grids this gives ~3x over the
heap-greedy variant and ~9x over sklearn.

Per-merge work:
- Pop outer heap; skip stale entries (cluster dead or distance changed).
- Build merged cluster's adjacency via union-find path compression on the
  append-only adjacency linked list (same trick as sklearn's _get_parents).
- Compute new cluster's NN by linear scan of its adjacency.
- For each neighbour n: if n's NN was i/j (now dead), rescan n's adj;
  else update n's NN only if the new cluster is closer than its current.

a is the feature dimension (number of independent variables), matching the
GLOW convention for x.shape == (a, n_subjects). When called from cluster()
it equals n_batch x projection_dim (the einsum-'b,a' axes of y after
projection, flattened).
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse.csgraph import connected_components


# --------------------------------------------------------------- primitives --


@njit(cache=True, boundscheck=False)
def _ward_dist(centroid, size, a, i, j):
    """Compute the Ward (variance-increase) distance between clusters i and j."""
    pa = 0.0
    for f in range(a):
        d = centroid[i, f] - centroid[j, f]
        pa += d * d
    sa = size[i]
    sb = size[j]
    return pa * (sa * sb) / (sa + sb)


@njit(cache=True, boundscheck=False)
def _find_root(parent, x):
    """Union-find with two-pass path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


@njit(cache=True, inline='always', boundscheck=False)
def _heap_less(d1, c1, d2, c2):
    """Lex (d, c) ordering, matches Python tuple comparison."""
    if d1 < d2:
        return True
    if d1 > d2:
        return False
    return c1 < c2


@njit(cache=True, boundscheck=False)
def _heap_push(heap_d, heap_c, heap_size, d, c):
    """Push (d, c) onto the min-heap; return False if the heap is full."""
    n = heap_size[0]
    if n >= heap_d.shape[0]:
        return False
    heap_d[n] = d
    heap_c[n] = c
    heap_size[0] = n + 1
    while n > 0:
        par = (n - 1) // 2
        if _heap_less(heap_d[n], heap_c[n], heap_d[par], heap_c[par]):
            heap_d[n], heap_d[par] = heap_d[par], heap_d[n]
            heap_c[n], heap_c[par] = heap_c[par], heap_c[n]
            n = par
        else:
            break
    return True


@njit(cache=True, boundscheck=False)
def _heap_pop(heap_d, heap_c, heap_size):
    """Pop and return the min (d, c) from the heap, re-heapifying."""
    n = heap_size[0] - 1
    d_out = heap_d[0]
    c_out = heap_c[0]
    heap_d[0] = heap_d[n]
    heap_c[0] = heap_c[n]
    heap_size[0] = n
    cur = 0
    while True:
        left = 2 * cur + 1
        right = 2 * cur + 2
        smallest = cur
        if left < n and _heap_less(heap_d[left], heap_c[left],
                                    heap_d[smallest], heap_c[smallest]):
            smallest = left
        if right < n and _heap_less(heap_d[right], heap_c[right],
                                    heap_d[smallest], heap_c[smallest]):
            smallest = right
        if smallest == cur:
            break
        heap_d[cur], heap_d[smallest] = heap_d[smallest], heap_d[cur]
        heap_c[cur], heap_c[smallest] = heap_c[smallest], heap_c[cur]
        cur = smallest
    return d_out, c_out


# Adjacency linked list pool.
#   adj_head[c]: index of c's first list node (-1 if empty)
#   adj_cluster[node]: cluster id at that node
#   adj_next[node]: next node id (-1 if end)


@njit(cache=True, boundscheck=False)
def _adj_prepend(adj_head, adj_cluster, adj_next, pool_top, c, x):
    """Prepend neighbour x to c's adjacency list; return False if pool full."""
    nid = pool_top[0]
    if nid >= adj_cluster.shape[0]:
        return False
    pool_top[0] = nid + 1
    adj_cluster[nid] = x
    adj_next[nid] = adj_head[c]
    adj_head[c] = nid
    return True


@njit(cache=True, boundscheck=False)
def _scan_nn(centroid, size, alive, a, c,
             adj_head, adj_cluster, adj_next):
    """Linear scan of c's adjacency for its current alive nearest neighbour.

    Returns (-1, inf) if c has no alive neighbours.
    """
    best_d = np.inf
    best = np.int64(-1)
    prev = np.int64(-1)
    nid = adj_head[c]
    while nid >= 0:
        x = adj_cluster[nid]
        nxt = adj_next[nid]
        if alive[x]:
            d = _ward_dist(centroid, size, a, c, x)
            # Lex tie-break: prefer smaller index on ties (matches sklearn's
            # global-heap behaviour which orders by (d, k, c) tuples).
            if d < best_d or (d == best_d and (best < 0 or x < best)):
                best_d = d
                best = x
            prev = nid
        else:
            # Unlink dead node so future scans don't re-walk it.  The pool
            # node is abandoned (append-only pool, no free list).
            if prev < 0:
                adj_head[c] = nxt
            else:
                adj_next[prev] = nxt
        nid = nxt
    return best, best_d


# ---------------------------------------------------------------- main loop --


@njit(cache=True, inline='always', boundscheck=False)
def _merge_adj_into(adj_head, adj_cluster, adj_next, pool_top,
                    parent, not_visited, src_head, k):
    """Splice src's adjacency list into cluster k's, dedup via not_visited.

    For each node in src's list, resolve its current root via union-find
    and, if not already attached to k, add the symmetric edge (root<->k).
    Returns False on pool overflow, True otherwise.
    """
    nid = src_head
    while nid >= 0:
        yy = adj_cluster[nid]
        nid = adj_next[nid]
        root = _find_root(parent, yy)
        if not_visited[root]:
            not_visited[root] = False
            if not _adj_prepend(adj_head, adj_cluster, adj_next,
                                pool_top, root, k):
                return False
            if not _adj_prepend(adj_head, adj_cluster, adj_next,
                                pool_top, k, root):
                return False
    return True


@njit(cache=True, boundscheck=False)
def _mullner_loop(
    centroid, size, alive, parent,
    adj_head, adj_cluster, adj_next, pool_top,
    heap_d, heap_c, heap_size,
    nn, nn_d,
    not_visited,
    out_a, out_b, out_d, a, n_samples, n_merges,
):
    """Mullner NN-array main loop.

    Returns 0 on success, -1 if outer heap overflows, -2 if pool overflows.
    """
    # Initial NN per leaf
    for c in range(n_samples):
        x, d = _scan_nn(centroid, size, alive, a, c,
                        adj_head, adj_cluster, adj_next)
        nn[c] = x
        nn_d[c] = d
        if x >= 0:
            if not _heap_push(heap_d, heap_c, heap_size, d, c):
                return -1

    for k in range(n_samples, n_samples + n_merges):
        # Pop the global-min cluster.  Skip dead or stale entries.
        c_merge = np.int64(-1)
        while heap_size[0] > 0:
            d_popped, c_popped = _heap_pop(heap_d, heap_c, heap_size)
            if not alive[c_popped]:
                continue
            if d_popped != nn_d[c_popped]:
                # stale: a newer entry was pushed for c_popped
                continue
            x = nn[c_popped]
            if x < 0 or not alive[x]:
                # nn died between push and pop; rescan
                new_x, new_d = _scan_nn(centroid, size, alive, a,
                                        c_popped, adj_head, adj_cluster,
                                        adj_next)
                nn[c_popped] = new_x
                nn_d[c_popped] = new_d
                if new_x >= 0:
                    if not _heap_push(heap_d, heap_c, heap_size, new_d,
                                       c_popped):
                        return -1
                continue
            c_merge = c_popped
            break
        if c_merge < 0:
            # heap exhausted: all merges in this component are done
            break

        i = c_merge
        j = nn[i]
        merged = k - n_samples
        if i < j:
            out_a[merged] = i
            out_b[merged] = j
        else:
            out_a[merged] = j
            out_b[merged] = i
        out_d[merged] = nn_d[i]

        # merge centroids
        si = size[i]
        sj = size[j]
        st = si + sj
        for f in range(a):
            centroid[k, f] = (si * centroid[i, f] + sj * centroid[j, f]) / st
        size[k] = st
        alive[k] = True
        alive[i] = False
        alive[j] = False
        parent[i] = k
        parent[j] = k

        # Build k's adjacency from i's and j's via union-find with
        # path compression.  Dedup via not_visited[].
        not_visited[k] = False
        if not _merge_adj_into(adj_head, adj_cluster, adj_next, pool_top,
                               parent, not_visited, adj_head[i], k):
            return -2
        if not _merge_adj_into(adj_head, adj_cluster, adj_next, pool_top,
                               parent, not_visited, adj_head[j], k):
            return -2

        # One pass over k's adjacency:
        # - reset not_visited
        # - compute nn[k] (linear scan)
        # - update each neighbour n's NN if needed
        best_d = np.inf
        best = np.int64(-1)
        nid = adj_head[k]
        while nid >= 0:
            n = adj_cluster[nid]
            nid = adj_next[nid]
            not_visited[n] = True

            d_kn = _ward_dist(centroid, size, a, k, n)
            if d_kn < best_d or (d_kn == best_d and (best < 0 or n < best)):
                best_d = d_kn
                best = n

            n_nn = nn[n]
            if n_nn == i or n_nn == j or n_nn < 0 or not alive[n_nn]:
                # n's old NN died → rescan n's adj
                new_x, new_d = _scan_nn(centroid, size, alive, a,
                                        n, adj_head, adj_cluster, adj_next)
                nn[n] = new_x
                nn_d[n] = new_d
                if new_x >= 0:
                    if not _heap_push(heap_d, heap_c, heap_size, new_d, n):
                        return -1
            elif d_kn < nn_d[n] or (d_kn == nn_d[n] and k < n_nn):
                # k is closer than n's current NN (with sklearn-matching
                # tie-break favouring smaller index)
                nn[n] = k
                nn_d[n] = d_kn
                if not _heap_push(heap_d, heap_c, heap_size, d_kn, n):
                    return -1
        not_visited[k] = True

        nn[k] = best
        nn_d[k] = best_d
        if best >= 0:
            if not _heap_push(heap_d, heap_c, heap_size, best_d, k):
                return -1

    return 0


# -------------------------------------------------------- public entry point --


def ward_tree(X, connectivity, return_distance: bool = False):
    """Cluster agglomeratively under a connectivity constraint (Ward).

    Matches the signature and dendrogram of
    sklearn.cluster.ward_tree(X, connectivity=...).

    Args:
        X (np.array): (n_samples, a) feature matrix where a is the
            feature/batch dimension.
        connectivity: scipy sparse matrix of shape (n_samples, n_samples).
            Symmetrised internally; non-zero entries define graph neighbours.
        return_distance (bool): if True, also return per-merge
            sqrt(2 * raw_ward) (sklearn's scaling).

    Returns:
        children (np.array): (n_merges, 2) child index pairs. Internal
            node n_samples + i corresponds to row i.
        n_connected_components (int): number of connected components
        n_leaves (int): number of leaf samples (n_samples)
        parents (np.array): (n_nodes,) parent of each node. Root(s)
            point to themselves.
        distances (np.array, optional): per-merge sqrt(2 * raw_ward),
            returned only when return_distance is True.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, a = X.shape

    A = sparse.csr_matrix(connectivity)
    A = (A + A.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    n_components, _ = connected_components(A, directed=False)
    n_merges_total = n_samples - n_components
    n_nodes = n_samples + n_merges_total

    centroid = np.zeros((n_nodes, a), dtype=np.float64)
    centroid[:n_samples] = X
    size = np.zeros(n_nodes, dtype=np.float64)
    size[:n_samples] = 1.0
    alive = np.zeros(n_nodes, dtype=np.bool_)
    alive[:n_samples] = True
    parent = np.arange(n_nodes, dtype=np.int64)
    not_visited = np.ones(n_nodes, dtype=np.bool_)
    nn = np.full(n_nodes, -1, dtype=np.int64)
    nn_d = np.full(n_nodes, np.inf, dtype=np.float64)

    out_a = np.full(n_merges_total, -1, dtype=np.int64)
    out_b = np.full(n_merges_total, -1, dtype=np.int64)
    out_d = np.full(n_merges_total, np.inf, dtype=np.float64)

    # Adjacency pool: 2 entries per edge added.  Per merge ≤ 2 * D adds.
    # Initial = 2 * E (full symmetric).  Generous: A.nnz + 32 * N, doubled
    # on overflow.
    init_pool = A.nnz + 32 * n_samples + 1024
    # Outer heap: one entry per cluster NN update.  Per merge ≤ D updates
    # plus the new k's entry.  Initial: N.  Generous: 16 * N, doubled on
    # overflow.
    init_heap = 16 * n_samples + 1024

    pool_size = init_pool
    heap_cap = init_heap

    for _retry in range(10):
        adj_head = np.full(n_nodes, -1, dtype=np.int64)
        adj_cluster = np.empty(pool_size, dtype=np.int64)
        adj_next = np.empty(pool_size, dtype=np.int64)
        pool_top = np.array([0], dtype=np.int64)
        heap_d = np.empty(heap_cap, dtype=np.float64)
        heap_c = np.empty(heap_cap, dtype=np.int64)
        heap_size = np.array([0], dtype=np.int64)

        # reset mutable state for retries
        centroid[n_samples:] = 0
        size[n_samples:] = 0
        alive[:n_samples] = True
        alive[n_samples:] = False
        parent[:] = np.arange(n_nodes)
        not_visited[:] = True
        nn[:] = -1
        nn_d[:] = np.inf
        out_a[:] = -1
        out_b[:] = -1
        out_d[:] = np.inf

        # Build initial adjacency from connectivity (both directions).
        indptr = A.indptr
        indices = A.indices
        overflow = False
        for i in range(n_samples):
            for p in range(indptr[i], indptr[i + 1]):
                j = int(indices[p])
                if not _adj_prepend(adj_head, adj_cluster, adj_next,
                                    pool_top, i, j):
                    overflow = True
                    break
            if overflow:
                break

        if not overflow:
            status = _mullner_loop(
                centroid, size, alive, parent,
                adj_head, adj_cluster, adj_next, pool_top,
                heap_d, heap_c, heap_size,
                nn, nn_d,
                not_visited,
                out_a, out_b, out_d, a, n_samples, n_merges_total,
            )
            if status == -1:
                heap_cap *= 2
                continue
            if status == -2:
                pool_size *= 2
                continue
            break
        else:
            pool_size *= 2
            continue
    else:
        raise RuntimeError('ward_tree: pool/heap overflow persisted')

    children = np.empty((n_merges_total, 2), dtype=np.intp)
    children[:, 0] = out_a
    children[:, 1] = out_b

    parents = np.arange(n_nodes, dtype=np.intp)
    for i in range(n_merges_total):
        pid = n_samples + i
        parents[children[i, 0]] = pid
        parents[children[i, 1]] = pid

    if return_distance:
        distances = np.sqrt(2.0 * out_d)
        return children, n_components, n_samples, parents, distances
    return children, n_components, n_samples, parents
