"""Connectivity-constrained Ward clustering via heap-greedy (sklearn-compat).

Faster drop-in for ``sklearn.cluster.ward_tree(X, connectivity=...)``.
Produces the same dendrogram as sklearn (same set of leaf-set partitions
at every cut level), so it's a lossless replacement.

Algorithm (matches sklearn's structured ward_tree):

- Heap of (distance, i, j) edges over alive cluster pairs.  Pop the min;
  if either endpoint has been merged away, discard and try again.
- When merging a, b → k, walk a's and b's adjacency lists; for each
  adjacency, follow ``parent`` pointers (union-find with path compression)
  to find the live ancestor.  Dedup ancestors → these are k's neighbours.
- Push (d(k, c), k, c) for each new neighbour c.  Append k to c's
  adjacency list (so future merges involving c can rediscover k via
  the union-find walk).
- Adjacency lists are append-only: dead entries never removed, just
  filtered at scan-time by union-find lookup.

Everything's in @njit with float64 centroids (matches sklearn's internal
precision so distance ties break the same way).

Output children pair convention: ``(smaller_idx, larger_idx)``, matching
sklearn's post-``[::-1]`` reversal.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse.csgraph import connected_components


# --------------------------------------------------------------- primitives --


@njit(cache=True, fastmath=False, boundscheck=False)
def _ward_dist(centroid, size, n_features, i, j):
    pa = 0.0
    for f in range(n_features):
        d = centroid[i, f] - centroid[j, f]
        pa += d * d
    sa = size[i]
    sb = size[j]
    return pa * (sa * sb) / (sa + sb)


@njit(cache=True, boundscheck=False)
def _find_root(parent, x):
    """Union-find find with two-pass path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    # compress: point every node on the walk directly to root
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


@njit(cache=True, inline='always', boundscheck=False)
def _heap_less(d1, i1, j1, d2, i2, j2):
    """Lex (d, i, j) ordering, matches Python's tuple comparison."""
    if d1 < d2:
        return True
    if d1 > d2:
        return False
    if i1 < i2:
        return True
    if i1 > i2:
        return False
    return j1 < j2


@njit(cache=True, boundscheck=False)
def _heap_push(heap_d, heap_i, heap_j, heap_size, d, i, j):
    n = heap_size[0]
    if n >= heap_d.shape[0]:
        return False
    heap_d[n] = d
    heap_i[n] = i
    heap_j[n] = j
    heap_size[0] = n + 1
    while n > 0:
        par = (n - 1) // 2
        if _heap_less(heap_d[n], heap_i[n], heap_j[n],
                      heap_d[par], heap_i[par], heap_j[par]):
            heap_d[n], heap_d[par] = heap_d[par], heap_d[n]
            heap_i[n], heap_i[par] = heap_i[par], heap_i[n]
            heap_j[n], heap_j[par] = heap_j[par], heap_j[n]
            n = par
        else:
            break
    return True


@njit(cache=True, boundscheck=False)
def _heap_pop(heap_d, heap_i, heap_j, heap_size):
    n = heap_size[0] - 1
    d_out = heap_d[0]
    i_out = heap_i[0]
    j_out = heap_j[0]
    heap_d[0] = heap_d[n]
    heap_i[0] = heap_i[n]
    heap_j[0] = heap_j[n]
    heap_size[0] = n
    cur = 0
    while True:
        left = 2 * cur + 1
        right = 2 * cur + 2
        smallest = cur
        if left < n and _heap_less(
                heap_d[left], heap_i[left], heap_j[left],
                heap_d[smallest], heap_i[smallest], heap_j[smallest]):
            smallest = left
        if right < n and _heap_less(
                heap_d[right], heap_i[right], heap_j[right],
                heap_d[smallest], heap_i[smallest], heap_j[smallest]):
            smallest = right
        if smallest == cur:
            break
        heap_d[cur], heap_d[smallest] = heap_d[smallest], heap_d[cur]
        heap_i[cur], heap_i[smallest] = heap_i[smallest], heap_i[cur]
        heap_j[cur], heap_j[smallest] = heap_j[smallest], heap_j[cur]
        cur = smallest
    return d_out, i_out, j_out


# ----------------------------------------------------------- adjacency list --
# Per-cluster adjacency list as a linked-list pool.
#   adj_head[c]: index of first node in c's list (-1 if empty)
#   adj_cluster[node_id]: cluster id at that node
#   adj_next[node_id]: next node id (-1 if end)


@njit(cache=True, boundscheck=False)
def _adj_prepend(adj_head, adj_cluster, adj_next, pool_top, c, x):
    nid = pool_top[0]
    if nid >= adj_cluster.shape[0]:
        return False
    pool_top[0] = nid + 1
    adj_cluster[nid] = x
    adj_next[nid] = adj_head[c]
    adj_head[c] = nid
    return True


# ----------------------------------------------------------------- main loop --


@njit(cache=True, boundscheck=False)
def _heap_greedy_loop(
    centroid, size, alive, parent,
    adj_head, adj_cluster, adj_next, pool_top,
    heap_d, heap_i, heap_j, heap_size,
    not_visited,
    out_a, out_b, out_d, n_features, n_samples, n_merges,
):
    """Run heap-greedy Ward clustering.

    Returns 0 on success.
    Returns -1 if the heap overflows; -2 if the adjacency pool overflows.
    """
    for k in range(n_samples, n_samples + n_merges):
        # Pop until we find a live pair.
        i = -1
        j = -1
        d = 0.0
        while heap_size[0] > 0:
            d, i, j = _heap_pop(heap_d, heap_i, heap_j, heap_size)
            if alive[i] and alive[j]:
                break
            i = -1
            j = -1
        if i < 0:
            # Heap exhausted before all merges done — only happens when
            # callers misconfigure (n_merges > actual edge-connected merges)
            return 0

        # Record merge.  Children layout: (smaller, larger) per sklearn.
        merged = k - n_samples
        if i < j:
            out_a[merged] = i
            out_b[merged] = j
        else:
            out_a[merged] = j
            out_b[merged] = i
        out_d[merged] = d

        # Update centroid (size-weighted) and size.
        sa = size[i]
        sb = size[j]
        st = sa + sb
        for f in range(n_features):
            centroid[k, f] = (sa * centroid[i, f] + sb * centroid[j, f]) / st
        size[k] = st
        alive[k] = True
        alive[i] = False
        alive[j] = False
        parent[i] = k
        parent[j] = k

        # Walk i's then j's adjacencies; collect live ancestors as k's
        # new neighbours (deduped via not_visited[]).
        not_visited[k] = False
        # walk i's adjacency
        nid = adj_head[i]
        while nid >= 0:
            x = adj_cluster[nid]
            nid_next = adj_next[nid]
            root = _find_root(parent, x)
            if not_visited[root]:
                not_visited[root] = False
                # add k to root's adjacency (so root can find k later)
                ok = _adj_prepend(adj_head, adj_cluster, adj_next, pool_top,
                                  root, k)
                if not ok:
                    return -2
                # add root to k's adjacency
                ok = _adj_prepend(adj_head, adj_cluster, adj_next, pool_top,
                                  k, root)
                if not ok:
                    return -2
                # compute ward distance and push to heap
                d_new = _ward_dist(centroid, size, n_features, k, root)
                if not _heap_push(heap_d, heap_i, heap_j, heap_size,
                                  d_new, k, root):
                    return -1
            nid = nid_next
        # walk j's adjacency
        nid = adj_head[j]
        while nid >= 0:
            x = adj_cluster[nid]
            nid_next = adj_next[nid]
            root = _find_root(parent, x)
            if not_visited[root]:
                not_visited[root] = False
                ok = _adj_prepend(adj_head, adj_cluster, adj_next, pool_top,
                                  root, k)
                if not ok:
                    return -2
                ok = _adj_prepend(adj_head, adj_cluster, adj_next, pool_top,
                                  k, root)
                if not ok:
                    return -2
                d_new = _ward_dist(centroid, size, n_features, k, root)
                if not _heap_push(heap_d, heap_i, heap_j, heap_size,
                                  d_new, k, root):
                    return -1
            nid = nid_next

        # Reset not_visited via a second pass over k's neighbours.
        nid = adj_head[k]
        while nid >= 0:
            not_visited[adj_cluster[nid]] = True
            nid = adj_next[nid]
        not_visited[k] = True

    return 0


# -------------------------------------------------------- public entry point --


def ward_tree(X, connectivity, return_distance=False):
    """Connectivity-constrained Ward agglomerative clustering.

    Matches the signature of ``sklearn.cluster.ward_tree(X, connectivity=...)``
    and produces the same dendrogram (same set of leaf-set partitions at
    every cut level).

    Args:
        X (np.ndarray): (n_samples, n_features) feature matrix.  Cast to
            float64 internally for sklearn-matching precision.
        connectivity: scipy sparse matrix of shape (n_samples, n_samples).
            Symmetrised internally; non-zero entries define graph neighbours.
        return_distance (bool): if True, also return per-merge
            ``sqrt(2 * raw_ward)`` (sklearn's scaling).

    Returns:
        children (np.ndarray): (n_merges, 2) child index pairs.  Internal
            node ``n_samples + i`` corresponds to row i in iteration order.
        n_connected_components (int)
        n_leaves (int): == n_samples
        parents (np.ndarray): (n_nodes,) parent of each node.  Root(s)
            point to themselves.
        distances (np.ndarray, optional): per-merge sqrt(2 * raw_ward),
            matching sklearn.  Only if ``return_distance``.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    A = sparse.csr_matrix(connectivity)
    A = (A + A.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    n_components, _labels = connected_components(A, directed=False)
    n_merges_total = n_samples - n_components
    n_nodes = n_samples + n_merges_total

    # cluster state
    centroid = np.zeros((n_nodes, n_features), dtype=np.float64)
    centroid[:n_samples] = X
    size = np.zeros(n_nodes, dtype=np.float64)
    size[:n_samples] = 1.0
    alive = np.zeros(n_nodes, dtype=np.bool_)
    alive[:n_samples] = True
    parent = np.arange(n_nodes, dtype=np.int64)
    not_visited = np.ones(n_nodes, dtype=np.bool_)

    # output
    out_a = np.full(n_merges_total, -1, dtype=np.int64)
    out_b = np.full(n_merges_total, -1, dtype=np.int64)
    out_d = np.full(n_merges_total, np.inf, dtype=np.float64)

    # adjacency pool sizing.  Per merge we add 2 entries per new neighbour
    # (one to k's list, one to that neighbour's list).  Initial entries =
    # 2 * E (full symmetric).  Per merge adds ≤ 2 * avg_deg.  Generous:
    # init_edges + 32 * n_samples, doubled on overflow.
    init_pool = A.nnz + 32 * n_samples + 1024

    # heap sizing.  Initial = E (upper triangular).  Plus ≤ avg_deg per
    # merge.  Stale entries pile up.  Generous: 8 * (E + n_samples).
    init_heap = (A.nnz // 2 + n_samples) * 8 + 1024

    pool_size = init_pool
    heap_cap = init_heap
    for _retry in range(10):
        adj_head = np.full(n_nodes, -1, dtype=np.int64)
        adj_cluster = np.empty(pool_size, dtype=np.int64)
        adj_next = np.empty(pool_size, dtype=np.int64)
        pool_top = np.array([0], dtype=np.int64)

        heap_d = np.empty(heap_cap, dtype=np.float64)
        heap_i = np.empty(heap_cap, dtype=np.int64)
        heap_j = np.empty(heap_cap, dtype=np.int64)
        heap_size = np.array([0], dtype=np.int64)

        # reset mutable state (centroid/size unchanged for leaves)
        centroid[n_samples:] = 0
        size[n_samples:] = 0
        alive[:n_samples] = True
        alive[n_samples:] = False
        parent[:] = np.arange(n_nodes)
        not_visited[:] = True
        out_a[:] = -1
        out_b[:] = -1
        out_d[:] = np.inf

        # Build initial adjacency.  For each leaf i, prepend its neighbours
        # (any order is fine; sklearn matches by partition not exact pair
        # order, and tie-breaking is by edge index so we use CSR order).
        indptr = A.indptr
        indices = A.indices
        overflow = False
        for i in range(n_samples):
            for p in range(indptr[i], indptr[i + 1]):
                j = int(indices[p])
                # adjacency is symmetric; we add both directions
                ok = _adj_prepend(adj_head, adj_cluster, adj_next, pool_top,
                                  i, j)
                if not ok:
                    overflow = True
                    break
            if overflow:
                break

        if not overflow:
            # Build initial heap (upper triangular: j < i).
            for i in range(n_samples):
                for p in range(indptr[i], indptr[i + 1]):
                    j = int(indices[p])
                    if j < i:
                        d = _ward_dist(centroid, size, n_features, i, j)
                        if not _heap_push(heap_d, heap_i, heap_j, heap_size,
                                          d, i, j):
                            overflow = True
                            break
                if overflow:
                    break

        if not overflow:
            status = _heap_greedy_loop(
                centroid, size, alive, parent,
                adj_head, adj_cluster, adj_next, pool_top,
                heap_d, heap_i, heap_j, heap_size,
                not_visited,
                out_a, out_b, out_d, n_features, n_samples, n_merges_total,
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
            heap_cap *= 2
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
