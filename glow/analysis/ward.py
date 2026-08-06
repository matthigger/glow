"""Connectivity-constrained Ward clustering via Mullner's NN-array algorithm.

Faster drop-in for sklearn.cluster.ward_tree(X, connectivity=...) (Mullner
2011). Produces the same dendrogram as sklearn: the children array matches
exactly, and distances match to float64 accumulation order (relative
difference up to ~5e-14 over the 224618 merges of a full-brain mask, from
sklearn updating centroids by a different formula).

Algorithm: each cluster c tracks its current nearest neighbour nn[c] and
that distance nn_d[c]. A global min-heap is keyed by (nn_d[c], c). The next
merge is always the global min over alive clusters' NN distances, which is
the same edge sklearn's heap-greedy picks.

Compared to sklearn's single-global-edge-heap approach, this dramatically
cuts heap pressure: instead of pushing every edge (and accumulating ~95%
stale entries when endpoints merge away), we push at most one entry per
cluster per NN change. On the full-brain HCP mask (224619 voxels) this runs
~24x (b=6) to ~29x (b=1) faster than sklearn.

Cost is dominated by nearest-neighbour churn, not by the distance
arithmetic: b=1 rescans a cluster's adjacency 8.2 times per merge against
2.6 for b >= 3, because 1d centroids sit on a line and a merge easily
reorders which neighbour is nearest. So b=1 is the slow case (sklearn
shows the same split), and widening b is nearly free until b ~ 12.

Per-merge work:
- Pop outer heap; skip stale entries (cluster dead or distance changed).
- Build merged cluster's adjacency via union-find path compression on the
  adjacency linked list (same trick as sklearn's _get_parents).
- Compute new cluster's NN by linear scan of its adjacency.
- For each neighbour n: if n's NN was i/j (now dead), rescan n's adj;
  else update n's NN only if the new cluster is closer than its current.

The merge loop is a pointer chase over the adjacency pool, so its cost is
set by cache lines touched rather than flops; see the pool comment below
for the layout and node recycling that keep both bounded.

b is the clustering feature dimension: how many values per sample Ward
sees. cluster() flattens the imaging-feature axis against the projection
basis, so b = imaging feats x projection rank -- exactly glow's b under
the default FOCUS mode with a single contrast column, a multiple of it
under GLM_ERROR (x projection rank) or NAIVE (x num_img). It is never the
design-matrix a; the two are unrelated.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse.csgraph import connected_components


# --------------------------------------------------------------- primitives --


@njit(cache=True, boundscheck=False)
def _ward_dist(centroid, size, b, i, j):
    """Compute the Ward (variance-increase) distance between clusters i, j."""
    pa = 0.0
    for f in range(b):
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
#   adj[2 * node]: cluster id at that node
#   adj[2 * node + 1]: next node id (-1 if end)
#
# The two fields are interleaved in one int32 array so walking a list touches
# one cache line per node rather than two; int32 then fits 8 nodes per line.
# The merge loop is a pointer chase over this pool, so its cost tracks lines
# touched (~1.35x at full-brain num_vox over int64 field-parallel arrays).
#
# Nodes are recycled through a free list threaded on the next field and
# rooted at free_head[0]. Every merge retires both children's whole lists and
# every scan unlinks the dead nodes it walks past, so without recycling the
# pool grows to ~90 * n_samples and overflows, forcing a full restart of the
# merge loop. Recycling holds it at O(nnz).


@njit(cache=True, inline='always', boundscheck=False)
def _adj_free(adj, free_head, nid):
    """Return pool node nid to the free list."""
    adj[2 * nid + 1] = free_head[0]
    free_head[0] = nid


@njit(cache=True, boundscheck=False)
def _adj_prepend(adj_head, adj, pool_top, free_head, c, x):
    """Prepend neighbour x to c's adjacency list; return False if pool full."""
    nid = free_head[0]
    if nid >= 0:
        free_head[0] = adj[2 * nid + 1]
    else:
        nid = pool_top[0]
        if 2 * nid + 1 >= adj.shape[0]:
            return False
        pool_top[0] = nid + 1
    adj[2 * nid] = x
    adj[2 * nid + 1] = adj_head[c]
    adj_head[c] = nid
    return True


@njit(cache=True, boundscheck=False)
def _build_initial_adj(indptr, indices, adj_head, adj, pool_top, free_head,
                       n_samples):
    """Seed every leaf's adjacency list from the CSR connectivity graph."""
    for i in range(n_samples):
        for p in range(indptr[i], indptr[i + 1]):
            if not _adj_prepend(adj_head, adj, pool_top, free_head, i,
                                indices[p]):
                return False
    return True


@njit(cache=True, boundscheck=False)
def _scan_nn(centroid, size, alive, b, c, adj_head, adj, free_head):
    """Linear scan of c's adjacency for its current alive nearest neighbour.

    Returns (-1, inf) if c has no alive neighbours.
    """
    best_d = np.inf
    best = np.int64(-1)
    prev = np.int64(-1)
    nid = np.int64(adj_head[c])
    while nid >= 0:
        x = adj[2 * nid]
        nxt = np.int64(adj[2 * nid + 1])
        if alive[x]:
            d = _ward_dist(centroid, size, b, c, x)
            # Lex tie-break: prefer smaller index on ties (matches sklearn's
            # global-heap behaviour which orders by (d, k, c) tuples).
            if d < best_d or (d == best_d and (best < 0 or x < best)):
                best_d = d
                best = x
            prev = nid
        else:
            # Unlink the dead node so future scans don't re-walk it, and
            # recycle it.
            if prev < 0:
                adj_head[c] = nxt
            else:
                adj[2 * prev + 1] = nxt
            _adj_free(adj, free_head, nid)
        nid = nxt
    return best, best_d


# ---------------------------------------------------------------- main loop --


@njit(cache=True, inline='always', boundscheck=False)
def _merge_adj_into(adj_head, adj, pool_top, free_head,
                    parent, not_visited, src_head, k):
    """Splice src's adjacency list into cluster k's, dedup via not_visited.

    For each node in src's list, resolve its current root via union-find
    and, if not already attached to k, add the symmetric edge (root<->k).
    src is dead once merged, so its nodes are recycled as they are walked.
    Returns False on pool overflow, True otherwise.
    """
    nid = np.int64(src_head)
    while nid >= 0:
        yy = adj[2 * nid]
        nxt = np.int64(adj[2 * nid + 1])
        _adj_free(adj, free_head, nid)
        nid = nxt
        root = _find_root(parent, yy)
        if not_visited[root]:
            not_visited[root] = False
            if not _adj_prepend(adj_head, adj, pool_top, free_head, root, k):
                return False
            if not _adj_prepend(adj_head, adj, pool_top, free_head, k, root):
                return False
    return True


@njit(cache=True, boundscheck=False)
def _mullner_loop(
    centroid, size, alive, parent,
    adj_head, adj, pool_top, free_head,
    heap_d, heap_c, heap_size,
    nn, nn_d,
    not_visited,
    out_c0, out_c1, out_d, b, n_samples, n_merges,
):
    """Mullner NN-array main loop.

    Returns 0 on success, -1 if outer heap overflows, -2 if pool overflows.
    """
    # Initial NN per leaf
    for c in range(n_samples):
        x, d = _scan_nn(centroid, size, alive, b, c, adj_head, adj,
                        free_head)
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
                new_x, new_d = _scan_nn(centroid, size, alive, b, c_popped,
                                        adj_head, adj, free_head)
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

        i = np.int64(c_merge)
        j = np.int64(nn[i])
        merged = k - n_samples
        if i < j:
            out_c0[merged] = i
            out_c1[merged] = j
        else:
            out_c0[merged] = j
            out_c1[merged] = i
        out_d[merged] = nn_d[i]

        # merge centroids
        si = size[i]
        sj = size[j]
        st = si + sj
        for f in range(b):
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
        src_i = np.int64(adj_head[i])
        src_j = np.int64(adj_head[j])
        adj_head[i] = -1
        adj_head[j] = -1
        if not _merge_adj_into(adj_head, adj, pool_top, free_head,
                               parent, not_visited, src_i, k):
            return -2
        if not _merge_adj_into(adj_head, adj, pool_top, free_head,
                               parent, not_visited, src_j, k):
            return -2

        # One pass over k's adjacency:
        # - reset not_visited
        # - compute nn[k] (linear scan)
        # - update each neighbour n's NN if needed
        best_d = np.inf
        best = np.int64(-1)
        nid = np.int64(adj_head[k])
        while nid >= 0:
            n = adj[2 * nid]
            nid = np.int64(adj[2 * nid + 1])
            not_visited[n] = True

            d_kn = _ward_dist(centroid, size, b, k, n)
            if d_kn < best_d or (d_kn == best_d and (best < 0 or n < best)):
                best_d = d_kn
                best = n

            n_nn = nn[n]
            if n_nn == i or n_nn == j or n_nn < 0 or not alive[n_nn]:
                # n's old NN died → rescan n's adj
                new_x, new_d = _scan_nn(centroid, size, alive, b, n,
                                        adj_head, adj, free_head)
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


# -------------------------------------------------------- public entry point -


def ward_tree(X, connectivity, return_distance: bool = False):
    """Cluster agglomeratively under a connectivity constraint (Ward).

    Matches the signature and dendrogram of
    sklearn.cluster.ward_tree(X, connectivity=...).

    Args:
        X (np.array): (n_samples, b) feature matrix, b values per
            sample (see the module docstring).
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
    n_samples, b = X.shape

    A = sparse.csr_matrix(connectivity)
    A = (A + A.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    n_components, _ = connected_components(A, directed=False)
    n_merges_total = n_samples - n_components
    n_nodes = n_samples + n_merges_total

    # Cluster and pool-node ids are int32 (see the adjacency pool comment).
    if n_nodes > np.iinfo(np.int32).max:
        raise ValueError(f'ward_tree: {n_samples} samples exceeds the int32 '
                         'node-index limit')

    centroid = np.zeros((n_nodes, b), dtype=np.float64)
    centroid[:n_samples] = X
    size = np.zeros(n_nodes, dtype=np.float64)
    size[:n_samples] = 1.0
    alive = np.zeros(n_nodes, dtype=np.bool_)
    alive[:n_samples] = True
    parent = np.arange(n_nodes, dtype=np.int32)
    not_visited = np.ones(n_nodes, dtype=np.bool_)
    nn = np.full(n_nodes, -1, dtype=np.int32)
    nn_d = np.full(n_nodes, np.inf, dtype=np.float64)

    out_c0 = np.full(n_merges_total, -1, dtype=np.int64)
    out_c1 = np.full(n_merges_total, -1, dtype=np.int64)
    out_d = np.full(n_merges_total, np.inf, dtype=np.float64)

    # Adjacency pool: seeded with A.nnz nodes, then held near that level by
    # the free list (a merge retires as many nodes as it allocates).  The
    # slack absorbs the transient peak; doubled on overflow.
    init_pool = A.nnz + 4 * n_samples + 1024
    # Outer heap: one entry per cluster NN update.  Per merge ≤ D updates
    # plus the new k's entry.  Initial: N.  Generous: 16 * N, doubled on
    # overflow.
    init_heap = 16 * n_samples + 1024

    pool_size = init_pool
    heap_cap = init_heap

    for _retry in range(10):
        adj_head = np.full(n_nodes, -1, dtype=np.int32)
        adj = np.empty(2 * pool_size, dtype=np.int32)
        pool_top = np.array([0], dtype=np.int64)
        free_head = np.array([-1], dtype=np.int64)
        heap_d = np.empty(heap_cap, dtype=np.float64)
        heap_c = np.empty(heap_cap, dtype=np.int32)
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
        out_c0[:] = -1
        out_c1[:] = -1
        out_d[:] = np.inf

        # Build initial adjacency from connectivity (both directions).
        overflow = not _build_initial_adj(
            A.indptr, A.indices, adj_head, adj, pool_top, free_head,
            n_samples)

        if not overflow:
            status = _mullner_loop(
                centroid, size, alive, parent,
                adj_head, adj, pool_top, free_head,
                heap_d, heap_c, heap_size,
                nn, nn_d,
                not_visited,
                out_c0, out_c1, out_d, b, n_samples, n_merges_total,
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
    children[:, 0] = out_c0
    children[:, 1] = out_c1

    parents = np.arange(n_nodes, dtype=np.intp)
    pid = np.arange(n_samples, n_nodes, dtype=np.intp)
    parents[children[:, 0]] = pid
    parents[children[:, 1]] = pid

    if return_distance:
        distances = np.sqrt(2.0 * out_d)
        return children, n_components, n_samples, parents, distances
    return children, n_components, n_samples, parents
