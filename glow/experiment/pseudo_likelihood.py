"""Pseudo-likelihood (PL) spatial model for the GLOW pipeline.

Provides an incremental bottom-up computation of PL for all regions
in a Ward hierarchy, suitable as a test statistic in permutation
testing.  The per-voxel PL-LLR (PL on full-model residuals minus PL
on null-model residuals, divided by region size) replaces the MANCOVA
LLR when activated.

The PL model for each voxel v in a region R is:

    obs_v ~ N(alpha * sum_{j in N(v) cap R} obs_j,  sigma^2)

where N(v) are the spatial neighbours of v within the analysis mask.
Parameters (alpha, sigma^2) are estimated by OLS from sufficient
statistics accumulated bottom-up through the Ward hierarchy.
"""

from collections import deque
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import generate_binary_structure

import glow.graph
from glow.mask import get_neighbor_offsets
from .mancova import decompose


# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------

class VoxelStats(NamedTuple):
    """Per-voxel sufficient statistics for the PL model."""
    xy_full: np.ndarray   # (V,) dot(obs, full_neighbor_sum) per voxel
    xx_full: np.ndarray   # (V,) dot(full_neighbor_sum, full_neighbor_sum)
    yy_vox: np.ndarray    # (V,) dot(obs, obs) per voxel
    degree: np.ndarray    # (V,) int32 neighbour count in the full mask


class MergeEdges(NamedTuple):
    """CSR-packed edge and affected-voxel data per merge step."""
    edge_v: np.ndarray    # (E,) int64, one endpoint per cross-edge
    edge_w: np.ndarray    # (E,) int64, other endpoint
    edge_ptr: np.ndarray  # (num_internal+1,) int64, CSR pointers into edge_v/w
    aff_arr: np.ndarray   # (A,) int64, unique affected voxels per merge
    aff_ptr: np.ndarray   # (num_internal+1,) int64, CSR pointers into aff_arr


# ---------------------------------------------------------------------------
# adjacency
# ---------------------------------------------------------------------------

def build_adjacency(mask_idx):
    """Build spatial adjacency from a voxel index mask.

    Uses 6-connectivity for 3D masks, 4-connectivity for 2D.

    Args:
        mask_idx: int array (D, H, W) or (H, W); -1 for inactive voxels,
            otherwise the global voxel index.

    Returns:
        neighbors: dict mapping voxel index -> list of neighbor indices
        W: (V, V) sparse CSR adjacency matrix
    """
    ndim = mask_idx.ndim
    struct = generate_binary_structure(ndim, 1)
    offsets = get_neighbor_offsets(struct)
    shape = np.array(mask_idx.shape)
    active = np.argwhere(mask_idx >= 0)
    num_vox = int((mask_idx >= 0).sum())

    neighbors = {}
    rows, cols = [], []
    for coord in active:
        v = int(mask_idx[tuple(coord)])
        nb = []
        for off in offsets:
            nc = coord + off
            if np.all(nc >= 0) and np.all(nc < shape):
                nv = int(mask_idx[tuple(nc)])
                if nv >= 0:
                    nb.append(nv)
                    rows.append(v)
                    cols.append(nv)
        neighbors[v] = nb

    W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                       shape=(num_vox, num_vox))
    return neighbors, W


# ---------------------------------------------------------------------------
# per-voxel sufficient statistics
# ---------------------------------------------------------------------------

def precompute_voxel_stats(obs, W):
    """Compute per-voxel sufficient statistics using full-mask neighbours.

    For each voxel v, computes sums over all K observation channels:
      xy_full[v] = sum_k  obs[k,v] * (sum_{j in N(v)} obs[k,j])
      xx_full[v] = sum_k  (sum_{j in N(v)} obs[k,j])^2
      yy_vox[v]  = sum_k  obs[k,v]^2

    Args:
        obs: (K, V) observation matrix
        W: (V, V) sparse adjacency matrix

    Returns:
        VoxelStats
    """
    m = W.dot(obs.T).T                        # (K, V) full neighbour sums
    xy_full = np.sum(obs * m, axis=0)          # (V,)
    xx_full = np.sum(m * m, axis=0)            # (V,)
    yy_vox = np.sum(obs * obs, axis=0)         # (V,)
    degree = np.diff(W.indptr).astype(np.int32)
    del m
    return VoxelStats(xy_full=xy_full, xx_full=xx_full,
                      yy_vox=yy_vox, degree=degree)


# ---------------------------------------------------------------------------
# LCA edge assignment
# ---------------------------------------------------------------------------

def compute_lca_edges(children, num_vox, neighbors):
    """Assign every adjacency edge to its lowest common ancestor merge node.

    For each pair of neighbouring voxels (v, w), finds the internal node
    where v and w first join the same region.  Returns flat CSR-style
    arrays for cache-friendly iteration during the bottom-up sweep.

    Args:
        children: (num_internal, 2) Ward children array
        num_vox: number of leaf voxels
        neighbors: dict voxel -> list of neighbor indices

    Returns:
        MergeEdges
    """
    num_internal = children.shape[0]
    num_nodes = num_vox + num_internal

    parent = np.full(num_nodes, -1, dtype=np.int32)
    for i in range(num_internal):
        node = num_vox + i
        parent[children[i, 0]] = node
        parent[children[i, 1]] = node

    # BFS from all roots to compute depth (supports forests)
    roots = np.where(parent == -1)[0]
    internal_roots = roots[roots >= num_vox]

    depth = np.zeros(num_nodes, dtype=np.int32)
    for r in internal_roots:
        q = deque([r])
        while q:
            u = q.popleft()
            if u >= num_vox:
                for ch in children[u - num_vox]:
                    depth[ch] = depth[u] + 1
                    q.append(ch)

    # walk up from both endpoints to find LCA for each edge
    merge_edges = {}
    for v, nbs in neighbors.items():
        for w in nbs:
            if w <= v:
                continue
            u, t = int(v), int(w)
            while u != t:
                if u == -1 or t == -1:
                    break
                if depth[u] > depth[t]:
                    u = parent[u]
                else:
                    t = parent[t]
            if u != t:
                continue
            merge_edges.setdefault(u, []).append((v, w))

    # flatten into CSR-style arrays
    ev, ew = [], []
    aff_flat = []
    edge_ptr = np.zeros(num_internal + 1, dtype=np.int64)
    aff_ptr = np.zeros(num_internal + 1, dtype=np.int64)
    for i in range(num_internal):
        node = num_vox + i
        edges = merge_edges.get(node, [])
        s = set()
        for v, w in edges:
            ev.append(v)
            ew.append(w)
            s.add(v)
            s.add(w)
        edge_ptr[i + 1] = len(ev)
        aff_flat.extend(sorted(s))
        aff_ptr[i + 1] = len(aff_flat)

    return MergeEdges(
        edge_v=np.array(ev, dtype=np.int64),
        edge_w=np.array(ew, dtype=np.int64),
        edge_ptr=edge_ptr,
        aff_arr=np.array(aff_flat, dtype=np.int64),
        aff_ptr=aff_ptr,
    )


# ---------------------------------------------------------------------------
# bottom-up PL (interior + boundary)
# ---------------------------------------------------------------------------

def compute_tree_pl(children, num_vox, obs_T, vstats, edges):
    """Incremental bottom-up pseudo-likelihood for all tree nodes.

    Maintains a global (V, K) array ``pm`` of partial-neighbour sums.
    At each merge, only the voxels on the merge interface are touched:

    1. Remove old boundary contributions for affected voxels.
    2. Update ``pm`` for each cross-edge endpoint.
    3. Reclassify affected voxels as interior or boundary.
    4. Add updated contributions.

    Args:
        children: (num_internal, 2) Ward children
        num_vox: number of leaf voxels
        obs_T: (V, K) row-major observation matrix
        vstats: VoxelStats with precomputed full-neighbor stats
        edges: MergeEdges with LCA edge assignments

    Returns:
        Tuple of 8 arrays, each (num_nodes,):
        (S_xy_int, S_xx_int, S_yy_int, n_int,
         S_xy_bnd, S_xx_bnd, S_yy_bnd, n_bnd)
    """
    num_internal = children.shape[0]
    num_nodes = num_vox + num_internal

    pm = np.zeros((num_vox, obs_T.shape[1]), dtype=obs_T.dtype)
    missing = vstats.degree.copy()

    S_xy_int = np.zeros(num_nodes)
    S_xx_int = np.zeros(num_nodes)
    S_yy_int = np.zeros(num_nodes)
    n_int = np.zeros(num_nodes, dtype=np.int32)

    S_xy_bnd = np.zeros(num_nodes)
    S_xx_bnd = np.zeros(num_nodes)
    S_yy_bnd = np.zeros(num_nodes)
    n_bnd = np.zeros(num_nodes, dtype=np.int32)

    S_yy_bnd[:num_vox] = vstats.yy_vox
    n_bnd[:num_vox] = 1

    for i in range(num_internal):
        node = num_vox + i
        left, right = children[i]

        ixy = S_xy_int[left] + S_xy_int[right]
        ixx = S_xx_int[left] + S_xx_int[right]
        iyy = S_yy_int[left] + S_yy_int[right]
        ni = int(n_int[left]) + int(n_int[right])

        bxy = S_xy_bnd[left] + S_xy_bnd[right]
        bxx = S_xx_bnd[left] + S_xx_bnd[right]
        byy = S_yy_bnd[left] + S_yy_bnd[right]
        nb = int(n_bnd[left]) + int(n_bnd[right])

        e_lo = int(edges.edge_ptr[i])
        e_hi = int(edges.edge_ptr[i + 1])
        a_lo = int(edges.aff_ptr[i])
        a_hi = int(edges.aff_ptr[i + 1])

        if e_lo < e_hi:
            aff = edges.aff_arr[a_lo:a_hi]

            o_a = obs_T[aff]
            p_a = pm[aff]
            bxy -= np.dot(o_a.ravel(), p_a.ravel())
            bxx -= np.dot(p_a.ravel(), p_a.ravel())

            vs = edges.edge_v[e_lo:e_hi]
            ws = edges.edge_w[e_lo:e_hi]
            np.add.at(pm, vs, obs_T[ws])
            np.add.at(pm, ws, obs_T[vs])
            np.subtract.at(missing, vs, 1)
            np.subtract.at(missing, ws, 1)

            now_int = missing[aff] == 0
            int_v = aff[now_int]
            bnd_v = aff[~now_int]

            if len(int_v) > 0:
                ixy += vstats.xy_full[int_v].sum()
                ixx += vstats.xx_full[int_v].sum()
                iyy += vstats.yy_vox[int_v].sum()
                ni += len(int_v)
                byy -= vstats.yy_vox[int_v].sum()
                nb -= len(int_v)

            if len(bnd_v) > 0:
                o_b = obs_T[bnd_v]
                p_b = pm[bnd_v]
                bxy += np.dot(o_b.ravel(), p_b.ravel())
                bxx += np.dot(p_b.ravel(), p_b.ravel())

        S_xy_int[node] = ixy
        S_xx_int[node] = ixx
        S_yy_int[node] = iyy
        n_int[node] = ni
        S_xy_bnd[node] = bxy
        S_xx_bnd[node] = bxx
        S_yy_bnd[node] = byy
        n_bnd[node] = nb

    return (S_xy_int, S_xx_int, S_yy_int, n_int,
            S_xy_bnd, S_xx_bnd, S_yy_bnd, n_bnd)


# ---------------------------------------------------------------------------
# vectorised PL / iid stats
# ---------------------------------------------------------------------------

def pl_stats(S_xy, S_xx, S_yy, n, K):
    """Compute PL and iid log-likelihoods from accumulated sums.

    Args:
        S_xy: (N,) sum of obs * neighbor_sum per node
        S_xx: (N,) sum of neighbor_sum^2 per node
        S_yy: (N,) sum of obs^2 per node
        n: (N,) int, voxel count per node
        K: int, number of observation channels (b * num_img)

    Returns:
        ll_pl: (N,) PL log-likelihood per node
        ll_iid: (N,) iid log-likelihood per node
    """
    valid = n >= 2

    # iid model:  obs[k,v] ~ N(0, sigma^2)
    s2_iid = np.where(valid, S_yy / (K * n), np.nan)
    s2_iid = np.where(s2_iid > 0, s2_iid, np.nan)
    ll_iid = np.where(
        np.isfinite(s2_iid),
        -0.5 * K * n * (np.log(2 * np.pi * s2_iid) + 1),
        np.nan)

    # PL model:  obs[k,v] ~ N(alpha * neighbor_sum[k,v], sigma^2)
    has_sp = valid & (S_xx > 0)
    S_xx_safe = np.where(has_sp, S_xx, 1.0)
    rss = np.where(has_sp, S_yy - S_xy ** 2 / S_xx_safe, S_yy)
    s2_pl = np.where(valid, rss / (K * n), np.nan)
    s2_pl = np.where(s2_pl > 0, s2_pl, np.nan)
    ll_pl = np.where(
        np.isfinite(s2_pl),
        -0.5 * K * n * (np.log(2 * np.pi * s2_pl) + 1),
        np.nan)

    return ll_pl, ll_iid


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------

def _run_pl_pass(P, y, num_vox, children, W, edges):
    """Project y through P and run one bottom-up PL pass.

    Returns (ll_pl, K) where ll_pl is (num_nodes,) and K is the
    number of observation channels.
    """
    obs = np.einsum('ij,bjv->biv', P, y).reshape(-1, num_vox)
    K = obs.shape[0]
    vstats = precompute_voxel_stats(obs, W)
    obs_T = np.ascontiguousarray(obs.T)
    del obs

    ixy, ixx, iyy, ni, bxy, bxx, byy, nb = compute_tree_pl(
        children, num_vox, obs_T, vstats, edges)
    del obs_T

    ll_pl, _ = pl_stats(ixy + bxy, ixx + bxx, iyy + byy, ni + nb, K)
    return ll_pl, K


def get_pl_stat(exp, children, mask_idx, neighbors=None, W=None):
    """Compute PL-LLR for all regions in a Ward hierarchy.

    PL-LLR = [PL(full-model residuals) - PL(null-model residuals)] / n,
    i.e. the per-voxel pseudo-likelihood ratio.  Normalising by region
    size is essential: the raw PL is a sum over voxels, so the total
    grows monotonically with n even after size-adjustment (by concavity
    of log).  Per-voxel normalisation restores the property that the
    adjusted stat is highest for tight-fitting regions.

    The two PL passes run sequentially to limit peak memory (each
    requires a (V, K) working array).

    Args:
        exp: Experiment with y (b, num_img, num_vox), x, contrast
        children: (num_internal, 2) Ward children array
        mask_idx: int mask array (3D or 2D)
        neighbors: optional pre-built neighbor dict (for caching)
        W: optional pre-built sparse adjacency (for caching)

    Returns:
        stat: (num_nodes,) per-voxel PL-LLR (NaN where n < 2)
        size: (num_nodes,) int region sizes
    """
    b, num_img, num_vox = exp.y.shape

    if neighbors is None or W is None:
        neighbors, W = build_adjacency(mask_idx)

    edges = compute_lca_edges(children, num_vox, neighbors)

    q0, q1, q2 = decompose(exp.x, exp.contrast)
    P_alt = q2.T @ q2
    P_null = np.eye(num_img) - q0.T @ q0

    ll_alt, _ = _run_pl_pass(P_alt, exp.y, num_vox, children, W, edges)
    ll_null, _ = _run_pl_pass(P_null, exp.y, num_vox, children, W, edges)

    size = glow.graph.node_sum(np.ones(num_vox, dtype=int), children)
    stat = (ll_alt - ll_null) / size.astype(float)

    return stat, size
