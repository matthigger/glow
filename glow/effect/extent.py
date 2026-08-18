"""Extenters: sample the contiguous voxel support (extent) of an effect.

An extenter is a spec: every knob that changes the sampled mask -- the
geometry (radius / n_vox / connected), the RNG seed, the seed voxel vox_init,
and whether to resample until contiguous (contiguous / max_iter) -- is an
attribute set at construction. Calling the extenter takes only the data to
sample over (mask_idx, optional y) plus a cosmetic verbose flag, so the mask
is a pure function of the spec and the data.
"""

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from scipy import sparse
from scipy.ndimage import label
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from sklearn.feature_extraction.image import grid_to_graph
from tqdm import tqdm

# 6-connectivity (face neighbours only) for 3D, matching Ward clustering, so
# effects grow and clustering merges share the same neighbour definition.
CONNECTIVITY_3D = generate_binary_structure(3, 1)


class ContiguousRegionNotFound(RuntimeError):
    """Raised when no contiguous region is sampled within max_iter tries."""


class Extenter(ABC):
    """Base extenter: a spec sampling an effect's spatial extent.

    Concrete extenters implement _sample (a single draw given a seed); this
    base wraps it in the resample-until-contiguous loop.

    Attributes:
        seed (int | None): RNG seed for reproducibility (None draws fresh).
        vox_init (int | None): seed voxel index; drawn from seed when None.
        contiguous (bool): resample until the mask is a single connected
            component (face-connectivity), up to max_iter tries.
        max_iter (int): maximum resample attempts when contiguous is True.
    """

    def __init__(self, *, seed=None, vox_init=None, contiguous=False,
                 max_iter=100):
        self.seed = None if seed is None else int(seed)
        self.vox_init = None if vox_init is None else int(vox_init)
        self.contiguous = bool(contiguous)
        self.max_iter = int(max_iter)

    def __repr__(self):
        """Class name + each public, set (non-None) attribute.

        Generic over the concrete extenter's own attributes (radius / n_vox /
        connected, plus the base seed / contiguity knobs), so a subclass needs
        no repr of its own: a set spec like ExtenterSphere(seed=0, n_vox=100)
        reads in a log or recorded cell instead of '<...object at 0x...>'.
        None-valued knobs (the unused half of an either/or, e.g. radius when
        n_vox is given) are dropped.
        """
        fields = ', '.join(f'{k}={v}' for k, v in vars(self).items()
                           if not k.startswith('_') and v is not None)
        return f'{type(self).__name__}({fields})'

    def __call__(self, mask_idx, y=None, verbose: bool = False):
        """Return a boolean mask defining the extent.

        Re-samples with a fresh derived seed until the mask is a single
        connected component when contiguous is True, otherwise returns the
        first draw.

        Args:
            mask_idx (np.array): voxel index array (-1 outside analysis)
            y (np.array): (b, num_img, num_vox) image intensities; required
                by data-driven extenters, ignored by geometric ones
            verbose (bool): if True, show a progress bar (data-driven only)

        Returns:
            mask (np.array): boolean, True within extent

        Raises:
            ContiguousRegionNotFound: if no contiguous region is found in
                max_iter tries (only when contiguous is True)
        """
        _seed = self.seed
        for _ in range(self.max_iter):
            mask = self._sample(mask_idx=mask_idx, y=y, seed=_seed,
                                verbose=verbose)
            if not self.contiguous:
                return mask

            structure = (CONNECTIVITY_3D if mask_idx.ndim == 3
                         else generate_binary_structure(2, 1))
            _, n_components = label((mask_idx >= 0) & mask,
                                    structure=structure)
            if n_components == 1:
                return mask

            # not contiguous: derive a fresh seed from the previous and retry
            rng = np.random.default_rng(seed=_seed)
            _seed = int(rng.integers(low=0, high=2 ** 32, size=1)[0])

        raise ContiguousRegionNotFound()

    @abstractmethod
    def _sample(self, *, mask_idx, y, seed, verbose):
        """Sample one boolean extent mask for the seed (no resampling)."""


class ExtenterSphere(Extenter):
    """Build effect extent as a randomly placed sphere.

    Specify exactly one of radius or n_vox. With radius, the extent is a
    dilation ball of that many steps; with n_vox, the region grows until it
    holds n_vox voxels (trimming the outer shell to hit the count exactly).

    Attributes:
        radius (int | None): dilation radius in voxels, XOR with n_vox
        n_vox (int | None): target voxel count, XOR with radius
        connected (bool): if True, restrict the seed and growth to a single
            connected component of the analysis mask
    """

    def __init__(self, *, radius=None, n_vox=None, connected=False,
                 seed=None, vox_init=None, contiguous=False, max_iter=100):
        super().__init__(seed=seed, vox_init=vox_init,
                         contiguous=contiguous, max_iter=max_iter)
        if radius is None and n_vox is None:
            raise ValueError('radius or n_vox required')
        if radius is not None and n_vox is not None:
            raise ValueError('specify radius or n_vox, not both')
        self.radius = None if radius is None else int(radius)
        self.n_vox = None if n_vox is None else int(n_vox)
        self.connected = bool(connected)

    def _sample(self, *, mask_idx, y=None, seed=None, verbose=False):
        rng = np.random.default_rng(seed=seed)
        vox_init = self.vox_init
        mask_bool = mask_idx > -1
        structure = (CONNECTIVITY_3D if mask_idx.ndim == 3
                     else generate_binary_structure(2, 1))
        if self.connected:
            comp_idx, num_comp = label(mask_bool, structure=structure)
            if num_comp == 0:
                raise ValueError('no voxels available in mask')
            comp_sizes = np.bincount(comp_idx.ravel())
            comp_sizes[0] = 0
            if self.n_vox is not None:
                eligible = np.flatnonzero(comp_sizes >= self.n_vox)
                if eligible.size == 0:
                    raise ValueError(
                        'no connected component has >= n_vox voxels')
            else:
                eligible = np.flatnonzero(comp_sizes > 0)

            if vox_init is None:
                comp_id = rng.choice(eligible)
                mask_bool = comp_idx == comp_id
                vox_init = rng.choice(mask_idx[mask_bool])
            else:
                comp_id = comp_idx[mask_idx == vox_init]
                if comp_id.size == 0 or comp_id[0] == 0:
                    raise ValueError('vox_init not in mask')
                comp_id = comp_id[0]
                if self.n_vox is not None and comp_sizes[comp_id] < self.n_vox:
                    raise ValueError(
                        'vox_init component has fewer voxels than n_vox')
                mask_bool = comp_idx == comp_id
        else:
            if vox_init is None:
                vox_init = rng.choice(mask_idx[mask_bool])

        mask = mask_idx == vox_init
        mask = np.logical_and(mask, mask_bool)

        if self.n_vox is None:
            mask = binary_dilation(mask, structure=structure,
                                   iterations=self.radius)
            # clip the dilation so the extent stays within the analysis mask
            return np.logical_and(mask, mask_bool)

        # grow until reaching desired voxel count
        count = int(mask.sum())
        while count < self.n_vox:
            prev_mask = mask
            mask = binary_dilation(mask, structure=structure, iterations=1)
            mask = np.logical_and(mask, mask_bool)
            count = int(mask.sum())

        # trim excess from the outer shell to land on n_vox exactly
        excess = count - self.n_vox
        if excess > 0:
            shell = np.logical_and(mask, np.logical_not(prev_mask))
            shell_idx = np.flatnonzero(shell)
            if shell_idx.size >= excess:
                mask_flat = mask.ravel()
                mask_flat[shell_idx[:excess]] = False
                mask = mask_flat.reshape(mask.shape)
        return mask


def iter_vox_neighbor(mask, mask_idx):
    """Yield voxel indices of all face-neighbours of mask.

    Uses 6-connectivity for 3D, matching Ward clustering.

    Args:
        mask (np.array): boolean, same shape as image
        mask_idx (np.array): voxel index array (-1 outside analysis)

    Yields:
        vox_idx (int): neighbour voxel index (non-reflexive)
    """
    # 6-connectivity (face neighbours) for 3D matches clustering connectivity
    structure = (CONNECTIVITY_3D if (mask_idx.ndim == 3)
                 else generate_binary_structure(2, 1))
    mask_neighbor = (binary_dilation(mask, structure=structure)
                     & np.logical_not(mask))
    for vox_idx in mask_idx[mask_neighbor]:
        if vox_idx > -1:
            yield vox_idx


class ExtenterMinVar(Extenter):
    """Grow effect extent from a seed voxel to greedily minimise variance.

    Attributes:
        n_vox (int): target voxel count for the grown extent
    """

    def __init__(self, *, n_vox=None, seed=None, vox_init=None,
                 contiguous=False, max_iter=100):
        super().__init__(seed=seed, vox_init=vox_init,
                         contiguous=contiguous, max_iter=max_iter)
        if n_vox is None:
            raise ValueError('n_vox required')
        self.n_vox = int(n_vox)

    def _sample(self, *, mask_idx, y=None, seed=None, verbose=False):
        assert y is not None, 'ExtenterMinVar requires y'

        # choose a random initial voxel
        vox_init = self.vox_init
        if vox_init is None:
            rng = np.random.default_rng(seed=seed)
            mask_bool = mask_idx > -1
            vox_init = rng.choice(mask_idx[mask_bool])
        mask = mask_idx == vox_init
        assert mask.sum(), 'vox_init not in mask_idx'

        mu = y[:, :, vox_init]
        y_norm_sq = (mu ** 2).sum()

        tqdm_dict = dict(total=self.n_vox - 1,
                         disable=not verbose,
                         desc='finding min var extent')
        for n in tqdm(range(1, self.n_vox), **tqdm_dict):
            min_var = np.inf
            vox_idx_best = None

            # incremental mean weights: adding the (n+1)-th voxel re-weights
            # the running mean mu by n/(n+1) and the candidate voxel by 1/(n+1)
            lam0 = n / (n + 1)
            lam1 = 1 / (n + 1)
            for vox_idx in iter_vox_neighbor(mask=mask, mask_idx=mask_idx):
                y_new = y[:, :, vox_idx]
                _mu = lam0 * mu + lam1 * y_new
                var = ((y_norm_sq + (y_new ** 2).sum()) / (n + 1) -
                       (_mu ** 2).sum())

                if var < min_var:
                    min_var = var
                    vox_idx_best = vox_idx

            assert vox_idx_best is not None

            mask[mask_idx == vox_idx_best] = True
            y_new = y[:, :, vox_idx_best]
            mu = lam0 * mu + lam1 * y_new
            y_norm_sq += (y_new ** 2).sum()

        return mask


class ExtenterSplit:
    """Bisect a base extent into two adjacent halves.

    A compound spec that holds one base Extenter defining the whole area;
    fit() grows that extent and spectrally bisects it (split_mask_spectral, a
    Fiedler cut) into two contiguous near-equal halves, returned as a pair.
    The two-effect cleaving trial plants one effect on each half. The base
    carries its own seed and geometry, so a single ExtenterSplit yields both
    halves -- no per-half instance.

    Unlike an Extenter (one mask via __call__), this returns a pair, so it is
    a sibling spec rather than an Extenter subclass.

    Attributes:
        base (Extenter): the extent to bisect; must yield a single connected
            component (e.g. ExtenterSphere(connected=True)), as
            split_mask_spectral requires.
    """

    def __init__(self, *, base=None):
        if base is None:
            raise ValueError('base required')
        self.base = base

    def __repr__(self):
        """Class name + the base extent it bisects (rendered recursively)."""
        return f'{type(self).__name__}(base={self.base!r})'

    def fit(self, mask_idx, y=None, verbose: bool = False):
        """Grow the base extent and bisect it into two spectral halves.

        Args:
            mask_idx (np.array): voxel index array (-1 outside analysis)
            y (np.array): (b, num_img, num_vox) intensities; required by
                data-driven bases (ExtenterMinVar), ignored by geometric ones
            verbose (bool): forwarded to the base extenter

        Returns:
            mask0 (np.array): boolean, the first spectral half
            mask1 (np.array): boolean, the other half. mask0 | mask1 is the
                base extent and the two are disjoint.
        """
        full = self.base(mask_idx=mask_idx, y=y, verbose=verbose)
        return split_mask_spectral(full)


def _face_offsets(ndim):
    """Return the face-neighbour index offsets for an ndim array.

    Four offsets in 2D, six in 3D -- the same 4-/6-connectivity that
    CONNECTIVITY_3D encodes for dilation, here as explicit index deltas for
    breadth-first traversal (one axis stepped by +-1, the rest held at 0).

    Args:
        ndim (int): number of array dimensions

    Returns:
        offsets (tuple): each element an (ndim,) int tuple with a single
            nonzero entry of +1 or -1
    """
    offsets = []
    for axis in range(ndim):
        for step in (-1, 1):
            off = [0] * ndim
            off[axis] = step
            offsets.append(tuple(off))
    return tuple(offsets)


def iter_bfs(mask, ijk, blocked=None):
    """Yield mask voxels in breadth-first order from a seed voxel.

    Neighbours are face-connected (4-connectivity in 2D, 6-connectivity in
    3D), matching CONNECTIVITY_3D and Ward clustering.

    Args:
        mask (np.array): boolean spatial mask, (X, Y) or (X, Y, Z), True for
            in-mask voxels
        ijk (tuple): (i, j[, k]) seed voxel, must be in the mask
        blocked (np.array): boolean, same shape as mask, voxels to neither
            yield nor expand through. Checked when a voxel is dequeued, not
            when it is enqueued, so the caller may mutate the array between
            pulls (two competing fronts can share it this way).

    Yields:
        ijk (tuple): (i, j[, k]) voxel
        d (int): graph distance (face-connectivity) from the seed
    """
    assert mask[ijk], 'seed voxel not in mask'
    shape = mask.shape
    offsets = _face_offsets(mask.ndim)
    visited = np.zeros(shape, dtype=bool)
    visited[ijk] = True
    queue = deque([(ijk, 0)])
    while queue:
        ijk, d = queue.popleft()
        if blocked is not None and blocked[ijk]:
            continue
        yield ijk, d
        for off in offsets:
            _ijk = tuple(c + o for c, o in zip(ijk, off))
            if (all(0 <= _ijk[ax] < shape[ax] for ax in range(mask.ndim))
                    and mask[_ijk]
                    and not visited[_ijk]):
                visited[_ijk] = True
                queue.append((_ijk, d + 1))


def get_diameter(mask):
    """Find two voxels at (approximately) maximal graph distance.

    Iterated BFS sweep (Handler 1973): BFS from a seed, jump to the farthest
    voxel found, repeat until the farthest distance stops increasing. Exact on
    trees, a lower bound on general graphs.

    Args:
        mask (np.array): boolean spatial mask, (X, Y) or (X, Y, Z), must be a
            single connected component

    Returns:
        ijk0 (tuple): (i, j[, k]) one endpoint of the diameter
        ijk1 (tuple): (i, j[, k]) the other endpoint

    Raises:
        ValueError: if mask has more than one connected component
    """
    ijk0 = tuple(np.argwhere(mask)[0])
    d_last = None
    while True:
        # the last voxel out of iter_bfs is the farthest from the seed
        num_visited = 0
        for ijk1, d in iter_bfs(mask, ijk0):
            num_visited += 1
        if num_visited != mask.sum():
            raise ValueError('mask is not a single connected component')
        if d == d_last:
            return ijk0, ijk1
        d_last = d
        ijk0 = ijk1


def _fiedler_endpoints(mask):
    """Find two seed voxels at the extremes of the mask's Fiedler vector.

    Builds the face-connectivity graph over the mask's voxels, forms the
    graph Laplacian, and returns the voxels at the minimum and maximum of
    its Fiedler vector (the second-smallest Laplacian eigenvector; Fiedler
    1973). These sit across the graph's minimum bisection, so they are the
    spectral analogue of get_diameter's endpoints -- the seed pair that
    splits the mask into two balanced, geometrically natural halves.

    Node i of grid_to_graph(mask=mask) is the i-th True voxel in row-major
    order, matching np.argwhere(mask), so a Fiedler-vector index maps back
    to a voxel by indexing the argwhere coordinates.

    Args:
        mask (np.array): boolean spatial mask, (X, Y) or (X, Y, Z), must be
            a single connected component

    Returns:
        ijk0 (tuple): (i, j[, k]) voxel at the Fiedler minimum
        ijk1 (tuple): (i, j[, k]) voxel at the Fiedler maximum

    Raises:
        ValueError: if mask has more than one connected component
    """
    coords = np.argwhere(mask)
    n_vox = len(coords)

    A = grid_to_graph(*mask.shape, mask=mask)
    A = (A + A.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    n_comp, _ = connected_components(A, directed=False)
    if n_comp != 1:
        raise ValueError('mask is not a single connected component')

    if n_vox <= 2:
        # too small for eigsh (needs k < n_vox); the lone voxel is its own
        # pair (n_vox == 1) and the two voxels are each other's (n_vox == 2)
        node0, node1 = 0, n_vox - 1
    else:
        degree = np.asarray(A.sum(axis=1)).ravel()
        L = sparse.diags(degree) - A
        # k=2 returns the constant vector + Fiedler. eigsh's Lanczos start
        # is random by default, and at a near-degenerate second eigenvalue
        # (symmetric regions) that returns an arbitrary vector of the
        # eigenspace, so pin a fixed start: ExtenterSplit's halves 0/1 must
        # agree across two independent calls.
        v0 = np.random.default_rng(0).standard_normal(n_vox)
        evals, evecs = eigsh(L, k=2, which='SM', v0=v0)
        fiedler = evecs[:, np.argsort(evals)[1]]
        # canonicalize the endpoint pair by node index too, so any residual
        # sign ambiguity cannot swap which piece is 0
        node0, node1 = sorted((int(np.argmin(fiedler)),
                               int(np.argmax(fiedler))))

    return tuple(coords[node0]), tuple(coords[node1])


def split_mask_spectral(mask):
    """Split a mask into two contiguous pieces of (near) equal size.

    BFS fronts grow from two seeds chosen by spectral bisection -- the
    extremes of the mask's Fiedler vector (_fiedler_endpoints), which puts
    the cut across the graph's minimum bisection -- alternately claiming one
    voxel each until the mask is exhausted. Each front blocks on the other
    piece, so both pieces come out contiguous.

    Note:
        Sizes differ by at most one voxel unless one front gets walled in by
        the other; the trapped front then stops and the open front claims the
        remainder, trading balance for contiguity.

    Args:
        mask (np.array): boolean spatial mask, (X, Y) or (X, Y, Z), must be a
            single connected component

    Returns:
        mask0 (np.array): boolean, same shape as mask, piece grown from the
            Fiedler-minimum seed
        mask1 (np.array): boolean, the other piece. mask0 | mask1 == mask and
            mask0 & mask1 is empty

    Raises:
        ValueError: if mask has more than one connected component
    """
    ijk0, ijk1 = _fiedler_endpoints(mask)
    pieces = (np.zeros_like(mask), np.zeros_like(mask))
    iters = (iter_bfs(mask, ijk0, blocked=pieces[1]),
             iter_bfs(mask, ijk1, blocked=pieces[0]))
    num_remain = int(mask.sum())
    while num_remain:
        num_remain_start = num_remain
        for piece, _iter in zip(pieces, iters):
            for ijk, _ in _iter:
                piece[ijk] = True
                num_remain -= 1
                break
            if not num_remain:
                break
        assert num_remain < num_remain_start, 'no front can advance'
    return pieces
