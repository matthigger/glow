"""Test the M_{ij}-kernel precompute for Y_r^*T Y_r^* under Freedman-Lane.

Under FL with the same permutation P per voxel,
    Y_v^* = M Y_v,  M = P(I - Q0 Q0^T) + Q0 Q0^T
the region's b x b outer-product sum decomposes as
    Y_r^*T Y_r^* = sum_v Y_v^T Y_v + C + C.T
where
    C_{ij} = <P, M_{ij}>_F  with  M_{ij} = sum_v (Y_v^par)[:,i] (Y_v^perp)[:,j]^T.

Because P is a permutation matrix, <P, M_{ij}>_F reduces to an N-element
gather along the "permutation diagonal":
    C_{ij} = sum_k M_{ij}[k, pi(k)].

These tests build a region with controlled energy in each of the q0/q1/q2
subspaces (q0 non-trivial — not just a constant vector) and confirm the
identity to machine precision.
"""
import numpy as np
import pytest


# --- self-contained helpers (no glow imports for math) -----------------------

def _decompose_x(x, contrast):
    """Orthonormal row-bases (q0, q1, q2) for nuisance/interest/residual."""
    x_nuis = x[~contrast]
    q0_T, _ = np.linalg.qr(x_nuis.T)                    # (N, a0)
    q_full_T, _ = np.linalg.qr(x.T)                     # (N, a)
    proj = q_full_T - q0_T @ (q0_T.T @ q_full_T)
    u, s, _ = np.linalg.svd(proj, full_matrices=False)
    q1_T = u[:, s > 1e-10]
    Q01 = np.hstack([q0_T, q1_T])
    p_resid = np.eye(x.shape[1]) - Q01 @ Q01.T
    u, s, _ = np.linalg.svd(p_resid, full_matrices=False)
    q2_T = u[:, s > 1e-10]
    return q0_T.T, q1_T.T, q2_T.T


def _build_y_with_subspace_energy(b, num_vox, q0, q1, q2, rng):
    """Y_v with equal-energy components in q0, q1, q2.  Shape (b, N, num_vox)."""
    def unit(shape):
        v = rng.standard_normal(shape)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)
    z0 = unit((b, num_vox, q0.shape[0]))
    z1 = unit((b, num_vox, q1.shape[0]))
    z2 = unit((b, num_vox, q2.shape[0]))
    return (np.einsum('avk,kn->anv', z0, q0)
            + np.einsum('avk,kn->anv', z1, q1)
            + np.einsum('avk,kn->anv', z2, q2))


def _fl_apply_naive(y, P, Q0Q0T):
    """Y_v^* = M Y_v applied along image axis n.  M = P(I-Q0Q0T) + Q0Q0T."""
    I = np.eye(P.shape[0])
    M = P @ (I - Q0Q0T) + Q0Q0T
    return np.einsum('nm,bmv->bnv', M, y)


def _yty_naive(y_perm):
    """sum_v y[:,:,v] @ y[:,:,v].T  ->  (b, b)."""
    return np.einsum('anv,bnv->ab', y_perm, y_perm)


def _build_kernels(y, Q0Q0T):
    """Returns (M, const) with
        M[i,j,k,l] = sum_v (Q0Q0T y)[i,k,v] * ((I-Q0Q0T) y)[j,l,v]
        const     = sum_v Y_v^T Y_v  (b, b)
    """
    y_par = np.einsum('nm,bmv->bnv', Q0Q0T, y)
    y_per = y - y_par
    M = np.einsum('ikv,jlv->ijkl', y_par, y_per)
    const = np.einsum('inv,jnv->ij', y, y)
    return M, const


def _yty_fast(M, const, pi):
    """C_{ij} = sum_k M[i,j,k,pi(k)];  return const + C + C.T."""
    N = M.shape[-1]
    C = M[:, :, np.arange(N), pi].sum(axis=-1)
    return const + C + C.T


def _yty_fast_batched(M, const, pi_stack):
    """Same as _yty_fast but for many perms in one go.  pi_stack: (P, N)."""
    p_count, N = pi_stack.shape
    b1, b2 = M.shape[:2]
    i_idx = np.arange(b1)[None, :, None, None]
    j_idx = np.arange(b2)[None, None, :, None]
    k_idx = np.arange(N)[None, None, None, :]
    l_idx = pi_stack[:, None, None, :]
    gathered = M[i_idx, j_idx, k_idx, l_idx]
    C = gathered.sum(axis=-1)
    return const[None] + C + C.transpose(0, 2, 1)


# --- fixtures -----------------------------------------------------------------

@pytest.fixture
def fl_problem():
    """Build a non-trivial-Q0 FL problem (bias + 2 WGN nuisance + 2 interest)."""
    rng = np.random.default_rng(0)
    num_img, b, num_vox = 30, 3, 50
    n_nuis, n_intrst = 3, 2

    x = np.empty((n_nuis + n_intrst, num_img))
    x[0] = 1.0
    x[1:n_nuis] = rng.standard_normal((n_nuis - 1, num_img))
    x[n_nuis:] = rng.standard_normal((n_intrst, num_img))
    contrast = np.array([False] * n_nuis + [True] * n_intrst)

    q0, q1, q2 = _decompose_x(x, contrast)
    y = _build_y_with_subspace_energy(b, num_vox, q0, q1, q2, rng)
    Q0Q0T = q0.T @ q0
    return dict(x=x, contrast=contrast, q0=q0, q1=q1, q2=q2,
                y=y, Q0Q0T=Q0Q0T, num_img=num_img, b=b, num_vox=num_vox)


# --- tests --------------------------------------------------------------------

def test_q0_is_nontrivial(fl_problem):
    """Q0 should span more than just the constant vector."""
    q0 = fl_problem['q0']
    # at least one row of q0 has meaningful spread (i.e. isn't constant)
    stds = q0.std(axis=1)
    assert (stds > 1e-3).sum() >= 2, (
        f'q0 rows should include non-constant directions; stds={stds}'
    )


def test_subspaces_orthonormal(fl_problem):
    """q0/q1/q2 are orthonormal row-bases of mutually orthogonal subspaces."""
    q0, q1, q2 = fl_problem['q0'], fl_problem['q1'], fl_problem['q2']
    assert np.allclose(q0 @ q0.T, np.eye(q0.shape[0]), atol=1e-12)
    assert np.allclose(q1 @ q1.T, np.eye(q1.shape[0]), atol=1e-12)
    assert np.allclose(q2 @ q2.T, np.eye(q2.shape[0]), atol=1e-12)
    assert np.abs(q0 @ q1.T).max() < 1e-12
    assert np.abs(q0 @ q2.T).max() < 1e-12
    assert np.abs(q1 @ q2.T).max() < 1e-12


def test_y_has_subspace_energy(fl_problem):
    """Y has the expected ~b*num_vox energy in each of q0, q1, q2."""
    y, b, num_vox = fl_problem['y'], fl_problem['b'], fl_problem['num_vox']
    expected = b * num_vox
    for q in (fl_problem['q0'], fl_problem['q1'], fl_problem['q2']):
        coef = np.einsum('bnv,kn->bvk', y, q)
        e = float((coef ** 2).sum())
        assert e == pytest.approx(expected, rel=1e-10), (
            f'expected energy {expected}, got {e}'
        )


@pytest.mark.parametrize('perm_seed', [1, 2, 3, 42, 1234, 99999])
def test_fast_matches_naive_per_perm(fl_problem, perm_seed):
    """C_{ij} = <P, M_{ij}>_F reproduces naive sum_v Y_v^*T Y_v^*."""
    y = fl_problem['y']
    Q0Q0T = fl_problem['Q0Q0T']
    num_img = fl_problem['num_img']

    pi = np.random.default_rng(perm_seed).permutation(num_img)
    P = np.eye(num_img)[pi, :]

    y_perm = _fl_apply_naive(y, P, Q0Q0T)
    yty_naive = _yty_naive(y_perm)

    M, const = _build_kernels(y, Q0Q0T)
    yty_fast = _yty_fast(M, const, pi)

    np.testing.assert_allclose(yty_fast, yty_naive, atol=1e-11, rtol=1e-11)


def test_identity_perm_recovers_unpermuted(fl_problem):
    """pi = arange(N) should produce the unpermuted outer product exactly."""
    y = fl_problem['y']
    Q0Q0T = fl_problem['Q0Q0T']
    num_img = fl_problem['num_img']

    M, const = _build_kernels(y, Q0Q0T)
    yty_fast_id = _yty_fast(M, const, np.arange(num_img))
    yty_unperm = _yty_naive(y)

    np.testing.assert_allclose(yty_fast_id, yty_unperm, atol=1e-12, rtol=1e-12)


def test_fast_yty_is_symmetric(fl_problem):
    """Y_r^*T Y_r^* is symmetric (since it's a sum of outer products)."""
    y = fl_problem['y']
    Q0Q0T = fl_problem['Q0Q0T']
    num_img = fl_problem['num_img']

    M, const = _build_kernels(y, Q0Q0T)
    pi = np.random.default_rng(7).permutation(num_img)
    yty = _yty_fast(M, const, pi)
    np.testing.assert_allclose(yty, yty.T, atol=1e-12, rtol=1e-12)


def test_batched_matches_per_perm(fl_problem):
    """fast_yty_batched gives the same answer as a loop of per-perm fast_yty."""
    y = fl_problem['y']
    Q0Q0T = fl_problem['Q0Q0T']
    num_img = fl_problem['num_img']

    pi_stack = np.stack([
        np.random.default_rng(s).permutation(num_img)
        for s in [1, 2, 3, 42, 1234, 99999]
    ])
    M, const = _build_kernels(y, Q0Q0T)

    by_loop = np.stack([_yty_fast(M, const, pi) for pi in pi_stack])
    by_batch = _yty_fast_batched(M, const, pi_stack)

    np.testing.assert_allclose(by_batch, by_loop, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('b', [1, 2, 5])
def test_fast_matches_naive_varied_b(b):
    """Equivalence holds for b = 1, 2, 5."""
    rng = np.random.default_rng(b * 7)
    num_img, num_vox = 25, 30
    n_nuis, n_intrst = 2, 1
    x = np.empty((n_nuis + n_intrst, num_img))
    x[0] = 1.0
    x[1:n_nuis] = rng.standard_normal((n_nuis - 1, num_img))
    x[n_nuis:] = rng.standard_normal((n_intrst, num_img))
    contrast = np.array([False] * n_nuis + [True] * n_intrst)
    q0, q1, q2 = _decompose_x(x, contrast)
    y = _build_y_with_subspace_energy(b, num_vox, q0, q1, q2, rng)
    Q0Q0T = q0.T @ q0

    pi = rng.permutation(num_img)
    P = np.eye(num_img)[pi, :]
    yty_naive = _yty_naive(_fl_apply_naive(y, P, Q0Q0T))
    M, const = _build_kernels(y, Q0Q0T)
    yty_fast = _yty_fast(M, const, pi)
    np.testing.assert_allclose(yty_fast, yty_naive, atol=1e-11, rtol=1e-11)
