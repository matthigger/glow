import numpy as np

a = 3
b = 2
num_img = 5
num_vox = 7

rng = np.random.default_rng(seed=0)
x = rng.standard_normal((a, num_img))
y3d = rng.standard_normal((b, num_img, num_vox))
y = y3d.reshape((b, -1), order='F')

# test: single voxel test projection as q.T @ q
q, r = np.linalg.qr(x.T)
q, r = q.T, r.T
p_trusted = x.T @ np.linalg.inv(x @ x.T) @ x
p_exp = q.T @ q
assert np.allclose(p_trusted, p_exp)

# test: multi vox q_multi = 1 / sqrt(r) [q, q, ...]
x_multi = np.kron(np.ones((1, num_vox)), x)
q_multi, r_multi = np.linalg.qr(x_multi.T)
q_multi, r_multi = q_multi.T, r_multi.T
_q_multi = np.kron(np.ones((1, num_vox)), q) / num_vox ** .5
assert np.allclose(_q_multi, q_multi)

# test: centering matrix for spatial cov
c = np.kron(np.ones((num_vox, 1)), np.eye(num_img)) / num_vox
y_mean = y3d.mean(axis=2)
assert np.allclose(y @ c, y_mean)

# test: spatial covariance compute
diff = y - np.kron(np.ones((1, num_vox)), y_mean)
space_cov_trusted = diff @ diff.T
space_cov = y @ (np.eye(num_vox * num_img) - num_vox * c @ c.T) @ y.T
assert np.allclose(space_cov_trusted, space_cov)

# test: H from QR factorization (space naive)
h_trusted = y @ x_multi.T @ np.linalg.inv(x_multi @ x_multi.T) @ x_multi @ y.T
h0 = y @ q_multi.T @ q_multi @ y.T
assert np.allclose(h0, h_trusted)

# test: H from QR factorization (quick compute across space)
h1 = num_vox * y_mean @ q.T @ q @ y_mean.T
assert np.allclose(h1, h0)

# test: e
e = space_cov + num_vox * y_mean @ (np.eye(num_img) - q.T @ q) @ y_mean.T
p = np.eye(num_img * num_vox) - np.linalg.pinv(x_multi) @ x_multi
e_trusted = y @ p @ y.T
assert np.allclose(e, e_trusted)

# test: control for covariates still works (WLOG assume first few rows are
# covariates while remaining rows of x are "of interest").
# to ensure orthogonality to covariates swap in q1 for all q's above
for a_prime in range(a + 1):
    q0 = q[:a_prime, :]
    q1 = q[a_prime:, :]
    np.allclose(q.T @ q, q0.T @ q0 + q1.T @ q1)
