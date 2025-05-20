import numpy as np


def project(x, y, orth=False):
    p = x.T @ np.linalg.inv(x @ x.T) @ x
    if orth:
        # project into space orthogonal to given x
        n = p.shape[0]
        p = np.eye(n) - p
    return y @ p


def flatten_exp(exp):
    """ flattens x, y from an experiment, slow compute but good for validation
    """
    b, num_img, num_vox = exp.y.shape
    x = np.kron(np.ones((1, num_vox)), exp.x)
    y = exp.y.reshape((b, -1)).copy()

    assert np.allclose(x[:, :num_img], exp.x), 'kronecker issue'
    return x, y


def get_manova_naive_sq(exp):
    x, y = flatten_exp(exp)

    if (~exp.contrast).any():
        # remove nuisance features
        y = project(x[~exp.contrast, :], y, orth=True)
        x = x[exp.contrast, :]

    # compute e, error sum of squares
    error = project(x, y, orth=True)
    e = error @ error.T

    # compute h, variance explained by model (orthogonal to covariates)
    t = y @ y.T
    h = t - e

    return e, h


def factor(x, contrast):
    assert contrast.all(), 'invalid assumption: no nuisance features'
    a = (~contrast).sum(), exp.contrast.size
    q, r = np.linalg.qr(x.T, mode='complete')
    q, r = q.T, r.T
    return q[:a[0], :], q[a[0]: a[1], :], q[a[1]:, :]


def get_manova_naive_s(exp):
    x, y = flatten_exp(exp)

    # compute e, error sum of squares
    q = factor(x, exp.contrast)
    e = y @ q[2].T @ q[2] @ y.T

    # compute h, variance explained by model (orthogonal to covariates)
    h = y @ q[1].T @ q[1] @ y.T

    return e, h


def get_manova_hier(exp):
    q = factor(exp.x, exp.contrast)

    b, num_img, num_vox = exp.y.shape

    # compute e, error sum of squares
    y_mean = exp.y.mean(axis=2)
    h = y_mean @ q[1].T @ q[1] @ y_mean.T * num_vox

    e = 2

    return e, h


# def get_iter_stats(exp):
#     b, num_img, num_vox = exp.y.shape
#
#     y_mean = exp.y.mean(axis=2)
#
#     _y = exp.y.reshape((b, -1))
#     y_out = _y @ _y.T
#
#     return exp.x, y_mean, y_out, num_vox


if __name__ == '__main__':
    import hglm




    # build experiment
    exp = hglm.experiment.ExperimentImageOnly.from_gauss(b=3)
    exp = exp.sample_x(contrast=np.array([True, True]), add_bias=False)

    # test q matrices
    x, y = flatten_exp(exp)
    q_single = factor(x, exp.contrast)
    q_multi = factor(exp.x, exp.contrast)
    b, num_img, num_vox = exp.y.shape
    q_multi = tuple(np.kron(np.ones((1, num_vox)), q) / num_vox ** .5
                    for q in q_multi)

    e0, h0 = get_manova_naive_sq(exp)
    e1, h1 = get_manova_naive_s(exp)
    e2, h2 = get_manova_hier(exp)

    assert np.allclose(e0, e1)
    assert np.allclose(h0, h1)

    # assert np.allclose(h0, h2)
    # assert np.allclose(e0, e2)
