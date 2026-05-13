import numpy as np

from glow.experiment import ExperimentImageOnly
from glow.experiment.permute import get_freed_lane
from glow.analysis.mancova import decompose


def _get_freed_lane_dense(x, contrast, perm_idx):
    """Dense reference for the column-layout Freedman-Lane construction.

    Builds ``freed_lane = (I - Q0Q0T) @ P + Q0Q0T`` where ``P`` is the
    column-permutation matrix with ``P[i, j] = 1`` iff ``i == perm[j]``,
    matching the indexing form ``(I - Q0Q0T)[:, perm] + Q0Q0T`` returned
    by ``get_freed_lane``.
    """
    assert perm_idx, 'perm_idx = 0 reserved for unpermuted data'
    q = decompose(x, contrast)
    num_img = x.shape[1]
    rng = np.random.default_rng(perm_idx)
    perm = np.argsort(rng.permutation(num_img))
    p = np.eye(num_img)[:, perm]
    q0 = q[0].T @ q[0]
    return (np.eye(num_img) - q0) @ p + q0


def test_get_freed_lane():
    """index-based freed_lane matches the dense permutation-matrix reference"""
    exp = ExperimentImageOnly.from_gauss(seed=0)
    exp = exp.sample_x(a=2, add_bias=True)

    for perm_idx in [1, 2, 42]:
        fl_new = get_freed_lane(x=exp.x, contrast=exp.contrast, perm_idx=perm_idx)
        fl_ref = _get_freed_lane_dense(x=exp.x, contrast=exp.contrast,
                                       perm_idx=perm_idx)
        assert np.allclose(fl_new, fl_ref), \
            f'freed_lane mismatch at perm_idx={perm_idx}'

    # covariate projection is unchanged (holds under intercept-only Q0,
    # which is what add_bias=True + a=2 gives here)
    q = decompose(x=exp.x, contrast=exp.contrast)
    p0 = q[0].T @ q[0]
    freed_lane = get_freed_lane(x=exp.x, contrast=exp.contrast, perm_idx=1)
    y0 = exp.y[:, :, 0]
    y0_perm = y0 @ freed_lane
    assert np.allclose(y0 @ p0, y0_perm @ p0)

    # residuals are shuffled by the expected permutation.  Under the
    # column-layout convention this is argsort(rng.permutation()), the
    # inverse of the raw permutation used in the row-layout reference.
    num_img = exp.y.shape[1]
    rng = np.random.default_rng(seed=1)
    new_idx = np.argsort(rng.permutation(num_img))
    to_resid = np.eye(num_img) - p0
    assert np.allclose((y0 @ to_resid)[:, new_idx], y0_perm @ to_resid)
