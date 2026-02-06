import time
from itertools import permutations
from math import factorial

import pytest

from glow.experiment import ExperimentImageOnly
from glow.experiment.permute import *


def _get_freed_lane_dense(x, contrast, perm_idx):
    """original dense-matrix implementation using explicit permutation matrix,
    kept as reference for testing"""
    assert perm_idx, 'perm_idx = 0 reserved for unpermuted data'
    q = decompose(x, contrast)
    num_img = x.shape[1]
    rng = np.random.default_rng(perm_idx)
    p = rng.permutation(np.eye(num_img)).T
    q0 = q[0].T @ q[0]
    return p @ (np.eye(num_img) - q0) + q0


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

    # covariate projection is unchanged
    q = decompose(x=exp.x, contrast=exp.contrast)
    p0 = q[0].T @ q[0]
    freed_lane = get_freed_lane(x=exp.x, contrast=exp.contrast, perm_idx=1)
    y0 = exp.y[:, :, 0]
    y0_perm = y0 @ freed_lane
    assert np.allclose(y0 @ p0, y0_perm @ p0)

    # residuals are shuffled by the expected permutation
    num_img = exp.y.shape[1]
    rng = np.random.default_rng(seed=1)
    new_idx = rng.permutation(num_img)
    to_resid = np.eye(num_img) - p0
    assert np.allclose((y0 @ to_resid)[:, new_idx], y0_perm @ to_resid)


def test_get_freed_lane_speed():
    """index-based construction avoids dense permutation matrix multiply"""
    perm_idx = 1
    n_rep = 200

    print()
    for num_img in (200, 1000):
        exp = ExperimentImageOnly.from_gauss(seed=0, b=3, num_img=num_img,
                                             shape=(100,))
        exp = exp.sample_x(a=2, add_bias=True)

        # warm up
        get_freed_lane(exp.x, exp.contrast, perm_idx)
        _get_freed_lane_dense(exp.x, exp.contrast, perm_idx)

        t0 = time.perf_counter()
        for _ in range(n_rep):
            fl_new = get_freed_lane(exp.x, exp.contrast, perm_idx)
        t_new = (time.perf_counter() - t0) / n_rep

        t0 = time.perf_counter()
        for _ in range(n_rep):
            fl_old = _get_freed_lane_dense(exp.x, exp.contrast, perm_idx)
        t_old = (time.perf_counter() - t0) / n_rep

        speedup = t_old / t_new
        print(f'  num_img={num_img:4d}: old={t_old*1000:.2f}ms  '
              f'new={t_new*1000:.2f}ms  speedup={speedup:.1f}x')
        assert np.allclose(fl_new, fl_old)

    # index-based row selection should be faster than P @ M matmul
    assert speedup > 2.0, f'expected speedup at num_img=1000, got {speedup:.2f}x'


def get_n_perm_possible_slow(partition):
    n = len(partition)
    counts = Counter(partition).values()
    return factorial(n) / np.prod([factorial(c) for c in counts])


case_list = ([0],
             [0, 1],
             [0, 0, 0, 1, 1, 1, 1],
             [0, 1, 2, 3, 3, 3],)


def test_perms_at_least():
    for partition in case_list:
        # given large threshold, its never enough (must compute all)
        enough, n_perm_obs = perms_at_least(partition, thresh=np.inf)
        assert not enough
        assert n_perm_obs == get_n_perm_possible_slow(partition)


def test_iter_perms():
    for partition in case_list:
        exp = set(permutations(partition))
        obs = set([tuple(p) for p in get_perm_iter_all(partition)])
        assert exp == obs


def test_get_perm_iter():
    partition = (0, 0, 1, 0)

    # if there are sufficient permutations, draw samples (repeats allowed)
    part_list = [tuple(p) for p in get_perm_iter(partition, n_perm=2)]
    assert part_list[0] == partition
    assert len(part_list) == 3

    # if there aren't sufficient permutations, go through the list exhaustively
    with pytest.warns(NotEnoughPermutations):
        part_list = [tuple(p) for p in get_perm_iter(partition, n_perm=1e6)]
        assert part_list[0] == partition
        assert len(set(part_list)) == len(part_list)


def test_perms_at_least_trivial_threshold():
    """test perms_at_least with thresh <= 1 (early return case)"""
    partition = [0, 0, 1, 1, 2, 2]
    
    # with thresh <= 1, should return True immediately with None
    enough, n_perm = perms_at_least(partition, thresh=1)
    assert enough is True
    assert n_perm is None
    
    # also test with thresh < 1
    enough, n_perm = perms_at_least(partition, thresh=0.5)
    assert enough is True
    assert n_perm is None
    
    # verify this is different from thresh > 1 behavior
    enough_high, n_perm_high = perms_at_least(partition, thresh=100)
    assert enough_high is False
    assert n_perm_high is not None
    assert n_perm_high > 1
