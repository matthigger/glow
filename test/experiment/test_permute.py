from itertools import permutations
from math import factorial

import pytest

from glow.experiment import ExperimentImageOnly
from glow.experiment.permute import *


def test_get_freed_lane():
    exp = ExperimentImageOnly.from_gauss(seed=0)
    exp = exp.sample_x(a=2, add_bias=True)

    # projections preserve the q0 directions but change data otherwise
    perm_idx = 1
    freed_lane = get_freed_lane(x=exp.x, contrast=exp.contrast, perm_idx=perm_idx)

    y0 = exp.y[:, :, 0]
    y0_perm = y0 @ freed_lane

    # zero direction (covariates) is unchanged by freed_lane
    q = decompose(x=exp.x, contrast=exp.contrast)
    p0 = q[0].T @ q[0]
    assert np.allclose(y0 @ p0, y0_perm @ p0)

    # residuals are just shuffled
    num_img = exp.y.shape[1]
    rng = np.random.default_rng(seed=perm_idx)
    new_idx = rng.permutation(num_img)
    to_resid = np.eye(num_img) - p0
    resid_before = y0 @ to_resid
    resid_after = y0_perm @ to_resid
    assert np.allclose(resid_before[:, new_idx], resid_after)


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
