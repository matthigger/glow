from collections import Counter
from itertools import permutations
from math import factorial

from hglm.experiment import Experiment
from hglm.experiment.permute import *
from hglm.experiment.permute import get_n_perm_possible, get_perm_iter_all


class TestPermuter:
    def test_get_perm_matrix(self):
        perm = get_perm_matrix(num_img=5, seed=0)
        np.testing.assert_array_almost_equal(perm, np.eye(5))

        perm = get_perm_matrix(num_img=5, seed=1)
        perm_expect = np.array([[0., 0., 0., 0., 1.],
                                [1., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0.],
                                [0., 0., 1., 0., 0.],
                                [0., 0., 0., 1., 0.]])

        np.testing.assert_array_almost_equal(perm, perm_expect)

    def test_call(self):
        perm_idx = 1
        shape = 10, 10
        num_img = 5
        exp = Experiment.from_gauss(shape=shape, seed=0, num_img=num_img)

        # prep
        perm = Permuter(x=exp.x[~exp.contrast, :])


def get_n_perm_possible_slow(partition):
    n = len(partition)
    counts = Counter(partition).values()
    return factorial(n) / np.prod([factorial(c) for c in counts])


case_list = ([0],
             [0, 1],
             [0, 0, 0, 1, 1, 1, 1],
             [0, 1, 2, 3, 3, 3])


def test_get_n_perm_possible():
    for partition in case_list:
        exp = get_n_perm_possible_slow(partition)
        obs = get_n_perm_possible(partition)
        assert exp == obs


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
    part_list = [tuple(p) for p in get_perm_iter(partition, n_perm=1e6)]
    assert part_list[0] == partition
    assert len(set(part_list)) == len(part_list)
