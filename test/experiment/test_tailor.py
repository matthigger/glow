from math import factorial

from hglm.experiment import Experiment
from hglm.experiment.tailor import *
from itertools import permutations

def test_tailor():
    # region 12 (covering regions 0, 1, 2, 3) is one effect
    # region 10 and 11 are identical effects, but we force region 13,
    # their union to be insignificant here to avoid their being merged
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5],
                         [6, 7],
                         [8, 9],
                         [10, 11],
                         [12, 13]])
    sig_reg_list = [5, 6, 7, 8, 9, 10, 11, 12, 14]
    exp = Experiment.from_gauss(shape=(8,), num_img=100)

    # ensure region 0, 1, 2, 3 have sufficiently different stats
    exp.y[:, :, :4] += 100

    reg_out, homo_pval_dict = tailor(sig_reg_list=sig_reg_list,
                                     children=children,
                                     exp=exp,
                                     alpha_tailor=.05,
                                     n_perm=100)

    assert np.array_equal(reg_out, [10, 11, 12])


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


