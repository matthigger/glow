import numpy as np

from hrba.graph import *


def test_iter_node_sum():
    val_dict = dict(enumerate(range(4)))
    dend = np.array([[0, 2, 4],
                     [1, 3, 5]])

    # each value is the sum of its children's values
    val_dict_exp = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 5, 6: 6}
    assert node_sum(dend, val_dict) == val_dict_exp
