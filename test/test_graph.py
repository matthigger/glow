from hrba.graph import *


def test_iter_node_sum():
    val_dict = dict(enumerate(range(4)))
    dend = np.array([[0, 2, 4],
                     [1, 3, 5]])

    # each value is the sum of its children's values
    val_dict_exp = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 5, 6: 6}
    assert node_sum(dend, val_dict) == val_dict_exp


def test_get_hit_miss():
    mask = np.array([0, 0, 1, 1])
    mask_idx = np.arange(4)
    dendro = np.array([[0, 2, 4],
                       [1, 3, 5]])
    miss_hit_dict_exp = {0: np.array([1, 0]),
                         1: np.array([1, 0]),
                         2: np.array([0, 1]),
                         3: np.array([0, 1]),
                         4: np.array([2, 0]),
                         5: np.array([0, 2]),
                         6: np.array([2, 2])}
    miss_hit_dict = get_hit_miss(mask=mask, mask_idx=mask_idx, dendro=dendro)

    assert miss_hit_dict.keys() == miss_hit_dict_exp.keys()
    for node in miss_hit_dict.keys():
        assert np.allclose(miss_hit_dict[node],
                           miss_hit_dict_exp[node]), f'failure node: {node}'
