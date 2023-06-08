from hrba.graph import *


def test_iter_node_sum():
    val_dict = dict(enumerate(range(4)))
    dendro = np.array([[0, 1],
                       [2, 3],
                       [4, 5]])

    # each value is the sum of its children's values
    val_dict_exp = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 5, 6: 6}
    assert node_sum(dendro, val_dict) == val_dict_exp


def test_get_hit_miss():
    mask = np.array([0, 0, 1, 1])
    mask_idx = np.arange(4)
    dendro = np.array([[0, 1],
                       [2, 3],
                       [4, 5]])

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


def test_topo_iter():
    # complete tree
    dendro = np.array([[0, 1],
                       [2, 3],
                       [4, 5]])

    assert list(iter_topo(dendro, node_start=4)) == [0, 1, 4]
    assert list(iter_topo(dendro, node_start=5)) == [2, 3, 5]
    assert list(iter_topo(dendro)) == [0, 1, 4, 2, 3, 5, 6]

    # incomplete tree
    dendro = np.array([[0, 1],
                       [2, 3]])

    assert list(iter_topo(dendro, num_leaf=4, node_start=4)) == [0, 1, 4]
    assert list(iter_topo(dendro, num_leaf=4, node_start=5)) == [2, 3, 5]
