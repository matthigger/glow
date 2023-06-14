from hrba.f_stat import get_f_stat
from hrba.graph import *
from .experiment.test_exper import get_rand_exp


def test_iter_node_sum():
    val_dict = dict(enumerate(range(4)))
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5]])

    # each value is the sum of its children's values
    val_dict_exp = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 5, 6: 6}
    assert node_sum(children, val_dict) == val_dict_exp


def test_get_f1():
    mask = np.array([0, 0, 1, 1])
    mask_idx = np.arange(4)
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5]])

    f1_exp = np.array([0, 0, 2 / 3, 2 / 3, 0, 1, 2 / 3])
    f1 = get_f1(mask=mask, mask_idx=mask_idx, children=children)

    assert np.allclose(f1, f1_exp)


def test_get_miss_hit():
    mask = np.array([0, 0, 1, 1])
    mask_idx = np.arange(4)
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5]])

    miss_hit_dict_exp = {0: np.array([1, 0]),
                         1: np.array([1, 0]),
                         2: np.array([0, 1]),
                         3: np.array([0, 1]),
                         4: np.array([2, 0]),
                         5: np.array([0, 2]),
                         6: np.array([2, 2])}
    miss_hit_dict = get_miss_hits(mask=mask, mask_idx=mask_idx,
                                  children=children)

    assert miss_hit_dict.keys() == miss_hit_dict_exp.keys()
    for node in miss_hit_dict.keys():
        assert np.allclose(miss_hit_dict[node],
                           miss_hit_dict_exp[node]), f'failure node: {node}'


def test_topo_iter():
    # complete tree
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5]])

    assert list(iter_topo(children, node_start=4)) == [0, 1, 4]
    assert list(iter_topo(children, node_start=5)) == [2, 3, 5]
    assert list(iter_topo(children)) == [0, 1, 4, 2, 3, 5, 6]
    assert list(iter_topo(children, only_leaf=True)) == [0, 1, 2, 3]

    # incomplete tree
    children = np.array([[0, 1],
                         [2, 3]])

    assert list(iter_topo(children, num_leaf=4, node_start=4)) == [0, 1, 4]
    assert list(iter_topo(children, num_leaf=4, node_start=5)) == [2, 3, 5]
    assert list(iter_topo(children, num_leaf=4, node_start=5,
                          only_leaf=True)) == [2, 3]


def test_iter_reg_stat():
    exp = get_rand_exp(shape=(10, 10), b=1)
    num_vox = exp.y.shape[2]
    children = np.arange((num_vox - 1) * 2).reshape((num_vox - 1), 2)

    for reg_idx, size, f_stat in iter_reg_stat_exp(children, exp):
        # compute size & f_stat (slowly)
        voxel_tup = tuple(v for v in iter_topo(children, node_start=reg_idx)
                          if v < num_vox)

        assert len(voxel_tup) == size

        y = exp.y[..., voxel_tup]
        f_stat_obs = get_f_stat(x=exp.x, y=y, contrast=exp.contrast)
        assert np.isclose(f_stat, f_stat_obs)


def test_children_to_parent():
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5]])
    parent_exp = np.array([4, 4, 5, 5, 6, 6, np.nan])
    parent = child_to_parent(children)
    assert np.allclose(parent_exp, parent, equal_nan=True)

    parent = child_to_parent(children, num_leaf=4)
    assert np.allclose(parent_exp, parent, equal_nan=True)

    parent = child_to_parent(children, num_leaf=4)
    assert np.allclose(parent_exp, parent, equal_nan=True)

    children = np.array([[0, 1],
                         [2, 3]])
    parent_exp = np.array([4, 4, 5, 5, np.nan, np.nan])
    parent = child_to_parent(children, num_leaf=4)
    assert np.allclose(parent_exp, parent, equal_nan=True)

    parent = child_to_parent(children, num_leaf=4)
    assert np.allclose(parent_exp, parent, equal_nan=True)


def test_iter_ancestor():
    parent = np.array([4, 4, 5, 5, 6, 6, np.nan])

    assert list(iter_ancestor(parent, node=0)) == [0, 4, 6]
    assert list(iter_ancestor(parent, node=0, include_self=False)) == [4, 6]
    assert list(iter_ancestor(parent, node=1)) == [1, 4, 6]
    assert list(iter_ancestor(parent, node=4)) == [4, 6]
    assert list(iter_ancestor(parent, node=3)) == [3, 5, 6]
