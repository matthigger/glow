import bisect
from itertools import product

from hglm.graph import *


def test_iter_node_sum():
    x = np.arange(4)
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5]])

    # each value is the sum of its children's values
    exp = np.array([0, 1, 2, 3, 1, 5, 6])
    assert np.allclose(node_sum(x, children), exp)


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

    miss_exp = np.array([1, 1, 0, 0, 2, 0, 2])
    hit_exp = np.array([0, 0, 1, 1, 0, 2, 2])

    miss, hit = get_miss_hits(mask=mask, mask_idx=mask_idx, children=children)
    assert np.allclose(miss, miss_exp)
    assert np.allclose(hit, hit_exp)

    # test incomplete tree
    children = children[:-1, :]
    miss_exp = miss_exp[:-1]
    hit_exp = hit_exp[:-1]

    miss, hit = get_miss_hits(mask=mask, mask_idx=mask_idx, children=children)
    assert np.allclose(miss, miss_exp)
    assert np.allclose(hit, hit_exp)


def test_topo_iter():
    # complete tree
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5]])

    assert list(iter_topo(children=children,
                          num_leaf=4,
                          node_start=4)) == [0, 1, 4]
    assert list(iter_topo(children=children,
                          num_leaf=4,
                          node_start=5)) == [2, 3, 5]
    assert list(iter_topo(children=children,
                          num_leaf=4)) == [0, 1, 4, 2, 3, 5, 6]
    assert list(iter_topo(children=children,
                          num_leaf=4,
                          only_leaf=True)) == [0, 1, 2,
                                               3]

    # incomplete tree
    children = np.array([[0, 1],
                         [2, 3]])

    assert list(iter_topo(children=children,
                          num_leaf=4,
                          node_start=4)) == [0, 1, 4]
    assert list(iter_topo(children=children,
                          num_leaf=4,
                          node_start=5)) == [2, 3, 5]
    assert list(iter_topo(children=children,
                          num_leaf=4,
                          node_start=5,
                          only_leaf=True)) == [2, 3]

    # no graph passed (iterate through leafs one by one)
    assert list(iter_topo(num_leaf=4)) == [0, 1, 2, 3]


def test_iter_size_e_h():
    rng = np.random.default_rng(seed=0)
    a, b, num_img, num_vox = 2, 3, 4, 5
    y = rng.standard_normal((b, num_img, num_vox))
    x = rng.standard_normal((a, num_img))
    children = np.arange(2 * num_vox - 2).reshape((-1, 2), order='C')

    for contrast in product([True, False], repeat=a):
        contrast = np.array(contrast)
        if not contrast.any():
            # ensure there is at least 1 feature of interest
            continue

        for reg_idx, size, e, h in iter_size_e_h(y=y, children=children,
                                                 x=x, contrast=contrast):
            # build reliable compute: get index of all voxels in region
            vox = np.array(list(iter_topo(children=children,
                                          num_leaf=num_vox,
                                          node_start=reg_idx,
                                          only_leaf=True)))

            # slow and steady compute of e and h
            yr = np.concatenate([y[:, :, _vox] for _vox in vox], axis=1)
            xr = np.concatenate([x for _vox in vox], axis=1)

            if not contrast.all():
                # project yr into nullspace of covariates
                _xr = xr[~contrast, :]

                p = np.eye(yr.shape[1]) - np.linalg.pinv(_xr) @ _xr
                yr = yr @ p
                xr = xr[contrast, :] @ p

            hat = yr @ np.linalg.pinv(xr) @ xr
            err = yr - hat

            h_exp = hat @ hat.T
            e_exp = err @ err.T

            assert np.allclose(h, h_exp[:, :, np.newaxis])
            assert np.allclose(e, e_exp[:, :, np.newaxis])


def test_iter_size_yout_ymean():
    rng = np.random.default_rng(seed=0)
    a, b, num_img, num_vox = 2, 3, 4, 5
    y = rng.standard_normal((b, num_img, num_vox))
    x = rng.standard_normal((a, num_img))
    children = np.arange(2 * num_vox - 2).reshape((-1, 2), order='C')

    for reg_idx, size, yout, ymean in iter_size_yout_ymean(y, children):
        # build reliable compute: get index of all voxels in region
        vox = np.array(list(iter_topo(children=children,
                                      num_leaf=num_vox,
                                      node_start=reg_idx,
                                      only_leaf=True)))
        _y = y[:, :, vox]

        assert vox.size == size
        assert np.allclose(_y.mean(axis=2), ymean[:, :, 0])

        _y = _y.reshape((b, -1), order='F')
        yout_exp = _y @ _y.T
        assert np.allclose(yout_exp, yout[:, :, 0])


def binary_tree(n_node=100, seed=0, merge_smallest=True):
    """ samples a binary tree via random agglomeration

    Args:
        n_node (int): number of nodes
        seed: random number seed
        merge_smallest (bool): if True, the tree will choose to merge the
            nodes representing the fewest leafs

    Returns:
        children (np.array): (n_node - 1, 2) row i contains the index of
            children of node i + n_node
    """

    rng = np.random.default_rng(seed)
    list_size_node = [(1, idx) for idx in range(n_node)]

    children = list()
    for node_idx in range(n_node, 2 * n_node - 1):
        # get max idx to merge
        if merge_smallest:
            # we get size of 2nd smallest node (if smallest has unique size,
            # we'd need to include this second to ensure there's something
            # to merge it with)
            _size = list_size_node[1][0]
            idx_max = bisect.bisect(list_size_node, (_size, np.inf))
        else:
            idx_max = len(list_size_node)

        # select two nodes from available nodes
        idx0, idx1 = rng.choice(range(idx_max), replace=False, size=2)

        # get size as node index of each chosen index (sort to ensure we pop
        # the later one first, to preserve the earlier index)
        idx0, idx1 = sorted((idx0, idx1))
        size1, node1 = list_size_node.pop(idx1)
        size0, node0 = list_size_node.pop(idx0)

        children.append(sorted((node0, node1)))
        size_node = size0 + size1, node_idx

        # insert new node to maintain sorted order (smallest to largest)
        idx = bisect.bisect(list_size_node, size_node)
        list_size_node.insert(idx, size_node)

    # check that everything merged into one node
    assert len(list_size_node) == 1
    assert list_size_node[0][0] == n_node

    return np.array(children)


def test_graph_merge():
    n_node_per_graph = 20
    n_graph = 11

    seed = 0
    for _ in range(1):
        for merge_smallest in (True, False):
            # sample some binary graphs
            children_list = list()
            for _seed in range(seed, seed + n_graph):
                children_list.append(binary_tree(n_node=n_node_per_graph,
                                                 seed=_seed,
                                                 merge_smallest=merge_smallest))
            # ensure the next batch is fresh
            seed += n_graph

            # merge them into one graph
            map_to_new, children, size = graph_merge(n_common=n_node_per_graph,
                                                     children_list=children_list)

            # ensure each _map_to_new is unique (observed in early version,
            # double checking)
            for _map_to_new in map_to_new:
                assert np.unique(_map_to_new).size == _map_to_new.size

            n_node_twin = np.zeros(children.shape[0], dtype=int)
            for idx in range(children.shape[0]):
                node = idx + n_node_per_graph

                # get list of represented leafs (from big graph)
                set_leaf = set(iter_topo(children=children,
                                         num_leaf=n_node_per_graph,
                                         node_start=node,
                                         only_leaf=True))

                for _children, _map_to_new in zip(children_list, map_to_new):
                    matches = np.where(_map_to_new == node)[0]
                    if not len(matches):
                        # subgraph doesn't contain this particular common
                        # node, nothing to check
                        continue
                    idx_subgraph = matches[0]
                    node_subgraph = idx_subgraph + n_node_per_graph

                    # get list of represented leafs (from subgraph)
                    set_leaf_subgraph = set(iter_topo(children=_children,
                                                      num_leaf=n_node_per_graph,
                                                      node_start=node_subgraph,
                                                      only_leaf=True))
                    if set_leaf != set_leaf_subgraph:
                        print('hi')
                        map_to_new, children, size = graph_merge(
                            n_common=n_node_per_graph,
                            children_list=children_list)

                    assert set_leaf == set_leaf_subgraph

                    # count
                    _map_to_new[idx_subgraph] = -1
                    n_node_twin[idx] += 1

            assert n_node_twin.min() >= 1, 'common node not in any subgraph'
            assert all((_x == -1).all() for _x in map_to_new), \
                'subgraph node not represented'
