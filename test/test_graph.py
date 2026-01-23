import bisect
import warnings
from itertools import product

import pytest

from glow.experiment import ExperimentImageOnly
from glow.experiment import get_mancova
from glow.experiment.exper import NoBiasTermWarning
from glow.graph import *


def test_iter_node_sum():
    x = np.arange(4)
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5]])

    # each value is the sum of its children's values
    exp = np.array([0, 1, 2, 3, 1, 5, 6])
    assert np.allclose(node_sum(x, children), exp)


def test_get_f1_sens_spec():
    mask = np.array([0, 0, 1, 1])
    mask_idx = np.arange(4)
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5]])

    # Expected per region: leaves 0..3, then internal nodes 4..6
    f1_exp = np.array([0.0, 0.0, 2 / 3, 2 / 3, 0.0, 1.0, 2 / 3])
    sens_exp = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0])
    spec_exp = np.array([0.5, 0.5, 1.0, 1.0, 0.0, 1.0, 0.0])

    f1, sens, spec = get_f1_sens_spec(mask=mask, mask_idx=mask_idx,
                                      children=children)

    assert np.allclose(f1, f1_exp)
    assert np.allclose(sens, sens_exp)
    assert np.allclose(spec, spec_exp)


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


def test_get_parent():
    parent = get_parent(children=np.array([[0, 1],
                                           [2, 3]]), num_leaf=4)

    assert np.allclose(parent, [4, 4, 5, 5, -1, -1])


@pytest.fixture
def exp():
    return ExperimentImageOnly.from_gauss(seed=0, b=3, num_img=20, shape=(5,))


@pytest.fixture
def children():
    return np.arange(2 * 5 - 2).reshape((-1, 2), order='C')


def test_iter_size_ysum_yout(exp, children):
    b, num_img, num_vox = exp.y.shape
    for reg_idx, size, ysum, yout in iter_size_ysum_yout(exp.y, children=children):
        # build reliable compute: get index of all voxels in region
        vox = np.array(list(iter_topo(children=children,
                                      num_leaf=num_vox,
                                      node_start=reg_idx,
                                      only_leaf=True)))
        _y = exp.y[:, :, vox]

        # test basic stats
        assert size == vox.size
        assert np.allclose(ysum, _y.sum(axis=2))

        y_flat = _y.reshape((b, -1), order='F')
        yout_exp = y_flat @ y_flat.T
        assert np.allclose(yout_exp, yout)


def test_iter_stat(exp, children):
    a = 2
    b, num_img, num_vox = exp.y.shape
    for add_bias, n_perm in product(range(2), [None, 10]):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', NoBiasTermWarning)
            exp = exp.sample_x(a=a, seed=0, add_bias=add_bias)

            for reg_idx, e, h in iter_stat(exp, children=children, n_perm=n_perm):
                # build reliable compute: get index of all voxels in region
                vox = np.array(list(iter_topo(children=children,
                                              num_leaf=num_vox,
                                              node_start=reg_idx,
                                              only_leaf=True)))

                for perm_idx in range(e.shape[2]):
                    # permute (reliable via get_freed_lane())
                    _exp = exp.permute(perm_idx)
                    e_exp, h_exp, _ = get_mancova(x=_exp.x,
                                                  y=_exp.y[:, :, vox],
                                                  contrast=_exp.contrast)

                    # test mancova stats (use float32 tolerance)
                    assert np.allclose(h[:, :, perm_idx], h_exp, rtol=1e-5, atol=1e-5)
                    assert np.allclose(e[:, :, perm_idx], e_exp, rtol=1e-5, atol=1e-5)


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


def test_get_mask_cases():
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5]])
    mask_idx = np.arange(4)  # constant for most cases
    base_kwargs = dict(children=children, mask_idx=mask_idx)

    case_list = [
        # 1) Single internal region 4 -> covers leaves {0,1}
        (dict(reg_idx_list=[4]), np.array([4, 4, -1, -1])),

        # 2) Two disjoint regions 4->{0,1}, 5->{2,3}
        (dict(reg_idx_list=[4, 5]), np.array([4, 4, 5, 5])),

        # 3) Overlap: 4 and 1 overlap on leaf 1 -> smallest reg_idx wins
        (dict(reg_idx_list=[1, 4]), np.array([4, 1, -1, -1]))
    ]
    for idx, (kwargs, expected) in enumerate(case_list):
        out = get_label_map(**(kwargs | base_kwargs))
        msg = f'Failed for case {idx} kwargs={kwargs}'
        assert np.array_equal(out, expected), msg

    # Error case: overlap with check_disjoint=True
    with pytest.raises(RegIntersectError) as e:
        get_label_map(reg_idx_list=[1, 4], **base_kwargs, check_disjoint=True)
        assert str(e.value) == '1 intersects [4]'


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


SE = GRAPH_EXCLUDE


class TestSubgraph:
    # [4, 4, 5, 5, 6, 6, -1]
    parent_tree = get_parent(children=np.array([[0, 1],
                                                [2, 3],
                                                [4, 5]]), num_leaf=4)

    def test_line(self):
        # init
        parent = [1, 2, 3, -1]
        g = SCGraph(parent=parent)
        assert np.allclose(g.included, [1, 1, 1, 1])
        assert np.allclose(g.parent, parent)
        assert g.children == {0: [], 1: [0], 2: [1], 3: [2]}

        # rm node 0
        g.modify(nodes_rm=[0])
        assert np.allclose(g.included, [0, 1, 1, 1])
        assert np.allclose(g.parent, [SE, 2, 3, SE])
        assert g.children == {1: [], 2: [1], 3: [2]}

        # add node 0 back in, rm node 2
        g.modify(nodes_add=[0], nodes_rm=[2])
        assert np.allclose(g.included, [1, 1, 0, 1])
        assert np.allclose(g.parent, [1, 3, SE, SE])
        assert g.children == {0: [], 1: [0], 3: [1]}

    def test_tree(self):
        g = SCGraph(parent=self.parent_tree)
        assert np.allclose(g.included, [1, 1, 1, 1, 1, 1, 1])
        assert np.allclose(g.parent, self.parent_tree)
        assert g.children == {0: [], 1: [], 2: [], 3: [],
                              4: [0, 1], 5: [2, 3], 6: [4, 5]}

        # rm node 0
        g.modify(nodes_rm=[0])
        assert np.allclose(g.included, [0, 1, 1, 1, 1, 1, 1])
        assert np.allclose(g.parent, [SE, 4, 5, 5, 6, 6, SE])
        assert g.children == {1: [], 2: [], 3: [],
                              4: [1], 5: [2, 3], 6: [4, 5]}

        # add node 0 back in, rm node 2 and 4
        g.modify(nodes_add=[0], nodes_rm=[2, 4])
        assert np.allclose(g.included, [1, 1, 0, 1, 0, 1, 1])
        assert np.allclose(g.parent, [6, 6, SE, 5, SE, 6, SE])
        assert g.children == {0: [], 1: [], 3: [],
                              5: [3], 6: [0, 1, 5]}

    def test_iter_desc_full_tree(self):
        g = SCGraph(self.parent_tree)

        # root sees everything in DFS order
        assert list(g.iter_desc(6)) == [4, 0, 1, 5, 2, 3]
        assert list(g.iter_desc(6, incl_self=True)) == [6, 4, 0, 1, 5, 2, 3]

        # internal node sees its children
        assert list(g.iter_desc(4)) == [0, 1]
        assert list(g.iter_desc(5)) == [2, 3]

        # leaf sees nothing unless incl_self
        assert list(g.iter_desc(0)) == []
        assert list(g.iter_desc(0, incl_self=True)) == [0]

    def test_iter_ancest_full_tree(self):
        g = SCGraph(self.parent_tree)

        assert list(g.iter_ancest(0)) == [4, 6]
        assert list(g.iter_ancest(0, incl_self=True)) == [0, 4, 6]

        assert list(g.iter_ancest(5)) == [6]

        assert list(g.iter_ancest(6)) == []
        assert list(g.iter_ancest(6, incl_self=True)) == [6]

    def test_iter_desc_full_tree(self):
        g = SCGraph(self.parent_tree)

        assert list(g.iter_desc(0)) == []
        assert list(g.iter_desc(0, incl_self=True)) == [0]

        assert list(g.iter_desc(5)) == [2, 3]
        assert list(g.iter_desc(5, incl_self=True)) == [5, 2, 3]

        assert list(g.iter_desc(6)) == [4, 0, 1, 5, 2, 3]
        assert list(g.iter_desc(6, incl_self=True)) == [6, 4, 0, 1, 5, 2, 3]

    def test_iter_desc_with_removals(self):
        g = SCGraph(self.parent_tree)
        g.modify(nodes_rm=[4])

        # root no longer has child 4 directly, 0 and 1 should short-circuit
        assert list(g.iter_desc(6)) == [0, 1, 5, 2, 3]

        # 0 and 1 are now direct children of 6
        assert list(g.iter_ancest(0)) == [6]
        assert list(g.iter_ancest(1)) == [6]

    def test_iter_ancest_with_removals(self):
        g = SCGraph(self.parent_tree)
        g.modify(nodes_rm=[0, 2, 4])

        # 2 and 3 should now attach directly to 6
        assert list(g.iter_ancest(3)) == [5, 6]

        # descendants of 6 should show 4 branch intact, 2 and 3 under 6
        assert list(g.iter_desc(6)) == [1, 5, 3]
