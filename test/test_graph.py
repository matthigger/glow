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


def test_get_f1_sens_spec_with_inactive_voxels():
    """Specificity must ignore spatial positions outside the analysis mask."""
    # 2x4 spatial grid, only 4 of 8 positions are analysis voxels
    mask_idx = np.array([[-1, 0, 1, -1],
                         [-1, 2, 3, -1]])
    mask = np.array([[False, False, False, False],
                     [False, True,  True,  False]])
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5]])

    # 4 analysis voxels, 2 in target (vox 2, 3)
    # Region 6 (root): all 4 voxels → tp=2, fp=2, fn=0, tn=0
    #   spec = 0/(0+2) = 0, NOT ~0.75 which you'd get using mask.size=8
    f1, sens, spec = get_f1_sens_spec(mask=mask, mask_idx=mask_idx,
                                      children=children)
    root = 6
    assert spec[root] == 0.0
    assert sens[root] == 1.0

    # Region 5 (vox 2,3 = the target): tp=2, fp=0, fn=0, tn=2 → spec=1
    assert spec[5] == 1.0

    # Region 4 (vox 0,1 = no target): tp=0, fp=2, fn=2, tn=0 → spec=0
    assert spec[4] == 0.0


def test_dp_antichain_basic():
    """dp_antichain with lam=0 selects the gain-maximising antichain."""
    from glow.graph import dp_antichain

    #     6
    #    / \
    #   4   5
    #  /\  /\
    # 0  1 2  3
    children_map = {6: [4, 5], 4: [], 5: []}
    gain = {4: 5.0, 5: 3.0, 6: 7.0}
    # split(6) = best(4)+best(5) = 5+3 = 8 > 7, so split
    selected, info = dp_antichain(
        nodes=[4, 5, 6], children_map=children_map, gain=gain, lam=0.0)
    assert selected == [4, 5]

    # with higher gain on root, should select root
    gain[6] = 10.0
    selected, _ = dp_antichain(
        nodes=[4, 5, 6], children_map=children_map, gain=gain, lam=0.0)
    assert selected == [6]


def test_dp_antichain_with_penalty():
    """Positive lam should suppress small-gain nodes."""
    from glow.graph import dp_antichain

    children_map = {6: [4, 5], 4: [], 5: []}
    gain = {4: 5.0, 5: 3.0, 6: 7.0}
    # lam=4: net gains are 1, -1, 3
    # best(4) = max(1, 0) = 1; best(5) = max(-1, 0) = 0
    # split(6) = 1; chose 6 since 3 >= 1
    selected, _ = dp_antichain(
        nodes=[4, 5, 6], children_map=children_map, gain=gain, lam=4.0)
    assert selected == [6]


def test_dp_antichain_empty():
    """Empty node list returns empty selection."""
    from glow.graph import dp_antichain
    selected, _ = dp_antichain(nodes=[], children_map={}, gain={}, lam=0.0)
    assert selected == []


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

            for reg_idx, size, e, h in iter_stat(exp, children=children, n_perm=n_perm):
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
