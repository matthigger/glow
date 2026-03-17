import numpy as np
from glow.experiment.prune import prune_greedy


def _make_tree_8():
    """8-leaf balanced binary tree for DP tests.

    Tree:          14
                 /    \\
               12      13
              / \\    / \\
             8   9  10  11
            /\\ /\\ /\\ /\\
           0 1 2 3 4 5 6 7
    """
    return np.array([[0, 1], [2, 3], [4, 5], [6, 7],
                     [8, 9], [10, 11], [12, 13]])


def _make_llr_8():
    """Raw LLR for the 8-leaf tree with effect in left subtree."""
    llr = np.zeros(15)
    llr[8], llr[9] = 5.0, 4.0
    llr[10], llr[11] = 0.5, 0.3
    llr[12] = 8.0
    llr[13] = 0.7
    llr[14] = 6.0
    return llr


class TestPruneGreedy:
    """Test greedy LLR-maximising antichain selection."""

    def test_single_leaf(self):
        """Single significant leaf should be selected."""
        children = _make_tree_8()
        llr = _make_llr_8()
        reg_out, info = prune_greedy([8], children, llr)
        assert 8 in reg_out

    def test_empty_sig_list(self):
        """Empty input should return empty output."""
        children = _make_tree_8()
        llr = _make_llr_8()
        reg_out, info = prune_greedy([], children, llr)
        assert reg_out == []

    def test_prefers_split_when_children_stronger(self):
        """When children sum > parent, the DP should split."""
        children = _make_tree_8()
        llr = _make_llr_8()
        # nodes 8+9 = 5+4 = 9 > node 12 = 8
        reg_out, _ = prune_greedy([8, 9, 12], children, llr)
        assert 8 in reg_out and 9 in reg_out
        assert 12 not in reg_out

    def test_prefers_parent_when_parent_stronger(self):
        """When parent > sum of children, keep parent."""
        children = _make_tree_8()
        llr = np.zeros(15)
        llr[8], llr[9] = 3.0, 3.0
        llr[12] = 10.0
        reg_out, _ = prune_greedy([8, 9, 12], children, llr)
        assert reg_out == [12]

    def test_output_is_antichain(self):
        """Selected regions must be disjoint (no ancestor-descendant)."""
        children = _make_tree_8()
        llr = _make_llr_8()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        reg_out, _ = prune_greedy(sig_all, children, llr)

        from glow.graph import get_parent
        parent = get_parent(children, num_leaf=8)
        selected = set(reg_out)
        for node in selected:
            p = parent[node]
            while p != -1:
                assert p not in selected, \
                    f'node {node} and its ancestor {p} both selected'
                p = parent[p]

    def test_negative_llr_not_selected(self):
        """Regions with negative LLR should not appear."""
        children = _make_tree_8()
        llr = np.zeros(15)
        llr[8] = -1.0
        reg_out, _ = prune_greedy([8], children, llr)
        assert reg_out == []

    def test_dp_info_keys(self):
        """dp_info should contain expected keys."""
        children = _make_tree_8()
        llr = _make_llr_8()
        _, info = prune_greedy([8, 9, 12], children, llr)
        assert 'sig_reg_list' in info
        assert 'subgraph_children' in info
        assert 'best' in info
        assert 'chose' in info
