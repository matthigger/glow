import numpy as np
from glow.analysis.prune import prune_greedy


def _make_tree_8():
    """8-leaf balanced binary tree.

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
    """Test greedy LLR pruning (pick highest, remove intersecting, repeat)."""

    def test_single_leaf(self):
        """Single significant leaf should be selected."""
        children = _make_tree_8()
        llr = _make_llr_8()
        reg_out, _ = prune_greedy([8], children, llr)
        assert reg_out == [8]

    def test_empty_sig_list(self):
        """Empty input should return empty output."""
        children = _make_tree_8()
        llr = _make_llr_8()
        reg_out, _ = prune_greedy([], children, llr)
        assert reg_out == []

    def test_picks_highest_first(self):
        """Greedy picks the region with highest LLR first."""
        children = _make_tree_8()
        llr = _make_llr_8()
        # node 12 has LLR=8, highest among [8,9,12]
        # greedy picks 12, which removes children 8 and 9
        reg_out, _ = prune_greedy([8, 9, 12], children, llr)
        assert reg_out == [12]

    def test_disjoint_regions_both_selected(self):
        """Non-overlapping regions should both be selected."""
        children = _make_tree_8()
        llr = _make_llr_8()
        # 8 (LLR=5) and 10 (LLR=0.5) are disjoint
        reg_out, _ = prune_greedy([8, 10], children, llr)
        assert 8 in reg_out and 10 in reg_out

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

    def test_info_keys(self):
        """info dict should contain sig_reg_list."""
        children = _make_tree_8()
        llr = _make_llr_8()
        _, info = prune_greedy([8, 9, 12], children, llr)
        assert 'sig_reg_list' in info
        assert set(info['sig_reg_list']) == {8, 9, 12}

    def test_ancestor_removed_after_pick(self):
        """Picking a child should block its ancestor from being selected."""
        children = _make_tree_8()
        llr = np.zeros(15)
        llr[8] = 10.0   # highest
        llr[12] = 5.0    # ancestor of 8
        llr[10] = 3.0    # disjoint
        reg_out, _ = prune_greedy([8, 10, 12], children, llr)
        assert 8 in reg_out
        assert 12 not in reg_out
        assert 10 in reg_out


