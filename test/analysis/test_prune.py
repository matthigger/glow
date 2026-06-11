import numpy as np
from glow.analysis.prune import prune_greedy, prune_dp, dp_antichain


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


class TestPruneDp:
    """Test DP antichain pruning."""

    def test_empty_sig_list(self):
        children = _make_tree_8()
        llr = _make_llr_8()
        reg_out, _ = prune_dp([], children, llr)
        assert reg_out == []

    def test_output_is_antichain(self):
        children = _make_tree_8()
        llr = _make_llr_8()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        reg_out, _ = prune_dp(sig_all, children, llr, lam=0.0)

        from glow.graph import get_parent
        parent = get_parent(children, num_leaf=8)
        selected = set(reg_out)
        for node in selected:
            p = parent[node]
            while p != -1:
                assert p not in selected, \
                    f'node {node} and ancestor {p} both selected'
                p = parent[p]

    def test_dp_at_least_as_good_as_greedy(self):
        """DP total stat >= greedy total stat (globally optimal)."""
        children = _make_tree_8()
        llr = _make_llr_8()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        greedy_out, _ = prune_greedy(sig_all, children, llr)
        dp_out, _ = prune_dp(sig_all, children, llr, lam=0.0)
        assert sum(llr[r] for r in dp_out) >= sum(llr[r] for r in greedy_out)

    def test_penalty_reduces_selection(self):
        """Higher lam should select fewer or equal regions."""
        children = _make_tree_8()
        llr = _make_llr_8()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        out_0, _ = prune_dp(sig_all, children, llr, lam=0.0)
        out_big, _ = prune_dp(sig_all, children, llr, lam=100.0)
        assert len(out_big) <= len(out_0)

    def test_exp_n_eff_overrides_lam(self):
        """exp_n_eff=3 should use lam=log(1+1/3)."""
        children = _make_tree_8()
        llr = _make_llr_8()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        out_eff, _ = prune_dp(sig_all, children, llr, exp_n_eff=3.0)
        out_lam, _ = prune_dp(sig_all, children, llr,
                              lam=np.log(1 + 1.0 / 3.0))
        assert out_eff == out_lam

    def test_splits_when_children_beat_parent(self):
        """When children sum > parent, DP should split."""
        children = _make_tree_8()
        llr = np.zeros(15)
        llr[8], llr[9] = 5.0, 4.0  # sum = 9
        llr[12] = 8.0               # parent < sum
        reg_out, _ = prune_dp([8, 9, 12], children, llr, lam=0.0)
        assert 8 in reg_out and 9 in reg_out
        assert 12 not in reg_out


class TestDpAntichain:
    """Test the bottom-up antichain DP that prune_dp delegates to."""

    def test_basic(self):
        """lam=0 selects the gain-maximising antichain."""
        #     6
        #    / \
        #   4   5
        children_map = {6: [4, 5], 4: [], 5: []}
        gain = {4: 5.0, 5: 3.0, 6: 7.0}
        # split(6) = best(4)+best(5) = 5+3 = 8 > 7, so split
        selected, _ = dp_antichain(
            nodes=[4, 5, 6], children_map=children_map, gain=gain, lam=0.0)
        assert selected == [4, 5]

        # with higher gain on root, should select root
        gain[6] = 10.0
        selected, _ = dp_antichain(
            nodes=[4, 5, 6], children_map=children_map, gain=gain, lam=0.0)
        assert selected == [6]

    def test_with_penalty(self):
        """Positive lam should suppress small-gain nodes."""
        children_map = {6: [4, 5], 4: [], 5: []}
        gain = {4: 5.0, 5: 3.0, 6: 7.0}
        # lam=4: net gains are 1, -1, 3
        # best(4) = max(1, 0) = 1; best(5) = max(-1, 0) = 0
        # split(6) = 1; chose 6 since 3 >= 1
        selected, _ = dp_antichain(
            nodes=[4, 5, 6], children_map=children_map, gain=gain, lam=4.0)
        assert selected == [6]

    def test_empty(self):
        """Empty node list returns empty selection."""
        selected, _ = dp_antichain(nodes=[], children_map={}, gain={}, lam=0.0)
        assert selected == []


