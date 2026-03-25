import numpy as np
from glow.experiment.prune import prune_greedy, prune_dp, prune_greedy_full_adjust


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


class TestPruneGreedyFullAdjust:
    """Test tree-wide adjusted-LL greedy pruning."""

    @staticmethod
    def _make_exp(num_vox=8, num_img=20, b=2, seed=42):
        """Build a minimal Experiment for testing."""
        from glow.experiment.exper import Experiment
        rng = np.random.RandomState(seed)
        y = rng.randn(b, num_img, num_vox)
        x = np.vstack([np.ones(num_img), rng.randn(num_img)])
        contrast = np.array([False, True])
        mask_idx = np.arange(num_vox)
        return Experiment(y=y, x=x, contrast=contrast, mask_idx=mask_idx)

    def test_empty_sig_list(self):
        exp = self._make_exp()
        children = _make_tree_8()
        reg_out, info = prune_greedy_full_adjust([], children, exp)
        assert reg_out == []
        assert 'sig_reg_list' in info

    def test_output_is_antichain(self):
        exp = self._make_exp()
        children = _make_tree_8()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        reg_out, _ = prune_greedy_full_adjust(sig_all, children, exp)

        from glow.graph import get_parent
        parent = get_parent(children, num_leaf=8)
        selected = set(reg_out)
        for node in selected:
            p = parent[node]
            while p != -1:
                assert p not in selected, \
                    f'node {node} and ancestor {p} both selected'
                p = parent[p]

    def test_cost_history_monotonic(self):
        """Each greedy step should increase (or maintain) the cost."""
        exp = self._make_exp()
        children = _make_tree_8()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        _, info = prune_greedy_full_adjust(sig_all, children, exp)
        history = info['cost_history']
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1]

    def test_single_significant_region(self):
        """Single significant region should be selected."""
        exp = self._make_exp()
        children = _make_tree_8()
        reg_out, _ = prune_greedy_full_adjust([12], children, exp)
        assert reg_out == [12]
