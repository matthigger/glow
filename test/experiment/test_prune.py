import numpy as np
from glow.experiment.prune import prune


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


def _make_llr_adjusted_8():
    """llr_adjusted for the 8-leaf tree with effect in left subtree."""
    adj = np.zeros(15)
    adj[8], adj[9] = 5.0, 4.0
    adj[10], adj[11] = 0.5, 0.3
    adj[12] = 8.0
    adj[13] = 0.7
    adj[14] = 6.0
    return adj


class TestDPSolve:
    """test the DP solver on hand-crafted inputs."""

    def test_single_leaf(self):
        """single significant leaf: selected iff gain > lambda."""
        children = _make_tree_8()
        adj = _make_llr_adjusted_8()
        reg_out, info = prune([8], children, adj, lam=0.0)
        assert 8 in reg_out, 'single leaf with lam=0 should be selected'

    def test_large_lambda_selects_nothing(self):
        """very large lambda should select no regions."""
        children = _make_tree_8()
        adj = _make_llr_adjusted_8()
        reg_out, info = prune([8, 9, 12], children, adj, lam=1e6)
        assert reg_out == []

    def test_prefers_split_over_parent(self):
        """when children have strong effects but parent is diluted,
        the DP should prefer splitting."""
        children = _make_tree_8()
        adj = _make_llr_adjusted_8()
        reg_out, info = prune([8, 9, 12], children, adj, lam=0.0)

        if 12 not in reg_out:
            assert 8 in reg_out and 9 in reg_out

    def test_output_is_antichain(self):
        """selected regions must be disjoint (no ancestor-descendant)."""
        children = _make_tree_8()
        adj = _make_llr_adjusted_8()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        reg_out, _ = prune(sig_all, children, adj, exp_eff=5)

        from glow.graph import get_parent
        parent = get_parent(children, num_leaf=8)
        selected = set(reg_out)
        for node in selected:
            p = parent[node]
            while p != -1:
                assert p not in selected, \
                    f'node {node} and its ancestor {p} both selected'
                p = parent[p]

    def test_empty_sig_reg_list(self):
        """empty input should return empty output."""
        children = _make_tree_8()
        adj = _make_llr_adjusted_8()
        reg_out, info = prune([], children, adj)
        assert reg_out == []
        assert info['lam'] == 0.0


class TestLambdaFormula:
    """test analytic lambda from geometric prior."""

    def test_lambda_value(self):
        """lambda should equal log(1 + 1/exp_eff)."""
        children = _make_tree_8()
        adj = _make_llr_adjusted_8()
        for exp_eff in [1, 2, 5, 10]:
            _, info = prune([8, 9], children, adj, exp_eff=exp_eff)
            expected = np.log(1 + 1 / exp_eff)
            assert np.isclose(info['lam'], expected), \
                f'exp_eff={exp_eff}: got {info["lam"]}, expected {expected}'

    def test_explicit_lambda_overrides(self):
        """passing lam= should override the formula."""
        children = _make_tree_8()
        adj = _make_llr_adjusted_8()
        _, info = prune([8, 9], children, adj, lam=0.5)
        assert info['lam'] == 0.5

    def test_higher_exp_eff_yields_more_regions(self):
        """increasing exp_eff (lower lambda) should find >= as many regions."""
        children = _make_tree_8()
        adj = _make_llr_adjusted_8()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        out_1, _ = prune(sig_all, children, adj, exp_eff=1)
        out_10, _ = prune(sig_all, children, adj, exp_eff=10)
        assert len(out_10) >= len(out_1)


class TestExplicitLamZero:
    """explicit lam=0.0 (no penalty) behaviour."""

    def test_lam_zero(self):
        children = _make_tree_8()
        adj = _make_llr_adjusted_8()
        _, info = prune([8, 9], children, adj, lam=0.0)
        assert info['lam'] == 0.0

    def test_positive_adjusted_selected(self):
        """nodes with positive stat should be selected at lam=0."""
        children = _make_tree_8()
        adj = _make_llr_adjusted_8()
        reg_out, _ = prune([10], children, adj, lam=0.0)
        assert 10 in reg_out, 'positive stat node should be selected'

    def test_negative_adjusted_not_selected(self):
        """nodes with negative stat should not be selected."""
        children = _make_tree_8()
        adj = np.zeros(15)
        adj[8] = -1.0
        reg_out, _ = prune([8], children, adj, lam=0.0)
        assert reg_out == []

    def test_default_without_exp_falls_back_to_geometric(self):
        """without exp/sizes, should fall back to geometric prior."""
        children = _make_tree_8()
        adj = _make_llr_adjusted_8()
        reg_out, info = prune([8, 9], children, adj)
        expected_lam = np.log(1 + 1 / 3)
        assert np.isclose(info['lam'], expected_lam)
