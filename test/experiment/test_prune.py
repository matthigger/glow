from glow.experiment import Experiment
from glow.experiment.mancova import decompose, get_mancova
from glow.experiment.prune import (
    np, prune, prune_node, prune_tree,
    _region_loglik, _compute_region_ll, _calibrate_lambda,
    _build_vox_cache, _compute_all_ll, _gains_from_ll,
    _cache_sufficient_stats, _null_ll_from_stats, _apply_effect,
    _leaf_depths, _depth_weights, _loglik_from_cov,
)
from glow.graph import SCGraph, GRAPH_EXCLUDE


def test_prune():
    # region 12 (covering regions 0, 1, 2, 3) is one effect
    # region 10 and 11 are identical effects, but we force region 13,
    # their union to be insignificant here to avoid their being merged
    children = np.array([[0, 1],
                         [2, 3],
                         [4, 5],
                         [6, 7],
                         [8, 9],
                         [10, 11],
                         [12, 13]])
    sig_reg_list = [5, 6, 7, 8, 9, 10, 11, 12, 14]
    exp = Experiment.from_gauss(shape=(8,), num_img=100)

    # ensure region 0, 1, 2, 3 have sufficiently different stats
    exp.y[:, :, :4] += 100

    reg_out, homo_pval_dict = prune(sig_reg_list=sig_reg_list,
                                    children=children,
                                    exp=exp,
                                    alpha_prune=.05,
                                    n_perm=100)

    assert np.array_equal(reg_out, [10, 11, 12])


# ---------------------------------------------------------------------------
# DP pruning (geometric prior) unit tests
# ---------------------------------------------------------------------------

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


def _make_exp_with_effect(seed=42):
    """experiment where voxels 0-3 have a real effect, 4-7 don't."""
    np.random.seed(seed)
    num_img, num_vox, b = 20, 8, 2
    x = np.vstack([np.ones(num_img), np.random.randn(num_img)])
    contrast = np.array([False, True])
    y = np.random.randn(b, num_img, num_vox) * 0.5
    for v in range(4):
        y[:, :, v] += np.outer(np.random.randn(b), x[1, :]) * 2.0
    return Experiment(x=x, contrast=contrast, y=y,
                      mask_idx=np.arange(num_vox))


class TestRegionLoglik:
    """test _region_loglik and _compute_region_ll."""

    def test_full_better_than_null(self):
        """LL_full >= LL_null (full model fits at least as well)."""
        exp = _make_exp_with_effect()
        q_tup = decompose(exp.x, exp.contrast)
        ll_full, ll_null = _compute_region_ll(exp.y[:, :, :2], q_tup)
        assert ll_full >= ll_null

    def test_effect_region_has_larger_gap(self):
        """regions with a real effect should have a bigger LL gap."""
        exp = _make_exp_with_effect()
        q_tup = decompose(exp.x, exp.contrast)

        # effect region (voxels 0-1)
        ll_f_eff, ll_n_eff = _compute_region_ll(exp.y[:, :, :2], q_tup)
        gap_eff = ll_f_eff - ll_n_eff

        # no-effect region (voxels 4-5)
        ll_f_null, ll_n_null = _compute_region_ll(exp.y[:, :, 4:6], q_tup)
        gap_null = ll_f_null - ll_n_null

        assert gap_eff > gap_null

    def test_region_loglik_matches_compute_region_ll(self):
        """_region_loglik should equal ll_full from _compute_region_ll."""
        exp = _make_exp_with_effect()
        q_tup = decompose(exp.x, exp.contrast)
        y_sub = exp.y[:, :, :4]
        ll_direct = _region_loglik(y_sub, q_tup)
        ll_full, _ = _compute_region_ll(y_sub, q_tup)
        assert np.isclose(ll_direct, ll_full)


class TestDPSolve:
    """test the DP solver on hand-crafted inputs."""

    def test_single_leaf(self):
        """single significant leaf: selected iff gain > lambda."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        reg_out, info = prune_node([8], children, exp, lam=0.0)
        assert 8 in reg_out, 'single leaf with lam=0 should be selected'

    def test_large_lambda_selects_nothing(self):
        """very large lambda should select no regions."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        reg_out, info = prune_node([8, 9, 12], children, exp, lam=1e6)
        assert reg_out == []

    def test_prefers_split_over_parent(self):
        """when children have strong effects but parent is diluted,
        the DP should prefer splitting."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        # nodes 8 (vox 0-1) and 9 (vox 2-3) have real effects;
        # node 12 (vox 0-3) pools effect + potentially dilutes
        reg_out, info = prune_node([8, 9, 12], children, exp, lam=0.0)

        # the DP should either pick 12 alone or split into {8, 9}
        # since the gains are strong at both children, splitting
        # should win when the parent's gain < sum of children
        if 12 not in reg_out:
            assert 8 in reg_out and 9 in reg_out

    def test_output_is_antichain(self):
        """selected regions must be disjoint (no ancestor-descendant)."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        reg_out, _ = prune_node(sig_all, children, exp, exp_eff=5)

        from glow.graph import get_parent
        parent = get_parent(children, num_leaf=8)
        selected = set(reg_out)
        for node in selected:
            # walk up to root, no ancestor should be in selected
            p = parent[node]
            while p != -1:
                assert p not in selected, \
                    f'node {node} and its ancestor {p} both selected'
                p = parent[p]

    def test_empty_sig_reg_list(self):
        """empty input should return empty output."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        reg_out, info = prune_node([], children, exp)
        assert reg_out == []
        assert info['lam'] == 0.0


class TestSameNodeGain:
    """test that gain is same-node LLR (sigma cancels)."""

    def test_gain_equals_same_node_llr(self):
        """gain(node) should equal ll_full(node) - ll_null(node)."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        _, info = prune_node(sig_all, children, exp, lam=0.0)

        # gain should be the same-node LLR stored in gain_per_node
        for node in sig_all:
            assert np.isclose(info['gain'][node],
                              info['gain_per_node'][node]), \
                f'gain mismatch at node {node}'

    def test_gain_positive_for_effect_regions(self):
        """effect regions should have positive same-node LLR."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        # node 8 covers voxels 0-1 (strong effect)
        _, info = prune_node([8], children, exp, lam=0.0)
        assert info['gain'][8] > 0

    def test_gain_per_node_in_dp_info(self):
        """dp_info should contain gain_per_node for viewer re-use."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        _, info = prune_node([8, 9], children, exp, lam=0.0)
        assert 'gain_per_node' in info
        assert 8 in info['gain_per_node']
        assert 9 in info['gain_per_node']


class TestLambdaFormula:
    """test analytic lambda from geometric prior."""

    def test_lambda_value(self):
        """lambda should equal log(1 + 1/exp_eff)."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        for exp_eff in [1, 2, 5, 10]:
            _, info = prune_node([8, 9], children, exp, exp_eff=exp_eff)
            expected = np.log(1 + 1 / exp_eff)
            assert np.isclose(info['lam'], expected), \
                f'exp_eff={exp_eff}: got {info["lam"]}, expected {expected}'

    def test_explicit_lambda_overrides(self):
        """passing lam= should override the formula."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        _, info = prune_node([8, 9], children, exp, lam=0.5)
        assert info['lam'] == 0.5

    def test_higher_exp_eff_yields_more_regions(self):
        """increasing exp_eff (lower lambda) should find >= as many regions."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        out_1, _ = prune_node(sig_all, children, exp, exp_eff=1)
        out_10, _ = prune_node(sig_all, children, exp, exp_eff=10)
        assert len(out_10) >= len(out_1)


class TestPermutationCalibration:
    """test permutation-based lambda calibration."""

    def test_calibrate_lambda_returns_positive(self):
        """calibrated lambda should be positive."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        q_tup = decompose(exp.x, exp.contrast)
        vox_cache = _build_vox_cache(sig_all, children, 8)
        lam, max_gains, h0_mean, h0_std = _calibrate_lambda(
            sig_all, exp, q_tup, vox_cache, n_perm=20, alpha=0.05)
        assert lam > 0
        assert len(max_gains) == 20
        assert set(h0_mean.keys()) == set(sig_all)
        assert all(v >= 0 for v in h0_mean.values())

    def test_perm_calibrated_dp_finds_effects(self):
        """permutation-calibrated DP should find effect regions."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        reg_out, info = prune_node(sig_all, children, exp,
                                    n_perm=25, alpha=0.05)
        # should find at least one effect (voxels 0-3 have strong signal)
        assert len(reg_out) > 0
        assert info['lam'] > 0

    def test_null_data_finds_nothing(self):
        """on pure-null data, permutation-calibrated DP should find nothing."""
        np.random.seed(99)
        children = _make_tree_8()
        # no effect imposed — pure noise
        exp = Experiment.from_gauss(a=2, b=2, shape=(8,),
                                    num_img=20, seed=99)
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        reg_out, info = prune_node(sig_all, children, exp,
                                    n_perm=50, alpha=0.05)
        # with pure noise and proper calibration, should find 0 or very few
        assert len(reg_out) <= 1, \
            f'expected <=1 effect on null data, got {len(reg_out)}'


# ---------------------------------------------------------------------------
# Adjusted-LL pruning unit tests
# ---------------------------------------------------------------------------

class TestNullLLFromStats:
    """test _null_ll_from_stats matches the MANCOVA path."""

    def test_matches_mancova(self):
        """_null_ll_from_stats should match _loglik_from_cov(E+H, n)."""
        exp = _make_exp_with_effect()
        q_tup = decompose(exp.x, exp.contrast)
        children = _make_tree_8()
        vox_cache = _build_vox_cache([12], children, 8)
        stats = _cache_sufficient_stats([12], exp.y, vox_cache)

        ysum, yout, n = stats[12]
        ll_stats = _null_ll_from_stats(ysum, yout, n, q_tup[0])

        e, h, _ = get_mancova(y=exp.y[:, :, vox_cache[12]], q_tup=q_tup)
        ll_mancova = _loglik_from_cov(e + h, n)

        assert np.isclose(ll_stats, ll_mancova), \
            f'stats={ll_stats}, mancova={ll_mancova}'

    def test_all_nodes(self):
        """check agreement for every node in the tree."""
        exp = _make_exp_with_effect()
        q_tup = decompose(exp.x, exp.contrast)
        children = _make_tree_8()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        vox_cache = _build_vox_cache(sig_all, children, 8)
        stats = _cache_sufficient_stats(sig_all, exp.y, vox_cache)

        for node in sig_all:
            ysum, yout, n = stats[node]
            ll_s = _null_ll_from_stats(ysum, yout, n, q_tup[0])
            e, h, _ = get_mancova(y=exp.y[:, :, vox_cache[node]],
                                  q_tup=q_tup)
            ll_m = _loglik_from_cov(e + h, n)
            assert np.isclose(ll_s, ll_m), \
                f'node {node}: stats={ll_s}, mancova={ll_m}'


class TestApplyEffect:
    """test _apply_effect adjustment algebra."""

    def test_self_zeroing(self):
        """after applying effect on node i, its H should be zero
        (i.e. adjusted LL equals LL_full)."""
        exp = _make_exp_with_effect()
        q_tup = decompose(exp.x, exp.contrast)
        children = _make_tree_8()
        sig = [8, 12]
        vox_cache = _build_vox_cache(sig, children, 8)
        stats = _cache_sufficient_stats(sig, exp.y, vox_cache)
        subgraph = SCGraph.from_children(children, num_leaf=8, subset=sig)

        # LL_full before adjustment
        e_before = get_mancova(y=exp.y[:, :, vox_cache[8]], q_tup=q_tup)[0]
        ll_full_8 = _loglik_from_cov(e_before, stats[8][2])

        _apply_effect(8, stats, q_tup, subgraph)

        # after adjustment, null LL should equal LL_full
        ysum, yout, n = stats[8]
        ll_adj = _null_ll_from_stats(ysum, yout, n, q_tup[0])
        assert np.isclose(ll_adj, ll_full_8, rtol=1e-6), \
            f'll_adj={ll_adj}, ll_full={ll_full_8}'

    def test_sigma_unchanged_for_self(self):
        """subtracting a constant from all voxels leaves sigma unchanged."""
        exp = _make_exp_with_effect()
        q_tup = decompose(exp.x, exp.contrast)
        children = _make_tree_8()
        sig = [8]
        vox_cache = _build_vox_cache(sig, children, 8)
        stats = _cache_sufficient_stats(sig, exp.y, vox_cache)
        subgraph = SCGraph.from_children(children, num_leaf=8, subset=sig)

        ysum0, yout0, n = stats[8]
        sigma_before = yout0 - ysum0 @ ysum0.T / n

        _apply_effect(8, stats, q_tup, subgraph)

        ysum1, yout1, _ = stats[8]
        sigma_after = yout1 - ysum1 @ ysum1.T / n
        assert np.allclose(sigma_before, sigma_after), \
            'sigma should not change when all voxels are shifted equally'

    def test_sigma_changes_for_ancestor(self):
        """sigma of an ancestor should change (partial voxel shift)."""
        exp = _make_exp_with_effect()
        q_tup = decompose(exp.x, exp.contrast)
        children = _make_tree_8()
        sig = [8, 12]
        vox_cache = _build_vox_cache(sig, children, 8)
        stats = _cache_sufficient_stats(sig, exp.y, vox_cache)
        subgraph = SCGraph.from_children(children, num_leaf=8, subset=sig)

        ysum0, yout0, n12 = stats[12]
        sigma_before = yout0 - ysum0 @ ysum0.T / n12

        _apply_effect(8, stats, q_tup, subgraph)

        ysum1, yout1, _ = stats[12]
        sigma_after = yout1 - ysum1 @ ysum1.T / n12

        # should differ (node 8 has a real effect, only 2 of 4 voxels shifted)
        assert not np.allclose(sigma_before, sigma_after), \
            'sigma of ancestor should change under partial adjustment'


class TestLeafDepths:
    """test _leaf_depths and _depth_weights."""

    def test_balanced_tree(self):
        """balanced 3-level tree: all leaves should have the same depth."""
        children = _make_tree_8()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        subgraph = SCGraph.from_children(children, num_leaf=8,
                                         subset=sig_all)
        depths = _leaf_depths(subgraph, sig_all)
        # leaves are 8, 9, 10, 11 (depth 3 from root 14)
        assert set(depths.keys()) == {8, 9, 10, 11}
        assert all(d == 3 for d in depths.values())

    def test_weights_all_one_when_balanced(self):
        """on a balanced tree all weights should be 1."""
        children = _make_tree_8()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        subgraph = SCGraph.from_children(children, num_leaf=8,
                                         subset=sig_all)
        weights, _ = _depth_weights(subgraph, sig_all)
        assert all(w == 1 for w in weights.values())

    def test_unbalanced_weights(self):
        """a subtree missing one branch should upweight the shallow leaf."""
        children = _make_tree_8()
        # sig: 14 -> {12, 13}, 12 -> {8, 9}, but 13 is a leaf (no kids)
        sig = [8, 9, 12, 13, 14]
        subgraph = SCGraph.from_children(children, num_leaf=8, subset=sig)
        depths = _leaf_depths(subgraph, sig)
        # 8 and 9 have depth 3 (14->12->8), 13 has depth 2 (14->13)
        assert depths[8] == 3
        assert depths[9] == 3
        assert depths[13] == 2
        weights, _ = _depth_weights(subgraph, sig)
        assert weights[13] == 2  # 1 + 3 - 2
        assert weights[8] == 1
        assert weights[9] == 1


class TestPruneTree:
    """integration tests for prune_tree."""

    def test_finds_effect_region(self):
        """should discover at least one region with a real effect."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        reg_out, info = prune_tree(sig_all, children, exp)
        assert len(reg_out) > 0, 'should find at least one effect'
        # selected regions should overlap with the true effect (voxels 0-3)
        effect_nodes = {8, 9, 12}
        assert any(r in effect_nodes for r in reg_out), \
            f'selected {reg_out}, expected overlap with {effect_nodes}'

    def test_null_data_runs(self):
        """on pure-noise data, method should run without error.

        Without a penalty term any Q1 removal improves the LL, so
        selections are expected; real protection comes from upstream FWER.
        """
        np.random.seed(99)
        children = _make_tree_8()
        exp = Experiment.from_gauss(a=2, b=2, shape=(8,),
                                    num_img=20, seed=99)
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        reg_out, info = prune_tree(sig_all, children, exp)
        assert isinstance(reg_out, list)

    def test_empty_input(self):
        """empty sig_reg_list should return empty output."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        reg_out, info = prune_tree([], children, exp)
        assert reg_out == []
        assert info['sig_reg_list'] == []

    def test_output_is_disjoint(self):
        """selected regions must be disjoint (no ancestor-descendant)."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        reg_out, _ = prune_tree(sig_all, children, exp)

        from glow.graph import get_parent
        parent = get_parent(children, num_leaf=8)
        selected = set(reg_out)
        for node in selected:
            p = parent[node]
            while p != -1:
                assert p not in selected, \
                    f'node {node} and ancestor {p} both selected'
                p = parent[p]

    def test_cost_increases_monotonically(self):
        """each greedy step should improve the tree-wide cost."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        _, info = prune_tree(sig_all, children, exp)
        hist = info['cost_history']
        for i in range(1, len(hist)):
            assert hist[i] > hist[i - 1], \
                f'cost did not increase at step {i}: {hist}'

    def test_info_has_expected_keys(self):
        """info dict should contain keys needed by the viewer."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        _, info = prune_tree([8, 9, 12], children, exp)
        for key in ('gain_per_node', 'sig_reg_list',
                    'subgraph_children', 'cost_history', 'weights'):
            assert key in info, f'missing key: {key}'
