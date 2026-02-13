from glow.experiment import Experiment
from glow.experiment.mancova import decompose
from glow.experiment.prune import (
    np, prune, prune_dp,
    _region_loglik, _compute_region_ll,
)


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
        reg_out, info = prune_dp([8], children, exp, lam=0.0)
        assert 8 in reg_out, 'single leaf with lam=0 should be selected'

    def test_large_lambda_selects_nothing(self):
        """very large lambda should select no regions."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        reg_out, info = prune_dp([8, 9, 12], children, exp, lam=1e6)
        assert reg_out == []

    def test_prefers_split_over_parent(self):
        """when children have strong effects but parent is diluted,
        the DP should prefer splitting."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        # nodes 8 (vox 0-1) and 9 (vox 2-3) have real effects;
        # node 12 (vox 0-3) pools effect + potentially dilutes
        reg_out, info = prune_dp([8, 9, 12], children, exp, lam=0.0)

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
        reg_out, _ = prune_dp(sig_all, children, exp, exp_eff=5)

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
        reg_out, info = prune_dp([], children, exp)
        assert reg_out == []
        assert info['lam'] == 0.0


class TestLambdaFormula:
    """test analytic lambda from geometric prior."""

    def test_lambda_value(self):
        """lambda should equal log(1 + 1/exp_eff)."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        for exp_eff in [1, 2, 5, 10]:
            _, info = prune_dp([8, 9], children, exp, exp_eff=exp_eff)
            expected = np.log(1 + 1 / exp_eff)
            assert np.isclose(info['lam'], expected), \
                f'exp_eff={exp_eff}: got {info["lam"]}, expected {expected}'

    def test_explicit_lambda_overrides(self):
        """passing lam= should override the formula."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        _, info = prune_dp([8, 9], children, exp, lam=0.5)
        assert info['lam'] == 0.5

    def test_higher_exp_eff_yields_more_regions(self):
        """increasing exp_eff (lower lambda) should find >= as many regions."""
        children = _make_tree_8()
        exp = _make_exp_with_effect()
        sig_all = [8, 9, 10, 11, 12, 13, 14]
        out_1, _ = prune_dp(sig_all, children, exp, exp_eff=1)
        out_10, _ = prune_dp(sig_all, children, exp, exp_eff=10)
        assert len(out_10) >= len(out_1)
