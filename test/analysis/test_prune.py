import itertools

import numpy as np
import pytest

from glow.analysis.prune import (PRUNE_RULE_LIST, check_prune_rule,
                                 dp_antichain, prune_by_rule, prune_dp,
                                 prune_greedy, prune_maxllr, prune_oracle)
from glow.graph import SCGraph, get_parent


def _region_size(children, num_vox: int):
    """Voxel count per region index (a leaf 1, a node its children's sum).

    Args:
        children (np.array): (num_internal, 2) Ward child-index pairs
        num_vox (int): leaf count

    Returns:
        size (np.array): (num_reg,) int voxels per region
    """
    size = np.zeros(num_vox + children.shape[0], dtype=int)
    size[:num_vox] = 1
    for i, (c0, c1) in enumerate(children):
        size[num_vox + i] = size[c0] + size[c1]
    return size


def _cover_frac(reg_idx_list, sig_reg_list, children, num_vox: int):
    """Share of each region's voxels its nearest significant descendants cover.

    prune_dp weighs a region against the antichain of its nearest significant
    descendants (the short-circuited SCGraph it runs over), not against its two
    Ward children, and those descendants tile the region only when the share is
    1. Below 1 the shortfall is sub-threshold voxels that no selection of
    significant regions can claim, which is what makes blooming the region
    worth more than the parts.

    Args:
        reg_idx_list (list): the regions to measure (a rule's output)
        sig_reg_list (list): int regions declared significant (via FWER)
        children (np.array): (num_internal, 2) Ward child-index pairs
        num_vox (int): leaf count

    Returns:
        list[float]: one share per region, in reg_idx_list order; 0.0 for a
            region with no significant descendant (already minimal)
    """
    size = _region_size(children, num_vox)
    subgraph = SCGraph.from_children(children, num_leaf=num_vox,
                                     subset=sig_reg_list)
    return [sum(size[k] for k in subgraph.children.get(reg, [])) / size[reg]
            for reg in reg_idx_list]


def _random_tree(num_vox: int, rng):
    """Sample a random binary merge tree over num_vox leaves.

    Merges two uniformly-chosen active nodes at a time, so the topology
    ranges over caterpillars and balanced trees alike -- Ward's own trees
    are neither, and the antichain optimum is a claim about topology.

    Args:
        num_vox (int): leaf count
        rng (np.random.Generator): source of the merge order

    Returns:
        children (np.array): (num_vox - 1, 2) child-index pairs, bottom-up
    """
    active = list(range(num_vox))
    children = []
    for i in range(num_vox - 1):
        lo, hi = sorted(rng.choice(len(active), size=2, replace=False))
        c1 = active.pop(int(hi))
        c0 = active.pop(int(lo))
        children.append((c0, c1))
        active.append(num_vox + i)
    return np.array(children, dtype=int)


def _relatives(reg: int, children, num_vox: int, parent):
    """Strict ancestors and descendants of reg (the regions it may not join).

    Args:
        reg (int): region index
        children (np.array): (num_internal, 2) child-index pairs
        num_vox (int): leaf count
        parent (np.array): (num_reg,) parent index, -1 at a root

    Returns:
        set: int region indices sharing a voxel with reg, excluding reg
    """
    out = set()
    p = parent[reg]
    while p != -1:
        out.add(int(p))
        p = parent[p]
    stack = [reg]
    while stack:
        node = stack.pop()
        if node >= num_vox:
            for kid in children[node - num_vox]:
                out.add(int(kid))
                stack.append(int(kid))
    return out


def _brute_force_best(sig_reg_list, children, stat, lam: float) -> float:
    """Exhaustive max of sum(stat[r] - lam) over antichains of the sig set.

    This is dp_antichain's objective computed the slow, unarguable way:
    every subset of the significant regions is tested for the
    ancestor-descendant conflicts that disqualify it, and the survivors are
    scored. The empty selection scores 0, so the optimum is never negative.
    Exponential in len(sig_reg_list) -- keep it small.

    Args:
        sig_reg_list (list): int regions declared significant
        children (np.array): (num_internal, 2) child-index pairs
        stat (np.array): (num_reg,) per-region stat (raw LLR)
        lam (float): per-region penalty

    Returns:
        float: the attainable optimum
    """
    num_vox = len(stat) - children.shape[0]
    parent = get_parent(children, num_leaf=num_vox)
    sig = sorted(sig_reg_list)
    conflict = {r: _relatives(r, children, num_vox, parent) for r in sig}

    best = 0.0
    for size in range(1, len(sig) + 1):
        for subset in itertools.combinations(sig, size):
            chosen = set(subset)
            if any(conflict[r] & chosen for r in subset):
                continue
            best = max(best, sum(stat[r] - lam for r in subset))
    return float(best)


def _mask_from_leaves(mask_idx, leaf_list):
    """Boolean target mask covering exactly the given leaf (voxel) indices.

    Args:
        mask_idx (np.array): (X, Y, Z) int voxel-index array
        leaf_list (iterable): leaf indices the target covers

    Returns:
        mask (np.array): (X, Y, Z) bool, True at those voxels
    """
    return np.isin(mask_idx, list(leaf_list))


def _dice_of(reg_out, children, mask_target, mask_idx) -> float:
    """Dice of a disjoint region set, counted straight off the masks.

    The independent reference for prune_oracle's own arithmetic: rebuild
    each region's voxel set by walking the tree and count the overlap,
    rather than summing the per-region counts prune_oracle sums.

    Args:
        reg_out (list): selected region indices
        children (np.array): (num_internal, 2) child-index pairs
        mask_target (np.array): (X, Y, Z) bool target support
        mask_idx (np.array): (X, Y, Z) int voxel-index array

    Returns:
        float: 2 tp / (2 tp + fp + fn), 0.0 for an empty selection
    """
    num_vox = int((mask_idx > -1).sum())
    leaf_set = set()
    for reg in reg_out:
        stack = [int(reg)]
        while stack:
            node = stack.pop()
            if node < num_vox:
                leaf_set.add(node)
            else:
                stack.extend(int(k) for k in children[node - num_vox])
    mask_pred = _mask_from_leaves(mask_idx, leaf_set)
    tp = int((mask_pred & mask_target).sum())
    denom = int(mask_pred.sum()) + int(mask_target.sum())
    return 0.0 if denom == 0 else 2.0 * tp / denom


def _brute_force_best_dice(sig_reg_list, children, mask_target,
                           mask_idx) -> float:
    """Exhaustive max Dice over antichains of the significant set.

    prune_oracle's objective computed the slow, unarguable way: every
    subset of the significant regions is tested for the
    ancestor-descendant conflicts that disqualify it, and the survivors are
    scored off the masks. Exponential in len(sig_reg_list) -- keep it small.

    Args:
        sig_reg_list (list): int regions declared significant
        children (np.array): (num_internal, 2) child-index pairs
        mask_target (np.array): (X, Y, Z) bool target support
        mask_idx (np.array): (X, Y, Z) int voxel-index array

    Returns:
        float: the attainable maximum Dice (0 when nothing scores)
    """
    num_vox = int((mask_idx > -1).sum())
    parent = get_parent(children, num_leaf=num_vox)
    sig = sorted(sig_reg_list)
    conflict = {r: _relatives(r, children, num_vox, parent) for r in sig}

    best = 0.0
    for size in range(1, len(sig) + 1):
        for subset in itertools.combinations(sig, size):
            chosen = set(subset)
            if any(conflict[r] & chosen for r in subset):
                continue
            best = max(best, _dice_of(subset, children, mask_target, mask_idx))
    return float(best)


def _assert_antichain(reg_out, children, num_vox: int):
    """Assert no selected region is an ancestor of another."""
    parent = get_parent(children, num_leaf=num_vox)
    selected = set(reg_out)
    for node in selected:
        p = parent[node]
        while p != -1:
            assert p not in selected, \
                f'node {node} and its ancestor {p} both selected'
            p = parent[p]


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

    def test_never_blooms_a_region_its_significant_parts_partition(self):
        """A region the significant regions tile is never output.

        Region LLR is subadditive under a Ward merge -- the fixture's
        llr[12] = 8 is below llr[8] + llr[9] = 9 -- so when both halves are
        themselves significant, splitting scores higher and the parent loses.
        Drop one half from the significant set and the comparison changes: the
        parent beats its only selectable part (8 > 5) and blooms, carrying the
        sub-threshold voxels of 9 that nothing else can claim. Greedy, ranking
        by raw LLR alone, does output the tiled parent.
        """
        children = _make_tree_8()
        llr = _make_llr_8()

        tiled, _ = prune_dp([8, 9, 12], children, llr)
        assert tiled == [8, 9]
        assert _cover_frac(tiled, [8, 9, 12], children, 8) == [0.0, 0.0]

        partial, _ = prune_dp([8, 12], children, llr)
        assert partial == [12]
        # region 8 is 2 of region 12's 4 voxels: a tiling would read 1.0
        assert _cover_frac(partial, [8, 12], children, 8) == [0.5]

        greedy_out, _ = prune_greedy([8, 9, 12], children, llr)
        assert greedy_out == [12]
        assert _cover_frac(greedy_out, [8, 9, 12], children, 8) == [1.0]

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

    @pytest.mark.parametrize('lam', [0.0, 0.5])
    def test_dp_attains_the_exhaustive_optimum(self, lam):
        """prune_dp attains the brute-forced optimum on random trees.

        The claim in prune_dp's docstring is global optimality of
        sum(stat[r] - lam) over antichains, so the reference is exhaustive
        enumeration, not greedy. Ties break toward blooming the parent, so
        the selected SET may differ from a brute-force argmax while the
        attained value may not -- hence the assert is on the value.

        Also asserts the check is not vacuous: greedy must come out
        strictly worse somewhere, or the test would equally pass on a
        greedy implementation.
        """
        rng = np.random.default_rng(0)
        n_beat_greedy = 0

        for trial in range(12):
            num_vox = int(rng.integers(6, 11))
            children = _random_tree(num_vox, rng)
            num_reg = num_vox + children.shape[0]
            # centred so a fair share of regions carry a negative stat, the
            # case where dropping a region beats blooming it
            stat = rng.standard_normal(num_reg) + 0.5

            # a random significant subset, small enough to enumerate
            n_sig = int(rng.integers(3, min(11, num_reg) + 1))
            sig = sorted(rng.choice(num_reg, size=n_sig, replace=False)
                         .tolist())

            dp_out, _ = prune_dp(sig, children, stat, lam=lam)
            _assert_antichain(dp_out, children, num_vox)

            got = sum(stat[r] - lam for r in dp_out)
            want = _brute_force_best(sig, children, stat, lam)
            assert np.isclose(got, want), (
                f'trial {trial} (num_vox={num_vox}, lam={lam}): prune_dp '
                f'scored {got:.6f}, exhaustive optimum is {want:.6f}')

            greedy_out, _ = prune_greedy(sig, children, stat)
            n_beat_greedy += sum(stat[r] - lam
                                 for r in greedy_out) < got - 1e-9

        assert n_beat_greedy, (
            'prune_dp never beat prune_greedy over 12 random trees -- the '
            'optimality check never saw a case greedy gets wrong')

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


class TestPruneOracle:
    """Test the max-Dice oracle rule (the headroom line, not a method)."""

    @staticmethod
    def _mask_idx_8():
        """Voxel-index array for the 8-leaf tree, one voxel per leaf."""
        return np.arange(8).reshape(2, 2, 2)

    def test_exact_region_scores_one(self):
        """A target that is exactly a significant region is recovered."""
        children = _make_tree_8()
        mask_idx = self._mask_idx_8()
        # region 8 is the pair of leaves 0, 1
        mask_target = _mask_from_leaves(mask_idx, [0, 1])
        reg_out, info = prune_oracle([8, 9, 12], children, mask_target,
                                     mask_idx)
        assert reg_out == [8]
        assert info['dice'] == pytest.approx(1.0)

    def test_unions_disjoint_regions(self):
        """A target spanning two regions is matched by their union."""
        children = _make_tree_8()
        mask_idx = self._mask_idx_8()
        # regions 8 (leaves 0, 1) and 10 (leaves 4, 5); their parents 12 / 14
        # each drag in leaves the target does not have
        mask_target = _mask_from_leaves(mask_idx, [0, 1, 4, 5])
        reg_out, info = prune_oracle([8, 9, 10, 12, 14], children,
                                     mask_target, mask_idx)
        assert reg_out == [8, 10]
        assert info['dice'] == pytest.approx(1.0)

    def test_beats_the_llr_ranking_when_it_over_covers(self):
        """The oracle drops the big region raw LLR blooms over the target."""
        children = _make_tree_8()
        mask_idx = self._mask_idx_8()
        llr = _make_llr_8()
        sig = [8, 9, 12]
        # the LLR peaks on the parent 12, so greedy blooms all four of its
        # leaves for a target of two
        mask_target = _mask_from_leaves(mask_idx, [0, 1])
        greedy_out, _ = prune_greedy(sig, children, llr)
        assert greedy_out == [12]

        reg_out, info = prune_oracle(sig, children, mask_target, mask_idx)
        assert reg_out == [8]
        assert info['dice'] > _dice_of(greedy_out, children, mask_target,
                                       mask_idx)

    def test_output_is_an_antichain(self):
        """No selected region is an ancestor of another."""
        children = _make_tree_8()
        mask_idx = self._mask_idx_8()
        mask_target = _mask_from_leaves(mask_idx, [0, 1, 2, 5])
        reg_out, _ = prune_oracle(list(range(8, 15)), children, mask_target,
                                  mask_idx)
        _assert_antichain(reg_out, children, num_vox=8)

    def test_empty_significant_set(self):
        """Nothing significant selects nothing, at Dice 0."""
        children = _make_tree_8()
        mask_idx = self._mask_idx_8()
        mask_target = _mask_from_leaves(mask_idx, [0, 1])
        reg_out, info = prune_oracle([], children, mask_target, mask_idx)
        assert reg_out == [] and info['dice'] == 0.0

    def test_empty_target(self):
        """An empty target selects nothing (every Dice is 0)."""
        children = _make_tree_8()
        mask_idx = self._mask_idx_8()
        mask_target = np.zeros_like(mask_idx, dtype=bool)
        reg_out, info = prune_oracle([8, 9, 12], children, mask_target,
                                     mask_idx)
        assert reg_out == [] and info['dice'] == 0.0

    def test_target_outside_the_analysis_is_ignored(self):
        """Voxels outside mask_idx do not count against the Dice."""
        children = _make_tree_8()
        # one extra voxel excluded from the analysis (-1)
        mask_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, -1])
        mask_target = _mask_from_leaves(mask_idx, [0, 1])
        mask_target[-1] = True
        reg_out, info = prune_oracle([8, 9, 12], children, mask_target,
                                     mask_idx)
        assert reg_out == [8]
        assert info['dice'] == pytest.approx(1.0)

    def test_attains_the_exhaustive_optimum(self):
        """prune_oracle attains the brute-forced max Dice on random trees.

        The docstring claims a global maximum over antichains, so the
        reference is exhaustive enumeration. Ties may pick a different set
        at the same Dice, so the assert is on the attained value -- as for
        prune_dp. Also asserts the check is not vacuous: greedy LLR pruning
        must come out strictly worse somewhere.
        """
        rng = np.random.default_rng(0)
        n_beat_greedy = 0

        for trial in range(12):
            num_vox = int(rng.integers(6, 11))
            children = _random_tree(num_vox, rng)
            num_reg = num_vox + children.shape[0]
            mask_idx = np.arange(num_vox)
            n_eff = int(rng.integers(1, num_vox))
            mask_target = _mask_from_leaves(
                mask_idx, rng.choice(num_vox, size=n_eff, replace=False))

            # a random significant subset, small enough to enumerate
            n_sig = int(rng.integers(3, min(11, num_reg) + 1))
            sig = sorted(rng.choice(num_reg, size=n_sig, replace=False)
                         .tolist())

            reg_out, info = prune_oracle(sig, children, mask_target, mask_idx)
            _assert_antichain(reg_out, children, num_vox)

            got = _dice_of(reg_out, children, mask_target, mask_idx)
            want = _brute_force_best_dice(sig, children, mask_target, mask_idx)
            assert np.isclose(got, want), (
                f'trial {trial} (num_vox={num_vox}): prune_oracle scored '
                f'{got:.6f}, exhaustive optimum is {want:.6f}')
            assert np.isclose(got, info['dice']), (
                f"trial {trial}: reported dice {info['dice']:.6f} is not the "
                f'selection\'s own {got:.6f}')

            stat = rng.standard_normal(num_reg) + 0.5
            greedy_out, _ = prune_greedy(sig, children, stat)
            n_beat_greedy += _dice_of(greedy_out, children, mask_target,
                                      mask_idx) < got - 1e-9

        assert n_beat_greedy, (
            'prune_oracle never beat prune_greedy over 12 random trees -- '
            'the optimality check never saw a case the LLR ranking misses')


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




# ---------------------------------------------------------------------------
# Empirical: the same invariant on real GLOW fits
# ---------------------------------------------------------------------------
# The synthetic case above fixes the LLRs by hand. These fit GLOW on planted
# data and check every region prune_dp outputs against the significant set it
# pruned, so the invariant is confirmed on trees and significance the analysis
# actually produced. Slow (a real permutation test per fit); the HCP case is
# skipped where the reference dataset is absent.

_CROP_N_VOX = 5_000
_N_PERM_FWER = 100
_N_PERM_INNER = 250
_EFFECT_LLR = 0.03


def _plant(exp_img, seed: int):
    """Crop an image-only experiment to a sphere and plant one effect."""
    from glow._extra.benchmark.data import _sample_x_and_crop
    from glow.effect import EffectSynthetic, ExtenterMinVar, ExtenterSphere

    # the benchmark's data_factory / effect_factory are memoised and recorded;
    # these call the builders underneath so the test writes no cache or records
    exp = _sample_x_and_crop(
        exp_img, a=1, contrast=None, has_bias=True, seed=seed,
        extenter=ExtenterSphere(n_vox=_CROP_N_VOX, connected=True,
                                contiguous=True, seed=seed))
    n_vox = round(0.1 * int((exp.mask_idx > -1).sum()))
    exp, _ = EffectSynthetic(
        extenter=ExtenterMinVar(n_vox=n_vox, seed=seed),
        effect_llr=_EFFECT_LLR).fit(exp)
    return exp


def _dp_cover(exp, cluster_mode):
    """Fit GLOW, prune by DP, and return (n_sig, selected, cover fractions)."""
    from glow.analysis import AnalysisGLOWSplit

    ana = AnalysisGLOWSplit(n_perm_fwer=_N_PERM_FWER,
                       alpha_fwer=0.05, cluster_mode=cluster_mode).fit(exp)
    sig = np.where(ana.fwer.pval <= ana.alpha_fwer)[0].tolist()
    if not sig:
        return 0, [], []
    # the rules rank by raw LLR, NaN / inf zeroed (glow_fit_for_prune)
    llr = np.nan_to_num(ana.llr.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    num_vox = len(llr) - ana.children.shape[0]
    selected, _ = prune_dp(sig_reg_list=sig, children=ana.children, stat=llr)
    cover = _cover_frac(selected, sig, ana.children, num_vox)
    return len(sig), selected, cover


def _assert_no_tiled_selection(exp_img, source: str):
    """Assert no DP selection is tiled by significant regions, both Ward modes.

    Also asserts the check is not vacuous: the fits must produce significant
    regions, and at least one selection must have a significant descendant (a
    bloom) -- the only case where a tiling could arise. Prints the per-fit
    tally (pytest -s to see it), the cover fractions being the measurement:
    how close the significant parts come to tiling a region that was output
    whole.
    """
    from glow.analysis.cluster import ClusterMode

    n_bloom = 0
    n_sel = 0
    worst = 0.0
    for seed in (0, 1):
        exp = _plant(exp_img, seed)
        for mode in (ClusterMode.FOCUS, ClusterMode.GLM_ERROR):
            n_sig, selected, cover = _dp_cover(exp, mode)
            for reg, frac in zip(selected, cover):
                assert frac < 1.0, (
                    f'{source} seed={seed} {mode}: region {reg} is tiled by '
                    f'significant regions (cover {frac:.4f}) yet prune_dp '
                    f'output it whole ({n_sig} significant regions)')
            n_bloom += sum(frac > 0 for frac in cover)
            n_sel += len(selected)
            worst = max([worst, *cover])
            print(f'{source} seed={seed} {str(mode):>9}: {n_sig:5d} '
                  f'significant, {len(selected):4d} output, '
                  f'{sum(frac > 0 for frac in cover):4d} bloomed over '
                  f'significant parts, max cover '
                  f'{max(cover, default=0.0):.4f}')
    print(f'{source}: {n_sel} regions output, {n_bloom} bloomed over '
          f'significant parts, none tiled (max cover {worst:.4f})')
    assert n_bloom, (
        f'{source}: no selection had a significant descendant -- the check '
        f'never saw a bloom, so it proves nothing')


@pytest.mark.slow
def test_dp_never_outputs_a_tiled_region_wgn():
    """No prune_dp output is tiled by significant regions, on WGN fits."""
    from glow.experiment import ExperimentImageOnly

    side = round(_CROP_N_VOX ** (1 / 3)) + 1
    exp_img = ExperimentImageOnly.from_gauss(shape=(side,) * 3, b=1,
                                             num_img=100, seed=0)
    _assert_no_tiled_selection(exp_img, 'WGN')


@pytest.mark.slow
def test_dp_never_outputs_a_tiled_region_hcp():
    """No prune_dp output is tiled by significant regions, on HCP fits."""
    from glow._extra.benchmark import hcp

    if not hcp.is_present():
        pytest.skip('HCP reference dataset not present '
                    '(see glow._extra.benchmark.hcp)')
    _assert_no_tiled_selection(hcp.build_exp_img_from_bundle(('fa',)), 'HCP')


# ---------- prune_maxllr: the single best region -----------------------------
class TestPruneMaxLLR:
    """Test the one-region rule."""

    def test_picks_the_highest(self):
        """The selection is the top-LLR region, whatever the tree."""
        reg_out, _ = prune_maxllr([8, 9, 12, 14], _make_llr_8())
        assert reg_out == [12]

    def test_selects_one_region(self):
        """However many are significant, exactly one comes back."""
        reg_out, _ = prune_maxllr(list(range(15)), _make_llr_8())
        assert len(reg_out) == 1

    def test_empty_sig_list(self):
        reg_out, info = prune_maxllr([], _make_llr_8())
        assert reg_out == []
        assert info['sig_reg_list'] == []

    def test_returns_a_python_int(self):
        """A numpy index in makes a python int out (it keys a record)."""
        reg_out, _ = prune_maxllr(list(np.array([8, 12])), _make_llr_8())
        assert type(reg_out[0]) is int


# ---------- prune_by_rule: one name -> one rule ------------------------------
class TestPruneByRule:
    """Test the rule dispatcher a GLOW recipe and a benchmark leaf share."""

    _SIG = [8, 9, 10, 11, 12, 13, 14]

    def test_dispatch_matches_each_rule_called_directly(self):
        """The dispatcher is a lookup, not a second implementation."""
        children = _make_tree_8()
        llr = _make_llr_8()
        direct = {
            'greedy': prune_greedy(self._SIG, children, llr)[0],
            'dp': prune_dp(self._SIG, children, llr)[0],
            'maxllr': prune_maxllr(self._SIG, llr)[0],
        }
        for rule, expect in direct.items():
            got, _ = prune_by_rule(rule, self._SIG, children, llr)
            assert got == expect, rule

    def test_every_listed_rule_dispatches(self):
        """PRUNE_RULE_LIST is the contract, so every entry has to resolve."""
        for rule in PRUNE_RULE_LIST:
            prune_by_rule(rule, self._SIG, _make_tree_8(), _make_llr_8())

    def test_oracle_is_not_a_dispatchable_rule(self):
        """The oracle needs the target, so a recipe cannot name it."""
        assert 'oracle' not in PRUNE_RULE_LIST
        with pytest.raises(ValueError, match='prune rule must be'):
            prune_by_rule('oracle', self._SIG, _make_tree_8(),
                          _make_llr_8())

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match='prune rule must be'):
            prune_by_rule('nope', self._SIG, _make_tree_8(), _make_llr_8())

    def test_penalty_reaches_dp(self):
        """lam passed through the dispatcher still shrinks the selection."""
        children = _make_tree_8()
        llr = _make_llr_8()
        few, _ = prune_by_rule('dp', self._SIG, children, llr, lam=100.0)
        many, _ = prune_by_rule('dp', self._SIG, children, llr)
        assert len(few) <= len(many)

    def test_exp_n_eff_reaches_dp(self):
        """exp_n_eff is dp's own penalty, so it must agree with prune_dp."""
        children = _make_tree_8()
        llr = _make_llr_8()
        got, _ = prune_by_rule('dp', self._SIG, children, llr,
                               exp_n_eff=1.5)
        expect, _ = prune_dp(self._SIG, children, llr, exp_n_eff=1.5)
        assert got == expect

    @pytest.mark.parametrize('rule', ['greedy', 'maxllr'])
    @pytest.mark.parametrize('kwargs', [dict(lam=1.0),
                                        dict(exp_n_eff=2.0)])
    def test_penalty_on_a_non_dp_rule_raises(self, rule, kwargs):
        """A penalty no rule would read is an error, not a no-op."""
        with pytest.raises(ValueError, match='dp rule only'):
            prune_by_rule(rule, self._SIG, _make_tree_8(), _make_llr_8(),
                          **kwargs)

    def test_both_penalties_at_once_raises(self):
        """exp_n_eff sets lam, so naming both leaves one of them unread."""
        with pytest.raises(ValueError, match='alternatives'):
            prune_by_rule('dp', self._SIG, _make_tree_8(), _make_llr_8(),
                          lam=1.0, exp_n_eff=2.0)

    @pytest.mark.parametrize('exp_n_eff', [0.0, -1.0])
    def test_non_positive_exp_n_eff_raises(self, exp_n_eff):
        """It counts regions; log(1 + 1/n) is undefined at or below 0."""
        with pytest.raises(ValueError, match='must be'):
            prune_by_rule('dp', self._SIG, _make_tree_8(), _make_llr_8(),
                          exp_n_eff=exp_n_eff)

    def test_lam_zero_is_not_a_penalty(self):
        """dp's own default has to stay reachable from any rule's default."""
        check_prune_rule('greedy', lam=0.0, exp_n_eff=None)
        check_prune_rule('maxllr', lam=0.0, exp_n_eff=None)
