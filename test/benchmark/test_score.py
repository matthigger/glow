"""Schema + arithmetic tests for the paper scoring functions (paper/score.py).

score_effects turns one fitted Analysis into the trial's score dict: global
num_vox / min_pval / n_pred, a per-region pred list, and the confusion blocks
(always "target" = prediction vs the union of planted effects; "target0..N"
per effect when several are planted). These guard that shape and its counts
without paying for a real fit, by feeding a stand-in Analysis. score_oracle_tree
and the min-size curve helpers get a small hand-built tree / staircase.
"""
import json

import numpy as np

from glow.benchmark.paper.score import (score_effects, score_oracle_tree,
                                        size_max_z_curve, curve_json)


class _FakeEffect:
    """Stand-in for an EffectEstimate: a support mask + reg_idx + pval_fwer.

    reg_idx / pval_fwer are None for the voxel-wise methods (VBA / CET), so
    the scorer must tolerate either.
    """

    def __init__(self, mask, reg_idx=None, pval_fwer=None):
        self.mask = mask
        self.reg_idx = reg_idx
        self.pval_fwer = pval_fwer


class _FakeAna:
    """Stand-in for a fitted Analysis: just an effect_list and a pval array."""

    def __init__(self, effect_list, pval):
        self.effect_list = effect_list
        self.pval = pval


def _vec(*idx, n):
    """A length-n bool vector, True at the given indices."""
    v = np.zeros(n, dtype=bool)
    v[list(idx)] = True
    return v


def test_score_effects_single_target():
    # 20 analyzed voxels; one planted effect on 0..9; one discovered region on
    # 5..14 -> 5 voxels overlap, 5 false positives, 5 missed.
    mask_active = np.ones(20, dtype=bool)
    target = _vec(*range(0, 10), n=20)
    region = _vec(*range(5, 15), n=20)
    ana = _FakeAna([_FakeEffect(region, reg_idx=0, pval_fwer=0.02)],
                   pval=np.array([0.2, 0.02, np.nan]))

    out = score_effects(ana, [target], mask_active)

    assert out['num_vox'] == 20
    assert out['min_pval'] == 0.02
    assert out['n_pred'] == 1
    # one planted effect -> only the aggregate "target" block, no target0
    assert out['target'] == {'tp': 5, 'fp': 5, 'tn': 5, 'fn': 5}
    assert 'target0' not in out
    # counts only -- the overlap metrics are derived downstream
    assert not ({'dice', 'sens', 'ppv', 'spec'} & set(out['target']))
    assert out['pred'] == [{'reg_idx': 0, 'num_vox': 10, 'pval': 0.02,
                            'target': 5}]


def test_score_effects_multi_target_per_effect_blocks_and_overlaps():
    # two planted effects (0..9, 10..19) over 30 voxels; two discovered
    # regions -- one straddling both effects (5..14), one pure false (20..24).
    mask_active = np.ones(30, dtype=bool)
    target0 = _vec(*range(0, 10), n=30)
    target1 = _vec(*range(10, 20), n=30)
    region_a = _vec(*range(5, 15), n=30)
    region_b = _vec(*range(20, 25), n=30)
    ana = _FakeAna(
        [_FakeEffect(region_a, reg_idx=100, pval_fwer=0.01),
         _FakeEffect(region_b, reg_idx=200, pval_fwer=0.5)],
        pval=np.array([0.01, 0.5]))

    out = score_effects(ana, [target0, target1], mask_active)

    assert out['n_pred'] == 2
    assert out['min_pval'] == 0.01
    # prediction union = {5..14, 20..24} (15 vox). vs each effect (others as
    # background) and vs the union of both effects (0..19).
    assert out['target0'] == {'tp': 5, 'fp': 10, 'tn': 10, 'fn': 5}
    assert out['target1'] == {'tp': 5, 'fp': 10, 'tn': 10, 'fn': 5}
    assert out['target'] == {'tp': 10, 'fp': 5, 'tn': 5, 'fn': 10}
    # each discovered region reports its voxel overlap with the union and each
    # effect: region A splits 5/5 across the two effects, region B hits neither
    assert out['pred'][0] == {'reg_idx': 100, 'num_vox': 10, 'pval': 0.01,
                              'target': 10, 'target0': 5, 'target1': 5}
    assert out['pred'][1] == {'reg_idx': 200, 'num_vox': 5, 'pval': 0.5,
                              'target': 0, 'target0': 0, 'target1': 0}


def test_score_effects_null_uses_min_pval_only():
    # the null calibration plants nothing and (here) discovers nothing: the
    # only meaningful field is min_pval; the target block is all-background.
    mask_active = np.ones(10, dtype=bool)
    ana = _FakeAna(effect_list=[], pval=np.array([0.3, 0.5, np.nan]))

    out = score_effects(ana, [], mask_active)

    assert out['n_pred'] == 0
    assert out['pred'] == []
    assert out['min_pval'] == 0.3
    assert out['target'] == {'tp': 0, 'fp': 0, 'tn': 10, 'fn': 0}
    assert 'target0' not in out


def test_score_effects_tolerates_voxel_method_none_fields():
    # VBA / CET effects carry no reg_idx / pval_fwer -> recorded as None / nan
    mask_active = np.ones(8, dtype=bool)
    target = _vec(0, 1, n=8)
    region = _vec(0, 1, 2, n=8)
    ana = _FakeAna([_FakeEffect(region)], pval=np.array([np.nan, np.nan]))

    out = score_effects(ana, [target], mask_active)

    assert out['pred'][0]['reg_idx'] is None
    assert np.isnan(out['pred'][0]['pval'])
    # all-nan pval -> min_pval is nan (no active region scored)
    assert np.isnan(out['min_pval'])
    assert out['target'] == {'tp': 2, 'fp': 1, 'tn': 5, 'fn': 0}


def test_score_oracle_tree_picks_best_matching_region():
    # 4 leaves merged as {0,1}, {2,3}, then {0,1,2,3}; target is {0,1}, so the
    # oracle region is node 4 ({0,1}) with a perfect match.
    children = np.array([[0, 1], [2, 3], [4, 5]])
    mask_idx = np.array([0, 1, 2, 3])
    mask_target = _vec(0, 1, n=4)

    out = score_oracle_tree(children=children, mask_target=mask_target,
                            mask_idx=mask_idx)

    assert out == {'tp': 2, 'fp': 0, 'tn': 2, 'fn': 0}


def test_size_max_z_curve_keeps_step_corners_and_serializes():
    # three regions (size, z) = (1, .5), (2, .9), (3, .2). E(m)=max z over
    # size>=m is .9 for m<=2 then .2 at m=3, so the corners are (2,.9),(3,.2).
    size = np.array([1, 2, 3])
    z = np.array([0.5, 0.9, 0.2])
    consider = np.ones(3, dtype=bool)

    curve = size_max_z_curve(size, z, consider)
    assert curve.tolist() == [[2.0, 0.9], [3.0, 0.2]]

    # an empty consider yields an empty (0, 2) curve
    assert size_max_z_curve(size, z, np.zeros(3, dtype=bool)).shape == (0, 2)

    # curve_json round-trips the per-perm corner lists
    assert json.loads(curve_json([curve])) == [[[2, 0.9], [3, 0.2]]]
