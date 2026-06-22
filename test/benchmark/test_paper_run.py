"""Schema tests for the paper trial scoring helpers (paper/run.py).

Each result row stores the four confusion counts (tp/fp/tn/fn -- per planted
effect, suffixed 0/1/.. when there is more than one); the overlap metrics
(dice/sens/ppv/spec) are derived at load by add_metric_cols. These guard that
contract at the scoring boundary -- _score_regions (shared by run_ana /
run_mancova / run_prune / run_two_effect) and the load-time derivation --
without paying for a full GLOW fit.
"""
import numpy as np
import pandas as pd

import pytest

import glow.benchmark
from glow.benchmark.paper.run import _effect_llr, _plant, _score, _score_regions


class _FakeExp:
    """Stand-in for an experiment: just the mask_idx _plant reads for shape."""

    def __init__(self, mask_idx):
        self.mask_idx = mask_idx


class _FakeDS:
    """Stand-in for a data source: exposes a clean .exp."""

    def __init__(self, exp):
        self.exp = exp


class _FakeEffect:
    """Stand-in for an Analysis effect: a reg_idx and a support mask."""

    def __init__(self, reg_idx, mask):
        self.reg_idx = reg_idx
        self.mask = mask


class _FakeAna:
    """Stand-in for a fitted Analysis: just an effect_list and pval."""

    def __init__(self, effect_list, pval):
        self.effect_list = effect_list
        self.pval = pval


def _masks():
    # 6 "voxels" in a line, 4 analyzed; target is the first 2 analyzed voxels
    mask_active = np.array([True, True, True, True, False, False])
    mask_target = np.array([True, True, False, False, False, False])
    return mask_active, mask_target


def test_effect_llr_resolves_per_voxel_total_or_null():
    # per-voxel target passes through; total target divides by the extent
    assert _effect_llr(0.03, None, 614) == 0.03
    assert _effect_llr(None, 18.42, 614) == 18.42 / 614
    # neither knob set -> None, the null / FWER-calibration signal (NOT 0)
    assert _effect_llr(None, None, 614) is None
    # both knobs set is a misconfiguration
    with pytest.raises(ValueError):
        _effect_llr(0.03, 18.42, 614)


def test_plant_none_leaves_data_untouched():
    # effect_llr=None must plant nothing: the same exp back (no scrubbing of
    # incidental effect, as effect_llr=0 would do) and an all-False target.
    exp = _FakeExp(mask_idx=np.arange(6).reshape(2, 3))
    ds = _FakeDS(exp)

    out_exp, out_eff, mask = _plant(ds, extenter=None, effect_llr=None)

    assert out_exp is exp
    assert out_eff is exp
    assert mask.shape == exp.mask_idx.shape
    assert mask.dtype == bool
    assert not mask.any()


def test_score_regions_emits_counts_not_metrics():
    mask_active, mask_target = _masks()
    # one region: both target voxels plus one false positive
    region = np.array([True, True, True, False, False, False])

    out = _score_regions([(7, region)], [mask_target], mask_active)

    # the stored row is counts only; metrics are derived downstream. A single
    # planted effect leaves the four counts unsuffixed.
    assert {'tp', 'fp', 'tn', 'fn', 'n_selected'}.issubset(out)
    assert not ({'dice', 'sens', 'ppv', 'spec'} & set(out))
    assert (out['tp'], out['fp'], out['fn'], out['tn']) == (2, 1, 0, 1)
    assert out['n_selected'] == 1


def test_score_regions_aggregate_is_the_union():
    # the aggregate counts come from the union of regions, not their sum:
    # two regions overlapping on a target voxel must not double-count it
    mask_active, mask_target = _masks()
    r1 = np.array([True, True, False, False, False, False])
    r2 = np.array([True, False, True, False, False, False])

    out = _score_regions([(1, r1), (2, r2)], [mask_target], mask_active)

    # union predicts {0, 1, 2}: tp = {0, 1} = 2, fp = {2} = 1
    assert (out['tp'], out['fp']) == (2, 1)
    assert out['n_selected'] == 2


def test_score_regions_suffixes_counts_per_effect():
    # two planted effects -> one set of counts per effect, suffixed 0/1; each
    # effect scores the prediction with the other effect treated as background
    mask_active = np.array([True, True, True, True, False, False])
    mask0 = np.array([True, False, False, False, False, False])  # voxel 0
    mask1 = np.array([False, True, False, False, False, False])  # voxel 1
    region = np.array([True, True, True, False, False, False])   # predicts 0,1,2

    out = _score_regions([(7, region)], [mask0, mask1], mask_active)

    # no unsuffixed counts when there is more than one effect
    assert not ({'tp', 'fp', 'tn', 'fn'} & set(out))
    # vs effect0: tp={0}=1, fp={1,2}=2, fn=0, tn={3}=1
    assert (out['tp0'], out['fp0'], out['fn0'], out['tn0']) == (1, 2, 0, 1)
    # vs effect1: tp={1}=1, fp={0,2}=2, fn=0, tn={3}=1
    assert (out['tp1'], out['fp1'], out['fn1'], out['tn1']) == (1, 2, 0, 1)
    assert out['n_selected'] == 1


def test_score_adds_min_pval():
    mask_active, mask_target = _masks()
    region = np.array([True, True, False, False, False, False])
    ana = _FakeAna(effect_list=[_FakeEffect(3, region)],
                   pval=np.array([0.2, np.nan, 0.04]))

    out = _score(ana, [mask_target], mask_active)

    assert out['min_pval'] == 0.04
    assert (out['tp'], out['fp']) == (2, 0)


def test_add_metric_cols_round_trip():
    # a counts-only row gains dice/sens/ppv/spec at load time
    df = pd.DataFrame([{'tp': 2, 'fp': 1, 'tn': 1, 'fn': 0}])

    out = glow.benchmark.add_metric_cols(df)

    assert np.isclose(out['dice'].iloc[0], 2 * 2 / (2 * 2 + 1 + 0))
    assert np.isclose(out['sens'].iloc[0], 1.0)
    assert np.isclose(out['ppv'].iloc[0], 2 / 3)
    assert np.isclose(out['spec'].iloc[0], 1 / 2)
