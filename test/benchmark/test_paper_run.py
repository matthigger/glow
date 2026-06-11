"""Schema tests for the paper trial scoring helpers (paper/run.py).

Each result row stores the four confusion counts (tp/fp/tn/fn); the overlap
metrics (dice/sens/ppv/spec) are derived at load by add_metric_cols. These
guard that contract at the scoring boundary -- _score_regions (shared by
run_segment / run_mancova / run_prune) and the load-time derivation -- without
paying for a full GLOW fit.
"""
import json

import numpy as np
import pandas as pd

import glow.benchmark
from glow.benchmark.paper.run import _score, _score_regions


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


def test_score_regions_emits_counts_not_metrics():
    mask_active, mask_target = _masks()
    # one region: both target voxels plus one false positive
    region = np.array([True, True, True, False, False, False])

    out = _score_regions([(7, region)], mask_target, mask_active)

    # the stored row is counts only; metrics are derived downstream
    assert {'tp', 'fp', 'tn', 'fn', 'n_selected',
            'effect_reg_json'}.issubset(out)
    assert not ({'dice', 'sens', 'ppv', 'spec'} & set(out))
    assert (out['tp'], out['fp'], out['fn'], out['tn']) == (2, 1, 0, 1)
    assert out['n_selected'] == 1

    # the diagnostic json carries each region's own reg_idx + tp/fp
    assert json.loads(out['effect_reg_json']) == \
        [{'reg_idx': 7, 'tp': 2, 'fp': 1}]


def test_score_regions_aggregate_is_the_union():
    # the aggregate counts come from the union of regions, not their sum:
    # two regions overlapping on a target voxel must not double-count it
    mask_active, mask_target = _masks()
    r1 = np.array([True, True, False, False, False, False])
    r2 = np.array([True, False, True, False, False, False])

    out = _score_regions([(1, r1), (2, r2)], mask_target, mask_active)

    # union predicts {0, 1, 2}: tp = {0, 1} = 2, fp = {2} = 1
    assert (out['tp'], out['fp']) == (2, 1)
    assert out['n_selected'] == 2


def test_score_adds_min_pval():
    mask_active, mask_target = _masks()
    region = np.array([True, True, False, False, False, False])
    ana = _FakeAna(effect_list=[_FakeEffect(3, region)],
                   pval=np.array([0.2, np.nan, 0.04]))

    out = _score(ana, mask_target, mask_active)

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
