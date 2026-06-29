"""Detection scoring for the benchmark run functions.

score_effects turns one fitted Analysis into the trial's score: it compares
the Analysis's discovered EffectEstimates against the planted EffectSynthetic
target(s) and emits one JSON-friendly dict. run_ana fits a recipe and calls
this on the result, returning (and so recording / caching) the dict as the
run's output -- the heavy fitted Analysis stays an in-memory local and is
discarded, so only the small score dict reaches disk. That dict is the leaf of
RECORDER.flatten_to_df: its exp ancestor chains back through the plant to the
data build, giving one score-bearing row per (trial, recipe). See
glow._extra.benchmark.run / recorder.

Ported from the deprecating paper layer (glow._extra.benchmark.paper.score):
score_oracle_tree (the segment cache's segmentation score) and the min-size
staircase scorers (size_max_z_curve / curve_json) now live here beside
score_effects.

score_effects output (one dict per fitted Analysis), keys:

    {num_vox, min_pval, n_pred,
     pred: [{reg_idx, num_vox, pval, target[, target0, target1, ...]}],
     target: {tp, fp, tn, fn},
     target0: {tp, fp, tn, fn}, ...}

num_vox is the analyzed voxel count (mask_active.sum()), min_pval the smallest
region p-value (nanmin(ana.pval)), n_pred the count of discovered regions. The
target0/target1/... blocks are present only with more than one target.

The single target block is always the prediction (the union of all discovered
regions) scored against the union of all planted effects -- so for one planted
effect it is that effect, and the bare tp/fp/tn/fn a reader flattens from it
stay backward compatible. The target0/target1/... blocks appear only with
several planted effects: they score the same prediction against each effect in
turn, the others' support treated as background (the cleaving / merge-cost
signal). Each pred region additionally reports how many of its voxels land in
each target, so the per-region geometry is visible without the masks (which
never reach disk; see recorder).

Every metric (Dice, sensitivity, PPV, specificity) is a function of the four
confusion counts and is derived downstream (glow.mask.stats_from_counts), so
only the counts are stored.
"""
import json

import numpy as np

import glow.graph
import glow.mask


def _union(mask_list, shape):
    """OR a list of bool masks into one; all-False on an empty list.

    Args:
        mask_list (list): (X, Y, Z) bool masks (possibly empty).
        shape (tuple): the (X, Y, Z) shape for the empty / accumulator mask.

    Returns:
        mask (np.array): (X, Y, Z) bool, the union of mask_list.
    """
    out = np.zeros(shape, dtype=bool)
    for m in mask_list:
        out |= m
    return out


def score_effects(ana, mask_target_list, mask_active) -> dict:
    """Score a fitted Analysis's effects against the planted target(s).

    The detection score shared by every effect-discovery cache. The
    prediction is the union of the Analysis's discovered EffectEstimate
    masks; it is scored (confusion counts; glow.mask.confusion_counts)
    against the union of the planted supports (the always-present "target"
    block) and, when more than one effect is planted, against each effect in
    turn ("target0", "target1", ...). Each discovered region additionally
    reports how many of its voxels fall inside each target. See the module
    docstring for the full output shape.

    Args:
        ana: a fitted Analysis. Its effect_list supplies the discovered
            regions (each EffectEstimate's mask, reg_idx, pval_fwer) and its
            pval ((num_reg,) array) the min_pval. reg_idx / pval_fwer are
            None for the voxel-wise methods (VBA / CET), recorded as such.
        mask_target_list (list): the planted effect supports, one (X, Y, Z)
            bool mask per EffectSynthetic; empty for the null calibration.
        mask_active (np.array): (X, Y, Z) bool, the analyzed voxels.

    Returns:
        the score dict: global num_vox / min_pval / n_pred, a per-region
        pred list, and the target (+ per-effect target0..N) confusion blocks.
    """
    shape = mask_active.shape
    pred_list = list(ana.effect_list or ())
    pred_masks = [eff.mask for eff in pred_list]
    pred_union = _union(pred_masks, shape)
    target_union = _union(mask_target_list, shape)
    n_target = len(mask_target_list)

    pval = getattr(ana, 'pval', None)
    min_pval = (float(np.nanmin(pval))
                if pval is not None and np.isfinite(pval).any()
                else float('nan'))

    out = {'num_vox': int(mask_active.sum()),
           'min_pval': min_pval,
           'n_pred': len(pred_list)}

    # per-region geometry: each discovered region's size and how many of its
    # voxels land in each target (the union, then each effect when several)
    def overlap(region, target):
        return int((region & target & mask_active).sum())

    pred = []
    for eff, m in zip(pred_list, pred_masks):
        rec = {'reg_idx': None if eff.reg_idx is None else int(eff.reg_idx),
               'num_vox': int((m & mask_active).sum()),
               'pval': (float(eff.pval_fwer)
                        if eff.pval_fwer is not None else float('nan')),
               'target': overlap(m, target_union)}
        if n_target > 1:
            for i, tm in enumerate(mask_target_list):
                rec[f'target{i}'] = overlap(m, tm)
        pred.append(rec)
    out['pred'] = pred

    # confusion of the whole prediction vs the union of targets (always),
    # then vs each planted effect when several were planted
    out['target'] = glow.mask.confusion_counts(
        mask_pred=pred_union, mask_target=target_union,
        mask_active=mask_active)
    if n_target > 1:
        for i, tm in enumerate(mask_target_list):
            out[f'target{i}'] = glow.mask.confusion_counts(
                mask_pred=pred_union, mask_target=tm, mask_active=mask_active)
    return out


def score_oracle_tree(children, mask_target, mask_idx) -> dict:
    """Return the confusion counts of the best-Dice region over a Ward tree.

    The segmentation-only (oracle) score: scan every tree region and keep the
    one whose Dice against the planted support is largest -- the best a perfect
    selector could do on this segmentation, no significance test or pruning.
    Used by the segment cache to isolate segmentation quality across Ward
    modes; Dice / sensitivity / PPV derive downstream from the counts.

    Args:
        children (np.array): (num_reg - num_vox, 2) Ward merge pairs.
        mask_target (np.array): (X, Y, Z) bool, the planted effect support.
        mask_idx (np.array): (X, Y, Z) int voxel-index array (-1 outside).

    Returns:
        {tp, fp, tn, fn}: the counts of the single best-matching region.
    """
    counts = glow.graph.confusion_counts_tree(
        mask=mask_target, mask_idx=mask_idx, children=children)
    dice = glow.mask.stats_from_counts(**counts)['dice']
    i = int(np.nanargmax(dice))
    return {k: int(counts[k][i]) for k in ('tp', 'fp', 'tn', 'fn')}


def size_max_z_curve(size, z, consider):
    """Build one outer perm's (size, max-z-at-size-or-larger) staircase.

    The max-z FWER null restricts the per-perm max to regions of size >=
    min_vox, so a min_vox sweep needs only E(m) = max{z_r : size_r >= m}, a
    non-increasing step function. This returns its corners: dedupe regions by
    size, take the running max from the largest size down, and keep the
    largest size of each distinct max-z. E(m) is then the value of the first
    corner with size >= m, so a handful of corners recovers the perm's max-z
    at any m >= floor without storing every region.

    Args:
        size (np.array): (num_reg,) region sizes
        z (np.array): (num_reg,) per-region z = (llr - mu) / std
        consider (np.array): (num_reg,) bool, regions eligible for the max
            (here size >= floor and z finite)

    Returns:
        curve (np.array): (L, 2) corners (size, max_z), ascending in size (so
            max_z descending); (0, 2) if none.
    """
    s, zz = size[consider], z[consider]
    if s.size == 0:
        return np.empty((0, 2))
    order = np.argsort(s)
    uniq, idx = np.unique(s[order], return_index=True)
    z_at = np.maximum.reduceat(zz[order], idx)
    suffix = np.maximum.accumulate(z_at[::-1])[::-1]
    # keep the largest size of each max-z plateau (right edge), so a
    # "first corner with size >= m" lookup returns the right value
    keep = np.append(np.diff(suffix) != 0, True)
    return np.column_stack([uniq, suffix])[keep]


def curve_json(curve_list) -> str:
    """Serialize the per-perm staircases to one results cell.

    Args:
        curve_list (list): one (L_k, 2) corner array per outer perm, index k
            matching the perm number (k=0 observed).

    Returns:
        a JSON string: a list (per perm) of [size, max_z] corner pairs.
    """
    return json.dumps([[[int(s), float(z)] for s, z in c] for c in curve_list])
