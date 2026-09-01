"""Detection scoring for the benchmark run functions.

score_effects compares a fitted Analysis's discovered EffectEstimates against
the planted EffectSynthetic target(s) and emits one JSON-friendly dict. run_ana
returns that dict as the run's output, so only it reaches disk while the heavy
fitted Analysis stays a local; it is the leaf of RECORDER.flatten_to_df, whose
exp ancestor chains back through the plant to the data build.

Also here: score_oracle_tree (segment), the min-size staircase scorers
(size_max_z_curve / curve_json), score_prune (the prune cache's region-index
scorer) and score_max_z_region (which region the max-z statistic came from).

score_effects output (one dict per fitted Analysis), keys:

    {num_vox, min_pval, n_pred,
     pred: [{reg_idx, num_vox, pval, target[, target0, target1, ...]}],
     target: {tp, fp, tn, fn},
     target0: {tp, fp, tn, fn}, ...}

num_vox is the analyzed voxel count, min_pval the smallest region p-value,
n_pred the count of discovered regions.

The target block scores the prediction (the union of the discovered regions)
against the union of the planted effects. target0/target1/... appear only with
several planted effects, scoring that same prediction against each in turn with
the others' support as background -- the cleaving / merge-cost signal. Each
pred region also reports how many of its voxels land in each target, so the
per-region geometry survives without the masks, which never reach disk.

Every metric (Dice, sensitivity, PPV, specificity) follows from the four
confusion counts (glow.mask.stats_from_counts), so only the counts are stored.
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
            fwer.pval ((num_reg,) array) the min_pval. reg_idx / pval_fwer are
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

    fwer = getattr(ana, 'fwer', None)
    pval = None if fwer is None else fwer.pval
    min_pval = (float(np.nanmin(pval))
                if pval is not None and np.isfinite(pval).any()
                else float('nan'))

    out = {'num_vox': int(mask_active.sum()),
           'min_pval': min_pval,
           'n_pred': len(pred_list)}

    # per-region geometry: each discovered region's size and how many of its
    # voxels land in each target (the union, then each effect when several)
    def overlap(region, target):
        """Count region voxels that fall inside target (within mask_active)."""
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


def _score_regions(reg_mask_list, mask_target_list, mask_active) -> dict:
    """Per-effect confusion scoring of a set of output regions.

    Unions the output region masks into one prediction, then scores it against
    each planted effect in turn. With a single planted effect the four counts
    are the bare tp/fp/tn/fn; with several they are suffixed by effect index
    (tp0/fp0/tn0/fn0 vs effect 0, etc.) -- each effect's counts treat the
    others' support as background. Dice / sensitivity / PPV derive downstream
    from the counts (glow.mask.stats_from_counts).

    Args:
        reg_mask_list (list): (reg_idx, mask) per output region, in output
            order; reg_idx is the Ward region index or None.
        mask_target_list (list): the planted supports, one (X, Y, Z) bool
            mask each (length 1 for the single-effect caches).
        mask_active (np.array): (X, Y, Z) bool, the analyzed voxels.

    Returns:
        {n_selected, tp, fp, tn, fn, pred}: the output-region count, the
            per-effect counts (suffixed by effect index when more than one),
            and the per-region pred block (_pred_records).
    """
    mask_pred = np.zeros(mask_active.shape, dtype=bool)
    for _, mask in reg_mask_list:
        mask_pred |= mask
    out = {'n_selected': len(reg_mask_list)}
    single = len(mask_target_list) == 1
    for i, mask_target in enumerate(mask_target_list):
        counts = glow.mask.confusion_counts(
            mask_pred=mask_pred, mask_target=mask_target,
            mask_active=mask_active)
        suffix = '' if single else str(i)
        out.update({f'{k}{suffix}': v for k, v in counts.items()})
    out['pred'] = _pred_records(reg_mask_list, mask_target_list, mask_active)
    return out


def _pred_records(reg_mask_list, mask_target_list, mask_active) -> list:
    """Record each output region's size and its overlap with the plant.

    The confusion counts union the output regions before scoring, which
    discards how the predicted volume was divided up. Keeping the per-region
    pair lets the structural scores (homogeneity and completeness, Rosenberg
    & Hirschberg 2007) be derived downstream off the records instead of a
    re-fit: with one planted effect, region r holds target voxels of the
    effect and num_vox - target of the null, which is the whole contingency
    table between the output regions and the true labels.

    Same record shape score_effects writes, less the pval that a pruning
    rule's selection has no per-region equivalent of.

    Args:
        reg_mask_list (list): (reg_idx, mask) per output region, in output
            order; reg_idx is the Ward region index or None.
        mask_target_list (list): the planted supports, one (X, Y, Z) bool
            mask each.
        mask_active (np.array): (X, Y, Z) bool, the analyzed voxels.

    Returns:
        one {reg_idx, num_vox, target} dict per region, target being the
        overlap with the union of the planted supports, plus target0..N when
        more than one effect is planted.
    """
    target_union = _union(mask_target_list, mask_active.shape)
    pred = []
    for reg_idx, mask in reg_mask_list:
        rec = {'reg_idx': None if reg_idx is None else int(reg_idx),
               'num_vox': int((mask & mask_active).sum()),
               'target': int((mask & target_union & mask_active).sum())}
        if len(mask_target_list) > 1:
            for i, tm in enumerate(mask_target_list):
                rec[f'target{i}'] = int((mask & tm & mask_active).sum())
        pred.append(rec)
    return pred


def score_prune(reg_out_list, children, mask_idx, mask_target_list,
                mask_active) -> dict:
    """Score a pruning rule's selected regions against the planted effect(s).

    The prune leaf keeps only region indices (masks are heavy), so the score
    is derived here: each selected region's (X, Y, Z) bool mask is rebuilt from
    its Ward index (glow.graph.get_label_map, as AnalysisGLOW.fit does),
    then _score_regions unions them and counts tp/fp/tn/fn vs the planted
    support, so dice / sens / ppv derive downstream like every arm. Used for
    the greedy / DP selections and the single max-LLR region.

    Args:
        reg_out_list (list): selected region indices (the rule's output, or
            [max-LLR region]); empty when nothing was selected.
        children (np.array): (num_reg - num_vox, 2) Ward child-index pairs.
        mask_idx (np.array): (X, Y, Z) int voxel-index array (-1 outside).
        mask_target_list (list): the planted (X, Y, Z) bool supports.
        mask_active (np.array): (X, Y, Z) bool, the analyzed voxels.

    Returns:
        {n_selected, tp, fp, tn, fn}: the _score_regions dict.
    """
    reg_mask_list = []
    for reg_idx in reg_out_list:
        label_map = glow.graph.get_label_map(
            reg_idx_list=[reg_idx], mask_idx=mask_idx, children=children)
        reg_mask_list.append((reg_idx, label_map > -1))
    return _score_regions(reg_mask_list, mask_target_list, mask_active)


def score_max_z_region(z_obs, reg_active, size, children, mask_idx,
                       mask_target_list, mask_active) -> dict:
    """Score the region the observed max-z statistic comes from.

    A max-stat test reports one number per fit; this says which region
    carried it and how well that region matches the plant, so a knob's
    effect on the argmax is readable at all (see run.run_inner_perm). The
    argmax runs over the same comparison set the max does
    (glow.analysis.fwer.max_over_active), so z is the fit's observed max-z.

    Args:
        z_obs (np.array): (num_reg,) observed z per region.
        reg_active (np.array): (num_reg,) boolean comparison set.
        size (np.array): (num_reg,) region sizes.
        children (np.array): (num_reg - num_vox, 2) Ward child-index pairs.
        mask_idx (np.array): (X, Y, Z) int voxel-index array (-1 outside).
        mask_target_list (list): the planted (X, Y, Z) bool supports.
        mask_active (np.array): (X, Y, Z) bool, the analyzed voxels.

    Returns:
        {reg_idx, num_vox, z, tp, fp, tn, fn}: the argmax region's index,
            voxel count, z, and confusion counts against the union of the
            planted supports. reg_idx None, num_vox 0, z NaN and
            all-background counts where no active region has a finite z.
    """
    z = np.where(reg_active & np.isfinite(z_obs), z_obs, np.nan)
    found = bool(np.isfinite(z).any())
    reg_idx = int(np.nanargmax(z)) if found else None
    if found:
        label_map = glow.graph.get_label_map(
            reg_idx_list=[reg_idx], mask_idx=mask_idx, children=children)
        mask_pred = label_map > -1
    else:
        mask_pred = np.zeros(mask_active.shape, dtype=bool)
    counts = glow.mask.confusion_counts(
        mask_pred=mask_pred,
        mask_target=_union(mask_target_list, mask_active.shape),
        mask_active=mask_active)
    return {'reg_idx': reg_idx,
            'num_vox': int(size[reg_idx]) if found else 0,
            'z': float(z[reg_idx]) if found else float('nan'),
            **counts}
