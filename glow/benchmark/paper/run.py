"""Trial functions for the paper benchmarks (recorder-wired).

Every trial fn takes the recorder as its first argument and only
scalar/categorical axes after it (source, b, num_img, n_vox_eff, seed,
effect strength), rebuilding the heavy DataSource / Extenter objects via
factory.build_ds so iter_kwargs stays a flat scalar grid. The fns return
nothing: each wraps its steps with the recorder at the call site (no
decorators), so a trial's provenance (the imposed Experiment + planted
effects), per-method timing, any failure, and the score land in the
records, grouped under the trial's cache hash. Scoring is recorded as its
own step (run_ana / run_segment) or derived afterward from the records
(run_mancova / run_prune / run_two_effect). build_ds raises on an
infeasible cell (e.g. HCP b > pool); the setup step records that failure
and swallows it, so the trial ends without a try/except (see recorder).

Effect strength is given one of two ways, resolved by _effect_llr:
  - effect_llr: the per-voxel (size-normalized) target, held fixed when a
    structural axis (b, num_img) is swept.
  - effect_total_llr: the whole-region target; the per-voxel llr is
    effect_total_llr / n_vox_eff, so the total stays fixed as the extent
    grows (the extent sweep's choice).

The trial kinds (each registered with its analysis recipe in config.py):
  - run_ana: fit and score every entry of an ana_kwargs_dict.
  - run_segment: oracle-Dice of the best-matching region in each Ward
    hierarchy (segmentation quality, no significance test or pruning).
  - run_mancova: one shared voxel-stat walk dispatched across VBA /
    VBA-TFCE / CET x 5 MANCOVA stats x {raw, z}.
  - run_prune: greedy vs DP pruning on one shared GLOW fit.
  - run_two_effect: two adjacent equal-LLR effects at a controlled angle.
  - run_min_size: per-perm (size -> max-z) staircases for a min_vox sweep.
"""
import numpy as np

import glow
import glow.graph
import glow.mask
from glow.analysis import (
    AnalysisGLOW, AnalysisVoxel, AnalysisVBA, AnalysisCET,
    DEFAULT_CET_CFT_PVAL, inner_perm)
from glow.analysis.cluster import cluster, ClusterMode
from glow.analysis.mancova import stat_dict, stat_dict_inv
from glow.analysis.prune import prune_greedy, prune_dp
from glow.effect import (EffectSynthetic, ExtenterMinVar, ExtenterSphere,
                         ExtenterSplit)
from .factory import build_ds_for_seed
from .score import (score_effects, score_oracle_tree, size_max_z_curve,
                    curve_json)


def _effect_llr(effect_llr, effect_total_llr, n_vox_eff: int):
    """Resolve the per-voxel effect_llr from whichever knob the cache set.

    Args:
        effect_llr (float | None): per-voxel target (used as-is)
        effect_total_llr (float | None): whole-region target; divided by
            n_vox_eff to hold the total fixed as extent varies
        n_vox_eff (int): requested effect support size

    Returns:
        the per-voxel effect_llr to plant, or None when neither knob is set
        -- the null / FWER-calibration path, which plants no effect (see
        _plant). Note effect_llr=0 is NOT this path: 0 is a real target the
        offset solver hits by scrubbing the region's incidental effect.

    Raises:
        ValueError: if both knobs are given
    """
    if effect_llr is not None and effect_total_llr is not None:
        raise ValueError('pass at most one of effect_llr / effect_total_llr')
    if effect_total_llr is not None:
        return float(effect_total_llr) / n_vox_eff
    if effect_llr is not None:
        return float(effect_llr)
    return None


def _plant(ds, extenter, effect_llr):
    """Build the exp + planted-effect pair shared by every trial fn.

    effect_llr is None for the null / FWER-calibration path: plant nothing
    and return the data untouched with an empty target mask. effect_llr=0 is
    deliberately NOT this case -- compute_offset would solve for the offset
    that drives the sampled region's natural LLR to exactly zero, scrubbing
    any incidental effect out of it. A calibration on the global null wants
    the data left exactly as the source produced it, so the null cache
    passes None (see config._cache('null', ...)), not 0.

    The effect seed lives in the extenter (it carries its own seed), so the
    planted support is a pure function of the passed extenter and ds.exp.

    Args:
        ds: data source whose .exp gives the clean experiment
        extenter: Extenter (seed baked in) sampling the planted support
        effect_llr (float | None): per-voxel planted effect strength, or
            None to plant no effect (null calibration)

    Returns:
        exp: the clean experiment
        exp_eff: the experiment with the synthetic effect added (exp itself,
            unmodified, when effect_llr is None)
        mask_ (np.array): (X, Y, Z) bool, the realized effect support
            (all-False when effect_llr is None)
    """
    exp = ds.exp
    if effect_llr is None:
        return exp, exp, np.zeros(exp.mask_idx.shape, dtype=bool)
    exp_eff, mask = EffectSynthetic(extenter=extenter,
                                    effect_llr=effect_llr).fit(exp)
    return exp, exp_eff, mask


def _score_regions(reg_mask_list, mask_target_list, mask_active) -> dict:
    """Per-effect confusion scoring of a method's output regions.

    Shared by _score (a fitted Analysis's effect_list), run_prune (each
    pruning rule's selected regions) and the two-effect trial: unions the
    output region masks into one prediction, then scores that prediction
    against each planted effect in turn. With a single planted effect the
    four counts are the bare tp/fp/tn/fn; with several they are suffixed by
    effect index (tp0/fp0/tn0/fn0 vs effect 0, tp1/.. vs effect 1, ...) --
    each effect's counts treat the others' support as background. Dice,
    sensitivity, PPV and specificity are derived from the counts at load
    time (glow.mask.stats_from_counts).

    Args:
        reg_mask_list (list): (reg_idx, mask) per output region, in output
            order; reg_idx is the Ward region index or None (voxel methods)
        mask_target_list (list): the planted effect supports, one (X, Y, Z)
            bool mask each (length 1 for the single-effect trials)
        mask_active (np.array): (X, Y, Z) bool, the analyzed voxels

    Returns:
        the per-effect tp/fp/tn/fn counts (unsuffixed for one effect, else
        suffixed by effect index) plus n_selected (output-region count)
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
    return out


def _score_prune(reg_out_list, children, mask_idx, mask_target_list,
                 mask_active) -> dict:
    """Score a pruning rule's selected regions against the planted effect(s).

    The prune records keep only region indices (the masks are heavy), so the
    score is derived at fit time: each selected region's (X, Y, Z) bool mask is
    rebuilt from its Ward index (glow.graph.get_label_map, exactly as
    AnalysisGLOW.finalize does), then the shared per-effect confusion scoring
    (_score_regions) unions them and counts tp/fp/tn/fn vs the planted support
    -- so dice / sens / ppv derive at load time like every other arm. Used for
    the greedy / DP selections and for the single max-LLR region.

    Args:
        reg_out_list (list): selected Ward region indices (the pruning rule's
            output, or [max-LLR region] for the max-LLR arm); empty when the
            rule selected nothing (no FWER-significant region).
        children (np.array): (num_reg - num_vox, 2) Ward child-index pairs
        mask_idx (np.array): (X, Y, Z) int voxel-index array (-1 outside)
        mask_target_list (list): the planted effect supports, one (X, Y, Z)
            bool mask each (length 1 for the single-effect prune trials)
        mask_active (np.array): (X, Y, Z) bool, the analyzed voxels

    Returns:
        the _score_regions dict (per-effect tp/fp/tn/fn, n_selected)
    """
    reg_mask_list = []
    for reg_idx in reg_out_list:
        label_map = glow.graph.get_label_map(
            reg_idx_list=[reg_idx], mask_idx=mask_idx, children=children)
        reg_mask_list.append((reg_idx, label_map > -1))
    return _score_regions(reg_mask_list, mask_target_list, mask_active)


def _score(ana, mask_target_list, mask_active) -> dict:
    """Score a fitted Analysis's effect_list against the planted effect(s).

    The per-effect confusion scoring of _score_regions (run by every
    Analysis trial) plus min_pval, the smallest region p-value reported.

    Args:
        ana: a fitted Analysis whose .effect_list / .pval are scored
        mask_target_list (list): the planted effect supports, one (X, Y, Z)
            bool mask each (length 1 for the single-effect trials)
        mask_active (np.array): (X, Y, Z) bool, voxels inside the mask

    Returns:
        the _score_regions dict (per-effect tp/fp/tn/fn, n_selected) plus
        min_pval
    """
    reg_mask_list = [(getattr(eff, 'reg_idx', None), eff.mask)
                     for eff in (ana.effect_list or ())]
    scored = _score_regions(reg_mask_list, mask_target_list, mask_active)
    scored['min_pval'] = (float(np.nanmin(ana.pval))
                          if getattr(ana, 'pval', None) is not None
                          else np.nan)
    return scored


def _plant_effect(exp, *, n_vox_eff: int, seed: int, llr):
    """Plant one ExtenterMinVar effect on exp; the planting tail of _setup_trial.

    Given the clean experiment, the support size, the effect sub-seed, and the
    resolved per-voxel llr, plant one synthetic effect (or nothing for the null
    path) and return the recorder-friendly (exp_eff, effect_list,
    mask_target_list) triple.

    Args:
        exp: the clean experiment to impose the effect on.
        n_vox_eff (int): requested effect support size.
        seed (int): the effect RNG seed (ExtenterMinVar support + direction).
        llr (float | None): resolved per-voxel effect target, or None to plant
            no effect (the null / FWER-calibration path).

    Returns:
        exp_eff: the experiment with the effect imposed (exp itself when llr
            is None).
        effect_list (list): the planted EffectSynthetic specs (empty when llr
            is None).
        mask_target_list (list): each planted effect's realized (X, Y, Z) bool
            support, in effect_list order (empty when llr is None).
    """
    if llr is None:
        return exp, [], []
    effect = EffectSynthetic(
        extenter=ExtenterMinVar(n_vox=n_vox_eff, seed=seed), effect_llr=llr)
    exp_eff, mask = effect.fit(exp)
    return exp_eff, [effect], [mask]


def _setup_trial(*, source: str, b: int, num_img: int, n_vox_eff: int,
                 seed: int, effect_llr=None, effect_total_llr=None):
    """Build one planted-effect trial: the imposed experiment and its effects.

    A plain function, wrapped with the recorder at the call site (run_ana) like
    every other recorded step -- the experiment + planted effects it returns
    capture the trial's data lineage through their to_record(). build_ds raises
    on an infeasible cell (e.g. HCP b > pool); the recorder records that failure
    and swallows it, so the caller sees None and ends the trial without a
    try/except.

    The trial seed is split (derive_seeds) into mutually independent DataSource /
    feature / effect sub-seeds, so each seed is its own data realization -- an
    independent WGN noise field (or HCP crop + design draw) -- with the planted
    effect kept independent of that realization. Without the split a sweep's
    seeds would share one base experiment (the DataSource seed pinned to
    DS_SEED), leaving the per-seed replicates correlated; the ds sub-seed gives
    each its own draw. The sub-seeds are a deterministic function of the trial
    seed, so every effect_llr / method trial of one seed still reuses that
    seed's single memoised ds build.

    Args:
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count
        num_img (int): subject count (WGN; HCP uses its cohort)
        n_vox_eff (int): requested effect support size
        seed (int): the trial seed, split by derive_seeds into independent
            DataSource, feature (HCP subset), and effect sub-seeds
        effect_llr (float | None): per-voxel effect target
        effect_total_llr (float | None): whole-region effect target

    Returns:
        exp_eff: the experiment with the synthetic effect imposed (the clean
            experiment unchanged when no effect is planted).
        effect_list (list): the planted EffectSynthetic specs (empty for the
            null / FWER-calibration path). One today; this factory will grow to
            plant several.
        mask_target_list (list): the realized (X, Y, Z) bool support of each
            planted effect, in effect_list order (empty for the null path).
            Kept alongside the recorded specs so the score step (score_effects)
            scores the prediction against the actual planted voxels.
    """
    ds, effect_seed = build_ds_for_seed(source, b=b, num_img=num_img, seed=seed)
    llr = _effect_llr(effect_llr, effect_total_llr, n_vox_eff)
    return _plant_effect(ds.exp, n_vox_eff=n_vox_eff, seed=effect_seed, llr=llr)


def run_ana(recorder, *, source: str, b: int, num_img: int, n_vox_eff: int,
            seed: int, ana_kwargs_dict: dict, effect_llr=None,
            effect_total_llr=None):
    """Fit and score every analysis in ana_kwargs_dict on one synthetic trial.

    The recorder is passed in (by the driver, which got it from the TrialCache
    that scoped this trial) and every step is wrapped with it at the call site
    -- no decorators: _setup_trial records the experiment + planted effects for
    provenance, then for each analysis a fit step records self (the recipe,
    which identifies the variant), its timing, and any traceback, and a score
    step records that fit's detection score against the planted effects
    (score_effects). Both the fit and score steps are tagged with the
    ana_kwargs_dict label (e.g. 'GLOW-GLM'), so the records carry the method
    name and a fit pairs with its score on (trial_id, label). All share one
    trial id (the cache hash, set by the iterator's scope). A fit that fails
    marks the trial failed, so its score step (and every later fit/score)
    short-circuits unrecorded (see recorder).

    Args:
        recorder (Recorder): the trial's recorder (its trial scope is already
            open); calls are wrapped with it.
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count
        num_img (int): subject count (WGN; HCP uses its cohort)
        n_vox_eff (int): requested effect support size
        seed (int): effect RNG seed (also selects the HCP feature subset)
        ana_kwargs_dict (dict): label -> (Analysis class, init kwargs)
        effect_llr (float | None): per-voxel effect target
        effect_total_llr (float | None): whole-region effect target
    """
    # build_ds failure is recorded and swallowed -> setup is None -> end here
    setup = recorder(output_name_list=(
        'exp_eff', 'effect_list', 'mask_target_list'))(_setup_trial)(
        source=source, b=b, num_img=num_img, n_vox_eff=n_vox_eff, seed=seed,
        effect_llr=effect_llr, effect_total_llr=effect_total_llr)
    if setup is None:
        return
    exp_eff, _effect_list, mask_target_list = setup
    mask_active = exp_eff.mask_idx > -1

    for label, (Ana, kw) in ana_kwargs_dict.items():
        ana = recorder(output_name='ana', label=label)(Ana(exp=exp_eff, **kw).fit)()
        recorder(output_name='score', label=label)(score_effects)(
            ana=ana, mask_target_list=mask_target_list, mask_active=mask_active)


# ---------------------------------------------------------------------------
# min-size sweep: capture the per-perm (size -> max-z) staircase

# A large seed offset reused to keep distinct seed regimes apart: outer-perm
# k draws its inner FL perms from the block starting at
# (k + 1) * _SEED_OFFSET_DISTINCT (matching AnalysisGLOW's per-outer-perm
# spacing), and config offsets the min_size trial seeds by it so they sit
# clear of the other sweeps' seeds.
_SEED_OFFSET_DISTINCT = 100_000


def _min_size_curves(exp_eff, *, n_perm_fwer: int, n_perm_inner: int,
                     min_vox_floor: int, cluster_mode) -> str:
    """Capture each outer perm's (size -> max-z) staircase.

    The min_size sweep's compute step. Borrows AnalysisGLOW for its scaling +
    (q0, q1) decomposition (so the captured curves match a real fit), then runs
    the outer-perm loop by hand with the exact inner_perm.cpu_perm kernel,
    recording per perm the size_max_z_curve staircase. With those curves
    GLOW's max-z FWER null -- hence its rejection / power -- can be recomputed at
    any min_vox >= min_vox_floor without re-fitting (the curve at the fit-time
    min_vox reproduces AnalysisGLOW.max_z_null exactly).

    inner_perm.cpu_perm gives every region >= min_vox_floor an exact z, so the
    null can be re-thresholded at any min_vox the sweep visits without bias.

    Args:
        exp_eff: the experiment with the synthetic effect imposed.
        n_perm_fwer (int): outer FL perms (n_perm_fwer + 1 curves, incl. k=0).
        n_perm_inner (int): inner FL draws per outer perm.
        min_vox_floor (int): smallest region size given a z; the sweep's lower
            bound. 1 keeps the whole range available.
        cluster_mode (ClusterMode): Ward projection.

    Returns:
        a JSON string of the per-perm [size, max_z] corner staircases
        (curve_json; parse with json.loads).
    """
    # borrow AnalysisGLOW only for its scaling + (q0, q1) decomposition, so
    # the captured curves match a real fit; the outer loop below is run by
    # hand to swap the racing kernel for the exact cpu_perm.
    ana = AnalysisGLOW(exp=exp_eff, n_perm_fwer=n_perm_fwer,
                       n_perm_inner=n_perm_inner, min_vox=min_vox_floor,
                       cluster_mode=cluster_mode)
    exp_s, q0, q1 = ana.exp, ana._q0, ana._q1

    curve_list = []
    for k in range(n_perm_fwer + 1):
        _exp = exp_s.permute(k) if k else exp_s
        children = cluster(_exp, mode=cluster_mode)
        llr_k, size = glow.graph.compute_llr_batched(
            _exp, children=children, q0=q0, q1=q1)
        mu, std = inner_perm.cpu_perm(
            exp=_exp, base_seed=(k + 1) * _SEED_OFFSET_DISTINCT,
            n_perm=n_perm_inner, q0=q0, q1=q1, children=children,
            min_vox=min_vox_floor)
        std_safe = np.where(std < 1e-12, 1.0, std)
        z = np.nan_to_num((llr_k - mu) / std_safe,
                          nan=0.0, posinf=0.0, neginf=np.nan)
        consider = (size >= min_vox_floor) & np.isfinite(z)
        curve_list.append(size_max_z_curve(size, z, consider))

    return curve_json(curve_list)


def run_min_size(recorder, *, source: str, b: int, num_img: int,
                 n_vox_eff: int, seed: int, n_perm_fwer: int,
                 n_perm_inner: int, min_vox_floor: int = 1,
                 cluster_mode=ClusterMode.FOCUS, effect_llr=None,
                 effect_total_llr=None):
    """Capture each outer perm's (size -> max-z) curve for a min_vox sweep.

    Records, it does not score. _setup_trial plants one synthetic effect on an
    independent data realization (recorded for provenance; the trial seed is
    split per source, so each seed is its own null), then a single curve step
    records the per-perm (size -> max-z) staircases (_min_size_curves ->
    curve_json) plus its inputs (n_perm_fwer / n_perm_inner / min_vox_floor /
    cluster_mode) and timing, tagged with the 'GLOW' method label (the cache's
    one method, so the records-to-csv reader keys on (trial_id, 'GLOW')). With
    those, GLOW's max-z FWER null can be swept over min_vox post hoc without
    re-fitting. Sweeping itself is derived afterward from the records, not here.

    Args:
        recorder (Recorder): the trial's recorder; calls are wrapped with it.
        source (str): 'wgn' or 'hcp' (the cache registers 'hcp').
        b (int): imaging-feature count.
        num_img (int): subject count (WGN; HCP uses its cohort).
        n_vox_eff (int): requested effect support size.
        seed (int): trial seed -- split by derive_seeds into independent
            DataSource, feature, and effect sub-seeds.
        n_perm_fwer (int): outer FL perms (n_perm_fwer + 1 curves, incl. k=0).
        n_perm_inner (int): inner FL draws per outer perm.
        min_vox_floor (int): smallest region size given a z; the sweep's lower
            bound. 1 keeps the whole range available.
        cluster_mode (ClusterMode): Ward projection (default FOCUS).
        effect_llr (float | None): per-voxel effect target.
        effect_total_llr (float | None): whole-region effect target.
    """
    setup = recorder(output_name_list=(
        'exp_eff', 'effect_list', 'mask_target_list'))(_setup_trial)(
        source=source, b=b, num_img=num_img, n_vox_eff=n_vox_eff, seed=seed,
        effect_llr=effect_llr, effect_total_llr=effect_total_llr)
    if setup is None:
        return
    exp_eff, _effect_list, _mask_target_list = setup

    recorder(output_name='curve', label='GLOW')(_min_size_curves)(
        exp_eff=exp_eff, n_perm_fwer=n_perm_fwer, n_perm_inner=n_perm_inner,
        min_vox_floor=min_vox_floor, cluster_mode=cluster_mode)


def _segment_oracle(exp_eff, mode, mask_target_list) -> dict:
    """Best-Dice tree region for one ClusterMode: the segment trial's score.

    Builds the Ward hierarchy in `mode` on the effect-bearing images and
    returns score_oracle_tree's confusion counts for the planted support -- the
    best a perfect selector could do on this segmentation, with no significance
    test or pruning. The mode and the experiment are the recorded inputs (its
    provenance); the timing is the segmentation + tree-scan cost.

    Args:
        exp_eff: the experiment with the synthetic effect imposed.
        mode (ClusterMode): the Ward projection to segment with.
        mask_target_list (list): the planted (X, Y, Z) bool supports (one for
            the segment cache); their union is the target scored (all-False,
            hence empty counts, for the null path).

    Returns:
        the {tp, fp, tn, fn} counts of the single best-matching tree region.
    """
    mask_target = np.zeros(exp_eff.mask_idx.shape, dtype=bool)
    for m in mask_target_list:
        mask_target |= m
    children = cluster(exp_eff, mode=mode)
    return score_oracle_tree(children=children, mask_target=mask_target,
                             mask_idx=exp_eff.mask_idx)


def run_segment(recorder, *, source: str, b: int, num_img: int, n_vox_eff: int,
                seed: int, modes, effect_llr=None, effect_total_llr=None):
    """Oracle Dice of the best region in each Ward hierarchy, per mode.

    Isolates segmentation quality from significance testing and pruning.
    _setup_trial plants one synthetic effect (recorded for provenance), then for
    each ClusterMode a score step records the oracle best-Dice region of that
    Ward tree (_segment_oracle): the maximum Dice over all regions, unavailable
    in practice but a clean measure of how well the segmentation alone recovers
    the planted support. The mode is both the score step's recorded input and
    its method label (e.g. 'Focus'), so the records-to-csv reader keys on
    (trial_id, mode).

    Args:
        recorder (Recorder): the trial's recorder; calls are wrapped with it.
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count
        num_img (int): subject count (WGN; HCP uses its cohort)
        n_vox_eff (int): requested effect support size
        seed (int): effect RNG seed (also selects the HCP feature subset)
        modes: iterable of ClusterMode (or their string values) to compare
        effect_llr (float | None): per-voxel effect target
        effect_total_llr (float | None): whole-region effect target
    """
    setup = recorder(output_name_list=(
        'exp_eff', 'effect_list', 'mask_target_list'))(_setup_trial)(
        source=source, b=b, num_img=num_img, n_vox_eff=n_vox_eff, seed=seed,
        effect_llr=effect_llr, effect_total_llr=effect_total_llr)
    if setup is None:
        return
    exp_eff, _effect_list, mask_target_list = setup

    for mode in modes:
        mode = ClusterMode(mode)
        recorder(output_name='score', label=str(mode))(_segment_oracle)(
            exp_eff=exp_eff, mode=mode, mask_target_list=mask_target_list)


def run_prune(recorder, *, source: str, b: int, num_img: int, n_vox_eff: int,
              seed: int, effect_llr=None, effect_total_llr=None):
    """Greedy vs DP pruning scored on one shared GLOW fit.

    Fits GLOW once (recorded -- self is the recipe), then applies both pruning
    rules to the SAME FWER-significant region set, so the comparison isolates
    the rule from the permutation test. Both rank candidates by raw LLR
    (mirrors AnalysisGLOW.finalize; the z-score fragments under pruning).
    prune_greedy blooms the max-LLR region and removes its tree relatives
    (GLOW's default; undersegments); prune_dp takes the exact max-total-LLR
    antichain (oversegments). Each prune call is recorded and tagged with its
    rule label (GLOW-Greedy / GLOW-DP), its output the selected regions, then a
    score step records that selection's detection counts (_score_prune) against
    the planted effect. A third GLOW-MaxLLR arm scores the single highest-LLR
    significant region on its own -- the headline "best region" dice / sens /
    ppv (n_selected = 1), what greedy blooms first.

    Args:
        recorder (Recorder): the trial's recorder; calls are wrapped with it.
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count
        num_img (int): subject count (WGN; HCP uses its cohort)
        n_vox_eff (int): requested effect support size
        seed (int): effect RNG seed (also selects the HCP feature subset)
        effect_llr (float | None): per-voxel effect target
        effect_total_llr (float | None): whole-region effect target
    """
    setup = recorder(output_name_list=(
        'exp_eff', 'effect_list', 'mask_target_list'))(_setup_trial)(
        source=source, b=b, num_img=num_img, n_vox_eff=n_vox_eff, seed=seed,
        effect_llr=effect_llr, effect_total_llr=effect_total_llr)
    if setup is None:
        return
    exp_eff, _effect_list, mask_target_list = setup
    mask_active = exp_eff.mask_idx > -1

    # GLOW's default recipe, shared with the GLOW-Focus arm so both pruning
    # rules prune the regions GLOW would actually report. Deferred import:
    # config imports this module, so a module-level import would be circular.
    from .config import _GLOW_BASE
    glow_kwargs = {**_GLOW_BASE, 'cluster_mode': ClusterMode.FOCUS}

    ana = recorder(output_name='ana')(
        AnalysisGLOW(exp=exp_eff, **glow_kwargs).fit)()
    if ana is None:
        return

    # the FWER-significant regions both rules prune, ranked by raw LLR
    # (mirrors AnalysisGLOW.finalize; the z-score fragments under pruning)
    sig_reg_list = np.where(ana.pval <= ana.alpha_fwer)[0].tolist()
    llr_gain = np.nan_to_num(ana.llr.astype(float), nan=0.0,
                             posinf=0.0, neginf=0.0)

    def score(reg_out_list, label):
        recorder(output_name='score', label=label)(_score_prune)(
            reg_out_list=reg_out_list, children=ana.children,
            mask_idx=exp_eff.mask_idx, mask_target_list=mask_target_list,
            mask_active=mask_active)

    # the single highest-LLR significant region -- what greedy blooms first,
    # scored on its own as the headline "best region" detection quality
    max_llr_reg = (max(sig_reg_list, key=lambda r: llr_gain[r])
                   if sig_reg_list else None)
    score([] if max_llr_reg is None else [max_llr_reg], 'GLOW-MaxLLR')

    for prune_fn, label in ((prune_greedy, 'GLOW-Greedy'), (prune_dp, 'GLOW-DP')):
        out = recorder(output_name_list=('reg_out_list', 'prune_info'),
                       label=label)(prune_fn)(
            sig_reg_list=sig_reg_list, children=ana.children, stat=llr_gain)
        if out is None:
            continue
        reg_out_list, _info = out
        score(reg_out_list, label)


# ---------------------------------------------------------------------------
# MANCOVA stat comparison
# ---------------------------------------------------------------------------
# This matrix is VBA / VBA-TFCE / CET only, by design rather than omission.
# GLOW uses the LLR throughout (it falls out of the cluster-level normal
# likelihood; see the manuscript's normal-identity appendix), so it is not
# swept across the classical MANCOVA stats here. Pinning GLOW to any single
# such stat could only understate its advantage over the voxel-wise methods,
# so the bake-off is among those methods.

def _shared_voxel_walk(exp_eff, n_perm_fwer: int) -> dict:
    """One stat matrix per MANCOVA stat, keyed by stat name, shared across families.

    Keyed by the stat's name (not the function) so the result is JSON-friendly
    -- the recorder serialises it as a step output (each matrix as a content
    hash); run_mancova indexes it by stat_dict_inv[fn].

    Args:
        exp_eff: the experiment with the synthetic effect added
        n_perm_fwer (int): number of FWER permutations

    Returns:
        stat name -> (n_perm_fwer + 1, num_vox) array; row 0 is observed,
            rows 1: are Freedman-Lane nulls
    """
    num_vox = exp_eff.y.shape[2]
    stat_fns = list(stat_dict.values())
    out = {stat_dict_inv[fn]: np.full((n_perm_fwer + 1, num_vox), np.nan)
           for fn in stat_fns}
    for k in range(n_perm_fwer + 1):
        _exp = exp_eff.permute(k) if k else exp_eff
        row = AnalysisVoxel.get_stat_perm_multi(_exp, stat_fns, children=None)
        for fn in stat_fns:
            out[stat_dict_inv[fn]][k, :] = row[fn]
    return out


def _build_specs(n_perm_fwer: int, alpha_fwer: float, cft_pval: float):
    """Yield (label, Ana, kw, stat_fn) for every VBA / VBA-TFCE / CET variant.

    Args:
        n_perm_fwer (int): number of FWER permutations
        alpha_fwer (float): FWER significance level
        cft_pval (float): cluster-forming threshold p-value (CET family)

    Yields:
        (label, Ana, kw, stat_fn) crossed over MANCOVA stats and {raw, z}
    """
    for fn in stat_dict.values():
        name = stat_dict_inv[fn]
        for z_flag in (False, True):
            suffix = '-z' if z_flag else ''
            yield (f'VBA-{name}{suffix}', AnalysisVBA,
                   dict(get_stat=fn, n_perm_fwer=n_perm_fwer,
                        alpha_fwer=alpha_fwer, z_flag=z_flag, tfce_flag=False),
                   fn)
            yield (f'VBA-TFCE-{name}{suffix}', AnalysisVBA,
                   dict(get_stat=fn, n_perm_fwer=n_perm_fwer,
                        alpha_fwer=alpha_fwer, z_flag=z_flag, tfce_flag=True),
                   fn)
            yield (f'CET-{name}{suffix}', AnalysisCET,
                   dict(get_stat=fn, n_perm_fwer=n_perm_fwer,
                        alpha_fwer=alpha_fwer, z_flag=z_flag, cft_pval=cft_pval),
                   fn)


def run_mancova(recorder, *, source: str, b: int, num_img: int, n_vox_eff: int,
                seed: int, n_perm_fwer: int, alpha_fwer: float = 0.05,
                cft_pval: float = DEFAULT_CET_CFT_PVAL,
                effect_llr=None, effect_total_llr=None):
    """VBA / VBA-TFCE / CET x 5 MANCOVA stats x {raw, z} on one trial.

    Records, it does not score. The shared voxel-stat walk is recorded as one
    step (its own timing), then each variant's fit records self (the recipe --
    the stat fn name, z_flag, tfce/cft -- which identifies the variant) tagged
    with its _build_specs label (e.g. 'VBA-TFCE-Wilks-z'). The big _stat matrix
    passed to fit records as a content hash. Scoring is derived afterward from
    the records, not here.

    Args:
        recorder (Recorder): the trial's recorder; calls are wrapped with it.
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count (>1 for a meaningful stat comparison)
        num_img (int): subject count (WGN; HCP uses its cohort)
        n_vox_eff (int): requested effect support size
        seed (int): effect RNG seed (also selects the HCP feature subset)
        n_perm_fwer (int): number of FWER permutations
        alpha_fwer (float): FWER significance level
        cft_pval (float): cluster-forming threshold p-value (CET family)
        effect_llr (float | None): per-voxel effect target
        effect_total_llr (float | None): whole-region effect target
    """
    setup = recorder(output_name_list=(
        'exp_eff', 'effect_list', 'mask_target_list'))(_setup_trial)(
        source=source, b=b, num_img=num_img, n_vox_eff=n_vox_eff, seed=seed,
        effect_llr=effect_llr, effect_total_llr=effect_total_llr)
    if setup is None:
        return
    exp_eff, _effect_list, _mask_target_list = setup

    # the shared walk is recorded once (its own time_sec); a failure stops the
    # trial (swallowed -> None) before the per-variant fits
    stat_by_name = recorder(output_name='stat')(_shared_voxel_walk)(
        exp_eff, n_perm_fwer)
    if stat_by_name is None:
        return

    for label, Ana, kw, fn in _build_specs(n_perm_fwer, alpha_fwer, cft_pval):
        recorder(output_name='ana', label=label)(Ana(exp=exp_eff, **kw).fit)(
            _stat=stat_by_name[stat_dict_inv[fn]].copy())


# ---------------------------------------------------------------------------
# two adjacent effects (cleaving)
# ---------------------------------------------------------------------------

def _setup_two_effect(*, source: str, b: int, num_img: int, n_vox_eff: int,
                      seed: int, angle: float, effect_llr=None,
                      effect_total_llr=None):
    """Build the two-adjacent-effect trial: the imposed experiment + provenance.

    A sphere centred in the mask is spectrally bisected (ExtenterSplit) into two
    contiguous halves; one effect is planted on each, same llr, with feature
    directions `angle` apart (shared effect sub-seed, angles 0 and angle).
    build_ds raises on an infeasible cell (recorded + swallowed by the recorder).

    The trial seed is split (derive_seeds) into independent DataSource / feature
    / effect sub-seeds, so each seed is its own data realization with the two
    effects independent of it -- matching _setup_trial.

    A sphere splits cleanly into two equal halves; a min-variance extent's
    irregular shape splits unevenly (verified at 25k: 34/66..58/42), which would
    break the "two equal effects" premise. Centre = in-mask voxel nearest the
    centroid (the analysis mask is itself a sphere, so this is its middle).

    Returns:
        exp_eff: the experiment with both effects imposed.
        effect_list (list): the two EffectSynthetic specs (angles 0 and angle).
        splitter (ExtenterSplit): the bisection that placed them -- the
            provenance of where each effect landed (the effects carry the masks
            verbatim, so their own to_record omits the support).
    """
    ds, effect_seed = build_ds_for_seed(source, b=b, num_img=num_img, seed=seed)
    exp = ds.exp
    llr = _effect_llr(effect_llr, effect_total_llr, n_vox_eff)

    coords = np.argwhere(exp.mask_idx > -1)
    ctr = coords.mean(axis=0)
    vox_init = int(exp.mask_idx[tuple(
        coords[np.argmin(((coords - ctr) ** 2).sum(axis=1))])])
    splitter = ExtenterSplit(
        base=ExtenterSphere(n_vox=n_vox_eff, connected=True, vox_init=vox_init))
    mask0, mask1 = splitter.fit(mask_idx=exp.mask_idx, y=exp.y)
    e0 = EffectSynthetic(mask=mask0, effect_llr=llr, angle=0.0, seed=effect_seed)
    e1 = EffectSynthetic(mask=mask1, effect_llr=llr, angle=float(angle),
                         seed=effect_seed)
    exp_eff = e1.fit(e0.fit(exp)[0])[0]
    return exp_eff, [e0, e1], splitter


def run_two_effect(recorder, *, source: str, b: int, num_img: int,
                   n_vox_eff: int, seed: int, angle: float,
                   ana_kwargs_dict: dict, effect_llr=None,
                   effect_total_llr=None):
    """Two adjacent equal-LLR effects at a controlled feature-direction angle.

    Records, it does not score. _setup_two_effect plants the two effects and
    records the experiment, the effect specs, and the ExtenterSplit (the
    provenance of the bisection); then each analysis fit records self (the
    recipe) tagged with its ana_kwargs_dict label (e.g. 'GLOW-GLM'). Scoring
    against each planted half is derived afterward from the records, not here.

    Args:
        recorder (Recorder): the trial's recorder; calls are wrapped with it.
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count (>= 2; the direction rotation needs a
            plane)
        num_img (int): subject count (WGN; HCP uses its cohort)
        n_vox_eff (int): the combined two-effect support, split into halves
        seed (int): effect RNG seed (extent grow, direction plane, HCP feats)
        angle (float): feature-direction angle between the two effects (deg)
        ana_kwargs_dict (dict): label -> (Analysis class, init kwargs)
        effect_llr (float | None): per-voxel target for each effect
        effect_total_llr (float | None): whole-region target per effect
    """
    setup = recorder(
        output_name_list=('exp_eff', 'effect_list', 'splitter'))(
        _setup_two_effect)(
        source=source, b=b, num_img=num_img, n_vox_eff=n_vox_eff, seed=seed,
        angle=angle, effect_llr=effect_llr, effect_total_llr=effect_total_llr)
    if setup is None:
        return
    exp_eff, _effect_list, _splitter = setup

    for label, (Ana, kw) in ana_kwargs_dict.items():
        recorder(output_name='ana', label=label)(Ana(exp=exp_eff, **kw).fit)()
