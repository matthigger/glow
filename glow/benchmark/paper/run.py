"""Trial functions for the paper benchmarks (flattened-axis edition).

Every trial fn takes only scalar/categorical axes (source, b, num_img,
n_vox_eff, seed, effect strength) and rebuilds the heavy DataSource /
Extenter objects via factory.build_ds, so iter_kwargs stays a flat scalar
grid and results.csv is tidy long-format: save_result merges each scalar
axis in as a column, and the fns below add only the method outputs plus a
few realized diagnostics (vox counts, the resolved per-voxel effect_llr,
the HCP feature subset).

Effect strength is given one of two ways, resolved by _effect_llr:
  - effect_llr: the per-voxel (size-normalized) target, held fixed when a
    structural axis (b, num_img) is swept.
  - effect_total_llr: the whole-region target; the per-voxel llr is
    effect_total_llr / n_vox_eff, so the total stays fixed as the extent
    grows (the extent sweep's choice).

Three trial kinds share this contract:
  - run_ana: fit every entry of an ana_kwargs_dict; one row per entry.
  - run_segment: oracle-Dice of the best-matching region in each Ward
    hierarchy; one row per ClusterMode (segmentation quality, no
    significance test or pruning).
  - run_mancova: one shared voxel-stat walk dispatched across VBA /
    VBA-TFCE / CET x 5 MANCOVA stats x {raw, z}; 30 rows per trial.
"""
import time
import traceback
from collections import namedtuple

import numpy as np
import pandas as pd

import glow
import glow.graph
import glow.mask
from glow.analysis import (
    AnalysisGLOW, AnalysisVoxel, AnalysisVBA, AnalysisCET,
    DEFAULT_CET_CFT_PVAL, inner_perm)
from glow.analysis.cluster import cluster, ClusterMode
from glow.analysis.mancova import stat_dict, stat_dict_inv
from glow.analysis.prune import prune_greedy, prune_dp
from glow.benchmark.trial_cache import SKIP_LABEL
from glow.effect import (EffectSynthetic, ExtenterMinVar, ExtenterSphere,
                         ExtenterSplit)
from .factory import build_ds, derive_seeds
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


def _trial_diag(exp, mask_target, effect_llr: float, feats) -> dict:
    """Realized per-trial diagnostics shared by every emitted row.

    These are the realized values (as opposed to the requested axes that
    save_result already merges in): the actual feature count / subject
    count, the planted support size, the resolved per-voxel effect_llr,
    and the HCP feature subset drawn for this seed.

    Args:
        exp: the clean experiment (y is (b, num_img, num_vox))
        mask_target (np.array): (X, Y, Z) bool, realized effect support
        effect_llr (float): resolved per-voxel planted strength
        feats (tuple | None): HCP feature subset, or None for WGN

    Returns:
        a dict of diagnostic columns
    """
    return {
        'b_real': int(exp.y.shape[0]),
        'num_img_real': int(exp.y.shape[1]),
        'vox_total': int(exp.y.shape[2]),
        'vox_effect': int(mask_target.sum()),
        'effect_llr': effect_llr,
        'feats': ','.join(feats) if feats else '',
    }


def _skip_frame(exc) -> pd.DataFrame:
    """A one-row SKIP frame for an infeasible cell (e.g. HCP b > pool).

    build_ds raises ValueError when a requested cell cannot be realized.
    Recording a SKIP row marks the cell done and auditable but keeps it out
    of the plots -- an intentional absence, not an ERROR (a failed trial).

    Args:
        exc (Exception): the build_ds error to record in the row

    Returns:
        a single-row DataFrame labelled SKIP
    """
    return pd.DataFrame([{'label': SKIP_LABEL, 'error': str(exc)}])


_Trial = namedtuple('_Trial', 'exp exp_eff mask_target mask_active diag')


def _setup_legacy(*, source: str, b: int, num_img: int, n_vox_eff: int,
                  seed: int, effect_llr=None, effect_total_llr=None) -> _Trial:
    """Build one planted-effect trial for the not-yet-recorder-wired fns.

    The common front half of run_segment / run_mancova / run_prune: resolve the
    data source, plant the synthetic effect at the resolved per-voxel llr, and
    assemble the per-row diagnostics. Raises ValueError for an infeasible cell
    -- callers turn that into a SKIP row via _skip_frame. (run_ana now uses the
    recorded _setup_trial instead; this remains until those fns are converted.)

    Args:
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count
        num_img (int): subject count (WGN; HCP uses its cohort)
        n_vox_eff (int): requested effect support size
        seed (int): effect RNG seed (also selects the HCP feature subset)
        effect_llr (float | None): per-voxel effect target
        effect_total_llr (float | None): whole-region effect target

    Returns:
        a _Trial(exp, exp_eff, mask_target, mask_active, diag)

    Raises:
        ValueError: infeasible cell (build_ds); the caller records a SKIP row
    """
    ds, feats = build_ds(source, b=b, num_img=num_img, seed=seed)
    extenter = ExtenterMinVar(n_vox=n_vox_eff, seed=seed)
    llr = _effect_llr(effect_llr, effect_total_llr, n_vox_eff)
    exp, exp_eff, mask_target = _plant(ds, extenter, llr)
    return _Trial(exp=exp, exp_eff=exp_eff, mask_target=mask_target,
                  mask_active=exp.mask_idx > -1,
                  diag=_trial_diag(exp, mask_target, llr, feats))


def _setup_trial(*, source: str, b: int, num_img: int, n_vox_eff: int,
                 seed: int, effect_llr=None, effect_total_llr=None):
    """Build one planted-effect trial: the imposed experiment and its effects.

    A plain function, wrapped with the recorder at the call site (run_ana) like
    every other recorded step -- the experiment + planted effects it returns
    capture the trial's data lineage through their to_record(). build_ds raises
    on an infeasible cell (e.g. HCP b > pool); the recorder records that failure
    and swallows it, so the caller sees None and ends the trial without a
    try/except.

    Args:
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count
        num_img (int): subject count (WGN; HCP uses its cohort)
        n_vox_eff (int): requested effect support size
        seed (int): effect RNG seed (also selects the HCP feature subset)
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
    ds, _ = build_ds(source, b=b, num_img=num_img, seed=seed)
    exp = ds.exp
    llr = _effect_llr(effect_llr, effect_total_llr, n_vox_eff)
    if llr is None:
        return exp, [], []
    effect = EffectSynthetic(extenter=ExtenterMinVar(n_vox=n_vox_eff, seed=seed),
                             effect_llr=llr)
    exp_eff, mask = effect.fit(exp)
    return exp_eff, [effect], [mask]


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
    (score_effects). All share one trial id (the cache hash, set by the
    iterator's scope). A fit that fails marks the trial failed, so its score
    step (and every later fit/score) short-circuits unrecorded (see recorder).

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

    for Ana, kw in ana_kwargs_dict.values():
        ana = recorder(output_name='ana')(Ana(exp=exp_eff, **kw).fit)()
        recorder(output_name='score')(score_effects)(
            ana=ana, mask_target_list=mask_target_list, mask_active=mask_active)


# ---------------------------------------------------------------------------
# min-size sweep: capture the per-perm (size -> max-z) staircase

# A large seed offset reused to keep distinct seed regimes apart: outer-perm
# k draws its inner FL perms from the block starting at
# (k + 1) * _SEED_OFFSET_DISTINCT (matching AnalysisGLOW's per-outer-perm
# spacing), and config offsets the min_size trial seeds by it so they sit
# clear of the other sweeps' seeds.
_SEED_OFFSET_DISTINCT = 100_000


def run_min_size(*, source: str, b: int, num_img: int, n_vox_eff: int,
                 seed: int, n_perm_fwer: int, n_perm_inner: int,
                 min_vox_floor: int = 1, cluster_mode=ClusterMode.FOCUS,
                 effect_llr=None, effect_total_llr=None):
    """Capture each outer perm's (size -> max-z) curve for a min_vox sweep.

    Mirrors run_ana's HCP trial setup (plant one synthetic effect, sweep
    effect_llr), but instead of fitting GLOW at a single min_vox it runs the
    outer-perm loop directly and records, per perm, the staircase of max-z
    over a size threshold (score.size_max_z_curve). With those curves in hand,
    GLOW's max-z FWER null -- hence its rejection / power -- can be recomputed
    at any min_vox >= min_vox_floor without re-fitting (the curve evaluated at
    the fit-time min_vox reproduces AnalysisGLOW.max_z_null exactly).

    Two deliberate departures from run_ana:
      - The trial seed is split (derive_seeds) into independent ds / feat /
        effect sub-seeds; the ds sub-seed drives the DataSource, so each seed
        is an independent data realization rather than the shared DS_SEED=0
        base -- the sweep wants independent nulls, with the effect kept
        independent of the data realization.
      - Inner perms use inner_perm.cpu_perm (no race), so every region >=
        min_vox_floor gets an exact z. The race only keeps the single global
        max accurate; raising min_vox past that region would read a frozen
        non-survivor z, biasing the swept null. cpu_perm avoids that, at the
        cost of the race's speedup.

    Args:
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

    Returns:
        a one-row DataFrame: the trial diagnostics plus n_perm_fwer,
        n_perm_inner, min_vox_floor, time_sec, and curve_json (the per-perm
        staircases, parse with json.loads).
    """
    s = derive_seeds(seed)
    try:
        ds, feats = build_ds(source, b=b, num_img=num_img,
                             seed=s.feat, ds_seed=s.ds)
    except ValueError as e:
        return _skip_frame(e)

    extenter = ExtenterMinVar(n_vox=n_vox_eff, seed=s.effect)
    llr = _effect_llr(effect_llr, effect_total_llr, n_vox_eff)
    exp, exp_eff, mask_target = _plant(ds, extenter, llr)
    diag = _trial_diag(exp, mask_target, llr, feats)

    # borrow AnalysisGLOW only for its scaling + (q0, q1) decomposition, so
    # the captured curves match a real fit; the outer loop below is run by
    # hand to swap the racing kernel for the exact cpu_perm.
    ana = AnalysisGLOW(exp=exp_eff, n_perm_fwer=n_perm_fwer,
                       n_perm_inner=n_perm_inner, min_vox=min_vox_floor,
                       cluster_mode=cluster_mode)
    exp_s, q0, q1 = ana.exp, ana._q0, ana._q1

    t0 = time.time()
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

    row = {'label': 'min_size', 'analysis_cls': 'AnalysisGLOW',
           'time_sec': time.time() - t0,
           'n_perm_fwer': n_perm_fwer, 'n_perm_inner': n_perm_inner,
           'min_vox_floor': min_vox_floor,
           'curve_json': curve_json(curve_list), **diag}
    return pd.DataFrame([row])


def run_segment(*, source: str, b: int, num_img: int, n_vox_eff: int,
                seed: int, modes, effect_llr=None, effect_total_llr=None):
    """Oracle Dice of the best region in each Ward hierarchy, per mode.

    Isolates segmentation quality from significance testing and pruning:
    for each ClusterMode it builds the hierarchy on the effect-bearing
    images and reports the maximum Dice over all regions of the tree (the
    oracle best match), unavailable in practice but a clean measure of how
    well the segmentation alone recovers the planted support.

    Args:
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count
        num_img (int): subject count (WGN; HCP uses its cohort)
        n_vox_eff (int): requested effect support size
        seed (int): effect RNG seed (also selects the HCP feature subset)
        modes: iterable of ClusterMode (or their string values) to compare
        effect_llr (float | None): per-voxel effect target
        effect_total_llr (float | None): whole-region effect target

    Returns:
        a DataFrame with one row per ClusterMode (label = mode name)
    """
    try:
        trial = _setup_legacy(source=source, b=b, num_img=num_img,
                             n_vox_eff=n_vox_eff, seed=seed,
                             effect_llr=effect_llr,
                             effect_total_llr=effect_total_llr)
    except ValueError as e:
        return _skip_frame(e)

    rows = []
    for mode in modes:
        mode = ClusterMode(mode)
        row = {'label': str(mode), **trial.diag}
        t0 = time.time()
        try:
            children = cluster(trial.exp_eff, mode=mode)
            # oracle: the single tree region best matching the planted support
            counts = score_oracle_tree(children=children,
                                       mask_target=trial.mask_target,
                                       mask_idx=trial.exp.mask_idx)
        except Exception:
            row['time_sec'] = time.time() - t0
            row['error'] = traceback.format_exc()
            rows.append(row)
            continue
        row['time_sec'] = time.time() - t0
        # store its counts (metrics derived downstream, as for every other row)
        row.update(counts)
        rows.append(row)

    return pd.DataFrame(rows)


def run_prune(*, source: str, b: int, num_img: int, n_vox_eff: int, seed: int,
              effect_llr=None, effect_total_llr=None):
    """Greedy vs DP pruning scored on one shared GLOW fit.

    Fits GLOW once with its default recipe -- Focus clustering and the
    shared FWER perms -- then applies both pruning rules to the SAME
    FWER-significant region set, so the comparison isolates the rule from
    the permutation test (and both see the regions GLOW would actually
    report). Both rank candidates by raw LLR, as AnalysisGLOW.finalize does
    (the z-score, right for FWER thresholding, fragments under pruning).
    prune_greedy blooms the max-LLR region and removes its tree relatives
    (GLOW's default; tends to undersegment); prune_dp takes the exact
    max-total-LLR antichain (tends to oversegment).

    Emits one row per rule (label Greedy / DP) with the usual aggregate
    tp/fp/tn/fn (over the union of the rule's output regions), n_selected
    (output-region count, ideal 1 per planted effect -- the direct
    over/under-segmentation measure), and n_sig (significant regions fed to
    both rules).

    Args:
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count
        num_img (int): subject count (WGN; HCP uses its cohort)
        n_vox_eff (int): requested effect support size
        seed (int): effect RNG seed (also selects the HCP feature subset)
        effect_llr (float | None): per-voxel effect target
        effect_total_llr (float | None): whole-region effect target

    Returns:
        a DataFrame with one row per pruning rule
    """
    try:
        trial = _setup_legacy(source=source, b=b, num_img=num_img,
                             n_vox_eff=n_vox_eff, seed=seed,
                             effect_llr=effect_llr,
                             effect_total_llr=effect_total_llr)
    except ValueError as e:
        return _skip_frame(e)

    # GLOW's default recipe, shared with the GLOW-Focus arm so both pruning
    # rules prune the regions GLOW would actually report. Deferred import:
    # config imports this module, so a module-level import would be circular.
    from .config import _GLOW_BASE
    glow_kwargs = {**_GLOW_BASE, 'cluster_mode': ClusterMode.FOCUS}

    t0 = time.time()
    ana = AnalysisGLOW(exp=trial.exp_eff, **glow_kwargs).fit()
    fit_time = time.time() - t0

    # the FWER-significant regions both rules prune, ranked by raw LLR
    # (mirrors AnalysisGLOW.finalize; the z-score fragments under pruning)
    sig_reg_list = list(np.where(ana.pval <= ana.alpha_fwer)[0])
    llr_gain = np.nan_to_num(ana.llr.astype(float), nan=0.0,
                             posinf=0.0, neginf=0.0)

    rows = []
    for label, prune_fn in (('Greedy', prune_greedy), ('DP', prune_dp)):
        reg_out_list, _ = prune_fn(sig_reg_list=sig_reg_list,
                                   children=ana.children, stat=llr_gain)
        reg_mask_list = [
            (r, glow.graph.get_label_map(reg_idx_list=[r],
                                         mask_idx=trial.exp.mask_idx,
                                         children=ana.children) > -1)
            for r in reg_out_list]
        rows.append({'label': label, 'analysis_cls': 'AnalysisGLOW',
                     **_score_regions(reg_mask_list, [trial.mask_target],
                                      trial.mask_active),
                     'n_sig': len(sig_reg_list),
                     'time_sec': fit_time, **trial.diag})
    return pd.DataFrame(rows)


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
    """One stat matrix per MANCOVA stat fn, shared across families.

    Args:
        exp_eff: the experiment with the synthetic effect added
        n_perm_fwer (int): number of FWER permutations

    Returns:
        stat_fn -> (n_perm_fwer + 1, num_vox) array; row 0 is observed,
            rows 1: are Freedman-Lane nulls
    """
    num_vox = exp_eff.y.shape[2]
    stat_fns = list(stat_dict.values())
    out = {fn: np.full((n_perm_fwer + 1, num_vox), np.nan) for fn in stat_fns}
    for k in range(n_perm_fwer + 1):
        _exp = exp_eff.permute(k) if k else exp_eff
        row = AnalysisVoxel.get_stat_perm_multi(_exp, stat_fns, children=None)
        for fn in stat_fns:
            out[fn][k, :] = row[fn]
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


def run_mancova(*, source: str, b: int, num_img: int, n_vox_eff: int,
                seed: int, n_perm_fwer: int, alpha_fwer: float = 0.05,
                cft_pval: float = DEFAULT_CET_CFT_PVAL,
                effect_llr=None, effect_total_llr=None):
    """VBA / VBA-TFCE / CET x 5 MANCOVA stats x {raw, z} on one trial.

    Shares one voxel-stat walk across families; each row's time_sec is
    walk_time + own_post (the isolated-run cost of that variant).

    Args:
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

    Returns:
        a DataFrame with one row per (family, stat, z) variant
    """
    try:
        trial = _setup_legacy(source=source, b=b, num_img=num_img,
                             n_vox_eff=n_vox_eff, seed=seed,
                             effect_llr=effect_llr,
                             effect_total_llr=effect_total_llr)
    except ValueError as e:
        return _skip_frame(e)

    walk_start = time.time()
    stat_by_fn = _shared_voxel_walk(trial.exp_eff, n_perm_fwer)
    walk_time = time.time() - walk_start

    rows = []
    for label, Ana, kw, fn in _build_specs(n_perm_fwer, alpha_fwer, cft_pval):
        row = {'label': label, 'analysis_cls': Ana.__name__,
               'stat': stat_dict_inv[fn], 'z_flag': kw['z_flag'], **trial.diag}
        post_start = time.time()
        try:
            ana = Ana(exp=trial.exp_eff, **kw).fit(_stat=stat_by_fn[fn].copy())
        except Exception:
            row['time_sec'] = walk_time + (time.time() - post_start)
            row['error'] = traceback.format_exc()
            rows.append(row)
            continue
        row['time_sec'] = walk_time + (time.time() - post_start)
        row.update(_score(ana, [trial.mask_target], trial.mask_active))
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# two adjacent effects (cleaving)
# ---------------------------------------------------------------------------

def run_two_effect(*, source: str, b: int, num_img: int, n_vox_eff: int,
                   seed: int, angle: float, ana_kwargs_dict: dict,
                   effect_llr=None, effect_total_llr=None):
    """Two adjacent equal-LLR effects at a controlled feature-direction angle.

    Grows one min-variance extent of n_vox_eff voxels, splits it into two
    contiguous halves (split_mask_spectral) and plants an effect on each:
    both at the same effect_llr, with feature directions `angle` degrees
    apart (shared seed, angles 0 and `angle`; see EffectSynthetic /
    sample_beta_direction).
    Fits every analysis and scores the prediction against each planted half
    separately (per-effect tp/fp/tn/fn suffixed 0/1; see _score_regions).

    Args:
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

    Returns:
        a DataFrame with one row per analysis label
    """
    try:
        ds, feats = build_ds(source, b=b, num_img=num_img, seed=seed)
    except ValueError as e:
        return _skip_frame(e)

    exp = ds.exp
    mask_active = exp.mask_idx > -1
    llr = _effect_llr(effect_llr, effect_total_llr, n_vox_eff)

    # A sphere centred in the middle of the data splits cleanly into two equal
    # halves; a min-variance extent's irregular shape splits unevenly (verified
    # at 25k: 34/66..58/42), which would break the "two equal effects" premise.
    # Centre = in-mask voxel nearest the centroid (the analysis mask is itself
    # a sphere, so this is its middle).
    coords = np.argwhere(mask_active)
    ctr = coords.mean(axis=0)
    vox_init = int(exp.mask_idx[tuple(
        coords[np.argmin(((coords - ctr) ** 2).sum(axis=1))])])
    # one sphere, spectrally bisected into two adjacent halves by a single
    # ExtenterSplit; plant one effect on each half (same llr, feature
    # directions `angle` apart).
    splitter = ExtenterSplit(
        base=ExtenterSphere(n_vox=n_vox_eff, connected=True, vox_init=vox_init))
    mask0, mask1 = splitter.fit(mask_idx=exp.mask_idx, y=exp.y)
    e0 = EffectSynthetic(mask=mask0, effect_llr=llr, angle=0.0, seed=seed)
    e1 = EffectSynthetic(mask=mask1, effect_llr=llr, angle=float(angle),
                         seed=seed)
    exp_eff0, _ = e0.fit(exp)
    exp_eff, _ = e1.fit(exp_eff0)

    diag = {**_trial_diag(exp, mask0 | mask1, llr, feats),
            'angle': float(angle)}
    rows = []
    for label, (Ana, kw) in ana_kwargs_dict.items():
        row = {'label': label, 'analysis_cls': Ana.__name__, **diag}
        t0 = time.time()
        try:
            ana = Ana(exp=exp_eff, **kw).fit()
        except Exception:
            row['time_sec'] = time.time() - t0
            row['error'] = traceback.format_exc()
            rows.append(row)
            continue
        row['time_sec'] = time.time() - t0
        row.update(_score(ana, [mask0, mask1], mask_active))
        rows.append(row)
    return pd.DataFrame(rows)
