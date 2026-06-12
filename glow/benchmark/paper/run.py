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
import json
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
    DEFAULT_CET_CFT_PVAL)
from glow.analysis.cluster import cluster, ClusterMode
from glow.analysis.mancova import stat_dict, stat_dict_inv
from glow.analysis.prune import prune_greedy, prune_dp
from glow.benchmark.trial_cache import SKIP_LABEL
from glow.effect import EffectSynthetic, ExtenterMinVar, ExtenterSphere
from glow.effect.extent import split_mask
from .factory import build_ds


def _effect_llr(effect_llr, effect_total_llr, n_vox_eff: int) -> float:
    """Resolve the per-voxel effect_llr from whichever knob the cache set.

    Args:
        effect_llr (float | None): per-voxel target (used as-is)
        effect_total_llr (float | None): whole-region target; divided by
            n_vox_eff to hold the total fixed as extent varies
        n_vox_eff (int): requested effect support size

    Returns:
        the per-voxel effect_llr to plant

    Raises:
        ValueError: if neither (or both) knobs are given
    """
    if (effect_llr is None) == (effect_total_llr is None):
        raise ValueError('pass exactly one of effect_llr / effect_total_llr')
    if effect_total_llr is not None:
        return float(effect_total_llr) / n_vox_eff
    return float(effect_llr)


def _plant(ds, extenter, effect_llr: float, seed: int):
    """Build the exp + planted-effect pair shared by every trial fn.

    Args:
        ds: data source whose .exp gives the clean experiment
        extenter: Extenter that samples the planted effect's support
        effect_llr (float): per-voxel planted effect strength
        seed (int): RNG seed for the synthetic effect

    Returns:
        exp: the clean experiment
        exp_eff: the experiment with the synthetic effect added
        mask_ (np.array): (X, Y, Z) bool, the realized effect support
    """
    exp = ds.exp
    synth = EffectSynthetic(extenter=extenter, effect_llr=effect_llr,
                            seed=seed)
    exp_eff = synth.fit(exp)
    return exp, exp_eff, synth.mask_


def _effect_reg_json(reg_tp_fp) -> str:
    """Pack per-output-region (reg_idx, tp, fp) triples into a JSON column.

    A trial outputs a variable number of effect regions, so rather than a
    ragged set of columns we stash one tidy row per method and serialize
    its regions into a single results.csv cell -- parse back with
    json.loads. reg_idx is the Ward-tree region index for GLOW (None for a
    voxel-method cluster that is not a tree node); tp/fp are that one
    region's counts against the planted support (over the active voxels).
    Use is diagnostic (e.g. debugging the per-region statistics); the
    plotted scores come from the row's aggregate tp/fp/tn/fn instead.

    Args:
        reg_tp_fp (list): (reg_idx, tp, fp) per output region, in output order

    Returns:
        a JSON list-of-dicts string, e.g. '[{"reg_idx": 12, "tp": 90, "fp": 4}]'
    """
    return json.dumps([{'reg_idx': None if r is None else int(r),
                        'tp': int(tp), 'fp': int(fp)}
                       for r, tp, fp in reg_tp_fp])


def _score_regions(reg_mask_list, mask_target, mask_active) -> dict:
    """Aggregate + per-region confusion scoring of a method's output regions.

    Shared by _score (a fitted Analysis's effect_list) and run_prune (each
    pruning rule's selected regions): unions the region masks for the
    aggregate tp/fp/tn/fn, and records each region's own tp/fp. Dice,
    sensitivity, PPV and specificity are derived from the aggregate counts
    at load time (glow.mask.stats_from_counts); the per-region detail is a
    cheap diagnostic the plots ignore.

    Args:
        reg_mask_list (list): (reg_idx, mask) per output region, in output
            order; reg_idx is the Ward region index or None (voxel methods)
        mask_target (np.array): (X, Y, Z) bool, the planted effect support
        mask_active (np.array): (X, Y, Z) bool, the analyzed voxels

    Returns:
        the aggregate tp/fp/tn/fn counts plus n_selected (output-region
        count) and effect_reg_json (each region's reg_idx + tp/fp)
    """
    mask_pred = np.zeros(mask_active.shape, dtype=bool)
    reg_tp_fp = []
    for reg_idx, mask in reg_mask_list:
        mask_pred |= mask
        c = glow.mask.confusion_counts(
            mask_pred=mask, mask_target=mask_target, mask_active=mask_active)
        reg_tp_fp.append((reg_idx, c['tp'], c['fp']))
    counts = glow.mask.confusion_counts(
        mask_pred=mask_pred, mask_target=mask_target, mask_active=mask_active)
    return {**counts,
            'n_selected': len(reg_tp_fp),
            'effect_reg_json': _effect_reg_json(reg_tp_fp)}


def _score(ana, mask_target, mask_active) -> dict:
    """Score a fitted Analysis's effect_list against the planted mask.

    The per-region confusion scoring of _score_regions (run by every
    Analysis trial) plus min_pval, the smallest region p-value reported.

    Args:
        ana: a fitted Analysis whose .effect_list / .pval are scored
        mask_target (np.array): (X, Y, Z) bool, the planted effect support
        mask_active (np.array): (X, Y, Z) bool, voxels inside the mask

    Returns:
        the _score_regions dict (tp/fp/tn/fn, n_selected, effect_reg_json)
        plus min_pval
    """
    reg_mask_list = [(getattr(eff, 'reg_idx', None), eff.mask)
                     for eff in (ana.effect_list or ())]
    scored = _score_regions(reg_mask_list, mask_target, mask_active)
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


def _setup_trial(*, source: str, b: int, num_img: int, n_vox_eff: int,
                 seed: int, effect_llr=None, effect_total_llr=None) -> _Trial:
    """Build one planted-effect trial shared by the scalar-axis trial fns.

    The common front half of run_segment / run_mancova / run_prune (run_ana
    routes through _run_ana_obj instead): resolve the data source, plant the
    synthetic effect at the resolved per-voxel llr, and assemble the per-row
    diagnostics. Raises ValueError for an infeasible cell -- callers turn
    that into a SKIP row via _skip_frame.

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
    extenter = ExtenterMinVar(n_vox=n_vox_eff)
    llr = _effect_llr(effect_llr, effect_total_llr, n_vox_eff)
    exp, exp_eff, mask_target = _plant(ds, extenter, llr, seed)
    return _Trial(exp=exp, exp_eff=exp_eff, mask_target=mask_target,
                  mask_active=exp.mask_idx > -1,
                  diag=_trial_diag(exp, mask_target, llr, feats))


def _run_ana_obj(*, ds, extenter, effect_llr: float, seed: int,
                 ana_kwargs_dict: dict, feats=None):
    """Fit every analysis in ana_kwargs_dict on one already-built trial.

    The object-level core shared by run_ana (which resolves scalar axes
    via the factory) and the mothballed catalogue (which builds its own
    2-D / sphere-extent objects directly).

    Args:
        ds: data source whose .exp gives the clean experiment
        extenter: Extenter that samples the planted effect's support
        effect_llr (float): per-voxel planted effect strength
        seed (int): effect RNG seed
        ana_kwargs_dict (dict): label -> (Analysis class, init kwargs)
        feats (tuple | None): HCP feature subset for the diagnostics, or None

    Returns:
        a DataFrame with one row per analysis label
    """
    exp, exp_eff, mask_target = _plant(ds, extenter, effect_llr, seed)
    mask_active = exp.mask_idx > -1
    diag = _trial_diag(exp, mask_target, effect_llr, feats)

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
        row.update(_score(ana, mask_target, mask_active))
        rows.append(row)

    return pd.DataFrame(rows)


def run_ana(*, source: str, b: int, num_img: int, n_vox_eff: int, seed: int,
            ana_kwargs_dict: dict, effect_llr=None, effect_total_llr=None):
    """Fit every analysis in ana_kwargs_dict on one synthetic trial.

    Args:
        source (str): 'wgn' or 'hcp'
        b (int): imaging-feature count
        num_img (int): subject count (WGN; HCP uses its cohort)
        n_vox_eff (int): requested effect support size
        seed (int): effect RNG seed (also selects the HCP feature subset)
        ana_kwargs_dict (dict): label -> (Analysis class, init kwargs)
        effect_llr (float | None): per-voxel effect target
        effect_total_llr (float | None): whole-region effect target

    Returns:
        a DataFrame with one row per analysis label
    """
    try:
        ds, feats = build_ds(source, b=b, num_img=num_img, seed=seed)
    except ValueError as e:
        return _skip_frame(e)
    extenter = ExtenterMinVar(n_vox=n_vox_eff)
    llr = _effect_llr(effect_llr, effect_total_llr, n_vox_eff)
    return _run_ana_obj(ds=ds, extenter=extenter, effect_llr=llr, seed=seed,
                        ana_kwargs_dict=ana_kwargs_dict, feats=feats)


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
        trial = _setup_trial(source=source, b=b, num_img=num_img,
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
            counts = glow.graph.confusion_counts_tree(
                mask=trial.mask_target, mask_idx=trial.exp.mask_idx,
                children=children)
            # the oracle picks the single tree region best matching the
            # target, so the max is taken here over the whole tree
            dice = glow.mask.stats_from_counts(**counts)['dice']
            i = int(np.nanargmax(dice))
        except Exception:
            row['time_sec'] = time.time() - t0
            row['error'] = traceback.format_exc()
            rows.append(row)
            continue
        row['time_sec'] = time.time() - t0
        # oracle: the single region best matching the planted support; store
        # its counts (metrics derived downstream, as for every other row)
        row.update(tp=int(counts['tp'][i]), fp=int(counts['fp'][i]),
                   tn=int(counts['tn'][i]), fn=int(counts['fn'][i]))
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
    over/under-segmentation measure), n_sig (significant regions fed to both
    rules), and effect_reg_json (each output region's reg_idx + tp/fp).

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
        trial = _setup_trial(source=source, b=b, num_img=num_img,
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
                     **_score_regions(reg_mask_list, trial.mask_target,
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
        trial = _setup_trial(source=source, b=b, num_img=num_img,
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
        row.update(_score(ana, trial.mask_target, trial.mask_active))
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# two adjacent effects (cleaving)
# ---------------------------------------------------------------------------

def _score_regions_two(reg_mask_list, mask0, mask1, mask_active) -> dict:
    """Region x truth-class overlap for the two-effect cleaving trial.

    Records, per output region, its voxel overlap with each planted effect
    (n0, n1) and with the analysed background (nbg). This overlap table is the
    sufficient statistic for the downstream metrics -- instance separation
    (ARI of the recovered partition vs the {effect0, effect1} truth) and
    per-effect detection both derive from it -- so results.csv stores raw
    counts only. Also returns the aggregate confusion of the union of regions
    against the whole effect (mask0 | mask1), which feeds the detectability
    (Dice / sens / PPV) panel through the existing stats_from_counts path.

    Output regions are disjoint (a Ward antichain, or connected components for
    the voxel methods), so the per-region overlaps partition the detection.

    Args:
        reg_mask_list (list): (reg_idx, mask) per output region, in output
            order; reg_idx is the Ward region index or None (voxel methods)
        mask0, mask1 (np.array): (X, Y, Z) bool, the two planted effect halves
        mask_active (np.array): (X, Y, Z) bool, the analysed voxels

    Returns:
        the union-vs-(mask0|mask1) aggregate tp/fp/tn/fn, plus vox_eff0,
        vox_eff1, n_selected and region_overlap_json (a JSON list of
        {reg_idx, n0, n1, nbg} dicts)
    """
    bg = mask_active & ~(mask0 | mask1)
    mask_pred = np.zeros(mask_active.shape, dtype=bool)
    overlap = []
    for reg_idx, mask in reg_mask_list:
        m = mask & mask_active
        mask_pred |= m
        overlap.append({'reg_idx': None if reg_idx is None else int(reg_idx),
                        'n0': int((m & mask0).sum()),
                        'n1': int((m & mask1).sum()),
                        'nbg': int((m & bg).sum())})
    counts = glow.mask.confusion_counts(
        mask_pred=mask_pred, mask_target=(mask0 | mask1),
        mask_active=mask_active)
    return {**counts,
            'vox_eff0': int(mask0.sum()), 'vox_eff1': int(mask1.sum()),
            'n_selected': len(overlap),
            'region_overlap_json': json.dumps(overlap)}


def run_two_effect(*, source: str, b: int, num_img: int, n_vox_eff: int,
                   seed: int, angle: float, ana_kwargs_dict: dict,
                   effect_llr=None, effect_total_llr=None):
    """Two adjacent equal-LLR effects at a controlled feature-direction angle.

    Grows one min-variance extent of n_vox_eff voxels, splits it into two
    contiguous halves (split_mask), and plants an effect on each: both at the
    same effect_llr, with feature directions `angle` degrees apart (shared
    seed, angles 0 and `angle`; see EffectSynthetic / sample_beta_direction).
    Fits every analysis and scores the region x truth-class overlap, so ARI
    (cleaving) and per-effect detection are recoverable downstream.

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

    # A sphere centred in the middle of the data splits (split_mask) into two
    # equal halves; a min-variance extent's irregular shape splits unevenly
    # (verified at 25k: 34/66..58/42, vs the sphere's exact 50/50), which would
    # break the "two equal effects" premise. Centre = in-mask voxel nearest the
    # centroid (the analysis mask is itself a sphere, so this is its middle).
    coords = np.argwhere(mask_active)
    ctr = coords.mean(axis=0)
    vox_init = int(exp.mask_idx[tuple(
        coords[np.argmin(((coords - ctr) ** 2).sum(axis=1))])])
    extent = ExtenterSphere(n_vox=n_vox_eff, connected=True)(
        mask_idx=exp.mask_idx, y=exp.y, vox_init=vox_init)
    mask0, mask1 = split_mask(extent)
    e0 = EffectSynthetic(mask=mask0, effect_llr=llr, angle=0.0, seed=seed)
    e1 = EffectSynthetic(mask=mask1, effect_llr=llr, angle=float(angle),
                         seed=seed)
    exp_eff = e1.fit(e0.fit(exp))

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
        reg_mask_list = [(getattr(eff, 'reg_idx', None), eff.mask)
                         for eff in (ana.effect_list or ())]
        row.update(_score_regions_two(reg_mask_list, mask0, mask1, mask_active))
        row['min_pval'] = (float(np.nanmin(ana.pval))
                           if getattr(ana, 'pval', None) is not None
                           else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)
