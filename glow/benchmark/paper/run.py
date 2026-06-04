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

import numpy as np
import pandas as pd

import glow
import glow.graph
from glow.analysis import (
    AnalysisVoxel, AnalysisVBA, AnalysisCET, DEFAULT_CET_CFT_PVAL)
from glow.analysis.cluster import cluster, ClusterMode
from glow.analysis.mancova import stat_dict, stat_dict_inv
from glow.benchmark.trial_cache import SKIP_LABEL
from glow.effect import EffectSynthetic, ExtenterMinVar

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


def _score(ana, mask_target, mask_active) -> dict:
    """Dice/sens/spec + min_pval against the planted mask.

    Args:
        ana: a fitted Analysis whose .effect_list / .pval are scored
        mask_target (np.array): (X, Y, Z) bool, the planted effect support
        mask_active (np.array): (X, Y, Z) bool, voxels inside the mask

    Returns:
        the dice, sens, spec, and min_pval scores as a dict
    """
    mask_pred = np.zeros(mask_active.shape, dtype=bool)
    for eff in (ana.effect_list or ()):
        mask_pred |= eff.mask
    dice, sens, spec = glow.mask.get_score(
        mask_pred=mask_pred, mask_target=mask_target,
        mask_active=mask_active)
    return {
        'dice': dice, 'sens': sens, 'spec': spec,
        'min_pval': (float(np.nanmin(ana.pval))
                     if getattr(ana, 'pval', None) is not None
                     else np.nan),
    }


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
        # infeasible cell (e.g. HCP b > feature pool): record a SKIP row so
        # it's marked done and auditable, but excluded from plots. Not an
        # ERROR -- an intentional absence, not a failure.
        return pd.DataFrame([{'label': SKIP_LABEL, 'error': str(e)}])
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
        ds, feats = build_ds(source, b=b, num_img=num_img, seed=seed)
    except ValueError as e:
        # infeasible cell (e.g. HCP b > feature pool): record a SKIP row so
        # it's marked done and auditable, but excluded from plots. Not an
        # ERROR -- an intentional absence, not a failure.
        return pd.DataFrame([{'label': SKIP_LABEL, 'error': str(e)}])
    extenter = ExtenterMinVar(n_vox=n_vox_eff)
    llr = _effect_llr(effect_llr, effect_total_llr, n_vox_eff)

    exp, exp_eff, mask_target = _plant(ds, extenter, llr, seed)
    diag = _trial_diag(exp, mask_target, llr, feats)

    rows = []
    for mode in modes:
        mode = ClusterMode(mode)
        row = {'label': str(mode), **diag}
        t0 = time.time()
        try:
            children = cluster(exp_eff, mode=mode)
            dice, sens, spec = glow.graph.get_dice_sens_spec(
                mask=mask_target, mask_idx=exp.mask_idx, children=children)
            i = int(np.nanargmax(dice))
        except Exception:
            row['time_sec'] = time.time() - t0
            row['error'] = traceback.format_exc()
            rows.append(row)
            continue
        row['time_sec'] = time.time() - t0
        # oracle: the single region best matching the planted support
        row.update(dice=float(dice[i]), sens=float(sens[i]),
                   spec=float(spec[i]))
        rows.append(row)

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
        ds, feats = build_ds(source, b=b, num_img=num_img, seed=seed)
    except ValueError as e:
        # infeasible cell (e.g. HCP b > feature pool): record a SKIP row so
        # it's marked done and auditable, but excluded from plots. Not an
        # ERROR -- an intentional absence, not a failure.
        return pd.DataFrame([{'label': SKIP_LABEL, 'error': str(e)}])
    extenter = ExtenterMinVar(n_vox=n_vox_eff)
    llr = _effect_llr(effect_llr, effect_total_llr, n_vox_eff)

    exp, exp_eff, mask_target = _plant(ds, extenter, llr, seed)
    mask_active = exp.mask_idx > -1
    diag = _trial_diag(exp, mask_target, llr, feats)

    walk_start = time.time()
    stat_by_fn = _shared_voxel_walk(exp_eff, n_perm_fwer)
    walk_time = time.time() - walk_start

    rows = []
    for label, Ana, kw, fn in _build_specs(n_perm_fwer, alpha_fwer, cft_pval):
        row = {'label': label, 'analysis_cls': Ana.__name__,
               'stat': stat_dict_inv[fn], 'z_flag': kw['z_flag'], **diag}
        post_start = time.time()
        try:
            ana = Ana(exp=exp_eff, **kw).fit(_stat=stat_by_fn[fn].copy())
        except Exception:
            row['time_sec'] = walk_time + (time.time() - post_start)
            row['error'] = traceback.format_exc()
            rows.append(row)
            continue
        row['time_sec'] = walk_time + (time.time() - post_start)
        row.update(_score(ana, mask_target, mask_active))
        rows.append(row)

    return pd.DataFrame(rows)
