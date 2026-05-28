"""Trial functions for the paper benchmarks.

Both run_ana and run_mancova have the same outer signature,
(ds, extenter, effect_llr, seed, ...) -> DataFrame, so the CLI can
dispatch either one through the same driver_local call. The difference
is what each trial emits:

  - run_ana fits every entry of an ana_kwargs_dict and emits one row
    per entry.

  - run_mancova runs one shared voxel-stat walk and dispatches it
    across VBA / VBA-TFCE / CET x 5 MANCOVA stats x {raw, z}, emitting
    30 rows per trial.

Per-trial wall time is reported as the cost the variant would have
incurred run in isolation (shared walk amortised into each row).
"""
import time
import traceback

import numpy as np
import pandas as pd

import glow
from glow.analysis import (
    AnalysisVoxel, AnalysisVBA, AnalysisCET, DEFAULT_CET_CFT_PVAL)
from glow.analysis.mancova import stat_dict, stat_dict_inv, get_wilks
from glow.effect import EffectSynthetic


def _plant(ds, extenter, effect_llr: float, seed: int):
    """Build the exp + planted-effect pair shared by every trial fn.

    Args:
        ds: data source whose .exp gives the clean experiment
        extenter: Extenter that samples the planted effect's support
        effect_llr (float): planted effect strength (log-likelihood ratio)
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

    The returned dict carries no per-trial bookkeeping columns; the
    caller merges it into the row it's accumulating.

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


def run_ana(*, ds, extenter, effect_llr: float, seed: int,
            ana_kwargs_dict: dict):
    """Fit every analysis in ana_kwargs_dict on one synthetic trial.

    Args:
        ds: data source whose .exp gives the clean experiment
        extenter: Extenter that samples the planted effect's support
        effect_llr (float): planted effect strength (log-likelihood ratio)
        seed (int): RNG seed for the synthetic effect
        ana_kwargs_dict (dict): label -> (Analysis class, init kwargs)

    Returns:
        a DataFrame with one row per analysis label
    """
    exp, exp_eff, mask_target = _plant(ds, extenter, effect_llr, seed)
    mask_active = exp.mask_idx > -1

    rows = []
    for label, (Ana, kw) in ana_kwargs_dict.items():
        row = {
            'label': label,
            'analysis_cls': Ana.__name__,
            'vox_total': int(exp.y.shape[2]),
            'vox_effect': int(mask_target.sum()),
        }
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


# ---------------------------------------------------------------------------
# MANCOVA stat comparison
# ---------------------------------------------------------------------------
_MANCOVA_FAMILIES = ('VBA', 'VBA-TFCE', 'CET')


def _shared_voxel_walk(exp_eff, n_perm_fwer: int) -> dict:
    """One stat matrix per MANCOVA stat fn, shared across families.

    This is the expensive part; each family then post-processes its copy
    (TFCE smoothing, CET thresholding, raw vs z-score).

    Args:
        exp_eff: the experiment with the synthetic effect added
        n_perm_fwer (int): number of FWER permutations

    Returns:
        stat_fn -> (n_perm_fwer + 1, num_vox) array; row 0 is observed,
            rows 1: are Freedman-Lane nulls
    """
    num_vox = exp_eff.y.shape[2]
    stat_fns = list(stat_dict.values())
    out = {fn: np.full((n_perm_fwer + 1, num_vox), np.nan)
           for fn in stat_fns}
    for k in range(n_perm_fwer + 1):
        _exp = exp_eff.permute(k) if k else exp_eff
        row = AnalysisVoxel.get_stat_perm_multi(
            _exp, stat_fns, children=None)
        for fn in stat_fns:
            out[fn][k, :] = row[fn]
    return out


def _build_specs(n_perm_fwer: int, alpha_fwer: float, cft_pval: float):
    """Build the per-variant specs for the MANCOVA stat comparison.

    Args:
        n_perm_fwer (int): number of FWER permutations
        alpha_fwer (float): FWER significance level
        cft_pval (float): cluster-forming threshold p-value (CET family)

    Yields:
        (label, Ana, kw, stat_fn) for every VBA / VBA-TFCE / CET variant
            crossed with the MANCOVA stats and {raw, z}
    """
    for fn in stat_dict.values():
        name = stat_dict_inv[fn]
        for z_flag in (False, True):
            suffix = '-z' if z_flag else ''
            yield (f'VBA-{name}{suffix}', AnalysisVBA,
                   dict(get_stat=fn, n_perm_fwer=n_perm_fwer,
                        alpha_fwer=alpha_fwer, z_flag=z_flag,
                        tfce_flag=False),
                   fn)
            yield (f'VBA-TFCE-{name}{suffix}', AnalysisVBA,
                   dict(get_stat=fn, n_perm_fwer=n_perm_fwer,
                        alpha_fwer=alpha_fwer, z_flag=z_flag,
                        tfce_flag=True),
                   fn)
            yield (f'CET-{name}{suffix}', AnalysisCET,
                   dict(get_stat=fn, n_perm_fwer=n_perm_fwer,
                        alpha_fwer=alpha_fwer, z_flag=z_flag,
                        cft_pval=cft_pval),
                   fn)


def run_mancova(*, ds, extenter, effect_llr: float, seed: int,
                n_perm_fwer: int, alpha_fwer: float = 0.05,
                cft_pval: float = DEFAULT_CET_CFT_PVAL):
    """VBA / VBA-TFCE / CET x 5 MANCOVA stats x {raw, z} on one trial.

    Shares one voxel-stat walk across families. Each emitted row's
    time_sec is walk_time + own_post, the cost the variant would incur
    if run in isolation.

    Args:
        ds: data source whose .exp gives the clean experiment
        extenter: Extenter that samples the planted effect's support
        effect_llr (float): planted effect strength (log-likelihood ratio)
        seed (int): RNG seed for the synthetic effect
        n_perm_fwer (int): number of FWER permutations
        alpha_fwer (float): FWER significance level
        cft_pval (float): cluster-forming threshold p-value (CET family)

    Returns:
        a DataFrame with one row per (family, stat, z) variant
    """
    exp, exp_eff, mask_target = _plant(ds, extenter, effect_llr, seed)
    mask_active = exp.mask_idx > -1

    walk_start = time.time()
    stat_by_fn = _shared_voxel_walk(exp_eff, n_perm_fwer)
    walk_time = time.time() - walk_start

    rows = []
    for label, Ana, kw, fn in _build_specs(n_perm_fwer, alpha_fwer, cft_pval):
        row = {
            'label': label,
            'analysis_cls': Ana.__name__,
            'stat': stat_dict_inv[fn],
            'z_flag': kw['z_flag'],
            'vox_total': int(exp.y.shape[2]),
            'vox_effect': int(mask_target.sum()),
        }
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
