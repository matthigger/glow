"""Trial functions for the paper benchmarks.

Both ``run_ana`` and ``run_mancova`` have the same outer signature —
``(ds, extenter, effect_llr, seed, ...) -> DataFrame`` — so the CLI can
dispatch either one through the same ``driver_local`` call.  The
difference is what each trial emits:

  - ``run_ana`` fits every entry of an ``ana_kwargs_dict`` and emits
    one row per entry.

  - ``run_mancova`` runs one shared voxel-stat walk and dispatches it
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


def _plant(ds, extenter, effect_llr, seed):
    """Build the exp + planted-effect pair shared by every trial fn."""
    exp = ds.exp
    synth = EffectSynthetic(extenter=extenter, effect_llr=effect_llr,
                            seed=seed)
    exp_eff = synth.fit(exp)
    return exp, exp_eff, synth.mask_


def _score(ana, mask_target, mask_active):
    """Dice/sens/spec + min_pval against the planted mask.

    Returns a dict (without the per-trial bookkeeping columns); merged
    by the caller into the row it's accumulating.
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


def run_ana(*, ds, extenter, effect_llr, seed, ana_kwargs_dict):
    """Fit every analysis in ``ana_kwargs_dict`` on one synthetic trial.

    Returns a DataFrame with one row per analysis label.
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


def _shared_voxel_walk(exp_eff, n_perm_fwer):
    """One stat matrix per MANCOVA stat fn, shared across families.

    Returns ``{stat_fn: (n_perm_fwer + 1, num_vox) array}``: row 0 is
    observed; rows 1: are Freedman-Lane nulls.  This is the expensive
    part — each family then post-processes its copy (TFCE smoothing,
    CET thresholding, raw vs z-score).
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


def _build_specs(n_perm_fwer, alpha_fwer, cft_pval):
    """Yield (label, Ana, kw, z_flag, stat_fn) for every mancova variant."""
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


def run_mancova(*, ds, extenter, effect_llr, seed,
                n_perm_fwer, alpha_fwer=0.05,
                cft_pval=DEFAULT_CET_CFT_PVAL):
    """VBA / VBA-TFCE / CET x 5 MANCOVA stats x {raw, z} on one trial.

    Shares one voxel-stat walk across families.  Each emitted row's
    ``time_sec`` is ``walk_time + own_post`` — the cost the variant
    would incur if run in isolation.
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
