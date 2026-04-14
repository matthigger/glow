import gzip
import json
import pathlib
import time
import traceback

import cloudpickle as pickle
import numpy as np

import glow
from glow.benchmark.file import OUT, ERROR, short_uuid


def _jsonable(v):
    """coerce numpy scalars / small types to JSON-safe Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _merge_iter_kw(d, iter_kw):
    """add swept iteration parameters to result dict (skip duplicates)."""
    for k, v in iter_kw.items():
        if k not in d:
            d[k] = _jsonable(v)


def _write_result(config, d, *, subfolder=OUT):
    """Generate UUID, write result dict as JSON. Returns the UUID string."""
    uuid_str = short_uuid()
    d['uuid'] = uuid_str
    file_out = config.folder / subfolder / f'{uuid_str}_result.json'
    file_out.parent.mkdir(exist_ok=True, parents=True)
    with open(file_out, 'w') as f:
        json.dump(d, f, sort_keys=True, indent=4)
    return uuid_str


def run_segment(config, **iter_kw):
    """run Ward's clustering and score against the imposed effect."""
    exp, effect = config.get_exp_eff(**iter_kw)

    from glow.analysis.cluster import _MODES, cluster
    for mode in _MODES:
        start = time.time()
        children = cluster(exp, mode=mode)
        total_time_sec = time.time() - start

        dice, sens, spec = glow.graph.get_dice_sens_spec(mask=effect.mask,
                                                         mask_idx=exp.mask_idx,
                                                         children=children)
        idx = np.argmax(dice)

        d = {'effect_llr': effect.effect_llr,
             'seed': int(effect.seed),
             'dice': dice[idx],
             'label': mode,
             'sens': sens[idx],
             'spec': spec[idx],
             'vox_total': int(exp.y.shape[2]),
             'vox_effect': int(effect.mask.sum()),
             'time_sec': total_time_sec,
             'config_hash': config._config_hash()}
        _merge_iter_kw(d, iter_kw)
        _write_result(config, d)


def _score_and_emit(ana, effect, config, label, total_time_sec, iter_kw):
    """Score a completed analysis and write the result JSON."""
    exp = ana.exp

    mask_pred = np.zeros(exp.mask_idx.shape, dtype=bool)
    for _effect in ana.effect_list:
        mask_pred |= _effect.mask

    mask_active = exp.mask_idx > -1
    dice, sens, spec = glow.mask.get_score(mask_pred=mask_pred,
                                           mask_target=effect.mask,
                                           mask_active=mask_active)

    pct_max_dice = 0.0
    if hasattr(ana, 'sig_reg_list') and hasattr(ana, 'children'):
        dice_all, _, _ = glow.graph.get_dice_sens_spec(
            mask=effect.mask, mask_idx=exp.mask_idx,
            children=ana.children)
        sig = ana.sig_reg_list
        max_dice_sig = max((dice_all[i] for i in sig), default=0.0)
        if max_dice_sig > 0:
            out_regs = [eff.reg_idx for eff in ana.effect_list]
            if out_regs:
                pct_max_dice = float(
                    max(dice_all[i] for i in out_regs) / max_dice_sig)

    min_pval = float(np.nanmin(ana.pval)) if hasattr(ana, 'pval') else None

    d = {'effect_llr': effect.effect_llr,
         'seed': int(effect.seed),
         'stat': ana.get_stat.__name__.replace('get_', ''),
         'label': label,
         'Analysis': type(ana).__name__,
         'dice': dice,
         'sens': sens,
         'spec': spec,
         'pct_max_dice': pct_max_dice,
         'min_pval': min_pval,
         'vox_total': int(exp.y.shape[2]),
         'vox_effect': int(effect.mask.sum()),
         'time_sec': total_time_sec,
         'config_hash': config._config_hash()}
    _merge_iter_kw(d, iter_kw)
    uuid_str = _write_result(config, d)

    if config.detail_save:
        file_out = config.folder / OUT / f'{uuid_str}_detail.p.gz'
        with gzip.open(file_out, 'wb') as f:
            pickle.dump((ana, effect), f)


def run_ana(config, _skip_labels=None, **iter_kw):
    """run all analyses defined in config.ana_kwargs_dict.

    Args:
        _skip_labels: optional set of labels to skip (already cached).
    """
    exp, effect = config.get_exp_eff(**iter_kw)

    for ana_label, (Ana, ana_kw) in config.ana_kwargs_dict.items():
        if _skip_labels and ana_label in _skip_labels:
            continue
        start = time.time()
        if config.error_save:
            try:
                ana = Ana(exp=exp, **ana_kw)
            except Exception as e:
                d = {'error_msg': traceback.format_exc(),
                     'label': ana_label,
                     'method': Ana.__name__,
                     'effect_llr': effect.effect_llr,
                     'seed': int(effect.seed)}
                print(f'error: {d}')

                _write_result(config, d, subfolder=ERROR)
                continue
        else:
            ana = Ana(exp=exp, **ana_kw)
        total_time_sec = time.time() - start

        _score_and_emit(ana, effect, config, ana_label,
                        total_time_sec, iter_kw)


def run_prune_compare(config, **iter_kw):
    """Run one AnalysisGLOW and apply all four pruning methods.

    Emits one JSON result per method with the same schema as run_ana,
    so the plotting pipeline works unchanged.
    """
    from glow.analysis.prune import (prune_greedy, prune_dp,
                                     prune_greedy_full_adjust)

    exp, effect = config.get_exp_eff(**iter_kw)

    _, (Ana, ana_kw) = next(iter(config.ana_kwargs_dict.items()))

    start = time.time()
    ana = Ana(exp=exp, **ana_kw)
    total_time_sec = time.time() - start

    sig = ana.sig_reg_list
    children = ana.children
    stat = np.nan_to_num(ana.stat.ravel().astype(float),
                         nan=0.0, posinf=0.0, neginf=0.0)

    methods = {
        'greedy': prune_greedy(sig, children, stat),
        'dp_lam0': prune_dp(sig, children, stat, lam=0.0),
        'dp_geom3': prune_dp(sig, children, stat, exp_n_eff=3.0),
        'full_adjust': prune_greedy_full_adjust(sig, children, exp),
    }

    mask_active = exp.mask_idx > -1

    for prune_label, (reg_out_list, _info) in methods.items():
        mask_pred = np.zeros(ana.exp.mask_idx.shape, dtype=bool)
        for reg_idx in reg_out_list:
            label_map = glow.graph.get_label_map(
                reg_idx_list=[reg_idx],
                mask_idx=exp.mask_idx,
                children=children)
            mask_pred |= (label_map > -1)

        dice, sens, spec = glow.mask.get_score(mask_pred=mask_pred,
                                               mask_target=effect.mask,
                                               mask_active=mask_active)
        d = {'effect_llr': effect.effect_llr,
             'seed': int(effect.seed),
             'stat': ana.get_stat.__name__.replace('get_', ''),
             'label': prune_label,
             'Analysis': Ana.__name__,
             'dice': dice,
             'sens': sens,
             'spec': spec,
             'vox_total': int(ana.exp.y.shape[2]),
             'vox_effect': int(effect.mask.sum()),
             'n_sig': len(sig),
             'n_selected': len(reg_out_list),
             'time_sec': total_time_sec,
             'config_hash': config._config_hash()}
        _merge_iter_kw(d, iter_kw)
        _write_result(config, d)


def run_mancova_glow(config, **iter_kw):
    """Run GLOW with all MANCOVA stats, sharing E/H across stats.

    Performs one tree walk per permutation and evaluates every stat in
    stat_dict on the same (E, H) matrices.  Size regression, FWER
    p-values and pruning are computed independently per stat.  Pruning
    always uses LLR regardless of the test statistic.
    """
    from glow.analysis import (
        Analysis, AnalysisGLOW, get_best_model, _sanitize_adjusted_stat)
    from glow.analysis.cluster import cluster
    from glow.analysis.mancova import (
        stat_dict, stat_dict_inv, get_llr)

    exp, effect = config.get_exp_eff(**iter_kw)
    start = time.time()

    stat_fns = list(stat_dict.values())
    _, ana_kw = next(iter(config.ana_kwargs_dict.values()))
    n_perm_fwer = ana_kw['n_perm_fwer']
    n_perm_sa = ana_kw.get('n_perm_fwer_size_adjust', 50)
    alpha_fwer = ana_kw.get('alpha_fwer', 0.05)
    min_size = ana_kw.get('min_size', 1)

    fit_start = n_perm_fwer + 1
    fit_end = n_perm_fwer + n_perm_sa
    num_vox = exp.y.shape[2]

    models = {fn: get_best_model(fn) for fn in stat_fns}
    XtX = {fn: None for fn in stat_fns}
    Xty = {fn: None for fn in stat_fns}

    # phase 1: fit permutations (held-out, for size regression)
    for perm_idx in range(fit_start, fit_end + 1):
        _exp = exp.permute(perm_idx)
        children = cluster(exp=_exp)
        multi = Analysis.get_stat_perm_multi(
            exp=_exp, get_stat_list=stat_fns, children=children)
        size = glow.graph.node_sum(
            np.ones(num_vox, dtype=int), children)
        for fn in stat_fns:
            XtX[fn], Xty[fn] = AnalysisGLOW.accumulate_regression(
                size, multi[fn].ravel(), models[fn], XtX[fn], Xty[fn])

    mu_fns = {}
    betas = {}
    for fn in stat_fns:
        mu_fn, _, beta = AnalysisGLOW.fit_size_regression_online(
            XtX[fn], Xty[fn], models[fn], fn)
        mu_fns[fn] = mu_fn
        betas[fn] = beta

    # phase 2: observed (perm 0) + FWER permutations
    stat_max = {fn: [] for fn in stat_fns}
    children_0 = size_0 = stat_0 = None

    for perm_idx in range(n_perm_fwer + 1):
        _exp = exp.permute(perm_idx)
        children = cluster(exp=_exp)
        multi = Analysis.get_stat_perm_multi(
            exp=_exp, get_stat_list=stat_fns, children=children)
        size = glow.graph.node_sum(
            np.ones(num_vox, dtype=int), children).astype(float)

        active = size >= min_size
        for fn in stat_fns:
            adj = multi[fn].ravel() - mu_fns[fn](size)
            adj = _sanitize_adjusted_stat(adj)
            stat_max[fn].append(
                float(np.nanmax(adj[active])) if active.any()
                else float('-inf'))

        if perm_idx == 0:
            children_0 = children
            size_0 = size
            stat_0 = {fn: multi[fn].ravel().astype(float)
                       for fn in stat_fns}

    stat_max_sorted = {fn: np.sort(stat_max[fn]) for fn in stat_fns}
    total_time = time.time() - start

    # phase 3: finalize each stat via AnalysisGLOW._finalize_analysis
    llr_prune = stat_0[get_llr]

    for fn in stat_fns:
        name = stat_dict_inv[fn]

        ana = AnalysisGLOW.from_precomputed(
            exp=exp, get_stat=fn,
            adj_model=models[fn], adj_beta=betas[fn])

        ana._finalize_analysis(
            exp, n_perm_fwer,
            stat_0[fn], size_0, children_0,
            mu_fns[fn], stat_max_sorted[fn],
            alpha_fwer, min_size,
            prune_stat=llr_prune)

        _score_and_emit(ana, effect, config,
                        f'GLOW-{name}', total_time, iter_kw)


def _run_variant(config, effect, exp, fn, name, stat, alpha_fwer,
                 walk_time, iter_kw, variant_start, AnalysisCls, **factory_kw):
    """Build an analysis variant via factory and emit its score."""
    ana = AnalysisCls.from_precomputed(
        exp=exp, get_stat=fn, stat=stat, alpha_fwer=alpha_fwer, **factory_kw)
    _score_and_emit(ana, effect, config, name,
                    walk_time + (time.time() - variant_start), iter_kw)


def run_mancova_vba(config, **iter_kw):
    """Run VBA, VBA-TFCE, and CET with all 5 MANCOVA stats.

    Computes all stats from a single voxel walk (shared E/H), then
    loops over variants: 5 stats x {raw, z} x {VBA, VBA-TFCE} + CET.
    The permutation walk is the expensive step; TFCE/CET/p-values are cheap.
    """
    from glow.analysis import (
        Analysis, AnalysisVBA, AnalysisCET, DEFAULT_CET_CFT_PVAL)
    from glow.analysis.mancova import (
        stat_dict, stat_dict_inv)

    exp, effect = config.get_exp_eff(**iter_kw)
    start = time.time()

    stat_fns = list(stat_dict.values())
    _, (Ana, ana_kw) = next(iter(config.ana_kwargs_dict.items()))
    n_perm_fwer = ana_kw['n_perm_fwer']
    alpha_fwer = ana_kw.get('alpha_fwer', 0.05)

    # one voxel walk, all 5 stats
    multi = Analysis.get_stat_perm_multi(
        exp, stat_fns, n_perm=n_perm_fwer, children=None)
    walk_time = time.time() - start

    for fn in stat_fns:
        name = stat_dict_inv[fn]

        for z_flag in [False, True]:
            variant_start = time.time()
            stat = multi[fn].copy()

            if z_flag:
                stat = Analysis.z_score_stat(stat)

            suffix = '-z' if z_flag else ''

            # VBA
            _run_variant(config, effect, exp, fn, f'VBA-{name}{suffix}',
                         stat, alpha_fwer, walk_time, iter_kw, variant_start,
                         AnalysisVBA)

            # VBA-TFCE
            stat_tfce = AnalysisVBA.apply_tfce(stat=stat, mask_idx=exp.mask_idx)
            _run_variant(config, effect, exp, fn, f'VBA-TFCE-{name}{suffix}',
                         stat_tfce, alpha_fwer, walk_time, iter_kw, variant_start,
                         AnalysisVBA)

    # CET variants (fixed cft_pval, sweep stats x {raw, z})
    cft_pval = DEFAULT_CET_CFT_PVAL
    for fn in stat_fns:
        name = stat_dict_inv[fn]

        for z_flag in [False, True]:
            variant_start = time.time()
            stat = multi[fn].copy()

            if z_flag:
                stat = Analysis.z_score_stat(stat)

            suffix = '-z' if z_flag else ''

            # empirical CFT from pooled null
            null_pool = stat[1:, :].ravel()
            cft = np.quantile(null_pool, 1 - cft_pval)

            _run_variant(config, effect, exp, fn, f'CET-{name}{suffix}',
                         stat, alpha_fwer, walk_time, iter_kw, variant_start,
                         AnalysisCET, cft=cft, cft_pval=cft_pval, z_flag=z_flag)


if __name__ == '__main__':
    from glow.benchmark.config import Config

    # quick test
    ana_kwargs_dict = {'GLOW': (glow.analysis.AnalysisGLOW,
                                dict(n_perm_fwer=100)),
                       'VBA': (glow.analysis.AnalysisVBA, dict(n_perm_fwer=100))}
    config = Config(label='quick_test', source='wgn', n_seed=3,
                    effect_llr_all=[0, 0.5], wgn_shape=(3, 3),
                    ana_kwargs_dict=ana_kwargs_dict)
    config.run_all(run_fnc=run_ana)
