import gzip
import json
import pathlib
import time
import traceback
from uuid import uuid4

import cloudpickle as pickle
import numpy as np

import glow
from glow.benchmark.file import OUT, ERROR


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


def run_segment(config, **iter_kw):
    """run Ward's clustering and score against the imposed effect."""
    exp, effect = config.get_exp_eff(**iter_kw)

    from glow.experiment.cluster import _MODES
    for mode in _MODES:
        start = time.time()
        children = glow.experiment.cluster(exp, mode=mode)
        total_time_sec = time.time() - start

        f1, sens, spec = glow.graph.get_f1_sens_spec(mask=effect.mask,
                                                     mask_idx=exp.mask_idx,
                                                     children=children)
        idx = np.argmax(f1)

        uuid = str(uuid4())[:8]
        file_out = config.folder / OUT / f'{uuid}_result.json'
        d = {'effect_llr': effect.effect_llr,
             'seed': int(effect.seed),
             'f1': f1[idx],
             'label': mode,
             'sens': sens[idx],
             'spec': spec[idx],
             'uuid': uuid,
             'vox_total': int(exp.y.shape[2]),
             'vox_effect': int(effect.mask.sum()),
             'time_sec': total_time_sec,
             'config_hash': config._config_hash()}
        _merge_iter_kw(d, iter_kw)
        file_out.parent.mkdir(exist_ok=True, parents=True)
        with open(file_out, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)


def run_ana(config, **iter_kw):
    """run all analyses defined in config.ana_kwargs_dict."""
    exp, effect = config.get_exp_eff(**iter_kw)

    for ana_label, (Ana, ana_kw) in config.ana_kwargs_dict.items():
        uuid = str(uuid4())[:8]
        file_out = config.folder / OUT / f'{uuid}_result.json'

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

                file_out = str(file_out).replace(OUT, ERROR)
                file_out = pathlib.Path(file_out)
                file_out.parent.mkdir(exist_ok=True, parents=True)
                with open(file_out, 'w') as f:
                    json.dump(d, f, sort_keys=True, indent=4)
                continue
        else:
            ana = Ana(exp=exp, **ana_kw)
        total_time_sec = time.time() - start

        # build mask of predicted area (union of all effect masks)
        mask_pred = np.zeros(ana.exp.mask_idx.shape, dtype=bool)
        for _effect in ana.effect_list:
            mask_pred |= _effect.mask

        mask_active = exp.mask_idx > -1

        f1, sens, spec = glow.mask.get_score(mask_pred=mask_pred,
                                             mask_target=effect.mask,
                                             mask_active=mask_active)

        # pct_max_f1: fraction of max achievable per-region F1
        pct_max_f1 = 0.0
        if hasattr(ana, 'sig_reg_list') and hasattr(ana, 'children'):
            f1_all, _, _ = glow.graph.get_f1_sens_spec(
                mask=effect.mask, mask_idx=exp.mask_idx,
                children=ana.children)
            sig = ana.sig_reg_list
            max_f1_sig = max((f1_all[i] for i in sig), default=0.0)
            if max_f1_sig > 0:
                out_regs = [eff.reg_idx for eff in ana.effect_list]
                if out_regs:
                    pct_max_f1 = float(
                        max(f1_all[i] for i in out_regs) / max_f1_sig)

        # minimum FWER-corrected p-value (for type I error calibration)
        min_pval = float(np.nanmin(ana.pval)) if hasattr(ana, 'pval') else None

        d = {'effect_llr': effect.effect_llr,
             'seed': int(effect.seed),
             'stat': ana.get_stat.__name__.replace('get_', ''),
             'label': ana_label,
             'Analysis': Ana.__name__,
             'f1': f1,
             'sens': sens,
             'spec': spec,
             'pct_max_f1': pct_max_f1,
             'min_pval': min_pval,
             'uuid': uuid,
             'vox_total': int(ana.exp.y.shape[2]),
             'vox_effect': int(effect.mask.sum()),
             'time_sec': total_time_sec,
             'config_hash': config._config_hash()}
        _merge_iter_kw(d, iter_kw)
        file_out.parent.mkdir(exist_ok=True, parents=True)
        with open(file_out, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)

        if config.detail_save:
            file_out = config.folder / OUT / f'{uuid}_detail.p.gz'
            with gzip.open(file_out, 'wb') as f:
                pickle.dump((ana, effect), f)


def run_prune_compare(config, **iter_kw):
    """Run one AnalysisGLOW and apply all four pruning methods.

    Emits one JSON result per method with the same schema as run_ana,
    so the plotting pipeline works unchanged.
    """
    from glow.experiment.prune import (prune_greedy, prune_dp,
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

        f1, sens, spec = glow.mask.get_score(mask_pred=mask_pred,
                                             mask_target=effect.mask,
                                             mask_active=mask_active)
        uuid = str(uuid4())[:8]
        file_out = config.folder / OUT / f'{uuid}_result.json'
        d = {'effect_llr': effect.effect_llr,
             'seed': int(effect.seed),
             'stat': ana.get_stat.__name__.replace('get_', ''),
             'label': prune_label,
             'Analysis': Ana.__name__,
             'f1': f1,
             'sens': sens,
             'spec': spec,
             'uuid': uuid,
             'vox_total': int(ana.exp.y.shape[2]),
             'vox_effect': int(effect.mask.sum()),
             'n_sig': len(sig),
             'n_selected': len(reg_out_list),
             'time_sec': total_time_sec,
             'config_hash': config._config_hash()}
        _merge_iter_kw(d, iter_kw)
        file_out.parent.mkdir(exist_ok=True, parents=True)
        with open(file_out, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    from glow.benchmark.config import Config

    # quick test
    ana_kwargs_dict = {'GLOW': (glow.experiment.AnalysisGLOW,
                                dict(n_perm_fwer=100)),
                       'VBA': (glow.experiment.AnalysisVBA, dict(n_perm_fwer=100))}
    config = Config(label='quick_test', source='wgn', n_seed=3,
                    effect_llr_all=[0, 0.5], wgn_shape=(3, 3),
                    ana_kwargs_dict=ana_kwargs_dict)
    config.run_all(run_fnc=run_ana)
