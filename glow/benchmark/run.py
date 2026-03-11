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


def run_segment(config, **kwargs):
    """run Ward's clustering and score against the imposed effect."""
    # build a particular effect
    exp, effect = config.get_exp_eff(**kwargs)

    for mode in ('ward-naive', 'ward-glm'):
        # cluster
        start = time.time()
        children = glow.experiment.cluster(exp, mode=mode)
        total_time_sec = time.time() - start

        # find best (f1) region
        f1, sens, spec = glow.graph.get_f1_sens_spec(mask=effect.mask,
                                                     mask_idx=exp.mask_idx,
                                                     children=children)
        idx = np.argmax(f1)

        # dump summary
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
        file_out.parent.mkdir(exist_ok=True, parents=True)
        with open(file_out, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)


def run_ana(config, **kwargs):
    """run all analyses defined in config.ana_kwargs_dict."""
    # build a particular effect
    exp, effect = config.get_exp_eff(**kwargs)

    for label, (Ana, kwargs) in config.ana_kwargs_dict.items():
        # prep output file
        uuid = str(uuid4())[:8]
        file_out = config.folder / OUT / f'{uuid}_result.json'

        # run analysis
        start = time.time()
        if config.error_save:
            # catch errors and dump to json if any occur (allows us to
            # continue with experiment in event of errors)
            try:
                ana = Ana(exp=exp, **kwargs)
            except Exception as e:
                d = {'error_msg': traceback.format_exc(),
                     'label': label,
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
            # no error catching, will stop all experiments if any error
            ana = Ana(exp=exp, **kwargs)
        total_time_sec = time.time() - start

        # build mask of predicted area (union of all effect masks)
        mask_pred = np.zeros(ana.exp.mask_idx.shape, dtype=bool)
        for _effect in ana.effect_list:
            mask_pred |= _effect.mask

        mask_active = exp.mask_idx > -1

        # compute scores
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

        # dump summary
        d = {'effect_llr': effect.effect_llr,
             'seed': int(effect.seed),
             'stat': ana.get_stat.__name__.replace('get_', ''),
             'label': label,
             'Analysis': Ana.__name__,
             'f1': f1,
             'sens': sens,
             'spec': spec,
             'pct_max_f1': pct_max_f1,
             'uuid': uuid,
             'vox_total': int(ana.exp.y.shape[2]),
             'vox_effect': int(effect.mask.sum()),
             'time_sec': total_time_sec,
             'config_hash': config._config_hash()}
        file_out.parent.mkdir(exist_ok=True, parents=True)
        with open(file_out, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)

        if config.detail_save:
            # dump detail
            file_out = config.folder / OUT / f'{uuid}_detail.p.gz'
            with gzip.open(file_out, 'wb') as f:
                pickle.dump((ana, effect), f)


def run_stat_auc(config, **kwargs):
    """Compare MANCOVA statistics via DP-optimal antichain AUC.

    For each stat, fits a size model on null permutations, computes
    adjusted stats, finds the best antichain via DP, and measures
    separability (AUC) against the known target.  Also reports
    F1/sens/spec from the antichain for comparison.
    """
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm
    from glow.experiment.analysis import AnalysisGLOW, get_best_model
    from glow.experiment.cluster import cluster
    from glow.experiment.exper import ExperimentScaled
    from glow.experiment.mancova import stat_dict, stat_sign
    from glow.graph import (children_to_map, dp_antichain, get_miss_hits,
                            node_sum)

    n_perm_fit = kwargs.pop('n_perm_fit', 30)
    exp, effect = config.get_exp_eff(**kwargs)
    if not isinstance(exp, ExperimentScaled):
        exp = ExperimentScaled.from_exp(exp)

    num_vox = exp.y.shape[2]
    stat_funcs = dict(stat_dict)

    # ------------------------------------------------------------------
    # Phase 1: cluster + compute all stats for real data + permutations
    # ------------------------------------------------------------------
    # stats_all[perm_idx] = {stat_name: np.array(num_reg)}
    stats_all = {}
    sizes_all = {}

    start = time.time()
    for perm_idx in tqdm(range(n_perm_fit + 1), desc='stat_auc perms',
                         disable=not getattr(config, 'verbose', True)):
        _exp = exp.permute(perm_idx)
        children = cluster(exp=_exp)
        num_reg = num_vox + children.shape[0]

        sizes = node_sum(x=np.ones(num_vox, dtype=int), children=children)
        sizes_all[perm_idx] = sizes.astype(float)

        perm_stats = {}
        for reg_idx, size, e, h in glow.graph.iter_stat(exp=_exp,
                                                         children=children):
            _e, _h = e[:, :, 0], h[:, :, 0]
            for name, fn in stat_funcs.items():
                if name not in perm_stats:
                    perm_stats[name] = np.full(num_reg, np.nan)
                try:
                    perm_stats[name][reg_idx] = fn(e=_e, h=_h, n=size)
                except np.linalg.LinAlgError:
                    pass
        stats_all[perm_idx] = perm_stats
        del _exp

    total_time_sec = time.time() - start

    # real data
    children_real = cluster(exp=exp.permute(0))
    sizes_real = sizes_all[0]
    children_map = children_to_map(children_real)
    miss, hit = get_miss_hits(mask=effect.mask, mask_idx=exp.mask_idx,
                              children=children_real)
    num_reg = num_vox + children_real.shape[0]

    total_hit = int(hit[:num_vox].sum())
    total_miss = int(miss[:num_vox].sum())

    # ------------------------------------------------------------------
    # Phase 2: per-stat size adjustment + DP + AUC
    # ------------------------------------------------------------------
    for stat_label, stat_fn in stat_funcs.items():
        # collect null (size, stat) pairs for fitting
        null_sizes = np.concatenate([sizes_all[p] for p in range(1, n_perm_fit + 1)])
        null_stats = np.concatenate([stats_all[p][stat_label]
                                     for p in range(1, n_perm_fit + 1)])

        model = get_best_model(stat_fn)

        XtX, Xty = None, None
        for p in range(1, n_perm_fit + 1):
            XtX, Xty = AnalysisGLOW.accumulate_regression(
                sizes_all[p], stats_all[p][stat_label], model, XtX, Xty)
        mu_fn, _, beta = AnalysisGLOW.fit_size_regression_online(
            XtX, Xty, model)

        # compute adjusted stat on real data (perm 0)
        sign = stat_sign.get(stat_fn, 1)
        raw = stats_all[0][stat_label]
        predicted = AnalysisGLOW.predict_null_mean(sizes_real, model, beta)
        adjusted = np.nan_to_num(sign * (raw - predicted), nan=-np.inf)

        # DP antichain
        gain = {i: float(adjusted[i]) for i in range(num_reg)}
        selected, _info = dp_antichain(
            nodes=range(num_reg),
            children_map=children_map,
            gain=gain,
        )

        # AUC via weighted Mann-Whitney
        scores = []
        labels = []
        sel_hit_total = 0
        sel_miss_total = 0
        for reg in selected:
            h_r, m_r = int(hit[reg]), int(miss[reg])
            s_r = float(adjusted[reg])
            scores.extend([s_r] * (h_r + m_r))
            labels.extend([1] * h_r + [0] * m_r)
            sel_hit_total += h_r
            sel_miss_total += m_r

        # unselected voxels get score 0
        unseen_hit = total_hit - sel_hit_total
        unseen_miss = total_miss - sel_miss_total
        scores.extend([0.0] * (unseen_hit + unseen_miss))
        labels.extend([1] * unseen_hit + [0] * unseen_miss)

        scores = np.array(scores)
        labels = np.array(labels)

        if labels.sum() == 0 or labels.sum() == len(labels):
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels, scores))

        # F1/sens/spec from antichain
        mask_pred = np.zeros(exp.mask_idx.shape, dtype=bool)
        for reg_idx in selected:
            label_map = glow.graph.get_label_map(
                reg_idx_list=[reg_idx],
                mask_idx=exp.mask_idx,
                children=children_real)
            mask_pred |= (label_map > -1)

        f1, sens, spec = glow.mask.get_score(
            mask_pred=mask_pred,
            mask_target=effect.mask,
            mask_active=exp.mask_idx > -1)

        # save result
        uuid = str(uuid4())[:8]
        file_out = config.folder / OUT / f'{uuid}_result.json'
        d = {'effect_llr': effect.effect_llr,
             'seed': int(effect.seed),
             'stat': stat_fn.__name__.replace('get_', ''),
             'label': stat_label,
             'Analysis': 'stat_auc',
             'auc': auc,
             'f1': f1,
             'sens': sens,
             'spec': spec,
             'n_selected': len(selected),
             'uuid': uuid,
             'vox_total': int(num_vox),
             'vox_effect': int(effect.mask.sum()),
             'time_sec': total_time_sec / len(stat_funcs),
             'config_hash': config._config_hash()}
        file_out.parent.mkdir(exist_ok=True, parents=True)
        with open(file_out, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    from glow.benchmark.config import Config

    # quick test
    ana_kwargs_dict = {'GLOW': (glow.experiment.AnalysisGLOW,
                                dict(n_perm=100)),
                       'VBA': (glow.experiment.AnalysisVBA, dict(n_perm=100))}
    config = Config(label='quick_test', source='wgn', n_seed=3,
                    effect_llr_all=[0, 0.5], wgn_shape=(3, 3),
                    ana_kwargs_dict=ana_kwargs_dict)
    config.run_all(run_fnc=run_ana)
