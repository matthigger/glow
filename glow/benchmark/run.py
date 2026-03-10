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

        # compute scores
        f1, sens, spec = glow.mask.get_score(mask_pred=mask_pred,
                                             mask_target=effect.mask,
                                             mask_active=exp.mask_idx > -1)
        # dump summary
        d = {'effect_llr': effect.effect_llr,
             'seed': int(effect.seed),
             'stat': ana.get_stat.__name__.replace('get_', ''),
             'label': label,
             'Analysis': Ana.__name__,
             'f1': f1,
             'sens': sens,
             'spec': spec,
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


def _prune_diagnostics_df(sig, methods):
    """Build per-region diagnostics from the pruning results.

    Returns a DataFrame with one row per significant region and columns
    for each method's diagnostics, suitable for the viewer's --csv flag.
    """
    import pandas as pd

    homo_regs, homo_info = methods['homo']
    nd_regs, dp_info = methods['node']
    nd_fl_regs, dp_fl_info = methods['node_fl']
    tr_regs, tr_info = methods['tree']

    homo_set, nd_set, tr_set = (set(homo_regs), set(nd_regs),
                                 set(tr_regs))
    nd_fl_set = set(nd_fl_regs)
    lam = dp_info.get('lam', 0.0)
    lam_fl = dp_fl_info.get('lam', 0.0)
    gain_nd = dp_info.get('gain', {})
    h0_mean = dp_fl_info.get('node_gain_h0_mean', {})
    h0_std = dp_fl_info.get('node_gain_h0_std', {})
    gain_tr = tr_info.get('gain_per_node', {})
    weights = tr_info.get('weights', {})
    wt_passes = tr_info.get('wt_gain_sum_passes', [])

    has_dp = 'tree_dp' in methods
    if has_dp:
        tr_dp_regs, tr_dp_info = methods['tree_dp']
        tr_dp_set = set(tr_dp_regs)
        tw_gain = tr_dp_info.get('tree_wide_gain', {})
        lam_dp = tr_dp_info.get('lam', 0.0)

    rows = []
    for node in sig:
        g = gain_nd.get(node, np.nan)
        ga = gain_tr.get(node, np.nan)
        w = weights.get(node, 1.0)
        row = {
            'region_idx': node,
            'homo_pval': homo_info.get(node, np.nan),
            'homo_selected': node in homo_set,
            'node_gain': g,
            'node_gain_net': g - lam if np.isfinite(g) else np.nan,
            'node_selected': node in nd_set,
            'node_gain_net_fl': g - lam_fl if np.isfinite(g) else np.nan,
            'node_gain_h0_mean': h0_mean.get(node, np.nan),
            'node_gain_h0_std': h0_std.get(node, np.nan),
            'node_fl_selected': node in nd_fl_set,
            'tree_gain': ga,
            'tree_wt_gain': ga * w if np.isfinite(ga) else np.nan,
            'tree_selected': node in tr_set,
        }
        for k, pass_dict in enumerate(wt_passes):
            row[f'tree_wt_gain_sum{k}'] = pass_dict.get(node, np.nan)
        if has_dp:
            tw = tw_gain.get(node, np.nan)
            row['tree_dp_tree_gain'] = tw
            row['tree_dp_tree_gain_net'] = (tw - lam_dp
                                            if np.isfinite(tw) else np.nan)
            row['tree_dp_selected'] = node in tr_dp_set
        rows.append(row)
    return pd.DataFrame(rows)


def _score_region_masks(masks, effect_mask, mask_active):
    """Score a list of boolean region masks against ground truth.

    Returns dict with f1, sens, spec, n_regions, region_sizes,
    region_tp_fracs (sorted largest-first).
    """
    mask_pred = np.zeros_like(effect_mask)
    sizes, tp_fracs = [], []

    for rmask in masks:
        mask_pred |= rmask
        n_vox = int(rmask.sum())
        sizes.append(n_vox)
        overlap = int((rmask & effect_mask).sum())
        tp_fracs.append(overlap / n_vox if n_vox > 0 else 0.0)

    f1, sens, spec = glow.mask.get_score(
        mask_pred=mask_pred, mask_target=effect_mask,
        mask_active=mask_active)

    order = np.argsort(sizes)[::-1]
    return dict(
        f1=f1, sens=sens, spec=spec,
        n_regions=len(masks),
        region_sizes=[sizes[i] for i in order],
        region_tp_fracs=[round(tp_fracs[i], 4) for i in order],
    )


def run_prune_compare(config, **kwargs):
    """Run GLOW pruning strategies (+ optional VBA / VBA-TFCE).

    For each AnalysisGLOW entry in ``ana_kwargs_dict``, runs a shared
    analysis then applies all 5 pruning strategies.  Non-GLOW entries
    (e.g. VBA, VBA-TFCE) are run independently.

    Emits one JSON result per method with F1/sens/spec and region-level
    metrics (sizes, TP fractions).  Also writes pruning diagnostics CSV.
    """
    from glow.experiment.prune import (prune, prune_node,
                                       prune_tree, prune_tree_dp)

    exp, effect = config.get_exp_eff(**kwargs)
    mask_active = exp.mask_idx > -1
    config_hash = config._config_hash()

    base = dict(
        effect_llr=effect.effect_llr,
        seed=int(effect.seed),
        vox_total=int(exp.y.shape[2]),
        vox_effect=int(effect.mask.sum()),
        config_hash=config_hash,
    )

    def _save(scores, label, analysis_name, time_sec):
        uid = str(uuid4())[:8]
        file_out = config.folder / OUT / f'{uid}_result.json'
        d = {**base, **scores,
             'label': label,
             'Analysis': analysis_name,
             'time_sec': time_sec,
             'uuid': uid}
        file_out.parent.mkdir(exist_ok=True, parents=True)
        with open(file_out, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)

    # ---- GLOW: shared analysis, 5 pruning strategies ----
    for _lbl, (Ana, ana_kw) in config.ana_kwargs_dict.items():
        if Ana.__name__ != 'AnalysisGLOW':
            continue

        n_perm_prune = ana_kw.get('n_perm_prune', 100)
        alpha_prune = ana_kw.get('alpha_prune', 0.05)
        exp_eff = ana_kw.get('prune_geom_exp_eff')

        start = time.time()
        ana = Ana(exp=exp, **ana_kw)
        glow_time = time.time() - start

        sig = ana.sig_reg_list
        children = ana.child_dict[0]

        prune_methods = {
            'GLOW-homo': prune(sig, children, exp,
                               n_perm=n_perm_prune, alpha_prune=alpha_prune),
            'GLOW-node': prune_node(sig, children, exp,
                                    n_perm=n_perm_prune, alpha=alpha_prune,
                                    exp_eff=exp_eff),
            'GLOW-node_fl': prune_node(sig, children, exp,
                                       n_perm=n_perm_prune,
                                       alpha=alpha_prune),
            'GLOW-tree': prune_tree(sig, children, exp),
        }
        if exp_eff is not None:
            prune_methods['GLOW-tree_dp'] = prune_tree_dp(
                sig, children, exp, exp_eff=exp_eff)

        for label, (reg_out_list, _info) in prune_methods.items():
            masks = []
            for reg_idx in reg_out_list:
                lm = glow.graph.get_label_map(
                    reg_idx_list=[reg_idx],
                    mask_idx=exp.mask_idx,
                    children=children)
                masks.append(lm > -1)
            scores = _score_region_masks(masks, effect.mask, mask_active)
            _save(scores, label, Ana.__name__, glow_time)

        if sig:
            diag_methods = {k.replace('GLOW-', ''): v
                            for k, v in prune_methods.items()}
            diag_df = _prune_diagnostics_df(sig, diag_methods)
            run_uuid = str(uuid4())[:8]
            csv_out = config.folder / OUT / f'{run_uuid}_diagnostics.csv'
            diag_df.to_csv(csv_out, index=False)
        break  # only one GLOW entry expected

    # ---- VBA / VBA-TFCE ----
    for label, (Ana, ana_kw) in config.ana_kwargs_dict.items():
        if Ana.__name__ == 'AnalysisGLOW':
            continue

        start = time.time()
        ana_vba = Ana(exp=exp, **ana_kw)
        vba_time = time.time() - start

        masks = [e.mask for e in ana_vba.effect_list]
        scores = _score_region_masks(masks, effect.mask, mask_active)
        _save(scores, label, Ana.__name__, vba_time)


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
