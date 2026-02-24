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
        d = {'hotel_tr': effect.hotel_tr,
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
                     'hotel_tr': effect.hotel_tr,
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
        d = {'hotel_tr': effect.hotel_tr,
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
    """Build per-region diagnostics from the three pruning results.

    Returns a DataFrame with one row per significant region and columns
    for each method's diagnostics, suitable for the viewer's --csv flag.
    """
    import pandas as pd

    homo_regs, homo_info = methods['homo']
    geom_regs, dp_info = methods['geom_prior']
    adj_regs, adj_info = methods['adjusted_ll']

    homo_set, geom_set, adj_set = set(homo_regs), set(geom_regs), set(adj_regs)
    lam = dp_info.get('lam', 0.0)
    gain_geom = dp_info.get('gain', {})
    gain_adj = adj_info.get('gain_per_node', {})
    weights = adj_info.get('weights', {})

    rows = []
    for node in sig:
        g = gain_geom.get(node, np.nan)
        ga = gain_adj.get(node, np.nan)
        w = weights.get(node, 1.0)
        rows.append({
            'region_idx': node,
            'homo_pval': homo_info.get(node, np.nan),
            'homo_selected': node in homo_set,
            'geom_gain': g,
            'geom_gain_net': g - lam if np.isfinite(g) else np.nan,
            'geom_selected': node in geom_set,
            'adj_ll_gain': ga,
            'adj_ll_wt_gain': ga * w if np.isfinite(ga) else np.nan,
            'adj_ll_selected': node in adj_set,
        })
    return pd.DataFrame(rows)


def run_prune_compare(config, **kwargs):
    """Run one AnalysisGLOW and apply all three pruning methods.

    Emits one JSON result per method (homo, geom_prior, adjusted_ll)
    with the same schema as run_ana, so plot.py works unchanged.
    Also writes a per-region diagnostics CSV for the viewer's --csv flag.
    """
    from glow.experiment.prune import prune, prune_dp, prune_adjusted_ll

    exp, effect = config.get_exp_eff(**kwargs)

    _, (Ana, ana_kw) = next(iter(config.ana_kwargs_dict.items()))
    n_perm_prune = ana_kw.get('n_perm_prune', 100)
    alpha_prune = ana_kw.get('alpha_prune', 0.05)

    start = time.time()
    ana = Ana(exp=exp, **ana_kw)
    total_time_sec = time.time() - start

    sig = ana.sig_reg_list
    children = ana.child_dict[0]

    methods = {
        'homo': prune(sig, children, exp,
                       n_perm=n_perm_prune, alpha_prune=alpha_prune),
        'geom_prior': prune_dp(sig, children, exp,
                                n_perm=n_perm_prune, alpha=alpha_prune),
        'adjusted_ll': prune_adjusted_ll(sig, children, exp),
    }

    run_uuid = str(uuid4())[:8]

    for label, (reg_out_list, _info) in methods.items():
        mask_pred = np.zeros(ana.exp.mask_idx.shape, dtype=bool)
        for reg_idx in reg_out_list:
            label_map = glow.graph.get_label_map(
                reg_idx_list=[reg_idx],
                mask_idx=exp.mask_idx,
                children=children)
            mask_pred |= (label_map > -1)

        f1, sens, spec = glow.mask.get_score(mask_pred=mask_pred,
                                             mask_target=effect.mask,
                                             mask_active=exp.mask_idx > -1)
        uuid = str(uuid4())[:8]
        file_out = config.folder / OUT / f'{uuid}_result.json'
        d = {'hotel_tr': effect.hotel_tr,
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

    if sig:
        diag_df = _prune_diagnostics_df(sig, methods)
        csv_out = config.folder / OUT / f'{run_uuid}_diagnostics.csv'
        diag_df.to_csv(csv_out, index=False)


if __name__ == '__main__':
    from glow.benchmark.config import Config

    # quick test
    ana_kwargs_dict = {'GLOW': (glow.experiment.AnalysisGLOW,
                                dict(n_perm=100)),
                       'VBA': (glow.experiment.AnalysisVBA, dict(n_perm=100))}
    config = Config(label='quick_test', source='wgn', n_seed=3,
                    hotel_tr_all=[0, 1], wgn_shape=(3, 3),
                    ana_kwargs_dict=ana_kwargs_dict)
    config.run_all(run_fnc=run_ana)
