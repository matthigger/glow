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
