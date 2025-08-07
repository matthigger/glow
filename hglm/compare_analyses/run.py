import gzip
import json
import pathlib
import time
import traceback
import warnings
from uuid import uuid4

import cloudpickle as pickle
import numpy as np

import param
from hglm.effect import ExtenterSphere, ExtenterMinVar
from hglm.experiment import ExperimentScaled, AnalysisHGLM
from hglm.graph import get_f1, iter_topo
from hglm.mask import get_score


def get_max_f1(ana_hglm, mask_target):
    # load detail, find region corresponding to max f1
    f1 = get_f1(mask=mask_target,
                mask_idx=ana_hglm.exp.mask_idx,
                children=ana_hglm.child_dict[0])
    reg_max_f1 = f1.argmax()

    # compute scores
    mask_pred = np.zeros_like(mask_target)
    for vox in iter_topo(children=ana_hglm.child_dict[0],
                         num_leaf=ana_hglm.exp.y.shape[2],
                         node_start=reg_max_f1,
                         only_leaf=True):
        mask_pred[ana_hglm.exp.mask_idx == vox] = True
    f1, sens, spec = get_score(mask_pred=mask_pred,
                               mask_target=mask_target,
                               mask_active=ana_hglm.exp.mask_idx > -1)
    return dict(f1=f1, sens=sens, spec=spec, region=reg_max_f1)


def run_one_exp(seed, hotel_tr, rough=None):
    # allows us to catch numpy's warnings
    warnings.filterwarnings('error')
    np.seterr(all='warn')

    # trim experiment to reasonable size (for speedup)
    extenter = ExtenterSphere(radius=param.radius)
    mask = extenter(mask_idx=param.exp.mask_idx, seed=seed, contiguous=True)
    exp_masked = param.exp.apply_mask(mask)

    # scale normalize before sampling minimum variance (each feature given
    # equal weight in sampling extent)
    exp_masked = ExperimentScaled.from_exp(exp_masked)

    # sample effect space
    n = exp_masked.y.shape[2] * param.effect_perc
    extenter = ExtenterMinVar(n=n)

    # impose effect
    _exp, effect = exp_masked.impose_effect(extenter=extenter,
                                            seed=seed,
                                            hotel_tr=hotel_tr,
                                            rough=rough)

    for Ana, kwargs in param.ana_kwargs_list:
        # prep output file
        uuid = str(uuid4())[:8]
        file_out = folder_out / 'out' / f'{uuid}_result.json'

        # run analysis
        start = time.time()
        if param.error_save:
            # catch errors and dump to json if any occur (allows us to
            # continue with experiment in event of errors)
            try:
                ana = Ana(exp=_exp, alpha_fwer=param.alpha_fwer, **kwargs)
            except Exception as e:
                d = {'error_msg': traceback.format_exc(),
                     'method': Ana.__name__,
                     'hotel_tr': hotel_tr,
                     'seed': int(seed)}
                print(f'error: {d}')
                file_out = str(file_out).replace('out', 'error')

                file_out = pathlib.Path(file_out)
                file_out.parent.mkdir(exist_ok=True, parents=True)
                with open(file_out, 'w') as f:
                    json.dump(d, f, sort_keys=True, indent=4)
                continue

        else:
            # no error catching, will stop all experiments if any error
            ana = Ana(exp=_exp, alpha_fwer=param.alpha_fwer, **kwargs)
        total_time_sec = time.time() - start

        # build mask of predicted area (union of all effect masks)
        mask_pred = np.zeros(ana.exp.mask_idx.shape, dtype=bool)
        for _effect in ana.effect_list:
            mask_pred |= _effect.mask

        # compute scores
        f1, sens, spec = get_score(mask_pred=mask_pred,
                                   mask_target=effect.mask,
                                   mask_active=exp_masked.mask_idx > -1)

        stat_name = ana.get_stat.__name__.replace('get_', '')

        # dump summary
        d = {'hotel_tr': hotel_tr,
             'seed': int(seed),
             'rough': effect.rough,
             'stat': stat_name,
             'Analysis': Ana.__name__,
             'f1': f1,
             'sens': sens,
             'spec': spec,
             'uuid': uuid,
             'time_sec': total_time_sec}
        with open(file_out, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)

        if param.maxf1 and Ana == AnalysisHGLM:
            # compute max f1 stats, store as a distinct output
            # "analysis" (its not really)
            d.update(get_max_f1(ana_hglm=ana, mask_target=effect.mask))
            d['Analysis'] = 'AnalysisHGLM-maxF1'
            d['time_sec'] = ''
            d['region'] = float(d['region'])
            with open(file_out.with_stem(f'{uuid}_maxf1'), 'w') as f:
                json.dump(d, f, sort_keys=True, indent=4)

        if param.detail_save:
            # dump detail
            file_out = folder_out / 'out' / f'{uuid}_detail.p.gz'
            with gzip.open(file_out, 'wb') as f:
                pickle.dump((ana, effect), f)


if __name__ == '__main__':
    from data import prep_folder_out
    from tqdm import tqdm
    from joblib import Parallel, delayed
    from itertools import product
    from param import seed_all, hotel_tr_all, rough_all

    # prep folder_out
    folder_out = prep_folder_out(param.folder_out,
                                 files_to_copy=(param.__file__,))

    kwargs_list = [dict(seed=s, hotel_tr=h, rough=r)
                   for s, h, r in product(seed_all, hotel_tr_all, rough_all)]

    if param.n_jobs not in (0, 1):
        r = Parallel(n_jobs=param.n_jobs, verbose=10)(
            delayed(run_one_exp)(**kwargs) for kwargs in tqdm(kwargs_list))
    else:
        for kwargs in tqdm(kwargs_list):
            run_one_exp(**kwargs)
