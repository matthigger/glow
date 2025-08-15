import gzip
import json
import pathlib
import time
import traceback
import warnings
from uuid import uuid4

import cloudpickle as pickle
import numpy as np
import pandas as pd

import param
from hglm.effect import ExtenterSphere, ExtenterMinVar
from hglm.experiment import ExperimentScaled
from hglm.graph import get_label_map
from hglm.mask import get_score


def re_tailor(ana, alpha_tailor_all, mask_target, **kwargs):
    """ re-runs tailor, returns dataframe of scores at every alpha_tailor

    Args:
        alpha_tailor_all (np.array): all alpha_tailor values to run
        mask_target (mask): effect mask (target)

    Returns:
        df (pd.DataFrame):
    """
    rows = list()
    children = ana.child_dict[0]
    _pval_dict = ana.homo_pval_dict
    for _alpha_tailor in alpha_tailor_all:
        # re-tailor
        reg_out_list, _pval_dict = ana.tailor(sig_reg_list=ana.sig_reg_list,
                                              alpha_tailor=_alpha_tailor,
                                              exp=ana.exp,
                                              children=children,
                                              _pval_dict=_pval_dict,
                                              **kwargs)

        # compute scores
        label_map = get_label_map(reg_out_list,
                             mask_idx=ana.exp.mask_idx,
                             children=children)
        f1, sens, spec = get_score(mask_pred=label_map > -1,
                                   mask_target=mask_target,
                                   mask_active=ana.exp.mask_idx > -1)
        rows.append(dict(alpha_tailor=_alpha_tailor, f1=f1, sens=sens,
                         spec=spec))

    return pd.DataFrame(rows)


def run_one_exp(seed, hotel_tr):
    # allows us to catch numpy's warnings
    warnings.filterwarnings('error')
    np.seterr(all='warn')

    # trim experiment to reasonable size (for speedup)
    if param.radius is None:
        exp = param.exp
    else:
        extenter = ExtenterSphere(radius=param.radius)
        mask = extenter(mask_idx=param.exp.mask_idx, seed=seed,
                        contiguous=True)
        exp = param.exp.apply_mask(mask)

    # scale normalize before sampling minimum variance (each feature given
    # equal weight in sampling extent)
    exp = ExperimentScaled.from_exp(exp)

    # sample effect space
    n = exp.y.shape[2] * param.effect_perc
    extenter = ExtenterMinVar(n=n)

    # impose effect
    _exp, effect = exp.impose_effect(extenter=extenter,
                                     seed=seed,
                                     hotel_tr=hotel_tr)

    for label, (Ana, kwargs) in param.ana_kwargs_dict.items():
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
                     'label': label,
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
                                   mask_active=exp.mask_idx > -1)

        stat_name = ana.get_stat.__name__.replace('get_', '')

        # dump summary
        d = {'hotel_tr': hotel_tr,
             'seed': int(seed),
             'stat': stat_name,
             'label': label,
             'Analysis': Ana.__name__,
             'f1': f1,
             'sens': sens,
             'spec': spec,
             'uuid': uuid,
             'vox_total': int(ana.exp.y.shape[2]),
             'vox_effect': int(effect.mask.sum()),
             'time_sec': total_time_sec}
        with open(file_out, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)

        if param.detail_save:
            # dump detail
            file_out = folder_out / 'out' / f'{uuid}_detail.p.gz'
            with gzip.open(file_out, 'wb') as f:
                pickle.dump((ana, effect), f)

        if param.alpha_tailor_all is not None and ana.sig_reg_list:
            df = re_tailor(ana, param.alpha_tailor_all,
                           mask_target=effect.mask,
                           n_perm=param.kwargs_hglm['n_perm_tailor'])
            df['seed'] = int(seed)
            df['hotel_tr'] = hotel_tr
            df.to_csv(folder_out / 'out' / f'{uuid}_alpha_tailor.csv')


if __name__ == '__main__':
    from data import prep_folder_out
    from tqdm import tqdm
    from joblib import Parallel, delayed
    from itertools import product
    from param import seed_all, hotel_tr_all

    # prep folder_out
    folder_out = prep_folder_out(files_to_copy=(param.__file__,))

    kwargs_list = [dict(seed=s, hotel_tr=h)
                   for s, h in product(seed_all, hotel_tr_all)]

    if param.n_jobs not in (0, 1):
        r = Parallel(n_jobs=param.n_jobs, verbose=10)(
            delayed(run_one_exp)(**kwargs) for kwargs in tqdm(kwargs_list))
    else:
        for kwargs in tqdm(kwargs_list):
            run_one_exp(**kwargs)
