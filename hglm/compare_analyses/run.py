import gzip
import json
import time
import traceback
import warnings
from uuid import uuid4

import cloudpickle as pickle
import numpy as np
from sklearn.metrics import roc_auc_score

import param
from hglm.effect import ExtenterSphere, ExtenterMinVar
from hglm.experiment import ExperimentWhitened, AnalysisHGLM, AnalysisTFCE
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


def get_auc(ana, mask_target):
    # get stat for AUC compute
    if isinstance(ana, AnalysisHGLM):
        # setting alpha=2 makes all regions "significant", discover()
        # produces a disjoint set of regions which cover the space
        # of voxels, choosing to maximize z stat greedily (a bit of
        # wasted compute here ... full Effect not needed)
        effect_list = ana.discover(pval=ana.p_val, alpha=2,
                                   priority=ana.z_stat[0, :],
                                   children=ana.child_dict[0],
                                   exp=ana.exp)

        # build stat per voxel
        x = np.zeros(ana.exp.mask_idx.shape)
        for eff in effect_list:
            x += ana.z_stat[0, eff.reg_idx] * eff.mask
        y_score = x[ana.exp.mask_idx > -1]

    elif isinstance(ana, AnalysisTFCE):
        y_score = ana.tfce_stat[0, :]
    else:
        raise RuntimeError(f'Analysis not recognized: {type(ana)}')

    return roc_auc_score(y_score=y_score,
                         y_true=mask_target[ana.exp.mask_idx > -1])


def run_one_exp(seed):
    # allows us to catch numpy's warnings
    warnings.filterwarnings('error')
    np.seterr(all='warn')

    # trim experiment to reasonable size (for speedup)
    extenter = ExtenterSphere(radius=param.radius)
    mask = extenter(mask_idx=param.exp.mask_idx, seed=seed, contiguous=True)
    exp_masked = param.exp.apply_mask(mask)

    # pre-whiten exp
    exp_masked = ExperimentWhitened.from_exp(exp_masked)

    # sample effect space
    n = exp_masked.y.shape[2] * param.effect_perc
    extenter = ExtenterMinVar(n=n)
    mask_target = extenter(y=exp_masked.y,
                           mask_idx=exp_masked.mask_idx,
                           seed=seed)

    # shuffle p value order (better sampling across threads)
    np.random.shuffle(param.p_val_all)
    for p_val in param.p_val_all:
        # impose effect
        _exp, effect = exp_masked.impose_effect(seed=seed,
                                                mask=mask_target,
                                                p_val=p_val)

        for Ana in param.analysis_obj_tup:
            # prep output file
            uuid = str(uuid4())[:8]
            file_out = folder_out / 'out' / f'{uuid}_result.json'

            # run analysis
            kwargs = param.analysis_kwargs[Ana.__name__]
            start = time.time()
            if param.error_save:
                # catch errors and dump to json if any occur (allows us to
                # continue with experiment in event of errors)
                try:
                    ana = Ana(exp=_exp, alpha=param.alpha, **kwargs)
                except Exception as e:
                    d = {'error_msg': traceback.format_exc(),
                         'method': Ana.__name__,
                         'p_val': p_val,
                         'seed': seed}
                    print(f'error: {d}')
                    file_out = str(file_out).replace('out', 'error')
                    with open(file_out, 'w') as f:
                        json.dump(d, f, sort_keys=True, indent=4)
                    continue

            else:
                # no error catching, will stop all experiments if any error
                ana = Ana(exp=_exp, alpha=param.alpha, **kwargs)
            total_time_sec = time.time() - start

            # build mask of predicted area (union of all effect masks)
            mask_pred = np.zeros(ana.exp.mask_idx.shape, dtype=bool)
            for _effect in ana.effect_list:
                mask_pred |= _effect.mask

            # compute scores
            f1, sens, spec = get_score(mask_pred=mask_pred,
                                       mask_target=effect.mask,
                                       mask_active=exp_masked.mask_idx > -1)
            auc = get_auc(ana, mask_target=effect.mask)

            # dump summary
            d = {'p_val': p_val,
                 'seed': seed,
                 'Analysis': Ana.__name__,
                 'f1': f1,
                 'sens': sens,
                 'spec': spec,
                 'auc': auc,
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

    # prep folder_out
    folder_out = prep_folder_out(param.folder_out,
                                 files_to_copy=(param.__file__,))

    if param.n_jobs not in (0, 1):
        r = Parallel(n_jobs=param.n_jobs, verbose=10)(
            delayed(run_one_exp)(seed) for seed in tqdm(range(param.n_repeat)))
    else:
        for seed in tqdm(range(param.n_repeat)):
            run_one_exp(seed)
