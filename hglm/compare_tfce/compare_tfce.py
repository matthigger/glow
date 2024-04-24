import gzip
import json
import shutil
import time
import traceback
import warnings
from datetime import datetime
from uuid import uuid4

import cloudpickle as pickle
from sklearn.metrics import roc_auc_score

from hglm.effect import *
from hglm.experiment import *
from hglm.graph import get_f1
from hglm.mask import get_score

# where output results are stored (each run of script yields its own folder)
folder_out = '/home/matt/Dropbox/pnl_hglm/results'

# number of effects to model
n_repeat = 32 * 3

# p_val describes severity of effect (assuming typical F test assumptions
# ...not valid but still useful to quantify how difficult effect is)
p_val_all = np.geomspace(.25, .001, 9)

# to speed up analysis, random voxel is chosen and dilated to this radius.
# only these voxels are included in the analysis
radius = 4

# effect size, as ratio to total voxels in experiment
effect_perc = .2

# FWER control
alpha = .05

# toggles parallel, 0 or 1 processes non-parallel (good for debug).  else this
# is the number of threads to use.  (-1 for all of them)
n_jobs = -1

# parameters to be passed to Analysis constructor
analysis_kwargs = {'AnalysisHGLM': dict(n_perm=100,
                                        n_perm_adj=10,
                                        min_size_discover=1),
                   'AnalysisTFCE': dict(n_perm=100)}

# saves output python objects (memory expensive)
detail_save = True

# writes json with input state causing an errors in Analysis.run(),
# continues to next experiment
error_save = True

# if True, computes stats on max f1 region in HGLM analysis (allows us to
# distinguish between segmentation & discovery errors)
maxf1 = True

# input data
folder = '/home/matt/Dropbox/pnl_hglm/data/hcp100_lowres/image'
exp_hcp = ExperimentImageOnly.from_search(folder=folder,
                                          sbj_regex='[\d]{6}',
                                          img_glob_dict={'FA': '*_FA.nii.gz',
                                                         'MD': '*_MD.nii.gz'})
exp_hcp = exp_hcp.sample_x(a=2, seed=1)

analysis_obj_tup = (AnalysisHGLM, AnalysisTFCE)

# prep folder_out
timestamp = datetime.now().strftime('%y%b%d-%H%M')
folder_out = pathlib.Path(folder_out)
assert folder_out.exists()
folder_out = pathlib.Path(folder_out) / f'./exp_{timestamp}'
if folder_out.exists():
    choice = input(f'folder exists: {folder_out}\ndelete? [y/n]:')
    if choice != 'y':
        raise Exception('quitting')
    shutil.rmtree(folder_out)
folder_out.mkdir()

# store copy of this script (to read experiment params above)
shutil.copy(__file__, folder_out / pathlib.Path(__file__).name)
folder = pathlib.Path(__file__).parent
shutil.copy(folder / 'compare_tfce_plot.ipynb',
            folder_out / 'compare_tfce_plot.ipynb')


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


def run_one_exp(seed):
    # allows us to catch numpy's warnings
    warnings.filterwarnings('error')
    np.seterr(all='warn')

    # trim experiment to reasonable size (for speedup)
    extenter = ExtenterSphere(radius=radius)
    mask = extenter(mask_idx=exp_hcp.mask_idx, seed=seed, contiguous=True)
    exp = exp_hcp.apply_mask(mask)

    # pre-whiten exp
    exp = ExperimentWhitened.from_exp(exp)

    # sample effect space
    n = exp.y.shape[2] * effect_perc
    extenter = ExtenterMinVar(n=n)
    mask_target = extenter(y=exp.y, mask_idx=exp.mask_idx, seed=seed)

    # shuffle p value order (better sampling across threads)
    np.random.shuffle(p_val_all)
    for p_val in p_val_all:
        # impose effect
        _exp, effect = exp.impose_effect(seed=seed, mask=mask_target,
                                         p_val=p_val)

        for Ana in analysis_obj_tup:
            # prep output file
            uuid = str(uuid4())[:8]
            file_out = folder_out / f'out_{uuid}.json'

            # prep analysis
            kwargs = analysis_kwargs[Ana.__name__]

            start = time.time()
            if error_save:
                try:
                    ana = Ana(exp=_exp, alpha=alpha, **kwargs)
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
                ana = Ana(exp=_exp, alpha=alpha, **kwargs)
            total_time_sec = time.time() - start

            # build mask of predicted area (union of all effect masks)
            mask_pred = np.zeros(ana.exp.mask_idx.shape, dtype=bool)
            for _effect in ana.effect_list:
                mask_pred |= _effect.mask

            # score
            f1, sens, spec = get_score(mask_pred=mask_pred,
                                       mask_target=effect.mask,
                                       mask_active=exp.mask_idx > -1)

            # get stat for AUC compute
            if isinstance(ana, AnalysisHGLM):
                # setting alpha=2 makes all regions "significant", discover()
                # produces a disjoint set of regions which cover the space
                # of voxels, choosing to maximize z stat greedily (a bit of
                # wasted compute here ... full Effect not needed)
                effect_list = ana.discover(pval=ana.p_val, alpha=2,
                                           priority=ana.z_stat[0, :],
                                           children=ana.child_dict[0], exp=exp)

                # build stat per voxel
                x = np.zeros(exp.mask_idx.shape)
                for eff in effect_list:
                    x += ana.z_stat[0, eff.reg_idx] * eff.mask
                y_score = x[exp.mask_idx > -1]

            elif isinstance(ana, AnalysisTFCE):
                y_score = ana.tfce_stat[0, :]
            else:
                raise RuntimeError(f'Analysis not recognized: {type(ana)}')

            auc = roc_auc_score(y_score=y_score,
                                y_true=effect.mask[exp.mask_idx > -1])

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

            if maxf1 and Ana == AnalysisHGLM:
                # compute max f1 stats, store as a distinct output
                # "analysis" (its not really)
                d.update(get_max_f1(ana_hglm=ana, mask_target=effect.mask))
                d['Analysis'] = 'AnalysisHGLM-maxF1'
                d['time_sec'] = ''
                d['region'] = float(d['region'])
                with open(file_out.with_stem(f'out_{uuid}_maxf1'), 'w') as f:
                    json.dump(d, f, sort_keys=True, indent=4)

            if detail_save:
                # dump detail
                file_out = folder_out / f'out_{uuid}_detail.p.gz'
                with gzip.open(file_out, 'wb') as f:
                    pickle.dump((ana, effect), f)


if __name__ == '__main__':
    if n_jobs not in (0, 1):
        r = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_one_exp)(seed) for seed in tqdm(range(n_repeat)))
    else:
        for seed in tqdm(range(n_repeat)):
            run_one_exp(seed)
