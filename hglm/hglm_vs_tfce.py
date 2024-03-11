import gzip
import json
import shutil
import time
import traceback
import warnings
from datetime import datetime
from uuid import uuid4

import cloudpickle as pickle
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, recall_score, confusion_matrix

from hglm.effect import *
from hglm.experiment import *

# where output results are stored (each run of script yields its own folder)
folder_out = '/home/matt/Dropbox/pnl_hglm/results'

# number of effects to model
n_repeat = 100

# p_val describes severity of effect (assuming typical F test assumptions
# ...not valid but still useful to quantify how difficult effect is)
p_val_all = np.linspace(.15, .03, 7)

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
analysis_kwargs = {'AnalysisHGLM': {'n_permute': 100},
                   'AnalysisTFCE': {'n_permute': 100}}

# saves output python objects (memory expensive)
detail_save = True

# writes json with input state causing an errors in Analysis.run(),
# continues to next experiment
error_save = True

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

# store copy of script (to read experiment params above)
shutil.copy(__file__, folder_out / pathlib.Path(__file__).name)
shutil.copy('hglm_vs_tfce_plot.ipynb', folder_out / 'hglm_vs_tfce_plot.ipynb')

# input data
folder = '/home/matt/Dropbox/pnl_hglm/data/hcp100_lowres/image'
exp_hcp = ExperimentImageOnly.from_search(folder=folder,
                                          sbj_regex='[\d]{6}',
                                          img_glob_dict={'FA': '*_FA.nii.gz',
                                                         'MD': '*_MD.nii.gz'})
exp_hcp = exp_hcp.sample_x(a=2)

analysis_obj_tup = (AnalysisHGLM, AnalysisTFCE)


def get_score(ana, effect):
    """ gets f1, sens, spec scores per analysis given ground truth effect
    """
    # build mask of predicted area (union of all effect masks)
    mask_pred = np.zeros(ana.exp.mask_idx.shape, dtype=bool)
    for _effect in ana.effect_list:
        mask_pred |= _effect.mask

    # build y_true / y_pred in sklearn format
    mask_active = ana.exp.mask_idx > -1
    y_true = effect.mask[mask_active]
    y_pred = mask_pred[mask_active]

    # compute scores
    f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    sens = recall_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    spec = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[0, 1])

    return f1, sens, spec


def run_one_exp(seed):
    # allows us to catch numpy's warnings
    warnings.filterwarnings('error')
    np.seterr(all='warn')

    # trim experiment to reasonable size (for speedup)
    # this trimming is re-done until a contiguous experiment is found
    _seed_extenter = seed
    while True:
        extenter = ExtenterSphere(radius=radius)
        mask_all = extenter(mask_idx=exp_hcp.mask_idx, seed=_seed_extenter)
        exp = exp_hcp.apply_mask(mask_all)

        _, n_components = label(exp.mask_idx >= 0)
        if n_components == 1:
            break

        # get a new seed (from previous)
        rng = np.random.default_rng(seed=_seed_extenter)
        _seed_extenter = rng.integers(low=np.iinfo(_seed_extenter).min,
                                      high=np.iinfo(_seed_extenter).max,
                                      size=1)[0]

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

            # score
            f1, sens, spec = get_score(ana, effect)

            # dump summary
            d = {'p_val': p_val,
                 'seed': seed,
                 'Analysis': Ana.__name__,
                 'f1': f1,
                 'sens': sens,
                 'spec': spec,
                 'uuid': uuid,
                 'time_sec': total_time_sec}
            with open(file_out, 'w') as f:
                json.dump(d, f, sort_keys=True, indent=4)

            if detail_save:
                # dump detail
                file_out = folder_out / f'out_{uuid}_detail.p.gz'
                with gzip.open(file_out, 'wb') as f:
                    pickle.dump((ana, effect), f)


if n_jobs:
    r = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_one_exp)(seed) for seed in range(n_repeat))
else:
    for seed in range(n_repeat):
        run_one_exp(seed)
