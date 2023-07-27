import pickle
import shutil
from datetime import datetime
from uuid import uuid4

from joblib import Parallel, delayed
from sklearn.metrics import f1_score, recall_score, confusion_matrix

from hrba.experiment import *
from hrba.sample_effect import *

# number of effects to model
n_repeat = 100

# p_val describes severity of effect (assuming typical F test assumptions
# ...not valid but still useful to quantify how difficult effect is)
p_val_all = np.geomspace(.15, .03, 15)

# to speed up analysis, random voxel is chosen and dilated to this radius.
# only these voxels are included in the analysis
radius = 5

# effect size, as ratio to total voxels in experiment
effect_perc = .2

# number of permutations in permutation testing
n_permute = 100

# FWER control
alpha = .05

# parameters to be passed to Analysis constructor
# min_reg_size=5 implies HRBA will not test the hypothesis that any region
# smaller than 5 voxels contains an effect
analysis_kwargs = {'AnalysisHRBA': {'min_reg_size': 10},
                   'AnalysisTFCE': dict()}

# prep folder_out
timestamp = datetime.now().strftime('%y%b%d-%H%M')
folder_out = pathlib.Path(f'./exp_{timestamp}').resolve()
if folder_out.exists():
    choice = input('folder exists, delete? [y/n]:')
    if choice != 'y':
        raise Exception('quitting')
    shutil.rmtree(folder_out)
folder_out.mkdir()

# store copy of script (to read experiment params above)
shutil.copy(__file__, folder_out / pathlib.Path(__file__).name)

# input data
folder = '/home/matt/Dropbox/pnl_hrba/data/hcp100_lowres/image'
exp_hcp = Experiment.from_search(folder=folder,
                                 sbj_regex='[\d]{6}',
                                 img_glob_dict={'FA': '*_FA.nii.gz'})
exp_hcp.sample_x(a=2)

analysis_obj_tup = (AnalysisTFCE, AnalysisHRBA)


def get_score(ana, effect):
    """ gets f1, sens, spec scores per analysis given ground truth effect
    """
    # build mask of predicted area (union of all effect masks)
    mask_pred = np.zeros(ana.exp.mask_idx.shape, dtype=bool)
    for _effect in ana.effect_tup:
        mask_pred |= _effect.mask

    # build y_true / y_pred in sklearn format
    mask_active = ana.exp.mask_idx > -1
    y_true = effect.mask[mask_active]
    y_pred = mask_pred[mask_active]

    # compute scores
    f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    sens = recall_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    spec = conf_mat[0, 0] / (conf_mat[0, 0] + conf_mat[1, 0])

    return f1, sens, spec


def run_one_exp(seed):
    # trim experiment to reasonable size (for speedup)
    extenter = ExtenterSphere(radius=radius)
    mask_all = extenter(mask_idx=exp_hcp.mask_idx, seed=seed)
    exp = exp_hcp.apply_mask(mask_all)

    # sample effect space
    n = exp.y.shape[2] * effect_perc
    assert analysis_kwargs['AnalysisHRBA']['min_reg_size'] <= n, \
        'min_reg_size larger than target effect'
    extenter = ExtenterMinVar(n=n)
    mask_target = extenter(y=exp.y, mask_idx=exp.mask_idx, seed=seed)

    for p_val in p_val_all:
        # impose effect
        _exp, effect = exp.impose_effect(seed=seed, mask=mask_target,
                                         p_val=p_val)

        for Ana in analysis_obj_tup:
            # run analysis
            kwargs = analysis_kwargs[Ana.__name__]
            ana = Ana(exp=_exp, alpha=alpha, n_permute=n_permute, **kwargs)
            ana.run(verbose=False)

            # score
            f1, sens, spec = get_score(ana, effect)

            # dump
            file_out = folder_out / f'result_{str(uuid4())[:8]}.p'
            with open(file_out, 'wb') as file:
                pickle.dump((p_val, seed, Ana.__name__, f1, sens, spec),
                            file=file)


r = Parallel(n_jobs=-2, verbose=10)(
    delayed(run_one_exp)(seed) for seed in range(n_repeat))
