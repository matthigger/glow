import shutil
from datetime import datetime
from uuid import uuid4

import cloudpickle
from joblib import Parallel, delayed

from hrba.experiment import *
from hrba.sample_effect import *

# build experiment with strong effect to be found (whole region)
p_val_all = np.geomspace(.15, .03, 15)
effect_perc = .2
n_permute = 100
alpha = .05
radius = 7
n_repeat = 100

# prep folder_out
timestamp = datetime.now().strftime('%y%b%d-%H%M')
folder_out = pathlib.Path(f'./exp_{timestamp}').resolve()
if folder_out.exists():
    choice = input('folder exists, delete? [y/n]:')
    if choice != 'y':
        raise Exception('quitting')
    shutil.rmtree(folder_out)
folder_out.mkdir()

# store copy of script (to read hyperparams)
shutil.copy(__file__, folder_out / pathlib.Path(__file__).name)

# input data
folder = '/home/matt/Dropbox/pnl_hrba/data/hcp100_lowres/image'
exp_hcp = Experiment.from_search(folder=folder,
                                 sbj_regex='[\d]{6}',
                                 img_glob_dict={'FA': '*_FA.nii.gz'})
exp_hcp.sample_x(a=2)

analysis_obj_tup = (AnalysisTFCE, AnalysisHRBA)


def run_one_exp(seed):
    # trim experiment to reasonable size (for speedup)
    extenter = ExtenterSphere(radius=radius)
    mask_all = extenter(mask_idx=exp_hcp.mask_idx, seed=seed)
    exp = exp_hcp.apply_mask(mask_all)

    # sample effect space
    extenter = ExtenterMinVar(n=exp.y.shape[2] * effect_perc)
    mask_target = extenter(y=exp.y, mask_idx=exp.mask_idx, seed=seed)

    for p_val in p_val_all:
        # impose effect
        _exp, effect = exp.impose_effect(seed=seed, mask=mask_target,
                                         p_val=p_val)

        for Ana in analysis_obj_tup:
            ana = Ana(exp=_exp, alpha=alpha, n_permute=n_permute)
            ana.run(verbose=False)

            # dump
            file_out = folder_out / f'analysis_{str(uuid4())[:8]}.p'
            with open(file_out, 'wb') as file:
                cloudpickle.dump((p_val, seed, Ana.__name__, ana, effect),
                                 file=file)


r = Parallel(n_jobs=-2, verbose=10)(
    delayed(run_one_exp)(seed) for seed in range(n_repeat))
