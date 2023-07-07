import shutil
from datetime import datetime

import cloudpickle
from tqdm import tqdm

from hrba.experiment import *
from hrba.sample_effect import *

# build experiment with strong effect to be found (whole region)
p_val_all = np.geomspace(.1, .01, 11)
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

ana_idx_max = n_repeat * p_val_all.size * len(analysis_obj_tup)
z_width = int(np.ceil(np.log10(ana_idx_max)))
ana_idx = 0
with tqdm(desc='experiment (pval-seed-method pair)',
          total=p_val_all.size * n_repeat * len(analysis_obj_tup)) as pbar:
    for seed in range(n_repeat):
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
                s_ana_idx = str(ana_idx).zfill(z_width)
                file_out = folder_out / f'analysis{s_ana_idx}.p'
                with open(file_out, 'wb') as file:
                    cloudpickle.dump((p_val, seed, Ana.__name__, ana, effect),
                                     file=file)

                ana_idx += 1
                pbar.update(1)
