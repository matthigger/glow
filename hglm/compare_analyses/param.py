import numpy as np

import hglm

# where output results are stored (each run of script yields its own folder)
folder_out = '/home/matt/Dropbox/pnl_hglm/results'

# controls number of repetitions
seed_all = np.arange(100)

# hotelling's trace [0, inf) describes severity of effect
hotel_tr_all = np.logspace(np.log10(.001), np.log10(.05), 9)

# roughness (0 to 1 inclusive)
rough_all = None,

# to speed up analysis, random voxel is chosen and dilated to this radius.
# only these voxels are included in the analysis
radius = 4

# effect size, as ratio to total voxels in experiment
effect_perc = .2

# FWER control
alpha_fwer = .05

# toggles parallel, 0 or 1 processes serial (good for debug).  else this is the
# number of threads to use.  (-1 for all of them)
n_jobs = -1

# parameters to be passed to Analysis constructor
kwargs_hglm = dict(n_perm=100,
                   n_perm_adj=10,
                   n_perm_tailor=100,
                   min_size=1,
                   alpha_tailor=.05)
kwargs_tfce = dict(n_perm=100)

# # analyses to run
# ana_kwargs_list = ((hglm.experiment.AnalysisHGLM, kwargs_hglm),
#                    (hglm.experiment.AnalysisTFCE, kwargs_tfce))

# compare all stats
ana_kwargs_list = list()
for get_stat in hglm.experiment.mancova.stat_dict.values():
    kwargs = kwargs_hglm | dict(get_stat=get_stat)
    ana_kwargs_list.append((hglm.experiment.AnalysisHGLM, kwargs))

# saves output python objects (memory expensive)
detail_save = False

# writes json with input state causing an errors in Analysis.run(),
# continues to next experiment
error_save = True

# if True, computes stats on max f1 region in HGLM analysis (allows us to
# distinguish between segmentation & discovery errors)
maxf1 = True

source = 'hcp'

match source:
    case 'hcp':
        # human connectome project data
        folder = '/home/matt/Dropbox/pnl_hglm/data/hcp100_lowres/image'
        exp_hcp = hglm.experiment.ExperimentImageOnly.from_search(
            folder=folder,
            sbj_regex=r'[\d]{6}',
            img_glob_dict={'FA': '*_FA.nii.gz',
                           'MD': '*_MD.nii.gz'})
        exp = exp_hcp.sample_x(a=2, seed=0, add_bias=True)
    case 'awgn':
        # additive white gaussian noise
        exp = hglm.experiment.Experiment.from_gauss(seed=0,
                                                    shape=(5, 5, 5),
                                                    a=2,
                                                    b=2,
                                                    num_img=100)
    case _:
        raise AttributeError(f'data source not recognized: {source}')
