import numpy as np

import hglm

# controls number of repetitions
seed_all = np.arange(10)

# hotelling's trace [0, inf) describes severity of effect
# hotel_tr_all = np.logspace(np.log10(.03), np.log10(1), 15)
hotel_tr_all = .15,

# roughness (0 to 1 inclusive)
rough_all = None,
rough_all = np.linspace(0, 1, 15)

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

# analyses to run (use plot_result.ipynb to view result)
ana_kwargs_list = ((hglm.experiment.AnalysisHGLM, kwargs_hglm),
                   (hglm.experiment.AnalysisTFCE, kwargs_tfce))

# # compare all stats (use plot_result_stat.ipynb to view result)
# ana_kwargs_list = list()
# for get_stat in hglm.experiment.mancova.stat_dict.values():
#     kwargs = kwargs_hglm | dict(get_stat=get_stat)
#     ana_kwargs_list.append((hglm.experiment.AnalysisHGLM, kwargs))

# saves output python objects (memory expensive)
detail_save = False

# writes json with input state causing an errors in Analysis.run(),
# continues to next experiment
error_save = True

source = 'hcp'

match source:
    case 'hcp':
        # human connectome project data
        hcp_path = '/home/matt/Dropbox/pnl_hglm/data/hcp100_lowres/image'

        exp_hcp = hglm.experiment.ExperimentImageOnly.from_search(
            folder=hcp_path,
            sbj_regex=r'[\d]{6}',
            img_glob_dict={'FA': '*_FA.nii.gz',
                           'MD': '*_MD.nii.gz'})
        exp = exp_hcp.sample_x(a=2, seed=0, add_bias=True)

        # to speed up analysis, random voxel is chosen and dilated to this radius.
        # only these voxels are included in the analysis
        radius = 4
    case 'wgn':
        # additive white gaussian noise
        exp = hglm.experiment.Experiment.from_gauss(seed=0,
                                                    shape=(5, 5, 5),
                                                    a=2,
                                                    b=2,
                                                    num_img=100)

        # to speed up analysis, random voxel is chosen and dilated to this radius.
        # only these voxels are included in the analysis
        radius = None
    case _:
        raise AttributeError(f'data source not recognized: {source}')
