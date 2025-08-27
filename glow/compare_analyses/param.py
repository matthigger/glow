import numpy as np

import glow

# hcp or wgn
source = 'wgn'

# controls number of repetitions
seed_all = np.arange(100)

# hotelling's trace [0, inf) describes severity of effect
hotel_tr_all = np.logspace(np.log10(.03), np.log10(1), 15)
# hotel_tr_all = .15,

# effect size, as ratio to total voxels in experiment
effect_perc = .2

# FWER control
alpha_fwer = .05

# toggles parallel, 0 or 1 processes serial (good for debug).  else this is the
# number of threads to use.  (-1 for all of them)
n_jobs = -1

# parameters to be passed to Analysis constructor
kwargs_glow = dict(n_perm=100,
                   n_perm_adj=100,
                   n_perm_tailor=1000,
                   min_size=1,
                   alpha_tailor=.15)
kwargs_tfce = dict(n_perm=100, tfce_flag=True, cet_flag=False)
kwargs_vba = dict(n_perm=100, tfce_flag=False, cet_flag=False)
kwargs_cet = dict(n_perm=100, tfce_flag=False, cet_flag=True)

# analyses to run
# compare GLOW and TFCE
ana_kwargs_dict = {'GLOW': (glow.experiment.AnalysisGLOW, kwargs_glow),
                   'VBA': (glow.experiment.AnalysisVBA, kwargs_vba),
                   'CET': (glow.experiment.AnalysisVBA, kwargs_cet),
                   'TFCE': (glow.experiment.AnalysisVBA, kwargs_tfce)}

# # compare mancova stats
# ana_kwargs_dict = dict()
# for label, get_stat in glow.experiment.mancova.stat_dict.items():
#     kwargs = kwargs_glow | dict(get_stat=get_stat)
#     ana_kwargs_dict[label] = glow.experiment.AnalysisGLOW, kwargs

# # compare different minimum region size for GLOW
# ana_kwargs_dict = dict()
# for min_size in [2 ** idx for idx in range(5)]:
#     label = f'min_size{min_size}'
#     kwargs = kwargs_glow | dict(min_size=min_size)
#     ana_kwargs_dict[label] = glow.experiment.AnalysisGLOW, kwargs

# saves output python objects (memory expensive)
detail_save = True

# writes json with input state causing an errors in Analysis.run(),
# continues to next experiment
error_save = False

match source:
    case 'hcp':
        # human connectome project data
        hcp_path = '/home/matt/Dropbox/pnl_hglm/data/hcp100_lowres/image'

        exp_hcp = glow.experiment.ExperimentImageOnly.from_search(
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
        exp = glow.experiment.Experiment.from_gauss(seed=0,
                                                    shape=(5, 5, 5),
                                                    a=2,
                                                    b=2,
                                                    num_img=100)

        # to speed up analysis, random voxel is chosen and dilated to this radius.
        # only these voxels are included in the analysis
        radius = None
    case _:
        raise AttributeError(f'data source not recognized: {source}')
