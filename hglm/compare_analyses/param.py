import numpy as np

import hglm

# where output results are stored (each run of script yields its own folder)
folder_out = '/home/matt/Dropbox/pnl_hglm/results'

# number of effects to model
n_repeat = 32 * 3

# pval describes severity of effect (assuming typical F test assumptions
# ...not valid but still useful to quantify how difficult effect is)
pval_all = np.geomspace(.6, .08, 13)

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

# analyses to run
analysis_obj_tup = (hglm.experiment.AnalysisHGLM,
                    hglm.experiment.AnalysisTFCE)

# parameters to be passed to Analysis constructor
analysis_kwargs = {'AnalysisHGLM': dict(n_perm=100,
                                        n_perm_adj=10,
                                        min_size_discover=1),
                   'AnalysisTFCE': dict(n_perm=100)}

# saves output python objects (memory expensive)
detail_save = False

# writes json with input state causing an errors in Analysis.run(),
# continues to next experiment
error_save = True

# if True, computes stats on max f1 region in HGLM analysis (allows us to
# distinguish between segmentation & discovery errors)
maxf1 = False

source = 'awgn'
match source:
    case 'hcp':
        # human connectome project data
        folder = '/home/matt/Dropbox/pnl_hglm/data/hcp100_lowres/image'
        exp_hcp = hglm.experiment.ExperimentImageOnly.from_search(
            folder=folder,
            sbj_regex=r'[\d]{6}',
            img_glob_dict={
                'FA': '*_FA.nii.gz',
                'MD': '*_MD.nii.gz'})
        exp = exp_hcp.sample_x(a=2, seed=1)
    case 'awgn':
        # additive white gaussian noise
        seed = 0
        shape = 5, 5, 5
        a, b, num_img = 2, 3, 100
        correlated_y = True

        rng = np.random.default_rng(seed=seed)
        y = rng.standard_normal(size=(b, num_img, np.prod(shape)))

        if correlated_y:
            # induce correlated y features
            y_transform = rng.standard_normal(size=(b, b))
            y = np.einsum('bnr,bc->cnr', y, y_transform)

        mask_idx = hglm.mask.get_mask_idx(np.ones(shape))
        exp_awgn = hglm.experiment.ExperimentImageOnly(y=y, mask_idx=mask_idx)
        exp_awgn = exp_awgn.sample_x(a=a, seed=seed)

        exp = exp_awgn
    case _:
        raise AttributeError(f'data source not recognized: {source}')
