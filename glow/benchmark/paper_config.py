import numpy as np

import glow
from glow.benchmark.config import Config
from glow.benchmark.run import run_ana, run_segment, run_stat_auc

# common params
# run experiments serially (n_jobs=1) and keep permutations serial per experiment
# TEMP: quick test settings (2k vox, 100 perm, 4 seeds)
COMMON = dict(
    n_seed=4,
    effect_llr_all=np.logspace(np.log10(0.003), np.log10(0.3), 5),
    effect_perc=0.2,
    crop_n_vox=500,
    n_jobs=1,
    detail_save=False,
    error_save=False,
)

# source params
# TEMP: WGN shape sized for 2k crop (production: (25,25,25) for 10k crop)
SOURCES = {
    'wgn': dict(wgn_shape=(15, 15, 15),
                wgn_a=2,
                wgn_b=2,
                wgn_num_img=100,
                exp_seed=0),
    'hcp': dict(hcp_feats=['fa', 'md']),
}

# analysis params
# TEMP: 100 perms (production: 250)
N_PERM = 100
ALPHA_FWER = 0.05
N_JOBS_PERM = 1

ANALYSES = {
    'GLOW': dict(n_perm=N_PERM,
                 n_perm_prune=100,
                 min_size=1,
                 alpha_prune=0.05,
                 alpha_fwer=ALPHA_FWER,
                 n_jobs_perm=N_JOBS_PERM),
    'VBA': dict(n_perm=N_PERM,
                tfce_flag=False,
                alpha_fwer=ALPHA_FWER,
                n_jobs_perm=N_JOBS_PERM),
    'VBA-TFCE': dict(n_perm=N_PERM,
                     tfce_flag=True,
                     alpha_fwer=ALPHA_FWER,
                     n_jobs_perm=N_JOBS_PERM),
}


def make_config(label, source, run_fnc, ana_kwargs_dict=None, **overrides):
    params = {}
    params.update(COMMON)
    params.update(SOURCES[source])
    params.update(overrides)
    return Config(
        label=label,
        source=source,
        run_fnc=run_fnc,
        ana_kwargs_dict=ana_kwargs_dict,
        **params
    )


# build experiment configs
config_list = []

# vba experiment: compare methods
ana_kwargs_dict_vba = {
    'GLOW': (glow.experiment.AnalysisGLOW, ANALYSES['GLOW']),
    'VBA': (glow.experiment.AnalysisVBA, ANALYSES['VBA']),
    'VBA-TFCE': (glow.experiment.AnalysisVBA, ANALYSES['VBA-TFCE']),
}

config_list.append(make_config('vba_hcp', 'hcp', run_ana, ana_kwargs_dict_vba))
config_list.append(make_config('vba_wgn', 'wgn', run_ana, ana_kwargs_dict_vba))

# mancova stat experiment: compare statistics via DP-antichain AUC
# (replaces old mancova_stat_* which ran full FWER+prune per stat)
config_list.append(make_config('stat_auc_hcp', 'hcp', run_stat_auc,
                               fixed_params={'n_perm_fit': 30}))
config_list.append(make_config('stat_auc_wgn', 'wgn', run_stat_auc,
                               fixed_params={'n_perm_fit': 30}))


# segmentation configs
config_list.append(make_config('segment_hcp', 'hcp', run_segment))
config_list.append(make_config('segment_wgn', 'wgn', run_segment))

CONFIG_BY_LABEL = {config.label: config for config in config_list}
