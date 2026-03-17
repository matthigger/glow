import numpy as np

import glow
from glow.benchmark.config import Config
from glow.benchmark.run import run_ana, run_segment
from glow.experiment.mancova import stat_dict

# common params
# run experiments serially (n_jobs=1) and keep permutations serial per experiment
# TEMP: quick test settings (2k vox, 100 perm, 4 seeds)
COMMON = dict(
    n_seed=10,
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
N_PERM_FWER = 500
N_PERM_FWER_SIZE_ADJUST = 50
N_PERM_FWER_VBA = N_PERM_FWER + N_PERM_FWER_SIZE_ADJUST
ALPHA_FWER = 0.05
ANALYSES = {
    'GLOW': dict(n_perm_fwer=N_PERM_FWER,
                 n_perm_fwer_size_adjust=N_PERM_FWER_SIZE_ADJUST,
                 min_size=1,
                 alpha_fwer=ALPHA_FWER),
    'VBA': dict(n_perm_fwer=N_PERM_FWER_VBA,
                tfce_flag=False,
                alpha_fwer=ALPHA_FWER),
    'VBA-TFCE': dict(n_perm_fwer=N_PERM_FWER_VBA,
                     tfce_flag=True,
                     alpha_fwer=ALPHA_FWER),
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

# mancova stat experiment: compare all 5 statistics using full GLOW pipeline
ana_kwargs_dict_stat = {
    name: (glow.experiment.AnalysisGLOW, ANALYSES['GLOW'] | {'get_stat': fn})
    for name, fn in stat_dict.items()
}
config_list.append(make_config('mancova_stat_hcp', 'hcp', run_ana,
                               ana_kwargs_dict_stat))
config_list.append(make_config('mancova_stat_wgn', 'wgn', run_ana,
                               ana_kwargs_dict_stat))


# segmentation configs
config_list.append(make_config('segment_hcp', 'hcp', run_segment))
config_list.append(make_config('segment_wgn', 'wgn', run_segment))

CONFIG_BY_LABEL = {config.label: config for config in config_list}
