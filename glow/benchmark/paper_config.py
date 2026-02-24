import numpy as np

import glow
from glow.benchmark.config import Config
from glow.benchmark.run import run_ana, run_prune_compare, run_segment

# common params
# run experiments serially (n_jobs=1) and keep permutations serial per experiment
COMMON = dict(
    n_seed=50,
    hotel_tr_all=np.logspace(np.log10(0.005), np.log10(1.0), 10),
    effect_perc=0.2,
    n_jobs=1,
    detail_save=False,
    error_save=False,
)

# source params
SOURCES = {
    'wgn': dict(wgn_shape=(13, 13, 13),
                wgn_a=2,
                wgn_b=2,
                wgn_num_img=100,
                exp_seed=0),
    'hcp': dict(hcp_feats=['fa', 'md'],
                radius=8),
}

# analysis params
N_PERM = 250
ALPHA_FWER = 0.05
N_JOBS_PERM = 1

ANALYSES = {
    'GLOW': dict(n_perm=N_PERM,
                 n_perm_prune=100,
                 min_size=1,
                 alpha_prune=0.05,
                 alpha_fwer=ALPHA_FWER,
                 n_jobs_perm=N_JOBS_PERM,
                 prune_method='geom_prior'),
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

# mancova stat experiment: compare statistics
ana_kwargs_dict_mancova_stat = dict()
for label, get_stat in glow.experiment.mancova.stat_dict.items():
    kwargs = ANALYSES['GLOW'] | dict(get_stat=get_stat)
    ana_kwargs_dict_mancova_stat[label] = glow.experiment.AnalysisGLOW, kwargs

config_list.append(make_config('mancova_stat_hcp', 'hcp', run_ana, ana_kwargs_dict_mancova_stat))
config_list.append(make_config('mancova_stat_wgn', 'wgn', run_ana, ana_kwargs_dict_mancova_stat))

# alpha_prune experiment: vary alpha_prune for tradeoff
ana_kwargs_dict_alpha_prune = dict()
for _alpha_prune in [.01, .1, .5]:
    _kwargs_glow = ANALYSES['GLOW'] | dict(alpha_prune=_alpha_prune)
    ana_kwargs_dict_alpha_prune[f'alpha_prune={_alpha_prune}'] = (
        glow.experiment.AnalysisGLOW, _kwargs_glow)
config_list.append(make_config('alpha_prune', 'hcp', run_ana, ana_kwargs_dict_alpha_prune))

# pruning method experiment: compare pruning strategies
_GLOW_BASE = dict(n_perm=N_PERM, n_perm_prune=100, min_size=1,
                  alpha_prune=0.05, alpha_fwer=ALPHA_FWER,
                  n_jobs_perm=N_JOBS_PERM)
ana_kwargs_dict_prune_method = {
    'GLOW': (glow.experiment.AnalysisGLOW, _GLOW_BASE),
}
config_list.append(make_config('prune_method_wgn', 'wgn', run_prune_compare,
                               ana_kwargs_dict_prune_method, n_seed=10))
config_list.append(make_config('prune_method_hcp', 'hcp', run_prune_compare,
                               ana_kwargs_dict_prune_method, n_seed=10))

# segmentation configs
config_list.append(make_config('segment_hcp', 'hcp', run_segment))
config_list.append(make_config('segment_wgn', 'wgn', run_segment))

CONFIG_BY_LABEL = {config.label: config for config in config_list}
