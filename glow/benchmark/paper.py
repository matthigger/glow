import numpy as np

import glow
from glow.benchmark.config import Config

# common params
common_dict = dict(n_seed=100,
                   hotel_tr_all=np.logspace(np.log10(0.03), np.log10(1.0), 16),
                   effect_perc=.2,
                   n_jobs=-1,
                   detail_save=False,
                   error_save=False)

# wgn params
wgn_dict = dict(wgn_shape=(8, 8, 8),
                wgn_a=2,
                wgn_b=2,
                wgn_num_img=100,
                exp_seed=0)

# hcp params
hcp_dict = dict(hcp_feats=['FA', 'MD'],
                radius=8)

# Analysis params
n_perm = 100
alpha_fwer = .05

kwargs_glow = dict(n_perm=n_perm,
                   n_perm_adj=50,
                   n_perm_tailor=200,
                   min_size=1,
                   alpha_tailor=.05,
                   alpha_fwer=alpha_fwer)
kwargs_vba = dict(n_perm=n_perm,
                  tfce_flag=False,
                  cet_flag=False,
                  alpha_fwer=alpha_fwer)
kwargs_tfce = kwargs_vba | dict(tfce_flag=True)

# build configs of experiments
config_list = list()

# vba experiment: compare which method performs best
ana_kwargs_dict_vba = {'GLOW': (glow.experiment.AnalysisGLOW, kwargs_glow),
'VBA': (glow.experiment.AnalysisVBA, kwargs_vba),
'VBA-TFCE': (glow.experiment.AnalysisVBA, kwargs_tfce)}

config_list.append(Config(label='vba_hcp',
                          source='hcp',
                          ana_kwargs_dict=ana_kwargs_dict_vba,
                          **(common_dict | hcp_dict)))
config_list.append(Config(label='vba_wgn',
                          source='wgn',
                          ana_kwargs_dict=ana_kwargs_dict_vba,
                          **(common_dict | wgn_dict)))

# mancova stat experiment: compare which statistic performs best
ana_kwargs_dict_mancova_stat = dict()
for label, get_stat in glow.experiment.mancova.stat_dict.items():
    kwargs = kwargs_glow | dict(get_stat=get_stat)
    ana_kwargs_dict_mancova_stat[label] = glow.experiment.AnalysisGLOW, kwargs

config_list.append(Config(label='mancova_stat_hcp',
                          source='hcp',
                          ana_kwargs_dict=ana_kwargs_dict_mancova_stat,
                          **(common_dict | hcp_dict)))
config_list.append(Config(label='mancova_stat_wgn',
                          source='wgn',
                          ana_kwargs_dict=ana_kwargs_dict_mancova_stat,
                          **(common_dict | wgn_dict)))

# alpha_tailor experiment: vary alpha_tailor, which offers reasonable
# performance / speed tradeoff point?
ana_kwargs_dict_alpha_tailor = dict()
for _alpha_tailor in [.05, .15, .5]:
    _kwargs_glow = kwargs_glow | dict(alpha_tailor=_alpha_tailor)
    ana_kwargs_dict_alpha_tailor[f'alpha_tailor={_alpha_tailor}'] = (
        glow.experiment.AnalysisGLOW, _kwargs_glow)
config_list.append(Config(label='alpha_tailor',
                          source='hcp',
                          ana_kwargs_dict=ana_kwargs_dict_alpha_tailor,
                          **(common_dict | hcp_dict)))

# todo: needs config update (varies only hotel_tr & seed ...)
# runtime experiment, how does it vary with region size?
# radius_all = list(np.linspace(2, 70, 31).astype(int)) + [None]

# dataset experiment, how does it vary with b? (wgn with b=1 and HCP with
# just FA vs FA&MD



if __name__ == '__main__':
    from glow.benchmark.run import run_ana

    # run benchmarks
    for config in config_list:
        print(f'begin: {config.label}')
        config.run_all(run_fnc=run_ana, verbose=True)
