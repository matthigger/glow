import math

import numpy as np

import glow
from glow.benchmark.config import Config
from glow.benchmark.run import (run_ana, run_mancova_glow, run_prune_compare,
                                run_segment, run_mancova_vba)
from glow.analysis.mancova import get_hotel_tr, get_llr, get_wilks

# ---------- common parameters ----------
CROP_N_VOX = 25000

COMMON = dict(
    n_seed=5,
    effect_llr_all=np.logspace(np.log10(0.003), np.log10(0.3), 11),
    effect_perc=0.1,
    crop_n_vox=CROP_N_VOX,
    n_jobs=1,
    detail_save=False,
    error_save=False,
)

MODERATE_EFFECT_LLR = 0.03

# ---------- source parameters ----------
_wgn_side = math.ceil(CROP_N_VOX ** (1 / 3))

SOURCES = {
    'wgn': dict(wgn_shape=(_wgn_side, _wgn_side, _wgn_side), wgn_a=2, wgn_b=2,
                wgn_num_img=100, exp_seed=0),
    'hcp': dict(hcp_feats=['fa', 'md']),
}

# ---------- analysis parameters ----------
N_PERM_TOTAL = 250
SIZE_ADJUST_FRAC = 0.05
N_PERM_FWER_SIZE_ADJUST = max(1, round(N_PERM_TOTAL * SIZE_ADJUST_FRAC))
N_PERM_FWER = N_PERM_TOTAL - N_PERM_FWER_SIZE_ADJUST
ALPHA_FWER = 0.05
ANALYSES = {
    'GLOW': dict(n_perm_fwer=N_PERM_FWER,
                 n_perm_fwer_size_adjust=N_PERM_FWER_SIZE_ADJUST,
                 min_size=1,
                 alpha_fwer=ALPHA_FWER,
                 get_stat=get_llr),
    'VBA': dict(n_perm_fwer=N_PERM_TOTAL,
                tfce_flag=False,
                z_flag=True,
                alpha_fwer=ALPHA_FWER,
                get_stat=get_hotel_tr),
    'VBA-TFCE': dict(n_perm_fwer=N_PERM_TOTAL,
                     tfce_flag=True,
                     z_flag=True,
                     alpha_fwer=ALPHA_FWER,
                     get_stat=get_wilks),
    'CET': dict(n_perm_fwer=N_PERM_TOTAL,
                alpha_fwer=ALPHA_FWER,
                z_flag=True,
                get_stat=get_hotel_tr),
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


# ---------- shared analysis kwarg dicts ----------
ana_kwargs_dict_vba = {
    'GLOW': (glow.analysis.AnalysisGLOW, ANALYSES['GLOW']),
    'VBA': (glow.analysis.AnalysisVBA, ANALYSES['VBA']),
    'VBA-TFCE': (glow.analysis.AnalysisVBA, ANALYSES['VBA-TFCE']),
    'CET': (glow.analysis.AnalysisCET, ANALYSES['CET']),
}

ana_kwargs_dict_prune = {
    'GLOW': (glow.analysis.AnalysisGLOW, ANALYSES['GLOW']),
}

ana_kwargs_dict_mancova = {
    'GLOW': (glow.analysis.AnalysisGLOW, ANALYSES['GLOW']),
}

ana_kwargs_dict_mancova_vba = {
    'VBA-TFCE': (glow.analysis.AnalysisVBA, ANALYSES['VBA-TFCE']),
}


# =====================================================================
# build experiment configs
# =====================================================================
config_list = []

# ---------- A. type I error (null) ----------
# effect_llr=0, 500 seeds — produces calibration curves via min_pval
config_list.append(make_config(
    'null_hcp', 'hcp', run_ana, ana_kwargs_dict_vba,
    effect_llr_all=np.array([0.0]), n_seed=500))
config_list.append(make_config(
    'null_wgn', 'wgn', run_ana, ana_kwargs_dict_vba,
    effect_llr_all=np.array([0.0]), n_seed=500))

# ---------- B. VBA comparison split by feature count ----------
# HCP: FA only (b=1) and FA+MD (b=2)
config_list.append(make_config(
    'vba_hcp_fa', 'hcp', run_ana, ana_kwargs_dict_vba,
    hcp_feats=['fa']))
config_list.append(make_config(
    'vba_hcp_famd', 'hcp', run_ana, ana_kwargs_dict_vba))

# WGN: b=1 and b=2
config_list.append(make_config(
    'vba_wgn_b1', 'wgn', run_ana, ana_kwargs_dict_vba,
    wgn_b=1))
config_list.append(make_config(
    'vba_wgn_b2', 'wgn', run_ana, ana_kwargs_dict_vba))

# ---------- C. b sweep (WGN) ----------
# vary number of features 1..10 at fixed moderate effect
config_list.append(make_config(
    'sweep_b_wgn', 'wgn', run_ana, ana_kwargs_dict_vba,
    x_param='wgn_b',
    iter_params={'wgn_b': list(range(1, 11))},
    fixed_params={'effect_llr': MODERATE_EFFECT_LLR}))

# ---------- D. effect_perc sweep ----------
# 15 geometrically spaced points from 1% to 100% of volume
config_list.append(make_config(
    'sweep_perc_hcp', 'hcp', run_ana, ana_kwargs_dict_vba,
    x_param='effect_perc',
    iter_params={'effect_perc': np.geomspace(0.01, 1.0, 15).tolist()},
    fixed_params={'effect_llr': MODERATE_EFFECT_LLR}))
config_list.append(make_config(
    'sweep_perc_wgn', 'wgn', run_ana, ana_kwargs_dict_vba,
    x_param='effect_perc',
    iter_params={'effect_perc': np.geomspace(0.01, 1.0, 15).tolist()},
    fixed_params={'effect_llr': MODERATE_EFFECT_LLR}))

# ---------- E. num_img sweep (WGN only) ----------
# 7 geometrically spaced points from 10 to 300 subjects
config_list.append(make_config(
    'sweep_nimg_wgn', 'wgn', run_ana, ana_kwargs_dict_vba,
    x_param='wgn_num_img',
    iter_params={'wgn_num_img': [10, 18, 30, 55, 100, 180, 300]},
    fixed_params={'effect_llr': MODERATE_EFFECT_LLR}))

# ---------- F. 2D images (WGN) ----------
_wgn_side_2d = math.ceil(CROP_N_VOX ** (1 / 2))
config_list.append(make_config(
    'vba_wgn_2d', 'wgn', run_ana, ana_kwargs_dict_vba,
    wgn_shape=(_wgn_side_2d, _wgn_side_2d)))

# ---------- G. pruning method comparison ----------
config_list.append(make_config(
    'prune_method_hcp', 'hcp', run_prune_compare, ana_kwargs_dict_prune))
config_list.append(make_config(
    'prune_method_wgn', 'wgn', run_prune_compare, ana_kwargs_dict_prune))

# ---------- H. MANCOVA stat comparison (GLOW) ----------
config_list.append(make_config(
    'mancova_glow_wgn', 'wgn', run_mancova_glow, ana_kwargs_dict_mancova,
    crop_n_vox=5000))
config_list.append(make_config(
    'mancova_glow_hcp', 'hcp', run_mancova_glow, ana_kwargs_dict_mancova,
    crop_n_vox=5000))

# ---------- I. MANCOVA stat comparison (VBA / VBA-TFCE / CET) ----------
config_list.append(make_config(
    'mancova_vba_wgn', 'wgn', run_mancova_vba, ana_kwargs_dict_mancova_vba))
config_list.append(make_config(
    'mancova_vba_hcp', 'hcp', run_mancova_vba, ana_kwargs_dict_mancova_vba))

# ---------- J. segmentation comparison ----------
config_list.append(make_config('segment_hcp', 'hcp', run_segment))
config_list.append(make_config('segment_wgn', 'wgn', run_segment))


CONFIG_BY_LABEL = {config.label: config for config in config_list}
