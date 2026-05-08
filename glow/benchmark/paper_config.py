import math

import numpy as np

import glow
from glow.benchmark.config import Config
from glow.benchmark.runner import (RunAna, RunSegment, RunPruneCompare,
                                   RunMancovaGlow, RunMancovaVba)
from glow.analysis.mancova import get_hotel_tr, get_llr, get_wilks

# ---------- common parameters ----------
# CROP_N_VOX = 25_000
CROP_N_VOX = 1_000

COMMON = dict(
    n_seed=10,
    # n_seed=50,
    effect_llr_all=np.logspace(np.log10(0.003), np.log10(0.3), 11),
    effect_perc=0.1,
    crop_n_vox=CROP_N_VOX,
    n_jobs=1,
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
N_PERM_FWER = 250          # outer permutations for FWER
N_PERM_INNER = 200         # inner permutations for per-region (mu, std)
ALPHA_FWER = 0.05
_GLOW_BASE = dict(n_perm_fwer=N_PERM_FWER,
                  n_perm_inner=N_PERM_INNER,
                  min_vox=4,
                  alpha_fwer=ALPHA_FWER,
                  get_stat=get_llr)
ANALYSES = {
    'GLOW-Focus': {**_GLOW_BASE, 'cluster_mode': "q1"},
    'GLOW-GLM':   {**_GLOW_BASE, 'cluster_mode': "q0, q1"},
    'VBA': dict(n_perm_fwer=N_PERM_FWER,
                tfce_flag=False,
                z_flag=True,
                alpha_fwer=ALPHA_FWER,
                get_stat=get_hotel_tr),
    'VBA-TFCE': dict(n_perm_fwer=N_PERM_FWER,
                     tfce_flag=True,
                     z_flag=True,
                     alpha_fwer=ALPHA_FWER,
                     get_stat=get_wilks),
    'CET': dict(n_perm_fwer=N_PERM_FWER,
                alpha_fwer=ALPHA_FWER,
                z_flag=True,
                get_stat=get_hotel_tr),
}


def make_config(label, source, runner, **overrides):
    params = {}
    params.update(COMMON)
    params.update(SOURCES[source])
    params.update(overrides)
    return Config(
        label=label,
        source=source,
        runner=runner,
        **params
    )


# ---------- shared ana_kwargs_dicts used by RunAna configs ----------
ana_kwargs_dict_vba = {
    'GLOW-Focus': (glow.analysis.AnalysisGLOW, ANALYSES['GLOW-Focus']),
    'GLOW-GLM':   (glow.analysis.AnalysisGLOW, ANALYSES['GLOW-GLM']),
    'VBA':        (glow.analysis.AnalysisVBA, ANALYSES['VBA']),
    'VBA-TFCE':   (glow.analysis.AnalysisVBA, ANALYSES['VBA-TFCE']),
    'CET':        (glow.analysis.AnalysisCET, ANALYSES['CET']),
}


# =====================================================================
# build experiment configs
# =====================================================================
config_list = []

# ---------- A. type I error (null) ----------
# effect_llr=0, 500 seeds — produces calibration curves via min_pval
config_list.append(make_config(
    'null_hcp', 'hcp', RunAna(ana_kwargs_dict_vba),
    effect_llr_all=np.array([0.0]), n_seed=500))
config_list.append(make_config(
    'null_wgn', 'wgn', RunAna(ana_kwargs_dict_vba),
    effect_llr_all=np.array([0.0]), n_seed=500))

# ---------- B. VBA comparison split by feature count ----------
# HCP: FA only (b=1) and FA+MD (b=2)
config_list.append(make_config(
    'vba_hcp_fa', 'hcp', RunAna(ana_kwargs_dict_vba),
    hcp_feats=['fa']))
config_list.append(make_config(
    'vba_hcp_famd', 'hcp', RunAna(ana_kwargs_dict_vba)))

# WGN: b=1 and b=2
config_list.append(make_config(
    'vba_wgn_b1', 'wgn', RunAna(ana_kwargs_dict_vba),
    wgn_b=1))
config_list.append(make_config(
    'vba_wgn_b2', 'wgn', RunAna(ana_kwargs_dict_vba)))

# ---------- C. b sweep (WGN) ----------
config_list.append(make_config(
    'sweep_b_wgn', 'wgn', RunAna(ana_kwargs_dict_vba),
    x_param='wgn_b',
    iter_params={'wgn_b': list(range(1, 11))},
    fixed_params={'effect_llr': MODERATE_EFFECT_LLR}))

# ---------- D. effect_perc sweep ----------
config_list.append(make_config(
    'sweep_perc_hcp', 'hcp', RunAna(ana_kwargs_dict_vba),
    x_param='effect_perc',
    iter_params={'effect_perc': np.geomspace(0.01, 1.0, 15).tolist()},
    fixed_params={'effect_llr': MODERATE_EFFECT_LLR}))
config_list.append(make_config(
    'sweep_perc_wgn', 'wgn', RunAna(ana_kwargs_dict_vba),
    x_param='effect_perc',
    iter_params={'effect_perc': np.geomspace(0.01, 1.0, 15).tolist()},
    fixed_params={'effect_llr': MODERATE_EFFECT_LLR}))

# ---------- E. num_img sweep (WGN only) ----------
config_list.append(make_config(
    'sweep_nimg_wgn', 'wgn', RunAna(ana_kwargs_dict_vba),
    x_param='wgn_num_img',
    iter_params={'wgn_num_img': [10, 18, 30, 55, 100, 180, 300]},
    fixed_params={'effect_llr': MODERATE_EFFECT_LLR}))

# ---------- F. 2D images (WGN) ----------
_wgn_side_2d = math.ceil(CROP_N_VOX ** (1 / 2))
config_list.append(make_config(
    'vba_wgn_2d', 'wgn', RunAna(ana_kwargs_dict_vba),
    wgn_shape=(_wgn_side_2d, _wgn_side_2d)))

# ---------- G. pruning method comparison ----------
config_list.append(make_config(
    'prune_method_hcp', 'hcp', RunPruneCompare(ANALYSES['GLOW-GLM'])))
config_list.append(make_config(
    'prune_method_wgn', 'wgn', RunPruneCompare(ANALYSES['GLOW-GLM'])))

# ---------- H. MANCOVA stat comparison (GLOW) ----------
config_list.append(make_config(
    'mancova_glow_wgn', 'wgn', RunMancovaGlow(ANALYSES['GLOW-GLM'])))
config_list.append(make_config(
    'mancova_glow_hcp', 'hcp', RunMancovaGlow(ANALYSES['GLOW-GLM'])))

# ---------- I. MANCOVA stat comparison (VBA / VBA-TFCE / CET) ----------
config_list.append(make_config(
    'mancova_vba_wgn', 'wgn', RunMancovaVba(ANALYSES['VBA-TFCE'])))
config_list.append(make_config(
    'mancova_vba_hcp', 'hcp', RunMancovaVba(ANALYSES['VBA-TFCE'])))

# ---------- J. segmentation comparison ----------
# minvar: greedy variance-minimising effect extent (the paper's default)
# sphere: random-centre dilated sphere — segmentation when the effect has
#         no special covariance structure
config_list.append(make_config('segment_minvar_hcp', 'hcp', RunSegment()))
config_list.append(make_config('segment_minvar_wgn', 'wgn', RunSegment()))
config_list.append(make_config('segment_sphere_hcp', 'hcp', RunSegment(),
                               effect_extenter='sphere'))
config_list.append(make_config('segment_sphere_wgn', 'wgn', RunSegment(),
                               effect_extenter='sphere'))


CONFIG_BY_LABEL = {config.label: config for config in config_list}
