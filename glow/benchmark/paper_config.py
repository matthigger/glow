"""Paper-benchmark TrialCache definitions.

Each entry of ``CACHE_BY_LABEL`` is a ``(TrialCache, ana_kwargs_dict)``
pair: the cache owns iteration + result IO for one (data source,
analysis-family) combination; the dict tells ``paper.run_trial`` which
analyses to fit on every trial.

The trial space is intentionally small: ``seed`` x ``effect_llr``, with
a single ``DataSource`` and effect ``Extenter`` held constant per
cache.  Sweeps that varied a structural parameter in the old setup
(b, num_img, effect n_vox, ...) are encoded by emitting one cache per
value.
"""
import math

import numpy as np

import glow
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import get_hotel_tr, get_wilks
from glow.benchmark.data import DataSourceHCP, DataSourceWGN
from glow.benchmark.trial_cache import TrialCache
from glow.effect import ExtenterMinVar, ExtenterSphere


# ---------- shared knobs -----------------------------------------------------
N_SEED = 10
N_SEED_NULL = 100
EFFECT_LLR_GRID = np.logspace(np.log10(0.003), np.log10(0.3), 11)
MODERATE_EFFECT_LLR = 0.03

CROP_N_VOX = 25_000
EFFECT_N_VOX = 2500            # 10 % of the cropped volume
_WGN_SIDE_3D = math.ceil(CROP_N_VOX ** (1 / 3))
_WGN_SIDE_2D = math.ceil(CROP_N_VOX ** (1 / 2))

N_PERM_FWER = 250
N_PERM_INNER = 250
ALPHA_FWER = 0.05

_CROP_EXTENTER = ExtenterSphere(n_vox=CROP_N_VOX, connected=True)
_DEFAULT_EFFECT_EXTENTER = ExtenterMinVar(n_vox=EFFECT_N_VOX)


# ---------- analysis recipes -------------------------------------------------
_GLOW_BASE = dict(n_perm_fwer=N_PERM_FWER,
                  n_perm_inner=N_PERM_INNER,
                  min_vox=4,
                  alpha_fwer=ALPHA_FWER)

ANALYSES_VBA = {
    'GLOW-Focus': (glow.analysis.AnalysisGLOW,
                   {**_GLOW_BASE, 'cluster_mode': ClusterMode.FOCUS}),
    'GLOW-GLM':   (glow.analysis.AnalysisGLOW,
                   {**_GLOW_BASE, 'cluster_mode': ClusterMode.GLM_ERROR}),
    'VBA':        (glow.analysis.AnalysisVBA,
                   dict(n_perm_fwer=N_PERM_FWER, tfce_flag=False,
                        z_flag=True, alpha_fwer=ALPHA_FWER,
                        get_stat=get_hotel_tr)),
    'VBA-TFCE':   (glow.analysis.AnalysisVBA,
                   dict(n_perm_fwer=N_PERM_FWER, tfce_flag=True,
                        z_flag=True, alpha_fwer=ALPHA_FWER,
                        get_stat=get_wilks)),
    'CET':        (glow.analysis.AnalysisCET,
                   dict(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER,
                        z_flag=True, get_stat=get_hotel_tr)),
}


# ---------- data-source factories -------------------------------------------
def _ds_wgn(*, shape=None, b=2, num_img=100, seed=0, crop=True):
    if shape is None:
        shape = (_WGN_SIDE_3D,) * 3
    return DataSourceWGN(
        shape=shape, b=b, num_img=num_img, seed=seed,
        extenter=_CROP_EXTENTER if crop else None)


def _ds_hcp(*, hcp_feats=('fa', 'md'), seed=0):
    return DataSourceHCP(hcp_feats=hcp_feats, seed=seed,
                         extenter=_CROP_EXTENTER)


# ---------- TrialCache assembly ---------------------------------------------
def _cache(label, *, ds, ana_kwargs_dict, extenter=None,
           effect_llr_all=None, n_seed=N_SEED):
    """Build one paper-benchmark TrialCache + analysis dict.

    Args:
        label: cache folder name (under ``get_path_result()``).
        ds: DataSource instance (held constant across trials).
        ana_kwargs_dict: ``{label: (AnalysisCls, kwargs)}`` to fit per
            trial.  Not part of the cache hash — change it freely
            without invalidating results.
        extenter: synthetic-effect extenter (held constant).  Defaults
            to ``ExtenterMinVar(n_vox=EFFECT_N_VOX)``.
        effect_llr_all: 1-D iterable of effect LLR values.
        n_seed: number of seeds per LLR.
    """
    if extenter is None:
        extenter = _DEFAULT_EFFECT_EXTENTER
    if effect_llr_all is None:
        effect_llr_all = EFFECT_LLR_GRID

    cache = TrialCache(
        name=label,
        iter_kwargs={
            'seed': list(range(n_seed)),
            'effect_llr': [float(x) for x in effect_llr_all],
        },
        kwargs={
            'ds': ds,
            'extenter': extenter,
        },
    )
    return cache, ana_kwargs_dict


# ---------- benchmark catalogue ---------------------------------------------
# label -> (TrialCache, ana_kwargs_dict)
CACHE_BY_LABEL = {}


def _add(label, *, ds, ana_kwargs_dict=None, **kwargs):
    if ana_kwargs_dict is None:
        ana_kwargs_dict = ANALYSES_VBA
    CACHE_BY_LABEL[label] = _cache(
        label, ds=ds, ana_kwargs_dict=ana_kwargs_dict, **kwargs)


# A. type I error (null): effect_llr = 0, many seeds
_add('null_hcp', ds=_ds_hcp(),
     effect_llr_all=[0.0], n_seed=N_SEED_NULL)
_add('null_wgn', ds=_ds_wgn(),
     effect_llr_all=[0.0], n_seed=N_SEED_NULL)

# B. VBA comparison split by feature count
_add('vba_hcp_fa',   ds=_ds_hcp(hcp_feats=('fa',)))
_add('vba_hcp_famd', ds=_ds_hcp(hcp_feats=('fa', 'md')))
_add('vba_wgn_b1',   ds=_ds_wgn(b=1))
_add('vba_wgn_b2',   ds=_ds_wgn(b=2))

# C. b sweep (WGN) — one cache per b at the moderate LLR
for _b in range(1, 11):
    _add(f'sweep_b_wgn_b{_b}',
         ds=_ds_wgn(b=_b),
         effect_llr_all=[MODERATE_EFFECT_LLR])

# D. effect-extent sweep — one cache per n_vox, both data sources
_EFFECT_N_VOX_GRID = [int(round(p * CROP_N_VOX))
                      for p in np.geomspace(0.01, 1.0, 15)]
for _n_vox in _EFFECT_N_VOX_GRID:
    _add(f'sweep_extent_hcp_n{_n_vox}',
         ds=_ds_hcp(),
         extenter=ExtenterMinVar(n_vox=_n_vox),
         effect_llr_all=[MODERATE_EFFECT_LLR])
    _add(f'sweep_extent_wgn_n{_n_vox}',
         ds=_ds_wgn(),
         extenter=ExtenterMinVar(n_vox=_n_vox),
         effect_llr_all=[MODERATE_EFFECT_LLR])

# E. num_img sweep (WGN only)
for _n in (10, 18, 30, 55, 100, 180, 300):
    _add(f'sweep_nimg_wgn_n{_n}',
         ds=_ds_wgn(num_img=_n),
         effect_llr_all=[MODERATE_EFFECT_LLR])

# F. 2D images (WGN, no 3D crop)
_add('vba_wgn_2d',
     ds=_ds_wgn(shape=(_WGN_SIDE_2D, _WGN_SIDE_2D), crop=False))

# G. sphere-extenter variants (geometric effect support, no min-var search)
_add('sphere_hcp',
     ds=_ds_hcp(),
     extenter=ExtenterSphere(n_vox=EFFECT_N_VOX))
_add('sphere_wgn',
     ds=_ds_wgn(),
     extenter=ExtenterSphere(n_vox=EFFECT_N_VOX))
