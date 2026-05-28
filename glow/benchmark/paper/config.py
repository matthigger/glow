"""Paper-benchmark catalogue.

Each entry of CACHE_BY_LABEL is a (TrialCache, run_fnc) pair: the cache
owns iteration + result IO for one (data source, analysis-family)
combination; the run_fnc is bound to its analysis recipe via
functools.partial so the CLI doesn't need to know whether the trial is
a run_ana or run_mancova job.

The trial space is intentionally small: seed x effect_llr, with a
single DataSource and effect Extenter held constant per cache. Sweeps
that varied a structural parameter (b, num_img, effect n_vox, ...) are
encoded by emitting one cache per value.
"""
import math
from functools import partial

import numpy as np

import glow
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import get_hotel_tr, get_wilks
from glow.benchmark.data import DataSourceHCP, DataSourceWGN
from glow.benchmark.trial_cache import TrialCache
from glow.effect import ExtenterMinVar, ExtenterSphere

from .run import run_ana, run_mancova


# ---------- shared knobs -----------------------------------------------------
N_SEED = 10
N_SEED_NULL = 100
EFFECT_LLR_GRID = np.logspace(np.log10(0.003), np.log10(0.3), 11)
MODERATE_EFFECT_LLR = 0.03

CROP_N_VOX = 25_000
EFFECT_PERC_TOTAL_VOLUME = .1
EFFECT_N_VOX = int(EFFECT_PERC_TOTAL_VOLUME * CROP_N_VOX)

_WGN_SIDE_3D = math.ceil(CROP_N_VOX ** (1 / 3))
_WGN_SIDE_2D = math.ceil(CROP_N_VOX ** (1 / 2))

N_PERM_FWER = 250
N_PERM_INNER = 250
ALPHA_FWER = 0.05

# Per-family VBA design decisions hoisted out of the ANALYSIS_DICT below
# so they're easy to scan and override.
VBA_Z_FLAG = True
VBA_GET_STAT = get_hotel_tr
VBA_TFCE_GET_STAT = get_wilks
CET_GET_STAT = get_hotel_tr

_CROP_EXTENTER = ExtenterSphere(n_vox=CROP_N_VOX, connected=True)
_DEFAULT_EFFECT_EXTENTER = ExtenterMinVar(n_vox=EFFECT_N_VOX)


# ---------- analysis recipes -------------------------------------------------
_GLOW_BASE = dict(n_perm_fwer=N_PERM_FWER,
                  n_perm_inner=N_PERM_INNER,
                  alpha_fwer=ALPHA_FWER)
_VBA_BASE = dict(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER,
                 z_flag=VBA_Z_FLAG)

ANALYSIS_DICT = {
    'GLOW-Focus': (glow.analysis.AnalysisGLOW,
                   {**_GLOW_BASE, 'cluster_mode': ClusterMode.FOCUS}),
    'GLOW-GLM':   (glow.analysis.AnalysisGLOW,
                   {**_GLOW_BASE, 'cluster_mode': ClusterMode.GLM_ERROR}),
    'VBA':        (glow.analysis.AnalysisVBA,
                   {**_VBA_BASE, 'tfce_flag': False,
                    'get_stat': VBA_GET_STAT}),
    'VBA-TFCE':   (glow.analysis.AnalysisVBA,
                   {**_VBA_BASE, 'tfce_flag': True,
                    'get_stat': VBA_TFCE_GET_STAT}),
    'CET':        (glow.analysis.AnalysisCET,
                   {**_VBA_BASE, 'get_stat': CET_GET_STAT}),
}


# ---------- data-source factories -------------------------------------------
def _ds_wgn(*, shape=None, b: int = 2, num_img: int = 100, seed: int = 0,
            crop: bool = True):
    """Build a white-Gaussian-noise data source, optionally cropped to a sphere."""
    if shape is None:
        shape = (_WGN_SIDE_3D,) * 3
    return DataSourceWGN(
        shape=shape, b=b, num_img=num_img, seed=seed,
        extenter=_CROP_EXTENTER if crop else None)


def _ds_hcp(*, hcp_feats=('fa', 'md'), seed: int = 0):
    """Build an HCP data source over the given features, cropped to a sphere."""
    return DataSourceHCP(hcp_feats=hcp_feats, seed=seed,
                         extenter=_CROP_EXTENTER)


# ---------- TrialCache assembly ---------------------------------------------
# label -> (TrialCache, run_fnc)
CACHE_BY_LABEL = {}


def _make_cache(label: str, *, ds, extenter, effect_llr_all,
                n_seed: int) -> TrialCache:
    """Build a TrialCache over the seed x effect_llr grid for one (ds, extenter)."""
    return TrialCache(
        name=label,
        iter_kwargs={
            'seed': list(range(n_seed)),
            'effect_llr': [float(x) for x in effect_llr_all],
        },
        kwargs={'ds': ds, 'extenter': extenter},
    )


def _add_ana(label: str, *, ds, extenter=None, effect_llr_all=None,
             n_seed: int = N_SEED, ana_kwargs_dict: dict = None) -> None:
    """Register a CACHE_BY_LABEL entry that runs every analysis in ana_kwargs_dict.

    Args:
        label (str): catalogue key for the new cache
        ds: data source for the trials
        extenter: effect Extenter (defaults to the shared MinVar extenter)
        effect_llr_all: effect-strength grid (defaults to EFFECT_LLR_GRID)
        n_seed (int): number of seeds to sweep
        ana_kwargs_dict (dict): label -> (Analysis class, init kwargs);
            defaults to ANALYSIS_DICT
    """
    if extenter is None:
        extenter = _DEFAULT_EFFECT_EXTENTER
    if effect_llr_all is None:
        effect_llr_all = EFFECT_LLR_GRID
    if ana_kwargs_dict is None:
        ana_kwargs_dict = ANALYSIS_DICT

    cache = _make_cache(label, ds=ds, extenter=extenter,
                        effect_llr_all=effect_llr_all, n_seed=n_seed)
    run_fnc = partial(run_ana, ana_kwargs_dict=ana_kwargs_dict)
    CACHE_BY_LABEL[label] = (cache, run_fnc)


def _add_mancova(label: str, *, ds, extenter=None, effect_llr_all=None,
                 n_seed: int = N_SEED, n_perm_fwer: int = N_PERM_FWER,
                 alpha_fwer: float = ALPHA_FWER) -> None:
    """Register a CACHE_BY_LABEL entry that runs the VBA/TFCE/CET x stats x {raw,z} matrix.

    Args:
        label (str): catalogue key for the new cache
        ds: data source for the trials
        extenter: effect Extenter (defaults to the shared MinVar extenter)
        effect_llr_all: effect-strength grid (defaults to EFFECT_LLR_GRID)
        n_seed (int): number of seeds to sweep
        n_perm_fwer (int): number of FWER permutations
        alpha_fwer (float): FWER significance level
    """
    if extenter is None:
        extenter = _DEFAULT_EFFECT_EXTENTER
    if effect_llr_all is None:
        effect_llr_all = EFFECT_LLR_GRID

    cache = _make_cache(label, ds=ds, extenter=extenter,
                        effect_llr_all=effect_llr_all, n_seed=n_seed)
    run_fnc = partial(run_mancova, n_perm_fwer=n_perm_fwer,
                      alpha_fwer=alpha_fwer)
    CACHE_BY_LABEL[label] = (cache, run_fnc)


# ---------- benchmark catalogue ---------------------------------------------
# A. type I error (null): effect_llr = 0, many seeds
_add_ana('null_hcp', ds=_ds_hcp(),
         effect_llr_all=[0.0], n_seed=N_SEED_NULL)
_add_ana('null_wgn', ds=_ds_wgn(),
         effect_llr_all=[0.0], n_seed=N_SEED_NULL)

# B. VBA comparison split by feature count
_add_ana('vba_hcp_fa',   ds=_ds_hcp(hcp_feats=('fa',)))
_add_ana('vba_hcp_famd', ds=_ds_hcp(hcp_feats=('fa', 'md')))
_add_ana('vba_wgn_b1',   ds=_ds_wgn(b=1))
_add_ana('vba_wgn_b2',   ds=_ds_wgn(b=2))

# C. effect-extent sweep — one cache per n_vox, both data sources
_EFFECT_N_VOX_GRID = [int(round(p * CROP_N_VOX))
                      for p in np.geomspace(0.01, 1.0, 15)]
for _n_vox in _EFFECT_N_VOX_GRID:
    _add_ana(f'sweep_extent_hcp_n{_n_vox}',
             ds=_ds_hcp(),
             extenter=ExtenterMinVar(n_vox=_n_vox),
             effect_llr_all=[MODERATE_EFFECT_LLR])
    _add_ana(f'sweep_extent_wgn_n{_n_vox}',
             ds=_ds_wgn(),
             extenter=ExtenterMinVar(n_vox=_n_vox),
             effect_llr_all=[MODERATE_EFFECT_LLR])

# D. num_img sweep (WGN only)
for _n in (10, 18, 30, 55, 100, 180, 300):
    _add_ana(f'sweep_nimg_wgn_n{_n}',
             ds=_ds_wgn(num_img=_n),
             effect_llr_all=[MODERATE_EFFECT_LLR])

# E. 2D images (WGN, no 3D crop)
_add_ana('vba_wgn_2d',
         ds=_ds_wgn(shape=(_WGN_SIDE_2D, _WGN_SIDE_2D), crop=False))

# F. sphere-extenter variants
_add_ana('sphere_hcp',
         ds=_ds_hcp(),
         extenter=ExtenterSphere(n_vox=EFFECT_N_VOX))
_add_ana('sphere_wgn',
         ds=_ds_wgn(),
         extenter=ExtenterSphere(n_vox=EFFECT_N_VOX))

# G. MANCOVA stat comparison — VBA / VBA-TFCE / CET x 5 stats x {raw, z}
_add_mancova('mancova_vba_hcp', ds=_ds_hcp())
_add_mancova('mancova_vba_wgn', ds=_ds_wgn())
