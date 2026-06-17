"""Shelved paper-benchmark caches, kept for storage (not in the catalogue).

These three experiments were dropped from the active paper run but are
preserved here so they can be revived without re-deriving them:

  - vba_wgn_2d: detection on 2-D WGN with no 3-D sphere crop.
  - sphere_wgn / sphere_hcp: detection where the planted effect's support
    is a sphere (ExtenterSphere) rather than the min-variance blob
    (ExtenterMinVar) used everywhere in the active catalogue.

They need axes the active scalar factory deliberately does not build (a
2-D no-crop shape; a sphere *effect* extenter), so each entry carries its
DataSource / Extenter as constant object kwargs and runs through
run._run_ana_obj (the object-level core of run_ana) rather than the
scalar run_ana. MOTHBALL_BY_LABEL mirrors config.CACHE_BY_LABEL, so to
revive one just drive it the same way the CLI drives a catalogue entry:

    from glow.benchmark.driver import driver_local
    from glow.benchmark.paper.config_mothball import MOTHBALL_BY_LABEL
    cache, run_fnc = MOTHBALL_BY_LABEL['sphere_hcp']
    driver_local(cache, run_fnc)
"""
import math
from functools import partial

from glow.benchmark import hcp
from glow.benchmark.data import DataSourceHCP, DataSourceWGN
from glow.benchmark.trial_cache import TrialCache
from glow.effect import ExtenterMinVar, ExtenterSphere

from .config import ANALYSIS_DICT, EFFECT_LLR_GRID, EFFECT_N_VOX, N_SEED
from .factory import CROP_N_VOX, DS_SEED, _CROP_EXTENTER, _ds_factory
from .run import _run_ana_obj


_WGN_SIDE_2D = math.ceil(CROP_N_VOX ** (1 / 2))

# label -> (TrialCache, run_fnc); intentionally NOT registered in the
# active CACHE_BY_LABEL.
MOTHBALL_BY_LABEL = {}


def _mothball(label: str, *, ds, extenter) -> None:
    """Register a shelved cache that sweeps seed x effect_llr over one (ds, extenter).

    Args:
        label (str): storage key / on-disk result folder name
        ds: the (already-built) DataSource for every trial
        extenter: the (already-built) effect Extenter for every trial
    """
    cache = TrialCache(
        name=label,
        iter_kwargs={'seed': list(range(N_SEED)),
                     'effect_llr': [float(x) for x in EFFECT_LLR_GRID]},
        kwargs={'ds': ds, 'extenter': extenter})
    run_fnc = partial(_run_ana_obj, ana_kwargs_dict=ANALYSIS_DICT)
    MOTHBALL_BY_LABEL[label] = (cache, run_fnc)


# 2-D WGN, no 3-D crop; min-variance effect support.
_mothball(
    'vba_wgn_2d',
    ds=DataSourceWGN(shape=(_WGN_SIDE_2D, _WGN_SIDE_2D), b=1, num_img=100,
                     seed=DS_SEED, extenter=None),
    extenter=ExtenterMinVar(n_vox=EFFECT_N_VOX))

# Cropped WGN / HCP, but the planted effect is a sphere.
_mothball('sphere_wgn',
          ds=_ds_factory('wgn', 1, 100, None, DS_SEED),
          extenter=ExtenterSphere(n_vox=EFFECT_N_VOX))
_mothball('sphere_hcp',
          ds=DataSourceHCP(hcp_feats=hcp.HCP_FEATS, seed=DS_SEED,
                           extenter=_CROP_EXTENTER),
          extenter=ExtenterSphere(n_vox=EFFECT_N_VOX))
