"""Paper-benchmark catalogue: the inputs to driver.drive for each figure.

CONFIG maps a cache name (e.g. 'sweep_llr') to the four-tuple drive consumes
-- (kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc): the data
grid, the effect grid, the leaf-function kwargs grid, and the leaf function
itself. So one benchmark figure is drive(*CONFIG[name]). Running the sweep is
the driver's / a CLI's job, not this module's -- config only declares it.

The two upstream grids come from builder functions whose keyword arguments
are the swept axes: get_kwargs_data_list returns the list of data_factory
kwargs over (source, b, num_img, seed); get_kwargs_effect_list returns the
list of effect_factory kwargs over (effect_llr, n_vox) -- or [None] for the
null path. Each cache calls them with the axes it sweeps, leaving the rest at
their defaults. The leaf is run_ana over the shared analysis recipes
(RUN_ANA_LIST).

This replaces the old scalar-grid catalogue (paper/config_old.py): there the
heavy objects (DataSource, Extenter) were rebuilt from scalar axes inside the
trial fn; here a cell carries the Extenter (effect support, analysis crop)
and the realized HCP feature subset directly, so the catalogue is what runs
-- no factory indirection.

Seeding. The old layer split one trial seed (derive_seeds) into independent
data / feature / effect sub-seeds so the effect placement tracked the data
realization. The driver decouples the data and effect loops (one effect grid
is shared across all data cells), so here a data cell's seed drives its whole
realization (the WGN draw / HCP feature subset, the x design, and the
analysis crop location), and the effect support is placed with seed_from_exp
-- effect_factory derives its placement seed from a hash of the experiment,
so each realization gets its own (reproducible) effect location even though
the effect grid is shared and carries no per-data seed (see effect_factory).

Effect strength. effect_llr is the per-voxel (size-normalized) target; the
observed whole-region LLR is ~ effect_llr * n_vox (see glow.effect.impose).
Sweeps hold effect_llr fixed: across a structural axis (b, num_img) that is a
clean power curve, and across the extent (n_vox) sweep the whole-region LLR
grows with the region. (The factory has no whole-region-LLR knob -- only the
per-voxel effect_llr -- so an extent sweep at fixed total would have to back
it out here; we don't.)

WGN and HCP share each cache (both sources in one data grid); they face apart
on the recorded source column downstream. HCP has no num_img axis (its N is
the cohort), so the num_img sweep is WGN-only.

Scope. The leaf here is run_ana (fit + score one Analysis per cell), so this
covers the five effect-detection caches that are exactly that shape: null,
sweep_llr, sweep_b, sweep_extent, sweep_nimg. The old catalogue's segment /
stat / prune / two-effect / min_size caches need other leaf functions (a
segmentation oracle, a shared voxel-stat walk, two pruning rules on one fit,
two planted effects, per-perm staircases); each becomes its own fnc +
kwargs_fnc_list in CONFIG once written to run_ana's contract
(fnc(exp, mask_target_list=..., **kwargs), memoised + recorded), but those
leaf functions do not exist yet.
"""
import itertools
import math
import warnings

import numpy as np

from glow.analysis import AnalysisCET, AnalysisGLOW, AnalysisVBA
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import get_hotel_tr, get_wilks
from glow.effect import ExtenterMinVar, ExtenterSphere

from . import hcp
from .run import run_ana


# ---------- shared knobs (mirror paper/config_old.py) ------------------------
SOURCES = ['wgn', 'hcp']

N_SEED = 15
N_SEED_NULL = 1000

EFFECT_LLR_GRID = np.logspace(np.log10(0.003), np.log10(0.3), 11)
# the sweeps' shared centre: the grid's middle element. Taking it off the grid
# (not typing 0.03, which misses grid[5] == 0.030000000000000013) is what makes
# sweep_llr's midpoint hash equal to the default-effect anchor the other caches
# plant. float() for a clean python-float hash matching effect_factory's
# float(effect_llr) cast; a single midpoint needs an odd-length grid.
if len(EFFECT_LLR_GRID) % 2 == 0:
    warnings.warn('EFFECT_LLR_GRID is even-length; it has no single midpoint')
MODERATE_EFFECT_LLR = float(EFFECT_LLR_GRID[len(EFFECT_LLR_GRID) // 2])

# Every source is cropped to one connected sphere of this many voxels, so
# num_vox matches across WGN and HCP (get_kwargs_data_list sizes the WGN box
# from it).
CROP_N_VOX = 25_000

# Effect support: 10% of the cropped volume.
EFFECT_N_VOX = int(0.1 * CROP_N_VOX)

N_PERM_FWER = 250
N_PERM_INNER = 1000
ALPHA_FWER = 0.05

# Structural grids. B caps at the HCP pool (6) so every HCP cell is feasible;
# the extent grid spans 1%..100% of the volume; the subject grid is WGN-only.
B_GRID = list(range(1, len(hcp.HCP_FEATS) + 1))
EXTENT_N_VOX_GRID = [int(round(p * CROP_N_VOX))
                     for p in np.geomspace(0.01, 1.0, 15)]
NIMG_GRID = [10, 18, 30, 55, 100, 180, 300]


# ---------- analysis recipes -------------------------------------------------
# label -> recipe. The label is for the reader: it rides into each run_ana
# cell as a label kwarg the function ignores -- recorded as the in.label
# column but dropped from the cache key (run_ana's
# @MEMORY.cache(ignore=['label'])) -- so a method is named beside its score
# without entering the computation. GLOW uses the LLR throughout, so the two
# GLOW arms differ only in Ward projection; the voxel-wise arms z-score
# before the max-stat null.
kwargs = dict(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER)
ana_kwargs_dict = {
    'GLOW-Focus': AnalysisGLOW(n_perm_inner=N_PERM_INNER,
                               cluster_mode=ClusterMode.FOCUS,
                               **kwargs),
    'GLOW-GLM':   AnalysisGLOW(n_perm_inner=N_PERM_INNER,
                               cluster_mode=ClusterMode.GLM_ERROR,
                               **kwargs),
    'VBA':        AnalysisVBA(z_flag=True, tfce_flag=False,
                              get_stat=get_hotel_tr, **kwargs),
    'VBA-TFCE':   AnalysisVBA(z_flag=True, tfce_flag=True, get_stat=get_wilks,
                              **kwargs),
    'CET':        AnalysisCET(z_flag=True, get_stat=get_hotel_tr,
                              **kwargs),
}

# the leaf kwargs grid: one run_ana call per recipe, shared by every cache.
# The ana_kwargs_dict key rides as a label kwarg run_ana ignores (see
# run.run_ana): recorded beside the score in the output, but dropped from the
# cache key, so renaming a method does not invalidate its cached fit.
RUN_ANA_LIST = [dict(ana=ana, label=label)
                for label, ana in ana_kwargs_dict.items()]


# ---------- stage builders (swept axes are the keyword arguments) ------------
def get_kwargs_data_list(*, sources=SOURCES, seeds=range(N_SEED), b_list=(1,),
                         num_img_list=(100,)):
    """Return the list of data_factory kwargs dicts over the swept axes.

    The cartesian product of (source, b, seed); WGN additionally sweeps num_img
    (HCP's N is its cohort, so its cells omit it and never duplicate). The
    analysis crop -- a connected CROP_N_VOX sphere seeded by the cell's seed --
    is built once per (source, b, seed) and shared across a WGN cell's num_img
    values; the HCP feature subset is a sorted random b-subset of the pool.

    Args:
        sources (list[str]): 'wgn' and/or 'hcp'.
        seeds (iterable[int]): per-cell realization seeds.
        b_list (iterable[int]): imaging-feature counts (HCP draws a subset).
        num_img_list (iterable[int]): subject counts (WGN only).

    Returns:
        list[dict]: kwargs for data_factory, one per cell (source selects
            wgn / hcp).
    """
    wgn_side = math.ceil(CROP_N_VOX ** (1 / 3))

    def sample_hcp_feats(b, seed):
        # a sorted random b-subset of the HCP feature pool, per seed
        idx = np.random.default_rng(seed).choice(len(hcp.HCP_FEATS), size=b,
                                                 replace=False)
        return tuple(sorted(hcp.HCP_FEATS[i] for i in idx))

    kwargs_data_list = []
    for source, b, seed in itertools.product(sources, b_list, seeds):
        extenter = ExtenterSphere(n_vox=CROP_N_VOX, connected=True,
                                  contiguous=True, seed=seed)
        if source == 'wgn':
            for num_img in num_img_list:
                kwargs_data_list.append(dict(
                    source='wgn', shape=(wgn_side,) * 3, b=b, num_img=num_img,
                    seed=seed, extenter=extenter))
        else:
            kwargs_data_list.append(dict(
                source='hcp', hcp_feats=sample_hcp_feats(b, seed), seed=seed,
                extenter=extenter))
    return kwargs_data_list


def get_kwargs_effect_list(*, llr_list=(MODERATE_EFFECT_LLR,),
                           n_vox_list=(EFFECT_N_VOX,)):
    """Return the list of effect_factory kwargs dicts over the swept axes.

    The cartesian product of (effect_llr, n_vox): a per-voxel strength and a
    support size. Each cell carries the ingredients effect_factory builds the
    support from -- the ExtenterMinVar class, n_vox, and seed_from_exp=True so
    the placement is derived from the experiment (see the module docstring).
    llr_list=None is the null / FWER-calibration path -- the list [None]
    (plant nothing).

    Args:
        llr_list (iterable[float] | None): per-voxel effect strengths; None is
            the null path.
        n_vox_list (iterable[int]): effect support sizes.

    Returns:
        list[dict | None]: kwargs for effect_factory (exp is supplied by the
            driver), or [None] for the null path.
    """
    if llr_list is None:
        return [None]
    kwargs_effect_list = []
    for llr, n_vox in itertools.product(llr_list, n_vox_list):
        kwargs_effect_list.append(dict(
            effect_llr=float(llr), extenter_cls=ExtenterMinVar,
            n_vox=int(n_vox), seed_from_exp=True))
    return kwargs_effect_list


# ---------- catalogue: name -> (data, effect, fnc kwargs, fnc) ---------------
# Building the grids runs no experiments and reads no data -- the cells are
# just kwargs (the Extenters are built later, in effect_factory /
# data_factory).
CONFIG = {
    # A. Type I error: no effect, many seeds, both sources (null path).
    'null': (
        get_kwargs_data_list(seeds=range(N_SEED_NULL)),
        get_kwargs_effect_list(llr_list=None),
        RUN_ANA_LIST, run_ana),
    # B. Detection vs effect strength (b=1; HCP draws one random feature/seed).
    'sweep_llr': (
        get_kwargs_data_list(),
        get_kwargs_effect_list(llr_list=EFFECT_LLR_GRID),
        RUN_ANA_LIST, run_ana),
    # C. Detection vs feature count (the multivariate story).
    'sweep_b': (
        get_kwargs_data_list(b_list=B_GRID),
        get_kwargs_effect_list(),
        RUN_ANA_LIST, run_ana),
    # D. Detection vs effect extent (fixed per-voxel effect_llr).
    'sweep_extent': (
        get_kwargs_data_list(),
        get_kwargs_effect_list(n_vox_list=EXTENT_N_VOX_GRID),
        RUN_ANA_LIST, run_ana),
    # E. Detection vs subject count (WGN only; HCP's N is its cohort).
    'sweep_nimg': (
        get_kwargs_data_list(sources=['wgn'], num_img_list=NIMG_GRID),
        get_kwargs_effect_list(),
        RUN_ANA_LIST, run_ana),
}
