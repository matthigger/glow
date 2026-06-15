"""Paper-benchmark catalogue (flattened-axis edition).

Each entry of CACHE_BY_LABEL is a (TrialCache, run_fnc) pair. Every trial
axis is a plain scalar/categorical in iter_kwargs -- source, seed, b,
num_img, n_vox_eff, and an effect-strength knob -- so a cache's whole
trial grid is the cartesian product of those lists. The heavy objects
(DataSource, Extenter) are rebuilt from the scalars inside the trial fn
via factory.build_ds, and TrialCache.save_result merges each scalar axis
into every result row. results.csv is therefore tidy long-format: one row
per (trial, method), self-describing by its scalar columns.

WGN and HCP live in one cache per experiment (source=['wgn','hcp']); they
plot side by side, faceted on the source column (see PLOT + plot.py). The
companion PLOT dict declares each cache's plot roles -- which scalar is
the x-axis, which is the facet -- so the plotter never introspects the
data to guess.

Effect strength. effect_llr is the per-voxel (size-normalized) target;
the observed whole-region LLR is ~ effect_llr * n_vox_eff (see
glow.effect.impose). Two consequences, both verified empirically:
  - holding effect_llr fixed while sweeping num_img holds the per-subject
    effect fixed (a clean power curve), so sweep_nimg needs no special
    handling;
  - holding effect_llr fixed while sweeping extent holds the per-voxel
    effect fixed, so the total grows with the region. sweep_extent instead
    passes effect_total_llr (the whole-region target) and the trial fn
    sets effect_llr = effect_total_llr / n_vox_eff to hold the total fixed.

Mothballed experiments (2-D WGN, sphere-extent variants) live in
config_mothball.py and are not imported here.
"""
from functools import partial

import numpy as np

import glow
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import get_hotel_tr, get_wilks
from glow.benchmark import hcp
from glow.benchmark.trial_cache import TrialCache

from .factory import CROP_N_VOX
from .run import (run_ana, run_mancova, run_prune, run_segment,
                  run_two_effect)


# ---------- shared knobs -----------------------------------------------------
SOURCES = ['wgn', 'hcp']

N_SEED = 2
N_SEED_NULL = 10

EFFECT_LLR_GRID = np.logspace(np.log10(0.003), np.log10(0.3), 11)
MODERATE_EFFECT_LLR = 0.03

EFFECT_PERC_TOTAL_VOLUME = 0.1
EFFECT_N_VOX = int(EFFECT_PERC_TOTAL_VOLUME * CROP_N_VOX)

# Whole-region LLR target for the extent sweep, set so the default extent
# (EFFECT_N_VOX) reproduces MODERATE_EFFECT_LLR per voxel.
EXTENT_TOTAL_LLR = MODERATE_EFFECT_LLR * EFFECT_N_VOX

N_PERM_FWER = 250
N_PERM_INNER = 1000
ALPHA_FWER = 0.05

# GLOW inner-perm-race speed/power knobs (never validity knobs; see
# AnalysisGLOW). RACE_INIT is the burn-in inner draws over all regions before
# the survivor trim; RACE_P_KEEP_THRESH keeps any region with > this
# probability of being the per-perm max-z region (scale-free, smaller keeps
# more). Defaults are lossless at RACE_INIT=15 on HCP.
RACE_INIT = 15
RACE_P_KEEP_THRESH = 1e-6

# Per-family VBA design decisions (hoisted for easy scanning / override).
VBA_Z_FLAG = True
VBA_GET_STAT = get_hotel_tr
VBA_TFCE_GET_STAT = get_wilks
CET_GET_STAT = get_hotel_tr


# ---------- analysis recipes -------------------------------------------------
_GLOW_BASE = dict(n_perm_fwer=N_PERM_FWER,
                  n_perm_inner=N_PERM_INNER,
                  alpha_fwer=ALPHA_FWER,
                  race_init=RACE_INIT,
                  race_p_keep_thresh=RACE_P_KEEP_THRESH)
_VBA_BASE = dict(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER,
                 z_flag=VBA_Z_FLAG)

ANALYSIS_DICT = {
    'GLOW-Focus': (glow.analysis.AnalysisGLOW,
                   {**_GLOW_BASE, 'cluster_mode': ClusterMode.FOCUS}),
    'GLOW-GLM':   (glow.analysis.AnalysisGLOW,
                   {**_GLOW_BASE, 'cluster_mode': ClusterMode.GLM_ERROR}),
    'VBA':        (glow.analysis.AnalysisVBA,
                   {**_VBA_BASE, 'tfce_flag': False, 'get_stat': VBA_GET_STAT}),
    'VBA-TFCE':   (glow.analysis.AnalysisVBA,
                   {**_VBA_BASE, 'tfce_flag': True,
                    'get_stat': VBA_TFCE_GET_STAT}),
    'CET':        (glow.analysis.AnalysisCET,
                   {**_VBA_BASE, 'get_stat': CET_GET_STAT}),
}

SEGMENT_MODES = [ClusterMode.NAIVE, ClusterMode.GLM_ERROR, ClusterMode.FOCUS]


# ---------- structural grids ------------------------------------------------
# Feature-count grid. WGN runs the whole grid; HCP clamps to its feature
# pool (run_ana would raise past it), so HCP's facet simply ends earlier.
B_GRID = list(range(1, 7))
_B_GRID_HCP_MAX = len(hcp.HCP_FEATS)

# Effect-extent grid: 1% .. 100% of the cropped volume.
EXTENT_N_VOX_GRID = [int(round(p * CROP_N_VOX))
                     for p in np.geomspace(0.01, 1.0, 15)]

# Subject-count grid (WGN only; HCP's N is its cohort size).
NIMG_GRID = [10, 18, 30, 55, 100, 180, 300]


# ---------- catalogue assembly ----------------------------------------------
# label -> (TrialCache, run_fnc)
CACHE_BY_LABEL = {}


def _cache(label: str, *, run_fnc, **iter_kwargs) -> None:
    """Register one (TrialCache, run_fnc) entry from a flat scalar grid.

    Args:
        label (str): catalogue key / on-disk result folder name
        run_fnc (Callable): bound trial fn (run_ana/run_segment/run_mancova)
        **iter_kwargs: the scalar axes; each value is a list whose
            cartesian product is the cache's trial grid
    """
    # cast numpy scalars to plain python so result columns stay readable
    clean = {k: [v.item() if isinstance(v, np.generic) else v for v in vals]
             for k, vals in iter_kwargs.items()}
    CACHE_BY_LABEL[label] = (TrialCache(name=label, iter_kwargs=clean), run_fnc)


_ana = partial(run_ana, ana_kwargs_dict=ANALYSIS_DICT)
_segment = partial(run_segment, modes=SEGMENT_MODES)
_mancova = partial(run_mancova, n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER)
_two_effect = partial(run_two_effect, ana_kwargs_dict=ANALYSIS_DICT)
# run_prune needs no partial: it has one call site and builds GLOW's default
# recipe (Focus, shared perms) itself.

# Angle (deg) between the two effects' feature directions: 0..90 in 10 steps.
ANGLE_GRID = [float(a) for a in np.linspace(0, 90, 10)]
# Per-voxel effect_llr grid for the two-effect cache. At 25k each ~1250-vox
# half is very high-SNR at 0.03 (per-region LLR ~37 -> GLOW favours the merged
# region), so span weaker SNRs too; the right level is read off the resulting
# ARI(angle) curves and the dice/sens detectability columns.
TWO_EFFECT_LLR_GRID = [0.003, 0.01, 0.03]


# A. Type I error (null): no effect, many seeds, both sources.
_cache('null', run_fnc=_ana,
       source=SOURCES, seed=list(range(N_SEED_NULL)),
       effect_llr=[0.0], b=[1], num_img=[100], n_vox_eff=[EFFECT_N_VOX])

# B. Detection vs effect strength (b=1; HCP draws one random feature/seed).
_cache('sweep_llr', run_fnc=_ana,
       source=SOURCES, seed=list(range(N_SEED)),
       effect_llr=EFFECT_LLR_GRID, b=[1], num_img=[100],
       n_vox_eff=[EFFECT_N_VOX])

# C. Detection vs feature count (the multivariate story).
_cache('sweep_b', run_fnc=_ana,
       source=SOURCES, seed=list(range(N_SEED)),
       effect_llr=[MODERATE_EFFECT_LLR], b=B_GRID, num_img=[100],
       n_vox_eff=[EFFECT_N_VOX])

# D. Detection vs effect extent, holding the whole-region LLR fixed
#    (effect_total_llr; per-voxel llr shrinks as the region grows).
_cache('sweep_extent', run_fnc=_ana,
       source=SOURCES, seed=list(range(N_SEED)),
       effect_total_llr=[EXTENT_TOTAL_LLR], b=[1], num_img=[100],
       n_vox_eff=EXTENT_N_VOX_GRID)

# E. Detection vs subject count (WGN only; fixed per-voxel effect = power
#    curve). HCP's N is its cohort, so it has no num_img axis to sweep.
_cache('sweep_nimg', run_fnc=_ana,
       source=['wgn'], seed=list(range(N_SEED)),
       effect_llr=[MODERATE_EFFECT_LLR], b=[1], num_img=NIMG_GRID,
       n_vox_eff=[EFFECT_N_VOX])

# F. Segmentation quality: oracle Dice of the best region per Ward mode
#    (Naive / GLM Error / Focus), no significance test or pruning.
_cache('segment', run_fnc=_segment,
       source=SOURCES, seed=list(range(N_SEED)),
       effect_llr=EFFECT_LLR_GRID, b=[1], num_img=[100],
       n_vox_eff=[EFFECT_N_VOX])

# G. MANCOVA stat comparison: VBA / VBA-TFCE / CET x 5 stats x {raw, z}
#    (b=2 so the multivariate stats differ). GLOW is excluded by design --
#    it uses LLR throughout (see run.py) -- not pending work.
_cache('stat', run_fnc=_mancova,
       source=SOURCES, seed=list(range(N_SEED)),
       effect_llr=EFFECT_LLR_GRID, b=[2], num_img=[100],
       n_vox_eff=[EFFECT_N_VOX])

# H. Pruning rule -- greedy max-LLR vs DP max-likelihood cut, scored on one
#    shared GLOW-Focus fit per trial (so the comparison isolates the rule,
#    not the permutation test). The unpenalized DP is expected to
#    oversegment -- one effect reported as several children -- which shows
#    up as n_selected (output-region count, ideal 1) climbing with effect
#    strength while greedy stays near 1. See Section ssec:prune.
_cache('prune', run_fnc=run_prune,
       source=SOURCES, seed=list(range(N_SEED)),
       effect_llr=EFFECT_LLR_GRID, b=[1], num_img=[100],
       n_vox_eff=[EFFECT_N_VOX])

# I. Cleaving: two adjacent equal-LLR effects, sweep the angle between their
#    feature directions (0..90 deg). Headline metric is instance separation
#    (ARI of the recovered partition vs the {effect0, effect1} truth), derived
#    downstream from region_overlap_json; the aggregate tp/fp/tn/fn (union vs
#    both effects) give the detectability check. b>=2 so the direction
#    rotation has a plane to turn in.
_cache('two-effect', run_fnc=_two_effect,
       source=SOURCES, seed=list(range(N_SEED)),
       angle=ANGLE_GRID, b=[3], num_img=[100],
       effect_llr=TWO_EFFECT_LLR_GRID, n_vox_eff=[EFFECT_N_VOX])


# ---------- plot specs -------------------------------------------------------
# Per-cache plot roles, read by plot.py instead of inferring from the data
# (robust to partial runs and to a cache that varies >1 ordered axis):
#   kind   -- 'metric' (faceted dice/sens/ppv sweep), 'calibration'
#             (FWER curve from min_pval), or 'mancova' (stat-comparison grid)
#   x      -- the swept scalar column to put on the x-axis
#   facet  -- categorical column(s) to split into side-by-side panels
#   metrics/hue -- optional overrides (default metrics dice/sens/ppv,
#             hue = the method 'label' column). PPV (not specificity) since
#             effects fill a small fraction of the volume; the extent sweep
#             varies that fraction, so it also shows prevalence-free spec.
PLOT = {
    'null':         dict(kind='calibration', facet='source'),
    'sweep_llr':    dict(kind='metric', x='effect_llr',  facet='source'),
    'sweep_b':      dict(kind='metric', x='b',           facet='source'),
    'sweep_extent': dict(kind='metric', x='effect_perc', facet='source',
                         metrics=['dice', 'sens', 'ppv', 'spec']),
    'sweep_nimg':   dict(kind='metric', x='num_img',     facet='source'),
    'segment':      dict(kind='metric', x='effect_llr',  facet='source',
                         metrics=['dice']),
    'stat':         dict(kind='mancova'),
    'prune':        dict(kind='metric', x='effect_llr',  facet='source',
                         metrics=['dice', 'sens', 'ppv', 'n_selected']),
    # ARI (cleaving) is computed downstream from region_overlap_json -- plot.py
    # needs a 'two_effect' branch (TODO); the run itself does not plot.
    'two-effect':   dict(kind='two_effect', x='angle',   facet='source'),
}
