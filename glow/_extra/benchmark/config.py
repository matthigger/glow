"""Paper-benchmark catalogue: the inputs to driver.drive for each figure.

CONFIG maps a cache name (e.g. 'sweep_llr') to the four-tuple drive consumes
-- (kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc): the data
grid, the effect grid, the leaf-function kwargs grid, and the leaf function
itself. So one benchmark figure is drive(*CONFIG[name]). Running the sweep is
the driver's / a CLI's job, not this module's -- config only declares it.

The two upstream grids come from the builders in benchmark.grid, whose keyword
arguments are the swept axes: get_kwargs_data_list over (source, b, num_img,
seed), get_kwargs_effect_list over (effect_llr, n_vox) -- or [None] for the
null path. This module holds no logic: it supplies the paper's values for those
axes (data_grid / effect_grid apply them as defaults) and each cache overrides
the axis it sweeps. A cell carries its heavy objects directly (the Extenter,
the realized HCP feature subset), so the catalogue is what runs.

Seeding. A data cell's seed drives its whole realization -- the WGN draw / HCP
feature subset, the x design, the analysis crop. The effect grid is shared
across data cells and carries no per-data seed, so the support is placed with
seed_from_exp: effect_factory derives its placement seed from a hash of the
experiment, giving each realization its own reproducible location.

Effect strength. effect_llr is the per-voxel (size-normalized) target, so the
whole-region LLR is ~ effect_llr * n_vox (see glow.effect.impose). Sweeps hold
effect_llr fixed, which across a structural axis (b, num_img) is a clean power
curve and across the extent sweep grows the whole-region LLR with the region.

WGN and HCP share each cache and face apart on the recorded source column
downstream. HCP has no num_img axis -- its N is the cohort -- so a subject
sweep cuts that cohort down at analysis time (run.run_ana_time_1perm).

Every cache here backs a figure, table or quantitative claim in the paper, bar
smoke (an end-to-end pipeline check).

Scope. Five caches share the run_ana leaf over the one recipe grid (fit +
score one Analysis per cell): null, sweep_llr, sweep_extent, sweep_b,
sweep_nimg. sweep_llr_glow_tune takes that leaf over a grid of its own, the
four GLOW variants, which is what picks the one the others report. Four swap
in their own leaf over much the same grids: segment (run_segment, a Ward-mode
oracle, no fit -- and segment_perc_llr, the same leaf and modes over the share
of the images the tree is built on), vba_stat (run_stat, a VBA / CET variant
reading a shared voxel-stat walk; HCP only, b=2), prune (run_prune, three
pruning rules plus the max-Dice oracle, all re-selected off one shared GLOW
fit), and sweep_n_perm_inner (run_inner_perm, the reported variant at every
inner-draw count off one shared capture; HCP only), which is what settles
N_PERM_INNER.

Runtime. Five caches measure time, not detection, and run locally only. They
answer two different questions and must not be read as one: runtime_num_vox is
the wall clock a user waits, every method given this machine's cores and card
(run_ana_time); the four runtime_1perm_* are one-at-a-time growth rates on a
single pinned core, one permutation deep (run_ana_time_1perm). See the runtime
section below.

"""
import warnings

import numpy as np

from glow.analysis import (AnalysisCET, AnalysisGLOW, AnalysisGLOWBase,
                           AnalysisVBA, DEFAULT_CET_CFT_PVAL)
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import get_hotel_tr, get_wilks

from . import grid, hcp
from .run import (run_ana, run_ana_time, run_ana_time_1perm, run_inner_perm,
                  run_prune, run_segment, run_stat)


# ---------- shared knobs ------------------------------------------------------
SOURCES = ['wgn', 'hcp']

N_SEED = 50
N_SEED_NULL = 1000

EFFECT_LLR_GRID = np.logspace(np.log10(0.003), np.log10(0.3), 11)
# the sweeps' shared centre: the grid's middle element. Taking it off the grid
# (not typing 0.03, which misses grid[5] == 0.030000000000000013) is what makes
# the llr sweep's b=1 midpoint hash equal to the default-effect anchor the
# other caches plant. float() for a clean python-float hash matching
# effect_factory's float(effect_llr) cast; a single midpoint needs an
# odd-length grid.
if len(EFFECT_LLR_GRID) % 2 == 0:
    warnings.warn('EFFECT_LLR_GRID is even-length; it has no single midpoint')
MODERATE_EFFECT_LLR = float(EFFECT_LLR_GRID[len(EFFECT_LLR_GRID) // 2])

# Every source is cropped to one connected sphere of this many voxels, so
# num_vox matches across WGN and HCP (grid.get_kwargs_data_list sizes the
# WGN box from it).
CROP_N_VOX = 25_000

# Effect support: 10% of the analysis volume. A per-cell fraction, so it
# tracks whatever num_vox each cell is cropped to (see
# grid.get_kwargs_effect_list).
EFFECT_N_VOX_FRAC = 0.1

N_PERM_FWER = 500
# inner Freedman-Lane draws standardizing each outer perm's own tree. It caps
# every z at n_perm_inner / sqrt(n_perm_inner + 1), so it is a resolution
# floor as much as a cost knob (see AnalysisGLOW), and it multiplies the draw
# count: a fit is n_perm_fwer x (n_perm_inner + 1) draws.
N_PERM_INNER = 250
ALPHA_FWER = 0.05

# Structural grids. B caps at the HCP pool (6) so every HCP cell is feasible;
# the extent grid spans 1%..100% of the volume; the subject grid tops out at
# the cohort (HCP_NUM_IMG), which is the sample the runtime_1perm_nimg axis
# cuts down from -- there is no extrapolating past it.
B_GRID = list(range(1, len(hcp.HCP_FEATS) + 1))
# the llr sweep's feature-count axis: b = 1 (the univariate power curve) plus
# its low-b multivariate counterpart. A subset of B_GRID, swept alongside
# effect_llr in one cache (see the sweep_llr entry).
B_LLR_SWEEP = (1, 2)
# sweep_b's own axis: detection vs b at the fixed moderate effect, llr held
# still so a wider b range costs no extra effect_llr cells.
SWEEP_B_GRID = (1, 2, 3, 4)
EXTENT_FRAC_GRID = list(np.geomspace(0.01, 1.0, 15))
HCP_NUM_IMG = 100
NIMG_GRID = [10, 18, 30, 55, HCP_NUM_IMG]
# sweep_nimg's own axis: detection vs subject count. Same 10 -> cohort span as
# NIMG_GRID but in linear steps, where the timing curve takes log-spaced
# strides -- a slope is read off a handful of points, a power curve is read
# point by point.
SWEEP_NIMG_GRID = list(range(10, HCP_NUM_IMG + 1, 10))


# ---------- analysis recipes -------------------------------------------------
# label -> recipe. The label is the reader-facing method name; it is the source
# of truth results / plot map a recorded recipe back to (a run function is not
# passed the label -- see run.py / benchmark.plot). GLOW is the per-perm arm
# (AnalysisGLOW) throughout: it rebuilds the Ward tree inside every outer
# permutation, so every image reaches both the tree and the statistics. Each
# voxel-wise arm takes the stat / z-scoring it wins the vba_stat bake-off with:
# the raw Hotelling-Lawley trace for VBA and CET, the z-scored 1 - Wilks for
# TFCE (z-scoring is what TFCE's single height grid needs to mean the same
# thing at every voxel). Written out rather than left to the recipe defaults,
# which agree -- the paper's arms should be readable here.
#
# The four GLOW entries are one arm under its two free choices: the Ward
# projection (Focus on the contrast subspace, GLM Error on the whole design
# space) crossed with the selection rule (greedy, or the exact max-total-LLR
# antichain). Only REPORTED_GLOW_LABEL rides the shared leaf grid; the other
# three are compared on one cache of their own (sweep_llr_glow_tune), which
# is what settles the choice.
kwargs_voxel = dict(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER)
kwargs = dict(n_perm_inner=N_PERM_INNER, **kwargs_voxel)
GLOW_LABEL_LIST = ('GLOW-Focus-greedy', 'GLOW-Focus-dp',
                   'GLOW-GLM-greedy', 'GLOW-GLM-dp')
ana_kwargs_dict = {
    'GLOW-Focus-greedy': AnalysisGLOW(cluster_mode=ClusterMode.FOCUS,
                                      prune_rule='greedy', **kwargs),
    'GLOW-Focus-dp':     AnalysisGLOW(cluster_mode=ClusterMode.FOCUS,
                                      prune_rule='dp', **kwargs),
    'GLOW-GLM-greedy':   AnalysisGLOW(cluster_mode=ClusterMode.GLM_ERROR,
                                      prune_rule='greedy', **kwargs),
    'GLOW-GLM-dp':       AnalysisGLOW(cluster_mode=ClusterMode.GLM_ERROR,
                                      prune_rule='dp', **kwargs),
    'VBA':        AnalysisVBA(z_flag=False, tfce_flag=False,
                              get_stat=get_hotel_tr, **kwargs_voxel),
    'VBA-TFCE':   AnalysisVBA(z_flag=True, tfce_flag=True,
                              get_stat=get_wilks, **kwargs_voxel),
    'CET':        AnalysisCET(z_flag=False, get_stat=get_hotel_tr,
                              **kwargs_voxel),
}

# The GLOW variant the figures report, and the only one on the shared leaf
# grid: benchmark.plot drops the other three (_ARMS_SKIP) and calls this one
# plainly GLOW. It is also the only variant the runtime caches time -- all
# four run the same permutation walk over the same shapes and differ in what
# Ward is handed and how the significant set is cut, neither of which moves a
# timing, so a second set would be a second copy of one curve.
REPORTED_GLOW_LABEL = 'GLOW-Focus-greedy'

# ---------- how a leaf's fit runs (never what it computes) -------------------
# fit_params is forwarded to Analysis.fit by the leaf and filtered out of the
# cache key and the record (run.FIT_IGNORE), so it may vary by machine without
# forking an artifact.
#
# GLOW alone parallelises deeply enough to be worth configuring. The count is
# an upper bound, clamped to the machine's cores at fit time
# (resolve_n_jobs), so a smaller box quietly uses what it has. What caps it is
# RAM, not cores: a GLOW worker holds its own copy of y, ~1 GB at full-brain
# num_vox.
#
# gpu is 'auto', not True: the device backend is a pure speedup (same seeds,
# float64, agreeing with the CPU fit to round-off), so a checked-in True would
# only break CPU-only runners. Both knobs multiply against the sweep's own -j;
# driver.check_fit_params refuses the products that would oversubscribe.
GLOW_FIT_N_JOBS = 10
GLOW_FIT_PARAMS = dict(n_jobs=GLOW_FIT_N_JOBS, gpu='auto')


# the leaf kwargs grid: one run_ana call per recipe, shared by every cache.
# Only the ana rides into the cell, plus how to run it; the method name (the
# ana_kwargs_dict key) is recovered from the recipe at read time (see
# benchmark.plot), so it never enters the call or the cache key.
#
# The reported GLOW variant plus the voxel-wise arms, NOT all four GLOW
# variants: this grid is shared by six caches, so an extra GLOW entry here
# costs a per-perm fit in each of them. The other three are compared on
# GLOW_ARM_LIST's cache alone.
def _run_ana(label: str) -> dict:
    """The run_ana leaf kwargs for one catalogue label."""
    ana = ana_kwargs_dict[label]
    return dict(ana=ana, fit_params=grid.fit_params_for(ana,
                                                        GLOW_FIT_PARAMS))


RUN_ANA_LIST = [_run_ana(label) for label in ana_kwargs_dict
                if label == REPORTED_GLOW_LABEL
                or label not in GLOW_LABEL_LIST]

# the sweep_llr_glow_tune cache's leaf grid: the four GLOW variants and nothing
# else. Ward projection x selection rule, on the llr sweep's own axes at b=1,
# which is what the choice of REPORTED_GLOW_LABEL rests on. The voxel-wise
# arms are absent -- sweep_llr already carries them over the same cells, so
# repeating them here would pay twice for one curve.
GLOW_ARM_LIST = [_run_ana(label) for label in GLOW_LABEL_LIST]


# the sweep_n_perm_inner cache's leaf grid: the reported GLOW variant at every
# inner-draw count on the grid, which is what settles N_PERM_INNER (on the grid
# itself, so the shipped setting is one of the points read). It spans a count
# too small to resolve the effect's z (the ceiling of AnalysisGLOW) and counts
# above the shipped one, since a knee is only visible from both sides.
#
# n_perm_inner rides as an explicit run_inner_perm argument rather than inside
# a recipe, so it keys the leaf and reaches the record as its own column. The
# grid rides along as well: a cell's counts share one capture
# (run.glow_inner_capture), and it is filtered out of the key because a prefix
# is the same draws whatever depth was sampled around it. The Ward projection
# and selection rule are read off the reported recipe rather than retyped, so
# this cache tunes the shipped variant by construction.
N_PERM_INNER_GRID = (25, 50, 100, N_PERM_INNER, 500, 1000)

# Its own outer-perm count, below the catalogue's N_PERM_FWER. A cell's cost is
# n_perm_fwer x (the deepest count + 1) draws, several times a shipped fit's,
# and the curve is read within a cell -- every count is tested against a null
# drawn from the same outer permutations, so what MC noise the threshold has is
# common to the points being compared rather than scattered between them. A
# detection level read off this cache is therefore compared along its own x,
# not against sweep_llr's.
INNER_N_PERM_FWER = 250
_INNER_ANA = ana_kwargs_dict[REPORTED_GLOW_LABEL]
RUN_INNER_PERM_LIST = [
    dict(n_perm_inner=n_perm_inner, n_perm_fwer=INNER_N_PERM_FWER,
         alpha_fwer=ALPHA_FWER, n_perm_inner_grid=N_PERM_INNER_GRID,
         cluster_mode=_INNER_ANA.cluster_mode,
         prune_rule=_INNER_ANA.prune_rule, fit_params=GLOW_FIT_PARAMS)
    for n_perm_inner in N_PERM_INNER_GRID]

# its effect axis: the moderate anchor and one step of EFFECT_LLR_GRID either
# side of it (two grid points out, so the three are visibly apart). The count a
# tree needs is not one number -- z scales with the effect, and the inner count
# caps z -- so the knee is read at three strengths rather than at the anchor
# alone. Every value is on EFFECT_LLR_GRID, so each plant is a cell sweep_llr
# has already built. float() for the clean python-float hash effect_factory
# casts to (see MODERATE_EFFECT_LLR).
_LLR_MID = len(EFFECT_LLR_GRID) // 2
INNER_EFFECT_LLR = tuple(float(EFFECT_LLR_GRID[i])
                         for i in (_LLR_MID - 2, _LLR_MID, _LLR_MID + 2))


# the segment cache's leaf grid: one run_segment call per Ward mode (Naive /
# GLM Error / Focus). The mode rides in as cluster_mode; the method name is
# str(mode), recovered from the record at read time.
SEGMENT_MODES = [ClusterMode.NAIVE, ClusterMode.GLM_ERROR, ClusterMode.FOCUS]
RUN_SEGMENT_LIST = [dict(cluster_mode=mode) for mode in SEGMENT_MODES]


# the segment_perc_llr cache's leaf grid: the same Ward modes crossed with the
# share of the images the tree is built on (run_segment's frac_segment). It
# stops at 0.9 because a split always holds a test fold back; the whole-cohort
# ceiling is the segment cache's own moderate-effect slice, which these cells
# share. 0.5 is on the grid, so what GLOW's default split costs the
# segmentation is read straight off the figure.
SEGMENT_FRAC_GRID = [round(0.1 * i, 2) for i in range(1, 10)]
RUN_SEGMENT_PERC_LIST = [dict(cluster_mode=mode, frac_segment=frac)
                         for mode in SEGMENT_MODES
                         for frac in SEGMENT_FRAC_GRID]


# the vba_stat cache's leaf grid: the voxel-wise stat bake-off
# (VBA / VBA-TFCE / CET x the stat pool x {raw, z}).
RUN_STAT_LIST = grid.get_run_stat_list(
    n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER,
    cft_pval=DEFAULT_CET_CFT_PVAL)

# the prune cache's leaf grid: four rules (greedy / DP / the single max-LLR
# region / the max-Dice oracle) crossed with the two Ward clustering modes
# (Focus / GLM Error). The oracle is the headroom line, not a method: it is
# handed the planted support the others are scored against, so the gap to it
# is what the ranking leaves on the table at a fixed fit (see
# prune.prune_oracle). All carry the same GLOW fit knobs, so run_prune's
# glow_fit_for_prune is shared across the rules at a given mode (one fit per
# (cell, mode), the first rule fits and the rest hit). The rule names the
# method (GLOW-<rule>) and
# cluster_mode is the second cache axis -- benchmark.plot splits it into one
# metric grid per mode. Mode is the outer loop so a mode's rules are
# contiguous (the shared-fit hits land back to back). fit_params is the same
# GLOW_FIT_PARAMS the RUN_ANA_LIST GLOW recipe takes: the shared fit is GLOW's
# alone, so it gets GLOW's device + worker-count knobs like every other GLOW
# leaf (driver.check_fit_params refuses to run it in parallel with a device
# visible, same as RUN_ANA_LIST's GLOW cell).
_PRUNE_GLOW_KWARGS = dict(n_perm_fwer=N_PERM_FWER, n_perm_inner=N_PERM_INNER,
                          alpha_fwer=ALPHA_FWER, fit_params=GLOW_FIT_PARAMS)
PRUNE_RULES = ['single_max', 'greedy', 'dp', 'oracle']
PRUNE_CLUSTER_MODES = [ClusterMode.FOCUS, ClusterMode.GLM_ERROR]
RUN_PRUNE_LIST = [
    dict(rule=rule, cluster_mode=mode, **_PRUNE_GLOW_KWARGS)
    for mode in PRUNE_CLUSTER_MODES
    for rule in PRUNE_RULES
]


# ---------- smoke test (a tiny non-paper cache for end-to-end checks) --------
# Not a paper figure: a tiny null sweep over both sources, to check the whole
# pipeline runs end to end -- the driver builds each cell from this CONFIG,
# fits it, and the records come back.
# It is the null path with two overrides only -- a small crop and few seeds --
# so a cell finishes fast; sources, num_img, the RUN_ANA_LIST recipes, and the
# paper permutation counts stay standard, so it exercises the real recipes.
SMOKE_CROP_N_VOX = 1_000
SMOKE_N_SEED = 3


# ---------- the paper's axes (grid builds them; here are its values) ---------
# Every cache starts from these axes and overrides the one it sweeps. The
# defaults live here rather than in grid's signatures so that module stays free
# of paper constants -- tuning a value here cannot change what grid's tests
# cover.
DATA_AXES = dict(sources=SOURCES, seeds=range(N_SEED), b_list=(1,),
                 num_img_list=(100,), crop_n_vox=CROP_N_VOX)
EFFECT_AXES = dict(llr_list=(MODERATE_EFFECT_LLR,),
                   n_vox_frac_list=(EFFECT_N_VOX_FRAC,))


def data_grid(**kwargs):
    """Build a data grid on the paper's axes, with per-cache overrides.

    Args:
        **kwargs: any DATA_AXES key, overriding the paper's value for it.

    Returns:
        list[dict]: kwargs for data_factory, one per cell.
    """
    return grid.get_kwargs_data_list(**{**DATA_AXES, **kwargs})


def effect_grid(**kwargs):
    """Build an effect grid on the paper's axes, with per-cache overrides.

    Args:
        **kwargs: any EFFECT_AXES key, overriding the paper's value for it.
            llr_list=None is the null path (plant nothing).

    Returns:
        list[dict | None]: kwargs for effect_factory, or [None] for the null.
    """
    return grid.get_kwargs_effect_list(**{**EFFECT_AXES, **kwargs})


# ---------- runtime benchmarks (HCP) -----------------------------------------
# Wall time, not detection. Every cache here is HCP, with the moderate default
# effect; only the timed axis varies. A recorded time_sec means the machine
# that recorded it, so these compare only within one run on one box. Each cache
# gets its own seed offset so its leaf timings are cold and independent.
#
# Two questions, deliberately not mixed:
#
# runtime_num_vox -- how long does a user wait? Every method fit with what the
# machine has (run_ana_time), so the answer includes this box's cores and card.
# A number to quote, not to extrapolate from: the parallel speedup itself
# varies along the axis. This is where the methods are compared, so it runs the
# three voxel-wise arms against the reported GLOW arm alone.
#
# runtime_1perm_* -- how does GLOW's cost grow? One core, no device, BLAS
# pinned to one thread, one outer permutation (run_ana_time_1perm), everything
# but the swept axis at the shared centre. Held that still, the ratio between
# two points is the growth in that axis, which is what the paper's cost model
# claims: linear in num_vox and num_img, quadratic in b, linear in each
# permutation count. Measured, not a formal complexity result. GLOW only -- it
# is GLOW's cost model being checked, and the arms are compared above.
RUNTIME_N_SEED = 3

# num_vox sweep: 1k -> the full HCP support (224,619 voxels, one connected
# component), roughly doubling. Shared by both runtime families.
RUNTIME_NUM_VOX_GRID = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000,
                        128_000, 224_619]

# The 1perm permutation-count axes. n_perm_fwer starts at the family's own
# baseline of 1 and doubles: the intercept (observed pass + synthesis) does not
# shrink with the count, so the slope is only readable against a point that is
# almost all intercept. n_perm_fwer is the only permutation axis: one tree
# means one draw matrix, so cost is linear in it alone.
ONE_PERM_N_PERM_FWER_GRID = [1, 2, 4, 8, 16]

# per-cache seed offsets, clear of each other, so no two runtime caches share a
# data cell (hence a cached leaf timing). runtime_num_vox keeps the offset the
# retired 'runtime' cache used, so its records and cached fits carry over.
RUNTIME_SEED_OFFSET = {
    'runtime_num_vox': 200_000,
    'runtime_1perm_num_vox': 210_000,
    'runtime_1perm_n_perm_fwer': 220_000,
    'runtime_1perm_b': 240_000,
    'runtime_1perm_nimg': 250_000,
}


def runtime_data_grid(**kwargs):
    """Build a runtime cache's data grid (a crop sweep, RUNTIME_N_SEED seeds).

    Args:
        **kwargs: grid.get_kwargs_data_runtime arguments; seed_offset and
            crop_n_vox_list are required.

    Returns:
        list[dict]: kwargs for data_factory, one per cell.
    """
    return grid.get_kwargs_data_runtime(n_seed=RUNTIME_N_SEED, **kwargs)


# runtime_num_vox's leaf grid: every method fit with what this machine has --
# the three voxel-wise arms and the reported GLOW arm (REPORTED_GLOW_LABEL;
# the unreported one measures the same walk).
# The voxel-wise arms take run_ana_time's default (all cores, no device --
# none of them has a backend); GLOW takes GLOW_FIT_PARAMS, which adds the
# device and caps the workers at GLOW_FIT_N_JOBS. The cap is RAM, not
# preference: a GLOW worker holds its own copy of y, so n_jobs=-1 exhausts
# memory at the grid's full-brain point.
#
# This makes the figure a wall-clock answer for one machine rather than a
# like-for-like algorithm comparison -- which is the question it is asked, and
# why the growth rates are runtime_1perm's job instead. GOTCHA fit_params does
# not key a leaf (run.FIT_IGNORE), so a timing is cached as whatever hardware
# reached it first: re-timing on another box, or with the card pulled, means
# clearing the entry rather than just re-running.
RUN_ANA_TIME_LIST = [dict(ana=ana, fit_params=grid.fit_params_for(
                              ana, GLOW_FIT_PARAMS))
                     for label, ana in ana_kwargs_dict.items()
                     if not isinstance(ana, AnalysisGLOWBase)
                     or label == REPORTED_GLOW_LABEL]

# The runtime_1perm leaf grids. GLOW alone: the cost model these caches back
# is GLOW's, and the voxel-wise arms are compared on the wall clock
# (runtime_num_vox) where the comparison is the point. The recipe names the
# method and only that -- every swept knob rides as an explicit
# run_ana_time_1perm argument, so it keys the cache and reaches the record as
# its own column, and the plotter recovers the method from in.ana exactly as it
# does elsewhere. No fit_params: the leaf pins its own core, device and BLAS
# thread (see run).
ONE_PERM_ANA = ana_kwargs_dict[REPORTED_GLOW_LABEL]
RUN_1PERM_LIST = [dict(ana=ONE_PERM_ANA)]
RUN_1PERM_FWER_LIST = [dict(ana=ONE_PERM_ANA, n_perm_fwer=n)
                       for n in ONE_PERM_N_PERM_FWER_GRID]
# num_img rides the leaf, not the data grid: see run.run_ana_time_1perm for why
# HCP is not given a subject-subset axis.
RUN_1PERM_NIMG_LIST = [dict(ana=ONE_PERM_ANA, num_img=n) for n in NIMG_GRID]


# ---------- catalogue: name -> (data, effect, fnc kwargs, fnc) ---------------
# Building the grids runs no experiments and reads no data -- the cells are
# just kwargs (the Extenters are built later, in effect_factory /
# data_factory).
CONFIG = {
    # A. Type I error: no effect, many seeds, both sources (null path).
    'null': (
        data_grid(seeds=range(N_SEED_NULL)),
        effect_grid(llr_list=None),
        RUN_ANA_LIST, run_ana),
    # B. Detection vs effect strength at b = 1 and b = 2 (the univariate power
    #    curve and its low-b multivariate counterpart) in one sweep over
    #    (b, effect_llr); HCP draws a random b-subset per seed. b rides the
    #    data grid alongside the full effect_llr grid, so the plot holds b
    #    fixed per figure (plot.plot_cache splits on it) -- one each. The b=1
    #    slice matches the standalone anchor the other caches plant, and every
    #    b's midpoint llr coincides with the moderate-effect anchor.
    'sweep_llr': (
        data_grid(b_list=B_LLR_SWEEP),
        effect_grid(llr_list=EFFECT_LLR_GRID),
        RUN_ANA_LIST, run_ana),
    # C. Which GLOW variant to report: Ward projection x selection rule, on
    #    sweep_llr's own axes at b=1. b=1 alone because the choice is between
    #    four variants of one arm rather than a power curve -- a second b
    #    would double the fits to re-read the same ranking -- and its cells
    #    are sweep_llr's b=1 cells, so every data / effect build is a hit and
    #    only the fits are new. REPORTED_GLOW_LABEL is the variant this cache
    #    picks out; benchmark.plot keeps all four under their own labels here
    #    (_BOTH_ARM_CACHES) and drops the unreported three everywhere else.
    'sweep_llr_glow_tune': (
        data_grid(b_list=(1,)),
        effect_grid(llr_list=EFFECT_LLR_GRID),
        GLOW_ARM_LIST, run_ana),
    # How many inner draws GLOW needs: the reported variant read at every
    # count on N_PERM_INNER_GRID, at three effect strengths, which is what
    # settles N_PERM_INNER. Two curves come out of one leaf grid -- which
    # region the max-z statistic comes from (a count is high enough only once
    # the argmax has stopped moving) and what the count costs detection.
    # HCP only, like the runtime family: the count is a property of the tree
    # and the data it is standardized against, and one source answers it at
    # half the fits. A cell's counts share one capture, so the whole grid
    # costs its deepest count (run.run_inner_perm), and its cells are
    # sweep_llr's own b=1 HCP cells, so every build is a hit.
    'sweep_n_perm_inner': (
        data_grid(sources=['hcp']),
        effect_grid(llr_list=INNER_EFFECT_LLR),
        RUN_INNER_PERM_LIST, run_inner_perm),
    # D. Detection vs effect extent (fixed per-voxel effect_llr).
    'sweep_extent': (
        data_grid(),
        effect_grid(n_vox_frac_list=EXTENT_FRAC_GRID),
        RUN_ANA_LIST, run_ana),
    # Detection vs feature count b (fixed moderate effect, both sources; HCP
    # draws a random b-subset per seed -- see grid.get_kwargs_data_list).
    'sweep_b': (
        data_grid(b_list=SWEEP_B_GRID),
        effect_grid(),
        RUN_ANA_LIST, run_ana),
    # Detection vs subject count (fixed moderate effect). WGN only: HCP's N is
    # its fixed cohort, and data_factory_hcp has no subject-subset axis to
    # build a smaller one with (see run.run_ana_time_1perm, which cuts the
    # cohort at analysis time for the timing curve instead).
    'sweep_nimg': (
        data_grid(sources=['wgn'], num_img_list=SWEEP_NIMG_GRID),
        effect_grid(),
        RUN_ANA_LIST, run_ana),
    # F. Segmentation quality: oracle best-Dice region per Ward mode (Naive /
    #    GLM Error / Focus), no significance test or pruning. Same grids as
    #    the b=1 llr sweep; the leaf is run_segment over the mode grid.
    'segment': (
        data_grid(),
        effect_grid(llr_list=EFFECT_LLR_GRID),
        RUN_SEGMENT_LIST, run_segment),
    # Segmentation quality vs sample size: the same oracle per Ward mode at the
    # Segmentation quality vs the share of the images the tree is built on
    # (SEGMENT_FRAC_GRID), across the whole llr axis: what a smaller
    # segmentation fold costs, read at every effect strength (benchmark.plot
    # draws it as one grid per Ward mode, a curve per fold share). Both grids
    # are segment's own, so its cells are segment's cells and only the fold
    # leaves are new work; the whole-cohort ceiling each curve is read against
    # is segment's llr sweep, which the record walk reaches from these same
    # cells. frac_segment is a knob of the split arm, which the catalogue no
    # longer reports -- this cache is the measurement of why.
    'segment_perc_llr': (
        data_grid(),
        effect_grid(llr_list=EFFECT_LLR_GRID),
        RUN_SEGMENT_PERC_LIST, run_segment),
    # G. MANCOVA stat comparison: VBA / VBA-TFCE / CET x 5 stats x {raw, z}
    #    (b=2 so the multivariate stats differ). The cell's variants share one
    #    voxel-stat walk (run_stat -> voxel_stat_walk). GLOW excluded. HCP only
    #    -- the comparison is within a planted cell, so the WGN half doubles
    #    the grid without sharpening the ranking; the b=2 HCP cells are shared
    #    with sweep_llr, so their data / effect builds are cache hits.
    'vba_stat': (
        data_grid(sources=['hcp'], b_list=[2]),
        effect_grid(llr_list=EFFECT_LLR_GRID),
        RUN_STAT_LIST, run_stat),
    # H. Pruning rule: greedy max-LLR vs DP max-likelihood cut vs the single
    #    max-LLR region, scored on one shared GLOW fit per (cell, Ward mode) so
    #    the comparison isolates the rule, not the permutation test. Crossed
    #    with both clustering modes (Focus / GLM Error); benchmark.plot draws
    #    one metric grid per mode.
    'prune': (
        data_grid(),
        effect_grid(llr_list=EFFECT_LLR_GRID),
        RUN_PRUNE_LIST, run_prune),
    # Smoke: tiny null sweep over both sources to confirm the pipeline end to
    # end (not a paper figure). The null path with only a small crop and few
    # seeds overridden (see SMOKE_* above). WGN and HCP cells (3 each).
    'smoke': (
        data_grid(seeds=range(SMOKE_N_SEED),
                             crop_n_vox=SMOKE_CROP_N_VOX),
        effect_grid(llr_list=None),
        RUN_ANA_LIST, run_ana),
    # Runtime (wall clock): time vs num_vox (1k -> full HCP), all methods, b=1,
    # the moderate effect. What a user waits for on this machine, so each
    # method is given everything it can use (RUN_ANA_TIME_LIST: all cores, and
    # the device for GLOW). HCP-only.
    'runtime_num_vox': (
        runtime_data_grid(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_num_vox'],
            crop_n_vox_list=RUNTIME_NUM_VOX_GRID),
        effect_grid(),
        RUN_ANA_TIME_LIST, run_ana_time),
    # Runtime (cost): four one-at-a-time sweeps around the shared centre --
    # b=1, the whole cohort, CROP_N_VOX voxels, the moderate effect, one
    # permutation -- each on one core with no device
    # and BLAS pinned (run_ana_time_1perm), GLOW only. One axis moves per
    # cache, so the ratio between two of its points is the growth in that axis
    # and nothing else. Full grids, since a cell costs two passes, not 501.
    #
    # num_vox on the wall-clock cache's own grid, so the two families are read
    # against the same volumes.
    'runtime_1perm_num_vox': (
        runtime_data_grid(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_1perm_num_vox'],
            crop_n_vox_list=RUNTIME_NUM_VOX_GRID),
        effect_grid(),
        RUN_1PERM_LIST, run_ana_time_1perm),
    # n_perm_fwer over ONE_PERM_N_PERM_FWER_GRID: the slope is the cost of an
    # outer permutation, the intercept the observed pass plus synthesis.
    'runtime_1perm_n_perm_fwer': (
        runtime_data_grid(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_1perm_n_perm_fwer'],
            crop_n_vox_list=[CROP_N_VOX]),
        effect_grid(),
        RUN_1PERM_FWER_LIST, run_ana_time_1perm),
    # b = 1..6, the paper's b-quadratic claim; b rides the data grid.
    'runtime_1perm_b': (
        runtime_data_grid(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_1perm_b'],
            crop_n_vox_list=[CROP_N_VOX], b_list=B_GRID),
        effect_grid(),
        RUN_1PERM_LIST, run_ana_time_1perm),
    # num_img over NIMG_GRID (10 -> the full cohort), the N-linearity claim.
    # The subjects are cut from the cohort by the leaf, so the sweep tops out
    # at the cohort rather than extrapolating past it.
    'runtime_1perm_nimg': (
        runtime_data_grid(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_1perm_nimg'],
            crop_n_vox_list=[CROP_N_VOX]),
        effect_grid(),
        RUN_1PERM_NIMG_LIST, run_ana_time_1perm),
}
