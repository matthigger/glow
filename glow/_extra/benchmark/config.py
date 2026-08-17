"""Paper-benchmark catalogue: the inputs to driver.drive for each figure.

CONFIG maps a cache name (e.g. 'sweep_llr') to the four-tuple drive consumes
-- (kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc): the data
grid, the effect grid, the leaf-function kwargs grid, and the leaf function
itself. So one benchmark figure is drive(*CONFIG[name]). Running the sweep is
the driver's / a CLI's job, not this module's -- config only declares it.

The two upstream grids come from the builders in benchmark.grid, whose keyword
arguments are the swept axes: grid.get_kwargs_data_list returns the list of
data_factory kwargs over (source, b, num_img, seed);
grid.get_kwargs_effect_list returns the list of effect_factory kwargs over
(effect_llr, n_vox) -- or [None]
for the null path. This module holds no logic: it supplies the paper's values
for those axes (data_grid / effect_grid apply them as defaults) and each cache
overrides the axis it sweeps. The leaf is run_ana over the shared analysis
recipes (RUN_ANA_LIST).

A cell carries its heavy objects directly -- the Extenter (effect support,
analysis crop) and the realized HCP feature subset -- so the catalogue is what
runs, with no factory indirection.

Seeding. A data cell's seed drives its whole realization (the WGN draw / HCP
feature subset, the x design, and the analysis crop location). The driver
decouples the data and effect loops (one effect grid is shared across all data
cells), so the effect support is placed with seed_from_exp -- effect_factory
derives its placement seed from a hash of the experiment, so each realization
gets its own (reproducible) effect location even though the effect grid is
shared and carries no per-data seed (see effect_factory).

Effect strength. effect_llr is the per-voxel (size-normalized) target; the
observed whole-region LLR is ~ effect_llr * n_vox (see glow.effect.impose).
Sweeps hold effect_llr fixed: across a structural axis (b, num_img) that is a
clean power curve, and across the extent (n_vox) sweep the whole-region LLR
grows with the region. (The factory has no whole-region-LLR knob -- only the
per-voxel effect_llr -- so an extent sweep at fixed total would have to back
it out here; we don't.)

WGN and HCP share each cache (both sources in one data grid); they face apart
on the recorded source column downstream. HCP has no num_img axis -- its N is
the cohort -- so a subject sweep cuts that cohort down at analysis time rather
than building to a count (see run.run_ana_time_1perm).

Every cache here backs a figure, table or quantitative claim in the paper, bar
smoke (an end-to-end pipeline check). Adding one is cheap; the catalogue is
kept at what is cited.

Scope. Five caches share the run_ana leaf (fit + score one Analysis per cell):
null, sweep_llr, sweep_extent, sweep_b, sweep_nimg. Three swap in their own
leaf over much the same grids: segment (run_segment, a Ward-mode oracle, no
fit -- and segment_perc / segment_perc_llr, the same leaf and modes over the
share of the images the tree is built on, at the moderate effect and across the
llr sweep), vba_stat (run_stat, a VBA / CET variant reading a shared voxel-stat
walk; HCP only, b=2), and prune (run_prune, three pruning rules on a shared
GLOW fit -- and sweep_llr_prune, the same leaf and rules over sweep_llr's own
(b, effect_llr) grid).

Runtime. Five caches measure time, not detection, and run locally only. They
answer two different questions and must not be read as one: runtime_num_vox is
the wall clock a user waits, every method given this machine's cores and card
(run_ana_time); the four runtime_1perm_* are one-at-a-time growth rates on a
single pinned core, one permutation deep (run_ana_time_1perm). See the runtime
section below.

"""
import warnings

import numpy as np

from glow.analysis import (AnalysisCET, AnalysisGLOW, AnalysisVBA,
                           DEFAULT_CET_CFT_PVAL)
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import get_hotel_tr, get_wilks

from . import grid, hcp
from .run import (run_ana, run_ana_time, run_ana_time_1perm, run_prune,
                  run_segment, run_stat)


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
# passed the label -- see run.py / benchmark.plot). GLOW uses the LLR
# throughout, so the two GLOW arms differ only in Ward projection: Focus
# clusters on the contrast subspace, GLM Error on the whole design space (the
# segment / prune families compare the same pair on their own leaves). Each
# voxel-wise arm takes the stat / z-scoring it wins the vba_stat bake-off with:
# the raw Hotelling-Lawley trace for VBA and CET, the z-scored 1 - Wilks for
# TFCE (z-scoring is what TFCE's single height grid needs to mean the same
# thing at every voxel). Written out rather than left to the recipe defaults,
# which agree -- the paper's arms should be readable here.
kwargs = dict(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER)
ana_kwargs_dict = {
    'GLOW-Focus': AnalysisGLOW(cluster_mode=ClusterMode.FOCUS,
                               **kwargs),
    'GLOW-GLM':   AnalysisGLOW(cluster_mode=ClusterMode.GLM_ERROR,
                               **kwargs),
    'VBA':        AnalysisVBA(z_flag=False, tfce_flag=False,
                              get_stat=get_hotel_tr, **kwargs),
    'VBA-TFCE':   AnalysisVBA(z_flag=True, tfce_flag=True, get_stat=get_wilks,
                              **kwargs),
    'CET':        AnalysisCET(z_flag=False, get_stat=get_hotel_tr,
                              **kwargs),
}

# The GLOW arm the figures report (benchmark.plot drops Focus and calls this
# one plainly GLOW) and, with it, the only arm the runtime caches time: the
# two arms run the same permutation walk over the same shapes and differ only
# in what Ward is handed, so a second set of timings would be a second copy of
# one curve at the cost of the grid's full-brain points. Detection is another
# matter -- there the arms can disagree, so every run_ana cache runs both.
REPORTED_GLOW_LABEL = 'GLOW-GLM'

# ---------- how a leaf's fit runs (never what it computes) -------------------
# fit_params is forwarded to Analysis.fit by the leaf (run.run_ana) and is
# filtered out of the cache key and the record (run.FIT_IGNORE), so it may vary
# by machine without forking an artifact: a cell fit here on the GPU and the
# same cell fit on another machine's cores are one recorded score.
#
# GLOW alone parallelises deeply enough to be worth configuring. The count is
# an upper bound, clamped to the machine's cores at fit time
# (glow.analysis.resolve_n_jobs), so this figure is the local workstation's and
# a smaller box quietly uses what it has. What caps it is RAM, not cores: a
# GLOW worker holds its own copy of y, ~1 GB at full-brain num_vox.
#
# gpu is 'auto', not True: the device backend is a pure speedup (same seeds,
# float64, agreeing with the CPU fit to round-off -- see AnalysisGLOW.fit), so
# taking it where a card exists and skipping it where none does is always
# right, and a checked-in True would break every CPU-only runner. Both knobs
# multiply against the sweep's own -j; driver.check_fit_params refuses the
# products that would oversubscribe.
GLOW_FIT_N_JOBS = 10
GLOW_FIT_PARAMS = dict(n_jobs=GLOW_FIT_N_JOBS, gpu='auto')


# the leaf kwargs grid: one run_ana call per recipe, shared by every cache.
# Only the ana rides into the cell, plus how to run it; the method name (the
# ana_kwargs_dict key) is recovered from the recipe at read time (see
# benchmark.plot), so it never enters the call or the cache key.
RUN_ANA_LIST = [dict(ana=ana, fit_params=grid.fit_params_for(ana,
                                                             GLOW_FIT_PARAMS))
                for ana in ana_kwargs_dict.values()]


# the segment cache's leaf grid: one run_segment call per Ward mode (Naive /
# GLM Error / Focus). The mode rides in as cluster_mode; the method name is
# str(mode), recovered from the record at read time.
SEGMENT_MODES = [ClusterMode.NAIVE, ClusterMode.GLM_ERROR, ClusterMode.FOCUS]
RUN_SEGMENT_LIST = [dict(cluster_mode=mode) for mode in SEGMENT_MODES]


# the segment_perc cache's leaf grid: the same Ward modes crossed with the
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

# the prune cache's leaf grid: three rules (greedy / DP / the single max-LLR
# region) crossed with the two Ward clustering modes (Focus / GLM Error). All
# carry the same GLOW fit knobs, so run_prune's glow_fit_for_prune is shared
# across the three rules at a given mode (one fit per (cell, mode), the first
# rule fits and the rest hit). The rule names the method (GLOW-<rule>) and
# cluster_mode is the second cache axis -- benchmark.plot splits it into one
# metric grid per mode. Mode is the outer loop so a mode's three rules are
# contiguous (the shared-fit hits land back to back). fit_params is the same
# GLOW_FIT_PARAMS the RUN_ANA_LIST GLOW recipe takes: the shared fit is GLOW's
# alone, so it gets GLOW's device + worker-count knobs like every other GLOW
# leaf (driver.check_fit_params refuses to run it in parallel with a device
# visible, same as RUN_ANA_LIST's GLOW cell).
_PRUNE_GLOW_KWARGS = dict(n_perm_fwer=N_PERM_FWER,
                          alpha_fwer=ALPHA_FWER, fit_params=GLOW_FIT_PARAMS)
PRUNE_RULES = ['maxllr', 'greedy', 'dp']
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
# Wall time, not detection. Every cache here is HCP -- the paper's real data,
# and the only cohort whose subject count means anything -- with the moderate
# default effect (10% of each cell's volume, see effect_grid); only the timed
# axis varies. A recorded time_sec means the machine that recorded it, so these
# are only comparable within one run on one box. Each cache gets its own seed
# offset so its leaf timings are cold (never served from another cache's cached
# fit) and independent.
#
# Two questions, deliberately not mixed:
#
# runtime_num_vox -- how long does a user wait? Every method fit with what the
# machine has (run_ana_time; see RUN_ANA_TIME_LIST for the per-method knobs),
# so the answer includes whatever these cores and this card contribute. A
# number to quote, not to extrapolate from: the parallel speedup itself varies
# along the axis, so the slope is the machine's as much as the algorithm's.
# This is where the methods are compared, so it runs the three voxel-wise arms
# against GLOW -- the reported arm alone (REPORTED_GLOW_LABEL); the other's
# wall clock is the same curve.
#
# runtime_1perm_* -- how does GLOW's cost grow? One core, no device, BLAS
# pinned to one thread, one outer permutation (run_ana_time_1perm), everything
# but the swept axis at the shared centre. Held that still, the ratio between
# two points is the growth in that axis, which is what the paper's cost model
# claims: linear in num_vox and num_img, quadratic in b, linear in each
# permutation count. Not a formal complexity result -- a measured middle ground
# short of one. GLOW only: it is GLOW's cost model being made good on, and the
# voxel-wise arms are already compared where the comparison belongs, above.
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
                     if not isinstance(ana, AnalysisGLOW)
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
    # moderate effect, sweeping the share of the images the tree is built on
    # (SEGMENT_FRAC_GRID). It shares both grids and the leaf with segment's
    # midpoint slice, so the record walk reaches either cache's leaves from the
    # other's cells; frac_segment is what separates them (plot.tidy_segment
    # selects on it).
    'segment_perc': (
        data_grid(),
        effect_grid(),
        RUN_SEGMENT_PERC_LIST, run_segment),
    # The same fold sweep across the whole llr axis: segment's effect grid on
    # segment_perc's leaf grid, so what a smaller segmentation fold costs is
    # read at every effect strength rather than at the moderate one alone
    # (benchmark.plot draws it as one grid per Ward mode, a curve per fold
    # share). Both grids are the ones those two caches already declare, so its
    # cells are segment's cells and its moderate-llr column is segment_perc's
    # own leaves -- only the off-midpoint fold leaves are new work. The
    # whole-cohort ceiling each curve is read against is segment's llr sweep,
    # which the record walk reaches from these same cells.
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
    # The same rules read on the llr sweep's own axes: sweep_llr's data grid
    # (both b) and effect grid on prune's leaf, so what the selection rule
    # costs is read at every effect strength the power curve reports rather
    # than at b=1 alone. Its b=1 half is prune's own grid, so those cells are
    # shared -- only the b=2 half is new work.
    #
    # A rule is a re-selection off a fitted tree, so this is not the same as
    # running sweep_llr twice with two GLOW recipes: run_prune scores all
    # three rules off one glow_fit_for_prune per (cell, Ward mode), which both
    # halves the fits and isolates the rule from the permutation test that
    # chose the significant set. Pooling the two b's in one panel would
    # average two power curves, so benchmark.plot facets the prune figures by
    # b (plot_prune).
    'sweep_llr_prune': (
        data_grid(b_list=B_LLR_SWEEP),
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
