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
on the recorded source column downstream. HCP has no num_img axis (its N is
the cohort), so the num_img sweep is WGN-only.

Every cache here backs a figure, table or quantitative claim in the paper, bar
smoke (an end-to-end pipeline check). Adding one is cheap; the catalogue is
kept at what is cited.

Scope. Five caches share the run_ana leaf (fit + score one Analysis per cell):
null, sweep_llr, sweep_extent, sweep_b, sweep_nimg. Three swap in their own
leaf over much the same grids: segment (run_segment, a Ward-mode oracle, no
fit), vba_stat (run_stat, a VBA / CET variant reading a shared voxel-stat
walk; HCP only, b=2), and prune (run_prune, three pruning rules on a shared
GLOW fit).

Runtime. A separate family measures wall time, not detection, and runs locally
only: runtime (run_ana_time over a num_vox sweep, 1k -> full HCP, all methods),
and three that time GLOW alone -- runtime_n_perm_fwer / runtime_n_perm_inner
(run_perm_fwer / run_perm_inner, GLOW's outer / inner perms at the shared crop)
and runtime_b / runtime_nimg (run_ana_time over the b and num_img sweeps).
run_ana_time fits every method at n_jobs=-1, so the curves are the wall-clock a
user waits on an N-core machine. See the runtime section below.

Convergence. sweep_n_perm_inner is the detection-side counterpart to
runtime_n_perm_inner: run_inner_edge samples each outer perm's inner null once
to MAX_INNER_PERM and records how the FWER max-z threshold converges as
num_inner_perm grows (HCP only, moderate effect). The recommended n_perm_inner
is read off where that threshold plateaus (benchmark.plot).

"""
import warnings

import numpy as np

from glow.analysis import (AnalysisCET, AnalysisGLOW, AnalysisVBA,
                           DEFAULT_CET_CFT_PVAL)
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import get_hotel_tr, get_wilks

from . import grid, hcp
from .run import (run_ana, run_ana_time, run_inner_edge, run_perm_fwer,
                  run_perm_inner, run_prune, run_segment, run_stat)


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
N_PERM_INNER = 250
ALPHA_FWER = 0.05

# Structural grids. B caps at the HCP pool (6) so every HCP cell is feasible;
# the extent grid spans 1%..100% of the volume; the subject grid is WGN-only
# (HCP's N is its cohort), and times the fit rather than scoring it -- it is
# the runtime_nimg axis (see the runtime section).
B_GRID = list(range(1, len(hcp.HCP_FEATS) + 1))
# the llr sweep's feature-count axis: b = 1 (the univariate power curve) plus
# its low-b multivariate counterpart. A subset of B_GRID, swept alongside
# effect_llr in one cache (see the sweep_llr entry).
B_LLR_SWEEP = (1, 2)
# sweep_b's own axis: detection vs b at the fixed moderate effect, llr held
# still so a wider b range costs no extra effect_llr cells.
SWEEP_B_GRID = (1, 2, 3, 4)
EXTENT_FRAC_GRID = list(np.geomspace(0.01, 1.0, 15))
NIMG_GRID = [10, 18, 30, 55, 100, 180, 300]
# sweep_nimg's own axis: detection vs subject count, 10 -> 100 linear (unlike
# NIMG_GRID's wider log spacing for the runtime timing curve).
SWEEP_NIMG_GRID = list(range(10, 101, 10))


# ---------- analysis recipes -------------------------------------------------
# label -> recipe. The label is the reader-facing method name; it is the source
# of truth results / plot map a recorded recipe back to (a run function is not
# passed the label -- see run.py / benchmark.plot). GLOW uses the LLR
# throughout, so the two GLOW arms differ only in Ward projection. Each
# voxel-wise arm takes the stat / z-scoring it wins the vba_stat bake-off with:
# the raw Hotelling-Lawley trace for VBA and CET, the z-scored 1 - Wilks for
# TFCE (z-scoring is what TFCE's single height grid needs to mean the same
# thing at every voxel). Written out rather than left to the recipe defaults,
# which agree -- the paper's arms should be readable here.
kwargs = dict(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER)
ana_kwargs_dict = {
    'GLOW':   AnalysisGLOW(n_perm_inner=N_PERM_INNER,
                           cluster_mode=ClusterMode.GLM_ERROR,
                           **kwargs),
    'VBA':        AnalysisVBA(z_flag=False, tfce_flag=False,
                              get_stat=get_hotel_tr, **kwargs),
    'VBA-TFCE':   AnalysisVBA(z_flag=True, tfce_flag=True, get_stat=get_wilks,
                              **kwargs),
    'CET':        AnalysisCET(z_flag=False, get_stat=get_hotel_tr,
                              **kwargs),
}

# ---------- how a leaf's fit runs (never what it computes) -------------------
# fit_params is forwarded to Analysis.fit by the leaf (run.run_ana) and is
# filtered out of the cache key and the record (run.FIT_IGNORE), so it may vary
# by machine without forking an artifact: a cell fit here on the GPU and the
# same cell fit on an AWS Batch worker's cores are one recorded score.
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
# contiguous (the shared-fit hits land back to back).
_PRUNE_GLOW_KWARGS = dict(n_perm_fwer=N_PERM_FWER, n_perm_inner=N_PERM_INNER,
                          alpha_fwer=ALPHA_FWER)
PRUNE_RULES = ['maxllr', 'greedy', 'dp']
PRUNE_CLUSTER_MODES = [ClusterMode.FOCUS, ClusterMode.GLM_ERROR]
RUN_PRUNE_LIST = [
    dict(rule=rule, cluster_mode=mode, **_PRUNE_GLOW_KWARGS)
    for mode in PRUNE_CLUSTER_MODES
    for rule in PRUNE_RULES
]


# ---------- smoke test (a tiny non-paper cache for end-to-end checks) --------
# Not a paper figure: a tiny null sweep over both sources, to check the whole
# pipeline runs end to end (in particular the AWS Batch path -- driver submits,
# a worker rebuilds its cell from this CONFIG, fits, and ships records back).
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


# ---------- runtime benchmarks (local-only) ---------------------------------
# Wall-time scaling of the methods, not detection. The effect is the moderate
# default (10% of each cell's volume, see effect_grid); only the
# timed axis varies. Local-only: a Batch array lands on whatever Spot instance
# type is free, so a time recorded there is hardware variance rather than cost
# (the CLI warns -- see benchmark.__main__.confirm_aws), and the HCP caches
# have no worker-side data anyway (see hcp / the aws package). Each cache gets
# its own seed offset so its leaf timings are cold (never served from another
# cache's cached fit) and independent. The timed leaves are run.run_perm_fwer /
# run_perm_inner; runtime / runtime_b / runtime_nimg use run_ana_time (fit at
# n_jobs=-1, time_sec the fit wall time, num_vox returned bare -- no scoring).
#
# Between them these caches cover the cost model the paper claims: linear in
# num_vox (runtime), linear in num_img (runtime_nimg) and quadratic in b
# (runtime_b), plus the two permutation counts.
RUNTIME_N_SEED = 3

# num_vox sweep: 1k -> the full HCP support (224,619 voxels, one connected
# component), roughly doubling.
RUNTIME_NUM_VOX_GRID = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000,
                        128_000, 224_619]

# the two GLOW arms, as (label, cluster_mode)
RUNTIME_GLOW_MODES = [('GLOW-Focus', ClusterMode.FOCUS),
                      ('GLOW-GLM', ClusterMode.GLM_ERROR)]

# permutation-count sweeps (tiny num_vox, GLOW only): one axis varies, the
# other holds at its paper value (N_PERM_INNER / N_PERM_FWER).
RUNTIME_N_PERM_FWER_GRID = [50, 100, 200, 400, 800]
RUNTIME_N_PERM_INNER_GRID = [250, 500, 1_000, 2_000, 4_000]

# The num_img sweep's crop. Small (not CROP_N_VOX): the claim is a slope in
# num_img, so the cheapest volume that still exercises the whole fit will do,
# and NIMG_GRID's 30x span is where the signal is.
RUNTIME_NIMG_CROP_N_VOX = 4_000

# per-cache seed offsets, clear of each other, so no two runtime caches share a
# data cell (hence a cached leaf timing).
RUNTIME_SEED_OFFSET = {
    'runtime': 200_000,
    'runtime_n_perm_fwer': 220_000,
    'runtime_n_perm_inner': 230_000,
    'runtime_b': 240_000,
    'runtime_nimg': 250_000,
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


# GLOW-only leaf grids for the perm sweeps: one run per (GLOW arm, count). The
# swept count rides as an explicit run_perm_* arg (recorded as an in.<count>
# column), the arm as the recorded label + cluster_mode.
RUN_PERM_FWER_LIST = [
    dict(n_perm_fwer=n, n_perm_inner=N_PERM_INNER, cluster_mode=mode,
         label=label)
    for label, mode in RUNTIME_GLOW_MODES
    for n in RUNTIME_N_PERM_FWER_GRID]
RUN_PERM_INNER_LIST = [
    dict(n_perm_inner=n, cluster_mode=mode, label=label)
    for label, mode in RUNTIME_GLOW_MODES
    for n in RUNTIME_N_PERM_INNER_GRID]

# GLOW-only leaf grid for the inner-perm edge (num_inner_perm convergence)
# cache: one run_inner_edge per GLOW arm, each sampling the inner null to
# MAX_INNER_PERM and reporting the max-z edge at every num_inner_perm <= it. No
# label is passed -- the arm is recovered from the recorded cluster_mode.
MAX_INNER_PERM = 2_000
RUN_INNER_EDGE_LIST = [
    dict(cluster_mode=mode, max_inner_perm=MAX_INNER_PERM,
         n_perm_fwer=N_PERM_FWER)
    for label, mode in RUNTIME_GLOW_MODES]

# The runtime family's leaf grids carry no fit_params, so every method is
# timed the way run_ana_time defaults: all cores, CPU. That is the honest
# comparison the figure claims -- timing GLOW on a device against VBA on the
# cores would compare hardware, not algorithms -- and it is also what the cache
# requires, since fit_params does not key a leaf (run.FIT_IGNORE): a GPU timing
# would be served from the CPU timing's entry rather than measured.
RUN_ANA_TIME_LIST = [dict(ana=ana) for ana in ana_kwargs_dict.values()]

# the two GLOW arms of it (the b / num_img runtime caches are GLOW only);
# selected by the ana_kwargs_dict key (the method name is not on the cell)
GLOW_ANA_LIST = [dict(ana=ana) for label, ana in ana_kwargs_dict.items()
                 if label.startswith('GLOW')]


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
    # Detection vs subject count (fixed moderate effect). WGN only -- HCP's N
    # is its fixed cohort (see runtime_data_grid's num_img note).
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
    # seeds overridden (see SMOKE_* above). WGN and HCP cells (3 each); on AWS
    # the HCP cells need the reference data staged to S3 first
    # (python -m glow._extra.aws stage_hcp).
    'smoke': (
        data_grid(seeds=range(SMOKE_N_SEED),
                             crop_n_vox=SMOKE_CROP_N_VOX),
        effect_grid(llr_list=None),
        RUN_ANA_LIST, run_ana),
    # Runtime: wall time vs num_vox (1k -> full HCP), all methods, b=1, the
    # moderate effect -- the paper's num_vox-linearity figure. run_ana_time
    # fits at n_jobs=-1 (all cores) so the curve is the wall-clock a user waits
    # on an N-core machine, every method parallelised alike. HCP-only and
    # local-only: the AWS worker has no HCP data, and Spot instance-type
    # variance would make time_sec meaningless anyway.
    'runtime': (
        runtime_data_grid(
            seed_offset=RUNTIME_SEED_OFFSET['runtime'],
            crop_n_vox_list=RUNTIME_NUM_VOX_GRID),
        effect_grid(),
        RUN_ANA_TIME_LIST, run_ana_time),
    # Runtime (n_perm_fwer): outer-loop wall time vs n_perm_fwer at the shared
    # crop (CROP_N_VOX), GLOW only, n_perm_inner held at N_PERM_INNER.
    'runtime_n_perm_fwer': (
        runtime_data_grid(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_n_perm_fwer'],
            crop_n_vox_list=[CROP_N_VOX]),
        effect_grid(),
        RUN_PERM_FWER_LIST, run_perm_fwer),
    # Runtime (n_perm_inner): inner-null wall time vs n_perm_inner at the
    # shared crop (CROP_N_VOX), GLOW only (one observed tree; run_perm_inner).
    'runtime_n_perm_inner': (
        runtime_data_grid(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_n_perm_inner'],
            crop_n_vox_list=[CROP_N_VOX]),
        effect_grid(),
        RUN_PERM_INNER_LIST, run_perm_inner),
    # Runtime (b): fit wall time vs feature count b (1..6) at the shared crop
    # (CROP_N_VOX), GLOW only -- the paper's b-quadratic claim. b rides the
    # data grid; the leaf is run_ana_time (n_jobs=-1).
    'runtime_b': (
        runtime_data_grid(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_b'],
            crop_n_vox_list=[CROP_N_VOX], b_list=B_GRID),
        effect_grid(),
        GLOW_ANA_LIST, run_ana_time),
    # Runtime (num_img): fit wall time vs subject count (NIMG_GRID, 10 -> 300)
    # at a small crop, GLOW only -- the paper's N-linearity claim. WGN, alone
    # in this family: HCP's N is its fixed cohort, and giving data_factory a
    # subject-subset axis would rehash every HCP cell everywhere. Timing turns
    # on the array shapes, not on what filled them, so WGN measures the slope
    # faithfully (see grid.get_kwargs_data_runtime).
    'runtime_nimg': (
        runtime_data_grid(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_nimg'],
            crop_n_vox_list=[RUNTIME_NIMG_CROP_N_VOX], sources=['wgn'],
            num_img_list=NIMG_GRID),
        effect_grid(),
        GLOW_ANA_LIST, run_ana_time),
    # n_perm_inner convergence: the detection-side counterpart to
    # runtime_n_perm_inner. Holds the data + moderate effect fixed and, per GLOW
    # arm, samples the inner null once to MAX_INNER_PERM (run_inner_edge),
    # recording how the FWER max-z threshold settles as num_inner_perm grows.
    # HCP only (the paper's real data; no point double-computing the WGN half),
    # so local-only like the runtime family; shares the HCP detection cells, so
    # their data / effect builds are cache hits off the other sweeps.
    'sweep_n_perm_inner': (
        data_grid(sources=['hcp']),
        effect_grid(),
        RUN_INNER_EDGE_LIST, run_inner_edge),
}
