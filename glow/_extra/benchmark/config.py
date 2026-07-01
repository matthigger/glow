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

Scope. Most caches share the run_ana leaf (fit + score one Analysis per cell):
null, sweep_llr, sweep_b, sweep_extent, sweep_nimg. Four caches swap in their
own leaf over those same grids: segment (run_segment, a Ward-mode oracle, no
fit), min_size (run_min_size, per-perm staircases, recorded not scored), stat
(run_stat, a VBA / CET variant reading a shared voxel-stat walk), and prune
(run_prune, three pruning rules on a shared GLOW fit). two-effect reuses the
run_ana leaf unchanged -- score_effects already scores each planted half
(target0 / target1) -- over a split effect stage (effect_factory kind='split').

Runtime. A separate family measures wall time, not detection (HCP-only, so
local-only): runtime (run_ana over a num_vox sweep, 1k -> full HCP, all
methods), and four that time one piece of GLOW each -- runtime_segment
(run_segment_time, Ward clustering per mode over the same num_vox sweep),
runtime_n_perm_fwer / runtime_n_perm_inner (run_perm_fwer / run_perm_inner,
GLOW's outer / inner perms at 1k voxels), and runtime_b (run_ana over the b
sweep). See the runtime section below.
"""
import itertools
import math
import warnings

import numpy as np

from glow.analysis import (AnalysisCET, AnalysisGLOW, AnalysisVBA,
                           DEFAULT_CET_CFT_PVAL)
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import (get_hotel_tr, get_wilks, stat_dict,
                                   stat_dict_inv)
from glow.effect import ExtenterMinVar, ExtenterSphere

from . import hcp
from .run import (run_ana, run_min_size, run_perm_fwer, run_perm_inner,
                  run_prune, run_segment, run_segment_time, run_stat)


# ---------- shared knobs ------------------------------------------------------
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

# Effect support: 10% of the analysis volume. A per-cell fraction, so it
# tracks whatever num_vox each cell is cropped to (see get_kwargs_effect_list).
EFFECT_N_VOX_FRAC = 0.1

N_PERM_FWER = 250
N_PERM_INNER = 1000
ALPHA_FWER = 0.05

# Structural grids. B caps at the HCP pool (6) so every HCP cell is feasible;
# the extent grid spans 1%..100% of the volume; the subject grid is WGN-only.
B_GRID = list(range(1, len(hcp.HCP_FEATS) + 1))
EXTENT_FRAC_GRID = list(np.geomspace(0.01, 1.0, 15))
NIMG_GRID = [10, 18, 30, 55, 100, 180, 300]

# Two-effect (cleaving) grids. The angle between the two effects' feature
# directions sweeps 0..90 deg in 10 steps; the per-voxel llr spans weaker SNRs
# (at 25k a ~1250-vox half is very high-SNR at the moderate llr, where GLOW
# favours the merged region), the right level read off the resulting ARI /
# dice curves. b=3 so the direction rotation has a plane to turn in.
B_TWO_EFFECT = 3
ANGLE_GRID = [float(a) for a in np.linspace(0, 90, 10)]
TWO_EFFECT_LLR_GRID = [0.003, 0.01, 0.03]


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

# the segment cache's leaf grid: one run_segment call per Ward mode (Naive /
# GLM Error / Focus). The mode rides as both the recorded label (its name) and
# the cluster_mode the leaf segments with.
SEGMENT_MODES = [ClusterMode.NAIVE, ClusterMode.GLM_ERROR, ClusterMode.FOCUS]
RUN_SEGMENT_LIST = [dict(cluster_mode=mode, label=str(mode))
                    for mode in SEGMENT_MODES]

# the min_size cache's leaf grid: one run_min_size call capturing GLOW's
# per-perm (size -> max-z) staircases (its one method, labelled GLOW), swept
# over min_vox post hoc from the recorded curves. Its trial seeds are offset
# clear of the other sweeps (MIN_SIZE_SEED_OFFSET), each its own HCP null.
MIN_SIZE_SEED_OFFSET = 100_000
RUN_MIN_SIZE_LIST = [dict(n_perm_fwer=N_PERM_FWER, n_perm_inner=N_PERM_INNER,
                          label='GLOW')]


def get_run_stat_list():
    """Build the stat cache's leaf grid (one run_stat call per stat variant).

    The bake-off among the voxel-wise methods: VBA / VBA-TFCE / CET x 5 stats x
    {raw, z} = 30 variants. GLOW is excluded by design (it uses the LLR
    throughout), so this is VBA / CET only. Each cell pairs a recipe with the
    stat_dict key naming the shared-walk matrix run_stat injects as _stat, and
    a record-only label (e.g. VBA-TFCE-Wilks-z).

    Returns:
        list[dict]: kwargs for run_stat (exp / mask_target_list supplied by the
            driver), one per variant.
    """
    kwargs = dict(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER)
    specs = []
    for fn in stat_dict.values():
        name = stat_dict_inv[fn]
        for z_flag in (False, True):
            suffix = '-z' if z_flag else ''
            specs.append(dict(
                ana=AnalysisVBA(get_stat=fn, z_flag=z_flag, tfce_flag=False,
                                **kwargs),
                stat_name=name, label=f'VBA-{name}{suffix}'))
            specs.append(dict(
                ana=AnalysisVBA(get_stat=fn, z_flag=z_flag, tfce_flag=True,
                                **kwargs),
                stat_name=name, label=f'VBA-TFCE-{name}{suffix}'))
            specs.append(dict(
                ana=AnalysisCET(get_stat=fn, z_flag=z_flag,
                                cft_pval=DEFAULT_CET_CFT_PVAL, **kwargs),
                stat_name=name, label=f'CET-{name}{suffix}'))
    return specs


RUN_STAT_LIST = get_run_stat_list()

# the prune cache's leaf grid: three rules scored on one shared GLOW fit per
# cell (greedy / DP / the single max-LLR region). All carry the same GLOW fit
# knobs (so run_prune's glow_fit_for_prune is shared across them); the rule
# rides as both the cache axis and the recorded label.
_PRUNE_GLOW_KWARGS = dict(n_perm_fwer=N_PERM_FWER, n_perm_inner=N_PERM_INNER,
                          alpha_fwer=ALPHA_FWER)
RUN_PRUNE_LIST = [
    dict(rule='maxllr', label='GLOW-MaxLLR', **_PRUNE_GLOW_KWARGS),
    dict(rule='greedy', label='GLOW-Greedy', **_PRUNE_GLOW_KWARGS),
    dict(rule='dp', label='GLOW-DP', **_PRUNE_GLOW_KWARGS),
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


# ---------- stage builders (swept axes are the keyword arguments) ------------
def get_kwargs_data_list(*, sources=SOURCES, seeds=range(N_SEED), b_list=(1,),
                         num_img_list=(100,), crop_n_vox=CROP_N_VOX):
    """Return the list of data_factory kwargs dicts over the swept axes.

    The cartesian product of (source, b, seed); WGN additionally sweeps num_img
    (HCP's N is its cohort, so its cells omit it and never duplicate). The
    analysis crop -- a connected crop_n_vox sphere seeded by the cell's seed --
    is built once per (source, b, seed) and shared across a WGN cell's num_img
    values; the HCP feature subset is a sorted random b-subset of the pool.

    Args:
        sources (list[str]): 'wgn' and/or 'hcp'.
        seeds (iterable[int]): per-cell realization seeds.
        b_list (iterable[int]): imaging-feature counts (HCP draws a subset).
        num_img_list (iterable[int]): subject counts (WGN only).
        crop_n_vox (int): voxels in the analysis-crop sphere (also sizes the
            WGN box). Defaults to the paper CROP_N_VOX; the smoke cache passes
            a small value for a fast end-to-end check.

    Returns:
        list[dict]: kwargs for data_factory, one per cell (source selects
            wgn / hcp).
    """
    wgn_side = math.ceil(crop_n_vox ** (1 / 3))

    def sample_hcp_feats(b, seed):
        """Return a sorted random b-subset of HCP_FEATS, per seed."""
        idx = np.random.default_rng(seed).choice(len(hcp.HCP_FEATS), size=b,
                                                 replace=False)
        return tuple(sorted(hcp.HCP_FEATS[i] for i in idx))

    kwargs_data_list = []
    for source, b, seed in itertools.product(sources, b_list, seeds):
        extenter = ExtenterSphere(n_vox=crop_n_vox, connected=True,
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
                           n_vox_frac_list=(EFFECT_N_VOX_FRAC,)):
    """Return the list of effect_factory kwargs dicts over the swept axes.

    The cartesian product of (effect_llr, n_vox_frac): a per-voxel strength and
    a support size as a fraction of each cell's analysis volume. Each cell
    carries kind='single' and the ingredients effect_factory_single builds the
    support from -- the ExtenterMinVar class, n_vox_frac, and
    seed_from_exp=True so the placement is derived from the experiment (see the
    module docstring).
    llr_list=None is the null / FWER-calibration path -- the list [None] (plant
    nothing).

    Args:
        llr_list (iterable[float] | None): per-voxel effect strengths; None is
            the null path.
        n_vox_frac_list (iterable[float]): effect support sizes, each a
            fraction of the analysis volume.

    Returns:
        list[dict | None]: kwargs for effect_factory (exp is supplied by the
            driver), or [None] for the null path.
    """
    if llr_list is None:
        return [None]
    kwargs_effect_list = []
    for llr, frac in itertools.product(llr_list, n_vox_frac_list):
        kwargs_effect_list.append(dict(
            kind='single', effect_llr=float(llr), extenter_cls=ExtenterMinVar,
            n_vox_frac=float(frac), seed_from_exp=True))
    return kwargs_effect_list


def get_kwargs_two_effect_list(*, llr_list=TWO_EFFECT_LLR_GRID,
                               angle_list=ANGLE_GRID,
                               n_vox_frac=EFFECT_N_VOX_FRAC,
                               extenter_cls=ExtenterMinVar):
    """Build the cleaving grid: effect_factory_split kwargs over (llr, angle).

    Two adjacent equal-LLR effects planted on the spectral halves of one n_vox
    extent, their feature directions angle degrees apart. Each cell carries
    kind='split' and seed_from_exp=True, so both the support placement and the
    direction pair are derived from the experiment (see effect_factory_split).
    The angle sweep at fixed llr is the cleaving / merge-cost curve.

    extenter_cls is the split base: ExtenterMinVar (the default) grows the
    lowest-variance region from its own seeded start -- the same data-driven
    support the single-effect caches use -- then bisects it into roughly equal
    halves. Pass ExtenterSphere for a geometric base.

    Args:
        llr_list (iterable[float]): per-voxel strengths (per effect).
        angle_list (iterable[float]): direction angles between the two effects
            (degrees).
        n_vox_frac (float): combined two-effect support as a fraction of the
            analysis volume (split into halves).
        extenter_cls (type[Extenter]): the split base extenter.

    Returns:
        list[dict]: kwargs for effect_factory (kind='split'), one per
            (llr, angle) cell.
    """
    return [dict(kind='split', effect_llr=float(llr),
                 extenter_cls=extenter_cls, n_vox_frac=float(n_vox_frac),
                 angle=float(angle), seed_from_exp=True)
            for llr, angle in itertools.product(llr_list, angle_list)]


# ---------- runtime benchmarks (HCP-only, local-only) -----------------------
# Wall-time scaling of the methods, not detection. The effect is the moderate
# default (10% of each cell's volume, see get_kwargs_effect_list); only the
# timed axis varies. HCP-only, so these run locally -- the AWS worker has no
# HCP data (see hcp / the aws package). Each cache gets its own seed offset so
# its leaf timings are cold (never served from another cache's cached fit) and
# independent. The timed leaves are run.run_perm_fwer / run_perm_inner /
# run_segment_time; runtime and runtime_b reuse run_ana (its score carries
# num_vox, and time_sec is the fit wall time).
RUNTIME_N_SEED = 3
RUNTIME_CROP_N_VOX = 1_000

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

# per-cache seed offsets, clear of each other and of MIN_SIZE_SEED_OFFSET, so
# no two runtime caches share a data cell (hence a cached leaf timing).
RUNTIME_SEED_OFFSET = {
    'runtime': 200_000,
    'runtime_segment': 210_000,
    'runtime_n_perm_fwer': 220_000,
    'runtime_n_perm_inner': 230_000,
    'runtime_b': 240_000,
}


def get_kwargs_data_runtime(*, seed_offset, crop_n_vox_list, b_list=(1,)):
    """Build the HCP data grid for a runtime cache (num_vox = crop, HCP only).

    Concatenates get_kwargs_data_list over crop_n_vox_list, so one grid spans
    several analysis volumes (each an ExtenterSphere crop of the HCP brain).
    HCP only (the runtime caches are local-only) and RUNTIME_N_SEED seeds from
    seed_offset, keeping each cache's cells (and their cached leaf timings)
    distinct.

    Args:
        seed_offset (int): first seed; the cache uses
            range(seed_offset, seed_offset + RUNTIME_N_SEED).
        crop_n_vox_list (iterable[int]): analysis-crop sizes to span (the
            num_vox axis); a single-element list for the fixed-size caches.
        b_list (iterable[int]): imaging-feature counts (HCP draws a subset).

    Returns:
        list[dict]: kwargs for data_factory (source='hcp'), one per cell.
    """
    seeds = range(seed_offset, seed_offset + RUNTIME_N_SEED)
    kwargs_data_list = []
    for crop_n_vox in crop_n_vox_list:
        kwargs_data_list += get_kwargs_data_list(
            sources=['hcp'], seeds=seeds, b_list=b_list, crop_n_vox=crop_n_vox)
    return kwargs_data_list


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

# segmentation timing: one run_segment_time per Ward mode (the segment cache's
# modes), the mode riding as both cluster_mode and label.
RUN_SEGMENT_TIME_LIST = [dict(cluster_mode=mode, label=str(mode))
                         for mode in SEGMENT_MODES]

# the two GLOW arms of RUN_ANA_LIST (runtime / runtime_b are GLOW only)
GLOW_ANA_LIST = [kw for kw in RUN_ANA_LIST if kw['label'].startswith('GLOW')]


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
        get_kwargs_effect_list(n_vox_frac_list=EXTENT_FRAC_GRID),
        RUN_ANA_LIST, run_ana),
    # E. Detection vs subject count (WGN only; HCP's N is its cohort).
    'sweep_nimg': (
        get_kwargs_data_list(sources=['wgn'], num_img_list=NIMG_GRID),
        get_kwargs_effect_list(),
        RUN_ANA_LIST, run_ana),
    # F. Segmentation quality: oracle best-Dice region per Ward mode (Naive /
    #    GLM Error / Focus), no significance test or pruning. Same grids as
    #    sweep_llr; the leaf is run_segment over the mode grid.
    'segment': (
        get_kwargs_data_list(),
        get_kwargs_effect_list(llr_list=EFFECT_LLR_GRID),
        RUN_SEGMENT_LIST, run_segment),
    # J. Min-size sweep: per-perm (size -> max-z) staircases on HCP, mirroring
    #    sweep_llr's effect grid, so min_vox sweeps post hoc from one run.
    #    Seeds are offset clear of the other sweeps; HCP only.
    'min_size': (
        get_kwargs_data_list(
            sources=['hcp'],
            seeds=range(MIN_SIZE_SEED_OFFSET, MIN_SIZE_SEED_OFFSET + N_SEED)),
        get_kwargs_effect_list(llr_list=EFFECT_LLR_GRID),
        RUN_MIN_SIZE_LIST, run_min_size),
    # G. MANCOVA stat comparison: VBA / VBA-TFCE / CET x 5 stats x {raw, z}
    #    (b=2 so the multivariate stats differ). The cell's variants share one
    #    voxel-stat walk (run_stat -> voxel_stat_walk). GLOW excluded.
    'stat': (
        get_kwargs_data_list(b_list=[2]),
        get_kwargs_effect_list(llr_list=EFFECT_LLR_GRID),
        RUN_STAT_LIST, run_stat),
    # H. Pruning rule: greedy max-LLR vs DP max-likelihood cut vs the single
    #    max-LLR region, scored on one shared GLOW-Focus fit per cell (so the
    #    comparison isolates the rule, not the permutation test).
    'prune': (
        get_kwargs_data_list(),
        get_kwargs_effect_list(llr_list=EFFECT_LLR_GRID),
        RUN_PRUNE_LIST, run_prune),
    # I. Cleaving: two adjacent equal-LLR effects; sweep the angle between
    #    their feature directions (0..90 deg). The leaf is run_ana unchanged --
    #    score_effects already scores the prediction against each planted half
    #    (target0 / target1); only the effect stage differs (kind='split').
    'two-effect': (
        get_kwargs_data_list(b_list=[B_TWO_EFFECT]),
        get_kwargs_two_effect_list(),
        RUN_ANA_LIST, run_ana),
    # Smoke: tiny null sweep over both sources to confirm the pipeline end to
    # end (not a paper figure). The null path with only a small crop and few
    # seeds overridden (see SMOKE_* above). WGN and HCP cells (3 each); on AWS
    # the HCP cells need the reference data staged to S3 first
    # (python -m glow._extra.aws stage_hcp).
    'smoke': (
        get_kwargs_data_list(seeds=range(SMOKE_N_SEED),
                             crop_n_vox=SMOKE_CROP_N_VOX),
        get_kwargs_effect_list(llr_list=None),
        RUN_ANA_LIST, run_ana),
    # Runtime: wall time vs num_vox (1k -> full HCP), all methods, b=1, the
    # moderate effect. HCP-only / local-only (see the runtime section above).
    'runtime': (
        get_kwargs_data_runtime(
            seed_offset=RUNTIME_SEED_OFFSET['runtime'],
            crop_n_vox_list=RUNTIME_NUM_VOX_GRID),
        get_kwargs_effect_list(),
        RUN_ANA_LIST, run_ana),
    # Runtime (segmentation): Ward-clustering wall time vs num_vox per mode
    # (Naive / GLM Error / Focus); run_segment_time times cluster only.
    'runtime_segment': (
        get_kwargs_data_runtime(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_segment'],
            crop_n_vox_list=RUNTIME_NUM_VOX_GRID),
        get_kwargs_effect_list(),
        RUN_SEGMENT_TIME_LIST, run_segment_time),
    # Runtime (n_perm_fwer): outer-loop wall time vs n_perm_fwer at 1k voxels,
    # GLOW only, n_perm_inner held at N_PERM_INNER.
    'runtime_n_perm_fwer': (
        get_kwargs_data_runtime(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_n_perm_fwer'],
            crop_n_vox_list=[RUNTIME_CROP_N_VOX]),
        get_kwargs_effect_list(),
        RUN_PERM_FWER_LIST, run_perm_fwer),
    # Runtime (n_perm_inner): inner-null wall time vs n_perm_inner at 1k
    # voxels, GLOW only (one observed tree; run_perm_inner).
    'runtime_n_perm_inner': (
        get_kwargs_data_runtime(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_n_perm_inner'],
            crop_n_vox_list=[RUNTIME_CROP_N_VOX]),
        get_kwargs_effect_list(),
        RUN_PERM_INNER_LIST, run_perm_inner),
    # Runtime (b): fit wall time vs feature count b (1..6) at 1k voxels, GLOW
    # only. b rides the data grid; the leaf is run_ana.
    'runtime_b': (
        get_kwargs_data_runtime(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_b'],
            crop_n_vox_list=[RUNTIME_CROP_N_VOX], b_list=B_GRID),
        get_kwargs_effect_list(),
        GLOW_ANA_LIST, run_ana),
}
