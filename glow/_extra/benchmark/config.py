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

Every cache here backs a figure, table or quantitative claim in the paper, bar
smoke (an end-to-end pipeline check). Adding one is cheap; the catalogue is
kept at what is cited.

Scope. Three caches share the run_ana leaf (fit + score one Analysis per cell):
null, sweep_llr, sweep_extent. Three swap in their own leaf over those same
grids: segment (run_segment, a Ward-mode oracle, no fit), vba_stat (run_stat,
a VBA / CET variant reading a shared voxel-stat walk), and prune (run_prune,
three pruning rules on a shared GLOW fit).

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
# num_vox matches across WGN and HCP (get_kwargs_data_list sizes the WGN box
# from it).
CROP_N_VOX = 25_000

# Effect support: 10% of the analysis volume. A per-cell fraction, so it
# tracks whatever num_vox each cell is cropped to (see get_kwargs_effect_list).
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
EXTENT_FRAC_GRID = list(np.geomspace(0.01, 1.0, 15))
NIMG_GRID = [10, 18, 30, 55, 100, 180, 300]


# ---------- analysis recipes -------------------------------------------------
# label -> recipe. The label is the reader-facing method name; it is the source
# of truth results / plot map a recorded recipe back to (a run function is not
# passed the label -- see run.py / benchmark.plot). GLOW uses the LLR
# throughout, so the two GLOW arms differ only in Ward projection; the
# voxel-wise arms z-score before the max-stat null.
kwargs = dict(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER)
ana_kwargs_dict = {
    'GLOW':   AnalysisGLOW(n_perm_inner=N_PERM_INNER,
                           cluster_mode=ClusterMode.GLM_ERROR,
                           **kwargs),
    'VBA':        AnalysisVBA(z_flag=True, tfce_flag=False,
                              get_stat=get_hotel_tr, **kwargs),
    'VBA-TFCE':   AnalysisVBA(z_flag=True, tfce_flag=True, get_stat=get_wilks,
                              **kwargs),
    'CET':        AnalysisCET(z_flag=True, get_stat=get_hotel_tr,
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
GLOW_FIT_N_JOBS = 32
GLOW_FIT_PARAMS = dict(n_jobs=GLOW_FIT_N_JOBS, gpu='auto')


def fit_params_for(ana):
    """Return the fit_params a recipe should run under, or None for defaults.

    Args:
        ana (Analysis): an analysis recipe.

    Returns:
        dict | None: GLOW_FIT_PARAMS for a GLOW recipe (the only one with a
            device backend and deep parallelism), else None -- the voxel-wise
            arms take fit's serial default and get their parallelism from the
            sweep's own n_jobs.
    """
    return GLOW_FIT_PARAMS if isinstance(ana, AnalysisGLOW) else None


# the leaf kwargs grid: one run_ana call per recipe, shared by every cache.
# Only the ana rides into the cell, plus how to run it; the method name (the
# ana_kwargs_dict key) is recovered from the recipe at read time (see
# benchmark.plot), so it never enters the call or the cache key.
RUN_ANA_LIST = [dict(ana=ana, fit_params=fit_params_for(ana))
                for ana in ana_kwargs_dict.values()]


def filter_ana_list(kwargs_fnc_list, labels) -> list:
    """Keep the fnc-kwargs cells whose recipe is one of the named methods.

    Narrows a cache's leaf grid to a subset of the analysis recipes, so a rerun
    touches only those methods. This is what makes a per-method rerun cheap:
    completeness is judged against the grid handed to the driver (see
    results.get_cell_complete), so a cell whose named-method leaves are all
    recorded is skipped, and the recipes left out are never called -- no
    already-computed fit is recomputed just because a sibling recipe changed
    (as one does whenever a recipe knob moves: a new knob is a new hash, hence
    a missing leaf).

    Cells are matched on the recipe repr (the address-free recipe id the read
    path identifies a leaf by), not identity, so a rebuilt equal recipe matches.
    A cell carrying no ana (a non-run_ana leaf grid -- segment / prune / ...)
    never matches, so filtering such a cache yields an empty grid: it has no
    per-method axis to select on.

    Args:
        kwargs_fnc_list (iterable[dict]): a leaf-fnc kwargs grid, e.g.
            RUN_ANA_LIST.
        labels (iterable[str]): ana_kwargs_dict keys (method names) to keep.

    Returns:
        list[dict]: the kept cells, in the input grid's order (empty when none
            match).

    Raises:
        ValueError: a label is not an ana_kwargs_dict key (a typo would
            otherwise silently select nothing).
    """
    labels = list(labels)
    unknown = [label for label in labels if label not in ana_kwargs_dict]
    if unknown:
        raise ValueError(f'unknown method label(s): {unknown}; '
                         f'known: {list(ana_kwargs_dict)}')
    keep = {repr(ana_kwargs_dict[label]) for label in labels}
    return [kwargs for kwargs in kwargs_fnc_list
            if 'ana' in kwargs and repr(kwargs['ana']) in keep]


def strip_gpu(kwargs_fnc_list) -> list:
    """Return the leaf grid with every fit_params gpu request removed.

    The CPU-parallel path for a machine that has a card: a device leaf and a
    parallel sweep cannot share it (driver.check_fit_params refuses the pair),
    so this is how one sweep opts out of the device without editing the
    catalogue. The fits it drops to the CPU compute the same thing (the two
    backends agree to round-off, see AnalysisGLOW.fit), so the scores and the
    records are unaffected.

    Cells are rebuilt rather than mutated: the grids are module-level
    singletons shared by every cache.

    Args:
        kwargs_fnc_list (iterable[dict]): a leaf-fnc kwargs grid.

    Returns:
        list[dict]: the same cells, each fit_params less its gpu key (dropped
            entirely when gpu was all it held).
    """
    out = []
    for kwargs in kwargs_fnc_list:
        fit_params = kwargs.get('fit_params')
        if not fit_params or 'gpu' not in fit_params:
            out.append(kwargs)
            continue
        rest = {k: v for k, v in fit_params.items() if k != 'gpu'}
        out.append({**kwargs, 'fit_params': rest or None})
    return out


# the segment cache's leaf grid: one run_segment call per Ward mode (Naive /
# GLM Error / Focus). The mode rides in as cluster_mode; the method name is
# str(mode), recovered from the record at read time.
SEGMENT_MODES = [ClusterMode.NAIVE, ClusterMode.GLM_ERROR, ClusterMode.FOCUS]
RUN_SEGMENT_LIST = [dict(cluster_mode=mode) for mode in SEGMENT_MODES]


def get_run_stat_list():
    """Build the vba_stat cache's leaf grid (one run_stat call per variant).

    The bake-off among the voxel-wise methods: VBA / VBA-TFCE / CET x 5 stats x
    {raw, z} = 30 variants. GLOW is excluded by design (it uses the LLR
    throughout), so this is VBA / CET only. Each cell pairs a recipe (its class
    / tfce_flag / z_flag identify the variant) with the stat_dict key naming the
    shared-walk matrix run_stat injects as _stat; the method name (e.g.
    VBA-TFCE-Wilks-z) is recovered from those at read time.

    Returns:
        list[dict]: kwargs for run_stat (exp / mask_target_list supplied by the
            driver), one per variant.
    """
    kwargs = dict(n_perm_fwer=N_PERM_FWER, alpha_fwer=ALPHA_FWER)
    specs = []
    for fn in stat_dict.values():
        name = stat_dict_inv[fn]
        for z_flag in (False, True):
            specs.append(dict(
                ana=AnalysisVBA(get_stat=fn, z_flag=z_flag, tfce_flag=False,
                                **kwargs),
                stat_name=name))
            specs.append(dict(
                ana=AnalysisVBA(get_stat=fn, z_flag=z_flag, tfce_flag=True,
                                **kwargs),
                stat_name=name))
            specs.append(dict(
                ana=AnalysisCET(get_stat=fn, z_flag=z_flag,
                                cft_pval=DEFAULT_CET_CFT_PVAL, **kwargs),
                stat_name=name))
    return specs


RUN_STAT_LIST = get_run_stat_list()

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


def get_kwargs_two_effect_list(*, llr_list, angle_list,
                               n_vox_frac=EFFECT_N_VOX_FRAC,
                               extenter_cls=ExtenterMinVar):
    """Build a cleaving grid: effect_factory_split kwargs over (llr, angle).

    Two adjacent equal-LLR effects planted on the spectral halves of one n_vox
    extent, their feature directions angle degrees apart. Each cell carries
    kind='split' and seed_from_exp=True, so both the support placement and the
    direction pair are derived from the experiment (see effect_factory_split).
    The angle sweep at fixed llr is the cleaving / merge-cost curve.

    No CONFIG cache declares this grid; it is the entry point for adding one
    (the split effect stage it feeds stays wired up and tested). Hence llr_list
    and angle_list are required -- the sweep's shape is the caller's to choose.
    b=3 or more gives the direction rotation a plane to turn in.

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


# ---------- runtime benchmarks (local-only) ---------------------------------
# Wall-time scaling of the methods, not detection. The effect is the moderate
# default (10% of each cell's volume, see get_kwargs_effect_list); only the
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


def get_kwargs_data_runtime(*, seed_offset, crop_n_vox_list, b_list=(1,),
                            sources=('hcp',), num_img_list=(100,)):
    """Build the data grid for a runtime cache (num_vox = the analysis crop).

    Concatenates get_kwargs_data_list over crop_n_vox_list, so one grid spans
    several analysis volumes (each an ExtenterSphere crop), with RUNTIME_N_SEED
    seeds from seed_offset -- keeping each cache's cells (and their cached leaf
    timings) distinct.

    HCP by default, the paper's real data. The num_img sweep passes
    sources=['wgn'] instead, because HCP's N is its fixed cohort: data_factory
    has no subject-subset axis, and adding one would rehash every HCP cell of
    every cache. Timing is a function of the (b, num_img, num_vox) shapes, not
    of what filled the array, so WGN measures the num_img slope faithfully.

    Args:
        seed_offset (int): first seed; the cache uses
            range(seed_offset, seed_offset + RUNTIME_N_SEED).
        crop_n_vox_list (iterable[int]): analysis-crop sizes to span (the
            num_vox axis); a single-element list for the fixed-size caches.
        b_list (iterable[int]): imaging-feature counts (HCP draws a subset).
        sources (iterable[str]): 'wgn' and/or 'hcp'.
        num_img_list (iterable[int]): subject counts (WGN only).

    Returns:
        list[dict]: kwargs for data_factory, one per cell.
    """
    seeds = range(seed_offset, seed_offset + RUNTIME_N_SEED)
    kwargs_data_list = []
    for crop_n_vox in crop_n_vox_list:
        kwargs_data_list += get_kwargs_data_list(
            sources=list(sources), seeds=seeds, b_list=b_list,
            num_img_list=num_img_list, crop_n_vox=crop_n_vox)
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
        get_kwargs_data_list(seeds=range(N_SEED_NULL)),
        get_kwargs_effect_list(llr_list=None),
        RUN_ANA_LIST, run_ana),
    # B. Detection vs effect strength at b = 1 and b = 2 (the univariate power
    #    curve and its low-b multivariate counterpart) in one sweep over
    #    (b, effect_llr); HCP draws a random b-subset per seed. b rides the
    #    data grid alongside the full effect_llr grid, so the plot holds b
    #    fixed per figure (plot.plot_cache splits on it) -- one each. The b=1
    #    slice matches the standalone anchor the other caches plant, and every
    #    b's midpoint llr coincides with the moderate-effect anchor.
    'sweep_llr': (
        get_kwargs_data_list(b_list=B_LLR_SWEEP),
        get_kwargs_effect_list(llr_list=EFFECT_LLR_GRID),
        RUN_ANA_LIST, run_ana),
    # D. Detection vs effect extent (fixed per-voxel effect_llr).
    'sweep_extent': (
        get_kwargs_data_list(),
        get_kwargs_effect_list(n_vox_frac_list=EXTENT_FRAC_GRID),
        RUN_ANA_LIST, run_ana),
    # F. Segmentation quality: oracle best-Dice region per Ward mode (Naive /
    #    GLM Error / Focus), no significance test or pruning. Same grids as
    #    the b=1 llr sweep; the leaf is run_segment over the mode grid.
    'segment': (
        get_kwargs_data_list(),
        get_kwargs_effect_list(llr_list=EFFECT_LLR_GRID),
        RUN_SEGMENT_LIST, run_segment),
    # G. MANCOVA stat comparison: VBA / VBA-TFCE / CET x 5 stats x {raw, z}
    #    (b=2 so the multivariate stats differ). The cell's variants share one
    #    voxel-stat walk (run_stat -> voxel_stat_walk). GLOW excluded.
    'vba_stat': (
        get_kwargs_data_list(b_list=[2]),
        get_kwargs_effect_list(llr_list=EFFECT_LLR_GRID),
        RUN_STAT_LIST, run_stat),
    # H. Pruning rule: greedy max-LLR vs DP max-likelihood cut vs the single
    #    max-LLR region, scored on one shared GLOW fit per (cell, Ward mode) so
    #    the comparison isolates the rule, not the permutation test. Crossed
    #    with both clustering modes (Focus / GLM Error); benchmark.plot draws
    #    one metric grid per mode.
    'prune': (
        get_kwargs_data_list(),
        get_kwargs_effect_list(llr_list=EFFECT_LLR_GRID),
        RUN_PRUNE_LIST, run_prune),
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
    # moderate effect -- the paper's num_vox-linearity figure. run_ana_time
    # fits at n_jobs=-1 (all cores) so the curve is the wall-clock a user waits
    # on an N-core machine, every method parallelised alike. HCP-only and
    # local-only: the AWS worker has no HCP data, and Spot instance-type
    # variance would make time_sec meaningless anyway.
    'runtime': (
        get_kwargs_data_runtime(
            seed_offset=RUNTIME_SEED_OFFSET['runtime'],
            crop_n_vox_list=RUNTIME_NUM_VOX_GRID),
        get_kwargs_effect_list(),
        RUN_ANA_TIME_LIST, run_ana_time),
    # Runtime (n_perm_fwer): outer-loop wall time vs n_perm_fwer at the shared
    # crop (CROP_N_VOX), GLOW only, n_perm_inner held at N_PERM_INNER.
    'runtime_n_perm_fwer': (
        get_kwargs_data_runtime(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_n_perm_fwer'],
            crop_n_vox_list=[CROP_N_VOX]),
        get_kwargs_effect_list(),
        RUN_PERM_FWER_LIST, run_perm_fwer),
    # Runtime (n_perm_inner): inner-null wall time vs n_perm_inner at the
    # shared crop (CROP_N_VOX), GLOW only (one observed tree; run_perm_inner).
    'runtime_n_perm_inner': (
        get_kwargs_data_runtime(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_n_perm_inner'],
            crop_n_vox_list=[CROP_N_VOX]),
        get_kwargs_effect_list(),
        RUN_PERM_INNER_LIST, run_perm_inner),
    # Runtime (b): fit wall time vs feature count b (1..6) at the shared crop
    # (CROP_N_VOX), GLOW only -- the paper's b-quadratic claim. b rides the
    # data grid; the leaf is run_ana_time (n_jobs=-1).
    'runtime_b': (
        get_kwargs_data_runtime(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_b'],
            crop_n_vox_list=[CROP_N_VOX], b_list=B_GRID),
        get_kwargs_effect_list(),
        GLOW_ANA_LIST, run_ana_time),
    # Runtime (num_img): fit wall time vs subject count (NIMG_GRID, 10 -> 300)
    # at a small crop, GLOW only -- the paper's N-linearity claim. WGN, alone
    # in this family: HCP's N is its fixed cohort, and giving data_factory a
    # subject-subset axis would rehash every HCP cell everywhere. Timing turns
    # on the array shapes, not on what filled them, so WGN measures the slope
    # faithfully (see get_kwargs_data_runtime).
    'runtime_nimg': (
        get_kwargs_data_runtime(
            seed_offset=RUNTIME_SEED_OFFSET['runtime_nimg'],
            crop_n_vox_list=[RUNTIME_NIMG_CROP_N_VOX], sources=['wgn'],
            num_img_list=NIMG_GRID),
        get_kwargs_effect_list(),
        GLOW_ANA_LIST, run_ana_time),
    # n_perm_inner convergence: the detection-side counterpart to
    # runtime_n_perm_inner. Holds the data + moderate effect fixed and, per GLOW
    # arm, samples the inner null once to MAX_INNER_PERM (run_inner_edge),
    # recording how the FWER max-z threshold settles as num_inner_perm grows.
    # HCP only (the paper's real data; no point double-computing the WGN half),
    # so local-only like the runtime family; shares the HCP detection cells, so
    # their data / effect builds are cache hits off the other sweeps.
    'sweep_n_perm_inner': (
        get_kwargs_data_list(sources=['hcp']),
        get_kwargs_effect_list(),
        RUN_INNER_EDGE_LIST, run_inner_edge),
}
