"""Builders for the benchmark catalogue's kwargs grids.

A cache is a cartesian product of cells, and each cell is a kwargs dict for one
pipeline stage. These builders make those lists; config declares which
axes each cache sweeps and holds the paper's values for them (see
config.CONFIG).

Nothing here knows a paper constant. Every axis arrives as an argument, so a
builder can be driven with whatever grid a caller wants -- which is also what
makes them testable without pinning the catalogue's current numbers. The
symmetric rule is that config holds no logic: tuning a knob there cannot change
behaviour tested here.

Three kinds of builder:

  data / effect / two-effect  the upstream grids, one cell per realization
  runtime data                a wall-time cache's data grid (a crop sweep)
  ana-grid utilities          build and narrow the leaf-fnc grid (the analysis
                              recipes each cell is run under)
"""
import itertools
import math

import numpy as np

from glow.analysis import AnalysisCET, AnalysisGLOWBase, AnalysisVBA
from glow.analysis.mancova import stat_dict, stat_dict_inv
from glow.effect import ExtenterMinVar, ExtenterSphere

from . import hcp


# ---------- upstream grids ---------------------------------------------------
def get_kwargs_data_list(*, sources, seeds, crop_n_vox, b_list=(1,),
                         num_img_list=(100,)):
    """Return the list of data_factory kwargs dicts over the swept axes.

    The cartesian product of (source, b, seed); WGN additionally sweeps num_img
    (HCP's N is its cohort, so its cells omit it and never duplicate). The
    analysis crop -- a connected crop_n_vox sphere seeded by the cell's seed --
    is built once per (source, b, seed) and shared across a WGN cell's num_img
    values; the HCP feature subset is a sorted random b-subset of the pool.

    Args:
        sources (iterable[str]): 'wgn' and/or 'hcp'.
        seeds (iterable[int]): per-cell realization seeds.
        crop_n_vox (int): voxels in the analysis-crop sphere (also sizes the
            WGN box).
        b_list (iterable[int]): imaging-feature counts (HCP draws a subset).
        num_img_list (iterable[int]): subject counts (WGN only).

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


def get_kwargs_effect_list(*, llr_list, n_vox_frac_list):
    """Return the list of effect_factory kwargs dicts over the swept axes.

    The cartesian product of (effect_llr, n_vox_frac): a per-voxel strength and
    a support size as a fraction of each cell's analysis volume. Each cell
    carries kind='single' and the ingredients effect_factory_single builds the
    support from -- the ExtenterMinVar class, n_vox_frac, and
    seed_from_exp=True so the placement is derived from the experiment (see
    config's module docstring).
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


def get_kwargs_two_effect_list(*, llr_list, angle_list, n_vox_frac,
                               extenter_cls=ExtenterMinVar):
    """Build a cleaving grid: effect_factory_split kwargs over (llr, angle).

    Two adjacent equal-LLR effects planted on the spectral halves of one n_vox
    extent, their feature directions angle degrees apart. Each cell carries
    kind='split' and seed_from_exp=True, so both the support placement and the
    direction pair are derived from the experiment (see effect_factory_split).
    The angle sweep at fixed llr is the cleaving / merge-cost curve.

    No CONFIG cache declares this grid; it is the entry point for adding one
    (the split effect stage it feeds stays wired up and tested). b=3 or more
    gives the direction rotation a plane to turn in.

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
        extenter_cls (type): the split base extenter class.

    Returns:
        list[dict]: kwargs for effect_factory (kind='split'), one per
            (llr, angle) cell.
    """
    return [dict(kind='split', effect_llr=float(llr),
                 extenter_cls=extenter_cls, n_vox_frac=float(n_vox_frac),
                 angle=float(angle), seed_from_exp=True)
            for llr, angle in itertools.product(llr_list, angle_list)]


def get_kwargs_data_runtime(*, seed_offset, crop_n_vox_list, n_seed,
                            b_list=(1,), sources=('hcp',),
                            num_img_list=(100,)):
    """Build the data grid for a runtime cache (num_vox = the analysis crop).

    Concatenates get_kwargs_data_list over crop_n_vox_list, so one grid spans
    several analysis volumes (each an ExtenterSphere crop), with n_seed seeds
    from seed_offset -- keeping each cache's cells (and their cached leaf
    timings) distinct.

    HCP by default, the paper's real data. The num_img sweep passes
    sources=['wgn'] instead, because HCP's N is its fixed cohort: data_factory
    has no subject-subset axis, and adding one would rehash every HCP cell of
    every cache. Timing is a function of the (b, num_img, num_vox) shapes, not
    of what filled the array, so WGN measures the num_img slope faithfully.

    Args:
        seed_offset (int): first seed; the cache uses
            range(seed_offset, seed_offset + n_seed).
        crop_n_vox_list (iterable[int]): analysis-crop sizes to span (the
            num_vox axis); a single-element list for the fixed-size caches.
        n_seed (int): seeds per crop size.
        b_list (iterable[int]): imaging-feature counts (HCP draws a subset).
        sources (iterable[str]): 'wgn' and/or 'hcp'.
        num_img_list (iterable[int]): subject counts (WGN only).

    Returns:
        list[dict]: kwargs for data_factory, one per cell.
    """
    seeds = range(seed_offset, seed_offset + n_seed)
    kwargs_data_list = []
    for crop_n_vox in crop_n_vox_list:
        kwargs_data_list += get_kwargs_data_list(
            sources=list(sources), seeds=seeds, b_list=b_list,
            num_img_list=num_img_list, crop_n_vox=crop_n_vox)
    return kwargs_data_list


# ---------- leaf-fnc (analysis recipe) grids ---------------------------------
def fit_params_for(ana, glow_fit_params):
    """Return the fit_params a recipe should run under, or None for defaults.

    Args:
        ana (Analysis): an analysis recipe.
        glow_fit_params (dict): what a GLOW recipe asks fit for.

    Returns:
        dict | None: glow_fit_params for a GLOW recipe (the only one with a
            device backend and deep parallelism), else None -- the voxel-wise
            arms take fit's serial default and get their parallelism from the
            sweep's own n_jobs.
    """
    return glow_fit_params if isinstance(ana, AnalysisGLOWBase) else None


def get_run_stat_list(*, n_perm_fwer: int, alpha_fwer: float,
                      cft_pval: float):
    """Build the vba_stat cache's leaf grid (one run_stat call per variant).

    The bake-off among the voxel-wise methods: VBA / VBA-TFCE / CET x the
    stat pool x {raw, z}. GLOW is excluded by design (it uses the LLR
    throughout), so this is VBA / CET only. Each cell pairs a recipe (its
    class / tfce_flag / z_flag identify the variant) with the stat_dict key
    naming the shared-walk matrix run_stat injects as _stat; the method name
    (e.g. VBA-TFCE-Wilks-z) is recovered from those at read time.

    Args:
        n_perm_fwer (int): outer permutations per recipe.
        alpha_fwer (float): family-wise error rate.
        cft_pval (float): CET's cluster-forming threshold.

    Returns:
        list[dict]: kwargs for run_stat (exp / mask_target_list supplied by the
            driver), one per variant.
    """
    kwargs = dict(n_perm_fwer=n_perm_fwer, alpha_fwer=alpha_fwer)
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
                ana=AnalysisCET(get_stat=fn, z_flag=z_flag, cft_pval=cft_pval,
                                **kwargs),
                stat_name=name))
    return specs


def filter_ana_list(kwargs_fnc_list, labels, ana_kwargs_dict) -> list:
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
    path identifies a leaf by), not identity, so a rebuilt equal recipe
    matches. A cell carrying no ana (a non-run_ana leaf grid -- segment /
    prune / ...) never matches, so filtering such a cache yields an empty grid:
    it has no per-method axis to select on.

    Args:
        kwargs_fnc_list (iterable[dict]): a leaf-fnc kwargs grid, e.g.
            config.RUN_ANA_LIST.
        labels (iterable[str]): method names to keep.
        ana_kwargs_dict (dict): the label -> recipe catalogue the names index,
            e.g. config.ana_kwargs_dict.

    Returns:
        list[dict]: the kept cells, in the input grid's order (empty when none
            match).

    Raises:
        ValueError: a label is not a known method name (a typo would otherwise
            silently select nothing).
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
