"""Benchmark leaf functions: measure one Experiment and score it.

Each leaf is one fnc(exp, mask_target_list=..., **kwargs) the driver runs on a
(data, effect) cell. run_ana is the canonical leaf; run_segment is a sibling
measuring segmentation quality (no fit); run_stat fits one VBA / CET
MANCOVA-stat variant reading a shared voxel-stat walk; run_prune scores one
pruning rule on a shared GLOW fit.
All are @MEMORY.cache'd (so a record's key equals its cache id) and, where they
score, score inline. The method name (GLOW-Focus, VBA-TFCE-Wilks-z, ...) is not
passed or recorded: it is recovered from the recipe at read time from config.py
(see config.ana_kwargs_dict / benchmark.plot).

A leaf may lean on a separately-memoised heavy intermediate rather than a
driver stage: run_stat reads voxel_stat_walk (every MANCOVA stat for one exp)
and run_prune reads glow_fit_for_prune (one GLOW fit's children / per-region
LLR / FWER-significant set), so the first of a cell's variants computes it and
the rest are cache hits -- one shared fit that a cell's N variants (rules or
stats) each score off. The intermediate is a plain memoised
helper, not a recorded DAG node: its output is not an Experiment, so it never
links as a leaf's ancestor, and the leaf already links to the build via exp.

The low-level benchmark primitive: run_ana takes an already-built
Experiment, an unfitted Analysis recipe, and the planted target(s), runs
ana.fit(exp), and returns the detection score of the discovered effects
against the target -- no data building, no effect planting (the config
layer wraps this with those). It is memoised + recorded exactly like the
data.py builders: it shares their MEMORY / RECORDER, so a run is cached on
disk and joins the same provenance DAG -- a run_ana record's exp input
links to the build (data_factory, or effect_factory's plant) that produced
it, so RECORDER.flatten_to_df chains data -> (plant ->) score into one row.

Scoring is inlined rather than a separate recorded step on purpose. The
fitted Analysis is the heavy object (per-region llr / z / children /
max_z_null arrays, ~tens of MB at scale); used as an in-memory local and
discarded, only the small score dict reaches the cache and the records.
Keeping it a downstream node would force either a bespoke link-typed
result carrier or recording that whole object -- and the planted target is
already baked into exp, so folding the score in adds no redundant refits.
The one method-uniform surface every Analysis exposes (effect_list + pval)
is read inside score_effects; the score dict it returns is what every
method is compared on (see .score).

score_effects records only the four confusion counts (+ per-region
geometry, min_pval); Dice / sensitivity / PPV / specificity derive
downstream, so a later metric change re-derives from the records without
re-fitting -- the bulk of the re-score flexibility a separate step would
have bought, at none of the linking cost.
"""

import copy
import json

import numpy as np

import glow.graph
from glow.analysis import Analysis, AnalysisGLOW, AnalysisVoxel
from glow.analysis.cluster import cluster, ClusterMode
from glow.analysis.inner_perm import cpu_perm, _welford_moments
from glow.analysis.mancova import decompose, stat_dict, stat_dict_inv
from glow.analysis.prune import prune_dp, prune_greedy
from glow.experiment import permute
from glow.experiment.exper import Experiment, ExperimentScaled

# share the data.py builders' disk cache + recorder, so a fit is memoised
# beside the builds and run_ana joins their provenance DAG (see module docs).
from .data import MEMORY, RECORDER
from .score import (curve_json, score_effects, score_oracle_tree, score_prune,
                    size_max_z_curve)


@MEMORY.cache
@RECORDER(output_name='score', recurse_out_list=['score'])
def run_ana(exp: Experiment, ana: Analysis, mask_target_list):
    """Fit ana on exp and score it against the planted target(s).

    Calls ana.fit(exp) (every Analysis scales exp on the way in and returns
    self), then scores the discovered effects against the planted supports
    with score_effects -- the uniform detection score a benchmark compares
    across GLOW, VBA, CET, etc. without knowing the concrete type.

    Memoised on disk (MEMORY) with the recorder nested inside the cache,
    keyed by joblib's hash of (exp, ana, mask_target_list): a repeat is
    served from the cache and only a real (cache-miss) run is recorded.
    Two distinct recipes hash distinctly, so each variant caches and
    records on its own. mask_target_list is a deterministic function of exp
    (the effect was planted into it), so it adds no independent cache key
    axis -- it is there because score_effects needs the realized supports,
    which exp does not itself carry.

    ana is never mutated -- fit runs on a private copy, leaving the caller's
    recipe (and so its hash) untouched. That is what makes the cache key
    stable when the same recipe object is reused, and the result
    deterministic in (exp, ana, mask_target_list). The fitted copy (and its
    heavy per-method arrays) is discarded; only the score dict is returned,
    cached, and recorded.

    Args:
        exp (Experiment): the experiment to analyze (raw or already
            scaled; fit idempotently scales it).
        ana (Analysis): an unfitted analysis recipe (its __init__ config
            knobs only -- the experiment is not stored on it).
        mask_target_list (list): the planted effect supports, one (X, Y, Z)
            bool mask per EffectSynthetic (effect_factory's mask output);
            empty for the null / FWER-calibration path.

    Returns:
        score (dict): the detection score (see .score.score_effects):
            global num_vox / min_pval / n_pred, a per-region pred list, and
            the target (+ per-effect target0..N) confusion blocks.
    """
    ana = copy.deepcopy(ana)
    ana.fit(exp)
    return score_effects(ana, mask_target_list, mask_active=exp.mask_idx > -1)


@MEMORY.cache
@RECORDER(output_name='score', recurse_out_list=['score'])
def run_segment(exp: Experiment, mask_target_list, cluster_mode):
    """Segment exp in one Ward mode and score the oracle best-Dice region.

    The segmentation-quality leaf: build the Ward tree in cluster_mode and
    return the confusion counts of the region whose Dice against the planted
    support is largest (score_oracle_tree) -- no significance test or pruning,
    swept across modes (Naive / GLM Error / Focus) by the config's fnc grid.
    exp is scaled (ExperimentScaled.from_exp) before clustering so the tree
    matches the one AnalysisGLOW fits (GLM_ERROR / FOCUS project y through the
    design). Memoised + recorded like run_ana.

    Args:
        exp (Experiment): the experiment to segment (raw or scaled).
        mask_target_list (list): planted (X, Y, Z) bool supports; their union
            is the target scored (empty -> all-background counts).
        cluster_mode (ClusterMode | str): the Ward projection to segment with.

    Returns:
        {tp, fp, tn, fn}: the counts of the best-matching tree region.
    """
    mask_target = np.zeros(exp.mask_idx.shape, dtype=bool)
    for m in mask_target_list:
        mask_target |= m
    children = cluster(ExperimentScaled.from_exp(exp),
                       mode=ClusterMode(cluster_mode))
    return score_oracle_tree(children=children, mask_target=mask_target,
                             mask_idx=exp.mask_idx)


# A large seed offset keeping inner-perm seed regimes apart: outer-perm k draws
# its inner FL perms from the block at (k + 1) * _SEED_OFFSET_DISTINCT,
# matching AnalysisGLOW's per-outer-perm spacing, so inner nulls never collide.
_SEED_OFFSET_DISTINCT = 100_000

# num_inner_perm grid the edge sweep (run_inner_edge) snapshots at: _INNER_GRID_N
# log-spaced points from _INNER_GRID_MIN up to max_inner_perm, dense at the low
# end where the max-z threshold still moves. The floor is >= 2 (std needs two
# draws; _welford_finalize leaves std NaN below).
_INNER_GRID_MIN = 25
_INNER_GRID_N = 20


@MEMORY.cache
def voxel_stat_walk(exp, n_perm_fwer: int) -> dict:
    """Compute the (n_perm+1, num_vox) matrix of every stat for one exp.

    The stat cache's shared heavy intermediate: one Freedman-Lane permutation
    walk over the voxels, computing all stats in stat_dict in a single pass
    (get_stat_perm_multi shares the per-region E / H decomposition across stat
    functions), keyed by stat name. The first run_stat variant of a cell
    computes and caches it; the others are cache hits, so the bake-off of 5
    stats x {raw, z} x {VBA, VBA-TFCE, CET} pays the walk once. A plain
    memoised helper, not a recorded DAG node (see the module docstring).

    exp is scaled (ExperimentScaled.from_exp) before the walk, so the matrix is
    byte-identical to the one AnalysisVoxel.fit would build (fit scales, then
    build_stat_matrix runs the same get_stat_perm on the scaled exp) -- that
    equivalence is what lets run_stat inject this as _stat and get exactly the
    standalone fit's result.

    Note: the cached matrix is ~250 MB per cell at the paper scale (251 perms,
    25k voxels, 5 stats) -- the cost of the per-variant-leaf model (the walk is
    shared on disk across the cell's variants); tune via its scale.

    Args:
        exp (Experiment): the experiment to walk (scaled here).
        n_perm_fwer (int): number of FWER permutations (the walk has n+1 rows,
            row 0 observed).

    Returns:
        {stat_name: (n_perm_fwer+1, num_vox) array}: row 0 observed, rows 1:
            the Freedman-Lane nulls.
    """
    exp = ExperimentScaled.from_exp(exp)
    num_vox = exp.y.shape[2]
    stat_fns = list(stat_dict.values())
    out = {stat_dict_inv[fn]: np.full((n_perm_fwer + 1, num_vox), np.nan)
           for fn in stat_fns}
    for k in range(n_perm_fwer + 1):
        _exp = exp.permute(k) if k else exp
        row = AnalysisVoxel.get_stat_perm_multi(_exp, stat_fns, children=None)
        for fn in stat_fns:
            out[stat_dict_inv[fn]][k, :] = row[fn]
    return out


@MEMORY.cache
@RECORDER(output_name='score', recurse_out_list=['score'])
def run_stat(exp: Experiment, mask_target_list, ana: Analysis, stat_name):
    """Fit one VBA / CET MANCOVA-stat variant (reading the shared walk), score.

    The stat bake-off's leaf: fit ana (a VBA / VBA-TFCE / CET recipe around one
    MANCOVA stat) on exp, but inject the precomputed stat matrix from
    voxel_stat_walk (indexed by stat_name) as _stat instead of re-walking the
    permutations, so a cell's variants share one walk. The result is identical
    to a standalone ana.fit(exp) (the walk reproduces the matrix fit would
    build), then scored with score_effects like run_ana. GLOW is excluded from
    this bake-off by design (it uses the LLR throughout), so this leaf is
    VBA / CET only. Memoised + recorded, keyed by (exp, ana, stat_name).

    Args:
        exp (Experiment): the experiment to analyze (raw or scaled).
        mask_target_list (list): the planted effect supports (score target).
        ana (Analysis): an unfitted AnalysisVBA / AnalysisCET recipe; its
            n_perm_fwer sizes the walk and its get_stat picks the stat.
        stat_name (str): the stat_dict key picking which walk matrix to inject
            (must match ana.get_stat's stat).

    Returns:
        score (dict): the detection score (see score.score_effects).
    """
    walk = voxel_stat_walk(exp, ana.n_perm_fwer)
    ana = copy.deepcopy(ana)
    ana.fit(exp, _stat=walk[stat_name].copy())
    return score_effects(ana, mask_target_list, mask_active=exp.mask_idx > -1)


@MEMORY.cache
def glow_fit_for_prune(exp, *, n_perm_fwer: int, n_perm_inner: int,
                       alpha_fwer: float,
                       cluster_mode=ClusterMode.FOCUS) -> tuple:
    """Fit GLOW once and return the pruning inputs (shared by the rules).

    The prune cache's shared intermediate: a full AnalysisGLOW fit reduced
    to the light triple every rule needs -- the Ward tree, the raw per-region
    LLR (candidates are ranked by raw LLR, as AnalysisGLOW.finalize does; the
    z-score fragments under pruning), and the FWER-significant region set. All
    three rules prune this same set, so the comparison isolates the rule from
    the permutation test; the first run_prune variant of a cell fits, the rest
    are cache hits. A plain memoised helper, not a DAG node (see
    voxel_stat_walk).

    Args:
        exp (Experiment): the experiment to fit (raw or scaled).
        n_perm_fwer (int): outer FWER permutations.
        n_perm_inner (int): inner FL draws per outer perm.
        alpha_fwer (float): FWER significance level (selects sig_reg_list).
        cluster_mode (ClusterMode): Ward projection (default FOCUS).

    Returns:
        children (np.array): (num_reg - num_vox, 2) Ward child-index pairs.
        llr (np.array): (num_reg,) raw per-region LLR, NaN/inf zeroed (the rank
            key the rules prune by).
        sig_reg_list (list): int indices of the FWER-significant regions.
    """
    ana = AnalysisGLOW(n_perm_fwer=n_perm_fwer, n_perm_inner=n_perm_inner,
                       alpha_fwer=alpha_fwer, cluster_mode=cluster_mode)
    ana.fit(exp)
    sig_reg_list = np.where(ana.pval <= ana.alpha_fwer)[0].tolist()
    llr = np.nan_to_num(ana.llr.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    return ana.children, llr, sig_reg_list


@MEMORY.cache
@RECORDER(output_name='score', recurse_out_list=['score'])
def run_prune(exp: Experiment, mask_target_list, rule, *, n_perm_fwer: int,
              n_perm_inner: int, alpha_fwer: float,
              cluster_mode=ClusterMode.FOCUS):
    """Score one pruning rule's selection on a shared GLOW fit.

    Reads the shared GLOW fit (glow_fit_for_prune), applies one rule to its
    FWER-significant regions, and scores the selection against the planted
    support (score_prune). The rules:
      - greedy: bloom the max-LLR region and drop its tree relatives (GLOW's
        default; undersegments).
      - dp: the exact max-total-LLR antichain (prune_dp; oversegments).
      - maxllr: the single highest-LLR significant region (the headline best
        region, n_selected = 1).
    All three prune the same fit, isolating the rule from the permutation test.
    Memoised + recorded, keyed by (exp, rule, the GLOW fit knobs).

    Args:
        exp (Experiment): the experiment to analyze (raw or scaled).
        mask_target_list (list): the planted effect supports (score target).
        rule (str): 'greedy', 'dp', or 'maxllr'.
        n_perm_fwer (int): outer FWER permutations (the shared fit's).
        n_perm_inner (int): inner FL draws per outer perm (the shared fit's).
        alpha_fwer (float): FWER significance level (the shared fit's).
        cluster_mode (ClusterMode): Ward projection (default FOCUS).

    Returns:
        score (dict): the prune counts (see score.score_prune):
            {n_selected, tp, fp, tn, fn}.

    Raises:
        ValueError: if rule is not 'greedy' / 'dp' / 'maxllr'.
    """
    children, llr, sig_reg_list = glow_fit_for_prune(
        exp, n_perm_fwer=n_perm_fwer, n_perm_inner=n_perm_inner,
        alpha_fwer=alpha_fwer, cluster_mode=cluster_mode)

    if rule == 'greedy':
        reg_out_list, _ = prune_greedy(sig_reg_list=sig_reg_list,
                                       children=children, stat=llr)
    elif rule == 'dp':
        reg_out_list, _ = prune_dp(sig_reg_list=sig_reg_list,
                                   children=children, stat=llr)
    elif rule == 'maxllr':
        reg_out_list = ([max(sig_reg_list, key=lambda r: llr[r])]
                        if sig_reg_list else [])
    else:
        raise ValueError(f"rule must be greedy / dp / maxllr, got {rule!r}")

    return score_prune(reg_out_list, children=children, mask_idx=exp.mask_idx,
                       mask_target_list=mask_target_list,
                       mask_active=exp.mask_idx > -1)


# ---------- runtime leaves (timed, not scored) -------------------------------
# The runtime caches measure wall time, not detection. Each leaf runs the one
# piece of GLOW its cache sweeps and returns the analyzed voxel count; the
# RECORDER's time_sec is the measurement, its swept knob rides as an explicit
# in.<name> column, and num_vox rides as the out column -- so runtime plots
# straight from the records. mask_target_list rides the uniform leaf contract
# but is unused (the effect is planted only to keep the run realistic; timing
# is effect-independent).


@MEMORY.cache(ignore=['label'])
@RECORDER(output_name='num_vox')
def run_perm_fwer(exp: Experiment, mask_target_list, *, n_perm_fwer: int,
                  n_perm_inner: int, cluster_mode=ClusterMode.FOCUS,
                  min_vox: int = 1, label=None) -> int:
    """Time GLOW's outer FWER loop for a fixed n_perm_inner (runtime leaf).

    The n_perm_fwer runtime sweep's leaf: run GLOW's outer loop by hand
    (AnalysisGLOW._run_outer for each of the n_perm_fwer + 1 outer perms -- the
    cluster + observed-LLR + inner-perm work) and stop before finalize /
    pruning / scoring. Holding n_perm_inner fixed, the recorded time_sec
    isolates the FWER-permutation cost, which grows linearly in n_perm_fwer.

    Args:
        exp (Experiment): the experiment with the synthetic effect imposed.
        mask_target_list (list): planted supports; unused (uniform contract).
        n_perm_fwer (int): outer FL perms (n_perm_fwer + 1 outer passes, incl.
            the observed k=0); the swept axis.
        n_perm_inner (int): inner FL draws per outer perm (held fixed).
        cluster_mode (ClusterMode): Ward projection (Focus / GLM_ERROR).
        min_vox (int): smallest region size admitted to the inner null.
        label (str): GLOW arm label recorded beside the timing; not a cache
            axis.

    Returns:
        num_vox (int): the analyzed voxel count (recorded beside time_sec as
            the sweep's size context).
    """
    exp_s = ExperimentScaled.from_exp(exp)
    q0, q1, _ = decompose(x=exp_s.x, contrast=exp_s.contrast)
    for k in range(n_perm_fwer + 1):
        AnalysisGLOW._run_outer(
            exp_s, k, q0=q0, q1=q1, n_perm_inner=n_perm_inner,
            min_vox=min_vox, cluster_mode=ClusterMode(cluster_mode))
    return int(exp_s.y.shape[2])


@MEMORY.cache(ignore=['label'])
@RECORDER(output_name='num_vox')
def run_perm_inner(exp: Experiment, mask_target_list, *, n_perm_inner: int,
                   cluster_mode=ClusterMode.FOCUS, min_vox: int = 1,
                   label=None) -> int:
    """Time one observed tree's inner Freedman-Lane null (runtime leaf).

    The n_perm_inner runtime sweep's leaf: cluster the observed (k=0) Ward tree
    once, then run its inner FL null (AnalysisGLOW.run_inner_perm) with
    n_perm_inner draws. Decoupled from the outer FWER loop, so the recorded
    time_sec is the pure inner-permutation cost, linear in n_perm_inner (the
    one-time clustering is a fixed intercept). Uses the full cpu_perm
    (use_race=False) so the swept cost stays the linear-in-n_perm_inner
    reference; the race's tail is not linear and would be a separate leaf.

    Args:
        exp (Experiment): the experiment with the synthetic effect imposed.
        mask_target_list (list): planted supports; unused (uniform contract).
        n_perm_inner (int): inner FL draws over the observed tree; swept axis.
        cluster_mode (ClusterMode): Ward projection (Focus / GLM_ERROR).
        min_vox (int): regions smaller than this are left NaN.
        label (str): GLOW arm label recorded beside the timing; not a cache
            axis.

    Returns:
        num_vox (int): the analyzed voxel count (recorded beside time_sec).
    """
    exp_s = ExperimentScaled.from_exp(exp)
    q0, q1, _ = decompose(x=exp_s.x, contrast=exp_s.contrast)
    children = cluster(exp_s, mode=ClusterMode(cluster_mode))
    AnalysisGLOW.run_inner_perm(exp_s, children, n_perm_inner, q0=q0, q1=q1,
                                min_vox=min_vox, use_race=False)
    return int(exp_s.y.shape[2])


@MEMORY.cache
@RECORDER(output_name='num_vox')
def run_ana_time(exp: Experiment, mask_target_list, ana: Analysis) -> int:
    """Time one method's fit at full local parallelism (runtime leaf).

    The cross-method runtime sweep's leaf: fit ana on exp with n_jobs=-1 (all
    cores) and record wall time only, no detection scoring. Every method's fit
    (GLOW / VBA / CET / TFCE) parallelises its permutation walk over
    joblib.Parallel(n_jobs), so -1 is the wall time a user on an N-core machine
    actually waits -- the wall-clock-in-practice counterpart to run_ana, which
    scores and times fit serially. The result is bit-identical to n_jobs=1 (the
    seed is derived from the permutation index, not the worker), so n_jobs is
    timing-only and never enters the recipe / cache identity.

    Kept distinct from run_ana so the detection caches (keyed on run_ana's code)
    are untouched, and so time_sec isolates fit alone (run_ana's spans fit plus
    score_effects). The method label is recovered from the ana recipe at read
    time (config.ana_kwargs_dict), as for run_ana.

    Args:
        exp (Experiment): the experiment to analyze (raw or scaled; fit scales
            it idempotently).
        mask_target_list (list): planted supports; unused (uniform contract).
        ana (Analysis): an unfitted analysis recipe (config knobs only).

    Returns:
        num_vox (int): analyzed voxel count (mask_active.sum()), recorded beside
            time_sec as the sweep's x-axis.
    """
    ana = copy.deepcopy(ana)
    ana.fit(exp, n_jobs=-1)
    return int((exp.mask_idx > -1).sum())


# ---------- inner-perm edge sweep (num_inner_perm convergence) ---------------
# The n_perm_inner counterpart to the runtime_n_perm_inner cost cache: instead
# of timing the inner null it captures how the max-z FWER threshold converges as
# num_inner_perm grows. The inner draws are seeded base_seed + i, so a single
# sampling to depth max_inner_perm contains every smaller run as a prefix -- one
# capture yields the whole curve, no re-fitting per num_inner_perm.


def _inner_grid(max_inner_perm: int) -> list:
    """Return the ascending num_inner_perm grid the edge is snapshotted at.

    A log-spaced grid (_INNER_GRID_MIN .. max_inner_perm, _INNER_GRID_N points),
    deduped to ints and always including max_inner_perm itself.

    Args:
        max_inner_perm (int): the deepest num_inner_perm (the sampled depth).

    Returns:
        grid (list[int]): ascending ints, >= _INNER_GRID_MIN, ending at
            max_inner_perm.
    """
    grid = np.geomspace(_INNER_GRID_MIN, max_inner_perm, _INNER_GRID_N)
    return sorted({int(round(m)) for m in grid} | {int(max_inner_perm)})


def _inner_draws(exp, *, base_seed: int, n_perm: int, q0, q1, children,
                 min_vox: int):
    """Materialize (n_perm, num_reg) inner FL LLR draws for one Ward tree.

    cpu_perm's construction (Freedman & Lane 1983), but the full draws matrix
    (np.vstack of the iter_llr_perm chunks) rather than the streamed (mu, std)
    reduction: draw i uses seed base_seed + i, so any prefix draws[:m] is the
    exact draw set a standalone n_perm=m cpu_perm produces. run_inner_edge slices
    these prefixes so one sampling covers every num_inner_perm <= n_perm.

    Args:
        exp (Experiment): experiment to sample inner perms from.
        base_seed (int): draw i uses RNG seed base_seed + i.
        n_perm (int): number of inner FL draws (the sweep's max depth).
        q0 (np.array): (a0, num_img) nuisance subspace.
        q1 (np.array): (a1, num_img) interest subspace.
        children (np.array): (num_reg - num_vox, 2) Ward tree.
        min_vox (int): regions smaller than this are left NaN.

    Returns:
        draws (np.array): (n_perm, num_reg) per-draw LLR, NaN where a region is
            smaller than min_vox.
    """
    num_vox = exp.y.shape[2]
    num_img = exp.y.shape[1]
    leaf_ord, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=num_vox)
    perms = np.empty((n_perm, num_img), dtype=np.int64)
    for i in range(n_perm):
        perms[i] = permute._perm_indices(base_seed + i, num_img)
    chunks = glow.graph.iter_llr_perm(
        y=exp.y, q0=q0, q1=q1, perms=perms, leaf_ord=leaf_ord,
        region_l=region_l, region_h=region_h, min_size=min_vox)
    return np.vstack(list(chunks))


def _inner_edge_curve(exp, *, cluster_mode, max_inner_perm: int,
                      n_perm_fwer: int, min_vox: int) -> str:
    """Capture each outer perm's max-z as a function of num_inner_perm.

    Runs GLOW's outer-perm loop by hand (mirroring AnalysisGLOW._run_outer),
    but samples the inner FL null once to depth
    max_inner_perm per outer perm and snapshots the per-region max-z at each
    num_inner_perm on _inner_grid. Because draws[:m] is the exact draw set a real
    n_perm_inner=m fit uses (nested seeds base + i, matching AnalysisGLOW's
    per-outer-perm spacing), each snapshot reproduces that fit's max_z_null[k] --
    one sampling gives the whole edge curve.

    The per-region z and the size >= min_vox max match AnalysisGLOW.fit (the
    1e-12 std floor, the nan/posinf/neginf handling), so the reduced threshold
    equals a real fit's FWER critical value.

    Args:
        exp (Experiment): the experiment with the synthetic effect imposed.
        cluster_mode (ClusterMode): Ward projection (Focus / GLM Error).
        max_inner_perm (int): inner FL draws sampled per outer perm; the deepest
            num_inner_perm the edge is reported at.
        n_perm_fwer (int): outer FL perms (n_perm_fwer + 1 rows, incl. k=0).
        min_vox (int): regions smaller than this are left out of the max.

    Returns:
        a JSON string {num_inner_perm, max_z_null, min_vox}: the grid, the
        (n_perm_fwer+1, len(grid)) per-outer-perm max-z (row 0 observed), and the
        size floor. The FWER critical-value curve is derived from max_z_null post
        hoc (see benchmark.plot); parse with json.loads.
    """
    exp_s = ExperimentScaled.from_exp(exp)
    q0, q1, _ = decompose(x=exp_s.x, contrast=exp_s.contrast)
    grid = _inner_grid(max_inner_perm)
    max_z = np.full((n_perm_fwer + 1, len(grid)), -np.inf)

    for k in range(n_perm_fwer + 1):
        _exp = exp_s.permute(k) if k else exp_s
        children = cluster(_exp, mode=ClusterMode(cluster_mode))
        llr_k, size = glow.graph.compute_llr_batched(
            _exp, children=children, q0=q0, q1=q1)
        draws = _inner_draws(
            _exp, base_seed=(k + 1) * _SEED_OFFSET_DISTINCT,
            n_perm=max_inner_perm, q0=q0, q1=q1, children=children,
            min_vox=min_vox)
        num_reg = draws.shape[1]
        for j, m in enumerate(grid):
            mu, std = _welford_moments([draws[:m]], num_reg)
            std_safe = np.where(std < 1e-12, 1.0, std)
            z = np.nan_to_num((llr_k - mu) / std_safe,
                              nan=0.0, posinf=0.0, neginf=np.nan)
            consider = (size >= min_vox) & np.isfinite(z)
            if consider.any():
                max_z[k, j] = float(np.nanmax(z[consider]))

    return json.dumps({'num_inner_perm': [int(m) for m in grid],
                       'max_z_null': [[float(v) for v in row] for row in max_z],
                       'min_vox': int(min_vox)})


@MEMORY.cache
@RECORDER(output_name='curve')
def run_inner_edge(exp: Experiment, mask_target_list, *, cluster_mode,
                   max_inner_perm: int, n_perm_fwer: int, min_vox: int = 1):
    """Capture GLOW's max-z edge as a function of num_inner_perm (one sampling).

    Records, it does not score. Samples each outer perm's inner FL null once to
    depth max_inner_perm and snapshots the per-region max-z at every
    num_inner_perm on the grid (_inner_edge_curve), exploiting the nested inner
    seeds so one sampling covers every num_inner_perm <= max_inner_perm without
    re-fitting. The convergence read -- how the FWER critical value settles with
    num_inner_perm -- is derived from the recorded max_z_null post hoc, not here.

    The GLOW arm is recovered from the recorded cluster_mode at read time; no
    label is passed or recorded. mask_target_list rides the uniform leaf contract
    but is unused (the edge is a pure function of exp); it stays in the cache key
    for uniformity with run_ana.

    Args:
        exp (Experiment): the experiment with the synthetic effect imposed.
        mask_target_list (list): planted supports; accepted for the uniform
            contract but unused (this leaf records a curve, not a score).
        cluster_mode (ClusterMode): Ward projection (Focus / GLM Error).
        max_inner_perm (int): inner FL draws sampled per outer perm; the deepest
            num_inner_perm the edge is reported at.
        n_perm_fwer (int): outer FL perms feeding the max-z null.
        min_vox (int): regions smaller than this are left out of the max.

    Returns:
        curve (str): a JSON string {num_inner_perm, max_z_null, min_vox}; parse
            with json.loads (see _inner_edge_curve).
    """
    return _inner_edge_curve(
        exp, cluster_mode=cluster_mode, max_inner_perm=max_inner_perm,
        n_perm_fwer=n_perm_fwer, min_vox=min_vox)


# ---------- max-z race retention (survivor race vs full cpu_perm) ------------
# The correctness counterpart to runtime_n_perm_inner's cost cache: per outer
# perm, reduce the inner Freedman-Lane null twice off ONE shared seed -- the
# full cpu_perm (use_race=False) and the survivor race (AnalysisGLOW's shipped
# default) -- and record each path's max-z. Sharing the seed makes the two draw
# identical inner permutations, so a matching max-z pair isolates the race's
# survivor trim as the only difference: it confirms the race retains the
# whole-tree max-z the outer FWER loop reads, on real data at scale (the unit
# tests pin the equivalence on small synthetic trees).


def _max_z(llr, mu, std, size, min_vox: int):
    """Reduce one outer perm's per-region (llr, mu, std) to its max z.

    The exact max-z AnalysisGLOW.fit assembles into max_z_null[k]: standardise
    with the 1e-12 std floor and the nan/posinf/neginf handling, then take the
    max over regions of size >= min_vox with a finite z. Returns (-inf, -1) when
    no region qualifies.

    Args:
        llr (np.array): (num_reg,) observed per-region LLR.
        mu (np.array): (num_reg,) inner-null mean per region.
        std (np.array): (num_reg,) inner-null std per region.
        size (np.array): (num_reg,) region sizes.
        min_vox (int): regions smaller than this are excluded.

    Returns:
        max_z (float): the max per-region z (-inf if none qualify).
        reg (int): the arg-max region index (-1 if none qualify).
    """
    std_safe = np.where(std < 1e-12, 1.0, std)
    z = np.nan_to_num((llr - mu) / std_safe, nan=0.0, posinf=0.0, neginf=np.nan)
    consider = (size >= min_vox) & np.isfinite(z)
    if not consider.any():
        return float('-inf'), -1
    z_masked = np.where(consider, z, -np.inf)
    reg = int(np.argmax(z_masked))
    return float(z_masked[reg]), reg


def _race_maxz_curve(exp, *, cluster_mode, n_perm_fwer: int, n_perm_inner: int,
                     n_perm_inner_race: int, p_keep_thresh: float, min_vox: int) -> str:
    """Capture each outer perm's max-z under the full inner null vs the race.

    Runs GLOW's outer-perm loop by hand (mirroring AnalysisGLOW._run_outer, as
    _inner_edge_curve does): per outer perm cluster the tree, compute the
    observed LLR, then reduce the inner Freedman-Lane null to per-region moments
    twice off the SAME base_seed -- the full cpu_perm (use_race=False) and the
    survivor race (use_race=True, cpu_perm_race) -- and record each path's max-z
    (and its arg-max region). Sharing base_seed makes the two draw identical
    inner permutations, so the only difference is the race's survivor trim.

    Args:
        exp (Experiment): the experiment with the synthetic effect imposed.
        cluster_mode (ClusterMode): Ward projection (Focus / GLM Error).
        n_perm_fwer (int): outer FL perms (n_perm_fwer + 1 rows, incl. k=0).
        n_perm_inner (int): inner FL draws per outer perm (both paths).
        n_perm_inner_race (int): race burn-in draws before the survivor trim.
        p_keep_thresh (float): race survivor keep-probability floor.
        min_vox (int): regions smaller than this are left out of the max.

    Returns:
        a JSON string {n_perm_inner, n_perm_inner_race, p_keep_thresh, min_vox,
        max_z_slow, max_z_race, reg_slow, reg_race}: the knobs, the two
        (n_perm_fwer+1,) per-outer-perm max-z arrays (row 0 observed), and the
        arg-max region each path selected (-1 if none). The race retains the
        max-z where max_z_race == max_z_slow to float round-off, with
        reg_slow == reg_race pinning the same region (a recall hit); parse with
        json.loads.
    """
    exp_s = ExperimentScaled.from_exp(exp)
    q0, q1, _ = decompose(x=exp_s.x, contrast=exp_s.contrast)

    max_z_slow, max_z_race, reg_slow, reg_race = [], [], [], []
    for k in range(n_perm_fwer + 1):
        _exp = exp_s.permute(k) if k else exp_s
        children = cluster(_exp, mode=ClusterMode(cluster_mode))
        llr_k, size = glow.graph.compute_llr_batched(
            _exp, children=children, q0=q0, q1=q1)
        base_seed = (k + 1) * _SEED_OFFSET_DISTINCT
        mu_s, std_s = AnalysisGLOW.run_inner_perm(
            _exp, children, n_perm_inner, q0=q0, q1=q1, min_vox=min_vox,
            base_seed=base_seed, use_race=False)
        mu_r, std_r = AnalysisGLOW.run_inner_perm(
            _exp, children, n_perm_inner, q0=q0, q1=q1, min_vox=min_vox,
            base_seed=base_seed, llr_obs=llr_k, n_perm_inner_race=n_perm_inner_race,
            p_keep_thresh=p_keep_thresh, use_race=True)
        z_s, r_s = _max_z(llr_k, mu_s, std_s, size, min_vox)
        z_r, r_r = _max_z(llr_k, mu_r, std_r, size, min_vox)
        max_z_slow.append(z_s)
        max_z_race.append(z_r)
        reg_slow.append(r_s)
        reg_race.append(r_r)

    return json.dumps({'n_perm_inner': int(n_perm_inner),
                       'n_perm_inner_race': int(n_perm_inner_race),
                       'p_keep_thresh': float(p_keep_thresh),
                       'min_vox': int(min_vox),
                       'max_z_slow': max_z_slow, 'max_z_race': max_z_race,
                       'reg_slow': reg_slow, 'reg_race': reg_race})


@MEMORY.cache
@RECORDER(output_name='curve')
def run_race_maxz(exp: Experiment, mask_target_list, *, cluster_mode,
                  n_perm_fwer: int, n_perm_inner: int, n_perm_inner_race: int,
                  p_keep_thresh: float, min_vox: int = 1):
    """Capture GLOW's per-outer-perm max-z under the full inner null vs race.

    Records, it does not score. Per outer perm reduces the inner Freedman-Lane
    null twice off one shared base_seed -- the full cpu_perm and the survivor
    race (cpu_perm_race, AnalysisGLOW's default inner null) -- and records each
    path's max-z (and arg-max region) via _race_maxz_curve. Because the two
    share the inner permutations, a matching max-z pair confirms the race
    retains the whole-tree max-z the outer FWER loop reads, on real data at
    scale (the unit tests pin the equivalence on small synthetic trees). The
    retention read is derived from the recorded pairs post hoc (benchmark.plot),
    not here.

    The GLOW arm is recovered from the recorded cluster_mode at read time; no
    label is passed or recorded. mask_target_list rides the uniform leaf
    contract but is unused (the curve is a pure function of exp); it stays in
    the cache key for uniformity with run_ana.

    Args:
        exp (Experiment): the experiment with the synthetic effect imposed.
        mask_target_list (list): planted supports; accepted for the uniform
            contract but unused (this leaf records a curve, not a score).
        cluster_mode (ClusterMode): Ward projection (Focus / GLM Error).
        n_perm_fwer (int): outer FL perms feeding the max-z null.
        n_perm_inner (int): inner FL draws per outer perm (both paths).
        n_perm_inner_race (int): race burn-in draws before the survivor trim.
        p_keep_thresh (float): race survivor keep-probability floor.
        min_vox (int): regions smaller than this are left out of the max.

    Returns:
        curve (str): a JSON string {n_perm_inner, n_perm_inner_race, p_keep_thresh,
            min_vox, max_z_slow, max_z_race, reg_slow, reg_race}; parse with
            json.loads (see _race_maxz_curve).
    """
    return _race_maxz_curve(
        exp, cluster_mode=cluster_mode, n_perm_fwer=n_perm_fwer,
        n_perm_inner=n_perm_inner, n_perm_inner_race=n_perm_inner_race,
        p_keep_thresh=p_keep_thresh, min_vox=min_vox)
