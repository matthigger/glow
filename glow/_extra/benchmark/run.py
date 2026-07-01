"""Benchmark leaf functions: measure one Experiment and score it.

Each leaf is one fnc(exp, mask_target_list=..., **kwargs) the driver runs on a
(data, effect) cell. run_ana is the canonical leaf; run_segment is a sibling
measuring segmentation quality (no fit); run_min_size captures GLOW's per-perm
(size -> max-z) staircases (recorded as a curve, swept over min_vox post hoc
rather than scored); run_stat fits one VBA / CET MANCOVA-stat variant reading a
shared voxel-stat walk; run_prune scores one pruning rule on a shared GLOW fit.
All share @MEMORY.cache(ignore=['label']) -- label is recorded beside the
output but dropped from the cache key -- and (where they score) score inline.

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

import numpy as np

import glow.graph
from glow.analysis import Analysis, AnalysisGLOW, AnalysisVoxel
from glow.analysis.cluster import cluster, ClusterMode
from glow.analysis.inner_perm import cpu_perm
from glow.analysis.mancova import decompose, stat_dict, stat_dict_inv
from glow.analysis.prune import prune_dp, prune_greedy
from glow.experiment.exper import Experiment, ExperimentScaled

# share the data.py builders' disk cache + recorder, so a fit is memoised
# beside the builds and run_ana joins their provenance DAG (see module docs).
from .data import MEMORY, RECORDER
from .score import (curve_json, score_effects, score_oracle_tree, score_prune,
                    size_max_z_curve)


@MEMORY.cache(ignore=['label'])
@RECORDER(output_name='score', recurse_out_list=['score'])
def run_ana(exp: Experiment, ana: Analysis, mask_target_list, label=None):
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

    label (the config layer's ana_kwargs_dict method name) is unused by the
    computation; it is recorded beside the score (the in.label column) so a
    method is named in the output, and @MEMORY.cache(ignore=['label'])
    drops it from the cache key so renaming a method does not invalidate its
    cache.

    Args:
        exp (Experiment): the experiment to analyze (raw or already
            scaled; fit idempotently scales it).
        ana (Analysis): an unfitted analysis recipe (its __init__ config
            knobs only -- the experiment is not stored on it).
        mask_target_list (list): the planted effect supports, one (X, Y, Z)
            bool mask per EffectSynthetic (effect_factory's mask output);
            empty for the null / FWER-calibration path.
        label (str): the method label recorded beside the score; unused by
            the computation and excluded from the cache key.

    Returns:
        score (dict): the detection score (see .score.score_effects):
            global num_vox / min_pval / n_pred, a per-region pred list, and
            the target (+ per-effect target0..N) confusion blocks.
    """
    ana = copy.deepcopy(ana)
    ana.fit(exp)
    return score_effects(ana, mask_target_list, mask_active=exp.mask_idx > -1)


@MEMORY.cache(ignore=['label'])
@RECORDER(output_name='score', recurse_out_list=['score'])
def run_segment(exp: Experiment, mask_target_list, cluster_mode, label=None):
    """Segment exp in one Ward mode and score the oracle best-Dice region.

    The segmentation-quality leaf: build the Ward tree in cluster_mode and
    return the confusion counts of the region whose Dice against the planted
    support is largest (score_oracle_tree) -- no significance test or pruning,
    swept across modes (Naive / GLM Error / Focus) by the config's fnc grid.
    exp is scaled (ExperimentScaled.from_exp) before clustering so the tree
    matches the one AnalysisGLOW fits (GLM_ERROR / FOCUS project y through the
    design). Memoised + recorded like run_ana; label (the mode name) is
    recorded but not a cache axis.

    Args:
        exp (Experiment): the experiment to segment (raw or scaled).
        mask_target_list (list): planted (X, Y, Z) bool supports; their union
            is the target scored (empty -> all-background counts).
        cluster_mode (ClusterMode | str): the Ward projection to segment with.
        label (str): method label recorded beside the score; not a cache axis.

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


def _min_size_curves(exp, *, n_perm_fwer, n_perm_inner, min_vox_floor,
                     cluster_mode) -> str:
    """Capture each outer perm's (size -> max-z) staircase as JSON.

    The min_size sweep's compute step. Borrows AnalysisGLOW's scaling +
    (q0, q1) decomposition (so the curves match a real fit), then runs the
    outer-perm loop by hand with the exact cpu_perm inner kernel, recording per
    perm the size_max_z_curve corners. cpu_perm gives every region >=
    min_vox_floor an exact z, so GLOW's max-z FWER null re-thresholds at any
    min_vox >= min_vox_floor post hoc without re-fitting (the curve at the
    fit-time min_vox reproduces AnalysisGLOW.max_z_null exactly).

    Args:
        exp (Experiment): the experiment with the synthetic effect imposed.
        n_perm_fwer (int): outer FL perms (n_perm_fwer + 1 curves, incl. k=0).
        n_perm_inner (int): inner FL draws per outer perm.
        min_vox_floor (int): smallest region size given a z; the sweep's lower
            bound (1 keeps the whole range available).
        cluster_mode (ClusterMode): Ward projection.

    Returns:
        a JSON string of the per-perm [size, max_z] corner staircases
        (curve_json; parse with json.loads).
    """
    # scale + decompose as AnalysisGLOW.fit does, so the curves match a real
    # fit; the loop below is by hand to swap the racing kernel for cpu_perm.
    exp_s = ExperimentScaled.from_exp(exp)
    q0, q1, _ = decompose(x=exp_s.x, contrast=exp_s.contrast)

    curve_list = []
    for k in range(n_perm_fwer + 1):
        _exp = exp_s.permute(k) if k else exp_s
        children = cluster(_exp, mode=ClusterMode(cluster_mode))
        llr_k, size = glow.graph.compute_llr_batched(
            _exp, children=children, q0=q0, q1=q1)
        mu, std = cpu_perm(
            exp=_exp, base_seed=(k + 1) * _SEED_OFFSET_DISTINCT,
            n_perm=n_perm_inner, q0=q0, q1=q1, children=children,
            min_vox=min_vox_floor)
        std_safe = np.where(std < 1e-12, 1.0, std)
        z = np.nan_to_num((llr_k - mu) / std_safe,
                          nan=0.0, posinf=0.0, neginf=np.nan)
        consider = (size >= min_vox_floor) & np.isfinite(z)
        curve_list.append(size_max_z_curve(size, z, consider))

    return curve_json(curve_list)


@MEMORY.cache(ignore=['label'])
@RECORDER(output_name='curve')
def run_min_size(exp: Experiment, mask_target_list, *, n_perm_fwer,
                 n_perm_inner, min_vox_floor=1, cluster_mode=ClusterMode.FOCUS,
                 label='GLOW'):
    """Capture GLOW's per-perm (size -> max-z) staircases for a min_vox sweep.

    Records, it does not score. Runs GLOW's outer-perm loop on exp by hand (the
    exact cpu_perm inner kernel; _min_size_curves) and returns the per-perm
    (size -> max-z) corner staircases as a JSON string. With those, GLOW's
    max-z FWER null is swept over min_vox post hoc without re-fitting -- the
    sweep is derived from the records, not here. mask_target_list rides the
    uniform leaf contract but is unused (the curves are a pure function of
    exp); it stays in the cache key (deterministic in exp, so no axis added)
    for uniformity with run_ana.

    Args:
        exp (Experiment): the experiment with the synthetic effect imposed.
        mask_target_list (list): planted supports; accepted for the uniform
            contract but unused (this leaf records curves, not a score).
        n_perm_fwer (int): outer FL perms (n_perm_fwer + 1 curves, incl. k=0).
        n_perm_inner (int): inner FL draws per outer perm.
        min_vox_floor (int): smallest region size given a z; the sweep's lower
            bound (1 keeps the whole range available).
        cluster_mode (ClusterMode): Ward projection (default FOCUS).
        label (str): method label recorded beside the curve; not a cache axis.

    Returns:
        curve (str): a JSON string of the per-perm [size, max_z] corner
            staircases (parse with json.loads; see score.curve_json).
    """
    return _min_size_curves(
        exp, n_perm_fwer=n_perm_fwer, n_perm_inner=n_perm_inner,
        min_vox_floor=min_vox_floor, cluster_mode=cluster_mode)


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


@MEMORY.cache(ignore=['label'])
@RECORDER(output_name='score', recurse_out_list=['score'])
def run_stat(exp: Experiment, mask_target_list, ana: Analysis, stat_name,
             label=None):
    """Fit one VBA / CET MANCOVA-stat variant (reading the shared walk), score.

    The stat bake-off's leaf: fit ana (a VBA / VBA-TFCE / CET recipe around one
    MANCOVA stat) on exp, but inject the precomputed stat matrix from
    voxel_stat_walk (indexed by stat_name) as _stat instead of re-walking the
    permutations, so a cell's variants share one walk. The result is identical
    to a standalone ana.fit(exp) (the walk reproduces the matrix fit would
    build), then scored with score_effects like run_ana. GLOW is excluded from
    this bake-off by design (it uses the LLR throughout), so this leaf is
    VBA / CET only. Memoised + recorded, keyed by (exp, ana, stat_name); label
    (e.g. VBA-TFCE-Wilks-z) is recorded but not a cache axis.

    Args:
        exp (Experiment): the experiment to analyze (raw or scaled).
        mask_target_list (list): the planted effect supports (score target).
        ana (Analysis): an unfitted AnalysisVBA / AnalysisCET recipe; its
            n_perm_fwer sizes the walk and its get_stat picks the stat.
        stat_name (str): the stat_dict key picking which walk matrix to inject
            (must match ana.get_stat's stat).
        label (str): method label recorded beside the score; not a cache axis.

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


@MEMORY.cache(ignore=['label'])
@RECORDER(output_name='score', recurse_out_list=['score'])
def run_prune(exp: Experiment, mask_target_list, rule, *, n_perm_fwer: int,
              n_perm_inner: int, alpha_fwer: float,
              cluster_mode=ClusterMode.FOCUS, label=None):
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
    Memoised + recorded, keyed by (exp, rule, the GLOW fit knobs); label
    (e.g. GLOW-Greedy) is recorded but not a cache axis.

    Args:
        exp (Experiment): the experiment to analyze (raw or scaled).
        mask_target_list (list): the planted effect supports (score target).
        rule (str): 'greedy', 'dp', or 'maxllr'.
        n_perm_fwer (int): outer FWER permutations (the shared fit's).
        n_perm_inner (int): inner FL draws per outer perm (the shared fit's).
        alpha_fwer (float): FWER significance level (the shared fit's).
        cluster_mode (ClusterMode): Ward projection (default FOCUS).
        label (str): method label recorded beside the score; not a cache axis.

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
    one-time clustering is a fixed intercept).

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
                                min_vox=min_vox)
    return int(exp_s.y.shape[2])


@MEMORY.cache(ignore=['label'])
@RECORDER(output_name='num_vox')
def run_segment_time(exp: Experiment, mask_target_list, cluster_mode,
                     label=None) -> int:
    """Time Ward segmentation in one mode (runtime leaf, cluster only).

    The segmentation runtime sweep's leaf: scale exp and build its Ward tree in
    cluster_mode (cluster), timing that and nothing else -- no significance
    test, pruning, or oracle scoring -- so the recorded time_sec isolates the
    clustering cost across modes (Naive / GLM Error / Focus). exp is scaled
    (ExperimentScaled.from_exp) before clustering so the tree matches the one
    AnalysisGLOW fits (as run_segment does).

    Args:
        exp (Experiment): the experiment to segment (raw or scaled).
        mask_target_list (list): planted supports; unused (uniform contract).
        cluster_mode (ClusterMode | str): the Ward projection to segment with.
        label (str): mode label recorded beside the timing; not a cache axis.

    Returns:
        num_vox (int): the analyzed voxel count (recorded beside time_sec).
    """
    exp_s = ExperimentScaled.from_exp(exp)
    cluster(exp_s, mode=ClusterMode(cluster_mode))
    return int(exp_s.y.shape[2])
