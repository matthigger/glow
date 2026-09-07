"""Benchmark leaf functions: measure one Experiment and score it.

Each leaf is one fnc(exp, mask_target_list=..., **kwargs) the driver runs on a
(data, effect) cell. run_ana is the canonical one -- fit an unfitted Analysis
recipe on an already-built Experiment and score the discovered effects against
the planted target(s), no data building and no planting. run_segment is a
sibling measuring segmentation quality (no fit); run_stat fits one VBA / CET
MANCOVA-stat variant off a shared voxel-stat walk; run_prune scores one pruning
rule on a shared GLOW fit; run_inner_perm scores GLOW at one inner-draw count
off a shared per-cell capture.

All are @MEMORY.cache'd (so a record's key equals its cache id) and share
data.py's MEMORY / RECORDER, so a leaf joins the same provenance DAG: its exp
input links to the build that produced it and RECORDER.flatten_to_df chains
data -> (plant ->) score into one row. The method name (GLOW-Focus-greedy,
VBA-TFCE-Wilks-z, ...) is not passed or recorded but recovered from the recipe
at read time (config.ana_kwargs_dict).

Every leaf requires parent_uid, the declared uid of the Experiment it measures
(.recipe): exp itself is kept out of the key, so a leaf's identity is the same
on any machine where the exp's bytes would not be. Two leaves given one
parent_uid claim to measure the same experiment, so a caller must never reuse
one across distinct experiments.

A leaf may lean on a shared heavy intermediate rather than a driver stage:
run_stat reads voxel_stat_walk (every MANCOVA stat for one exp), run_prune
reads glow_fit_for_prune (one GLOW fit's children / per-region LLR /
FWER-significant set) and run_inner_perm reads glow_inner_capture (one GLOW
walk's test inputs at every inner-draw count), so the first of a cell's
variants computes it and the rest reuse it. None is a recorded DAG node -- its
output is not an Experiment, and the leaf already links to the build via exp.
glow_fit_for_prune is memoised to disk, its triple being light; a
voxel_stat_walk matrix and a capture are too big to keep for a whole grid and
are shared in memory only.

Scoring is inlined rather than a separate recorded step: the fitted Analysis is
the heavy object, used as a local and discarded, so only the small score dict
reaches the cache and the records. score_effects stores just the four confusion
counts (plus per-region geometry and min_pval), so a later metric change
re-derives from the records without refitting.
"""

import copy

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

import glow.graph
from glow.analysis import Analysis, AnalysisGLOW, AnalysisVoxel, draws
from glow.analysis._base import resolve_n_jobs, Z_STD_FLOOR
from glow.analysis._fit_gpu import gpu_draws, resolve_gpu
from glow.analysis.cluster import cluster, ClusterMode
from glow.analysis.fwer import max_over_active, MaxStatPerm
from glow.analysis.mancova import decompose, stat_dict, stat_dict_inv
from glow.analysis.prune import prune_by_rule, prune_oracle
from glow.experiment.exper import Experiment, ExperimentScaled

# share the data.py builders' disk cache + recorder, so a fit is memoised
# beside the builds and run_ana joins their provenance DAG (see module docs).
from .data import MEMORY, RECORDER
from .score import (score_effects, score_max_z_region, score_oracle_tree,
                    score_prune)

# Neither exp nor its mask_target_list companion is an identity: exp is named
# by the parent_uid every leaf requires, and the masks are determined by that
# same parent. Keeping both out of every cache key and recipe is what makes a
# leaf's id portable -- and cheap, since a key no longer digests a (b, num_img,
# num_vox) array (see glow._extra.benchmark.recipe).
LEAF_IGNORE = ['exp', 'mask_target_list']

# The same list goes to @MEMORY.cache and @RECORDER: the recorder keys a record
# by joblib's args hash, so it must filter exactly what joblib filters or the
# record stops naming its own cache entry.
#
# fit_params (the kwargs a leaf forwards to Analysis.fit -- n_jobs, gpu) is an
# execution knob, not a recipe knob, so it is filtered like exp: the same cell
# fit on 32 CPU workers or on the GPU is one artifact. That holds only while
# fit_params carries no numerical knob -- a GpuConfig(acc_dtype=float32) does
# perturb fwer.max_stat, which is why gpu=True does not select it and a config
# must not sweep it.
FIT_IGNORE = [*LEAF_IGNORE, 'fit_params']

# parent_uid is declared before every defaulted parameter below, not last:
# joblib's filter_args resolves an omitted default by indexing from the end of
# the signature, which assumes the defaulted parameters are a suffix. A
# required parameter after a defaulted one makes that index run off the front
# and every call raise "Wrong number of arguments".


@MEMORY.cache(ignore=FIT_IGNORE)
@RECORDER(output_name='score', recurse_out_list=['score'],
          ignore=FIT_IGNORE)
def run_ana(exp: Experiment, ana: Analysis, mask_target_list, *,
            parent_uid: str, fit_params=None):
    """Fit ana on exp and score it against the planted target(s).

    Calls ana.fit(exp, **fit_params), then scores the discovered effects
    against the planted supports with score_effects -- the uniform detection
    score every method is compared on, whatever its concrete type.

    Memoised on disk with the recorder nested inside the cache, so a repeat
    is served from the cache and only a real run is recorded. Two distinct
    recipes hash distinctly. mask_target_list is a deterministic function of
    exp, so it adds no cache-key axis; it is there because score_effects
    needs the realized supports, which exp does not carry.

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
        parent_uid (str): the exp's declared uid (see the module docstring).
        fit_params (dict | None): kwargs forwarded to ana.fit -- how the
            fit runs (n_jobs, gpu), never what it computes. Filtered from
            the cache key and the record (FIT_IGNORE), so the same cell run
            on CPU or GPU is one artifact. None (default) takes fit's own
            defaults: serial, CPU.

    Returns:
        score (dict): the detection score (see .score.score_effects):
            global num_vox / min_pval / n_pred, a per-region pred list, and
            the target (+ per-effect target0..N) confusion blocks.
    """
    ana = copy.deepcopy(ana)
    ana.fit(exp, **(fit_params or {}))
    return score_effects(ana, mask_target_list, mask_active=exp.mask_idx > -1)


@MEMORY.cache(ignore=LEAF_IGNORE)
@RECORDER(output_name='score', recurse_out_list=['score'],
          ignore=LEAF_IGNORE)
def run_segment(exp: Experiment, mask_target_list, cluster_mode, *,
                parent_uid: str, frac_segment: float = None):
    """Segment exp in one Ward mode and score the oracle best-Dice region.

    The segmentation-quality leaf: build the Ward tree in cluster_mode and
    return the confusion counts of the region whose Dice against the planted
    support is largest (score_oracle_tree) -- no significance test or pruning,
    swept across modes (Naive / GLM Error / Focus) by the config's fnc grid.
    exp is scaled (ExperimentScaled.from_exp) before clustering so the tree
    matches the one a GLOW fit builds (GLM_ERROR / FOCUS project y through the
    design). Memoised + recorded like run_ana.

    frac_segment is the second axis (the segment_perc cache): the share of the
    images the tree is built on. The whole cohort segments by default; a
    fraction takes AnalysisGLOWSplit's own segmentation fold (split_img
    at its split_seed of 0, so the tree is the one a fit at that frac_segment
    would build) and drops the test fold, which an oracle region needs no more
    than it needs a permutation test. The folds are voxel-identical, so the
    tree still indexes exp's voxels and the target is scored unchanged.
    split_img cuts a prefix of one seeded shuffle, so a larger frac_segment
    holds a smaller one's images too -- the sweep is a nested learning curve,
    not an independent draw per point.

    Args:
        exp (Experiment): the experiment to segment (raw or scaled; raw when
            frac_segment is set -- ExperimentScaled refuses to split).
        mask_target_list (list): planted (X, Y, Z) bool supports; their union
            is the target scored (empty -> all-background counts).
        parent_uid (str): the exp's declared uid (see the module docstring).
        cluster_mode (ClusterMode | str): the Ward projection to segment with.
        frac_segment (float | None): share of the images to build the tree on,
            in (0, 1); None (default) segments the whole cohort.

    Returns:
        {tp, fp, tn, fn}: the counts of the best-matching tree region.
    """
    mask_target = np.zeros(exp.mask_idx.shape, dtype=bool)
    for m in mask_target_list:
        mask_target |= m
    if frac_segment is not None:
        exp, _ = exp.split_img(frac_segment=frac_segment, seed=0)
    children = cluster(ExperimentScaled.from_exp(exp),
                       mode=ClusterMode(cluster_mode))
    return score_oracle_tree(children=children, mask_target=mask_target,
                             mask_idx=exp.mask_idx)


# The edge sweep draws at AnalysisGLOWSplit's own base_seed of 0, so draw i is
# exp_test.permute(i) and a prefix of the matrix is exactly the draw set a
# real fit at that n_perm_fwer produces.



# The current cell's stat walk, {(parent_uid, n_perm_fwer): walk}, holding one
# entry (see voxel_stat_walk for why it is memory-only and bounded to one).
_WALK_MEMO = {}


def voxel_stat_walk(exp, n_perm_fwer: int, *, parent_uid: str) -> dict:
    """Compute the (n_perm+1, num_vox) matrix of every stat for one exp.

    The stat cache's shared heavy intermediate: one Freedman-Lane permutation
    walk over the voxels, computing all stats in stat_dict in a single pass
    (get_stat_perm_multi shares the per-region E / H decomposition across stat
    functions), keyed by stat name. The first run_stat variant of a cell
    computes it, the rest read it back, so the bake-off of 5 stats x {raw, z} x
    {VBA, VBA-TFCE, CET} pays the walk once. Not a recorded DAG node (see the
    module docstring).

    exp is scaled (ExperimentScaled.from_exp) before the walk, so the matrix is
    byte-identical to the one AnalysisVoxel.fit would build (fit scales, then
    build_stat_matrix runs the same get_stat_perm on the scaled exp) -- that
    equivalence is what lets run_stat inject this as _stat and get exactly the
    standalone fit's result.

    The sharing is in memory, not on disk: one walk runs to hundreds of MB,
    which persisted over a whole stat grid would dwarf every other cache.
    _WALK_MEMO holds the current cell's walk and drops the previous one, so
    peak cost is one walk. A cell's variants are consecutive -- drive's leaf
    grid is its innermost loop and its n_jobs splits by data cell -- so that
    one entry serves them all.

    Args:
        exp (Experiment): the experiment to walk (scaled here).
        parent_uid (str): the exp's declared uid (see the module docstring);
            what identifies the walk, since exp is out of the key.
        n_perm_fwer (int): number of FWER permutations (the walk has n+1 rows,
            row 0 observed).

    Returns:
        {stat_name: (n_perm_fwer+1, num_vox) array}: row 0 observed, rows 1:
            the Freedman-Lane nulls.
    """
    key = (parent_uid, n_perm_fwer)
    if key in _WALK_MEMO:
        return _WALK_MEMO[key]
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
    _WALK_MEMO.clear()
    _WALK_MEMO[key] = out
    return out


@MEMORY.cache(ignore=LEAF_IGNORE)
@RECORDER(output_name='score', recurse_out_list=['score'],
          ignore=LEAF_IGNORE)
def run_stat(exp: Experiment, mask_target_list, ana: Analysis, stat_name, *,
             parent_uid: str):
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
        parent_uid (str): the exp's declared uid (see the module docstring).
        ana (Analysis): an unfitted AnalysisVBA / AnalysisCET recipe; its
            n_perm_fwer sizes the walk and its get_stat picks the stat.
        stat_name (str): the stat_dict key picking which walk matrix to inject
            (must match ana.get_stat's stat).

    Returns:
        score (dict): the detection score (see score.score_effects).
    """
    walk = voxel_stat_walk(exp, ana.n_perm_fwer, parent_uid=parent_uid)
    ana = copy.deepcopy(ana)
    ana.fit(exp, _stat=walk[stat_name].copy())
    return score_effects(ana, mask_target_list, mask_active=exp.mask_idx > -1)


@MEMORY.cache(ignore=['exp', 'fit_params'])
def glow_fit_for_prune(exp, *, parent_uid: str, n_perm_fwer: int,
                       n_perm_inner: int, alpha_fwer: float,
                       cluster_mode=ClusterMode.FOCUS,
                       fit_params=None) -> tuple:
    """Fit GLOW once and return the pruning inputs (shared by the rules).

    The prune cache's shared intermediate: one AnalysisGLOW fit (the arm the
    catalogue reports) reduced to the light triple every rule needs -- the
    Ward tree, the raw per-region LLR (the rank key AnalysisGLOWBase._discover
    also prunes by, the z-score fragmenting under pruning), and the
    FWER-significant region set. All the rules prune this same set, so the
    comparison isolates the rule from the permutation test. A plain
    disk-memoised helper, not a DAG node.

    Args:
        exp (Experiment): the experiment to fit (raw or scaled).
        parent_uid (str): the exp's declared uid (see the module docstring);
            what identifies the fit, since exp is out of the key.
        n_perm_fwer (int): outer FL perms feeding the max-z null.
        n_perm_inner (int): inner FL draws standardizing each outer perm's
            own tree.
        alpha_fwer (float): FWER significance level (selects sig_reg_list).
        cluster_mode (ClusterMode): Ward projection (default FOCUS).
        fit_params (dict | None): kwargs forwarded to ana.fit -- how the fit
            runs (n_jobs, gpu), never what it computes. Ignored like exp, so a
            cell fit on CPU or GPU is one artifact (see run_ana). None
            (default) takes fit's own defaults: serial, CPU.

    Returns:
        children (np.array): (num_reg - num_vox, 2) Ward child-index pairs.
        llr (np.array): (num_reg,) raw per-region LLR, NaN/inf zeroed (the rank
            key the rules prune by).
        sig_reg_list (list): int indices of the FWER-significant regions.
    """
    ana = AnalysisGLOW(n_perm_fwer=n_perm_fwer, n_perm_inner=n_perm_inner,
                       alpha_fwer=alpha_fwer, cluster_mode=cluster_mode)
    ana.fit(exp, **(fit_params or {}))
    sig_reg_list = np.flatnonzero(ana.fwer.reg_sig).tolist()
    llr = np.nan_to_num(ana.llr.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    return ana.children, llr, sig_reg_list


@MEMORY.cache(ignore=FIT_IGNORE)
@RECORDER(output_name='score', recurse_out_list=['score'],
          ignore=FIT_IGNORE)
def run_prune(exp: Experiment, mask_target_list, rule, *, parent_uid: str,
              n_perm_fwer: int, n_perm_inner: int, alpha_fwer: float,
              cluster_mode=ClusterMode.FOCUS, fit_params=None):
    """Score one pruning rule's selection on a shared GLOW fit.

    Reads the shared GLOW fit (glow_fit_for_prune), applies one rule to its
    FWER-significant regions, and scores the selection against the planted
    support (score_prune). The rules:
      - greedy: bloom the max-LLR region and drop its tree relatives (GLOW's
        default; undersegments).
      - dp: the exact max-total-LLR antichain (prune_dp; oversegments).
      - single_max: the single highest-LLR significant region (the headline
        best region, n_selected = 1).
      - oracle: the max-Dice antichain against the planted support
        (prune_oracle). Not a method -- it is handed the target the others
        are scored against, so it draws the headroom the rules leave: the
        Dice this fit's significant set still has in it.
    All four prune the same fit, isolating the rule from the permutation test.
    Memoised + recorded, keyed by (exp, rule, the GLOW fit knobs).

    Args:
        exp (Experiment): the experiment to analyze (raw or scaled).
        mask_target_list (list): the planted effect supports (score target,
            and the oracle rule's input).
        parent_uid (str): the exp's declared uid (see the module docstring).
        rule (str): 'greedy', 'dp', 'single_max', or 'oracle'.
        n_perm_fwer (int): outer FL perms (the shared fit's).
        n_perm_inner (int): inner FL draws per outer perm (the shared fit's).
        alpha_fwer (float): FWER significance level (the shared fit's).
        cluster_mode (ClusterMode): Ward projection (default FOCUS).
        fit_params (dict | None): kwargs forwarded to the shared fit -- how it
            runs (n_jobs, gpu), never what it computes (see run_ana).

    Returns:
        score (dict): the prune counts (see score.score_prune):
            {n_selected, tp, fp, tn, fn, pred}. The pred block is one
            {reg_idx, num_vox, target} record per selected region, which is
            what lets a rule be scored on how it divided the volume up
            (completeness / homogeneity) and not only on how much of it
            overlapped.

    Raises:
        ValueError: if rule is not 'greedy' / 'dp' / 'single_max' / 'oracle'.
    """
    children, llr, sig_reg_list = glow_fit_for_prune(
        exp, parent_uid=parent_uid, n_perm_fwer=n_perm_fwer,
        n_perm_inner=n_perm_inner, alpha_fwer=alpha_fwer,
        cluster_mode=cluster_mode, fit_params=fit_params)

    if rule == 'oracle':
        # scored against the union of the planted supports, as
        # score_prune scores every rule's output
        mask_target = np.zeros(exp.mask_idx.shape, dtype=bool)
        for mask in mask_target_list:
            mask_target |= mask
        reg_out_list, _ = prune_oracle(sig_reg_list=sig_reg_list,
                                       children=children,
                                       mask_target=mask_target,
                                       mask_idx=exp.mask_idx)
    else:
        reg_out_list, _ = prune_by_rule(rule, sig_reg_list=sig_reg_list,
                                        children=children, stat=llr)

    return score_prune(reg_out_list, children=children, mask_idx=exp.mask_idx,
                       mask_target_list=mask_target_list,
                       mask_active=exp.mask_idx > -1)


# ---------- inner-draw sweep (one capture, every n_perm_inner) ---------------
# GLOW standardizes each outer permutation's tree against n_perm_inner inner
# Freedman-Lane draws, and draw i of that null is seeded base_seed + i
# (glow.analysis.draws), so the first m + 1 rows of a deeper matrix are exactly
# the draws a fit at n_perm_inner = m takes. One sampling to the deepest count
# on the grid therefore carries every shallower count as a prefix:
# glow_inner_capture summarizes each prefix per outer perm, and run_inner_perm
# reads one prefix per leaf. So the whole curve costs the deepest fit rather
# than a fit per point, and its points share their trees -- the comparison is
# paired within a cell.

# The current cell's capture, {key: capture}, holding one entry (see
# voxel_stat_walk for why the sharing is memory-only and bounded to one).
_INNER_MEMO = {}


def _capture_outer(exp, k: int, *, q0, q1, n_perm_inner_grid, min_vox: int,
                   cluster_mode, gpu_config=None):
    """Summarize one outer permutation at every inner count on the grid.

    glow.analysis._glow._run_outer, snapshotted: cluster this permutation,
    draw its inner null to the deepest count on the grid, and standardize
    its observed row against each prefix of that matrix. Every snapshot is
    the (mu, std) a real AnalysisGLOW fit at that n_perm_inner reduces this
    tree to, so the fit is not approximated -- it is re-read.

    The prefixes are folded one after another through the draw module's own
    Chan accumulators rather than re-reduced from scratch, which is both
    what makes the moments equal a fit's and what keeps the whole grid to
    one pass over the matrix. Only the observed row's z is formed: a
    prefix's other rows go into its moments and are then max'd over in the
    fits this stands in for, but only row 0 of an outer perm's own matrix
    reaches its FWER null.

    Args:
        exp (Experiment): the scaled experiment, unpermuted.
        k (int): outer-perm index; 0 is the observed data.
        q0 (np.array): (a0, num_img) nuisance subspace.
        q1 (np.array): (a1, num_img) interest subspace.
        n_perm_inner_grid (tuple): ascending inner counts to snapshot at.
        min_vox (int): regions smaller than this sit out of the max.
        cluster_mode (ClusterMode): Ward projection mode.
        gpu_config (GpuConfig | None): device knobs, None for the CPU.

    Returns:
        max_z (np.array): (len(n_perm_inner_grid),) this perm's max z per
            inner count -- its entry in the FWER null at each one.
        obs (tuple | None): (children, size, llr, z_obs) for k == 0, None
            otherwise; z_obs is (len(n_perm_inner_grid), num_reg).
    """
    exp_k = exp.permute(k) if k else exp
    children = cluster(exp_k, mode=cluster_mode)

    _, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=exp_k.y.shape[2])
    size = region_h - region_l
    reg_active = size >= min_vox

    mat = dict(exp=exp_k, base_seed=0, n_perm=max(n_perm_inner_grid) + 1,
               q0=q0, q1=q1, children=children, min_vox=min_vox)
    mat = (draws.cpu_batched(**mat) if gpu_config is None
           else gpu_draws(gpu_config, **mat))

    llr = mat[0]
    num_reg = mat.shape[1]
    n = np.zeros(num_reg)
    mean = np.zeros(num_reg)
    m2 = np.zeros(num_reg)
    row = 0
    max_z = np.empty(len(n_perm_inner_grid))
    z_list = []
    for j, n_perm_inner in enumerate(n_perm_inner_grid):
        n, mean, m2 = draws._chan_combine(mat[row:n_perm_inner + 1],
                                          n, mean, m2)
        row = n_perm_inner + 1
        mu, std = draws._chan_moments(n, mean, m2)
        z = (llr - mu) / np.where(std > Z_STD_FLOOR, std, 1.0)
        max_z[j] = max_over_active(z[None, :], reg_active)[0]
        if not k:
            z_list.append(z)

    if k:
        return max_z, None
    return max_z, (children, size, llr, np.stack(z_list))


def glow_inner_capture(exp, *, parent_uid: str, n_perm_fwer: int,
                       n_perm_inner_grid, cluster_mode=ClusterMode.FOCUS,
                       min_vox: int = 1, fit_params=None) -> dict:
    """Run GLOW's outer loop once, keeping every inner count's test inputs.

    The inner-draw sweep's shared heavy intermediate: one per-perm GLOW walk
    (AnalysisGLOW.fit's loop) at the deepest count on the grid, reduced to
    what a max-z test needs at each count -- the observed tree, its raw LLR,
    its observed z, and the per-outer-perm max-z null. run_inner_perm turns
    one column of that into the fit's significant set, so a cell's whole
    n_perm_inner grid costs one walk. Not a recorded DAG node (see the
    module docstring).

    The sharing is in memory, not on disk: a capture is
    (len(n_perm_inner_grid), num_reg), small beside a stat walk but not
    something a whole grid of cells should keep, and a cell's leaves are
    consecutive (drive's leaf grid is its innermost loop), so one entry
    serves them.

    Args:
        exp (Experiment): the experiment to fit (scaled here).
        parent_uid (str): the exp's declared uid (see the module
            docstring); what identifies the capture, since exp is out of
            the key.
        n_perm_fwer (int): outer FL perms feeding the max-z null.
        n_perm_inner_grid (tuple): ascending inner counts to capture.
        cluster_mode (ClusterMode): Ward projection (default FOCUS).
        min_vox (int): smallest region admitted to the comparison set.
        fit_params (dict | None): how the walk runs (n_jobs, gpu), resolved
            as AnalysisGLOW.fit resolves them; never what it computes.

    Returns:
        the capture dict:
            {n_perm_inner_grid: tuple,
             children: (num_reg - num_vox, 2),
             size: (num_reg,), llr: (num_reg,) observed raw LLR,
             z_obs: (len(grid), num_reg) observed z per inner count,
             max_z_null: (n_perm_fwer + 1, len(grid)) max z per outer perm
                 per inner count, row 0 the observed perm}
    """
    n_perm_inner_grid = tuple(int(m) for m in n_perm_inner_grid)
    key = (parent_uid, n_perm_fwer, n_perm_inner_grid,
           str(ClusterMode(cluster_mode)), min_vox)
    if key in _INNER_MEMO:
        return _INNER_MEMO[key]

    fit_params = fit_params or {}
    gpu_config = resolve_gpu(fit_params.get('gpu', False),
                             name='glow_inner_capture')
    n_jobs = resolve_n_jobs(fit_params.get('n_jobs', 1))

    exp = ExperimentScaled.from_exp(exp)
    q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)
    outer_kwargs = dict(q0=q0, q1=q1, n_perm_inner_grid=n_perm_inner_grid,
                        min_vox=min_vox,
                        cluster_mode=ClusterMode(cluster_mode),
                        gpu_config=gpu_config)
    results = Parallel(n_jobs=1 if gpu_config else n_jobs,
                       return_as='generator')(
        delayed(_capture_outer)(exp, k, **outer_kwargs)
        for k in range(n_perm_fwer + 1))

    max_z_null = np.empty((n_perm_fwer + 1, len(n_perm_inner_grid)))
    obs = None
    for k, (max_z, obs_k) in enumerate(results):
        max_z_null[k] = max_z
        if obs_k is not None:
            obs = obs_k
    children, size, llr, z_obs = obs

    capture = dict(n_perm_inner_grid=n_perm_inner_grid, children=children,
                   size=size, llr=llr, z_obs=z_obs, max_z_null=max_z_null)
    _INNER_MEMO.clear()
    _INNER_MEMO[key] = capture
    return capture


# n_perm_inner_grid is ignored like exp: it says which prefixes the shared
# capture snapshots, not what this leaf computes. A prefix is the same draws
# whatever depth was sampled around it, so one n_perm_inner is one artifact
# however the grid it was captured with is widened or narrowed.
INNER_IGNORE = [*FIT_IGNORE, 'n_perm_inner_grid']


@MEMORY.cache(ignore=INNER_IGNORE)
@RECORDER(output_name='score', recurse_out_list=['score'],
          ignore=INNER_IGNORE)
def run_inner_perm(exp: Experiment, mask_target_list, *, parent_uid: str,
                   n_perm_inner: int, n_perm_fwer: int, alpha_fwer: float,
                   n_perm_inner_grid, cluster_mode=ClusterMode.FOCUS,
                   prune_rule: str = 'greedy', min_vox: int = 1,
                   fit_params=None):
    """Score GLOW at one inner-draw count, off the cell's shared capture.

    The inner-draw sweep's leaf: read one n_perm_inner out of
    glow_inner_capture, threshold that column's max-z null, prune the
    significant regions by prune_rule, and score the selection against the
    planted support. Equivalent to fitting AnalysisGLOW at this
    n_perm_inner and scoring it, but every count on the grid shares one
    walk (see the section comment).

    Two things are recorded, because n_perm_inner acts on the test in two
    places. The selection counts are detection as usual. The max_z block is
    which region the max-z statistic came from: the inner draws set every
    region's (mu, std), so the argmax can move with the count, and a count
    is high enough only once it has stopped moving.

    Args:
        exp (Experiment): the experiment to analyze (raw or scaled).
        mask_target_list (list): the planted (X, Y, Z) bool supports.
        parent_uid (str): the exp's declared uid (see the module docstring).
        n_perm_inner (int): the inner count to report; must be on
            n_perm_inner_grid.
        n_perm_fwer (int): outer FL perms feeding the max-z null.
        alpha_fwer (float): FWER significance level.
        n_perm_inner_grid (tuple): the counts the shared capture snapshots,
            this one among them. Filtered from the key (INNER_IGNORE).
        cluster_mode (ClusterMode): Ward projection (default FOCUS).
        prune_rule (str): selection rule (glow.analysis.prune), its own
            knobs left at their defaults -- this cache sweeps the inner
            count, not the rule.
        min_vox (int): smallest region admitted to the comparison set.
        fit_params (dict | None): how the shared walk runs (n_jobs, gpu),
            never what it computes (see run_ana).

    Returns:
        score (dict): {n_selected, tp, fp, tn, fn} for the selection (see
            score.score_prune), plus n_sig (significant regions), min_pval,
            and a max_z block (see score.score_max_z_region).

    Raises:
        ValueError: n_perm_inner is not on n_perm_inner_grid.
    """
    capture = glow_inner_capture(
        exp, parent_uid=parent_uid, n_perm_fwer=n_perm_fwer,
        n_perm_inner_grid=n_perm_inner_grid, cluster_mode=cluster_mode,
        min_vox=min_vox, fit_params=fit_params)

    inner_grid = capture['n_perm_inner_grid']
    if int(n_perm_inner) not in inner_grid:
        raise ValueError(f'n_perm_inner={n_perm_inner} is not on the '
                         f'captured grid {inner_grid}')
    j = inner_grid.index(int(n_perm_inner))

    children, size = capture['children'], capture['size']
    reg_active = size >= min_vox
    z_obs = capture['z_obs'][j]
    fwer = MaxStatPerm.from_max(z_obs, capture['max_z_null'][:, j],
                                alpha=alpha_fwer, reg_active=reg_active)

    # ranked by raw LLR, as AnalysisGLOWBase._discover ranks them
    llr = np.nan_to_num(capture['llr'].astype(float), nan=0.0, posinf=0.0,
                        neginf=0.0)
    sig_reg_list = np.flatnonzero(fwer.reg_sig).tolist()
    reg_out_list, _ = prune_by_rule(prune_rule, sig_reg_list=sig_reg_list,
                                    children=children, stat=llr)

    score = score_prune(reg_out_list, children=children,
                        mask_idx=exp.mask_idx,
                        mask_target_list=mask_target_list,
                        mask_active=exp.mask_idx > -1)
    score['n_sig'] = len(sig_reg_list)
    score['min_pval'] = (float(np.nanmin(fwer.pval))
                         if np.isfinite(fwer.pval).any() else float('nan'))
    score['max_z'] = score_max_z_region(
        z_obs, reg_active=reg_active, size=size, children=children,
        mask_idx=exp.mask_idx, mask_target_list=mask_target_list,
        mask_active=exp.mask_idx > -1)
    return score


# ---------- runtime leaves (timed, not scored) -------------------------------
# The runtime caches measure wall time, not detection. Both leaves fit a real
# recipe and return the analyzed voxel count; the RECORDER's time_sec is the
# measurement and num_vox rides as the out column, so a runtime figure plots
# straight from the records. They differ only in what they hold fixed:
# run_ana_time takes every core and any device (wall clock as experienced),
# run_ana_time_1perm pins one core and one permutation (work, near enough to
# read a growth rate off). mask_target_list rides the uniform leaf contract but
# is unused (the effect is planted only to keep the run realistic; timing is
# effect-independent).


# run_ana_time's default fit_params: all cores, CPU. A dict literal rather
# than a mutable module constant, so a caller cannot edit the default.
def _default_time_fit_params() -> dict:
    """Return the fit kwargs a timing leaf uses when a cell names none."""
    return dict(n_jobs=-1)


@MEMORY.cache(ignore=FIT_IGNORE)
@RECORDER(output_name='num_vox', ignore=FIT_IGNORE)
def run_ana_time(exp: Experiment, mask_target_list, ana: Analysis, *,
                 parent_uid: str, fit_params=None) -> int:
    """Time one method's fit at full local parallelism (runtime leaf).

    The cross-method runtime sweep's leaf: fit ana on exp with n_jobs=-1 and
    record wall time only, no scoring. Every method parallelises its
    permutation walk, so -1 is the wall time a user on an N-core machine
    waits;
    the result is bit-identical to n_jobs=1 (seeds come from the permutation
    index, not the worker), so n_jobs stays out of the cache identity.

    GOTCHA fit_params is filtered like run_ana's (FIT_IGNORE), so a CPU timing
    and a GPU timing of one cell collide on the same key and the second is
    served from the first rather than measured. Timing a second backend means
    clearing that entry.

    Kept distinct from run_ana so the detection caches are untouched and
    time_sec isolates fit alone.

    Args:
        exp (Experiment): the experiment to analyze (raw or scaled; fit scales
            it idempotently).
        mask_target_list (list): planted supports; unused (uniform contract).
        parent_uid (str): the exp's declared uid (see the module docstring).
        ana (Analysis): an unfitted analysis recipe (config knobs only).
        fit_params (dict | None): kwargs forwarded to ana.fit; None
            (default) times it at n_jobs=-1 on the CPU.

    Returns:
        num_vox (int): analyzed voxel count, recorded beside time_sec as
            the sweep's x-axis.
    """
    ana = copy.deepcopy(ana)
    ana.fit(exp, **(fit_params if fit_params is not None
                    else _default_time_fit_params()))
    return int((exp.mask_idx > -1).sum())


def _take_img(exp: Experiment, num_img: int) -> Experiment:
    """Return exp cut to its leading num_img subjects (a timing helper).

    Both y and the design x lose the same columns, so the result is shaped
    like an experiment of that many subjects. y is copied, not sliced into a
    view: the analysis path expects the F-contiguous layout the builders
    produce (data._with_canonical_y), and timing a strided view would measure
    the stride rather than the size.

    Args:
        exp (Experiment): the experiment to cut.
        num_img (int): subjects to keep.

    Returns:
        Experiment: y (b, num_img, num_vox), x (a, num_img), same mask.

    Raises:
        ValueError: num_img exceeds the cohort.
    """
    if num_img > exp.y.shape[1]:
        raise ValueError(f'num_img={num_img} exceeds the {exp.y.shape[1]}'
                         f'-subject cohort')
    return Experiment(y=exp.y[:, :num_img].copy(order='F'),
                      x=exp.x[:, :num_img], contrast=exp.contrast,
                      mask_idx=exp.mask_idx)


@MEMORY.cache(ignore=LEAF_IGNORE)
@RECORDER(output_name='num_vox', ignore=LEAF_IGNORE)
def run_ana_time_1perm(exp: Experiment, mask_target_list, ana: Analysis, *,
                       parent_uid: str, n_perm_fwer: int = 1,
                       n_perm_inner: int = None,
                       num_img: int = None) -> int:
    """Time one method's fit on a single core, one permutation deep.

    The growth-rate leaf: fit ana with the permutation counts overridden and
    the machine pinned to one core -- n_jobs=1, no device, BLAS held to one
    thread (threadpool_limits). It therefore measures work rather than
    schedule, which matters because the parallel speedup is itself a function
    of the swept axis: a bandwidth-bound fit plateaus at a few workers
    where a compute-bound one keeps scaling, so an all-cores wall time
    conflates the algorithm's growth with the machine's.

    n_perm_fwer defaults to 1, the observed pass plus one outer permutation, a
    full run costing n_perm_fwer times that per-permutation term. The separate
    n_perm_fwer sweep is what splits that slope from the fixed intercept (the
    observed pass, synthesis and pruning) rather than assuming the split.

    n_perm_inner is the count inside that permutation: every region of every
    outer draw is standardized against its own inner draws, so it multiplies
    the per-region walk rather than the tree build. Left at the recipe's own
    value unless swept.

    Unlike run_ana_time this leaf takes no fit_params: the serial contract is
    the measurement. Every swept knob rides as an explicit argument so that it
    keys the cache (fit_params would not) and lands in the record as its own
    in.<name> column, leaving in.ana to name the method.

    num_img is swept here rather than by building a smaller experiment: the HCP
    cohort is the sample, and giving data_factory_hcp a subject-subset axis
    would put num_img in every HCP cell's declared recipe and rehash the whole
    catalogue. Timing is a function of the array shapes, not of which subjects
    fill them.

    Args:
        exp (Experiment): the experiment to analyze (raw or scaled; fit scales
            it idempotently).
        mask_target_list (list): planted supports; unused (uniform contract).
        parent_uid (str): the exp's declared uid (see the module docstring).
        ana (Analysis): an unfitted analysis recipe; deep-copied before its
            permutation counts are overridden, so the caller's is untouched.
        n_perm_fwer (int): permutations to time, the observed pass on
            top. 1 (default) is the per-permutation cost.
        n_perm_inner (int | None): inner draws standardizing each region.
            None (default) keeps the recipe's own count.
        num_img (int | None): subjects to keep, the leading num_img of them.
            None (default) is the whole cohort.

    Returns:
        num_vox (int): analyzed voxel count (mask_active.sum()), recorded
            beside time_sec as the sweep's size context.

    Raises:
        ValueError: num_img larger than the cohort (a silently short curve).
        ValueError: n_perm_inner given for a recipe without that knob, which
            would otherwise time an unswept fit under a swept label.
    """
    ana = copy.deepcopy(ana)
    ana.n_perm_fwer = n_perm_fwer
    if n_perm_inner is not None:
        if not hasattr(ana, 'n_perm_inner'):
            raise ValueError(f'{type(ana).__name__} has no n_perm_inner')
        ana.n_perm_inner = n_perm_inner
    if num_img is not None:
        exp = _take_img(exp, num_img)

    with threadpool_limits(limits=1):
        ana.fit(exp, n_jobs=1, gpu=False)
    return int((exp.mask_idx > -1).sum())
