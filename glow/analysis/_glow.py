"""GLOW analysis: per-permutation Ward segmentation with FWER control.

Two arms, kept side by side because they trade the same two things against
each other and which trade wins is measured rather than argued:

  - AnalysisGLOW (here) rebuilds the Ward tree inside every outer
    permutation and standardizes that tree's regions against inner
    Freedman-Lane draws of the same permuted data.
  - AnalysisGLOWSplit (glow.analysis._glow_split) builds one tree, on a
    held-out fold of the images, and needs no inner null.

Segmentation is what the per-perm arm buys: every image reaches the tree
and every image reaches the statistics, where the split arm spends
frac_segment of them on the tree and the rest on the statistics. The tree
is the whole family of hypotheses, so halving the sample that builds it is
not a small cost.

Validity is what it pays. Only the split arm has strong FWER control; see
AnalysisGLOW for what the per-perm arm does and does not control, and
AnalysisGLOWSplit for the argument the split turns on.

AnalysisGLOWBase holds what the two share: the recipe knobs, the kept-matrix
contract, and the prune-then-estimate synthesis that turns a max-stat test
into effects.
"""

import warnings

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import glow.effect
import glow.graph
from glow.experiment.exper import ExperimentScaled
from ._base import Analysis, resolve_n_jobs
from . import draws
from ._fit_gpu import gpu_draws, gpu_summary, resolve_gpu
from .cluster import cluster, ClusterMode
from .fwer import MaxStatPerm
from .mancova import decompose
from .prune import prune_greedy


class AnalysisGLOWBase(Analysis):
    """Ward-tree effect discovery: the parts both GLOW arms share.

    Not fit directly -- fit() belongs to the arms, which differ in where
    the tree comes from and what the regions are standardized against. What
    is common is everything either side of that: the four recipe knobs, the
    per-region arrays a fit reports, the max-stat test they feed, and the
    synthesis that turns significant regions into effects (_discover).

    Operation parameters (set at __init__):
        n_perm_fwer (int): draws in the FWER null.
        alpha_fwer (float): family-wise error rate.
        min_vox (int): smallest region size admitted to the FWER set.
            Must be pre-specified, never tuned against results, or it
            reintroduces selection one level up.
        cluster_mode (ClusterMode): Ward projection mode.
        keep_stat (bool): keep a draw matrix in .stat. Which matrix that is
            differs by arm; both cost it in memory and in any pickle of the
            fit, and neither changes a reported result -- see the arms.

    Fit outputs (on the observed data, against the observed tree):
        children (np.array): (num_reg - num_vox, 2) Ward tree.
        size (np.array): (num_reg,) region sizes.
        llr (np.array): (num_reg,) observed LLR per region.
        mu (np.array): (num_reg,) per-region mean over the draws.
        std (np.array): (num_reg,) per-region std over the draws.
        fwer (MaxStatPerm): the max-z test -- observed z per region in
            stat_obs, the max-z null in max_stat, the size >= min_vox
            comparison set, the p-values and the selected regions.
        effect_list (list): discovered EffectEstimate objects.
        stat (np.array): (n_perm + 1, num_reg) per-region LLR draws, row 0
            the observed one -- only under keep_stat, None otherwise. Same
            name, shape and role as AnalysisVoxel.stat, which the voxel
            arms always keep.
    """

    def __init__(self, n_perm_fwer: int, alpha_fwer: float = .05,
                 min_vox: int = 1,
                 cluster_mode: ClusterMode = ClusterMode.FOCUS,
                 keep_stat: bool = False):
        """Configure the knobs both arms take.

        Args:
            n_perm_fwer (int): draws in the FWER null.
            alpha_fwer (float): family-wise error rate.
            min_vox (int): smallest region size admitted to the FWER set.
            cluster_mode (ClusterMode): Ward projection. Default
                ClusterMode.FOCUS projects onto the contrast subspace;
                ClusterMode.GLM_ERROR keeps bias + contrast;
                ClusterMode.NAIVE clusters raw y.
            keep_stat (bool): keep a draw matrix in .stat instead of
                discarding it.
        """
        super().__init__()
        self.n_perm_fwer = n_perm_fwer
        self.alpha_fwer = alpha_fwer
        self.min_vox = min_vox
        self.cluster_mode = cluster_mode
        self.keep_stat = keep_stat

        self.children = None
        self.size = None
        self.llr = None
        self.mu = None
        self.std = None
        self.stat = None

    def __getstate__(self) -> dict:
        """Warn when a kept stat matrix is about to be written out.

        Every save route -- pickle, joblib's hash and cache, the viewer
        bundle, deepcopy -- reaches the object through here, so this is the
        one place the cost can be announced. A fit that kept its stat looks
        like a recipe and weighs like the matrix: ~16.7 GiB at 5001 draws
        and full-brain num_vox, against a few MB without it.

        The matrix is still written. Dropping it here would silently
        produce a bundle the viewer's PERMUTATION panel cannot open, and a
        diagnostic that vanishes on save is worse than a loud one.
        """
        if self.stat is not None:
            warnings.warn(
                f'pickling {type(self).__name__} with keep_stat=True: the '
                f'{self.stat.shape} {self.stat.dtype} stat matrix '
                f'({self.stat.nbytes / 2 ** 30:.2f} GiB) is written with '
                f'the fit. Refit with keep_stat=False to save the fit '
                f'alone.')
        return self.__dict__

    def _discover(self, exp, *, verbose: bool = False) -> None:
        """Prune the significant regions and estimate the effects they carry.

        Requires fwer, children and llr; sets effect_list. Shared by both
        arms so the pruning rule and the p-value each effect carries cannot
        drift between them.

        Args:
            exp (Experiment): the (scaled) experiment the statistics came
                from -- supplies mask_idx, and is carried into each
                EffectEstimate. The arm passes whichever images tested the
                regions, which for the split arm is the test fold alone.
            verbose (bool): print significant-region and discovery counts.
        """
        sig_reg_list = list(np.flatnonzero(self.fwer.reg_sig))
        if verbose:
            print(f'  {len(sig_reg_list)} significant regions '
                  f'(alpha_fwer={self.alpha_fwer})')

        # Rank pruning candidates by raw LLR.  z-score is right for
        # FWER thresholding (puts different-size regions on a common
        # scale), but it fragments under pruning: for a true effect of
        # size n with per-voxel strength alpha, z scales as ~sqrt(n),
        # so a small slice of the effect can outscore the whole region
        # on z.  Greedy z-pruning then locks out the parent and emits
        # fragments with very low Dice.  Raw LLR scales linearly with
        # n, picks the largest coherent region, and naturally caps
        # over-inclusion via the z-FWER filter.
        llr_gain = np.nan_to_num(self.llr.astype(float), nan=0.0,
                                 posinf=0.0, neginf=0.0)
        reg_out_list, _ = prune_greedy(
            sig_reg_list=sig_reg_list,
            children=self.children,
            stat=llr_gain)

        self.effect_list = []
        for reg_idx in reg_out_list:
            label_map = glow.graph.get_label_map(
                reg_idx_list=[reg_idx],
                mask_idx=exp.mask_idx,
                children=self.children)
            eff = glow.effect.EffectEstimate.from_exp_mask(
                mask=label_map > -1, exp=exp,
                reg_idx=reg_idx, pval_fwer=self.fwer.pval[reg_idx])
            self.effect_list.append(eff)

        if verbose:
            n_disc = len(self.effect_list)
            n_pruned = len(sig_reg_list) - n_disc
            print(f'  done: {n_disc} discovered, {n_pruned} pruned')


def _run_outer(exp, k: int, *, q0, q1, n_perm_inner: int, min_vox: int,
               cluster_mode: ClusterMode, gpu_config=None,
               keep_stat: bool = False):
    """Run one outer permutation: cluster it, then z-score it against itself.

    The unit of AnalysisGLOW.fit's outer loop, and everything a joblib
    worker needs of it -- pure, carrying no instance state, a deterministic
    function of k alone (exp.permute(k) is seeded by k).

    The inner draws are one call to the shared draw backends against this
    outer perm's own tree, base_seed=0, so their row 0 is the identity draw
    of the permuted data -- this perm's observed LLR -- and the moments over
    the whole matrix are what standardizes it. The same procedure runs for
    every k, the observed k=0 included, which is what keeps the per-perm
    max-z statistics exchangeable.

    Args:
        exp (Experiment): the scaled experiment, unpermuted.
        k (int): outer-perm index; 0 is the observed data.
        q0 (np.array): (a0, num_img) nuisance subspace.
        q1 (np.array): (a1, num_img) interest subspace.
        n_perm_inner (int): inner FL draws standardizing this tree; the
            call runs n_perm_inner + 1 rows counting the observed.
        min_vox (int): regions smaller than this are left NaN and sit out
            of the max.
        cluster_mode (ClusterMode): Ward projection mode.
        gpu_config (GpuConfig | None): device knobs, None for the CPU.
        keep_stat (bool): materialize the k=0 inner matrix and return it.

    Returns:
        max_z (float): max z over this tree's regions of size >= min_vox,
            NaN where none qualify -- outer perm k's entry in the FWER null.
        obs (tuple | None): (children, size, summary, stat) for k == 0,
            None otherwise. Only the observed tree's arrays are reported, so
            a worker returning them for every k would ship one
            (num_reg - num_vox, 2) tree per perm back for nothing.
    """
    exp_k = exp.permute(k) if k else exp
    children = cluster(exp_k, mode=cluster_mode)

    _, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=exp_k.y.shape[2])
    size = region_h - region_l

    # size >= min_vox is a function of this outer perm's own tree, so the
    # comparison set is drawn afresh per perm -- unlike the split arm, where
    # one tree fixes it for the whole fit.
    reg_active = size >= min_vox

    draws_kwargs = dict(exp=exp_k, base_seed=0, n_perm=n_perm_inner + 1,
                        q0=q0, q1=q1, children=children, min_vox=min_vox)

    # The matrix is formed only for the observed tree under keep_stat, which
    # is the only one a reader can ask about afterwards; every other perm
    # streams and holds no more than one chunk. Summarizing what was kept,
    # rather than keeping a second copy alongside a streamed summary, is what
    # makes .stat the very matrix llr/mu/std came out of.
    stat = None
    if keep_stat and not k:
        stat = (draws.cpu_batched(**draws_kwargs) if gpu_config is None
                else gpu_draws(gpu_config, **draws_kwargs))
        summary = draws.summarize_draws(stat, reg_active=reg_active)
    elif gpu_config is None:
        summary = draws.cpu_summary(reg_active=reg_active, **draws_kwargs)
    else:
        summary = gpu_summary(gpu_config, reg_active=reg_active,
                              **draws_kwargs)

    # entry 0 of a DrawSummary's max_stat is the observed draw's max z over
    # the comparison set -- here the identity draw of exp_k, so this outer
    # perm's own max-z, which is its entry in the FWER null.
    max_z = float(summary.max_stat[0])
    if k:
        return max_z, None
    return max_z, (children, size, summary, stat)


class AnalysisGLOW(AnalysisGLOWBase):
    """Segment every permutation, standardize each tree against itself.

    Each outer permutation gets its own Ward tree, built on that
    permutation's images, and its regions are z-scored against inner
    Freedman-Lane draws (Freedman & Lane 1983) of the same permuted data.
    The max z over each outer perm's regions is one entry of the FWER null;
    the observed data is outer perm 0, so it enters that null on the same
    footing as the rest (Westfall & Young 1993; Phipson & Smyth 2010).

    Every image reaches both jobs, and that is the whole reason to run this
    arm: the tree is built from the full sample rather than a fold of it,
    and Ward's tree is the family of hypotheses the fit can discover at all.
    AnalysisGLOWSplit gives up half the sample on each side of that trade.

    What it controls. The map data -> (tree, statistic) is applied
    identically to every outer perm, so the per-perm max-z values are
    exchangeable and the complete-null rejection rate is nominal (0.040-0.053
    measured; Lehmann & Romano Thm 15.2.1; Hemerik & Goeman 2018) -- weak
    FWER control.

    What it does not. Strong control fails, and the reason is structural
    rather than a matter of tuning: under a planted effect the observed tree
    is effect-bearing while every null tree is a pure-noise tree, so the
    observed draw is compared against a null drawn from easier data. In
    trials with a planted effect, 30-45% emitted a region that carried none
    of it. The inflation is real on both sides -- on white noise per-region z
    reaches 70-81 at regions of 1154-8353 voxels, ~1700x what the same
    statistic gives on a region fixed in advance -- and the asymmetry is that
    the inner null conditions on a tree the outer perms rebuild. A fit here
    is a segmentation to inspect and to compare, not a p-value to report; for
    that, run AnalysisGLOWSplit.

    Cost. n_perm_fwer * (n_perm_inner + 1) draws and n_perm_fwer + 1 Ward
    builds against the split arm's n_perm_fwer + 1 draws and one build --
    ~250x the draws at the benchmark's 500 x 250. n_jobs splits the outer
    perms; gpu draws each perm's inner null on device.

    n_perm_inner is not merely a precision knob. The moments come from
    n_perm_inner + 1 samples, and a z over N samples cannot exceed
    (N - 1) / sqrt(N), so it caps every region's z at
    n_perm_inner / sqrt(n_perm_inner + 1): 3.0 at 10, 7.0 at 50, 14.1 at
    200. Once the observed tree and the null trees all sit on that ceiling
    the max-z test has nothing left to separate them -- measured p = 0.31
    at n_perm_inner = 10 against 0.0099 at 50 on the same planted data --
    so the count has to leave headroom above the z the effect reaches.

    The experiment is supplied to fit(), not stored (see Analysis). fit()
    scales it (ExperimentScaled.from_exp, idempotent) and decomposes its
    design; there is no fold, so an already-scaled experiment is accepted.

    Operation parameters (set at __init__), beyond AnalysisGLOWBase's:
        n_perm_inner (int): inner FL draws standardizing each outer perm's
            tree.

    Fit outputs: AnalysisGLOWBase's, all for the observed tree. mu and std
    are that tree's inner-null moments, and fwer.max_stat is
    (n_perm_fwer + 1,) -- one max-z per outer perm, entry 0 the observed.
    Under keep_stat, .stat is the observed tree's (n_perm_inner + 1,
    num_reg) inner matrix, the one its mu and std came out of; the other
    perms' matrices are never formed.
    """

    RECORD_FIELDS = ('n_perm_fwer', 'n_perm_inner', 'alpha_fwer', 'min_vox',
                     'cluster_mode')

    def __init__(self, n_perm_fwer: int, n_perm_inner: int = 500,
                 alpha_fwer: float = .05, min_vox: int = 1,
                 cluster_mode: ClusterMode = ClusterMode.FOCUS,
                 keep_stat: bool = False):
        """Configure a per-perm-segmentation GLOW analysis.

        Args:
            n_perm_fwer (int): outer FL perms feeding the max-z null; the
                fit runs n_perm_fwer + 1 counting the observed.
            n_perm_inner (int): inner FL draws per outer perm. Caps every
                z at n_perm_inner / sqrt(n_perm_inner + 1); see the class
                docstring before lowering it.
            alpha_fwer (float): family-wise error rate.
            min_vox (int): smallest region size admitted to the FWER set.
            cluster_mode (ClusterMode): Ward projection (see
                AnalysisGLOWBase).
            keep_stat (bool): keep the observed tree's inner draw matrix in
                .stat. A diagnostic that changes no result -- see the class
                docstring.
        """
        super().__init__(n_perm_fwer=n_perm_fwer, alpha_fwer=alpha_fwer,
                         min_vox=min_vox, cluster_mode=cluster_mode,
                         keep_stat=keep_stat)
        self.n_perm_inner = n_perm_inner

    def fit(self, exp, *, n_jobs: int = 1, gpu=False,
            verbose: bool = False):
        """Run the analysis on exp and return self.

        Populates the observed tree's attributes (children, size, llr, mu,
        std), the max-stat test they feed (fwer), and the synthesis output
        (effect_list).

        Args:
            exp (Experiment): experiment to analyze. Scaled on the way in;
                an already-scaled one passes through.
            n_jobs (int): joblib workers over the outer perms, capped at the
                machine's core count (resolve_n_jobs). The result is
                identical at any n_jobs -- outer perm k is seeded by k --
                and it is ignored on device, where one worker per perm would
                build a CUDA context each and then queue on the one device
                anyway.
            gpu: False (default) to draw on the CPU, True or a GpuConfig to
                require a device, 'auto' to take one when visible. The inner
                draws are the only thing the device changes; see ._fit_gpu
                on why float64 is its default dtype.
            verbose (bool): print progress and show a tqdm bar.

        Returns:
            self
        """
        gpu_config = resolve_gpu(gpu, name='AnalysisGLOW.fit')
        n_jobs = resolve_n_jobs(n_jobs)

        exp = ExperimentScaled.from_exp(exp)
        q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)

        n_total = self.n_perm_fwer + 1
        max_z_null = np.empty(n_total)

        if verbose:
            print(f'  [1/2] {n_total} outer perms x '
                  f'{self.n_perm_inner + 1} inner draws '
                  f'({exp.y.shape[1]} images, {exp.y.shape[2]} voxels, '
                  f'n_jobs={n_jobs}) ...')

        outer_kwargs = dict(q0=q0, q1=q1, n_perm_inner=self.n_perm_inner,
                            min_vox=self.min_vox,
                            cluster_mode=self.cluster_mode,
                            gpu_config=gpu_config, keep_stat=self.keep_stat)
        results = Parallel(n_jobs=1 if gpu_config else n_jobs,
                           return_as='generator')(
            delayed(_run_outer)(exp, k, **outer_kwargs)
            for k in range(n_total))

        summary = None
        for k, (max_z, obs) in enumerate(
                tqdm(results, total=n_total, desc='outer perms',
                     disable=not verbose)):
            max_z_null[k] = max_z
            if obs is not None:
                self.children, self.size, summary, self.stat = obs

        self.llr = summary.llr
        self.mu = summary.mu
        self.std = summary.std

        self.fwer = MaxStatPerm.from_max(
            summary.z_obs, max_z_null, alpha=self.alpha_fwer,
            reg_active=self.size >= self.min_vox)

        if verbose:
            print('  [2/2] FWER synthesis ...')
        self._discover(exp, verbose=verbose)

        return self
