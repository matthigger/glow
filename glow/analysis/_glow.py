"""GLOW analysis: per-permutation Ward segmentation with FWER control.

The Ward tree is the whole family of hypotheses, and the two arms here
differ in where it comes from:

  - AnalysisGLOW rebuilds it inside every outer permutation and
    standardizes its regions against inner Freedman-Lane draws of the same
    permuted data. Every image reaches both the tree and the statistics;
    weak FWER control only.
  - AnalysisGLOWSplit (glow.analysis._glow_split) builds one tree on a
    held-out fold of the images, needs no inner null, and controls FWER
    strongly.

AnalysisGLOWBase holds what they share: the recipe knobs, the per-region
fit outputs, and the prune-then-estimate synthesis.
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
from .prune import check_prune_rule, prune_by_rule


class AnalysisGLOWBase(Analysis):
    """Ward-tree effect discovery: the parts both GLOW arms share.

    Not fit directly; fit() belongs to the arms, which differ in where the
    tree comes from and what its regions are standardized against.

    Operation parameters (set at __init__):
        n_perm_fwer (int): draws in the FWER null.
        alpha_fwer (float): family-wise error rate.
        min_vox (int): smallest region size admitted to the FWER set.
            Pre-specify it; tuning it against results reintroduces
            selection one level up.
        cluster_mode (ClusterMode): Ward projection mode.
        prune_rule (str): the rule that turns the significant regions into
            disjoint effects, one of prune.PRUNE_RULE_LIST.
        prune_lam (float): prune_dp's per-region penalty; the dp rule only.
        prune_exp_n_eff (float | None): prune_dp's expected region count
            under a geometric prior; the dp rule only, and it sets the
            penalty in place of prune_lam.
        keep_stat (bool): keep a draw matrix in .stat. Which matrix that is
            differs by arm; neither changes a reported result.

    Fit outputs (on the observed data, against the observed tree):
        children (np.array): (num_reg - num_vox, 2) Ward tree.
        size (np.array): (num_reg,) region sizes.
        llr (np.array): (num_reg,) observed LLR per region.
        mu (np.array): (num_reg,) per-region mean over the draws.
        std (np.array): (num_reg,) per-region std over the draws.
        fwer (MaxStatPerm): the max-z test -- observed z per region, the
            max-z null, the size >= min_vox comparison set, the p-values
            and the selected regions.
        effect_list (list): discovered EffectEstimate objects.
        stat (np.array): (n_perm + 1, num_reg) per-region LLR draws, row 0
            the observed one -- only under keep_stat, None otherwise.
    """

    def __init__(self, n_perm_fwer: int, alpha_fwer: float = .05,
                 min_vox: int = 1,
                 cluster_mode: ClusterMode = ClusterMode.FOCUS,
                 prune_rule: str = 'greedy', prune_lam: float = 0.0,
                 prune_exp_n_eff: float = None,
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
            prune_rule (str): selection rule, one of
                prune.PRUNE_RULE_LIST. Default 'greedy'.
            prune_lam (float): per-region penalty handed to prune_dp;
                valid only with prune_rule='dp'.
            prune_exp_n_eff (float | None): expected region count handed to
                prune_dp, which turns it into the penalty; valid only with
                prune_rule='dp', and an alternative to prune_lam.
            keep_stat (bool): keep a draw matrix in .stat instead of
                discarding it.

        Raises:
            ValueError: the pruning arguments do not go together (see
                prune.check_prune_rule).
        """
        super().__init__()
        check_prune_rule(prune_rule, lam=prune_lam,
                         exp_n_eff=prune_exp_n_eff)
        self.n_perm_fwer = n_perm_fwer
        self.alpha_fwer = alpha_fwer
        self.min_vox = min_vox
        self.cluster_mode = cluster_mode
        self.prune_rule = prune_rule
        self.prune_lam = prune_lam
        self.prune_exp_n_eff = prune_exp_n_eff
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
        bundle, deepcopy -- reaches the object through here. The matrix is
        still written: dropping it would leave a bundle the viewer's
        PERMUTATION panel cannot open.
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
        arms, so the pruning rule and the p-value each effect carries
        cannot drift between them.

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

        # Rank candidates by raw LLR, not z.  z is right for FWER
        # thresholding, putting different-size regions on a common scale,
        # but it scales as ~sqrt(n): a slice of an effect can outscore the
        # whole region, and greedy pruning then emits fragments.  Raw LLR
        # scales linearly with n, so it picks the largest coherent region.
        llr_gain = np.nan_to_num(self.llr.astype(float), nan=0.0,
                                 posinf=0.0, neginf=0.0)
        reg_out_list, _ = prune_by_rule(
            self.prune_rule,
            sig_reg_list=sig_reg_list,
            children=self.children,
            stat=llr_gain,
            lam=self.prune_lam,
            exp_n_eff=self.prune_exp_n_eff)

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


def _cluster_tree(exp_k, cluster_mode: ClusterMode):
    """Ward-cluster one permuted experiment and size its regions.

    Args:
        exp_k (Experiment): the permuted (or observed) experiment.
        cluster_mode (ClusterMode): Ward projection mode.

    Returns:
        children (np.array): (num_reg - num_vox, 2) Ward tree.
        size (np.array): (num_reg,) region voxel counts.
    """
    children = cluster(exp_k, mode=cluster_mode)
    _, region_l, region_h = glow.graph.build_dfs_preorder(
        children=children, num_vox=exp_k.y.shape[2])
    return children, region_h - region_l


def _cluster_outer(exp, k: int, *, cluster_mode: ClusterMode):
    """Build outer permutation k's Ward tree: the CPU half of a perm.

    The device pipeline's producer (see AnalysisGLOW.fit). Only the tree
    and its region sizes come back, hundreds of KB where the permuted y
    behind them is tens of MB, so a worker pool feeding one device stream
    does not pay to ship experiments; the consumer re-derives
    exp.permute(k), which is far cheaper than that transfer.

    Args:
        exp (Experiment): the scaled experiment, unpermuted.
        k (int): outer-perm index; 0 is the observed data.
        cluster_mode (ClusterMode): Ward projection mode.

    Returns:
        k (int): the index, so an unordered generator still keys results.
        children (np.array): (num_reg - num_vox, 2) this perm's Ward tree.
        size (np.array): (num_reg,) region voxel counts.
    """
    exp_k = exp.permute(k) if k else exp
    children, size = _cluster_tree(exp_k, cluster_mode)
    return k, children, size


def _draw_outer(exp_k, k: int, children, size, *, q0, q1,
                n_perm_inner: int, min_vox: int, gpu_config=None,
                keep_stat: bool = False):
    """Standardize one outer perm's tree against its own inner draws.

    The device pipeline's consumer, and the second half of _run_outer.
    The inner draws run against this perm's own tree with base_seed=0, so
    their row 0 is the identity draw -- this perm's observed LLR -- and the
    moments over the whole matrix standardize it.

    Args:
        exp_k (Experiment): the permuted (or observed) experiment whose
            tree this is.
        k (int): outer-perm index; 0 is the observed data.
        children (np.array): (num_reg - num_vox, 2) this perm's Ward tree.
        size (np.array): (num_reg,) region voxel counts.
        q0 (np.array): (a0, num_img) nuisance subspace.
        q1 (np.array): (a1, num_img) interest subspace.
        n_perm_inner (int): inner FL draws standardizing this tree; the
            call runs n_perm_inner + 1 rows counting the observed.
        min_vox (int): regions smaller than this are left NaN and sit out
            of the max.
        gpu_config (GpuConfig | None): device knobs, None for the CPU.
        keep_stat (bool): materialize the k=0 inner matrix and return it.

    Returns:
        max_z (float): max z over this tree's regions of size >= min_vox,
            NaN where none qualify -- outer perm k's entry in the FWER null.
        obs (tuple | None): (children, size, summary, stat) for k == 0,
            None otherwise -- only the observed tree's arrays are reported.
    """
    # the comparison set is drawn afresh per perm: size >= min_vox is a
    # function of this perm's own tree, not of one tree fixed for the fit
    reg_active = size >= min_vox

    draws_kwargs = dict(exp=exp_k, base_seed=0, n_perm=n_perm_inner + 1,
                        q0=q0, q1=q1, children=children, min_vox=min_vox)

    # Only the observed tree's matrix can be asked about after the fit, so
    # it is the only one formed; every other perm streams a chunk at a
    # time. Summarizing what was kept is what makes .stat the very matrix
    # llr/mu/std came out of.
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

    # a DrawSummary's max_stat[0] is the observed draw's max z over the
    # comparison set -- here the identity draw of exp_k, so this outer
    # perm's own max-z, its entry in the FWER null
    max_z = float(summary.max_stat[0])
    if k:
        return max_z, None
    return max_z, (children, size, summary, stat)


def _run_outer(exp, k: int, *, q0, q1, n_perm_inner: int, min_vox: int,
               cluster_mode: ClusterMode, gpu_config=None,
               keep_stat: bool = False):
    """Run one outer permutation: cluster it, then z-score it against itself.

    The unit of AnalysisGLOW.fit's CPU outer loop, carrying no instance
    state and a deterministic function of k alone (exp.permute(k) is seeded
    by k). Every k is treated the same way, k=0 included, which keeps the
    per-perm max-z values exchangeable. The device path runs the same two
    halves split across a pool and a stream (_cluster_outer, _draw_outer),
    and both orderings give the same numbers.

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
        max_z (float): see _draw_outer.
        obs (tuple | None): see _draw_outer.
    """
    exp_k = exp.permute(k) if k else exp
    children, size = _cluster_tree(exp_k, cluster_mode)
    return _draw_outer(exp_k, k, children, size, q0=q0, q1=q1,
                       n_perm_inner=n_perm_inner, min_vox=min_vox,
                       gpu_config=gpu_config, keep_stat=keep_stat)


class AnalysisGLOW(AnalysisGLOWBase):
    """Segment every permutation, standardize each tree against itself.

    Each outer perm gets its own Ward tree, built on that permutation's
    images, and its regions are z-scored against inner Freedman-Lane draws
    (Freedman & Lane 1983) of the same permuted data. Max z per outer perm
    is one entry of the FWER null, the observed data being perm 0 (Westfall
    & Young 1993; Phipson & Smyth 2010).

    Weak FWER control only: the observed tree bears the effect while every
    null tree is noise, so the observed draw meets an easier null. A fit
    here is a segmentation to inspect and compare; for a reported p-value
    use AnalysisGLOWSplit.

    Operation parameters (set at __init__), beyond AnalysisGLOWBase's:
        n_perm_inner (int): inner FL draws standardizing each outer perm's
            tree. Caps every region's z at
            n_perm_inner / sqrt(n_perm_inner + 1), so leave headroom above
            the z the effect reaches.

    Fit outputs: AnalysisGLOWBase's, all for the observed tree. mu and std
    are that tree's inner-null moments, and fwer.max_stat is
    (n_perm_fwer + 1,), one max-z per outer perm. Under keep_stat, .stat is
    the observed tree's (n_perm_inner + 1, num_reg) inner matrix; the other
    perms' matrices are never formed.
    """

    RECORD_FIELDS = ('n_perm_fwer', 'n_perm_inner', 'alpha_fwer', 'min_vox',
                     'cluster_mode', 'prune_rule', 'prune_lam',
                     'prune_exp_n_eff')

    def __init__(self, n_perm_fwer: int, n_perm_inner: int = 500,
                 alpha_fwer: float = .05, min_vox: int = 1,
                 cluster_mode: ClusterMode = ClusterMode.FOCUS,
                 prune_rule: str = 'greedy', prune_lam: float = 0.0,
                 prune_exp_n_eff: float = None,
                 keep_stat: bool = False):
        """Configure a per-perm-segmentation GLOW analysis.

        Args:
            n_perm_fwer (int): outer FL perms feeding the max-z null; the
                fit runs n_perm_fwer + 1 counting the observed.
            n_perm_inner (int): inner FL draws per outer perm. Caps every
                z; see the class docstring before lowering it.
            alpha_fwer (float): family-wise error rate.
            min_vox (int): smallest region size admitted to the FWER set.
            cluster_mode (ClusterMode): Ward projection (see
                AnalysisGLOWBase).
            prune_rule (str): selection rule (see AnalysisGLOWBase).
            prune_lam (float): dp-only per-region penalty (see
                AnalysisGLOWBase).
            prune_exp_n_eff (float | None): dp-only region count (see
                AnalysisGLOWBase).
            keep_stat (bool): keep the observed tree's inner draw matrix in
                .stat.
        """
        super().__init__(n_perm_fwer=n_perm_fwer, alpha_fwer=alpha_fwer,
                         min_vox=min_vox, cluster_mode=cluster_mode,
                         prune_rule=prune_rule, prune_lam=prune_lam,
                         prune_exp_n_eff=prune_exp_n_eff,
                         keep_stat=keep_stat)
        self.n_perm_inner = n_perm_inner

    def _pipeline_device(self, exp, n_total: int, *, n_jobs: int,
                         draw_kwargs: dict):
        """Yield (k, _draw_outer result), pool on Ward, one stream on device.

        The device pipeline. A fit alternates a single-threaded CPU stage
        (permute + Ward) with a device stage, so running the outer perms
        one at a time leaves whichever device is not busy idle for the
        whole of the other's turn. Here n_jobs workers build trees ahead
        of the device while this process draws against the trees already
        built, making the per-perm cost the larger of the two stages
        rather than their sum.

        One stream, not n_jobs of them: a worker touching the device would
        build its own CUDA context (hundreds of MB) and the calls would
        serialise on the one card regardless. Workers therefore return
        trees only, and the device work stays here.

        Args:
            exp (Experiment): the scaled experiment, unpermuted.
            n_total (int): outer perms to run, counting the observed.
            n_jobs (int): tree-building workers.
            draw_kwargs (dict): forwarded to _draw_outer.

        Yields:
            k (int): outer-perm index, since trees arrive out of order.
            result (tuple): that perm's (max_z, obs).
        """
        # dispatched before the first device call so the pool's processes
        # exist before this one holds a CUDA context
        trees = Parallel(n_jobs=n_jobs, return_as='generator_unordered')(
            delayed(_cluster_outer)(exp, k, cluster_mode=self.cluster_mode)
            for k in range(n_total))

        for k, children, size in trees:
            exp_k = exp.permute(k) if k else exp
            yield k, _draw_outer(exp_k, k, children, size, **draw_kwargs)

    def fit(self, exp, *, n_jobs: int = 1, gpu=False,
            verbose: bool = False):
        """Run the analysis on exp and return self.

        Populates the observed tree's attributes (children, size, llr, mu,
        std), the max-stat test they feed (fwer), and effect_list.

        Args:
            exp (Experiment): experiment to analyze. Scaled on the way in;
                an already-scaled one passes through.
            n_jobs (int): joblib workers, capped at the machine's core
                count (resolve_n_jobs). On the CPU each runs whole outer
                perms; on device they build Ward trees to feed the one
                stream (_pipeline_device). The result is identical at any
                n_jobs -- outer perm k is seeded by k.
            gpu: False (default) to draw on the CPU, True or a GpuConfig to
                require a device, 'auto' to take one when visible. The
                inner draws are the only thing the device changes; see
                ._fit_gpu on why float64 is its default dtype.
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

        draw_kwargs = dict(q0=q0, q1=q1, n_perm_inner=self.n_perm_inner,
                           min_vox=self.min_vox, gpu_config=gpu_config,
                           keep_stat=self.keep_stat)
        if gpu_config is None:
            results = Parallel(n_jobs=n_jobs, return_as='generator')(
                delayed(_run_outer)(exp, k, cluster_mode=self.cluster_mode,
                                    **draw_kwargs)
                for k in range(n_total))
            keyed = enumerate(results)
        else:
            keyed = self._pipeline_device(exp, n_total, n_jobs=n_jobs,
                                          draw_kwargs=draw_kwargs)

        summary = None
        for k, (max_z, obs) in tqdm(keyed, total=n_total,
                                    desc='outer perms', disable=not verbose):
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
