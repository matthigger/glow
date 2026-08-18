"""Split-fold GLOW: one Ward tree, on a fold the statistics never see.

The arm with strong FWER control, and the one to report. Its counterpart,
AnalysisGLOW in glow.analysis._glow, segments inside every permutation
instead.
"""

import glow.graph
from glow.experiment.exper import ExperimentScaled
from . import draws
from ._fit_gpu import (describe_backend, gpu_draws, gpu_summary,
                       resolve_gpu)
from ._glow import AnalysisGLOWBase
from .cluster import cluster, ClusterMode
from .fwer import MaxStatPerm
from .mancova import decompose


class AnalysisGLOWSplit(AnalysisGLOWBase):
    """Build one Ward tree on a held-out fold, test on the other.

    split_img partitions the images, the tree comes from fold A, and every
    statistic is computed on fold B. The whole fit is one
    (n_perm_fwer + 1, num_reg) matrix of Freedman-Lane draws (Freedman &
    Lane 1983) against that tree, row i being exp_test.permute(i) and row 0
    the identity, hence the observed draw: its column moments standardize
    the regions onto a common scale, its row maxima are the max-z null.

    The fold-A tree is constant with respect to the fold-B permutations, so
    the observed draw is exchangeable with the permuted ones and the
    family-wise bound is exact (Westfall & Young 1993; Hemerik & Goeman
    2018).

    Operation parameters (set at __init__), beyond AnalysisGLOWBase's:
        frac_segment (float): share of the images going to the
            segmentation fold. Pre-specify it, as with min_vox.
        split_seed (int): seed for the image partition.

    Fit outputs: AnalysisGLOWBase's, all on the test fold and against the
    fold-A tree, plus:
        img_segment (np.array): (num_img,) boolean, True for the images the
            tree was built on. The realized partition, not recoverable from
            frac_segment and split_seed once split_group is passed.

    llr, mu and std are that matrix's row 0 and column moments,
    fwer.max_stat its (n_perm_fwer + 1,) row maxima. keep_stat keeps the
    matrix itself, so a column is the null a region's z was measured
    against; it changes no result (std moves at fp64 round-off), hence its
    absence from RECORD_FIELDS.
    """

    RECORD_FIELDS = ('n_perm_fwer', 'alpha_fwer', 'min_vox', 'cluster_mode',
                     'frac_segment', 'split_seed', 'prune_rule', 'prune_lam',
                     'prune_exp_n_eff')

    def __init__(self, n_perm_fwer: int, alpha_fwer: float = .05,
                 min_vox: int = 1,
                 cluster_mode: ClusterMode = ClusterMode.FOCUS,
                 frac_segment: float = .5, split_seed: int = 0,
                 prune_rule: str = 'greedy', prune_lam: float = 0.0,
                 prune_exp_n_eff: float = None,
                 keep_stat: bool = False):
        """Configure a split-fold GLOW analysis.

        Args:
            n_perm_fwer (int): FL draws in the null.
            alpha_fwer (float): family-wise error rate.
            min_vox (int): smallest region size admitted to the FWER set.
            cluster_mode (ClusterMode): Ward projection (see
                AnalysisGLOWBase).
            frac_segment (float): share of the images used to build the
                Ward tree; the rest carry every statistic.
            split_seed (int): seed for the image partition.
            prune_rule (str): selection rule (see AnalysisGLOWBase).
            prune_lam (float): dp-only per-region penalty (see
                AnalysisGLOWBase).
            prune_exp_n_eff (float | None): dp-only region count (see
                AnalysisGLOWBase).
            keep_stat (bool): keep the whole (n_perm_fwer + 1, num_reg)
                draw matrix in .stat instead of discarding it.
        """
        super().__init__(n_perm_fwer=n_perm_fwer, alpha_fwer=alpha_fwer,
                         min_vox=min_vox, cluster_mode=cluster_mode,
                         prune_rule=prune_rule, prune_lam=prune_lam,
                         prune_exp_n_eff=prune_exp_n_eff,
                         keep_stat=keep_stat)
        self.frac_segment = frac_segment
        self.split_seed = split_seed

        self.img_segment = None

    def fit(self, exp, *, n_jobs: int = 1, gpu=False,
            cpu_anchor: bool = False, split_group=None,
            verbose: bool = False):
        """Run the analysis on exp and return self.

        Populates the observed attributes (children, size, llr, mu, std),
        the max-stat test they feed (fwer), and effect_list.

        Args:
            exp (Experiment): experiment to analyze. Must be raw, not an
                ExperimentScaled: fit splits it into folds and scales
                each fold separately.
            n_jobs (int): accepted for the execution contract every recipe
                honours (see Analysis.fit) but unused -- the draws come
                from one serial call.
            gpu: False (default) to draw on the CPU, True or a GpuConfig
                to require a device, 'auto' to take one when visible. The
                draws are the only thing the device changes; see
                ._fit_gpu on why float64 is its default dtype.
            cpu_anchor (bool): draw through draws.cpu_reliable, the slow
                trust anchor, instead of the batched draws.cpu_summary.
                Same answer to fp64 round-off at ~35x the cost, so this is
                for holding the fast path honest, not for fitting. Forces
                the CPU, and contradicts an explicit gpu=True / GpuConfig.
            split_group (np.array): (num_img,) labels held together by
                the split -- family IDs, subject IDs for repeat scans.
                Omitting it when the images are related leaves the two
                folds dependent, which voids the FWER argument.
            verbose (bool): print progress.

        Returns:
            self
        """
        # cpu_anchor names a CPU backend, so a device request alongside it
        # is a contradiction rather than a preference; 'auto' is only a
        # preference, and the anchor is the more specific instruction
        if cpu_anchor and gpu and gpu != 'auto':
            raise ValueError(
                'AnalysisGLOWSplit.fit(cpu_anchor=True) draws on the CPU '
                f'and cannot also honour gpu={gpu!r}')
        gpu_config = None if cpu_anchor else resolve_gpu(
            gpu, name='AnalysisGLOWSplit.fit')
        del n_jobs

        split_kwargs = dict(frac_segment=self.frac_segment,
                            seed=self.split_seed, group=split_group)
        # kept because a reader of the fit cannot redraw it: with a
        # split_group the partition depends on labels the fit does not store
        self.img_segment = exp.get_img_segment(**split_kwargs)
        exp_seg, exp_test = exp.split_img(**split_kwargs)

        # Each fold is scaled on its own images. Fold A's scaling reaches
        # the tree, Ward distances not being invariant to a map on the b
        # axis. Fold B's does not -- every MANCOVA statistic is exactly
        # invariant to an invertible b x b map on y -- and is here only for
        # the conditioning of the slogdet in get_llr.
        self.children = cluster(ExperimentScaled.from_exp(exp_seg),
                                mode=self.cluster_mode)
        exp_test = ExperimentScaled.from_exp(exp_test)
        q0, q1, _ = decompose(x=exp_test.x, contrast=exp_test.contrast)

        _, region_l, region_h = glow.graph.build_dfs_preorder(
            children=self.children, num_vox=exp_test.y.shape[2])
        self.size = region_h - region_l

        if verbose:
            print(f'  [1/2] {self.n_perm_fwer + 1} draws '
                  f'({exp_test.y.shape[1]} test images, '
                  f'{exp_test.y.shape[2]} voxels) ...')
            # the backends differ by orders of magnitude and gpu='auto'
            # falls back silently, so say which one ran
            backend = describe_backend(gpu, gpu_config,
                                       cpu_anchor=cpu_anchor)
            print(f'        backend: {backend}')

        # size >= min_vox is a function of the fold-A tree alone, hence a
        # constant with respect to the fold-B permutations -- the condition
        # MaxStatPerm needs of a comparison set
        reg_active = self.size >= self.min_vox

        # base_seed=0 makes draw i exp_test.permute(i), and seed 0 is the
        # identity -- so row 0 is the observed draw and needs no separate
        # code path (permute._perm_indices reserves it). All three backends
        # return the same DrawSummary, differing in speed and peak memory
        # only.
        draws_kwargs = dict(
            exp=exp_test, base_seed=0, n_perm=self.n_perm_fwer + 1,
            q0=q0, q1=q1, children=self.children, min_vox=self.min_vox)

        # The matrix is formed only when something needs it whole: the
        # anchor has no streaming form, and keep_stat asks for it outright.
        # Summarizing what was kept, rather than streaming a second copy,
        # is what makes .stat the very matrix llr/mu/std came out of.
        if cpu_anchor:
            matrix = draws.cpu_reliable(**draws_kwargs)
        elif not self.keep_stat:
            matrix = None
        elif gpu_config is None:
            matrix = draws.cpu_batched(**draws_kwargs)
        else:
            matrix = gpu_draws(gpu_config, **draws_kwargs)

        if matrix is not None:
            summary = draws.summarize_draws(matrix, reg_active=reg_active)
        elif gpu_config is None:
            summary = draws.cpu_summary(reg_active=reg_active,
                                        **draws_kwargs)
        else:
            summary = gpu_summary(gpu_config, reg_active=reg_active,
                                  **draws_kwargs)

        self.stat = matrix if self.keep_stat else None

        self.llr = summary.llr
        self.mu = summary.mu
        self.std = summary.std

        self.fwer = MaxStatPerm.from_max(
            summary.z_obs, summary.max_stat, alpha=self.alpha_fwer,
            reg_active=reg_active)

        if verbose:
            print('  [2/2] FWER synthesis ...')

        # effects are estimated on the test fold: the segmentation fold
        # chose the regions, so only fold B gives an estimate the region's
        # own selection did not shape
        self._discover(exp_test, verbose=verbose)

        return self
