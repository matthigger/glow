"""GLOW analysis: split-fold Ward-tree effect discovery with FWER control."""

import numpy as np

import glow.effect
import glow.graph
from glow.experiment.exper import ExperimentScaled
from ._base import Analysis
from . import inner_perm
from ._fit_gpu import gpu_draws, resolve_gpu
from .cluster import cluster, ClusterMode
from .fwer import MaxStatPerm
from .mancova import decompose
from .prune import prune_greedy


class AnalysisGLOW(Analysis):
    """Hierarchical-segmentation search for significant effects.

    One Ward tree, built on a held-out segmentation fold, is the whole
    hypothesis family: split_img partitions the images, the tree comes
    from fold A, and every statistic is computed on fold B.

    That split is what makes the procedure valid, and fixing the tree is
    not. A tree chosen with the same images that then test it gives the
    observed draw an advantage no permuted draw can have (per-region z
    reaches 70+ on white noise, ~1700x the fixed-region value), and
    holding such a tree across the permutations does not repair that --
    it is a function of the observed data, so the observed draw is
    privileged. Here the tree is a function of fold A alone, hence a
    constant with respect to the fold-B permutation group: the observed
    draw is exchangeable with the permuted draws, subset pivotality holds
    (each region's E and H are built from its own voxels, so an effect
    elsewhere cannot shift a signal-free region's null), and the
    family-wise bound is exact (Westfall & Young 1993; Lehmann & Romano
    Thm 15.2.1; Hemerik & Goeman 2018).

    With the family fixed there is no outer loop that re-clusters and no
    nested inner null. The whole fit is one (n_perm_fwer + 1, num_reg)
    matrix of Freedman-Lane draws (Freedman & Lane 1983) against that one
    tree, row i being exp_test.permute(i). Row 0 is the identity, so it
    is the observed draw and rows 1: are the null. That single matrix
    serves both jobs: its column moments standardize the regions onto a
    common scale, and its row maxima are the max-z null.

    The observed row contributes to those moments on equal footing with
    the permuted rows, which is required rather than merely tidy -- see
    Analysis.z_score_stat, whose docstring records what happens when it
    does not (Phipson & Smyth 2010; Winkler et al. 2014).

    The experiment is supplied to fit(), not stored (see Analysis). It
    must be a raw Experiment: fit() splits it first and scales each fold
    separately, because ExperimentScaled fits pre_scale on every image at
    once and a fold cut out afterwards would carry a transform the other
    fold helped choose (ExperimentScaled.split_img refuses for the same
    reason).

    Operation parameters (set at __init__):
        n_perm_fwer (int): FL draws in the null; the fit runs
            n_perm_fwer + 1 rows counting the observed.
        alpha_fwer (float): family-wise error rate
        min_vox (int): smallest region size admitted to the FWER set.
            Must be pre-specified, never tuned against results, or it
            reintroduces selection one level up.
        cluster_mode (ClusterMode): Ward projection mode
        frac_segment (float): share of the images going to the
            segmentation fold. Pre-specified, for the same reason.
        split_seed (int): seed for the image partition.

    The draw matrix itself is not kept: at full-brain num_vox it runs to
    gigabytes, and only its first row, its column moments and its row
    maxima outlive it. Each is copied out, since a row sliced from it is
    a view and would hold the whole matrix alive.

    Fit outputs (all on the test fold, against the fold-A tree):
        children (np.array): (num_reg - num_vox, 2) Ward tree.
        size (np.array): (num_reg,) region sizes.
        llr (np.array): (num_reg,) observed LLR per region -- the
            matrix's row 0.
        mu (np.array): (num_reg,) per-region mean over the draws.
        std (np.array): (num_reg,) per-region std over the draws.
        fwer (MaxStatPerm): the max-z test -- observed z per region
            in stat_obs, max-z per draw in max_stat, the size >= min_vox
            comparison set, the p-values and the selected regions.
        effect_list (list): discovered EffectEstimate objects.
    """

    RECORD_FIELDS = ('n_perm_fwer', 'alpha_fwer', 'min_vox', 'cluster_mode',
                     'frac_segment', 'split_seed')

    def __init__(self, n_perm_fwer: int, alpha_fwer: float = .05,
                 min_vox: int = 1,
                 cluster_mode: ClusterMode = ClusterMode.FOCUS,
                 frac_segment: float = .5, split_seed: int = 0):
        """Configure a GLOW analysis.

        Args:
            n_perm_fwer (int): FL draws in the null.
            alpha_fwer (float): family-wise error rate.
            min_vox (int): smallest region size admitted to the FWER set.
            cluster_mode (ClusterMode): Ward projection. Default
                ClusterMode.FOCUS projects onto the contrast subspace;
                ClusterMode.GLM_ERROR keeps bias + contrast;
                ClusterMode.NAIVE clusters raw y.
            frac_segment (float): share of the images used to build the
                Ward tree; the rest carry every statistic.
            split_seed (int): seed for the image partition.
        """
        super().__init__()
        self.n_perm_fwer = n_perm_fwer
        self.alpha_fwer = alpha_fwer
        self.min_vox = min_vox
        self.cluster_mode = cluster_mode
        self.frac_segment = frac_segment
        self.split_seed = split_seed

        self.children = None
        self.size = None
        self.llr = None
        self.mu = None
        self.std = None

    def fit(self, exp, *, n_jobs: int = 1, gpu=False, split_group=None,
            verbose: bool = False):
        """Run the analysis on exp and return self.

        Populates the observed attributes (children, size, llr, mu, std),
        the max-stat test they feed (fwer), and the synthesis output
        (effect_list).

        Args:
            exp (Experiment): experiment to analyze. Must be raw, not an
                ExperimentScaled: fit splits it into folds and scales
                each fold separately.
            n_jobs (int): accepted for the execution contract every
                recipe honours (see Analysis.fit) but unused -- the draws
                come from one serial call. Parallelising them is a
                straight win and simply has not been done yet.
            gpu: False (default) to draw on the CPU, True or a GpuConfig
                to require a device, 'auto' to take one when visible. The
                draws are the only thing the device changes; see
                ._fit_gpu on why float64 is its default dtype.
            split_group (np.array): (num_img,) labels held together by
                the split -- family IDs, subject IDs for repeat scans.
                Omitting it when the images are related leaves the two
                folds dependent, which voids the FWER argument.
            verbose (bool): print progress.

        Returns:
            self
        """
        gpu_config = resolve_gpu(gpu, name='AnalysisGLOW.fit')
        del n_jobs

        exp_seg, exp_test = exp.split_img(frac_segment=self.frac_segment,
                                          seed=self.split_seed,
                                          group=split_group)

        # Each fold is scaled on its own images. Fold A's scaling is the
        # one that matters: Ward distances are not invariant to a map on
        # the b axis, so pre_scale reaches the tree. Fold B's does not --
        # every MANCOVA statistic is exactly invariant to an invertible
        # b x b map on y -- and is here only for the conditioning of the
        # slogdet in get_llr.
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

        # base_seed=0 makes draw i exp_test.permute(i), and seed 0 is the
        # identity -- so row 0 is the observed draw and needs no separate
        # code path (permute._perm_indices reserves it).
        #
        # The two backends differ only in speed: the CPU one is the
        # trust anchor, an independent per-region implementation of the
        # statistic (see inner_perm), while the device one is ~3 orders
        # faster at full-brain num_vox and holds float64 to reproduce it.
        draws_kwargs = dict(
            exp=exp_test, base_seed=0, n_perm=self.n_perm_fwer + 1,
            q0=q0, q1=q1, children=self.children, min_vox=self.min_vox)
        llr_all = (inner_perm.cpu_reliable_full(**draws_kwargs)
                   if gpu_config is None
                   else gpu_draws(gpu_config, **draws_kwargs))

        # One matrix, both jobs: column moments standardize the regions,
        # row maxima are the null. The observed row is inside both.
        z, self.mu, self.std = self.z_score_stat(llr_all)

        # Copy, never slice: llr_all[0] is a view, so keeping it would
        # hold the whole (n_perm_fwer + 1, num_reg) matrix -- gigabytes at
        # full-brain num_vox -- alive for one row of it. from_stat copies
        # the row it keeps for the same reason.
        self.llr = llr_all[0].copy()
        del llr_all

        # size >= min_vox is a function of the fold-A tree alone, hence a
        # constant with respect to the fold-B permutations -- the
        # condition MaxStatPerm needs of a comparison set.
        self.fwer = MaxStatPerm.from_stat(
            z, alpha=self.alpha_fwer, reg_active=self.size >= self.min_vox)
        del z

        if verbose:
            print('  [2/2] FWER synthesis ...')

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

        # Effects are estimated on the test fold, not on the whole cohort:
        # the segmentation fold chose the regions, so only fold B gives an
        # estimate that the region's own selection did not shape.
        self.effect_list = []
        for reg_idx in reg_out_list:
            label_map = glow.graph.get_label_map(
                reg_idx_list=[reg_idx],
                mask_idx=exp_test.mask_idx,
                children=self.children)
            eff = glow.effect.EffectEstimate.from_exp_mask(
                mask=label_map > -1, exp=exp_test,
                reg_idx=reg_idx, pval_fwer=self.fwer.pval[reg_idx])
            self.effect_list.append(eff)

        if verbose:
            n_disc = len(self.effect_list)
            n_pruned = len(sig_reg_list) - n_disc
            print(f'  done: {n_disc} discovered, {n_pruned} pruned')

        return self
