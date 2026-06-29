"""GLOW analysis: per-perm Ward-tree effect discovery with FWER control."""

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import glow.effect
import glow.graph
from glow.experiment.exper import ExperimentScaled
from ._base import Analysis
from . import inner_perm
from .cluster import cluster, ClusterMode
from .mancova import decompose
from .prune import prune_greedy


_INNER_SEED_BLOCK = 100_000


class AnalysisGLOW(Analysis):
    """Hierarchical-segmentation search for significant effects.

    Each outer perm gets its own Ward tree. Per-region z-scoring uses
    inner Freedman-Lane draws (Freedman & Lane 1983) against that tree;
    Westfall-Young FWER (Westfall & Young 1993) runs on the max-z null
    assembled across outer perms.

    Each outer perm runs the SAME inner Freedman-Lane procedure (including
    the observed k=0), so the per-perm max-z statistics stay exchangeable and
    the FWER bound is exact (Lehmann & Romano Thm 15.2.1; Hemerik & Goeman
    2018).

    The experiment is supplied to fit(), not stored (see Analysis); fit()
    scales it (ExperimentScaled.from_exp) and decomposes its design.

    Operation parameters (set at __init__):
        n_perm_fwer (int): outer FL permutations feeding the max-z null
        n_perm_inner (int): inner FL permutations per outer perm
        alpha_fwer (float): family-wise error rate
        min_vox (int): smallest region size admitted to the FWER set
        cluster_mode (ClusterMode): Ward projection mode

    Fit outputs (populated by fit, for the observed k=0 tree):
        children (np.array): (num_reg - num_vox, 2) Ward tree.
        size (np.array): (num_reg,) region sizes.
        llr (np.array): (num_reg,) observed LLR per region.
        mu (np.array): (num_reg,) inner-null mean per region.
        std (np.array): (num_reg,) inner-null std per region.
        z (np.array): (num_reg,) per-region z-score (llr - mu)/std.
        max_z_null (np.array): (n_perm_fwer + 1,) max-z per outer perm,
            indexed by outer-perm number; max_z_null[0] is the observed.
        pval (np.array): (num_reg,) FWER-controlled p-values.
        effect_list (list): discovered EffectEstimate objects.
    """

    RECORD_FIELDS = ('n_perm_fwer', 'n_perm_inner', 'alpha_fwer', 'min_vox',
                     'cluster_mode')

    def __init__(self, n_perm_fwer: int, n_perm_inner: int = 500,
                 alpha_fwer: float = .05, min_vox: int = 1,
                 cluster_mode: ClusterMode = ClusterMode.FOCUS):
        """Configure a GLOW analysis.

        Args:
            n_perm_fwer (int): outer FL permutations feeding the max-z null.
            n_perm_inner (int): inner FL permutations per outer perm.
            alpha_fwer (float): family-wise error rate.
            min_vox (int): smallest region size admitted to the FWER set.
            cluster_mode (ClusterMode): Ward projection. Default
                ClusterMode.FOCUS projects onto the contrast subspace;
                ClusterMode.GLM_ERROR keeps bias + contrast;
                ClusterMode.NAIVE clusters raw y.
        """
        super().__init__()
        self.n_perm_fwer = n_perm_fwer
        self.n_perm_inner = n_perm_inner
        self.alpha_fwer = alpha_fwer
        self.min_vox = min_vox
        self.cluster_mode = cluster_mode

        self.children = None
        self.size = None
        self.llr = None
        self.mu = None
        self.std = None
        self.z = None
        self.max_z_null = None

    @classmethod
    def run_inner_perm(cls, exp, children, n_perm: int, *, q0, q1,
                       min_vox: int = 4, base_seed: int = 0):
        """Compute per-region inner-null (mu, std) for the given Ward tree.

        Runs n_perm Freedman-Lane (Freedman & Lane 1983) inner draws against
        the tree via inner_perm.cpu_perm and reduces them to per-region
        moments.

        Args:
            exp (Experiment): pre-permute if drawing against an
                outer-permuted tree.
            children (np.array): (num_reg - num_vox, 2) Ward tree.
            n_perm (int): number of inner FL draws.
            q0 (np.array): (a0, num_img) nuisance subspace.
            q1 (np.array): (a1, num_img) interest subspace.
            min_vox (int): regions smaller than this are left NaN.
            base_seed (int): draw i uses seed base_seed + i.

        Returns:
            mu (np.array): (num_reg,) inner-null mean per region
            std (np.array): (num_reg,) inner-null std per region
        """
        return inner_perm.cpu_perm(
            exp=exp, base_seed=base_seed, n_perm=n_perm,
            q0=q0, q1=q1, children=children, min_vox=min_vox)

    @classmethod
    def _run_outer(cls, exp, k: int, *, q0, q1, n_perm_inner: int,
                   min_vox: int, cluster_mode: ClusterMode):
        """Run one outer perm: cluster, observed LLR, inner perms.

        Pure (no self, no shared state) so joblib workers can run it.
        Returns (children, size, llr, mu, std) for outer-perm index k:
        children is (num_reg - num_vox, 2); size, llr, mu, std are each
        (num_reg,).
        """
        _exp = exp.permute(k) if k else exp
        children = cluster(_exp, mode=cluster_mode)
        llr, size = glow.graph.compute_llr_batched(
            _exp, children=children, q0=q0, q1=q1)
        mu, std = cls.run_inner_perm(
            _exp, children, n_perm_inner, q0=q0, q1=q1,
            min_vox=min_vox, base_seed=(k + 1) * _INNER_SEED_BLOCK)
        return children, size, llr, mu, std

    def fit(self, exp, *, n_jobs: int = 1, verbose: bool = False):
        """Run the analysis on exp and return self.

        Populates the observed-tree attributes (children, size, llr,
        mu, std, z), the FWER null (max_z_null), and the synthesis
        output (pval, effect_list).

        Args:
            exp (Experiment): experiment to analyze. fit scales it
                (ExperimentScaled.from_exp) and decomposes its design into
                the q0/q1 subspaces.
            n_jobs (int): outer-perm parallelism via joblib. 1 (default)
                runs in-process; -1 uses all cores. Results are
                identical regardless of n_jobs (seed is derived from
                outer-perm index).
            verbose (bool): print progress and show tqdm bar.
        """
        exp = ExperimentScaled.from_exp(exp)
        q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)

        n_total = self.n_perm_fwer + 1
        self.max_z_null = np.empty(n_total)

        if verbose:
            num_vox = exp.y.shape[2]
            print(f'  [1/2] outer perms: {n_total} perms '
                  f'({num_vox} voxels, {self.n_perm_fwer} FWER, '
                  f'{self.n_perm_inner} inner, n_jobs={n_jobs}) ...')

        results = Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(self._run_outer)(
                exp, k, q0=q0, q1=q1,
                n_perm_inner=self.n_perm_inner,
                min_vox=self.min_vox,
                cluster_mode=self.cluster_mode)
            for k in range(n_total))

        for k, (children, size, llr, mu, std) in enumerate(
                tqdm(results, total=n_total, desc='outer perms',
                     disable=not verbose)):

            std_safe = np.where(std < 1e-12, 1.0, std)
            z = np.nan_to_num((llr - mu) / std_safe,
                              nan=0.0, posinf=0.0, neginf=np.nan)

            consider = size >= self.min_vox
            if consider.any() and np.isfinite(z[consider]).any():
                self.max_z_null[k] = float(np.nanmax(z[consider]))
            else:
                self.max_z_null[k] = float('-inf')

            if k == 0:
                self.children = children
                self.size = size
                self.llr = llr
                self.mu = mu
                self.std = std
                self.z = z

        if verbose:
            print('  [2/2] FWER synthesis ...')
        self.finalize(exp, verbose=verbose)
        return self

    def finalize(self, exp, *, verbose: bool = False):
        """Compute FWER p-values, prune the observed tree, discover effects.

        Requires max_z_null and the observed-tree attributes. Sets
        pval and effect_list.

        Args:
            exp (Experiment): the (scaled) experiment fit ran on; supplies
                mask_idx and is carried into each discovered EffectEstimate.
            verbose (bool): print significant-region and discovery counts.
        """
        reg_active = self.size >= self.min_vox
        self.pval = self.get_pval(self.z, reg_active=reg_active,
                                  stat_null=self.max_z_null)

        sig_reg_list = list(np.where(self.pval <= self.alpha_fwer)[0])
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
                reg_idx=reg_idx, pval_fwer=self.pval[reg_idx])
            self.effect_list.append(eff)

        if verbose:
            n_disc = len(self.effect_list)
            n_pruned = len(sig_reg_list) - n_disc
            print(f'  done: {n_disc} discovered, {n_pruned} pruned')
