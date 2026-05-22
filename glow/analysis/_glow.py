from bisect import bisect_left

import numpy as np
from tqdm import tqdm

import glow.effect
import glow.graph
from ._base import Analysis
from . import inner_perm
from .cluster import cluster, ClusterMode
from .mancova import decompose, is_intercept_only_nuisance
from .prune import prune_greedy


_INNER_SEED_BLOCK = 100_000


class AnalysisGLOW(Analysis):
    """Hierarchical-segmentation search for significant effects.

    Each outer perm gets its own Ward tree.  Per-region z-scoring uses
    inner Freedman-Lane draws against that tree; Westfall-Young FWER
    runs on the max-z null assembled across outer perms.

    Attributes:
        children: (num_reg - num_vox, 2) Ward tree for the observed
            (k=0) pass.
        size: (num_reg,) region sizes for the observed tree.
        llr: (num_reg,) observed LLR per region.
        mu, sigma, z: (num_reg,) inner-null mean, std, and per-region
            z-score (llr - mu)/sigma, observed tree.
        max_z_null: (n_perm_fwer + 1,) max-z per outer perm, indexed by
            outer-perm number; max_z_null[0] is the observed max-z.
        pval: (num_reg,) FWER-controlled p-values.
        effect_list: discovered EffectEstimate objects.
    """

    def __init__(self, exp, n_perm_fwer, n_perm_inner=200,
                 alpha_fwer=.05, min_vox=4,
                 cluster_mode=ClusterMode.FOCUS):
        """Configure a GLOW analysis.

        Args:
            exp: experiment to analyze.
            n_perm_fwer: outer FL permutations feeding the max-z null.
            n_perm_inner: inner FL permutations per outer perm.
            min_vox: smallest region size admitted to the FWER set.
            alpha_fwer: family-wise error rate.
            cluster_mode (ClusterMode): Ward projection.  Default
                ``ClusterMode.FOCUS`` projects onto the contrast
                subspace; ``ClusterMode.GLM_ERROR`` keeps bias +
                contrast; ``ClusterMode.NAIVE`` clusters raw y.
        """
        super().__init__(exp)
        self.n_perm_fwer = n_perm_fwer
        self.n_perm_inner = n_perm_inner
        self.alpha_fwer = alpha_fwer
        self.min_vox = min_vox
        self.cluster_mode = cluster_mode

        self._q0, self._q1, _ = decompose(x=self.exp.x,
                                          contrast=self.exp.contrast)

        self.children = None
        self.size = None
        self.llr = None
        self.mu = None
        self.sigma = None
        self.z = None
        self.max_z_null = None
        self.pval = None
        self.effect_list = None

    @classmethod
    def run_inner_perm(cls, exp, children, n_perm, *, q0, q1,
                       use_gpu=False, min_vox=4, base_seed=0):
        """Inner-null mean and std per region, from n_perm Freedman-Lane
        draws against children.

        Args:
            exp: pre-permute if drawing against an outer-permuted tree.
            children: Ward tree, (num_reg - num_vox, 2).
            n_perm: number of inner FL draws.
            q0: nuisance subspace.
            q1: interest subspace.
            use_gpu: run inner draws on GPU if available.
            min_vox: regions smaller than this are left NaN.
            base_seed: draw uses base_seed + i.
        """
        if exp.y is None:
            exp.rehydrate()

        use_fast = is_intercept_only_nuisance(exp.x, exp.contrast)
        if use_gpu:
            if use_fast:
                run = inner_perm.gpu_fast
            else:
                run = inner_perm.gpu_slow
        else:
            if use_fast:
                run = inner_perm.cpu_fast
            else:
                run = inner_perm.cpu_slow

        return run(
            exp=exp, base_seed=base_seed, n_perm=n_perm,
            q0=q0, q1=q1, children=children,
            min_vox=min_vox)

    def fit(self, *, use_gpu=False, verbose=False):
        """Run the analysis.

        Populates the observed-tree attributes (children, size, llr,
        mu, sigma, z), the FWER null (max_z_null), and the synthesis
        output (pval, effect_list).
        """
        n_total = self.n_perm_fwer + 1
        self.max_z_null = np.empty(n_total)

        if verbose:
            num_vox = self.exp.y.shape[2]
            print(f'  [1/2] outer perms: {n_total} perms '
                  f'({num_vox} voxels, {self.n_perm_fwer} FWER, '
                  f'{self.n_perm_inner} inner) ...')

        for k in tqdm(range(n_total), desc='outer perms',
                      disable=not verbose):
            _exp = self.exp.permute(k) if k else self.exp
            children = cluster(_exp, mode=self.cluster_mode)

            llr, size = glow.graph.compute_llr_batched(
                _exp, children=children,
                q0=self._q0, q1=self._q1)

            mu, sigma = self.run_inner_perm(
                _exp, children, self.n_perm_inner,
                q0=self._q0, q1=self._q1,
                use_gpu=use_gpu, min_vox=self.min_vox,
                base_seed=(k + 1) * _INNER_SEED_BLOCK)

            sigma_safe = np.where(sigma < 1e-12, 1.0, sigma)
            z = np.nan_to_num((llr - mu) / sigma_safe,
                              nan=0.0, posinf=0.0, neginf=np.nan)

            active = size >= self.min_vox
            if active.any() and np.isfinite(z[active]).any():
                self.max_z_null[k] = float(np.nanmax(z[active]))
            else:
                self.max_z_null[k] = float('-inf')

            if k == 0:
                self.children = children
                self.size = size
                self.llr = llr
                self.mu = mu
                self.sigma = sigma
                self.z = z

        if verbose:
            print('  [2/2] FWER synthesis ...')
        self.finalize(verbose=verbose)
        return self

    def finalize(self, *, verbose=False):
        """Compute FWER p-values, prune the observed tree, discover effects.

        Requires max_z_null and the observed-tree attributes.  Sets
        pval and effect_list.
        """
        null_sorted = np.sort(self.max_z_null)
        n_null = len(null_sorted)
        num_reg = self.llr.shape[0]

        reg_active = self.size >= self.min_vox
        if not reg_active.any():
            self.pval = np.full(num_reg, fill_value=np.nan)
        else:
            pval = np.full(num_reg, fill_value=-1.0)
            for r, z_r in enumerate(self.z):
                if np.isnan(z_r):
                    pval[r] = np.nan
                    continue
                pval[r] = max(
                    1 - bisect_left(null_sorted, z_r) / n_null,
                    1 / n_null)
            pval[~reg_active] = np.nan
            self.pval = pval

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
                mask_idx=self.exp.mask_idx,
                children=self.children)
            eff = glow.effect.EffectEstimate.from_exp_mask(
                mask=label_map > -1, exp=self.exp,
                reg_idx=reg_idx, pval_fwer=self.pval[reg_idx])
            self.effect_list.append(eff)

        if verbose:
            n_disc = len(self.effect_list)
            n_pruned = len(sig_reg_list) - n_disc
            print(f'  done: {n_disc} discovered, {n_pruned} pruned')
