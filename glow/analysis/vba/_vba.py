import numpy as np
from tqdm import tqdm

from .._base import AnalysisVoxel


class AnalysisVBA(AnalysisVoxel):
    def __init__(self, exp, n_perm_fwer, alpha_fwer=.05, verbose=False,
                 tfce_flag=False, z_flag=False,
                 get_stat=None):
        """
        Args:
            exp: Experiment to analyze
            n_perm_fwer: Number of permutations for FWER control
            alpha_fwer: Family-wise error rate
            verbose: Print progress
            tfce_flag: Apply TFCE enhancement
            z_flag: Z-score voxel-wise using the permutation null before
                TFCE.  Makes the null distribution spatially homogeneous
                (pivotal), improving power under max-stat correction.
        """
        super().__init__(exp, get_stat=get_stat)
        self.n_perm_fwer = n_perm_fwer
        self.alpha_fwer = alpha_fwer
        self.tfce_flag = tfce_flag
        self.z_flag = z_flag
        self.verbose = verbose

    def fit(self, _stat=None):
        """Run the permutation walk and compute p-values.

        Args:
            _stat: optional (n_perm_fwer+1, num_vox) pre-computed stat matrix
                (raw, before z-scoring or TFCE). Caller is responsible for
                passing a copy. Must match n_perm_fwer.

        Returns:
            self
        """
        exp = self.exp
        if _stat is None:
            num_vox = exp.y.shape[2]
            _stat = np.full((self.n_perm_fwer + 1, num_vox), np.nan)
            for k in range(self.n_perm_fwer + 1):
                _exp = exp.permute(k) if k else exp
                _stat[k, :] = self.get_stat_perm(_exp, children=None)
        elif _stat.shape[0] - 1 != self.n_perm_fwer:
            raise ValueError(
                f'_stat has {_stat.shape[0] - 1} permutations but '
                f'n_perm_fwer={self.n_perm_fwer}')

        self.stat = _stat
        if self.z_flag:
            self.stat = self.z_score_stat(self.stat)
        if self.tfce_flag:
            self.stat = self.apply_tfce(stat=self.stat,
                                        mask_idx=exp.mask_idx,
                                        verbose=self.verbose)
        self.pval = self.get_pval(self.stat)
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[exp.mask_idx > -1] = self.pval <= self.alpha_fwer
        self.effect_list = self.discover_mask(mask=mask, exp=exp)
        return self

    @classmethod
    def apply_tfce(cls, stat, mask_idx, verbose=False):
        """apply TFCE to every permutation image.

        Args:
            stat (np.array): (num_permute, num_vox) statistics
            mask_idx (np.array): 3d voxel index array (-1 outside analysis)
            verbose (bool): print progress

        Returns:
            tfce (np.array): (num_permute, num_vox) TFCE-enhanced stats
        """
        from . import _tfce as _tfce_mod  # lazy import (requires FSL)
        tfce = np.full(shape=stat.shape,
                       fill_value=np.nanmin(stat))
        for perm_idx, _stat in tqdm(enumerate(stat),
                                    desc='tfce per permutation',
                                    disable=not verbose):
            tfce[perm_idx, :] = _tfce_mod.apply_tfce_x(_stat,
                                                       mask_idx=mask_idx)
        return tfce
