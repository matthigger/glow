import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ._base import Analysis


class AnalysisVBA(Analysis):
    def __init__(self, exp, n_perm_fwer, alpha_fwer=.05, verbose=False,
                 tfce_flag=False, z_flag=False, n_jobs_perm=1,
                 get_stat=None, **kwargs):
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
            n_jobs_perm: Number of parallel jobs for permutations (1=serial, -1=all cores)
        """
        if get_stat is None:
            from .mancova import get_wilks
            get_stat = get_wilks
        super().__init__(exp, get_stat=get_stat, n_jobs_perm=n_jobs_perm,
                         **kwargs)
        self.tfce_flag = tfce_flag
        self.z_flag = z_flag

        # compute stat per voxel: row 0 = observed, rows 1..n_perm_fwer = FL nulls
        num_vox = exp.y.shape[2]
        self.stat = np.full((n_perm_fwer + 1, num_vox), np.nan)
        for k in range(n_perm_fwer + 1):
            _exp = exp.permute(k) if k else exp
            self.stat[k, :] = self.get_stat_perm(_exp, children=None)

        # z-score voxel-wise using the permutation null
        if self.z_flag:
            self.stat = self.z_score_stat(self.stat)

        # apply TFCE per image
        if self.tfce_flag:
            self.stat = self.apply_tfce(stat=self.stat,
                                        mask_idx=exp.mask_idx,
                                        verbose=verbose,
                                        n_jobs_perm=self.n_jobs_perm)

        # compute p-values
        self.pval = self.get_pval(self.stat)

        # discover effects
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[exp.mask_idx > -1] = self.pval <= alpha_fwer

        self.effect_list = self.discover_mask(mask=mask, exp=exp)

    @classmethod
    def from_precomputed(cls, *, exp, get_stat, stat, alpha_fwer):
        """Construct from pre-computed stat matrix without running __init__.

        Computes p-values and discovers effects from the given stat matrix.
        """
        from glow.experiment.exper import ExperimentScaled
        obj = cls.__new__(cls)
        obj.exp = exp if isinstance(exp, ExperimentScaled) else ExperimentScaled.from_exp(exp)
        obj.get_stat = get_stat
        obj.stat = stat
        obj.pval = cls.get_pval(stat)
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[exp.mask_idx > -1] = obj.pval <= alpha_fwer
        obj.effect_list = cls.discover_mask(mask=mask, exp=exp)
        return obj

    @classmethod
    def apply_tfce(cls, stat, mask_idx, verbose=False, n_jobs_perm=1):
        """apply TFCE to every permutation image.

        Args:
            stat (np.array): (num_permute, num_vox) statistics
            mask_idx (np.array): 3d voxel index array (-1 outside analysis)
            verbose (bool): print progress
            n_jobs_perm (int): parallel jobs for TFCE (1=serial)

        Returns:
            tfce (np.array): (num_permute, num_vox) TFCE-enhanced stats
        """
        from . import _tfce as _tfce_mod  # lazy import (requires FSL)
        # apply & store tfce
        tqdm_dict = dict(desc='tfce per permutation',
                         disable=not verbose)
        tfce = np.full(shape=stat.shape,
                       fill_value=np.nanmin(stat))

        # helper function for parallel processing
        def process_tfce_permutation(perm_idx_local):
            return _tfce_mod.apply_tfce_x(stat[perm_idx_local, :],
                                        mask_idx=mask_idx)

        if n_jobs_perm not in (0, 1):
            # parallel execution
            results = Parallel(n_jobs=n_jobs_perm, verbose=0)(
                delayed(process_tfce_permutation)(perm_idx)
                for perm_idx in tqdm(range(stat.shape[0]), **tqdm_dict)
            )
            for perm_idx, tfce_result in enumerate(results):
                tfce[perm_idx, :] = tfce_result
        else:
            # serial execution
            for perm_idx, _stat in tqdm(enumerate(stat), **tqdm_dict):
                tfce[perm_idx, :] = _tfce_mod.apply_tfce_x(_stat,
                                                          mask_idx=mask_idx)

        return tfce
