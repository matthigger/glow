"""Voxel-based analysis with permutation FWER and optional TFCE."""

from typing import Callable

import numpy as np
from tqdm import tqdm

from .._base import AnalysisVoxel


class AnalysisVBA(AnalysisVoxel):
    """Per-voxel MANCOVA analysis with max-stat permutation FWER.

    Computes one stat per voxel, optionally z-scores and TFCE-enhances
    the permutation null, then derives FWER-controlled p-values.

    Attributes:
        exp (Experiment): source data (scaled)
        n_perm_fwer (int): permutations for FWER control
        alpha_fwer (float): family-wise error rate
        tfce_flag (bool): whether TFCE enhancement is applied
        z_flag (bool): whether stats are z-scored before TFCE
        verbose (bool): whether progress is printed
        stat (np.array): (n_perm_fwer+1, num_vox) stats (populated by fit)
        pval (np.array): (num_vox,) FWER p-values (populated by fit)
    """

    def __init__(self, exp, n_perm_fwer: int, alpha_fwer: float = .05,
                 verbose: bool = False, tfce_flag: bool = False,
                 z_flag: bool = False, get_stat: Callable = None):
        """Configure a voxel-based analysis.

        Args:
            exp (Experiment): experiment to analyze
            n_perm_fwer (int): number of permutations for FWER control
            alpha_fwer (float): family-wise error rate
            verbose (bool): print progress
            tfce_flag (bool): apply TFCE enhancement
            z_flag (bool): z-score voxel-wise using the permutation null
                before TFCE. Makes the null distribution spatially
                homogeneous (pivotal), improving power under max-stat
                correction.
            get_stat (Callable): per-region stat function (e, h, n);
                defaults to Wilks lambda.
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
            _stat (np.array): optional (n_perm_fwer+1, num_vox) pre-computed
                stat matrix (raw, before z-scoring or TFCE). Row 0 is the
                observed draw. Caller is responsible for passing a copy.
                Must match n_perm_fwer.

        Returns:
            self
        """
        exp = self.exp
        self.stat = self.build_stat_matrix(_stat)
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
    def apply_tfce(cls, stat, mask_idx, verbose: bool = False):
        """Apply TFCE to every permutation image.

        Args:
            stat (np.array): (n_perm+1, num_vox) statistics (row 0 observed)
            mask_idx (np.array): 3d voxel index array (-1 outside analysis)
            verbose (bool): print progress

        Returns:
            tfce (np.array): (n_perm+1, num_vox) TFCE-enhanced stats
        """
        # lazy import: TFCE validation against FSL is opt-in
        from . import _tfce as _tfce_mod
        tfce = np.full(shape=stat.shape,
                       fill_value=np.nanmin(stat))
        for perm_idx, _stat in tqdm(enumerate(stat),
                                    desc='tfce per permutation',
                                    disable=not verbose):
            tfce[perm_idx, :] = _tfce_mod.apply_tfce_x(_stat,
                                                       mask_idx=mask_idx)
        return tfce
