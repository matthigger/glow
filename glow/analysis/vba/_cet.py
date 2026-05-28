"""Cluster-extent thresholding with permutation FWER."""

from bisect import bisect_left
from typing import Callable

import numpy as np
from scipy.ndimage import label

from .._base import AnalysisVoxel

DEFAULT_CET_CFT_PVAL = 0.001


class AnalysisCET(AnalysisVoxel):
    """Cluster Extent Thresholding with permutation-based FWER.

    Thresholds voxel-wise stats at a cluster forming threshold (CFT)
    derived from the permutation null at cft_pval, finds connected
    components, and compares cluster sizes to the permutation null
    of max cluster sizes.

    Attributes:
        exp (Experiment): source data (scaled)
        n_perm_fwer (int): permutations for FWER control
        alpha_fwer (float): family-wise error rate
        cft_pval (float): tail probability defining the cluster-forming
            threshold from the permutation null
        z_flag (bool): whether stats are z-scored before thresholding
        cft (float): cluster-forming threshold in stat units
            (populated by fit)
        stat (np.array): (n_perm_fwer+1, num_vox) stats (populated by fit)
        pval (np.array): (num_vox,) FWER p-values (populated by fit)
    """

    def __init__(self, exp, n_perm_fwer: int, alpha_fwer: float = .05,
                 cft_pval: float = DEFAULT_CET_CFT_PVAL, z_flag: bool = False,
                 get_stat: Callable = None):
        """Configure a cluster-extent thresholding analysis.

        Args:
            exp (Experiment): experiment to analyze
            n_perm_fwer (int): number of permutations for FWER control
            alpha_fwer (float): family-wise error rate
            cft_pval (float): tail probability defining the cluster-forming
                threshold from the permutation null
            z_flag (bool): z-score voxel-wise before thresholding
            get_stat (Callable): per-region stat function (e, h, n);
                defaults to Wilks lambda.
        """
        super().__init__(exp, get_stat=get_stat)
        self.n_perm_fwer = n_perm_fwer
        self.alpha_fwer = alpha_fwer
        self.cft_pval = cft_pval
        self.z_flag = z_flag
        self.cft = None

    def fit(self, _stat=None):
        """Run the permutation walk and compute cluster-extent p-values.

        Args:
            _stat (np.array): optional (n_perm_fwer+1, num_vox) pre-computed
                stat matrix (raw, before z-scoring). Row 0 is the observed
                draw. Caller is responsible for passing a copy. Must match
                n_perm_fwer.

        Returns:
            self
        """
        self.stat = self.build_stat_matrix(_stat)
        if self.z_flag:
            self.stat = self.z_score_stat(self.stat)
        null_pool = self.stat[1:, :].ravel()
        self.cft = np.quantile(null_pool, 1 - self.cft_pval)
        self.pval = self._get_pval_cet(self.stat, self.exp.mask_idx, self.cft)
        mask = np.zeros(self.exp.mask_idx.shape, dtype=bool)
        mask[self.exp.mask_idx > -1] = self.pval <= self.alpha_fwer
        self.effect_list = self.discover_mask(mask=mask, exp=self.exp)
        return self

    @staticmethod
    def _get_pval_cet(stat, mask_idx, cft):
        """Compute FWER p-values via permutation null of max cluster sizes.

        For each permutation (and the observed data), thresholds the stat
        map at cft, labels connected components, and records the max
        cluster size. The observed clusters are then compared to the
        sorted null of max cluster sizes (Nichols & Holmes 2002).

        Args:
            stat (np.array): (n_perm+1, num_vox) stats (row 0 = observed)
            mask_idx (np.array): 2d or 3d voxel index array (-1 outside)
            cft (float): cluster-forming threshold in stat units

        Returns:
            pval (np.array): (num_vox,) p-value per voxel; voxels in the
                same cluster share a p-value
        """
        n_rows, num_vox = stat.shape
        vox_mask = mask_idx > -1

        max_sizes = np.zeros(n_rows)
        obs_labels = np.zeros(mask_idx.shape, dtype=int)
        obs_sizes = np.array([])

        for i in range(n_rows):
            vol = np.zeros(mask_idx.shape)
            vol[vox_mask] = stat[i, :]
            # default face-connectivity
            labeled, n_cl = label(vol >= cft)
            if n_cl == 0:
                if i == 0:
                    obs_labels, obs_sizes = labeled, np.array([])
                continue
            # skip background (label 0)
            sizes = np.bincount(labeled.ravel())[1:]
            max_sizes[i] = sizes.max()
            if i == 0:
                obs_labels, obs_sizes = labeled, sizes

        null_sorted = np.sort(max_sizes[1:])
        n_perm = len(null_sorted)

        pval = np.ones(num_vox)
        if len(obs_sizes) > 0:
            obs_flat = obs_labels[vox_mask]
            for cid in range(1, len(obs_sizes) + 1):
                p = max(
                    1 - bisect_left(null_sorted, obs_sizes[cid - 1]) / n_perm,
                    1 / n_perm)
                pval[obs_flat == cid] = p

        return pval
