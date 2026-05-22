from bisect import bisect_left

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
    """

    def __init__(self, exp, n_perm_fwer, alpha_fwer=.05,
                 cft_pval=DEFAULT_CET_CFT_PVAL, z_flag=False,
                 get_stat=None):
        super().__init__(exp, get_stat=get_stat)
        self.n_perm_fwer = n_perm_fwer
        self.alpha_fwer = alpha_fwer
        self.cft_pval = cft_pval
        self.z_flag = z_flag
        self.cft = None

    def fit(self, _stat=None):
        """Run the permutation walk and compute cluster-extent p-values.

        Args:
            _stat: optional (n_perm_fwer+1, num_vox) pre-computed stat matrix
                (raw, before z-scoring). Caller is responsible for passing a
                copy. Must match n_perm_fwer.

        Returns:
            self
        """
        if _stat is None:
            num_vox = self.exp.y.shape[2]
            _stat = np.full((self.n_perm_fwer + 1, num_vox), np.nan)
            for k in range(self.n_perm_fwer + 1):
                _exp = self.exp.permute(k) if k else self.exp
                _stat[k, :] = self.get_stat_perm(_exp, children=None)
        elif _stat.shape[0] - 1 != self.n_perm_fwer:
            raise ValueError(
                f'_stat has {_stat.shape[0] - 1} permutations but '
                f'n_perm_fwer={self.n_perm_fwer}')

        self.stat = _stat
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
        """FWER via permutation null of max cluster sizes.

        For each permutation (and the observed data), thresholds the stat
        map at cft, labels connected components, and records the max
        cluster size.  The observed clusters are then compared to the
        sorted null of max cluster sizes.

        Args:
            stat: (n_perm+1, num_vox) stats (row 0 = observed)
            mask_idx: voxel index array (2D or 3D, -1 outside)
            cft: cluster-forming threshold (in stat units)

        Returns:
            pval: (num_vox,) p-value per voxel (cluster members share p)
        """
        n_rows, num_vox = stat.shape
        vox_mask = mask_idx > -1

        max_sizes = np.zeros(n_rows)
        obs_labels = np.zeros(mask_idx.shape, dtype=int)
        obs_sizes = np.array([])

        for i in range(n_rows):
            vol = np.zeros(mask_idx.shape)
            vol[vox_mask] = stat[i, :]
            labeled, n_cl = label(vol >= cft)  # default face-connectivity
            if n_cl == 0:
                if i == 0:
                    obs_labels, obs_sizes = labeled, np.array([])
                continue
            sizes = np.bincount(labeled.ravel())[1:]  # skip background
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
