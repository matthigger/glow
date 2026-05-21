from bisect import bisect_left

import numpy as np
from scipy.ndimage import label

from glow.experiment.exper import ExperimentScaled
from ._base import AnalysisVoxel

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
                 get_stat=None, **kwargs):
        if get_stat is None:
            from .mancova import get_wilks
            get_stat = get_wilks
        super().__init__(exp, get_stat=get_stat, **kwargs)
        self.cft_pval = cft_pval
        self.z_flag = z_flag

        # voxel-wise stats: row 0 = observed, rows 1..n_perm_fwer = FL nulls
        num_vox = exp.y.shape[2]
        self.stat = np.full((n_perm_fwer + 1, num_vox), np.nan)
        for k in range(n_perm_fwer + 1):
            _exp = exp.permute(k) if k else exp
            self.stat[k, :] = self.get_stat_perm(_exp, children=None)

        # z-score voxel-wise using the permutation null
        if self.z_flag:
            self.stat = self.z_score_stat(self.stat)

        # CFT from empirical null: pool all permutation stats
        null_pool = self.stat[1:, :].ravel()
        self.cft = np.quantile(null_pool, 1 - self.cft_pval)

        # cluster-extent FWER
        self.pval = self._get_pval_cet(self.stat, exp.mask_idx, self.cft)

        # discover effects
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[exp.mask_idx > -1] = self.pval <= alpha_fwer
        self.effect_list = self.discover_mask(mask=mask, exp=exp)

    @classmethod
    def from_precomputed(cls, *, exp, get_stat, stat, cft, cft_pval,
                         alpha_fwer, z_flag=False):
        """Construct from pre-computed stat matrix without running __init__.

        Computes cluster-extent p-values and discovers effects.
        """
        obj = cls.__new__(cls)
        if isinstance(exp, ExperimentScaled):
            obj.exp = exp
        else:
            obj.exp = ExperimentScaled.from_exp(exp)

        obj.get_stat = get_stat
        obj.stat = stat
        obj.cft_pval = cft_pval
        obj.cft = cft
        obj.z_flag = z_flag
        obj.pval = cls._get_pval_cet(stat, exp.mask_idx, cft)
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[exp.mask_idx > -1] = obj.pval <= alpha_fwer
        obj.effect_list = cls.discover_mask(mask=mask, exp=exp)
        return obj

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
        obs_labels = None
        obs_sizes = None

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
