"""Cluster-extent thresholding with permutation FWER."""

from bisect import bisect_left
from typing import Callable

import numpy as np
from scipy.ndimage import label

from glow.experiment.exper import ExperimentScaled
from .._base import AnalysisVoxel, reject_gpu
from ..mancova import get_hotel_tr

DEFAULT_CET_CFT_PVAL = 0.001


class AnalysisCET(AnalysisVoxel):
    """Cluster Extent Thresholding with permutation-based FWER.

    Thresholds voxel-wise stats at a cluster forming threshold (CFT)
    derived from the permutation null at cft_pval, finds connected
    components, and compares cluster sizes to the permutation null
    of max cluster sizes.

    Defaults to the raw (un-z-scored) Hotelling-Lawley trace, its most
    powerful setting in the stat bake-off (glow._extra.benchmark's
    vba_stat cache); get_stat / z_flag override.

    The experiment is supplied to fit(), not stored (see Analysis).

    Attributes:
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

    RECORD_FIELDS = ('get_stat', 'n_perm_fwer', 'alpha_fwer', 'cft_pval',
                     'z_flag')

    def __init__(self, n_perm_fwer: int, alpha_fwer: float = .05,
                 cft_pval: float = DEFAULT_CET_CFT_PVAL, z_flag: bool = False,
                 get_stat: Callable = None):
        """Configure a cluster-extent thresholding analysis.

        Args:
            n_perm_fwer (int): number of permutations for FWER control
            alpha_fwer (float): family-wise error rate
            cft_pval (float): tail probability defining the cluster-forming
                threshold from the permutation null
            z_flag (bool): z-score voxel-wise before thresholding
            get_stat (Callable): per-region stat function (e, h, n);
                defaults to the Hotelling-Lawley trace.
        """
        if get_stat is None:
            get_stat = get_hotel_tr
        super().__init__(get_stat=get_stat)
        self.n_perm_fwer = n_perm_fwer
        self.alpha_fwer = alpha_fwer
        self.cft_pval = cft_pval
        self.z_flag = z_flag
        self.cft = None

    def fit(self, exp, _stat=None, *, n_jobs: int = 1, gpu=False):
        """Run the permutation walk on exp and compute cluster-extent p-values.

        Args:
            exp (Experiment): experiment to analyze (scaled on the way in).
            _stat (np.array): optional (n_perm_fwer+1, num_vox) pre-computed
                stat matrix (raw, before z-scoring). Row 0 is the observed
                draw. Caller is responsible for passing a copy. Must match
                n_perm_fwer.
            n_jobs (int): permutation-level parallelism via joblib for the
                stat walk. 1 (default) runs in-process; -1 uses all cores.
                Results are identical regardless of n_jobs (each
                permutation is seeded by its index).
            gpu: accepted for the uniform fit contract (see Analysis.fit);
                there is no device backend here, so 'auto' is a no-op and
                an explicit True raises.

        Returns:
            self
        """
        reject_gpu(gpu, type(self).__name__)
        exp = ExperimentScaled.from_exp(exp)
        self.stat = self.build_stat_matrix(exp, _stat, n_jobs=n_jobs)
        if self.z_flag:
            self.stat, _, _ = self.z_score_stat(self.stat)
        null_pool = self.stat[1:, :].ravel()
        # nanquantile, not quantile: one dropped voxel is NaN in every row,
        # and np.quantile propagates that to the threshold, after which no
        # cluster forms anywhere and the whole fit returns p = 1
        self.cft = np.nanquantile(null_pool, 1 - self.cft_pval)
        self.pval = self._get_pval_cet(self.stat, exp.mask_idx, self.cft)
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[exp.mask_idx > -1] = self.pval <= self.alpha_fwer
        self.effect_list = self.discover_mask(mask=mask, exp=exp)
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
                same cluster share a p-value, and a voxel dropped from the
                analysis (NaN in stat) gets NaN
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

        # a dropped voxel is NaN in every row; left at the default 1.0 it
        # would sit in the tested family as a voxel that merely failed to
        # reach significance (get_pval marks the same case NaN)
        pval[np.isnan(stat[0, :])] = np.nan

        return pval
