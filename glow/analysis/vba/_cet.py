"""Cluster-extent thresholding with permutation FWER."""

from typing import Callable

import numpy as np
from scipy.ndimage import label

from glow.experiment.exper import ExperimentScaled
from .._base import AnalysisVoxel, reject_gpu
from ..fwer import MaxStatPerm
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
        fwer (MaxStatPerm): the max-cluster-size test, stat_obs
            carrying each voxel's observed cluster size (populated by fit;
            see Analysis and _get_fwer_cet)
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
        self.fwer = self._get_fwer_cet(self.stat, exp.mask_idx, self.cft,
                                       alpha=self.alpha_fwer)
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[exp.mask_idx > -1] = self.fwer.reg_sig
        self.effect_list = self.discover_mask(mask=mask, exp=exp)
        return self

    @staticmethod
    def _get_fwer_cet(stat, mask_idx, cft, *, alpha: float):
        """Test observed clusters against the null of max cluster sizes.

        For each draw, the observed included, thresholds the stat map at
        cft, labels the connected components and records the largest
        cluster size. Every voxel of an observed cluster then carries that
        cluster's size as its statistic and is tested against the per-draw
        maxima (Nichols & Holmes 2002).

        The clusters reform in every draw, so there is no fixed region
        family to index a (n_perm+1, num_reg) matrix of sizes into and the
        null is accumulated a draw at a time. That is the whole reason
        this builds the null itself instead of handing a matrix to
        MaxStatPerm.from_stat; the comparison is still the shared
        MaxStatPerm.from_max, which is what keeps this arm on the same
        p-value convention as VBA and GLOW.

        Args:
            stat (np.array): (n_perm+1, num_vox) stats (row 0 = observed)
            mask_idx (np.array): 2d or 3d voxel index array (-1 outside)
            cft (float): cluster-forming threshold in stat units
            alpha (float): family-wise error rate

        Returns:
            MaxStatPerm: stat_obs is each voxel's observed cluster
                size, max_stat the largest cluster size per draw in draw
                order. Voxels of one cluster share a p-value; a
                sub-threshold voxel has size 0, hence p = 1.
        """
        n_rows, num_vox = stat.shape
        vox_mask = mask_idx > -1

        max_stat = np.zeros(n_rows)
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
            max_stat[i] = sizes.max()
            if i == 0:
                obs_labels, obs_sizes = labeled, sizes

        # a sub-threshold voxel keeps size 0, which needs no special case:
        # every draw's largest cluster is at least 0, so the shared
        # comparison hands it p = 1 on its own
        stat_obs = np.zeros(num_vox)
        if len(obs_sizes) > 0:
            obs_flat = obs_labels[vox_mask]
            in_cluster = obs_flat > 0
            stat_obs[in_cluster] = obs_sizes[obs_flat[in_cluster] - 1]

        # a voxel dropped from the analysis is NaN in every row; left at 0
        # it would sit in the tested family as one that merely failed to
        # reach significance
        reg_active = ~np.isnan(stat[0, :])
        stat_obs[~reg_active] = np.nan

        return MaxStatPerm.from_max(stat_obs, max_stat, alpha=alpha,
                                    reg_active=reg_active)
