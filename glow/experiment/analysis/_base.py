from bisect import bisect_left

import numpy as np
from scipy.ndimage import label

import glow.effect
import glow.graph

DEFAULT_CET_CFT_PVAL = 0.0001
"""Default cluster-forming-threshold p-value used for CET variants."""


def _sanitize_adjusted_stat(adj):
    """Replace non-finite values in an adjusted stat array.

    nan -> 0.0, posinf -> 0.0, neginf -> -30.0
    """
    return np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=-30.0)


class Analysis:
    """performs effect discovery (glow or TFCE) and computes FWER p-values.

    Attributes:
        exp (Experiment): source data
        get_stat (callable): accepts (e, h, n) and returns a scalar
            statistic (see mancova.py)
    """

    def __init__(self, exp, get_stat=None, n_jobs_perm=1):
        if get_stat is None:
            from ..mancova import get_llr
            get_stat = get_llr
        from ..exper import ExperimentScaled
        if not isinstance(exp, ExperimentScaled):
            # pre-process
            exp = ExperimentScaled.from_exp(exp)
        self.exp = exp
        self.get_stat = get_stat
        self.n_jobs_perm = n_jobs_perm

    @classmethod
    def get_pval(cls, stat, reg_active=None):
        """compute FWER-adjusted p-values via Westfall-Young permutation.

        Args:
            stat (np.array): (num_permute, num_reg) statistics per region
            reg_active (np.array): (num_reg) boolean mask. only active
                regions have a p-value computed; inactive get np.nan.
                discarding a-priori small regions from the comparison
                set preserves power for larger regions. defaults to all
                regions active.

        Returns:
            pval (np.array): (num_reg) FWER-controlled p-values
        """
        if reg_active is None:
            reg_active = np.ones(stat.shape[1], dtype=bool)
        elif not reg_active.any():
            # no active regions, return all nan
            num_reg = stat.shape[1]
            return np.full(num_reg, fill_value=np.nan)

        # max stat per permutation (sorted from low to high)
        stat_max = np.sort(np.nanmax(stat[:, reg_active], axis=1))

        # compute pvalues (what percentage of permuted, or unpermuted,
        # stats were >= to observed value?)
        num_perm, num_reg = stat.shape
        pval = np.full(num_reg, fill_value=-1.0)
        for reg_idx, z in enumerate(stat[0, :]):
            if np.isnan(z):
                pval[reg_idx] = np.nan
                continue
            pval[reg_idx] = max(1 - bisect_left(stat_max, z) / num_perm,
                               1 / num_perm)

        # inactive regions get no pvalue (otherwise we don't control FWER!)
        pval[~reg_active] = np.nan

        return pval

    @classmethod
    def z_score_stat(cls, stat):
        """Z-score each voxel across permutations using the null rows.

        For each voxel, the mean and std are computed from the permutation
        null (rows 1:).  All rows (including the observed row 0) are then
        standardized by that voxel's null mean and std.  This makes each
        voxel's null distribution ~N(0,1), removing spatial heterogeneity
        so that max-stat FWER is not biased by regionally varying noise
        (Salimi-Khorshidi et al. 2011, NeuroImage).

        Args:
            stat (np.array): (n_perm+1, num_vox) statistics.
                Row 0 is the observed (unpermuted) statistic.

        Returns:
            z (np.array): same shape, voxel-wise z-scored
        """
        null = stat[1:, :]
        mu = np.nanmean(null, axis=0)
        sigma = np.nanstd(null, axis=0, ddof=1)
        sigma[sigma < 1e-12] = 1.0
        return (stat - mu) / sigma

    @classmethod
    def get_stat_perm_multi(cls, exp, get_stat_list, n_perm=None,
                            children=None):
        """compute multiple test statistics from a single tree walk.

        Avoids redundant E/H computation when comparing stat functions.

        Args:
            exp (Experiment): experiment data
            get_stat_list (list): stat functions (each accepts e, h, n)
            n_perm (int): number of permutations (excluding unpermuted)
            children (np.array): (num_leaf - 1, 2) child index array

        Returns:
            dict mapping each stat function to (n_perm + 1, num_reg) array
        """
        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox
        if children is not None:
            num_reg += children.shape[0]

        n_rows = 1 if n_perm is None else n_perm + 1
        result = {fn: np.full((n_rows, num_reg), fill_value=np.nan)
                  for fn in get_stat_list}

        for reg_idx, size, e, h in glow.graph.iter_stat(
                exp=exp, children=children, n_perm=n_perm):
            for perm_idx in range(n_rows):
                _e = e[:, :, perm_idx]
                _h = h[:, :, perm_idx]
                for fn in get_stat_list:
                    try:
                        result[fn][perm_idx, reg_idx] = fn(
                            e=_e, h=_h, n=size)
                    except np.linalg.LinAlgError:
                        pass
        return result

    def get_stat_perm(self, exp, n_perm=None, children=None):
        """compute test statistic for each region under each permutation.

        Args:
            exp (Experiment): experiment to evaluate
            n_perm (int): number of permutations (in addition to unpermuted)
            children (np.array): (num_reg, 2) child index array. if None,
                only iterates through individual voxels.

        Returns:
            stat (np.array): (n_perm + 1, num_reg) test statistics
        """
        # compute wilks per region
        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox
        if children is not None:
            num_reg += children.shape[0]

        n_rows = 1 if n_perm is None else n_perm + 1
        stat = np.full((n_rows, num_reg), fill_value=np.nan)
        for reg_idx, size, e, h in glow.graph.iter_stat(exp=exp,
                                                       children=children,
                                                       n_perm=n_perm):
            for perm_idx in range(n_rows):
                stat[perm_idx, reg_idx] = self.get_stat(e=e[:, :, perm_idx],
                                                        h=h[:, :, perm_idx],
                                                        n=size)
        return stat

    @classmethod
    def discover_mask(cls, mask, exp):
        """split a boolean mask into connected-component effects.

        Args:
            mask (np.array): boolean mask, same shape as exp.mask_idx
            exp (Experiment): experiment for constructing Effect objects

        Returns:
            effect_list (list): discovered Effect objects
        """
        # split discovered regions into disjoint effects (all adjacent are
        # same effect)
        mask_est, num_effect = label(mask.astype(bool))

        effect_list = list()
        for eff_idx in range(1, num_effect + 1):
            # build effect for each contiguous effect found
            _mask = mask_est == eff_idx
            eff = glow.effect.Effect.from_exp_mask(exp=exp, mask=_mask)
            effect_list.append(eff)

        return effect_list
