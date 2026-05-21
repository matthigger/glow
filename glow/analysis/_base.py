from bisect import bisect_left

import numpy as np
from scipy.ndimage import label

import glow.effect
import glow.graph


class Analysis:
    """performs effect discovery (glow or TFCE) and computes FWER p-values.

    Attributes:
        exp (Experiment): source data
    """

    def __init__(self, exp, get_stat=None):
        if get_stat is None:
            from .mancova import get_llr
            get_stat = get_llr
        from glow.experiment.exper import ExperimentScaled
        if not isinstance(exp, ExperimentScaled):
            # pre-process
            exp = ExperimentScaled.from_exp(exp)
        self.exp = exp
        self.get_stat = get_stat

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
        """Z-score each voxel across permutations (observed row included).

        For each voxel, the mean and std are computed from **all** rows
        — the observed row (0) together with the permutation null (1:) —
        then every row is standardized by that voxel's empirical mean and
        std.  This equalizes per-voxel scale so max-stat FWER is not
        biased by regional heterogeneity.

        Under H0 the observed row is exchangeable with the permuted rows
        (Phipson & Smyth 2010; Winkler et al. 2014), so it must contribute
        to the standardization on equal footing — otherwise row 0 is
        divided by a sigma it did not contribute to while rows 1: are
        divided by a sigma they did, and max-stat FWER drifts above
        nominal at finite B (see test_stat_reliability.py).

        Args:
            stat (np.array): (n_perm+1, num_vox) statistics.
                Row 0 is the observed (unpermuted) statistic.

        Returns:
            z (np.array): same shape, voxel-wise z-scored
        """
        mu = np.nanmean(stat, axis=0)
        sigma = np.nanstd(stat, axis=0, ddof=1)
        sigma[sigma < 1e-12] = 1.0
        return (stat - mu) / sigma

    @classmethod
    def get_stat_perm_multi(cls, exp, get_stat_list, children=None):
        """compute multiple test statistics from a single tree walk.

        Avoids redundant E/H computation when comparing stat functions.
        Computes one stat value per region per stat function, on the
        given (possibly permuted) experiment.  For permutation nulls,
        callers must loop externally over ``exp.permute(k)``.

        Args:
            exp (Experiment): experiment data (already permuted if
                this is a permutation draw)
            get_stat_list (list): stat functions (each accepts e, h, n)
            children (np.array): (num_leaf - 1, 2) child index array

        Returns:
            dict mapping each stat function to (num_reg,) array

        Raises:
            RuntimeError: if the fraction of LinAlgError failures
                exceeds ``_LINALG_FAIL_RATE`` (results too unreliable
                to proceed).
        """
        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox
        if children is not None:
            num_reg += children.shape[0]

        result = {fn: np.full(num_reg, fill_value=np.nan)
                  for fn in get_stat_list}

        n_linalg_err = 0
        n_total = 0
        for reg_idx, size, e, h in glow.graph.iter_stat(
                exp=exp, children=children):
            for fn in get_stat_list:
                n_total += 1
                try:
                    result[fn][reg_idx] = fn(e=e, h=h, n=size)
                except np.linalg.LinAlgError:
                    n_linalg_err += 1

        return result

    def get_stat_perm(self, exp, children=None):
        """compute test statistic for each region.

        Computes one stat per region on the given (possibly permuted)
        experiment.  For permutation nulls, callers must loop
        externally over ``exp.permute(k)``.

        Args:
            exp (Experiment): experiment to evaluate (already permuted
                if this is a permutation draw)
            children (np.array): (num_reg, 2) child index array. if None,
                only iterates through individual voxels.

        Returns:
            stat (np.array): (num_reg,) test statistics

        Raises:
            RuntimeError: if the fraction of LinAlgError failures
                exceeds ``_LINALG_FAIL_RATE`` (results too unreliable
                to proceed).
        """
        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox
        if children is not None:
            num_reg += children.shape[0]

        stat = np.full(num_reg, fill_value=np.nan)
        n_linalg_err = 0
        n_total = 0
        for reg_idx, size, e, h in glow.graph.iter_stat(exp=exp,
                                                        children=children):
            n_total += 1
            try:
                stat[reg_idx] = self.get_stat(e=e, h=h, n=size)
            except np.linalg.LinAlgError:
                n_linalg_err += 1

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
            eff = glow.effect.EffectEstimate.from_exp_mask(exp=exp, mask=_mask)
            effect_list.append(eff)

        return effect_list
