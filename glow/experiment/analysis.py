from _bisect import bisect_left

import numpy as np
from scipy.ndimage import label
from tqdm import tqdm

import glow.effect
import glow.graph
import glow.vba
from .cluster import cluster
from .exper import ExperimentScaled
from .mancova import get_hotel_tr
from .permute import Permuter
from .tailor import tailor


class Analysis:
    """ performs effect discovery (glow or TFCE) computes FWER p-val

    Attributes:
        exp (Experiment): the source data to run experiment on
        get_stat (fnc): accepts e, h and returns a scalar statistic (see
            mancova.py)
    """

    def __init__(self, exp, get_stat=get_hotel_tr):
        if not isinstance(exp, ExperimentScaled):
            # pre-process
            exp = ExperimentScaled.from_exp(exp)
        self.exp = exp
        self.get_stat = get_stat

    @classmethod
    def get_pval(cls, stat, reg_active=None):
        """ computes FWER adjusted pval Westfall-Young Permutation

        (percentile within max stat per permutation)

        Args:
            stat (np.array): (num_permute, num_reg) statistics per region
            reg_active (np.array): (num_reg) indexes into 2nd dimension
                above (bool).  only active regions have a pvalue
                computed for them, otherwise np.nan is returned for inactive
                regions.  (use case: discarding regions which are too small
                a priori we needn't consider their stats in building our
                comparison set for H0 which controls for FWER ... more stat
                power is preserved for the larger regions of interest).
                default behavior is all regions are included in analysis

        Returns:
            pval (np.array): (num_reg) Family Wise Error Rate controlled
                p-values
        """
        if reg_active is None:
            reg_active = np.ones(stat.shape[1], dtype=bool)

        # max stat per permutation (sorted from low to high)
        stat_max = np.sort(np.nanmax(stat[:, reg_active], axis=1))

        # compute pvalues (what percentage of permuted, or unpermuted,
        # stats were >= to observed value?)
        num_perm, num_reg = stat.shape
        pval = np.full(num_reg, fill_value=-1, dtype=float)
        for reg_idx, z in enumerate(stat[0, :]):
            if np.isnan(z):
                pval[reg_idx] = np.nan
                continue
            pval[reg_idx] = 1 - bisect_left(stat_max, z) / num_perm

        # inactive regions get no pvalue (otherwise we don't control FWER!)
        pval[~reg_active] = np.nan

        return pval

    def get_stat_perm(self, exp, n_perm=1, children=None):
        """ computes stats for each region (fixed) under different permutations

        Args:
            exp (Experiment):
            n_perm (int): number of permutations (includes unpermuted data)
            children (np.array): (num_reg, 2) each col are index of child
                regions, if none passed then iterates only through voxels

        Returns:
            stat (np.array): (num_reg, n_perm) statistic (see self.get_stat
                attribute)
        """
        # compute wilks per region
        b, num_img, num_reg = exp.y.shape
        if children is not None:
            num_reg += children.shape[0]
        stat = np.zeros((n_perm, num_reg))

        # prep Permuter object (if needed)
        x0 = exp.x[~exp.contrast, :]
        perm = None if n_perm == 1 else Permuter(x=x0)
        for reg_idx, size, e, h in glow.graph.iter_size_e_h(
                x=exp.x,
                contrast=exp.contrast,
                y=exp.y,
                children=children,
                perm=perm,
                n_perm=n_perm,
                keep_orig=True):
            for perm_idx in range(n_perm):
                stat[perm_idx, reg_idx] = self.get_stat(e=e[:, :, perm_idx],
                                                        h=h[:, :, perm_idx])
        return stat


class AnalysisVBA(Analysis):
    def __init__(self, exp, n_perm, alpha_fwer=.05, verbose=False,
                 tfce_flag=False, cet_flag=False, mask_eff=None, **kwargs):
        super().__init__(exp, **kwargs)
        self.tfce_flag = tfce_flag
        self.cet_flag = cet_flag

        # compute stat per each voxel (for every permutation)
        self.stat = self.get_stat_perm(exp, n_perm=n_perm + 1, children=None)

        # apply TFCE per image
        if self.tfce_flag:
            self.stat = self.apply_tfce(stat=self.stat,
                                        mask_idx=exp.mask_idx,
                                        verbose=verbose)

        # compute p-values
        self.pval = self.get_pval(self.stat)

        # discover effects
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[exp.mask_idx > -1] = self.pval <= alpha_fwer
        self.effect_list = self.discover_mask(mask=mask, exp=exp,
                                              cet_flag=cet_flag,
                                              mask_eff=mask_eff)

    @classmethod
    def apply_tfce(cls, stat, mask_idx, verbose=False):
        """ writes images to nii, applies TFCE, loads and returns results

         Args:
            stat (np.array): (num_permute, num_vox) stats across all
                permutations
            mask_idx (np.array): same shape as image.  -1 where voxel not
                included in analysis, otherwise contains voxel index
            verbose (bool): toggles command line output

        Returns
            tfce (np.array): (num_permute, num_vox) tfce stats
        """
        # apply & store tfce
        tqdm_dict = dict(desc='tfce per permutation',
                         disable=not verbose)
        tfce = np.full(shape=stat.shape, dtype=float,
                       fill_value=np.nanmin(stat))
        for perm_idx, _stat in tqdm(enumerate(stat), **tqdm_dict):
            tfce[perm_idx, :] = glow.vba.apply_tfce_x(_stat,
                                                      mask_idx=mask_idx)

        return tfce

    @classmethod
    def discover_mask(cls, mask, exp, cet_flag=False, mask_eff=None):
        """ each connected component in mask yields an effect region

        Args:
            mask (np.array): boolean mask same size as exp.mask_idx
            exp (Experiment):
            cet_flag (bool): toggles cluster extent thresholding flag, if True
                uses the ground truth mask (eff_mask) to get the cluster
                extent threshold which maxmizes f1 score
            mask_eff (np.array):

        Returns:
            effect_list (list): list of effects (largest first)
        """
        # split discovered regions into disjoint effects (all adjacent are
        # same effect)
        mask_est, num_effect = label(mask.astype(bool))

        if cet_flag:
            raise NotImplementedError
        else:
            thresh_size = 0

        effect_list = list()
        for eff_idx in range(1, num_effect + 1):
            # build effect for each contiguous effect found
            _mask = mask_est == eff_idx

            if _mask.sum() >= thresh_size:
                eff = glow.effect.Effect.from_exp_mask(exp=exp, mask=_mask)
                effect_list.append(eff)

        return effect_list


class AnalysisGLOW(Analysis):
    """ search a hierarchical segmentation for significant effects

    Attributes:
        child_dict (dict): keys are permutation indices, values are
            (2, n) graph arrays (equiv to sklearn.cluster.Ward.children_)
        stat (np.array): (n_perm + 1, n_perm_adj, num_reg)
        size (np.array): (n_perm + 1, num_reg) number of voxels in
            each region (for all permutations).  first row corresponds to
            unpermuted data
    """

    def __init__(self, exp, n_perm, n_perm_adj=10, n_perm_tailor=100,
                 alpha_fwer=.05, alpha_tailor=.05, min_size=1, verbose=False,
                 **kwargs):
        super().__init__(exp, **kwargs)

        # constants
        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox * 2 - 1

        # build hierarchy per permutation, compute stat per region
        self.child_dict = dict()
        tqdm_dict = dict(total=n_perm + 1,
                         desc='clustering per permutation',
                         disable=not verbose)

        self.stat = np.full((n_perm + 1, num_reg),
                            fill_value=-1,
                            dtype=float)
        for perm_idx in tqdm(range(n_perm + 1), **tqdm_dict):
            # permute data (get one permutation of experiment)
            _exp = exp.permute(perm_idx, block_exchange=True)

            # build hierarchical segmentation
            children = cluster(exp=_exp)
            self.child_dict[perm_idx] = children

            # build stat for each region in hierarchy
            self.stat[perm_idx, :] = self.get_stat_perm(exp=_exp,
                                                        children=children)

        # merge all graphs (many nodes are repeated across permutations above,
        # we adjust them all by same mu and std to minimize computation)
        map_to_new, children, _ = glow.graph.graph_merge(
            n_common=num_vox,
            children_list=list(self.child_dict.values()))

        # to ensure each of these permuted stats is new, we run one
        # permutation ahead of time
        _exp = exp.permute(1 << 31 - 1, block_exchange=True)
        # compute permutation stat for each region in common graph
        stat_perm = self.get_stat_perm(exp=_exp,
                                       children=children,
                                       n_perm=n_perm_adj)

        # adjust
        self.z_stat = np.empty_like(self.stat)
        for perm_idx, _map_to_new in enumerate(map_to_new):
            # look up stats per region in permutation perm_idx
            _stat_perm = np.concatenate((stat_perm[:, :num_vox],
                                         stat_perm[:, _map_to_new]), axis=1)
            mu = _stat_perm.mean(axis=0)
            std = _stat_perm.std(axis=0)
            self.z_stat[perm_idx, :] = (self.stat[perm_idx, :] - mu) / std

        # compute sizes of each region
        self.size = np.empty((n_perm + 1, num_reg))
        for perm_idx, children in self.child_dict.items():
            self.size[perm_idx, :] = glow.graph.node_sum(x=np.ones(num_vox,
                                                                   dtype=int),
                                                         children=children)

        # compute p-values (max stat across space)
        self.pval = self.get_pval(stat=self.z_stat,
                                  reg_active=self.size[0, :] >= min_size)

        # tailor significant regions (discard to make disjoint set)
        self.sig_reg_list = list(np.where(self.pval <= alpha_fwer)[0])
        reg_out_list, self.homo_pval_dict = tailor(
            sig_reg_list=self.sig_reg_list,
            alpha_tailor=alpha_tailor,
            n_perm=n_perm_tailor,
            exp=exp,
            children=self.child_dict[0])

        # build effects
        self.effect_list = list()
        for reg_idx in reg_out_list:
            label_map = glow.graph.get_label_map(reg_idx_list=[reg_idx, ],
                                                 mask_idx=exp.mask_idx,
                                                 children=self.child_dict[0])
            pval_fwer = self.pval[reg_idx]
            eff = glow.effect.Effect.from_exp_mask(mask=label_map > -1,
                                                   exp=exp,
                                                   reg_idx=reg_idx,
                                                   pval_fwer=pval_fwer)
            self.effect_list.append(eff)
