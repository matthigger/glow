import pathlib
import tempfile
from _bisect import bisect_left
from copy import copy

import nibabel
import numpy as np
from scipy.ndimage import label
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction import grid_to_graph
from tqdm import tqdm

from hrba.graph import iter_topo, node_sum
from hrba.tfce import apply_tfce_x
from .effect import Effect


def reshape(mask_idx, x=None, fill=0):
    img = np.full(shape=mask_idx.shape, fill_value=fill, dtype=float)
    if x is not None:
        img[mask_idx > -1] = x
    return img


class Epoch:
    """ a single round of region discovery (HRBA or TFCE) computes FWER p-val

    Attributes:
        exp (Experiment): the source data to run experiment on
        p_val (np.array): (num_reg) FWER controlled pval
        effect_list (list): list of effects discovered
    """

    @classmethod
    def get_pval(cls, stat, mask_exclude=None):
        """ computes FWER adjusted pval (percentile within max per permute)

        Args:
            stat (np.array): (num_permute, num_reg) statistics per region
                (larger assumed more significant)
            mask_exclude (np.array): (num_permute, num_reg) where True,
                statistics are excluded (resulting pval is 1 and this region's
                stats are counted among h0 permuted stats). defaults to all
                stats included when set to None

        Returns:
            pval (np.array): (num_reg) Family Wise Error Rate controlled
                p-values
        """
        if mask_exclude is not None:
            assert not np.all(mask_exclude, axis=1).any(), \
                'may not exclude an entire permutation'

            # set all excluded stats to the minimum observed
            stat = copy(stat).astype(float)
            stat[mask_exclude] = np.nan

        # max z_stat per permutation (sorted from low to high)
        z_stat_max = np.sort(np.nanmax(stat, axis=1))

        num_perm, num_reg = stat.shape
        pval = np.full(num_reg, fill_value=-1, dtype=float)
        for reg_idx, z in enumerate(stat[0, :]):
            if np.isnan(z):
                pval[reg_idx] = np.nan
                continue
            pval[reg_idx] = 1 - bisect_left(z_stat_max, z) / num_perm

        return pval


class EpochTFCE(Epoch):
    def __init__(self, exp, n_permute, alpha=.05, verbose=True):
        self.exp = exp

        raise NotImplementedError('need f stats')

        # apply TFCE per image
        self.tfce_stat = self.apply_tfce(stat=self.llr,
                                         mask_idx=exp.mask_idx,
                                         verbose=verbose)

        # compute p-values
        self.p_val = self.get_pval(self.tfce_stat)

        # discover effects
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[exp.mask_idx > -1] = self.p_val <= alpha
        self.effect_list = self.discover_mask(mask=mask, exp=exp)

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
        tfce = np.full(shape=stat.shape, dtype=float, fill_value=-1)
        for perm_idx, _stat in tqdm(enumerate(stat), **tqdm_dict):
            tfce[perm_idx, :] = apply_tfce_x(_stat, mask_idx=mask_idx)

        return tfce

    @classmethod
    def discover_mask(cls, mask, exp):
        """ each connected component in mask yields an effect region

        Args:
            mask (np.array): boolean mask same size as exp.mask_idx
            exp (Experiment):

        Returns:
            effect_list (list): list of effects (largest first)
        """
        # split discovered regions into disjoint effects
        effect_mask, num_effect = label(mask.astype(bool))

        effect_list = list()
        for eff_idx in range(1, num_effect + 1):
            # build effect for each contiguous effect found
            _mask = effect_mask == eff_idx
            effect_list.append(Effect.from_exp_mask(exp=exp, mask=_mask))

        return effect_list


class EpochHRBA(Epoch):
    """ a single round of region discovery, computes FWER p-val

    Attributes:
        child_dict (dict): keys are permutation indices, values are
            (2, n) graph arrays (equiv to sklearn.cluster.Ward.children_)
    """

    def __init__(self, exp, n_permute, alpha=.05, verbose=True,
                 n_permute_model=10):
        self.exp = exp

        # build hierarchy for all permutations (and unpermuted: perm_idx=0)
        self.child_dict = self.cluster(exp=exp,
                                       n_permute=n_permute,
                                       verbose=verbose)

        # compute sizes of each region
        b, num_img, num_vox = self.exp.y.shape
        num_reg = num_vox * 2 - 1
        self.size = np.empty((n_permute + 1, num_reg))
        for perm_idx, children in self.child_dict.items():
            self.size[perm_idx, :] = node_sum(x=np.ones(num_vox, dtype=int),
                                              children=children)

        # raise NotImplementedError('get_z_score_perm')
        #
        # # compute z stat
        # mu = self.llr_all[:, :, 1:].mean(axis=2)
        # std = self.llr_all[:, :, 1:].std(axis=2)
        # self.z_stat = (self.llr_all[:, :, 0] - mu) / std
        #
        # # compute p-values via max stat across permutation
        # self.p_val = self.get_pval(stat=self.z_stat)
        #
        # # discover effects with largest llr
        # self.effect_list = self.discover(pval=self.p_val, alpha=alpha,
        #                                  stat=self.llr_all[0, :, 0], exp=exp,
        #                                  children=self.child_dict[0])

    @classmethod
    def get_z_score_perm(cls, exp, child_dict, num_permute):
        # compute & apply freed lane permutation matrix

        # compute C_r per single voxel (we append y2 row to end).  store single
        # voxel regions as the last segmentation
        cr_reg = np.empty((a, b, num_vox - 1, num_segment + 1))

        # compute cr per multiple voxel region
        num_segment = len(child_dict)
        for perm_idx, children in child_dict.items():
            cr_reg = np.empty((a, b, num_vox - 1, num_segment))

            def get_cr(reg_idx):
                if reg_idx < num_vox:
                    # single voxel region
                    return cr_reg[:, :, reg_idx, -1]
                else:
                    return cr_reg[:, :, reg_idx - num_vox, perm_idx]

            for reg_idx, (c0, c1) in enumerate(children):
                # sum and replace
                cr_reg[:, :, reg_idx, perm_idx] = get_cr(c0) + get_cr(c1)

        # compute residuals (full & reduced)
        explained0 = (cr_reg[:a_prime, ...] ** 2).sum(axis=(0, 1))
        explained1 = (cr_reg[a_prime:-1, ...] ** 2).sum(axis=(0, 1))
        y_squared = (cr_reg[-1, ...] ** 2).sum(axis=(0, 1))

        # compute un-normalized f stats
        f_stat = explained1 / (y_squared - explained0 - explained1)

        # z score
        mu = f_stat.mean()
        std = f_stat.std()
        z_score = (f_stat - mu) / std

        return z_score

    @classmethod
    def cluster(cls, exp, n_permute, verbose=True):
        """ build child_dict """
        # prep
        b, num_img, num_vox = exp.y.shape
        x = exp.x[~exp.contrast, :], exp.x
        h = [np.linalg.pinv(_x) @ _x for _x in x]
        h_diff = h[1] - h[0]
        i = np.eye(num_img)

        # get connectivity (ensures only neighboring voxels joined)
        mask = exp.mask_idx >= 0
        if mask.ndim == 3:
            shape = mask.shape
        elif mask.ndim == 2:
            shape = (*mask.shape, 1)
        else:
            raise AttributeError('mask must be 2d or 3d')

        # prep ward clustering object
        child_dict = dict()
        connectivity = grid_to_graph(*shape, mask=mask)
        ward = AgglomerativeClustering(connectivity=connectivity,
                                       linkage='ward')
        tqdm_dict = dict(desc='clustering per permutation',
                         disable=not verbose)
        for perm_idx in tqdm(range(n_permute + 1), **tqdm_dict):
            # freedman lane permutation
            _exp = exp.permute(perm_idx)
            y = _exp.y.reshape((-1, num_vox))

            # cluster & store
            ward.fit(y.T)
            child_dict[perm_idx] = ward.children_

        return child_dict

    @classmethod
    def discover(cls, pval, stat, children, exp, alpha=.05):
        """ identifies most compelling disjoint effects while FWER < alpha

        by virtue of the hierarchical segmentation, significant regions may
        intersect.  we "discover" a significant effect if it has minimal
        p-value among all intersecting effects

        Args:
            pval (np.array): (num_reg) Family Wise Error Rate controlled
                p-values
            stat (np.array): (num_reg) some statistic (higher indicates
                more compelling effect associated with region)
            children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
                sklearn.cluster.Ward.children_)
            exp (Experiment): the source data to run experiment on
            alpha (float): upper bound on FWER

        Returns:
            effect_list (list): list of disjoint Effect
        """
        # get set of all significant regions
        bool_sig = pval <= alpha
        reg_idx = np.where(bool_sig)[0]
        stat_pval_reg_list = zip(stat[bool_sig], pval[bool_sig], reg_idx)
        stat_pval_reg_list = sorted(stat_pval_reg_list, reverse=True)

        vox_claimed = set()
        effect_list = list()
        for stat, p_val, reg_idx in stat_pval_reg_list:
            # check if region intersects with others discovered (no shared
            # ancestor)
            vox_contained = set(iter_topo(children=children,
                                          node_start=reg_idx,
                                          only_leaf=True))
            if vox_claimed.intersection(vox_contained):
                # region intersects some claimed region already discovered
                continue

            # claim intersecting voxels
            vox_claimed |= vox_contained

            # build mask corresponding to effect region
            mask = np.zeros(exp.mask_idx.shape, dtype=bool)
            for vox in vox_contained:
                mask[exp.mask_idx == vox] = True

            #  build effect & add to effect list
            effect = Effect.from_exp_mask(mask=mask, exp=exp, reg_idx=reg_idx,
                                          p_val_fwer=p_val)
            effect_list.append(effect)

        return effect_list
