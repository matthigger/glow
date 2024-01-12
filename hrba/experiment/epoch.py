from _bisect import bisect_left
from copy import copy

import numpy as np
from scipy.ndimage import label
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction import grid_to_graph
from tqdm import tqdm

from hrba.graph import iter_topo, node_sum
from hrba.tfce import apply_tfce_x
from .cholesky import CholeskyRegressCovariate
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

    def __init__(self, exp, n_permute, n_permute_z=100, alpha=.05):
        self.exp = exp

        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox * 2 - 1

        # compute z stat per region
        self.child_dict = dict()
        self.z_stat = np.empty((n_permute + 1, num_reg))
        for perm_idx in range(n_permute + 1):
            _exp = exp.permute(perm_idx)
            children = self.cluster(exp=_exp)
            _, z_stat, _, _ = self.get_z_score(_exp, children,
                                               num_permute=n_permute_z,
                                               seed_offset=n_permute)

            # store
            self.child_dict[perm_idx] = children
            self.z_stat[perm_idx, :] = z_stat

        # compute p-values (max stat across space)
        self.p_val = self.get_pval(stat=self.z_stat)

        # discover effects (greedily choose max z stat regions whose p_val is
        # significant.  continue so long as disjoint significant effect remain)
        self.effect_list = self.discover(pval=self.p_val, alpha=alpha,
                                         stat=self.z_stat[0, :], exp=exp,
                                         children=self.child_dict[0])

        # compute sizes of each region (not really necessary, but good to have)
        self.size = np.empty((n_permute + 1, num_reg))
        for perm_idx, children in self.child_dict.items():
            self.size[perm_idx, :] = node_sum(x=np.ones(num_vox, dtype=int),
                                              children=children)

    @classmethod
    def get_z_score(cls, exp, children, num_permute, seed_offset=0):
        """ computes z score of f stat (over permutations) per region

        voxels are aggregated with unique permutations.  (otherwise an effect's
        incidental performance under some permutation is repeated across other
        voxels).

        Args:
            exp (Experiment):
            children (np.array): (num_leaf - 1, 2) graph arrays (equiv to
                sklearn.cluster.Ward.children_)
            num_permute (int): number of permutations to use when computing
                z-score adjustment
            seed_offset (int): offsets seed used in permutation.  useful to
                ensure permutations used in z score process here are distinct
                from those used to generate segmentations elsewhere.

        Returns:
            f (np.array): f statistic per region (unpermuted)
            z (np.array): z score (of f stat) per region (unpermuted)
            mu (np.array): mean f stat per region across permutations
            std (np.array): std dev of f stat per region across permutations
        """
        b, num_img, num_vox = exp.y.shape
        num_multi_vox = children.shape[0]
        num_reg = num_vox + num_multi_vox

        # freed lane permutation matrix (for all permutations)
        perms_needed = num_permute + num_vox + children.shape[0]
        seed_iter = range(seed_offset, perms_needed + seed_offset)
        freed_lane = np.stack(list(map(exp.get_freed_lane, seed_iter)))

        # prep regression object: to computes f ratios
        chol_regr = CholeskyRegressCovariate(x=exp.x, contrast=exp.contrast)

        # count regions & initialize output arrays (assume children merges
        # until only a single region remains)
        f = np.empty(num_reg)
        mu = np.empty(num_reg)
        std = np.empty(num_reg)

        r_dict = dict()
        for reg_idx in iter_topo(children, num_leaf=num_vox):
            if reg_idx < num_vox:
                # single voxel region, permute y & compute r explicitly
                # note each voxel gets its own unique permutation matrix
                y = np.empty((b, num_img, num_permute + 1))
                y[:, :, 0] = exp.y[:, :, reg_idx]
                y[:, :, 1:] = np.einsum(
                    'bn,knm->bmk',
                    exp.y[:, :, reg_idx],
                    freed_lane[reg_idx: reg_idx + num_permute, :, :])
                r_dict[reg_idx] = chol_regr.get_r(y=y)
            else:
                # if multi voxel region, compute r via constituent regions
                c0, c1 = children[int(reg_idx - num_vox), :]
                r_dict[reg_idx] = r_dict.pop(c0) + r_dict.pop(c1)

            f_ratio = chol_regr.get_f_ratio(r=r_dict[reg_idx])

            # store
            f[reg_idx] = f_ratio[0]
            mu[reg_idx] = f_ratio[1:].mean()
            std[reg_idx] = f_ratio[1:].std(ddof=1)

        # compute z
        z = (f - mu) / std

        return f, z, mu, std

    @classmethod
    def cluster(cls, exp):
        """ build child_dict """
        b, num_img, num_vox = exp.y.shape

        # get connectivity (ensures only neighboring voxels joined)
        mask = exp.mask_idx >= 0
        if mask.ndim == 3:
            shape = mask.shape
        elif mask.ndim == 2:
            shape = (*mask.shape, 1)
        else:
            raise AttributeError('mask must be 2d or 3d')

        # prep ward clustering object
        connectivity = grid_to_graph(*shape, mask=mask)
        ward = AgglomerativeClustering(connectivity=connectivity,
                                       linkage='ward')

        # cluster
        y = exp.y.reshape((-1, num_vox))
        ward.fit(y.T)

        return ward.children_

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
