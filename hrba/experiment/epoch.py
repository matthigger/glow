from _bisect import bisect_left

import numpy as np
from scipy.ndimage import label
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction import grid_to_graph
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from hrba.graph import iter_reg_stat_exp, iter_topo
from hrba.tfce import apply_tfce_x
from .effect import Effect
from .permute import get_perm_matrix


class Epoch:
    @classmethod
    def get_f_stat(cls, exp, child_dict=None, n_permute=None, verbose=True):
        assert (child_dict is None) != (n_permute is None), \
            'either child_dict xor n_permute required'

        # compute f-stat per region in all permutations
        num_vox = exp.y.shape[2]
        if child_dict is None:
            shape = (n_permute + 1, num_vox)
            perm_child_iter = ((perm_idx, None)
                               for perm_idx in range(n_permute + 1))
        else:
            shape = (len(child_dict), 2 * num_vox - 1)
            perm_child_iter = child_dict.items()
        size = np.full(shape, fill_value=-1, dtype=int)
        f_stat = np.full(shape, fill_value=-1, dtype=float)

        tqdm_dict = dict(desc='compute stats per permutation',
                         disable=not verbose)
        for perm_idx, children in tqdm(perm_child_iter, **tqdm_dict):
            _exp = exp.permute(perm_idx)
            for reg_idx, _size, _f_stat in iter_reg_stat_exp(children=children,
                                                             exp=_exp):
                size[perm_idx, reg_idx] = _size
                f_stat[perm_idx, reg_idx] = _f_stat

        return size, f_stat

    @classmethod
    def get_pval(cls, stat):
        """ computes FWER adjusted pval (percentile within max per permute)

        Args:
            stat (np.array): (num_permute, num_reg) statistics per region
                (larger assumed more significant)

        Returns:
            pval (np.array): (num_reg) Family Wise Error Rate controlled
                p-values
        """
        # max z_stat per permutation (sorted from low to high)
        z_stat_max = np.sort(stat.max(axis=1))

        num_perm, num_reg = stat.shape
        pval = np.full(num_reg, fill_value=-1, dtype=float)
        for reg_idx, z in enumerate(stat[0, :]):
            pval[reg_idx] = 1 - bisect_left(z_stat_max, z) / num_perm

        return pval


class EpochTFCE(Epoch):
    def __init__(self, exp, n_permute, alpha=.05, verbose=True):
        self.exp = exp

        # compute f stat per every region in hierarchy (across all permutes)
        self.size, self.f_stat = self.get_f_stat(exp=exp,
                                                 n_permute=n_permute + 1,
                                                 verbose=verbose)

        # apply TFCE per image
        self.tfce_stat = self.apply_tfce(stat=self.f_stat,
                                         mask_idx=exp.mask_idx,
                                         verbose=verbose)

        # compute p-values
        self.p_val = self.get_pval(self.tfce_stat)

        # discover effects
        mask = (self.p_val <= alpha).reshape(exp.mask_idx.shape)
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
        exp (Experiment): the source data to run experiment on
        child_dict (dict): keys are permutation indices, values are
            (2, n) graph arrays (equiv to sklearn.cluster.Ward.children_)
        size (np.array): (n_permute + 1, num_reg) size of each region
        f_stat (np.array): (n_permute + 1, num_reg) raw f-stat of each region
        z_stat (np.array): (n_permute + 1, num_reg)  "z-score" of each f-stat
        p_val (np.array): (n_permute + 1, num_reg) FWER controlled pval
        model_f_mu (LinearRegression): the mean f stat as a function of
            region size (log10 F = m * log10 num_vox + b)
        model_f_std (float): std deviation of model f (in log space)
        effect_list (list): list of effects discovered
    """

    def __init__(self, exp, n_permute, alpha=.05, verbose=True):
        self.exp = exp

        # build hierarchy
        self.child_dict = self.cluster(exp=exp, n_permute=n_permute,
                                       verbose=verbose)

        # compute f stat per every region in hierarchy (across all permutes)
        self.size, self.f_stat = self.get_f_stat(exp=exp,
                                                 child_dict=self.child_dict,
                                                 verbose=verbose)

        # model f mu & var as a function of region size
        self.model_f_mu, self.model_f_std, self.z_stat = \
            self.model_adjust_f(self.size, self.f_stat)

        # compute p-values
        self.p_val = self.get_pval(self.z_stat)

        # discover effects
        self.effect_list = self.discover(pval=self.p_val, alpha=alpha,
                                         stat=self.z_stat[0, :], exp=exp,
                                         children=self.child_dict[0])

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

        # prep ward clustering object
        child_dict = dict()
        connectivity = grid_to_graph(*shape, mask=mask)
        ward = AgglomerativeClustering(connectivity=connectivity,
                                       linkage='ward')
        tqdm_dict = dict(desc='clustering per permutation',
                         disable=not verbose)
        for perm_idx in tqdm(range(n_permute + 1), **tqdm_dict):
            # permute data residuals under reduced model (freedman lane)
            p = get_perm_matrix(perm_idx, num_img)
            freed_lane = (i - h[0]) @ p + h[0]

            # prepare y
            y = np.einsum('ijk,jm->imk', exp.y, freed_lane @ h_diff)
            y = y.reshape((-1, num_vox))

            # cluster & store
            ward.fit(y.T)
            child_dict[perm_idx] = ward.children_

        return child_dict

    @classmethod
    def model_adjust_f(cls, size, f_stat, min_f=.1):
        """ models f stats as function of region size

        motivation: we need a size agnostic stat per region (i.e. z-stat)
        """
        # init

        # prep
        log_size = np.log10(size)
        log_f = np.log10(f_stat)

        # apply min_f
        min_log_f = np.log10(min_f)
        log_f[log_f < min_log_f] = min_log_f

        # fit model_f_mu
        model_f_mu = LinearRegression(fit_intercept=True)
        model_f_mu.fit(X=log_size.reshape(-1, 1), y=log_f.flatten(),
                       sample_weight=size.flatten())

        # adjust f stats per region size
        mu = model_f_mu.predict(log_size.reshape(-1, 1)).reshape(size.shape)
        error = log_f - mu
        model_f_std = error.flatten().std()
        z_stat = (np.log10(f_stat) - mu) / model_f_std

        return model_f_mu, model_f_std, z_stat

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
