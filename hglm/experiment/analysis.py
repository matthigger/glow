from _bisect import bisect_left
from copy import copy

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import label
from sklearn.cluster import ward_tree
from sklearn.feature_extraction import grid_to_graph
from tqdm import tqdm

from hglm.effect import Effect
from hglm.graph import iter_topo, node_sum, iter_size_yout_ybar
from hglm.tfce import apply_tfce_x
from .exper import ExperimentWhitened
from .regress import get_llr, ComputeRegress


class Analysis:
    """ performs effect discovery (hglm or TFCE) computes FWER p-val

    Attributes:
        exp (Experiment): the source data to run experiment on
        p_val (np.array): (num_reg) FWER controlled pval
        effect_list (list): list of Effect objects discovered
    """

    def __init__(self, exp):
        if not isinstance(exp, ExperimentWhitened):
            exp = ExperimentWhitened.from_exp(exp)
        self.exp = exp

    @classmethod
    def get_pval(cls, stat, mask_exclude=None):
        """ computes FWER adjusted pval Westfall-Young Permutation

        (percentile within max stat per permutation)

        Args:
            stat (np.array): (num_permute, num_reg) statistics per region
            mask_exclude (np.array): (num_permute, num_reg) where True,
                statistics are excluded (resulting pval is 1 and this region's
                stats are counted among h0 permuted stats). defaults to all
                stats included when set to None.

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

        # max stat per permutation (sorted from low to high)
        stat_max = np.sort(np.nanmax(stat, axis=1))

        # compute pvalues (what percentage of permuted, or unpermuted,
        # stats were >= to observed value?)
        num_perm, num_reg = stat.shape
        pval = np.full(num_reg, fill_value=-1, dtype=float)
        for reg_idx, z in enumerate(stat[0, :]):
            if np.isnan(z):
                pval[reg_idx] = np.nan
                continue
            pval[reg_idx] = 1 - bisect_left(stat_max, z) / num_perm

        return pval

    @classmethod
    def get_llr(cls, exp, children=None):
        """ computes log likelihood score (full over reduced) per region

        Args:
            exp (Experiment):
            children (np.array): (num_reg, 2) each col are index of child
                regions, if none passed then iterates only through voxels

        Returns:
            llr (np.array): (num_reg, ) log likelihood
        """
        # compute llr per region
        b, num_img, num_reg = exp.y.shape
        if children is not None:
            num_reg += children.shape[0]
        llr = np.zeros(num_reg)

        # prepare the get_eps functions (eps0 is only features not-of-interest)
        comp_reg0 = ComputeRegress(exp.x[~exp.contrast, :])
        comp_reg1 = ComputeRegress(exp.x)

        for reg_idx, size, yout, ybar in iter_size_yout_ybar(exp.y, children):
            eps0 = comp_reg0.get_eps(size, yout, ybar)
            eps1 = comp_reg1.get_eps(size, yout, ybar)
            llr[reg_idx] = get_llr(size=size, eps0=eps0, eps1=eps1)

        return llr


class AnalysisTFCE(Analysis):
    def __init__(self, exp, n_perm, alpha=.05, verbose=False):
        super().__init__(exp)

        # compute llr per each voxel (for every permutation)
        num_vox = exp.y.shape[2]
        self.llr = np.zeros((n_perm + 1, num_vox))
        tqdm_dict = dict(total=n_perm + 1,
                         desc='permuting',
                         disable=not verbose)
        for perm_idx in tqdm(range(n_perm + 1), **tqdm_dict):
            _exp = exp.permute(perm_idx, block_exchange=False)
            self.llr[perm_idx, :] = self.get_llr(_exp, children=None)

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
        # split discovered regions into disjoint effects (all adjacent are
        # same effect)
        effect_mask, num_effect = label(mask.astype(bool))

        effect_list = list()
        for eff_idx in range(1, num_effect + 1):
            # build effect for each contiguous effect found
            _mask = effect_mask == eff_idx
            effect_list.append(Effect.from_exp_mask(exp=exp, mask=_mask))

        return effect_list


class AnalysisHGLM(Analysis):
    """ search a hierarchical segmentation for significant effects

    Attributes:
        child_dict (dict): keys are permutation indices, values are
            (2, n) graph arrays (equiv to sklearn.cluster.Ward.children_)
        llr (np.array): (n_perm + n_perm_adj + 1, num_reg)
        size (np.array): (n_perm + n_perm_adj + 1, num_reg) number of voxels in
            each region (for all permutations).  first row corresponds to
            unpermuted data
    """

    def __init__(self, exp, n_perm, n_perm_adj=10, alpha=.05,
                 min_size_discover=1, verbose=False, n_jobs=0):
        super().__init__(exp)

        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox * 2 - 1

        # permute, cluster & llr per region in hierarchy
        self.child_dict = dict()
        self.llr = np.zeros((n_perm + n_perm_adj + 1, num_reg))
        tqdm_dict = dict(total=n_perm + n_perm_adj + 1,
                         desc='permuting',
                         disable=not verbose)

        def cluster_llr(perm_idx):
            # permute data
            _exp = exp.permute(perm_idx, block_exchange=False)

            # build hierarchical segmentation
            children = self.cluster(exp=_exp)

            # compute log likelihood ratio
            llr = self.get_llr(_exp, children)
            return children, llr

        perm_iter = tqdm(range(n_perm + n_perm_adj + 1), **tqdm_dict)
        if n_jobs not in (0, 1):
            # parallel
            r = Parallel(n_jobs=n_jobs)(delayed(cluster_llr)(perm)
                                        for perm in perm_iter)

            # store
            for p_idx, (child, llr) in enumerate(r):
                self.child_dict[p_idx] = child
                self.llr[p_idx, :] = llr
        else:
            # serial
            for p_idx in perm_iter:
                self.child_dict[p_idx], self.llr[p_idx, :] = cluster_llr(p_idx)

        # compute sizes of each region
        self.size = np.empty((n_perm + n_perm_adj + 1, num_reg))
        for perm_idx, children in self.child_dict.items():
            self.size[perm_idx, :] = node_sum(x=np.ones(num_vox, dtype=int),
                                              children=children)

        # adjust llr
        self.llr_beta = self.llr_fit(self.size[-n_perm_adj:, :],
                                     self.llr[-n_perm_adj:, :])
        llr_predict = self.llr_predict(self.llr_beta,
                                       self.size[:-n_perm_adj, :])
        self.llr_adjust = 10 ** (np.log10(self.llr[:-n_perm_adj, :]) -
                                 np.log10(llr_predict))

        # compute p-values (max stat across space)
        mask_exclude = self.size[:-n_perm_adj, :] < min_size_discover
        self.p_val = self.get_pval(stat=self.llr_adjust,
                                   mask_exclude=mask_exclude)

        # discover effects (greedily choose max z stat regions whose p_val is
        # significant.  continue so long as disjoint significant effect remain)
        self.effect_list = self.discover(pval=self.p_val, alpha=alpha,
                                         stat=self.llr_adjust[0, :], exp=exp,
                                         children=self.child_dict[0])

    @classmethod
    def llr_fit(cls, size, llr):
        """ build a model between size and llr

        log size * beta[1] + beta[0] = log llr

        Args:
            size (np.array): size of each region
            llr (np.array): log likelihood ratio of each region

        Returns:
            beta (np.array): (2) model params
        """
        x_size = np.vstack([np.ones(size.size), np.log10(size.flatten())])
        y_llr = np.log10(llr.flatten())
        return y_llr @ np.linalg.pinv(x_size)

    @classmethod
    def llr_predict(cls, beta, size):
        """ predicts llr via model.  see llr_fit()

        Args:
            beta (np.array): (2) model params
            size (np.array): size of each region

        Returns:
            beta (np.array): (2) model params
        """
        return 10 ** (beta[0] + beta[1] * np.log10(size))

    @classmethod
    def cluster(cls, exp, mode='full', chol_regr=None):
        """ build child_dict

        Args:
            exp (Experiment):
            mode (str): 'ward', 'full' or 'diff'
                'ward': reduces image-pooled spatial covariance
                'full': reduces error in the full model
        """
        # get connectivity (ensures only neighboring voxels joined)
        assert exp.mask_idx.ndim in (2, 3), 'mask must be 2d or 3d'
        assert mode in ('ward', 'full'), 'mode not recognized'

        if mode == 'ward':
            y = exp.y
        else:
            # compute qr decomposition
            q, r = np.linalg.qr(exp.x.T)
            q = q.T

            # map x into span of x
            y = np.einsum('bnr,na->bar', exp.y, q.T)

        # reshape to vector
        num_vox = y.shape[2]
        y = y.reshape((-1, num_vox))

        # build connectivity
        mask = exp.mask_idx >= 0
        connectivity = grid_to_graph(*mask.shape, mask=mask)

        # ensure contiguous input
        _, num_regions = label(mask)
        if num_regions > 1:
            raise NotImplementedError('non-contiguous inputs currently '
                                      'unsupported')

        # ward's clustering
        children = ward_tree(X=y.T, connectivity=connectivity)[0]

        return children

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
            children (np.array): (num_reg, 2) each col are index of child
                regions
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
                                          num_leaf=exp.y.shape[2],
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
