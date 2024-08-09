from _bisect import bisect_left

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import label
from sklearn.cluster import ward_tree
from sklearn.feature_extraction import grid_to_graph
from tqdm import tqdm

from hglm.effect import Effect
from hglm.graph import iter_topo, node_sum, iter_size_yout_ybar
from hglm.tfce import apply_tfce_x
from .exper import ExperimentScaled
from .permute import Permuter
from .regress import get_llr, ComputeRegress


class Analysis:
    """ performs effect discovery (hglm or TFCE) computes FWER p-val

    Attributes:
        exp (Experiment): the source data to run experiment on
        p_val (np.array): (num_reg) FWER controlled p value per region
        effect_list (list): list of Effect objects discovered
    """

    def __init__(self, exp):
        if exp.y.shape[0] > 1 and not isinstance(exp, ExperimentScaled):
            # scale if multiple features are given
            exp = ExperimentScaled.from_exp(exp)
        self.exp = exp

    @classmethod
    def get_pval(cls, stat):
        """ computes FWER adjusted pval Westfall-Young Permutation

        (percentile within max stat per permutation)

        Args:
            stat (np.array): (num_permute, num_reg) statistics per region

        Returns:
            pval (np.array): (num_reg) Family Wise Error Rate controlled
                p-values
        """
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
    def get_llr(cls, exp, n_perm, children=None):
        """ computes log likelihood score (full over reduced) per region

        Args:
            exp (Experiment):
            n_perm (int): number of permutations (includes unpermuted data)
            children (np.array): (num_reg, 2) each col are index of child
                regions, if none passed then iterates only through voxels

        Returns:
            llr (np.array): (num_reg, n_perm) log likelihood
        """
        # compute llr per region
        b, num_img, num_reg = exp.y.shape
        if children is not None:
            num_reg += children.shape[0]
        llr = np.zeros((n_perm, num_reg))

        # prepare the get_eps functions (eps0 is only features not-of-interest)
        x0 = exp.x[~exp.contrast, :]
        comp_reg = ComputeRegress(x0), ComputeRegress(exp.x)

        # prep Permuter object (if needed)
        perm = None if n_perm is None else Permuter(x=x0)
        for reg_idx, size, yout, ybar in iter_size_yout_ybar(exp.y, children,
                                                             perm=perm,
                                                             n_perm=n_perm,
                                                             keep_orig=True):
            for perm_idx in range(n_perm):
                # compute llr & store
                eps = [_comp_reg.get_eps(size,
                                         yout[..., perm_idx],
                                         ybar[..., perm_idx])
                       for _comp_reg in comp_reg]
                llr[perm_idx, reg_idx] = get_llr(size=size,
                                                 eps0=eps[0],
                                                 eps1=eps[1])
        return llr


class AnalysisTFCE(Analysis):
    def __init__(self, exp, n_perm, alpha=.05, verbose=False):
        super().__init__(exp)

        # compute llr per each voxel (for every permutation)
        self.llr = self.get_llr(exp, n_perm=n_perm + 1, children=None)

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
        tfce = np.full(shape=stat.shape, dtype=float,
                       fill_value=np.nanmin(stat))
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
        llr (np.array): (n_perm + 1, n_perm_adj, num_reg)
        size (np.array): (n_perm + 1, num_reg) number of voxels in
            each region (for all permutations).  first row corresponds to
            unpermuted data
    """

    def __init__(self, exp, n_perm, n_perm_adj=10, alpha=.05,
                 min_size_discover=1, verbose=False, n_jobs=0):
        super().__init__(exp)

        # constants
        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox * 2 - 1

        # permute, cluster & llr per region in hierarchy
        self.child_dict = dict()
        tqdm_dict = dict(total=n_perm + 1,
                         desc='permuting',
                         disable=not verbose)

        def permute_cluster_llr(perm_idx):
            # permute data (get one permutation of experiment)
            _exp = exp.permute(perm_idx, block_exchange=False)

            # build hierarchical segmentation
            children = self.cluster(exp=_exp)

            # compute log likeratio (get n_perm_adj permutations per region)
            llr = self.get_llr(exp=_exp, children=children, n_perm=n_perm_adj)

            return children, llr

        # run permute_cluster_llr (serial or parallel)
        perm_iter = tqdm(range(n_perm + 1), **tqdm_dict)
        if n_jobs not in (0, 1):
            # parallel
            r = Parallel(n_jobs=n_jobs)(delayed(permute_cluster_llr)(perm)
                                        for perm in perm_iter)
            llr_list = list()
            for p_idx, (child, llr) in enumerate(r):
                self.child_dict[p_idx] = child
                llr_list.append(llr)
            self.llr = np.stack(llr_list, axis=0)
        else:
            # serial
            self.llr = np.zeros((n_perm + 1, n_perm_adj, num_reg))
            for p_idx in perm_iter:
                child, llr = permute_cluster_llr(p_idx)
                self.child_dict[p_idx] = child
                self.llr[p_idx, :, :] = llr

        # compute sizes of each region
        self.size = np.empty((n_perm + 1, num_reg))
        for perm_idx, children in self.child_dict.items():
            self.size[perm_idx, :] = node_sum(x=np.ones(num_vox, dtype=int),
                                              children=children)

        # adjust llr
        mu = self.llr[:, 1:, :].mean(axis=1)
        std = self.llr[:, 1:, :].std(axis=1)
        self.z_stat = (self.llr[:, 0, :] - mu) / std

        # compute p-values (max stat across space)
        self.p_val = self.get_pval(stat=self.z_stat)

        # discover effects (greedily choose max stat regions whose p_val is
        # significant.  continue so long as disjoint significant effect remain)
        mask_exclude = self.size[0, :] < min_size_discover
        self.effect_list = self.discover(pval=self.p_val, alpha=alpha,
                                         mask_exclude=mask_exclude,
                                         priority=self.z_stat[0, :], exp=exp,
                                         children=self.child_dict[0])

    @classmethod
    def cluster(cls, exp, mode='full'):
        """ hierarchical segmentation of image

        Args:
            exp (Experiment):
            mode (str): 'ward', 'full'
                'ward': reduces image-pooled spatial covariance
                'full': reduces error in the full model

        Returns:
            children (np.array): (num_reg, 2) each col are index of child
                regions
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
            y = np.einsum('bnr,na->bar', exp.y, q.T, optimize=True)

        # reshape to vector
        num_vox = y.shape[2]
        y = y.reshape((-1, num_vox))

        # build connectivity
        mask = exp.mask_idx >= 0
        connectivity = grid_to_graph(*mask.shape, mask=mask)

        # ensure contiguous input (https://github.com/matthigger/hglm/issues/3)
        _, num_regions = label(mask)
        if num_regions > 1:
            raise NotImplementedError('non-contiguous inputs currently '
                                      'unsupported')

        # ward's clustering
        children = ward_tree(X=y.T, connectivity=connectivity)[0]

        return children

    @classmethod
    def discover(cls, pval, priority, children, exp, alpha=.05,
                 mask_exclude=None):
        """ regions with highest priority are discovered first

        Args:
            pval (np.array): (num_reg) Family Wise Error Rate controlled
                p-values
            priority (np.array): (num_reg) priority value per region,
                higher priority values are discovered first
            children (np.array): (num_reg, 2) each col are index of child
                regions
            exp (Experiment): the source data to run experiment on
            alpha (float): upper bound on FWER

        Returns:
            effect_list (list): list of disjoint Effect
        """
        # get set of all significant regions
        bool_sig = pval <= alpha

        # exclude regions as necessary
        if mask_exclude is not None:
            bool_sig &= np.logical_not(mask_exclude)

        # build stat_pval_reg_list, list of tuples (stat, pval, reg_idx)
        reg_idx = np.where(bool_sig)[0]
        stat_pval_reg_list = zip(priority[bool_sig], pval[bool_sig], reg_idx)
        stat_pval_reg_list = sorted(stat_pval_reg_list, reverse=True)

        vox_claimed = set()
        effect_list = list()
        for priority, p_val, reg_idx in stat_pval_reg_list:
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
