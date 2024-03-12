from _bisect import bisect_left
from copy import copy

import numpy as np
from scipy.ndimage import label
from sklearn.cluster import ward_tree
from sklearn.feature_extraction import grid_to_graph
from tqdm import tqdm

from hglm.effect import Effect
from hglm.graph import iter_topo, node_sum
from hglm.tfce import apply_tfce_x
from .regress import QRRegressCovariate


class Analysis:
    """ performs effect discovery (hglm or TFCE) computes FWER p-val

    Attributes:
        exp (Experiment): the source data to run experiment on
        p_val (np.array): (num_reg) FWER controlled pval
        effect_list (list): list of Effect objects discovered
    """

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

        # max z_stat per permutation (sorted from low to high)
        z_stat_max = np.sort(np.nanmax(stat, axis=1))

        # compute pvalues (what percentage of permuted, or unpermuted,
        # stats were >= to observed value?)
        num_perm, num_reg = stat.shape
        pval = np.full(num_reg, fill_value=-1, dtype=float)
        for reg_idx, z in enumerate(stat[0, :]):
            if np.isnan(z):
                pval[reg_idx] = np.nan
                continue
            pval[reg_idx] = 1 - bisect_left(z_stat_max, z) / num_perm

        return pval


class AnalysisTFCE(Analysis):
    def __init__(self, exp, n_perm, alpha=.05, verbose=False):
        self.exp = exp

        # compute f_ratio per each voxel
        num_vox = exp.y.shape[2]
        chol_regr = QRRegressCovariate(exp.x, exp.contrast)
        self.f_ratio = np.empty((n_perm + 1, num_vox))
        for perm_idx in range(n_perm + 1):
            _exp = exp.permute(perm_idx=perm_idx)
            qyt, yout = chol_regr.get_qyt_yout(_exp.y)
            self.f_ratio[perm_idx, :] = chol_regr.get_f_ratio(qyt, yout)

        # apply TFCE per image
        self.tfce_stat = self.apply_tfce(stat=self.f_ratio,
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
                 verbose=False):
        self.exp = exp

        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox * 2 - 1

        # pre-compute
        chol_regr = QRRegressCovariate(x=exp.x, contrast=exp.contrast)

        # compute z stat per region
        self.child_dict = dict()
        self.llr = np.empty((n_perm + n_perm_adj + 1, num_reg))
        tqdm_dict = dict(total=n_perm + 1,
                         desc='permuting',
                         disable=not verbose)
        for perm_idx in tqdm(range(n_perm + n_perm_adj + 1),
                             **tqdm_dict):
            _exp = exp.permute(perm_idx)
            children = self.cluster(exp=_exp, chol_regr=chol_regr)

            # store
            self.child_dict[perm_idx] = children
            self.llr[perm_idx, :] = self.get_llr(_exp, children,
                                                 seed_offset=n_perm)

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
        self.p_val = self.get_pval(stat=self.llr_adjust)

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
    def get_llr(cls, exp, children, num_permute=0, seed_offset=0):
        """ computes log likelihood score (full over reduced) per region

        Args:
            exp (Experiment):
            children (np.array): (num_reg, 2) each col are index of child
                regions
            num_permute (int): number of permutations to use when computing
                z-score adjustment
            seed_offset (int): offsets seed used in permutation.  useful to
                ensure permutations used in z score process here are distinct
                from those used to generate segmentations elsewhere.

        Returns:
            llr (np.array): (num_reg, ) log likelihood
        """
        b, num_img, num_vox = exp.y.shape
        num_multi_vox = children.shape[0]
        num_reg = num_vox + num_multi_vox

        # count regions & initialize output arrays (assume children merges
        # until only a single region remains)
        llr = np.empty(num_reg)

        chol_regr = QRRegressCovariate(x=exp.x, contrast=exp.contrast)

        for reg_idx, qyt, yout, size in \
                iter_qyt_yout_size(exp, chol_regr, children, num_permute,
                                   seed_offset):
            # compute f ratio
            llr[reg_idx] = chol_regr.get_llr(qyt, yout, size)[0]

        return llr

    @classmethod
    def cluster(cls, exp, mode='full', chol_regr=None):
        """ build child_dict

        Args:
            exp (Experiment):
            mode (str): 'ward', 'full' or 'diff'
                'ward': reduces image-pooled spatial covariance
                'full': reduces error in the full model
                'diff': reduces error exclusively in full model (no credit
                    given to error changes in the reduced model)
        """
        # get connectivity (ensures only neighboring voxels joined)
        assert exp.mask_idx.ndim in (2, 3), 'mask must be 2d or 3d'
        assert mode in ('ward', 'full', 'diff'), 'mode not recognized'

        if mode == 'ward':
            y = exp.y
        else:
            if chol_regr is None:
                # if chol_regr not pre-computed, compute it
                chol_regr = QRRegressCovariate(x=exp.x,
                                               contrast=exp.contrast)
            else:
                assert isinstance(chol_regr, QRRegressCovariate)

            if mode == 'full':
                # rows with same span as x
                q = chol_regr.q
            else:
                # mode == 'diff' only rows corresponding features of interest
                q = chol_regr.q[chol_regr.n_covariate:, :]

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


def iter_qyt_yout_size(exp, chol_regr, children, num_permute=0, seed_offset=0):
    b, num_img, num_vox = exp.y.shape

    # freed lane permutation matrix (for all permutations)
    perms_needed = num_permute + num_vox
    seed_iter = range(seed_offset, perms_needed + seed_offset)
    freed_lane = np.stack(list(map(exp.get_freed_lane, seed_iter)))

    qyt_yout_size_dict = dict()
    for reg_idx in iter_topo(children, num_leaf=num_vox):
        if reg_idx < num_vox:
            # single voxel region, permute y via freedman lane
            # note each voxel gets its own unique permutation matrix
            y = np.empty((b, num_img, num_permute + 1))
            y[:, :, 0] = exp.y[:, :, reg_idx]

            if num_permute:
                _freed_lane = freed_lane[reg_idx: reg_idx + num_permute, :, :]
                y[:, :, 1:] = np.einsum('bn,knm->bmk',
                                        exp.y[:, :, reg_idx],
                                        _freed_lane)

            qyt, yout = chol_regr.get_qyt_yout(y=y)
            size = 1
            qyt_yout_size_dict[reg_idx] = qyt, yout, size
        else:
            # if multi voxel region, compute r via constituent regions
            c0, c1 = children[int(reg_idx - num_vox), :]
            qyt0, yout0, size0 = qyt_yout_size_dict.pop(c0)
            qyt1, yout1, size1 = qyt_yout_size_dict.pop(c1)

            size = size0 + size1
            lam = size0 / size, size1 / size
            qyt = qyt0 * lam[0] + qyt1 * lam[1]
            yout = yout0 * lam[0] + yout1 * lam[1]
            qyt_yout_size_dict[reg_idx] = qyt, yout, size

        yield reg_idx, qyt, yout, size
