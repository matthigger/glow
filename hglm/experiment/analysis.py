from _bisect import bisect_left

import numpy as np
from scipy.ndimage import label
from sklearn.cluster import ward_tree
from sklearn.feature_extraction import grid_to_graph
from tqdm import tqdm

import hglm.effect
import hglm.graph
import hglm.tfce
from .exper import ExperimentScaled
from .permute import Permuter
from .regress import get_wilks, wilks_to_chi2


class Analysis:
    """ performs effect discovery (hglm or TFCE) computes FWER p-val

    Attributes:
        exp (Experiment): the source data to run experiment on
        pval (np.array): (num_reg) FWER controlled p value per region
        effect_list (list): list of Effect objects discovered
    """

    def __init__(self, exp):
        if not isinstance(exp, ExperimentScaled):
            # pre-process
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
    def get_stat(cls, exp, n_perm=1, children=None):
        """ computes chi2

        Args:
            exp (Experiment):
            n_perm (int): number of permutations (includes unpermuted data)
            children (np.array): (num_reg, 2) each col are index of child
                regions, if none passed then iterates only through voxels

        Returns:
            stat (np.array): (num_reg, n_perm) wilk's lambda
        """
        # compute wilks per region
        b, num_img, num_reg = exp.y.shape
        if children is not None:
            num_reg += children.shape[0]
        stat = np.zeros((n_perm, num_reg))

        # prep Permuter object (if needed)
        x0 = exp.x[~exp.contrast, :]
        perm = None if n_perm == 1 else Permuter(x=x0)
        for reg_idx, size, e, h in hglm.graph.iter_size_e_h(
                x=exp.x,
                contrast=exp.contrast,
                y=exp.y,
                children=children,
                perm=perm,
                n_perm=n_perm,
                keep_orig=True):
            for perm_idx in range(n_perm):
                wilks = get_wilks(e=e[:, :, perm_idx],
                                  h=h[:, :, perm_idx])
                stat[perm_idx, reg_idx], _ = wilks_to_chi2(wilks,
                                                           contrast=exp.contrast,
                                                           b=b,
                                                           n=num_img * size)
        return stat


class AnalysisTFCE(Analysis):
    def __init__(self, exp, n_perm, alpha=.05, verbose=False):
        super().__init__(exp)

        # compute stat per each voxel (for every permutation)
        self.stat = self.get_stat(exp, n_perm=n_perm + 1, children=None)

        # apply TFCE per image
        self.tfce_stat = self.apply_tfce(stat=self.stat,
                                         mask_idx=exp.mask_idx,
                                         verbose=verbose)

        # compute p-values
        self.pval = self.get_pval(self.tfce_stat)

        # discover effects
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[exp.mask_idx > -1] = self.pval <= alpha
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
            tfce[perm_idx, :] = hglm.tfce.apply_tfce_x(_stat,
                                                       mask_idx=mask_idx)

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
            effect_list.append(
                hglm.effect.Effect.from_exp_mask(exp=exp, mask=_mask))

        return effect_list


class AnalysisHGLM(Analysis):
    """ search a hierarchical segmentation for significant effects

    Attributes:
        child_dict (dict): keys are permutation indices, values are
            (2, n) graph arrays (equiv to sklearn.cluster.Ward.children_)
        stat (np.array): (n_perm + 1, n_perm_adj, num_reg)
        size (np.array): (n_perm + 1, num_reg) number of voxels in
            each region (for all permutations).  first row corresponds to
            unpermuted data
    """

    def __init__(self, exp, n_perm, n_perm_adj=10, alpha=.05,
                 min_size_discover=1, verbose=False):
        super().__init__(exp)

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
            children = self.cluster(exp=_exp)
            self.child_dict[perm_idx] = children

            # build stat for each region in hierarchy
            self.stat[perm_idx, :] = self.get_stat(exp=_exp,
                                                   children=children)

        # merge all graphs (many nodes are repeated across permutations above,
        # we adjust them all by same mu and std to minimize computation)
        # todo: child_dict to child_list
        map_to_new, children, _ = hglm.graph.graph_merge(
            n_common=num_vox,
            children_list=list(self.child_dict.values()))

        # to ensure each of these permuted stats is new, we run one
        # permutation ahead of time
        _exp = exp.permute(1 << 31 - 1, block_exchange=True)
        # compute permutation stat for each region in common graph
        stat_perm = self.get_stat(exp=_exp,
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
            self.size[perm_idx, :] = hglm.graph.node_sum(x=np.ones(num_vox,
                                                                   dtype=int),
                                                         children=children)

        # compute p-values (max stat across space)
        self.pval = self.get_pval(stat=self.z_stat)

        # discover effects (greedily choose max stat regions whose pval is
        # significant.  continue so long as disjoint significant effect remain)
        mask_exclude = self.size[0, :] < min_size_discover
        self.effect_list = self.discover(pval=self.pval, alpha=alpha,
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
        for priority, pval, reg_idx in stat_pval_reg_list:
            # check if region intersects with others discovered (no shared
            # ancestor)
            vox_contained = set(hglm.graph.iter_topo(children=children,
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
            effect = hglm.effect.Effect.from_exp_mask(mask=mask, exp=exp,
                                                      reg_idx=reg_idx,
                                                      pval_fwer=pval)
            effect_list.append(effect)

        return effect_list
