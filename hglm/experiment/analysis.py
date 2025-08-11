from _bisect import bisect_left
from collections import defaultdict
from collections import deque

import numpy as np
from scipy.ndimage import label
from sklearn.cluster import ward_tree
from sklearn.feature_extraction import grid_to_graph
from tqdm import tqdm

import hglm.effect
import hglm.graph
import hglm.tfce
from .exper import ExperimentScaled
from .mancova import get_hotel_tr
from .permute import Permuter
from .tailor import permute_llr_partition


class Analysis:
    """ performs effect discovery (hglm or TFCE) computes FWER p-val

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
        for reg_idx, size, e, h in hglm.graph.iter_size_e_h(
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


class AnalysisTFCE(Analysis):
    def __init__(self, exp, n_perm, alpha_fwer=.05, verbose=False, **kwargs):
        super().__init__(exp, **kwargs)

        # compute stat per each voxel (for every permutation)
        self.stat = self.get_stat_perm(exp, n_perm=n_perm + 1, children=None)

        # apply TFCE per image
        self.tfce_stat = self.apply_tfce(stat=self.stat,
                                         mask_idx=exp.mask_idx,
                                         verbose=verbose)

        # compute p-values
        self.pval = self.get_pval(self.tfce_stat)

        # discover effects
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[exp.mask_idx > -1] = self.pval <= alpha_fwer
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
            children = self.cluster(exp=_exp)
            self.child_dict[perm_idx] = children

            # build stat for each region in hierarchy
            self.stat[perm_idx, :] = self.get_stat_perm(exp=_exp,
                                                        children=children)

        # merge all graphs (many nodes are repeated across permutations above,
        # we adjust them all by same mu and std to minimize computation)
        map_to_new, children, _ = hglm.graph.graph_merge(
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
            self.size[perm_idx, :] = hglm.graph.node_sum(x=np.ones(num_vox,
                                                                   dtype=int),
                                                         children=children)

        # compute p-values (max stat across space)
        self.pval = self.get_pval(stat=self.z_stat,
                                  reg_active=self.size[0, :] >= min_size)

        # tailor significant regions (discard to make disjoint set)
        self.sig_reg_list = np.where(self.pval <= alpha_fwer)[0]
        reg_out_list, self.homo_pval_dict = self.tailor(
            sig_reg_list=self.sig_reg_list,
            alpha_tailor=alpha_tailor,
            n_perm=n_perm_tailor,
            exp=exp,
            children=self.child_dict[0])

        # build effects
        self.effect_list = list()
        for reg_idx in reg_out_list:
            mask = hglm.graph.get_mask(reg_idx=reg_idx,
                                       mask_idx=exp.mask_idx,
                                       children=self.child_dict[0])
            pval_fwer = self.pval[reg_idx]
            eff = hglm.effect.Effect.from_exp_mask(mask=mask,
                                                   exp=exp,
                                                   reg_idx=reg_idx,
                                                   pval_fwer=pval_fwer)
            self.effect_list.append(eff)

    @classmethod
    def cluster(cls, exp, mode='ward-proj'):
        """ hierarchical segmentation of image

        Args:
            exp (Experiment):
            mode (str): 'ward', 'proj'
                'ward-full': reduces image-pooled spatial covariance
                'ward-proj': removes error

        Returns:
            children (np.array): (num_reg, 2) each col are index of child
                regions
        """
        # get connectivity (ensures only neighboring voxels joined)
        assert exp.mask_idx.ndim in (2, 3), 'mask must be 2d or 3d'
        assert mode in ('ward-full', 'ward-proj'), 'mode not recognized'

        if mode == 'ward-full':
            y = exp.y
        else:
            # compute qr decomposition
            q, r = np.linalg.qr(exp.x.T)
            q = q.T

            # map y into span of x
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
    def tailor(cls, sig_reg_list, children, exp, alpha_tailor, n_perm):
        """ attempts to tailor to regions with all, and only, one effect

        Args:
            sig_reg_list (list): list of regions declared significant
            children (np.array): (num_reg, 2) each col are index of child
                regions
            exp (Experiment): the source data to run experiment on
            alpha_tailor (float): threshold at which a prune event happens (a
                significant region and all its ancestors are discarded)
            n_perm (int): number of permutations to perform

        Returns:
            reg_out_list (list): index of prune regions
        """
        assert 1 / n_perm <= alpha_tailor, \
            'inconsistent n_perm_tailor & alpha_tailor: all merge no prune'

        sig_reg_list = list(np.sort(sig_reg_list))

        # build parent representation of graph
        num_vox = exp.y.shape[2]
        parent_all = hglm.graph.get_parent(children, num_leaf=num_vox)

        def get_sig_parent(reg_idx):
            """Yield parent nodes of reg_idx, stopping at -1"""
            while True:
                reg_idx = parent_all[reg_idx]
                if reg_idx == -1:
                    return None
                elif reg_idx in sig_reg_list:
                    return reg_idx

        # List of all significant immediate descendants of a significant node.
        # "Immediate" means the largest nested region directly below it:
        # e.g., if region 1 ⊂ region 2 ⊂ region 3, and all are significant,
        # then region 2 is considered a significant child of reg 3 — not reg 1
        sig_kid_dict = defaultdict(list)
        sig_root_list = list()
        for reg_idx in sig_reg_list:
            _parent = get_sig_parent(reg_idx)
            if _parent is None:
                sig_root_list.append(reg_idx)
            else:
                sig_kid_dict[_parent].append(reg_idx)

        def get_pval(parent, kid_list):
            """ run homogeneity test (small pval = hetero) """
            # mask is zero for not included voxels or constituent region
            # index for corresponding voxels.  voxels in the parent but
            # not in any significant kid have value of parent
            mask = hglm.graph.get_mask(reg_idx=parent,
                                       mask_idx=exp.mask_idx,
                                       children=children) * parent
            for reg_idx in kid_list:
                # replace voxels in mask with constituent regions
                _mask = hglm.graph.get_mask(reg_idx=reg_idx,
                                            mask_idx=exp.mask_idx,
                                            children=children)
                mask[_mask] = reg_idx

            # permutation test: is the parent homogenous?
            b = mask.astype(bool)
            _b = exp.mask_idx[b]
            llr_list = permute_llr_partition(x=exp.x,
                                             y=exp.y[:, :, _b],
                                             partition=mask[b],
                                             n_perm=n_perm)
            return (llr_list[0] >= llr_list).mean()

        # top-down: start from significant roots and split while heterogeneous
        reg_out_set = set()
        pval_dict = dict()
        queue = deque(sig_root_list)
        while queue:
            reg_idx = queue.pop()

            kid_list = sig_kid_dict.get(reg_idx, None)
            if kid_list is None:
                # significant region has no kids, output it (its disjoint)
                reg_out_set.add(reg_idx)
                continue

            pval_dict[reg_idx] = get_pval(parent=reg_idx, kid_list=kid_list)

            if pval_dict[reg_idx] > alpha_tailor:
                # homogeneous, ready for output
                reg_out_set.add(reg_idx)
            else:
                # heterogeneous -> descend into its significant children
                queue.extend(kid_list)

        return sorted(reg_out_set), pval_dict
