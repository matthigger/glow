"""Oracle region-based analysis: one GLM per true effect support.

What a region-based test detects when its segmentation is already right.
The regions are read off the planted effects, so this is an upper bound on
an arm list, not a method anyone can run on data whose effects are unknown.
"""

import numpy as np
from joblib import Parallel, delayed

import glow.effect
from glow.experiment.exper import ExperimentScaled
from ._base import Analysis, reject_gpu
from .fwer import MaxStatPerm
from .mancova import decompose, get_hotel_tr, get_mancova


class AnalysisOracleSegment(Analysis):
    """Test the planted supports and their complement, one GLM each.

    A region-based analysis handed the segmentation it would otherwise
    have to find: one region per planted effect support, one more holding
    every analysed voxel no plant claimed, and one MANCOVA per region over
    all of that region's voxels. A region of n voxels contributes n
    observations, so the (e, h, n) scored here is the one GLOW's tree walk
    forms for a region (glow.graph.iter_mancova); what differs is where
    the regions came from, and that there are k+1 of them rather than a
    tree's worth. The gap to this arm is the search, not the statistic.
    (Classic ROI analysis instead averages a region to one signal and
    tests that, which is a different quantity; Poldrack 2007.)

    Subclasses Analysis rather than AnalysisVoxel: get_stat takes the same
    (e, h, n) as every other arm's, but AnalysisVoxel's walk indexes
    voxels and Ward nodes, where these regions are an arbitrary label map.

    The comparison set is the plant, which no permutation touches, so it is
    fixed with respect to the permutation group in the sense fwer.py
    requires.

    The experiment is supplied to fit(), not stored (see Analysis).

    Attributes:
        get_stat (Callable): per-region stat function f(e, h, n) -> float
        n_perm_fwer (int): permutations for FWER control
        alpha_fwer (float): family-wise error rate
        z_flag (bool): whether stats are z-scored across permutations
        label_map (np.array): (X, Y, Z) int, the regions tested: -1 where
            the voxel is not analysed, else 0..num_reg-1 (populated by
            fit; see get_label_map)
        stat (np.array): (n_perm_fwer+1, num_reg) stats (populated by fit)
        fwer (MaxStatPerm): the max-stat test over the regions, every one
            of them in the comparison set (populated by fit; see Analysis)
    """

    RECORD_FIELDS = ('get_stat', 'n_perm_fwer', 'alpha_fwer', 'z_flag')

    def __init__(self, n_perm_fwer: int, alpha_fwer: float = .05,
                 z_flag: bool = True, get_stat=None):
        """Configure an oracle region-based analysis.

        Args:
            n_perm_fwer (int): number of permutations for FWER control
            alpha_fwer (float): family-wise error rate
            z_flag (bool): z-score each region across permutations before
                the max-stat test (see fit for why it is on by default)
            get_stat (Callable): per-region stat function (e, h, n);
                defaults to the Hotelling-Lawley trace, which is what the
                voxel-wise arms this one is compared against take.
        """
        super().__init__()
        if get_stat is None:
            get_stat = get_hotel_tr
        self.get_stat = get_stat
        self.n_perm_fwer = n_perm_fwer
        self.alpha_fwer = alpha_fwer
        self.z_flag = z_flag
        self.label_map = None
        self.stat = None

    def fit(self, exp, *, n_jobs: int = 1, gpu=False):
        """Read the oracle regions off exp, test them, and discover.

        Args:
            exp (Experiment): experiment to analyze (scaled on the way
                in). Its patch_list is where the regions come from, so it
                must be the experiment the effects were planted into.
            n_jobs (int): permutation-level parallelism via joblib for the
                stat walk. 1 (default) runs in-process; -1 uses all cores.
                Results are identical regardless of n_jobs (each
                permutation is seeded by its index).
            gpu: accepted for the uniform fit contract (see Analysis.fit);
                there is no device backend here, so 'auto' is a no-op and
                an explicit True raises.

        Returns:
            self
        """
        reject_gpu(gpu, type(self).__name__)
        exp = ExperimentScaled.from_exp(exp)

        # Read the regions once, here, off the experiment handed in.
        # Experiment.permute builds its draw with no patch_list, so a
        # permuted experiment reports no plants, and re-reading them
        # inside the walk would test the whole volume as one region on
        # every null draw while the observed row tested k+1.
        self.label_map = self.get_label_map(exp)
        vox_list = self.get_vox_list(exp, self.label_map)

        self.stat = self.build_stat_matrix(exp, vox_list, n_jobs=n_jobs)
        # On by default, unlike the voxel-wise arms: e accumulates over a
        # region's voxels where h does not, so the statistic's null scale
        # falls with region size, and these regions differ in size by an
        # order of magnitude. Left raw, the max-stat family would compare
        # a small region against a large one on scale rather than on
        # evidence -- the effect-free remainder being the largest region
        # of all, it could not be rejected for a reason having nothing to
        # do with what it holds.
        if self.z_flag:
            self.stat, _, _ = self.z_score_stat(self.stat)

        self.fwer = MaxStatPerm.from_stat(self.stat, alpha=self.alpha_fwer)
        self.effect_list = self._discover(exp)
        return self

    @staticmethod
    def get_label_map(exp):
        """Label the planted supports and their complement, one region each.

        The oracle segmentation: one region per planted effect, clipped to
        the analysed voxels, then one region holding whatever analysed
        voxels no plant claimed. A cell with nothing planted therefore
        yields a single region, the whole analysed volume, and so does a
        plant that covers that volume -- a label no voxel ends up carrying
        is dropped rather than tested empty.

        Args:
            exp (Experiment): the experiment the effects were planted
                into; patch_list supplies the supports (as (X, Y, Z) bool
                masks) and mask_idx the analysed voxels.

        Returns:
            label_map (np.array): (X, Y, Z) int. -1 where the voxel is not
                analysed, else a region index in 0..num_reg-1: the plants
                in plant order, the complement last.
        """
        mask_active = exp.mask_idx > -1
        label_map = np.full(exp.mask_idx.shape, fill_value=-1)

        # assigned in plant order, so a voxel two supports both claim goes
        # to the later of them and the regions stay a partition
        for reg_idx, patch in enumerate(exp.patch_list):
            label_map[patch['mask'] & mask_active] = reg_idx
        label_map[mask_active & (label_map < 0)] = len(exp.patch_list)

        # renumber over the labels that landed, so num_reg is the region
        # count and no region is handed to the walk empty
        _, inv = np.unique(label_map[mask_active], return_inverse=True)
        label_map[mask_active] = inv
        return label_map

    @staticmethod
    def get_vox_list(exp, label_map) -> list:
        """List the voxels of each labelled region, as y column indices.

        The walk's own copy of the segmentation, taken before any
        permutation and passed down rather than re-derived (see fit).

        Args:
            exp (Experiment): the experiment whose mask_idx numbers the
                voxels.
            label_map (np.array): (X, Y, Z) int region labels, -1 outside
                the analysis (get_label_map).

        Returns:
            vox_list (list): one (num_vox_r,) int array of y column
                indices per region, in region order.
        """
        return [exp.mask_idx[label_map == reg_idx]
                for reg_idx in range(int(label_map.max()) + 1)]

    def build_stat_matrix(self, exp, vox_list, *, n_jobs: int = 1):
        """Walk the Freedman-Lane permutations, one stat row per draw.

        Row 0 is the observed draw, rows 1: the permutation null. Row k
        depends only on k (exp.permute(k) is seeded by k), so the matrix
        is identical however n_jobs splits it.

        Args:
            exp (Experiment): the (scaled) experiment to walk.
            vox_list (list): the regions, as y column indices per region
                (get_vox_list).
            n_jobs (int): permutation-level parallelism via joblib. 1
                (default) runs in-process; -1 uses all cores.

        Returns:
            stat (np.array): (n_perm_fwer+1, num_reg) statistics. A region
                with no usable statistic is left NaN, and every reader of
                this matrix skips NaN rather than repairing it.
        """
        n_total = self.n_perm_fwer + 1
        stat = np.full((n_total, len(vox_list)), fill_value=np.nan)
        rows = Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(self._stat_row)(exp, k, self.get_stat, vox_list)
            for k in range(n_total))
        for k, row in enumerate(rows):
            stat[k, :] = row
        return stat

    @staticmethod
    def _stat_row(exp, k: int, get_stat, vox_list):
        """Per-region stat row for outer perm k (pure, for joblib workers).

        Permutes exp by outer-perm index k (k=0 the observed data), then
        pools each region's voxels into one (e, h) and scores it at that
        region's voxel count. Kept free of instance state so joblib
        workers can run it; k seeds the permutation, so the row is a
        deterministic function of k alone.

        Args:
            exp (Experiment): the (scaled) experiment to permute and walk.
            k (int): outer permutation index; 0 is the observed data.
            get_stat (Callable): per-region stat function (e, h, n).
            vox_list (list): the regions, as y column indices per region
                (get_vox_list).

        Returns:
            stat (np.array): (num_reg,) test statistics.
        """
        exp = exp.permute(k) if k else exp
        # one design decomposition for every region, as the tree walk does
        q_tup = decompose(exp.x, exp.contrast)

        stat = np.full(len(vox_list), fill_value=np.nan)
        for reg_idx, vox_idx in enumerate(vox_list):
            e, h, _ = get_mancova(y=exp.y[:, :, vox_idx], q_tup=q_tup)
            try:
                stat[reg_idx] = get_stat(e=e, h=h, n=len(vox_idx))
            except np.linalg.LinAlgError:
                pass
        return stat

    def _discover(self, exp) -> list:
        """Estimate the effect in each region the test rejected.

        One discovery per rejected region rather than per connected
        component: the regions are the hypotheses, so each keeps the
        p-value it was rejected at, which merging adjacent regions
        (Analysis.discover_mask) would have none to give.

        Args:
            exp (Experiment): the (scaled) experiment the statistics came
                from; carried into each EffectEstimate.

        Returns:
            effect_list (list): one EffectEstimate per rejected region, in
                region order, carrying its reg_idx and pval_fwer.
        """
        effect_list = []
        for reg_idx in np.flatnonzero(self.fwer.reg_sig):
            eff = glow.effect.EffectEstimate.from_exp_mask(
                mask=self.label_map == reg_idx, exp=exp,
                reg_idx=int(reg_idx),
                pval_fwer=float(self.fwer.pval[reg_idx]))
            effect_list.append(eff)
        return effect_list
