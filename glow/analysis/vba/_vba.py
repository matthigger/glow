"""Voxel-based analysis with permutation FWER and optional TFCE."""

from typing import Callable

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from tqdm import tqdm

from glow.experiment.exper import ExperimentScaled
from .._base import AnalysisVoxel, reject_gpu
from ..mancova import get_hotel_tr, get_wilks


class AnalysisVBA(AnalysisVoxel):
    """Per-voxel MANCOVA analysis with max-stat permutation FWER.

    Computes one stat per voxel, optionally z-scores and TFCE-enhances
    the permutation null, then derives FWER-controlled p-values.

    The two arms take different defaults, each its most powerful setting
    in the stat bake-off (glow._extra.benchmark's vba_stat cache):
    plain VBA the raw Hotelling-Lawley trace, TFCE the z-scored
    1 - Wilks. Either is overridable via get_stat / z_flag.

    The experiment is supplied to fit(), not stored (see Analysis).

    Attributes:
        n_perm_fwer (int): permutations for FWER control
        alpha_fwer (float): family-wise error rate
        tfce_flag (bool): whether TFCE enhancement is applied
        z_flag (bool): whether stats are z-scored before TFCE
        verbose (bool): whether progress is printed
        stat (np.array): (n_perm_fwer+1, num_vox) stats (populated by fit)
        pval (np.array): (num_vox,) FWER p-values (populated by fit)
    """

    RECORD_FIELDS = ('get_stat', 'n_perm_fwer', 'alpha_fwer', 'tfce_flag',
                     'z_flag')

    def __init__(self, n_perm_fwer: int, alpha_fwer: float = .05,
                 verbose: bool = False, tfce_flag: bool = False,
                 z_flag: bool = None, get_stat: Callable = None):
        """Configure a voxel-based analysis.

        Args:
            n_perm_fwer (int): number of permutations for FWER control
            alpha_fwer (float): family-wise error rate
            verbose (bool): print progress
            tfce_flag (bool): apply TFCE enhancement
            z_flag (bool): z-score voxel-wise using the permutation null
                before TFCE. Makes the null distribution spatially
                homogeneous (pivotal), improving power under max-stat
                correction. None takes the arm's default: True under
                TFCE, False otherwise.
            get_stat (Callable): per-region stat function (e, h, n);
                None takes the arm's default: 1 - Wilks lambda under
                TFCE, the Hotelling-Lawley trace otherwise.
        """
        if get_stat is None:
            get_stat = get_wilks if tfce_flag else get_hotel_tr
        if z_flag is None:
            z_flag = bool(tfce_flag)
        super().__init__(get_stat=get_stat)
        self.n_perm_fwer = n_perm_fwer
        self.alpha_fwer = alpha_fwer
        self.tfce_flag = tfce_flag
        self.z_flag = z_flag
        self.verbose = verbose

    def fit(self, exp, _stat=None, *, n_jobs: int = 1, gpu=False,
            tfce_backend: str = None):
        """Run the permutation walk on exp and compute p-values.

        Args:
            exp (Experiment): experiment to analyze (scaled on the way in).
            _stat (np.array): optional (n_perm_fwer+1, num_vox) pre-computed
                stat matrix (raw, before z-scoring or TFCE). Row 0 is the
                observed draw. Caller is responsible for passing a copy.
                Must match n_perm_fwer.
            n_jobs (int): permutation-level parallelism via joblib, spanning
                both the stat walk and TFCE. 1 (default) runs in-process;
                -1 uses all cores. Results are identical regardless of
                n_jobs (each permutation is seeded by its index).
            gpu: accepted for the uniform fit contract (see Analysis.fit);
                there is no device backend here, so 'auto' is a no-op and
                an explicit True raises.
            tfce_backend (str or None): TFCE backend, see apply_tfce. None
                takes the module default (prefer fslmaths when installed).

        Returns:
            self
        """
        reject_gpu(gpu, type(self).__name__)
        exp = ExperimentScaled.from_exp(exp)
        self.stat = self.build_stat_matrix(exp, _stat, n_jobs=n_jobs,
                                           verbose=self.verbose)
        if self.z_flag:
            self.stat = self.z_score_stat(self.stat)
        if self.tfce_flag:
            self.stat = self.apply_tfce(stat=self.stat,
                                        mask_idx=exp.mask_idx,
                                        n_jobs=n_jobs,
                                        verbose=self.verbose,
                                        backend=tfce_backend)
        self.pval = self.get_pval(self.stat)
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[exp.mask_idx > -1] = self.pval <= self.alpha_fwer
        self.effect_list = self.discover_mask(mask=mask, exp=exp)
        return self

    @classmethod
    def apply_tfce(cls, stat, mask_idx, *, n_jobs: int = 1,
                   verbose: bool = False, backend: str = None):
        """Apply TFCE to every permutation image.

        Each permutation image is enhanced independently, so the work
        parallelises over permutations with joblib (n_jobs). The python
        backend dispatches one task per permutation; the fsl backend
        enhances a whole chunk of permutations per fslmaths call, so it
        splits the stack one chunk per worker instead.

        Args:
            stat (np.array): (n_perm+1, num_vox) statistics (row 0 observed)
            mask_idx (np.array): 3d voxel index array (-1 outside analysis)
            n_jobs (int): permutation-level parallelism via joblib. 1
                (default) runs in-process; -1 uses all cores.
            verbose (bool): print progress
            backend (str or None): 'auto', 'fsl' or 'python' (see
                _tfce.resolve_backend); None takes the module default.
                A speed knob, deliberately outside RECORD_FIELDS so a
                cached fit does not re-key on whether the machine had
                FSL. The backends use different height grids and so
                differ by ~5e-3 relative; see the _tfce module docstring.

        Returns:
            tfce (np.array): (n_perm+1, num_vox) TFCE-enhanced stats
        """
        # lazy import: TFCE validation against FSL is opt-in
        from . import _tfce as _tfce_mod

        tfce = np.full(shape=stat.shape,
                       fill_value=np.nanmin(stat))
        mask_bb = _tfce_mod.get_mask_bb(mask_idx)
        name = _tfce_mod.resolve_backend(backend, ndim=mask_bb.ndim)

        if name == 'fsl':
            sl_list = _tfce_mod.get_chunk_slices(
                stat.shape[0], mask_bb.size, effective_n_jobs(n_jobs))
            chunks = Parallel(n_jobs=n_jobs, return_as='generator')(
                delayed(_tfce_mod.apply_tfce_stat)(
                    stat[sl], mask_idx=mask_idx, backend='fsl')
                for sl in sl_list)
            for sl, chunk in zip(sl_list,
                                 tqdm(chunks, total=len(sl_list),
                                      desc='tfce per fsl chunk',
                                      disable=not verbose)):
                tfce[sl, :] = chunk
            return tfce

        rows = Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(_tfce_mod.apply_tfce_x)(_stat, mask_idx=mask_idx,
                                            backend='python')
            for _stat in stat)
        for perm_idx, row in enumerate(tqdm(rows, total=stat.shape[0],
                                            desc='tfce per permutation',
                                            disable=not verbose)):
            tfce[perm_idx, :] = row
        return tfce
