from bisect import bisect_left

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import label
from tqdm import tqdm

import glow.effect
import glow.graph
# glow.vba imported lazily when needed (requires FSL for TFCE)
from .cluster import cluster
from .exper import ExperimentScaled
from .mancova import get_hotel_tr
from .prune import prune


class Analysis:
    """performs effect discovery (glow or TFCE) and computes FWER p-values.

    Attributes:
        exp (Experiment): source data
        get_stat (callable): accepts e, h and returns a scalar statistic
            (see mancova.py)
    """

    def __init__(self, exp, get_stat=get_hotel_tr, n_jobs_perm=1):
        if not isinstance(exp, ExperimentScaled):
            # pre-process
            exp = ExperimentScaled.from_exp(exp)
        self.exp = exp
        self.get_stat = get_stat
        self.n_jobs_perm = n_jobs_perm

    @classmethod
    def get_pval(cls, stat, reg_active=None):
        """compute FWER-adjusted p-values via Westfall-Young permutation.

        Args:
            stat (np.array): (num_permute, num_reg) statistics per region
            reg_active (np.array): (num_reg) boolean mask. only active
                regions have a p-value computed; inactive get np.nan.
                discarding a-priori small regions from the comparison
                set preserves power for larger regions. defaults to all
                regions active.

        Returns:
            pval (np.array): (num_reg) FWER-controlled p-values
        """
        if reg_active is None:
            reg_active = np.ones(stat.shape[1], dtype=bool)
        elif not reg_active.any():
            # no active regions, return all nan
            num_reg = stat.shape[1]
            return np.full(num_reg, fill_value=np.nan)

        # max stat per permutation (sorted from low to high)
        stat_max = np.sort(np.nanmax(stat[:, reg_active], axis=1))

        # compute pvalues (what percentage of permuted, or unpermuted,
        # stats were >= to observed value?)
        num_perm, num_reg = stat.shape
        pval = np.full(num_reg, fill_value=-1.0)
        for reg_idx, z in enumerate(stat[0, :]):
            if np.isnan(z):
                pval[reg_idx] = np.nan
                continue
            pval[reg_idx] = max(1 - bisect_left(stat_max, z) / num_perm,
                               1 / num_perm)

        # inactive regions get no pvalue (otherwise we don't control FWER!)
        pval[~reg_active] = np.nan

        return pval

    def get_stat_perm(self, exp, n_perm=None, children=None):
        """compute test statistic for each region under each permutation.

        Args:
            exp (Experiment): experiment to evaluate
            n_perm (int): number of permutations (in addition to unpermuted)
            children (np.array): (num_reg, 2) child index array. if None,
                only iterates through individual voxels.

        Returns:
            stat (np.array): (n_perm + 1, num_reg) test statistics
        """
        # compute wilks per region
        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox
        if children is not None:
            num_reg += children.shape[0]

        n_rows = 1 if n_perm is None else n_perm + 1
        stat = np.full((n_rows, num_reg), fill_value=np.nan)
        for reg_idx, e, h in glow.graph.iter_stat(exp=exp,
                                                  children=children,
                                                  n_perm=n_perm):
            for perm_idx in range(n_rows):
                stat[perm_idx, reg_idx] = self.get_stat(e=e[:, :, perm_idx],
                                                        h=h[:, :, perm_idx])
        return stat


class AnalysisVBA(Analysis):
    def __init__(self, exp, n_perm, alpha_fwer=.05, verbose=False,
                 tfce_flag=False, conn=None, n_jobs_perm=1, **kwargs):
        """
        Args:
            exp: Experiment to analyze
            n_perm: Number of permutations
            alpha_fwer: Family-wise error rate
            verbose: Print progress
            tfce_flag: Apply TFCE enhancement
            conn: Connectivity for clustering
            n_jobs_perm: Number of parallel jobs for permutations (1=serial, -1=all cores)
        """
        super().__init__(exp, n_jobs_perm=n_jobs_perm, **kwargs)
        self.tfce_flag = tfce_flag

        # compute stat per each voxel (for every permutation)
        self.stat = self.get_stat_perm(exp, n_perm=n_perm, children=None)

        # apply TFCE per image
        if self.tfce_flag:
            self.stat = self.apply_tfce(stat=self.stat,
                                        mask_idx=exp.mask_idx,
                                        verbose=verbose,
                                        n_jobs_perm=self.n_jobs_perm)

        # compute p-values
        self.pval = self.get_pval(self.stat)

        # discover effects
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[exp.mask_idx > -1] = self.pval <= alpha_fwer

        self.effect_list = self.discover_mask(mask=mask, exp=exp)

    @classmethod
    def apply_tfce(cls, stat, mask_idx, verbose=False, n_jobs_perm=1):
        """apply TFCE to every permutation image.

        Args:
            stat (np.array): (num_permute, num_vox) statistics
            mask_idx (np.array): 3d voxel index array (-1 outside analysis)
            verbose (bool): print progress
            n_jobs_perm (int): parallel jobs for TFCE (1=serial)

        Returns:
            tfce (np.array): (num_permute, num_vox) TFCE-enhanced stats
        """
        import glow.vba  # lazy import (requires FSL)
        # apply & store tfce
        tqdm_dict = dict(desc='tfce per permutation',
                         disable=not verbose)
        tfce = np.full(shape=stat.shape,
                       fill_value=np.nanmin(stat))
        
        # helper function for parallel processing
        def process_tfce_permutation(perm_idx_local):
            return glow.vba.apply_tfce_x(stat[perm_idx_local, :],
                                        mask_idx=mask_idx)
        
        if n_jobs_perm not in (0, 1):
            # parallel execution
            results = Parallel(n_jobs=n_jobs_perm, verbose=0)(
                delayed(process_tfce_permutation)(perm_idx)
                for perm_idx in tqdm(range(stat.shape[0]), **tqdm_dict)
            )
            for perm_idx, tfce_result in enumerate(results):
                tfce[perm_idx, :] = tfce_result
        else:
            # serial execution
            for perm_idx, _stat in tqdm(enumerate(stat), **tqdm_dict):
                tfce[perm_idx, :] = glow.vba.apply_tfce_x(_stat,
                                                          mask_idx=mask_idx)

        return tfce

    @classmethod
    def discover_mask(cls, mask, exp):
        """split a boolean mask into connected-component effects.

        Args:
            mask (np.array): boolean mask, same shape as exp.mask_idx
            exp (Experiment): experiment for constructing Effect objects

        Returns:
            effect_list (list): discovered Effect objects
        """
        # split discovered regions into disjoint effects (all adjacent are
        # same effect)
        mask_est, num_effect = label(mask.astype(bool))

        effect_list = list()
        for eff_idx in range(1, num_effect + 1):
            # build effect for each contiguous effect found
            _mask = mask_est == eff_idx
            eff = glow.effect.Effect.from_exp_mask(exp=exp, mask=_mask)
            effect_list.append(eff)

        return effect_list


class AnalysisGLOW(Analysis):
    """search a hierarchical segmentation for significant effects.

    Attributes:
        child_dict (dict): perm_idx -> (num_node, 2) children array
        stat (np.array): (n_perm + 1, num_reg) raw test statistics
        size (np.array): (n_perm + 1, num_reg) voxel count per region
    """

    # how often to persist a checkpoint (every N permutations)
    _CHECKPOINT_INTERVAL = 25

    def __init__(self, exp, n_perm, n_perm_adj=10, n_perm_prune=100,
                 alpha_fwer=.05, alpha_prune=.05, min_size=1, verbose=False,
                 n_jobs_perm=1, cloud_config=None, checkpoint=None, **kwargs):
        """
        Args:
            exp: Experiment to analyze
            n_perm: Number of permutations
            n_perm_adj: Number of adjustment permutations
            n_perm_prune: Number of pruning permutations
            alpha_fwer: Family-wise error rate
            alpha_prune: Pruning alpha
            min_size: Minimum region size
            verbose: Print progress
            n_jobs_perm: Number of parallel jobs for permutations (1=serial, -1=all cores)
            cloud_config: CloudConfig for AWS execution (if None, runs locally)
            checkpoint: optional object with load/save/delete methods for
                resuming interrupted runs. only used in the serial path.
        """
        super().__init__(exp, **kwargs)
        
        # check if running on cloud
        if cloud_config is not None:
            self._run_on_cloud(exp, n_perm, n_perm_adj, n_perm_prune,
                              alpha_fwer, alpha_prune, min_size, verbose,
                              cloud_config, **kwargs)
            return

        # constants
        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox * 2 - 1

        # build hierarchy per permutation, compute stat per region
        self.child_dict = dict()
        self.stat = np.full((n_perm + 1, num_reg),
                            fill_value=-1.0)

        # try to resume from checkpoint
        start_perm = 0
        if checkpoint is not None:
            resume = checkpoint.load()
            if resume is not None:
                self.child_dict = resume['child_dict']
                self.stat = resume['stat']
                start_perm = resume['last_perm_idx'] + 1
                if verbose:
                    print(f'  resumed from checkpoint (perm {start_perm}/{n_perm + 1})')
        
        # helper function for single permutation (for parallelization)
        def process_permutation(perm_idx):
            """Process one permutation: cluster and compute stats."""
            # permute data (get one permutation of experiment)
            _exp = exp.permute(perm_idx)

            # build hierarchical segmentation
            children = cluster(exp=_exp)

            # build stat for each region in hierarchy
            stat_row = self.get_stat_perm(exp=_exp, children=children)
            
            # free memory immediately
            del _exp
            
            return perm_idx, children, stat_row
        
        # run permutations (parallel or serial)
        if n_jobs_perm not in (0, 1):
            # parallel execution (no checkpointing support)
            results = Parallel(n_jobs=n_jobs_perm, verbose=10 if verbose else 0)(
                delayed(process_permutation)(perm_idx)
                for perm_idx in range(n_perm + 1)
            )
            # collect results
            for perm_idx, children, stat_row in results:
                self.child_dict[perm_idx] = children
                self.stat[perm_idx, :] = stat_row
        else:
            # serial execution with progress bar
            remaining = n_perm + 1 - start_perm
            tqdm_dict = dict(total=remaining,
                           desc='clustering per permutation',
                           disable=not verbose)
            for perm_idx in tqdm(range(start_perm, n_perm + 1), **tqdm_dict):
                _, children, stat_row = process_permutation(perm_idx)
                self.child_dict[perm_idx] = children
                self.stat[perm_idx, :] = stat_row

                # periodic checkpoint
                if (checkpoint is not None
                        and (perm_idx + 1) % self._CHECKPOINT_INTERVAL == 0):
                    checkpoint.save(self.child_dict, self.stat, perm_idx)

        # clean up checkpoint now that all permutations are done
        if checkpoint is not None:
            checkpoint.delete()

        # finalize analysis (common to local and cloud execution)
        self._finalize_analysis(exp, n_perm, n_perm_adj, n_perm_prune,
                               alpha_fwer, alpha_prune, min_size)
    
    
    def _finalize_analysis(self, exp, n_perm, n_perm_adj, n_perm_prune,
                          alpha_fwer, alpha_prune, min_size):
        """post-process after permutations: merge graphs, z-normalise, prune."""
        b, num_img, num_vox = exp.y.shape
        num_reg = num_vox * 2 - 1

        # merge all graphs (many nodes are repeated across permutations above,
        # we adjust them all by same mu and std to minimize computation)
        # use perm index order so local and cloud match (cloud fills child_dict in arbitrary S3 order)
        children_list = [self.child_dict[i] for i in range(n_perm + 1)]
        map_to_new, children, _ = glow.graph.graph_merge(
            n_common=num_vox,
            children_list=children_list)

        # to ensure each of these permuted stats is new, we run one
        # permutation ahead of time
        _exp = exp.permute((1 << 31) - 1)
        # compute permutation stat for each region in common graph
        stat_perm = self.get_stat_perm(exp=_exp,
                                       children=children,
                                       n_perm=n_perm_adj)
        del _exp

        # adjust
        self.z_stat = np.empty_like(self.stat)
        for perm_idx, _map_to_new in enumerate(map_to_new):
            # look up stats per region in permutation perm_idx
            _stat_perm = np.concatenate((stat_perm[:, :num_vox],
                                         stat_perm[:, _map_to_new]), axis=1)
            mu = _stat_perm.mean(axis=0)
            std = _stat_perm.std(axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                self.z_stat[perm_idx, :] = np.where(
                    std > 0,
                    (self.stat[perm_idx, :] - mu) / std,
                    0.0)
            if perm_idx == 0:
                # store H0 null distribution parameters (for viewer)
                self.stat_mu = mu
                self.stat_std = std

        # store analysis thresholds (for viewer)
        self.alpha_fwer = alpha_fwer
        self.alpha_prune = alpha_prune

        # compute sizes of each region
        self.size = np.empty((n_perm + 1, num_reg))
        for perm_idx, children in self.child_dict.items():
            self.size[perm_idx, :] = glow.graph.node_sum(x=np.ones(num_vox,
                                                                   dtype=int),
                                                         children=children)

        # compute p-values (max stat across space)
        self.pval = self.get_pval(stat=self.z_stat,
                                  reg_active=self.size[0, :] >= min_size)

        # prune significant regions (discard to make disjoint set)
        self.sig_reg_list = list(np.where(self.pval <= alpha_fwer)[0])
        reg_out_list, self.homo_pval_dict = prune(
            sig_reg_list=self.sig_reg_list,
            alpha_prune=alpha_prune,
            n_perm=n_perm_prune,
            exp=exp,
            children=self.child_dict[0])

        # build effects
        self.effect_list = list()
        for reg_idx in reg_out_list:
            label_map = glow.graph.get_label_map(reg_idx_list=[reg_idx, ],
                                                 mask_idx=exp.mask_idx,
                                                 children=self.child_dict[0])
            pval_fwer = self.pval[reg_idx]
            eff = glow.effect.Effect.from_exp_mask(mask=label_map > -1,
                                                   exp=exp,
                                                   reg_idx=reg_idx,
                                                   pval_fwer=pval_fwer)
            self.effect_list.append(eff)

    def _run_on_cloud(self, exp, n_perm, n_perm_adj, n_perm_prune,
                     alpha_fwer, alpha_prune, min_size, verbose,
                     cloud_config, **kwargs):
        """run permutation processing on AWS Batch and finish locally."""
        from glow.aws import AWSBatchRunner
        import uuid
        
        print('running analysis on AWS cloud...')
        
        # generate experiment ID
        experiment_id = f'glow_{uuid.uuid4().hex[:8]}'
        
        # prepare analysis kwargs (without cloud_config)
        ana_kwargs = {
            'get_stat': self.get_stat,
            'n_perm_adj': n_perm_adj,
            'n_perm_prune': n_perm_prune,
            'alpha_fwer': alpha_fwer,
            'alpha_prune': alpha_prune,
            'min_size': min_size
        }
        ana_kwargs.update(kwargs)
        
        # initialize runner
        runner = AWSBatchRunner(cloud_config)
        
        # upload experiment
        print('uploading experiment data...')
        runner.upload_experiment(exp, ana_kwargs, experiment_id)
        
        # submit jobs
        print('submitting permutation jobs...')
        submission = runner.submit_jobs(
            experiment_id=experiment_id,
            n_perm=n_perm,
            skip_completed=True
        )
        
        if submission.get('cancelled'):
            raise RuntimeError('job submission cancelled')
        
        # monitor progress
        if verbose:
            runner.monitor_jobs(submission['job_ids'])
        else:
            print(f'submitted {submission["n_jobs"]} jobs')
            print('use runner.monitor_jobs(job_ids) to track progress')
        
        # download results
        print('downloading results...')
        from pathlib import Path
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = runner.download_results(
                experiment_id=experiment_id,
                n_perm=n_perm,
                output_dir=Path(tmpdir)
            )
            
            # reconstruct analysis state from results
            b, num_img, num_vox = exp.y.shape
            num_reg = num_vox * 2 - 1
            
            self.child_dict = {}
            self.stat = np.full((n_perm + 1, num_reg), fill_value=-1.0)
            
            for perm_idx, result in results.items():
                self.child_dict[perm_idx] = result['children']
                self.stat[perm_idx, :] = result['stat']
        
        # complete analysis locally using shared finalization method
        print('completing analysis locally...')
        self._finalize_analysis(exp, n_perm, n_perm_adj, n_perm_prune,
                               alpha_fwer, alpha_prune, min_size)
        
        print(f'cloud analysis complete: found {len(self.effect_list)} effects')