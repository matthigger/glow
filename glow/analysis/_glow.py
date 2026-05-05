import pickle
import shutil
import tempfile
from bisect import bisect_left
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

import glow.effect
import glow.graph
from ._base import Analysis, _sanitize_adjusted_stat
from glow.experiment.exper import ExperimentScaled
from .mancova import get_llr
from .prune import prune_greedy
from .cluster import cluster


class AnalysisGLOW(Analysis):
    """search a hierarchical segmentation for significant effects.

    Uses a disk-backed streaming pipeline: each permutation result is
    written to a temp directory, regression is accumulated online, and
    finalization reads the results in a single sweep.

    Attributes:
        children (np.array): (num_leaf - 1, 2) Ward children for observed data
        stat (np.array): (num_reg,) raw test statistics for observed
        size (np.array): (num_reg,) region sizes for observed
        pval (np.array): (num_reg,) FWER-controlled p-values
        effect_list (list): discovered Effect objects
    """

    def __init__(self, exp, n_perm_fwer,
                 n_perm_inner=200,
                 alpha_fwer=.05, min_size=1, verbose=False,
                 n_jobs_perm=1, cloud_config=None, perm_dir=None,
                 cluster_mode="ward's (q1)",
                 **kwargs):
        """
        Args:
            exp: Experiment to analyze
            n_perm_fwer: number of outer FL permutations for FWER control
            n_perm_inner: number of inner FL permutations used to
                estimate per-region (mu, std) of the H0 LLR distribution
                on the merged graph.  Inner seeds occupy indices
                ``n_perm_fwer + 1 .. n_perm_fwer + n_perm_inner``.
            alpha_fwer: family-wise error rate
            min_size: minimum region size
            verbose: print progress
            n_jobs_perm: parallel jobs for outer perms (1=serial, -1=all)
            cloud_config: CloudConfig for AWS execution (None = local)
            perm_dir: directory for per-perm result pickles.  If None
                a temp dir is created and cleaned up; if given, files
                are kept and reused on resume.
            cluster_mode: Ward projection.  Must be one of the keys in
                ``glow.analysis.cluster._MODES``.  Default
                ``"ward's (q1)"`` (Focus) projects onto the contrast
                subspace; ``"ward's (q0, q1)"`` (GLM Error) keeps bias
                + contrast; ``"ward's (all)"`` (Naive) clusters raw y.
        """
        super().__init__(exp, **kwargs)
        self.verbose = verbose
        self.cluster_mode = cluster_mode

        if cloud_config is not None:
            self._run_on_cloud(exp, n_perm_fwer, n_perm_inner,
                              alpha_fwer, min_size, verbose,
                              cloud_config,
                              perms_per_job=kwargs.pop('perms_per_job', None),
                              cluster_mode=cluster_mode,
                              **kwargs)
            return

        b, num_img, num_vox = exp.y.shape

        # Permutation index layout:
        #   0                                    = observed data
        #   1..n_perm_fwer                       = outer FL nulls (FWER)
        #   n_perm_fwer+1..n_perm_fwer+n_perm_inner = held-out inner FL nulls
        all_perm_indices = list(range(n_perm_fwer + 1))

        _cleanup_dir = perm_dir is None
        if perm_dir is None:
            perm_dir = Path(tempfile.mkdtemp(prefix='glow_perm_'))
        else:
            perm_dir = Path(perm_dir)
            perm_dir.mkdir(parents=True, exist_ok=True)

        # scan for existing per-perm result files (resume support)
        existing = set()
        for f in sorted(perm_dir.glob('*_result.pkl')):
            try:
                existing.add(int(f.name.split('_')[0]))
            except ValueError:
                continue

        todo = [i for i in all_perm_indices if i not in existing]
        if existing and verbose:
            print(f'  resumed: {len(existing)} permutations on disk, '
                  f'{len(todo)} remaining')

        if verbose:
            print(f'  [1/2] outer perms: clustering {len(todo)} '
                  f'permutations ({num_vox} voxels, '
                  f'{n_perm_fwer} FWER) ...')

        if n_jobs_perm not in (0, 1) and todo:
            # Pin BLAS threads to 1 inside each worker.  Without this,
            # numpy / scipy / sklearn each spawn O(N_cores) BLAS threads
            # inside every joblib worker, leading to thread-pool
            # explosion and occasional SIGSEGV on large WGN volumes.
            with parallel_config(backend='loky', inner_max_num_threads=1):
                results = Parallel(
                    n_jobs=n_jobs_perm,
                    verbose=10 if verbose else 0,
                )(delayed(self._process_permutation)(exp, perm_idx)
                  for perm_idx in todo)
            for r in results:
                p = r['perm_idx']
                with open(perm_dir / f'{p:06d}_result.pkl', 'wb') as fh:
                    pickle.dump(r, fh)
                del r
        else:
            for perm_idx in tqdm(todo, desc='outer perms',
                                 disable=not verbose):
                r = self._process_permutation(exp, perm_idx)
                with open(perm_dir / f'{r["perm_idx"]:06d}_result.pkl',
                          'wb') as fh:
                    pickle.dump(r, fh)
                del r

        if verbose:
            print(f'  [2/2] per_region_z finalization '
                  f'({n_perm_inner} inner perms) ...')

        self._finalize_per_region_z(
            exp, perm_dir, n_perm_fwer, n_perm_inner,
            alpha_fwer, min_size,
            inner_seed_offset=n_perm_fwer + 1)

        if _cleanup_dir:
            shutil.rmtree(perm_dir, ignore_errors=True)

    def _finalize_per_region_z(self, exp, perm_dir, n_perm_fwer,
                                n_perm_inner, alpha_fwer, min_size,
                                inner_seed_offset):
        """Per-region permutation z-scoring with Westfall-Young FWER.

        Builds the merged graph from all outer-perm trees, runs
        ``n_perm_inner`` held-out FL permutations to estimate
        ``(mu_r, std_r)`` per merged region, then z-scores each outer
        perm's LLRs against its tree's regions and computes the
        max-z null.
        """
        verbose = getattr(self, 'verbose', False)
        num_vox = exp.y.shape[2]

        children_list = []
        stat_list = []
        size_list = []
        for k in range(n_perm_fwer + 1):
            with open(perm_dir / f'{k:06d}_result.pkl', 'rb') as fh:
                r = pickle.load(fh)
            children_list.append(np.asarray(r['children']))
            stat_list.append(np.asarray(r['stat'], dtype=float))
            size_list.append(np.asarray(r['size'], dtype=float))

        if verbose:
            print(f'  per_region_z: merging {n_perm_fwer + 1} trees ...')
        map_to_new, merged_children, _ = glow.graph.graph_merge(
            n_common=num_vox, children_list=children_list)
        num_merged = num_vox + merged_children.shape[0]

        if verbose:
            print(f'  per_region_z: {n_perm_inner} inner perms on '
                  f'{num_merged} merged regions ...')
        LLR_inner = np.full((n_perm_inner, num_merged), np.nan)
        for k_inner in tqdm(range(n_perm_inner), desc='inner perms',
                             disable=not verbose):
            seed = inner_seed_offset + k_inner
            _exp = exp.permute(seed)
            LLR_inner[k_inner, :] = self.get_stat_perm(
                _exp, children=merged_children)

        mu_merged = np.nanmean(LLR_inner, axis=0)
        sigma_merged = np.nanstd(LLR_inner, axis=0, ddof=1)
        sigma_merged[sigma_merged < 1e-12] = 1.0

        # max-z null over outer perms
        max_z_list = []
        for k_outer in range(n_perm_fwer + 1):
            outer_to_merged = np.concatenate([
                np.arange(num_vox), map_to_new[k_outer]])
            mu_k = mu_merged[outer_to_merged]
            sigma_k = sigma_merged[outer_to_merged]
            z_k = (stat_list[k_outer] - mu_k) / sigma_k
            z_k = _sanitize_adjusted_stat(z_k)
            size_k = size_list[k_outer]
            active = size_k >= min_size
            if active.any() and np.isfinite(z_k[active]).any():
                max_z_list.append(float(np.nanmax(z_k[active])))
            else:
                max_z_list.append(float('-inf'))

        stat_max_sorted = np.sort(max_z_list)

        # observed (k_outer = 0) z-scored LLR, mapped from T_0's region
        # indexing back to merged-graph indexing.
        outer_to_merged_0 = np.concatenate([
            np.arange(num_vox), map_to_new[0]])
        mu_arr_0 = mu_merged[outer_to_merged_0]
        sigma_arr_0 = sigma_merged[outer_to_merged_0]
        llr_adjusted_0 = (stat_list[0] - mu_arr_0) / sigma_arr_0

        self._finalize_analysis(
            exp, n_perm_fwer,
            stat_list[0], size_list[0], children_list[0],
            llr_adjusted_0, stat_max_sorted,
            alpha_fwer, min_size)

        # diagnostics
        self._merged_children = merged_children
        self._map_to_new = map_to_new
        self._mu_per_region = mu_merged
        self._sigma_per_region = sigma_merged

    @classmethod
    def from_precomputed(cls, *, exp, get_stat=None, verbose=False,
                         cluster_mode="ward's (q1)"):
        """Construct an empty shell without running ``__init__``.

        Caller is responsible for invoking ``_finalize_per_region_z()``
        (or running outer perms first) to populate the analysis.
        """
        obj = cls.__new__(cls)
        obj.exp = (exp if isinstance(exp, ExperimentScaled)
                   else ExperimentScaled.from_exp(exp))
        if get_stat is None:
            get_stat = get_llr
        obj.get_stat = get_stat
        obj.verbose = verbose
        obj.cluster_mode = cluster_mode
        obj.n_jobs_perm = 1
        return obj

    def _process_permutation(self, exp, perm_idx):
        """Run one permutation: cluster, compute stats and sizes."""
        _exp = exp.permute(perm_idx)
        children = cluster(exp=_exp, mode=self.cluster_mode)

        stat = self.get_stat_perm(exp=_exp, children=children)
        num_vox = _exp.y.shape[2]
        size = glow.graph.node_sum(np.ones(num_vox, dtype=int), children)

        del _exp
        return {
            'perm_idx': perm_idx,
            'children': children,
            'stat': stat,
            'size': size,
        }

    @classmethod
    def rerun_permutation(cls, exp, perm_idx, get_stat=get_llr,
                          cluster_mode="ward's (q1)"):
        """Re-run a single permutation for inspection.

        Since permutations are deterministic given ``perm_idx``, this
        faithfully reproduces the result without needing stored data.

        Returns:
            dict with keys ``perm_idx``, ``children``, ``stat``, ``size``
        """
        ana = cls.from_precomputed(exp=exp, get_stat=get_stat)
        ana.cluster_mode = cluster_mode
        return ana._process_permutation(exp, perm_idx)

    def _finalize_analysis(self, exp, n_perm_fwer,
                          stat_0, size_0, children_0,
                          llr_adjusted_0, stat_max_sorted,
                          alpha_fwer, min_size,
                          prune_stat=None,
                          ):
        """Compute p-values from the max-z null and prune.

        Args:
            stat_0: raw LLR per region for the observed tree.
            size_0: region sizes for the observed tree.
            children_0: observed Ward children.
            llr_adjusted_0: per-region z-scored LLR for the observed
                tree (already adjusted by per-region (mu, std)).
            stat_max_sorted: sorted max-z null distribution from the
                outer permutations (length n_perm_fwer + 1).
            prune_stat: optional override for the array used to rank
                pruning candidates.  Defaults to ``llr_adjusted_0``
                (z-scored), which makes the pruning rank consistent
                with the FWER threshold.  Pass ``stat_0`` to rank by
                raw LLR instead.
        """
        verbose = getattr(self, 'verbose', False)
        num_reg = stat_0.shape[0]

        llr_adjusted_0 = _sanitize_adjusted_stat(np.asarray(llr_adjusted_0,
                                                            dtype=float))

        self.alpha_fwer = alpha_fwer
        self.size = size_0
        self.stat = stat_0
        self.llr_adjusted_0 = llr_adjusted_0
        self.children = children_0

        reg_active = size_0 >= min_size
        if not reg_active.any():
            pval = np.full(num_reg, fill_value=np.nan)
        else:
            pval = np.full(num_reg, fill_value=-1.0)
            for reg_idx, z in enumerate(llr_adjusted_0):
                if np.isnan(z):
                    pval[reg_idx] = np.nan
                    continue
                n_total = len(stat_max_sorted)
                pval[reg_idx] = max(
                    1 - bisect_left(stat_max_sorted, z) / n_total,
                    1 / n_total)
            pval[~reg_active] = np.nan
        self.pval = pval

        n_total = len(stat_max_sorted)
        if n_total > 0:
            crit_idx = min(int(np.ceil(n_total * (1 - alpha_fwer))),
                           n_total - 1)
            self.adj_crit = float(stat_max_sorted[crit_idx])
        else:
            self.adj_crit = None

        self.sig_reg_list = list(np.where(self.pval <= alpha_fwer)[0])
        if verbose:
            print(f'  {len(self.sig_reg_list)} significant regions '
                  f'(alpha_fwer={alpha_fwer})')
            print('  pruning (greedy on z-scored LLR) ...')

        # Default: rank pruning by the z-scored LLR (consistent with
        # the FWER threshold).  Caller may override via ``prune_stat``
        # — RunPruneCompare does this to compare raw-LLR vs LLR-z gain.
        _prune = llr_adjusted_0 if prune_stat is None else prune_stat
        stat_gain = np.nan_to_num(_prune.astype(float), nan=0.0,
                                  posinf=0.0, neginf=0.0)

        reg_out_list, self.prune_info = prune_greedy(
            sig_reg_list=self.sig_reg_list,
            children=children_0,
            stat=stat_gain)

        self.effect_list = list()
        for reg_idx in reg_out_list:
            label_map = glow.graph.get_label_map(
                reg_idx_list=[reg_idx],
                mask_idx=exp.mask_idx,
                children=children_0)
            pval_fwer = self.pval[reg_idx]
            eff = glow.effect.Effect.from_exp_mask(
                mask=label_map > -1, exp=exp,
                reg_idx=reg_idx, pval_fwer=pval_fwer)
            self.effect_list.append(eff)

        if verbose:
            n_disc = len(self.effect_list)
            n_pruned = len(self.sig_reg_list) - n_disc
            print(f'  done: {n_disc} discovered, {n_pruned} pruned')

    @staticmethod
    def _estimate_perm_sec(exp):
        """Estimate seconds per permutation from the runtime benchmark model.

        Raises:
            FileNotFoundError: if no GLOW runtime model is available.
                Previously this fell back to ``max(num_vox * 3e-4, 5.0)``,
                which gave a plausible-looking number with no grounding
                in measurement and would silently mis-size AWS batch jobs.
        """
        b, num_img, num_vox = exp.y.shape
        from glow.benchmark.runtime import (
            RUNTIME_MODEL_PATHS, load_runtime_model, predict_runtime_sec)
        model = load_runtime_model('GLOW', platform='aws')
        if model is None:
            path = RUNTIME_MODEL_PATHS['aws'].get('GLOW')
            raise FileNotFoundError(
                f'GLOW AWS runtime model not found at {path}.  '
                f'Regenerate with: '
                f'python -m glow.benchmark.runtime --profile experiment --cloud')
        return predict_runtime_sec(model, num_vox, b, num_img, n_perm=1)

    def _run_on_cloud(self, exp, n_perm_fwer, n_perm_inner,
                     alpha_fwer, min_size, verbose,
                     cloud_config,
                     perms_per_job=None,
                     cluster_mode="ward's (q1)",
                     **kwargs):
        """Run full analysis on AWS Batch (outer perms + synthesis).

        Submits ``n_perm_fwer + 1`` outer-perm jobs, then a synthesis
        job that polls S3 for all results, runs ``n_perm_inner`` inner
        FL perms locally on the synth worker, and runs the per-region-z
        finalization.  The final pickled AnalysisGLOW is downloaded and
        its attributes are copied onto ``self``.
        """
        from glow.aws import AWSBatchRunner
        import uuid

        print('running analysis on AWS cloud...')

        experiment_id = f'glow_{uuid.uuid4().hex[:8]}'

        ana_kwargs = {
            'get_stat': self.get_stat,
            'n_perm_inner': n_perm_inner,
            'alpha_fwer': alpha_fwer,
            'min_size': min_size,
            'cluster_mode': cluster_mode,
        }
        ana_kwargs.update(kwargs)

        runner = AWSBatchRunner(cloud_config)

        if perms_per_job is None:
            perm_sec = self._estimate_perm_sec(exp)
            AWSBatchRunner.estimate_batch_table(
                n_perm_fwer + 1, perm_sec)
            choice = input('\nperms_per_job (or "q" to cancel): ').strip()
            if choice.lower() in ('q', 'quit', 'cancel'):
                print('cancelled.')
                return
            perms_per_job = int(choice)
            print()

        print('uploading experiment data...')
        runner.upload_experiment(exp, ana_kwargs, experiment_id)

        print('submitting permutation jobs...')
        submission = runner.submit_jobs(
            experiment_id=experiment_id,
            n_perm=n_perm_fwer,
            skip_completed=True,
            perms_per_job=perms_per_job,
        )

        if submission.get('cancelled'):
            raise RuntimeError('job submission cancelled')

        perm_job_ids = submission['job_ids']
        job_info_map = submission.get('job_info_map', {})

        print('submitting synthesis job...')
        synth_job_id = runner.submit_synthesis_job(experiment_id, n_perm_fwer)

        all_job_ids = perm_job_ids + [synth_job_id]

        if verbose:
            runner.monitor_jobs(all_job_ids, job_info_map=job_info_map)
        else:
            print(f'submitted {len(perm_job_ids)} perm jobs + 1 synthesis job')

        print('downloading final analysis...')
        remote_ana = runner.download_final_analysis(experiment_id)

        _COPY_ATTRS = [
            'children', 'stat', 'size', 'pval', 'llr_adjusted_0',
            'sig_reg_list', 'effect_list', 'alpha_fwer', 'adj_crit',
            'prune_info',
            '_merged_children', '_map_to_new',
            '_mu_per_region', '_sigma_per_region',
        ]
        for attr in _COPY_ATTRS:
            if hasattr(remote_ana, attr):
                setattr(self, attr, getattr(remote_ana, attr))

        print(f'cloud analysis complete: found {len(self.effect_list)} effects')
