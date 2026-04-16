import pickle
import shutil
import tempfile
import warnings
from bisect import bisect_left
from collections import namedtuple
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pygam import LinearGAM, s
from tqdm import tqdm


GAMFitResult = namedtuple('GAMFitResult',
                          ['gam', 'mu_fn', 'r2', 'size_adjusted'])
"""Return type of fit_size_gam.

Attributes:
    gam: fitted LinearGAM object, or None if fallback used.
    mu_fn: callable(size_array) -> predicted E[stat | H0]. When
        size_adjusted is False, returns zeros (identity adjustment).
    r2: R² of the fit, or None if fallback used.
    size_adjusted (bool): True if the GAM was fit, False if the
        low-data fallback was used.
"""

import glow.effect
import glow.graph
from ._base import Analysis, _sanitize_adjusted_stat
from glow.experiment.exper import ExperimentScaled
from .mancova import get_llr, stat_dict_inv
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
                 n_perm_fwer_size_adjust=25,
                 alpha_fwer=.05, min_size=1, verbose=False,
                 n_jobs_perm=1, cloud_config=None, perm_dir=None,
                 **kwargs):
        """
        Args:
            exp: Experiment to analyze
            n_perm_fwer: Number of permutations for FWER control
            n_perm_fwer_size_adjust: Number of held-out permutations used
                exclusively to fit the size-adjustment GAM (default 25).
                These are independent of the n_perm_fwer FWER permutations.
            alpha_fwer: Family-wise error rate
            min_size: Minimum region size
            verbose: Print progress
            n_jobs_perm: Number of parallel jobs for permutations
                (1=serial, -1=all cores)
            cloud_config: CloudConfig for AWS execution (if None, runs
                locally)
            perm_dir: path for permutation result files.  If provided,
                results are kept on disk for post-hoc inspection; if
                None a temp directory is created and cleaned up.
                Existing results in the directory are reused (resume).
        """
        super().__init__(exp, **kwargs)
        self.verbose = verbose

        if cloud_config is not None:
            self._run_on_cloud(exp, n_perm_fwer,
                              alpha_fwer, min_size, verbose,
                              cloud_config,
                              n_perm_fwer_size_adjust=n_perm_fwer_size_adjust,
                              perms_per_job=kwargs.pop('perms_per_job', None),
                              **kwargs)
            return

        b, num_img, num_vox = exp.y.shape

        # Permutation index layout:
        #   0                                        = observed data
        #   1..n_perm_fwer                           = FWER null permutations
        #   n_perm_fwer+1..n_perm_fwer+n_perm_fwer_size_adjust = held-out fit permutations
        fit_start = n_perm_fwer + 1
        fit_end = n_perm_fwer + n_perm_fwer_size_adjust
        all_perm_indices = list(range(fit_end + 1))

        # set up directory for per-permutation result files
        _cleanup_dir = perm_dir is None
        if perm_dir is None:
            perm_dir = Path(tempfile.mkdtemp(prefix='glow_perm_'))
        else:
            perm_dir = Path(perm_dir)
            perm_dir.mkdir(parents=True, exist_ok=True)

        # scan for existing results (resume support)
        existing = set()
        fit_sizes, fit_stats = [], []
        for f in sorted(perm_dir.glob('*_result.pkl')):
            try:
                perm_idx = int(f.name.split('_')[0])
            except ValueError:
                continue
            existing.add(perm_idx)
            if perm_idx >= fit_start:
                with open(f, 'rb') as fh:
                    r = pickle.load(fh)
                fit_sizes.append(np.asarray(r['size'], dtype=float))
                fit_stats.append(np.asarray(r['stat'], dtype=float))
                del r

        todo = [i for i in all_perm_indices if i not in existing]
        if existing and verbose:
            print(f'  resumed: {len(existing)} permutations found on disk, '
                  f'{len(todo)} remaining')

        if verbose:
            print(f'  [1/3] clustering {len(todo)} permutations '
                  f'({num_vox} voxels, {n_perm_fwer} FWER + {n_perm_fwer_size_adjust} fit) ...')

        if n_jobs_perm not in (0, 1) and todo:
            results = Parallel(
                n_jobs=n_jobs_perm,
                verbose=10 if verbose else 0,
            )(delayed(self._process_permutation)(exp, perm_idx)
              for perm_idx in todo)
            for r in results:
                p = r['perm_idx']
                with open(perm_dir / f'{p:06d}_result.pkl', 'wb') as fh:
                    pickle.dump(r, fh)
                if p >= fit_start:
                    fit_sizes.append(np.asarray(r['size'], dtype=float))
                    fit_stats.append(np.asarray(r['stat'], dtype=float))
                del r
        else:
            for perm_idx in tqdm(todo, desc='permutations',
                                 disable=not verbose):
                r = self._process_permutation(exp, perm_idx)
                with open(perm_dir / f'{r["perm_idx"]:06d}_result.pkl',
                          'wb') as fh:
                    pickle.dump(r, fh)
                if perm_idx >= fit_start:
                    fit_sizes.append(np.asarray(r['size'], dtype=float))
                    fit_stats.append(np.asarray(r['stat'], dtype=float))
                del r

        # fit GAM from collected size/stat arrays
        if verbose:
            print(f'  [2/3] fitting size-adjustment GAM ...')
        all_sizes = np.concatenate(fit_sizes)
        all_stats = np.concatenate(fit_stats)
        fit = self.fit_size_gam(all_sizes, all_stats)
        self.adj_gam = fit.gam
        self.size_adjusted = fit.size_adjusted
        mu_fn = fit.mu_fn
        if verbose and fit.r2 is not None:
            print(f'         R²={fit.r2:.4f}')
        elif verbose:
            print(f'         size adjustment disabled (low-data fallback)')

        # load observed (perm 0) — kept permanently
        with open(perm_dir / f'{0:06d}_result.pkl', 'rb') as fh:
            r0 = pickle.load(fh)
        stat_0 = np.asarray(r0['stat'], dtype=float)
        size_0 = np.asarray(r0['size'], dtype=float)
        children_0 = r0['children']
        del r0

        # pass 2: sweep temp files for adjusted max-stats
        if verbose:
            print(f'  [3/3] computing FWER p-values + pruning ...')
        reg_active = size_0 >= min_size
        stat_max_list = []
        for perm_idx in range(n_perm_fwer + 1):
            with open(perm_dir / f'{perm_idx:06d}_result.pkl', 'rb') as fh:
                r = pickle.load(fh)
            adj = (np.asarray(r['stat'], dtype=float)
                   - mu_fn(np.asarray(r['size'], dtype=float)))
            adj = _sanitize_adjusted_stat(adj)
            if reg_active.any() and np.isfinite(adj[reg_active]).any():
                stat_max_list.append(float(np.nanmax(adj[reg_active])))
            else:
                stat_max_list.append(float('-inf'))
            del r, adj
        stat_max_sorted = np.sort(stat_max_list)

        self._finalize_analysis(
            exp, n_perm_fwer, stat_0, size_0, children_0,
            mu_fn, stat_max_sorted,
            alpha_fwer, min_size)

        if _cleanup_dir:
            shutil.rmtree(perm_dir, ignore_errors=True)

    @classmethod
    def from_precomputed(cls, *, exp, get_stat, adj_gam=None, verbose=False):
        """Construct a shell for finalization without running full __init__.

        The caller should then invoke _finalize_analysis() to compute p-values.
        """
        obj = cls.__new__(cls)
        obj.exp = exp if isinstance(exp, ExperimentScaled) else ExperimentScaled.from_exp(exp)
        obj.get_stat = get_stat
        obj.adj_gam = adj_gam
        obj.verbose = verbose
        return obj

    def _process_permutation(self, exp, perm_idx):
        """Run one permutation: cluster, compute stats and sizes."""
        _exp = exp.permute(perm_idx)
        children = cluster(exp=_exp)

        stat_row = self.get_stat_perm(exp=_exp, children=children)
        stat = stat_row.ravel()
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
    def rerun_permutation(cls, exp, perm_idx, get_stat=get_llr):
        """Re-run a single permutation for inspection.

        Since permutations are deterministic given ``perm_idx``, this
        faithfully reproduces the result without needing stored data.

        Returns:
            dict with keys ``perm_idx``, ``children``, ``stat``, ``size``
        """
        ana = cls.from_precomputed(exp=exp, get_stat=get_stat)
        return ana._process_permutation(exp, perm_idx)

    _N_SPLINES = 10

    _MIN_GAM_FIT_POINTS = 50

    @classmethod
    def fit_size_gam(cls, size, stat, n_splines=None):
        """Fit a GAM: stat ~ f(log10(size)) using pygam.

        Args:
            size: (N,) region sizes (voxels)
            stat: (N,) stat values
            n_splines: number of splines (default: cls._N_SPLINES)

        Returns:
            GAMFitResult(gam, mu_fn, r2, size_adjusted): see dataclass
            docstring.  When ``valid.sum() < _MIN_GAM_FIT_POINTS``, the
            fallback identity mu_fn is used and a ``RuntimeWarning`` is
            emitted; ``size_adjusted`` is False so callers can surface
            this to users.
        """
        if n_splines is None:
            n_splines = cls._N_SPLINES

        sz = np.asarray(size, dtype=float)
        y = np.asarray(stat, dtype=float)
        valid = np.isfinite(y) & np.isfinite(sz) & (sz > 0)

        n_valid = int(valid.sum())
        if n_valid < cls._MIN_GAM_FIT_POINTS:
            warnings.warn(
                f'fit_size_gam: only {n_valid} valid (size, stat) pairs '
                f'(threshold is {cls._MIN_GAM_FIT_POINTS}); falling back '
                f'to identity mu_fn.  No size adjustment will be applied '
                f'-- downstream FWER max-stats will reflect raw stats.  '
                f'Consider increasing n_perm_fwer_size_adjust or '
                f'verifying that the stat function is well-behaved on '
                f'small regions.',
                RuntimeWarning,
                stacklevel=2,
            )
            return GAMFitResult(
                gam=None,
                mu_fn=lambda sz: np.zeros_like(np.asarray(sz, dtype=float)),
                r2=None,
                size_adjusted=False,
            )

        log_size = np.log10(sz[valid])
        y_valid = y[valid]

        gam = LinearGAM(s(0, n_splines=n_splines))
        gam.gridsearch(log_size, y_valid, progress=False)

        pred = gam.predict(log_size)
        ss_res = np.sum((y_valid - pred) ** 2)
        ss_tot = np.sum((y_valid - y_valid.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        def mu_fn(sz, _gam=gam):
            sz = np.asarray(sz, dtype=float)
            return _gam.predict(np.log10(np.maximum(sz, 1)))

        return GAMFitResult(gam=gam, mu_fn=mu_fn, r2=r2,
                            size_adjusted=True)

    @staticmethod
    def mu_fn_from_gam(gam):
        """Build a mu_fn callable from a fitted GAM object.

        Useful when reconstructing from a pickled analysis.
        """
        if gam is None:
            return lambda sz: np.zeros_like(np.asarray(sz, dtype=float))

        def mu_fn(sz, _gam=gam):
            sz = np.asarray(sz, dtype=float)
            return _gam.predict(np.log10(np.maximum(sz, 1)))
        return mu_fn

    def _finalize_analysis(self, exp, n_perm_fwer,
                          stat_0, size_0, children_0,
                          mu_fn, stat_max_sorted,
                          alpha_fwer, min_size,
                          prune_stat=None,
                          ):
        """Finalize: compute p-values from max-stat distribution, prune.

        Args:
            prune_stat (np.array): optional override for the stat array
                used by greedy pruning.  When None (default), ``stat_0``
                is used.  Pass e.g. LLR values here when the test
                statistic differs from the desired pruning criterion.
        """
        verbose = getattr(self, 'verbose', False)
        num_reg = stat_0.shape[0]

        llr_adjusted_0 = stat_0 - mu_fn(size_0.astype(float))
        llr_adjusted_0 = _sanitize_adjusted_stat(llr_adjusted_0)

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
            print('  pruning (greedy LLR) ...')

        _prune = stat_0 if prune_stat is None else prune_stat
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
        model = load_runtime_model('GLOW')
        if model is None:
            path = RUNTIME_MODEL_PATHS.get('GLOW')
            raise FileNotFoundError(
                f'GLOW runtime model not found at {path}.  '
                f'Regenerate with: '
                f'python -m glow.benchmark.runtime --profile experiment')
        return predict_runtime_sec(model, num_vox, b, num_img, n_perm=1)

    def _run_on_cloud(self, exp, n_perm_fwer,
                     alpha_fwer, min_size, verbose,
                     cloud_config, n_perm_fwer_size_adjust=25,
                     perms_per_job=None,
                     **kwargs):
        """Run full analysis on AWS Batch (permutations + synthesis).

        Submits N+1+n_perm_fwer_size_adjust permutation jobs, then a
        synthesis job that polls S3 for all results before running
        ``_finalize_analysis`` on the cloud.  The final pickled
        AnalysisGLOW is downloaded and its attributes are copied onto
        ``self``.
        """
        from glow.aws import AWSBatchRunner
        import uuid

        print('running analysis on AWS cloud...')

        experiment_id = f'glow_{uuid.uuid4().hex[:8]}'

        ana_kwargs = {
            'get_stat': self.get_stat,
            'n_perm_fwer_size_adjust': n_perm_fwer_size_adjust,
            'alpha_fwer': alpha_fwer,
            'min_size': min_size,
        }
        ana_kwargs.update(kwargs)

        runner = AWSBatchRunner(cloud_config)
        n_perm = n_perm_fwer + n_perm_fwer_size_adjust

        # interactive batch-size selection when perms_per_job not specified
        if perms_per_job is None:
            perm_sec = self._estimate_perm_sec(exp)
            AWSBatchRunner.estimate_batch_table(
                n_perm + 1, perm_sec)
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
            n_perm=n_perm,
            skip_completed=True,
            perms_per_job=perms_per_job,
        )

        if submission.get('cancelled'):
            raise RuntimeError('job submission cancelled')

        perm_job_ids = submission['job_ids']
        job_info_map = submission.get('job_info_map', {})

        print('submitting synthesis job...')
        synth_job_id = runner.submit_synthesis_job(experiment_id, n_perm)

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
            'prune_info', 'adj_gam',
        ]
        for attr in _COPY_ATTRS:
            if hasattr(remote_ana, attr):
                setattr(self, attr, getattr(remote_ana, attr))

        print(f'cloud analysis complete: found {len(self.effect_list)} effects')
