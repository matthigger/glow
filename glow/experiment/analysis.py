import pickle
import shutil
import tempfile
from bisect import bisect_left
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import label
from tqdm import tqdm

import glow.effect
import glow.graph
# glow.vba imported lazily when needed (requires FSL for TFCE)
from .cluster import cluster, count_components
from .exper import ExperimentScaled
from .mancova import get_llr, get_neg_wilks
from .prune import prune, prune_node, prune_tree, prune_tree_dp


# best regression model per stat (from stat_vs_size benchmark)
_STAT_MODEL = {
    get_llr: 'sqrt',
    get_neg_wilks: 'sqrt',
}
_DEFAULT_MODEL = 'power_law'


class Analysis:
    """performs effect discovery (glow or TFCE) and computes FWER p-values.

    Attributes:
        exp (Experiment): source data
        get_stat (callable): accepts (e, h, n) and returns a scalar
            statistic (see mancova.py)
    """

    def __init__(self, exp, get_stat=get_llr, n_jobs_perm=1):
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
        for reg_idx, size, e, h in glow.graph.iter_stat(exp=exp,
                                                       children=children,
                                                       n_perm=n_perm):
            for perm_idx in range(n_rows):
                stat[perm_idx, reg_idx] = self.get_stat(e=e[:, :, perm_idx],
                                                        h=h[:, :, perm_idx],
                                                        n=size)
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

    Uses a disk-backed streaming pipeline: each permutation result is
    written to a temp directory, regression is accumulated online, and
    finalization reads the results in a single sweep.

    Attributes:
        child_dict (dict): {0: children_0} — observed-permutation children
        stat_0 (np.array): (num_reg,) raw test statistics for observed
        size_0 (np.array): (num_reg,) region sizes for observed
        pval (np.array): (num_reg,) FWER-controlled p-values
        effect_list (list): discovered Effect objects
    """

    def __init__(self, exp, n_perm, n_perm_prune=100,
                 alpha_fwer=.05, alpha_prune=.05, min_size=1, verbose=False,
                 n_jobs_perm=1, cloud_config=None, perm_dir=None,
                 prune_method='node', prune_geom_exp_eff=None, **kwargs):
        """
        Args:
            exp: Experiment to analyze
            n_perm: Number of permutations
            n_perm_prune: Number of pruning permutations (used for
                homogeneity-test pruning and node-gain calibration)
            alpha_fwer: Family-wise error rate
            alpha_prune: Pruning alpha (quantile level for both
                homogeneity and node-gain calibration)
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
            prune_method: 'node' for per-node-LLR DP pruning (default),
                'homo' for homogeneity-test pruning, 'tree' for greedy
                tree-wide adjusted-likelihood pruning, 'tree_dp' for
                DP tree-wide pruning
            prune_geom_exp_eff: expected number of effect regions under
                the geometric prior.  if None (default), lambda is
                calibrated from permutations.  if given, uses the
                analytic formula lambda = log(1 + 1/exp_eff) instead.
        """
        super().__init__(exp, **kwargs)
        self.verbose = verbose

        if cloud_config is not None:
            self._run_on_cloud(exp, n_perm, n_perm_prune,
                              alpha_fwer, alpha_prune, min_size, verbose,
                              cloud_config,
                              prune_method=prune_method,
                              prune_geom_exp_eff=prune_geom_exp_eff,
                              **kwargs)
            return

        b, num_img, num_vox = exp.y.shape
        model = _STAT_MODEL.get(self.get_stat, _DEFAULT_MODEL)

        # set up directory for per-permutation result files
        _cleanup_dir = perm_dir is None
        if perm_dir is None:
            perm_dir = Path(tempfile.mkdtemp(prefix='glow_perm_'))
        else:
            perm_dir = Path(perm_dir)
            perm_dir.mkdir(parents=True, exist_ok=True)

        # scan for existing results (resume support)
        existing = set()
        XtX, Xty = None, None
        for f in sorted(perm_dir.glob('*_result.pkl')):
            try:
                perm_idx = int(f.name.split('_')[0])
            except ValueError:
                continue
            existing.add(perm_idx)
            if perm_idx > 0:
                with open(f, 'rb') as fh:
                    r = pickle.load(fh)
                XtX, Xty = self.accumulate_regression(
                    r['size'], r['stat'], model, XtX, Xty)
                del r

        todo = [i for i in range(n_perm + 1) if i not in existing]
        if existing and verbose:
            print(f'  resumed: {len(existing)} permutations found on disk, '
                  f'{len(todo)} remaining')

        if verbose:
            print(f'  [1/3] clustering {len(todo)} permutations '
                  f'({num_vox} voxels) ...')

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
                if p > 0:
                    XtX, Xty = self.accumulate_regression(
                        r['size'], r['stat'], model, XtX, Xty)
                del r
        else:
            for perm_idx in tqdm(todo, desc='permutations',
                                 disable=not verbose):
                r = self._process_permutation(exp, perm_idx)
                with open(perm_dir / f'{r["perm_idx"]:06d}_result.pkl',
                          'wb') as fh:
                    pickle.dump(r, fh)
                if perm_idx > 0:
                    XtX, Xty = self.accumulate_regression(
                        r['size'], r['stat'], model, XtX, Xty)
                del r

        # fit regression from accumulated sufficient statistics
        if verbose:
            print(f'  [2/3] fitting size model ({model}) ...')
        mu_fn, _, beta = self.fit_size_regression_online(
            XtX, Xty, model, self.get_stat)
        self.adj_model = model
        self.adj_beta = beta

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
        for perm_idx in range(n_perm + 1):
            with open(perm_dir / f'{perm_idx:06d}_result.pkl', 'rb') as fh:
                r = pickle.load(fh)
            adj = (np.asarray(r['stat'], dtype=float)
                   - mu_fn(np.asarray(r['size'], dtype=float)))
            adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=-30.0)
            stat_max_list.append(
                float(np.nanmax(adj[reg_active])) if reg_active.any()
                else float('-inf'))
            del r, adj
        stat_max_sorted = np.sort(stat_max_list)

        self._finalize_analysis(
            exp, n_perm, stat_0, size_0, children_0,
            mu_fn, stat_max_sorted,
            n_perm_prune, alpha_fwer, alpha_prune, min_size,
            prune_method=prune_method,
            prune_geom_exp_eff=prune_geom_exp_eff)

        if _cleanup_dir:
            shutil.rmtree(perm_dir, ignore_errors=True)

    def _process_permutation(self, exp, perm_idx):
        """Run one permutation: cluster, compute stats and sizes."""
        _exp = exp.permute(perm_idx)
        children = cluster(exp=_exp)
        stat_row = self.get_stat_perm(exp=_exp, children=children)
        num_vox = _exp.y.shape[2]
        size = glow.graph.node_sum(np.ones(num_vox, dtype=int), children)
        del _exp
        return {
            'perm_idx': perm_idx,
            'children': children,
            'stat': stat_row.ravel(),
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
        ana = object.__new__(cls)
        ana.get_stat = get_stat
        return ana._process_permutation(exp, perm_idx)

    @staticmethod
    def _wls_fit(X, y, w):
        """Weighted least squares via sqrt-weight transformation."""
        sw = np.sqrt(w)[:, np.newaxis]
        beta, _, _, _ = np.linalg.lstsq(X * sw, y * sw.ravel(), rcond=None)
        return beta

    @staticmethod
    def predict_null_mean(size, model, beta):
        """Predict E[stat | H0] given region sizes, model name, and coeffs.

        Args:
            size: scalar or array of region sizes (voxels)
            model (str): 'sqrt' or 'power_law'
            beta (np.array): regression coefficients

        Returns:
            predicted mean stat under H0
        """
        sz = np.asarray(size, dtype=float)
        if model == 'sqrt':
            return beta[0] + beta[1] * np.sqrt(sz)
        elif model == 'power_law':
            return np.exp(beta[0] + beta[1] * np.log(np.maximum(sz, 1)))
        else:
            raise ValueError(f'Unknown model: {model}')

    @classmethod
    def fit_size_regression_online(cls, XtX, Xty, model, get_stat=None):
        """Fit size regression from pre-accumulated sufficient statistics.

        Args:
            XtX: (2, 2) accumulated X^T X matrix
            Xty: (2,) accumulated X^T y vector
            model: 'sqrt' or 'power_law'
            get_stat: stat function (stored on result for predict_null_mean)

        Returns:
            mu_fn: callable(size_array) -> predicted E[stat | H0]
            model: model name
            beta: regression coefficients
        """
        beta, _, _, _ = np.linalg.lstsq(XtX, Xty, rcond=None)
        def mu_fn(sz, _m=model, _b=beta):
            return cls.predict_null_mean(sz, _m, _b)
        return mu_fn, model, beta

    @staticmethod
    def accumulate_regression(size, stat, model, XtX=None, Xty=None):
        """Accumulate OLS sufficient statistics from one permutation row.

        Args:
            size: (num_reg,) region sizes for this permutation
            stat: (num_reg,) stat values for this permutation
            model: 'sqrt' or 'power_law'
            XtX: running (2, 2) matrix (None to initialize)
            Xty: running (2,) vector (None to initialize)

        Returns:
            XtX, Xty: updated sufficient statistics
        """
        s = np.asarray(size, dtype=float)
        y = np.asarray(stat, dtype=float)

        if model == 'sqrt':
            valid = np.isfinite(y) & (s > 0) & np.isfinite(s)
            x1 = np.sqrt(s[valid])
            yv = y[valid]
        else:
            valid = np.isfinite(y) & (y > 0) & (s > 0) & np.isfinite(s)
            x1 = np.log(s[valid])
            yv = np.log(y[valid])

        x0 = np.ones(x1.shape[0])
        if XtX is None:
            XtX = np.zeros((2, 2))
            Xty = np.zeros(2)
        XtX[0, 0] += x0 @ x0
        XtX[0, 1] += x0 @ x1
        XtX[1, 0] += x0 @ x1
        XtX[1, 1] += x1 @ x1
        Xty[0] += x0 @ yv
        Xty[1] += x1 @ yv
        return XtX, Xty

    def _finalize_analysis(self, exp, n_perm,
                          stat_0, size_0, children_0,
                          mu_fn, stat_max_sorted,
                          n_perm_prune, alpha_fwer, alpha_prune,
                          min_size, prune_method='node',
                          prune_geom_exp_eff=None):
        """Finalize: compute p-values from max-stat distribution, prune."""
        verbose = getattr(self, 'verbose', False)
        num_reg = stat_0.shape[0]

        llr_adjusted_0 = stat_0 - mu_fn(size_0.astype(float))
        llr_adjusted_0 = np.nan_to_num(llr_adjusted_0, nan=0.0,
                                        posinf=0.0, neginf=-30.0)

        self.alpha_fwer = alpha_fwer
        self.alpha_prune = alpha_prune
        self.size_0 = size_0
        self.stat_0 = stat_0
        self.llr_adjusted_0 = llr_adjusted_0
        self.child_dict = {0: children_0}

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

        self.sig_reg_list = list(np.where(self.pval <= alpha_fwer)[0])
        if verbose:
            print(f'  {len(self.sig_reg_list)} significant regions '
                  f'(alpha_fwer={alpha_fwer})')

        if prune_method == 'node':
            if verbose:
                _mode = (f'exp_eff={prune_geom_exp_eff}'
                         if prune_geom_exp_eff is not None
                         else f'{n_perm_prune} perms, alpha={alpha_prune}')
                print(f'  pruning ({_mode}) ...')
            reg_out_list, self.dp_info = prune_node(
                sig_reg_list=self.sig_reg_list,
                children=children_0, exp=exp,
                n_perm=n_perm_prune, alpha=alpha_prune,
                exp_eff=prune_geom_exp_eff)
            self.homo_pval_dict = {}
        elif prune_method == 'tree':
            reg_out_list, self.dp_info = prune_tree(
                sig_reg_list=self.sig_reg_list,
                children=children_0, exp=exp)
            self.homo_pval_dict = {}
        elif prune_method == 'tree_dp':
            if prune_geom_exp_eff is None:
                raise ValueError('tree_dp requires prune_geom_exp_eff')
            reg_out_list, self.dp_info = prune_tree_dp(
                sig_reg_list=self.sig_reg_list,
                children=children_0, exp=exp,
                exp_eff=prune_geom_exp_eff)
            self.homo_pval_dict = {}
        elif prune_method == 'homo':
            reg_out_list, self.homo_pval_dict = prune(
                sig_reg_list=self.sig_reg_list,
                alpha_prune=alpha_prune,
                n_perm=n_perm_prune, exp=exp,
                children=children_0)
            self.dp_info = {}
        else:
            raise ValueError(f'unknown prune_method: {prune_method!r}')

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
            msg = f'  done: {n_disc} discovered, {n_pruned} pruned'
            if prune_method == 'node' and self.dp_info:
                msg += f' (lambda={self.dp_info["lam"]:.4f})'
            print(msg)

    def _run_on_cloud(self, exp, n_perm, n_perm_prune,
                     alpha_fwer, alpha_prune, min_size, verbose,
                     cloud_config, prune_method='node',
                     prune_geom_exp_eff=None, **kwargs):
        """Run full analysis on AWS Batch (permutations + synthesis).

        Submits N+1 permutation jobs, then a synthesis job that polls S3
        for all results before running ``_finalize_analysis`` on the cloud.
        The final pickled AnalysisGLOW is downloaded and its attributes
        are copied onto ``self``.
        """
        from glow.aws import AWSBatchRunner
        import uuid

        print('running analysis on AWS cloud...')

        experiment_id = f'glow_{uuid.uuid4().hex[:8]}'

        ana_kwargs = {
            'get_stat': self.get_stat,
            'n_perm_prune': n_perm_prune,
            'alpha_fwer': alpha_fwer,
            'alpha_prune': alpha_prune,
            'min_size': min_size,
            'prune_method': prune_method,
            'prune_geom_exp_eff': prune_geom_exp_eff,
        }
        ana_kwargs.update(kwargs)

        runner = AWSBatchRunner(cloud_config)

        print('uploading experiment data...')
        runner.upload_experiment(exp, ana_kwargs, experiment_id)

        print('submitting permutation jobs...')
        submission = runner.submit_jobs(
            experiment_id=experiment_id,
            n_perm=n_perm,
            skip_completed=True,
        )

        if submission.get('cancelled'):
            raise RuntimeError('job submission cancelled')

        perm_job_ids = submission['job_ids']

        print('submitting synthesis job...')
        synth_job_id = runner.submit_synthesis_job(experiment_id, n_perm)

        all_job_ids = perm_job_ids + [synth_job_id]

        if verbose:
            runner.monitor_jobs(all_job_ids)
        else:
            print(f'submitted {len(perm_job_ids)} perm jobs + 1 synthesis job')

        print('downloading final analysis...')
        remote_ana = runner.download_final_analysis(experiment_id)

        _COPY_ATTRS = [
            'child_dict', 'stat_0', 'size_0', 'pval', 'llr_adjusted_0',
            'sig_reg_list', 'effect_list', 'alpha_fwer', 'alpha_prune',
            'dp_info', 'homo_pval_dict', 'adj_model', 'adj_beta',
        ]
        for attr in _COPY_ATTRS:
            if hasattr(remote_ana, attr):
                setattr(self, attr, getattr(remote_ana, attr))

        print(f'cloud analysis complete: found {len(self.effect_list)} effects')