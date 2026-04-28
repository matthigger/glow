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


GAMFitResult = namedtuple(
    'GAMFitResult',
    ['mu_gam', 'mu_fn', 'sigma_gam', 'sigma_fn', 'r2', 'size_adjusted'])
"""Return type of fit_size_gam.

Two-stage fit modelling both the conditional mean and the conditional
standard deviation of the null statistic as smooth functions of region
size (analogous to the GAMLSS framework, Rigby & Stasinopoulos 2005).
Studentizing by ``sigma_fn`` removes size-dependent heteroscedasticity
that mean-only adjustment leaves intact.

Attributes:
    mu_gam: fitted LinearGAM for E[stat | log10 size], or None if
        the low-data fallback was used.
    mu_fn: callable(size_array) -> predicted E[stat | H0].  When
        ``size_adjusted`` is False, returns zeros (identity adjustment).
    sigma_gam: fitted LinearGAM for log Var[stat | log10 size], or None
        when ``score_method='mean_adj'`` or under the low-data fallback.
    sigma_fn: callable(size_array) -> predicted SD[stat | H0].  Returns
        ones (identity scale) when ``sigma_gam`` is None.
    r2: R² of the mean GAM, or None if fallback used.
    size_adjusted (bool): True if the GAM(s) were fit, False if the
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
                 cluster_mode="ward's (q1)",
                 score_method='mean_adj',
                 keep_fit_data=False,
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
            score_method: Adjustment used for FWER threshold.
                ``'mean_adj'`` (default) subtracts the size-conditional
                null mean only -- this is the historical behaviour and
                stays the default until the studentized variant is
                evaluated more thoroughly.  ``'z_score'`` additionally
                divides by the fitted size-conditional null SD
                (studentization, GAMLSS-style).  Pruning rank is on
                raw LLR in either case.
            keep_fit_data: When True, store a log-uniformly stratified
                subsample of the (size, stat) pairs from the held-out
                fit permutations on ``self._gam_fit_data`` (~2k pairs,
                float32, ~16 KB).  Useful for the viewer's H0 mode
                which renders this cloud as the actual data the
                size-adjustment GAM was fit to.  Default False so
                production analyses don't carry the diagnostic data.
            cloud_config: CloudConfig for AWS execution (if None, runs
                locally)
            perm_dir: path for permutation result files.  If provided,
                results are kept on disk for post-hoc inspection; if
                None a temp directory is created and cleaned up.
                Existing results in the directory are reused (resume).
            cluster_mode: projection used by Ward clustering.  Must be
                one of the keys in ``glow.analysis.cluster._MODES``.
                Default ``"ward's (q1)"`` ("Focus") projects onto the
                contrast-of-interest subspace; ``"ward's (q0, q1)"``
                ("GLM Error") keeps bias + contrast; ``"ward's (all)"``
                ("Naive") clusters raw ``y``.
        """
        super().__init__(exp, **kwargs)
        self.verbose = verbose
        self.cluster_mode = cluster_mode
        self.score_method = score_method
        self.keep_fit_data = keep_fit_data
        self._gam_fit_data = None

        if cloud_config is not None:
            self._run_on_cloud(exp, n_perm_fwer,
                              alpha_fwer, min_size, verbose,
                              cloud_config,
                              n_perm_fwer_size_adjust=n_perm_fwer_size_adjust,
                              perms_per_job=kwargs.pop('perms_per_job', None),
                              cluster_mode=cluster_mode,
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
            print(f'  [2/3] fitting size-adjustment GAM(s) ...')
        all_sizes = np.concatenate(fit_sizes)
        all_stats = np.concatenate(fit_stats)
        fit = self.fit_size_gam(all_sizes, all_stats,
                                score_method=score_method)
        self.adj_gam = fit.mu_gam            # legacy alias
        self.mu_gam = fit.mu_gam
        self.sigma_gam = fit.sigma_gam
        self.size_adjusted = fit.size_adjusted
        mu_fn = fit.mu_fn
        sigma_fn = fit.sigma_fn
        if verbose and fit.r2 is not None:
            print(f'         R²(mu)={fit.r2:.4f}'
                  + (' (z_score: sigma_gam fitted)'
                     if fit.sigma_gam is not None else ''))
        elif verbose:
            print(f'         size adjustment disabled (low-data fallback)')

        # diagnostic subsample of the fit cloud (used by viewer H0 mode)
        if keep_fit_data:
            self._gam_fit_data = self._subsample_fit_data(
                all_sizes, all_stats)
            if verbose and self._gam_fit_data is not None:
                print(f'         retained {len(self._gam_fit_data["size"])}'
                      f' fit-cloud pairs')

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
            sz_p = np.asarray(r['size'], dtype=float)
            adj = ((np.asarray(r['stat'], dtype=float) - mu_fn(sz_p))
                   / sigma_fn(sz_p))
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
            alpha_fwer, min_size,
            sigma_fn=sigma_fn)

        if _cleanup_dir:
            shutil.rmtree(perm_dir, ignore_errors=True)

    @classmethod
    def from_precomputed(cls, *, exp, get_stat, adj_gam=None,
                         sigma_gam=None, verbose=False,
                         cluster_mode="ward's (q1)",
                         score_method=None):
        """Construct a shell for finalization without running full __init__.

        The caller should then invoke _finalize_analysis() to compute p-values.

        Args:
            adj_gam: legacy alias for the mean GAM (mu_gam).
            sigma_gam: optional log-variance GAM.  When None, the
                resulting analysis behaves as ``score_method='mean_adj'``;
                when provided, ``score_method='z_score'``.
            score_method: explicit override.  When None, inferred from
                whether ``sigma_gam`` is provided.
        """
        obj = cls.__new__(cls)
        obj.exp = exp if isinstance(exp, ExperimentScaled) else ExperimentScaled.from_exp(exp)
        obj.get_stat = get_stat
        obj.adj_gam = adj_gam        # legacy alias
        obj.mu_gam = adj_gam
        obj.sigma_gam = sigma_gam
        obj.verbose = verbose
        obj.cluster_mode = cluster_mode
        obj.score_method = (score_method
                            if score_method is not None
                            else ('z_score' if sigma_gam is not None
                                  else 'mean_adj'))
        return obj

    def _process_permutation(self, exp, perm_idx):
        """Run one permutation: cluster, compute stats and sizes."""
        _exp = exp.permute(perm_idx)
        children = cluster(exp=_exp, mode=self.cluster_mode)

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

    _N_SPLINES = 10

    _MIN_GAM_FIT_POINTS = 50

    def __setstate__(self, state):
        """Pickle restore with backward compatibility.

        Pre-z_score pickles only stored ``adj_gam`` (no ``sigma_gam``
        and no ``score_method``).  Restore them as legacy mean-only
        analyses so ``_finalize_analysis``-derived attributes remain
        consistent with the original computation.
        """
        self.__dict__.update(state)
        if 'mu_gam' not in self.__dict__:
            self.mu_gam = self.__dict__.get('adj_gam')
        if 'sigma_gam' not in self.__dict__:
            self.sigma_gam = None
        if 'score_method' not in self.__dict__:
            self.score_method = ('mean_adj' if self.sigma_gam is None
                                 else 'z_score')
        if '_gam_fit_data' not in self.__dict__:
            self._gam_fit_data = None
        if 'keep_fit_data' not in self.__dict__:
            self.keep_fit_data = self._gam_fit_data is not None

    @classmethod
    def fit_size_gam(cls, size, stat, n_splines=None,
                     score_method='mean_adj'):
        """Fit size-adjustment GAM(s) for the null statistic distribution.

        Stage 1 (always): mu_gam fits stat ~ f_mu(log10 size).
        Stage 2 (only when score_method='z_score'): sigma_gam fits
            log(e^2 + eps) ~ f_sigma(log10 size), where
            e = stat - mu_gam.predict(log10 size) and eps is a tiny
            positive floor to keep the log finite.  sigma_fn(s) is then
            recovered as exp(0.5 * sigma_gam.predict(log10 s)), positive
            by construction.

        Args:
            size: (N,) region sizes (voxels)
            stat: (N,) stat values
            n_splines: number of splines per stage (default: cls._N_SPLINES)
            score_method: 'mean_adj' (default) fits only mu_gam; 'z_score'
                additionally fits sigma_gam so callers can studentize.

        Returns:
            GAMFitResult: see namedtuple docstring.  When
            ``valid.sum() < _MIN_GAM_FIT_POINTS``, identity fallbacks are
            used and a ``RuntimeWarning`` is emitted; ``size_adjusted``
            is False so callers can surface this to users.
        """
        if n_splines is None:
            n_splines = cls._N_SPLINES
        if score_method not in ('mean_adj', 'z_score'):
            raise ValueError(
                f'score_method must be mean_adj or z_score; '
                f'got {score_method!r}')

        sz = np.asarray(size, dtype=float)
        y = np.asarray(stat, dtype=float)
        valid = np.isfinite(y) & np.isfinite(sz) & (sz > 0)

        n_valid = int(valid.sum())
        if n_valid < cls._MIN_GAM_FIT_POINTS:
            warnings.warn(
                f'fit_size_gam: only {n_valid} valid (size, stat) pairs '
                f'(threshold is {cls._MIN_GAM_FIT_POINTS}); falling back '
                f'to identity mu_fn / sigma_fn.  No size adjustment will '
                f'be applied -- downstream FWER max-stats will reflect '
                f'raw stats.  Consider increasing n_perm_fwer_size_adjust '
                f'or verifying that the stat function is well-behaved on '
                f'small regions.',
                RuntimeWarning,
                stacklevel=2,
            )
            return GAMFitResult(
                mu_gam=None,
                mu_fn=lambda sz: np.zeros_like(np.asarray(sz, dtype=float)),
                sigma_gam=None,
                sigma_fn=lambda sz: np.ones_like(np.asarray(sz, dtype=float)),
                r2=None,
                size_adjusted=False,
            )

        log_size = np.log10(sz[valid])
        y_valid = y[valid]

        # --- Stage 1: mean ---
        mu_gam = LinearGAM(s(0, n_splines=n_splines))
        mu_gam.gridsearch(log_size, y_valid, progress=False)

        mu_pred = mu_gam.predict(log_size)
        ss_res = np.sum((y_valid - mu_pred) ** 2)
        ss_tot = np.sum((y_valid - y_valid.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        mu_fn = cls.mu_fn_from_gam(mu_gam)

        # --- Stage 2: log variance (z_score only) ---
        sigma_gam = None
        if score_method == 'z_score':
            resid_sq = (y_valid - mu_pred) ** 2
            # eps: tiny floor preventing log(0) and bounding minimum
            # implied variance.  See note 2026_studentize_size_adjust.
            max_sq = float(np.maximum(resid_sq.max(), np.finfo(float).tiny))
            eps = np.finfo(float).eps * max_sq
            log_sq = np.log(resid_sq + eps)

            sigma_gam = LinearGAM(s(0, n_splines=n_splines))
            sigma_gam.gridsearch(log_size, log_sq, progress=False)

        sigma_fn = cls.sigma_fn_from_gam(sigma_gam)

        return GAMFitResult(mu_gam=mu_gam, mu_fn=mu_fn,
                            sigma_gam=sigma_gam, sigma_fn=sigma_fn,
                            r2=r2, size_adjusted=True)

    _GAM_FIT_DATA_BUDGET = 2000
    """Maximum number of (size, stat) pairs retained for the diagnostic
    H0 cloud render in the viewer.  ~16 KB at float32."""

    @classmethod
    def _subsample_fit_data(cls, sizes, stats, budget=None):
        """Log-uniform stratified subsample of the GAM fit cloud.

        The raw fit data is heavily skewed toward small regions (a Ward
        tree on N voxels has 2N-1 nodes, of which N are leaves).  For a
        diagnostic plot we want every size band visible, so we bucket
        log10-uniformly and sample within each bucket without
        replacement up to a per-bucket cap.

        Args:
            sizes: (N,) raw region sizes from fit permutations
            stats: (N,) raw stat values from fit permutations
            budget: target number of pairs to keep
                (default: cls._GAM_FIT_DATA_BUDGET).

        Returns:
            dict with 'size' and 'stat' float32 arrays of length
            <= budget, or None if input is empty.
        """
        if budget is None:
            budget = cls._GAM_FIT_DATA_BUDGET
        sz = np.asarray(sizes, dtype=float)
        st = np.asarray(stats, dtype=float)
        valid = np.isfinite(sz) & np.isfinite(st) & (sz > 0)
        sz, st = sz[valid], st[valid]
        if sz.size == 0:
            return None

        n_bins = 30
        log_sz = np.log10(sz)
        edges = np.linspace(log_sz.min(), log_sz.max(), n_bins + 1)
        per_bin = max(1, budget // n_bins)
        rng = np.random.default_rng(0)

        keep_idx = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = np.flatnonzero((log_sz >= lo) & (log_sz <= hi))
            if in_bin.size == 0:
                continue
            take = min(per_bin, in_bin.size)
            keep_idx.append(rng.choice(in_bin, size=take, replace=False))
        if not keep_idx:
            return None
        keep = np.concatenate(keep_idx)
        if keep.size > budget:
            keep = rng.choice(keep, size=budget, replace=False)

        return {
            'size': sz[keep].astype(np.float32),
            'stat': st[keep].astype(np.float32),
        }

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

    @staticmethod
    def sigma_fn_from_gam(gam):
        """Build a sigma_fn callable from a fitted log-variance GAM.

        The GAM stores E[log(e^2 + eps) | log10 size]; we recover
        sigma(s) = exp(0.5 * gam.predict(log10 s)).  When ``gam`` is
        None, returns the identity scale (ones) so callers can use a
        common (stat - mu) / sigma pipeline regardless of whether
        studentization is configured.
        """
        if gam is None:
            return lambda sz: np.ones_like(np.asarray(sz, dtype=float))

        def sigma_fn(sz, _gam=gam):
            sz = np.asarray(sz, dtype=float)
            log_sq = _gam.predict(np.log10(np.maximum(sz, 1)))
            return np.exp(0.5 * log_sq)
        return sigma_fn

    def _finalize_analysis(self, exp, n_perm_fwer,
                          stat_0, size_0, children_0,
                          mu_fn, stat_max_sorted,
                          alpha_fwer, min_size,
                          sigma_fn=None,
                          prune_stat=None,
                          ):
        """Finalize: compute p-values from max-stat distribution, prune.

        Args:
            sigma_fn: callable(size) -> SD estimate.  When None, the
                identity (ones) is used, reproducing the legacy
                mean-only adjustment.  When non-None, the observed
                statistic is studentized: z = (stat - mu) / sigma.
            prune_stat (np.array): optional override for the stat array
                used by greedy pruning.  When None (default), the
                studentized statistic ``llr_adjusted_0`` is used so the
                pruning rank is consistent with the FWER threshold.
        """
        verbose = getattr(self, 'verbose', False)
        num_reg = stat_0.shape[0]

        if sigma_fn is None:
            sigma_fn = lambda sz: np.ones_like(np.asarray(sz, dtype=float))

        size_f = size_0.astype(float)
        llr_adjusted_0 = (stat_0 - mu_fn(size_f)) / sigma_fn(size_f)
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
            print('  pruning (greedy on raw LLR) ...')

        # Rank pruning candidates by raw LLR.  Significance is gated by
        # the studentized statistic against the FWER threshold above;
        # within the surviving set, raw-LLR ranking keeps the original
        # ``largest spatially-coherent region wins'' behaviour.
        # Studentized ranking (z-score) was tried and rejected:
        # because sigma_fn(size) ~ sqrt(size) for CLT-ish nulls, z grows
        # only as sqrt(size) for a uniform effect, so on heterogeneous
        # within-region signal (e.g.\ mandrill colour gradients) z can
        # peak on a subregion of strongest pixels rather than the
        # parent node spanning the planted effect.  Raw-LLR ranking
        # naturally prefers the parent.
        # Caller may override via ``prune_stat``.
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
        model = load_runtime_model('GLOW', platform='aws')
        if model is None:
            path = RUNTIME_MODEL_PATHS['aws'].get('GLOW')
            raise FileNotFoundError(
                f'GLOW AWS runtime model not found at {path}.  '
                f'Regenerate with: '
                f'python -m glow.benchmark.runtime --profile experiment --cloud')
        return predict_runtime_sec(model, num_vox, b, num_img, n_perm=1)

    def _run_on_cloud(self, exp, n_perm_fwer,
                     alpha_fwer, min_size, verbose,
                     cloud_config, n_perm_fwer_size_adjust=25,
                     perms_per_job=None,
                     cluster_mode="ward's (q1)",
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
            'cluster_mode': cluster_mode,
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
