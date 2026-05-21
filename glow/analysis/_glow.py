from bisect import bisect_left

import numpy as np
from tqdm import tqdm

import glow.effect
import glow.graph
from glow.experiment.exper import ExperimentScaled
from . import inner_perm, Analysis
from .mancova import decompose, is_intercept_only_nuisance
from .prune import prune_greedy
from .cluster import cluster, ClusterMode


class AnalysisGLOW(Analysis):
    """search a hierarchical segmentation for significant effects.

    Uses a disk-backed streaming pipeline: each outer-perm worker is
    self-contained — it clusters, computes the observed LLR for that
    tree, and runs ``n_perm_inner`` fresh Freedman-Lane draws against
    that same tree to estimate per-region (mu_r, std_r).  The synth
    step then just gathers each worker's pre-computed max-z and runs
    Westfall-Young FWER + pruning.

    Attributes:
        children (np.array): (num_leaf - 1, 2) Ward children for observed data
        stat (np.array): (num_reg,) raw test statistics for observed
        size (np.array): (num_reg,) region sizes for observed
        pval (np.array): (num_reg,) FWER-controlled p-values
        effect_list (list): discovered Effect objects
    """

    def __init__(self, exp, n_perm_fwer,
                 n_perm_inner=200,
                 alpha_fwer=.05, min_vox=4, verbose=False,
                 cloud_config=None,
                 cluster_mode=ClusterMode.FOCUS):
        """
        Args:
            exp: Experiment to analyze
            n_perm_fwer: number of outer FL permutations for FWER control
            n_perm_inner: number of inner FL permutations run per
                outer-perm worker.  Every active region (size >=
                min_vox) gets exactly this many draws — no CI-based
                dropout, no adaptive resampling.
            alpha_fwer: family-wise error rate
            min_vox: minimum region size (in voxels) admitted to the
                FWER comparison set.  Regions smaller than this are
                excluded from the max-z null and assigned NaN p-values.
                Defaults to 4 — the merged-graph profiling showed that
                regions with size < 5 are >99% tree-private and
                contribute disproportionately to the max-z tail without
                ever being plausible scientific findings.
            verbose: print progress
            cloud_config: CloudConfig for AWS execution (None = local)
            cluster_mode (ClusterMode): Ward projection.  Default
                ``ClusterMode.FOCUS`` projects onto the contrast
                subspace; ``ClusterMode.GLM_ERROR`` keeps bias +
                contrast; ``ClusterMode.NAIVE`` clusters raw y.
        """
        super().__init__(exp)
        self.verbose = verbose
        self.cluster_mode = cluster_mode
        self.min_vox = min_vox
        self.n_perm_fwer = n_perm_fwer
        self.n_perm_inner = n_perm_inner

        b, num_img, num_vox = exp.y.shape

        # decompose() depends only on (x, contrast), which permute()
        # leaves untouched — hoist outside the per-perm loop and reuse
        # for every outer + inner FL draw.
        q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)

        # Permutation index layout:
        #   0                = observed data (outer perm 0)
        #   1..n_perm_fwer   = outer FL nulls (FWER comparison set)
        # Inner perms run inside each outer-perm worker (against that
        # worker's tree) and stay in memory.
        all_perm_indices = list(range(n_perm_fwer + 1))

        if verbose:
            print(f'  [1/2] outer perms: clustering + inner perms for '
                  f'{len(all_perm_indices)} permutations ({num_vox} voxels, '
                  f'{n_perm_fwer} FWER, {n_perm_inner} inner) ...')

        results = {}
        for perm_idx in tqdm(all_perm_indices, desc='outer perms',
                             disable=not verbose):
            results[perm_idx] = self._process_permutation(
                exp, perm_idx, n_perm_inner, min_vox, q0, q1)

        if verbose:
            print(f'  [2/2] per_region_z finalization ...')

        self._finalize_per_region_z(
            exp, results, n_perm_fwer, alpha_fwer, min_vox)

    def _finalize_per_region_z(self, exp, results, n_perm_fwer,
                               alpha_fwer, min_vox):
        """Per-region permutation z-scoring with Westfall-Young FWER.

        Reads each outer-perm worker's result (which already contains
        the worker-local mu/sigma/z + max_z), assembles the max-z null,
        and runs the FWER + pruning pipeline on the observed tree.
        The ``min_vox`` cutoff defends FWER power against the long
        tail of small (mostly tree-private) regions whose noise
        dominates the max-z null.

        Args:
            results: dict mapping perm_idx → result dict (see
                ``_process_permutation``).  Must contain keys 0..n_perm_fwer.
        """
        verbose = getattr(self, 'verbose', False)

        max_z_outer = [float(results[k]['max_z'])
                       for k in range(1, n_perm_fwer + 1)]
        r0 = results[0]

        if verbose:
            print(f'  per_region_z: assembled max-z null from '
                  f'{n_perm_fwer + 1} outer perms (min_vox={min_vox})')

        max_z_list = sorted(max_z_outer + [float(r0['max_z'])])
        self._finalize_analysis(
            exp, n_perm_fwer,
            np.asarray(r0['stat'], dtype=float),
            np.asarray(r0['size'], dtype=float),
            np.asarray(r0['children']),
            np.asarray(r0['z'], dtype=float),
            max_z_list, alpha_fwer, min_vox)
        self._mu_per_region = np.asarray(r0['mu'], dtype=float)
        self._sigma_per_region = np.asarray(r0['sigma'], dtype=float)

    @classmethod
    def from_precomputed(cls, *, exp, verbose=False, cluster_mode=ClusterMode.FOCUS):
        """Construct an empty shell without running ``__init__``.

        Caller is responsible for invoking ``_finalize_per_region_z()``
        (or running outer perms first) to populate the analysis.
        """
        obj = cls.__new__(cls)
        obj.exp = (exp if isinstance(exp, ExperimentScaled)
                   else ExperimentScaled.from_exp(exp))
        obj.verbose = verbose
        obj.cluster_mode = cluster_mode
        return obj

    def _process_permutation(self, exp, perm_idx, n_perm_inner, min_vox,
                             q0=None, q1=None):
        """Run one outer perm: cluster, observed LLR, inner perms, max_z.

        Each outer perm worker is fully self-contained — the inner FL
        permutations run against this perm's tree (not a merged graph),
        which makes the worker embarrassingly parallel and gives a
        bounded memory footprint.  The merged-graph approach was
        retired after profiling showed regions with size >= 5 are
        >99% tree-private (graph_merge bought nothing for the regions
        that ``min_vox`` admits to the FWER comparison set).

        Args:
            exp: source experiment (NOT yet permuted; ``perm_idx==0``
                is the observed data).
            perm_idx: outer permutation index.
            n_perm_inner: number of inner FL draws to run against
                this outer perm's tree.  Every active region gets
                exactly this many samples — no CI-based dropout, no
                resume.
            min_vox: minimum region size for the max-z comparison set.
            q0, q1: pre-decomposed contrast subspaces.  May be None
                when called via ``rerun_permutation``; in that case
                they are recomputed from ``exp``.

        Returns:
            dict with keys
                ``perm_idx``, ``children``, ``stat`` (raw LLR),
                ``size``, ``mu``, ``sigma``,
                ``n_inner_used``, ``z``, ``max_z``.
        """
        # In the joblib parallel path, ``exp`` arrives via pickle which
        # slims y/x when a recipe is registered.  Rehydrate before any
        # further use.
        if exp.y is None:
            exp.rehydrate()

        if q0 is None or q1 is None:
            q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)

        _exp = exp.permute(perm_idx)
        children = cluster(exp=_exp, mode=self.cluster_mode)

        # Per-node tree depth depends only on `children`; precompute
        # once and reuse across the inner-perm loop.  Without this,
        # the Python loop inside compute_llr_batched is ~34% of its
        # runtime on a 5k vox tree.
        num_vox = _exp.y.shape[2]
        layer = glow.graph.compute_tree_layers(children, num_vox)

        # observed (for this outer perm) LLR per region.  Computed
        # for ALL regions, including size < min_vox, so the viewer's
        # H1 scatter has full coverage (raw LLR is cheap to compute
        # at all sizes).
        llr_outer, size = glow.graph.compute_llr_batched(
            _exp, children=children, q0=q0, q1=q1, layer=layer)

        # Dispatch inner FL perms across the (cpu vs gpu) x (fast vs
        # slow) grid.  Fast = intercept-only Phase-1 hoist; slow =
        # general-Q0 full recompute per draw.  See
        # ``glow.analysis.inner_perm`` for the four backends; they
        # share one keyword signature.
        if n_perm_inner > 0:
            use_fast = (getattr(self, 'use_fast_path', True)
                        and is_intercept_only_nuisance(exp.x, exp.contrast))
            use_gpu = getattr(self, 'use_gpu', False)
            if use_gpu:
                run = inner_perm.gpu_fast if use_fast else inner_perm.gpu_slow
            else:
                run = inner_perm.cpu_fast if use_fast else inner_perm.cpu_slow

            # Seed scheme: each outer perm reserves a 100_000-wide
            # block, far above any realistic n_perm_inner, so inner
            # seeds never collide across outer perms.
            mu, sigma = run(
                exp=_exp, base_seed=(perm_idx + 1) * 100_000,
                n_perm=n_perm_inner,
                q0=q0, q1=q1, children=children, layer=layer,
                min_vox=min_vox)
            n_inner_used = n_perm_inner
        else:
            # rerun_permutation path: caller only wants children/stat/size.
            mu = np.full_like(llr_outer, fill_value=np.nan)
            sigma = np.full_like(llr_outer, fill_value=np.nan)
            n_inner_used = 0
        del _exp

        # zero-std guard (constant inner draws → divide-by-zero z).
        sigma_safe = np.where(sigma < 1e-12, 1.0, sigma)
        z = (llr_outer - mu) / sigma_safe
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=np.nan)

        # FWER comparison set: only regions with size >= min_vox feed
        # the max-z null.  Inactive regions are excluded entirely.
        active = size >= min_vox
        if active.any() and np.isfinite(z[active]).any():
            max_z = float(np.nanmax(z[active]))
        else:
            max_z = float('-inf')

        result = {
            'perm_idx': perm_idx,
            'children': children,
            'stat': llr_outer,
            'size': size,
            'mu': mu,
            'sigma': sigma,
            'z': z,
            'max_z': max_z,
            'n_inner_used': n_inner_used,
        }
        return result

    @classmethod
    def rerun_permutation(cls, exp, perm_idx,
                          cluster_mode=ClusterMode.FOCUS, n_perm_inner=0, min_vox=4):
        """Re-run a single outer permutation for inspection.

        Defaults to ``n_perm_inner=0`` (skips the inner FL loop), which
        reproduces the cheap "just give me children/stat/size" use
        case of the old method.

        Returns:
            dict — see ``_process_permutation``.  With
            ``n_perm_inner=0`` the ``mu``/``sigma`` arrays are NaN and
            ``z`` reduces to the (sanitised) raw LLR.
        """
        ana = cls.from_precomputed(exp=exp)
        ana.cluster_mode = cluster_mode
        return ana._process_permutation(
            exp, perm_idx, n_perm_inner, min_vox)

    def _finalize_analysis(self, exp, n_perm_fwer,
                           stat_0, size_0, children_0,
                           llr_z_0, stat_max_sorted,
                           alpha_fwer, min_size,
                           prune_stat=None,
                           ):
        """Compute p-values from the max-z null and prune.

        Args:
            stat_0: raw LLR per region for the observed tree.
            size_0: region sizes for the observed tree.
            children_0: observed Ward children.
            llr_z_0: per-region z-scored LLR for the observed tree
                (already adjusted by per-region (mu, std)).
            stat_max_sorted: sorted max-z null distribution from the
                outer permutations (length n_perm_fwer + 1).
            min_size: minimum region size for the FWER comparison set
                (synonym for ``min_vox`` at the call site).
            prune_stat: optional override for the array used to rank
                pruning candidates.  Defaults to ``stat_0`` (raw LLR).
                Pass ``llr_z_0`` to rank by per-region z instead.
                See note below.
        """
        verbose = getattr(self, 'verbose', False)
        num_reg = stat_0.shape[0]

        llr_z_0 = np.nan_to_num(np.asarray(llr_z_0, dtype=float), nan=0.0,
                                posinf=0.0, neginf=np.nan)

        self.alpha_fwer = alpha_fwer
        self.size = size_0
        self.stat = stat_0
        self.llr_z_0 = llr_z_0
        self.children = children_0
        self.stat_max_sorted = np.asarray(stat_max_sorted)

        reg_active = size_0 >= min_size
        if not reg_active.any():
            pval = np.full(num_reg, fill_value=np.nan)
        else:
            pval = np.full(num_reg, fill_value=-1.0)
            for reg_idx, z in enumerate(llr_z_0):
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

        # Rank pruning candidates by raw LLR.  z-score is right for
        # FWER thresholding (puts different-size regions on a common
        # scale), but it fragments under pruning: for a true effect of
        # size n with per-voxel strength alpha, z scales as ~sqrt(n),
        # so a small slice of the effect can outscore the whole region
        # on z.  Greedy z-pruning then locks out the parent and emits
        # fragments with very low Dice.  Raw LLR scales linearly with
        # n, picks the largest coherent region, and naturally caps
        # over-inclusion via the z-FWER filter (only z-significant
        # regions are pruning candidates; an over-large parent
        # typically fails z-FWER because its added voxels dilute the
        # per-region signal).  RunPruneCompare exposes both via the
        # greedy_llr / greedy_z labels for head-to-head benchmarking.
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
            eff = glow.effect.EffectEstimate.from_exp_mask(
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
                      alpha_fwer, min_vox, verbose,
                      cloud_config,
                      perms_per_job=None,
                      cluster_mode=ClusterMode.FOCUS,
                      **kwargs):
        """Run full analysis on AWS Batch (outer perms + synthesis).

        Submits ``n_perm_fwer + 1`` outer-perm jobs (each running its
        own inner FL loop) plus a synthesis job that gathers per-perm
        pickles and runs the FWER + pruning pipeline.  The final
        pickled AnalysisGLOW is downloaded and its attributes are
        copied onto ``self``.
        """
        from glow.aws import AWSBatchRunner
        import uuid

        print('running analysis on AWS cloud...')

        experiment_id = f'glow_{uuid.uuid4().hex[:8]}'

        ana_kwargs = {
            'n_perm_inner': n_perm_inner,
            'alpha_fwer': alpha_fwer,
            'min_vox': min_vox,
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
            'children', 'stat', 'size', 'pval', 'llr_z_0',
            'sig_reg_list', 'effect_list', 'alpha_fwer', 'adj_crit',
            'prune_info', 'stat_max_sorted',
            '_mu_per_region', '_sigma_per_region',
        ]
        for attr in _COPY_ATTRS:
            if hasattr(remote_ana, attr):
                setattr(self, attr, getattr(remote_ana, attr))

        print(
            f'cloud analysis complete: found {len(self.effect_list)} effects')

    def pickle_status(self):
        """Return :class:`PickleStatus` accounting for analysis arrays.

        Adds the bytesize of stored arrays
        (``pval``, ``stat``, ``size``, ``llr_z_0``, ``children``,
        ``stat_max_sorted``) to ``estimated_pickle_mb`` so callers can
        budget the on-disk size of the full :class:`AnalysisGLOW`.
        """
        from glow.experiment.regen import compute_pickle_status

        extra = 0
        for attr in ('pval', 'stat', 'size', 'llr_z_0', 'children',
                     'stat_max_sorted', '_mu_per_region',
                     '_sigma_per_region'):
            arr = getattr(self, attr, None)
            if arr is not None and hasattr(arr, 'nbytes'):
                extra += int(arr.nbytes)
        return compute_pickle_status(self.exp, extra_bytes=extra)
