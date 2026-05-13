import pickle
import shutil
import tempfile
import warnings
from bisect import bisect_left
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

import glow.effect
import glow.graph
from ._base import Analysis, _sanitize_adjusted_stat
from glow.experiment.exper import ExperimentScaled
from .mancova import decompose, get_llr, is_intercept_only_nuisance
from .prune import prune_greedy
from .cluster import cluster


def _run_inner_race(draw_one, n_max, llr_outer, size, min_vox,
                    race_init=50, race_batch=25, race_k_sigma=3.0,
                    z_threshold=None, on_warmup_done=None):
    """Adaptive inner-perm loop: drop decided regions, stop when none remain.

    Computational, not statistical: regions whose interim z is far
    enough from the decision boundary can't plausibly cross it, so
    they stop competing.  After each batch we update per-region z and
    an asymptotic Wald SE (``sqrt((1 + z^2/2)/n)``) and drop every
    active region whose ``±k*SE`` band already commits to a side of
    the boundary.

    Two modes (selected by ``z_threshold``):

    - **Leader race** (``z_threshold=None``): drop region r if
      ``z[r] + k*SE[r] < z[leader] - k*SE[leader]``.  Loop ends when
      only the leader is left in the active set.
    - **Threshold race** (``z_threshold=value``): drop region r if
      ``z[r] + k*SE[r] < value`` (clearly below) or
      ``z[r] - k*SE[r] > value`` (clearly above).  Loop ends when no
      region is still ambiguous.

    Per-region (mu, M2, n_per_reg) is updated online via Welford's
    algorithm — O(num_reg) per inner perm regardless of n_run, no
    growing buffer.  In Phase A, ``draw_one`` still returns LLR for
    every region per inner perm, so ``n_per_reg`` equals
    ``n_inner_used`` for every initially-active region; the field
    becomes meaningful in a future Phase-B kernel that learns to skip
    inactive regions per inner perm.

    Returns:
        dict with
            ``mu``, ``sigma`` (num_reg,) float — final per-region stats
            ``n_per_reg`` (num_reg,) int — n inner draws each region got
            ``n_inner_used`` int — total inner perms run
            ``leader`` int (leader mode only) — region with current max z
            ``sig_regions`` (num_sig,) int (threshold mode only) —
                indices of regions confidently above ``z_threshold``

    Args (callback):
        on_warmup_done (callable | None): one-shot hook called after
            warmup + the first trim, before the first batch.  Signature
            ``f(active) -> draw_one_post | None``.  If it returns a
            non-None callable, that callable replaces ``draw_one`` for
            the rest of the race.  Used by AnalysisGLOW to build
            M_{ij} kernels for the post-warmup survivor set and switch
            to a cheap survivor-only inner-perm draw.
    """
    num_reg = llr_outer.shape[0]

    # Welford running stats (per region)
    mean = np.zeros(num_reg, dtype=float)
    M2 = np.zeros(num_reg, dtype=float)
    n_per_reg = np.zeros(num_reg, dtype=np.int64)

    def update(x):
        # Online update for one inner-perm draw across all regions.
        finite = np.isfinite(x)
        n_per_reg[finite] += 1
        # Welford recurrence: mean_new = mean + (x - mean)/n_new
        delta = np.empty_like(mean)
        delta[finite] = x[finite] - mean[finite]
        mean[finite] += delta[finite] / n_per_reg[finite]
        delta2 = np.empty_like(mean)
        delta2[finite] = x[finite] - mean[finite]
        M2[finite] += delta[finite] * delta2[finite]

    def current_z_se(n_run):
        # Sample std via M2/(n-1); regions with n<2 → sigma=NaN → z=NaN.
        with np.errstate(invalid='ignore', divide='ignore'):
            var = np.where(n_per_reg >= 2, M2 / (n_per_reg - 1), np.nan)
        sigma = np.sqrt(var)
        sigma_safe = np.where(sigma < 1e-12, 1.0, sigma)
        z = (llr_outer - mean) / sigma_safe
        # Wald SE depends on n_per_reg (per-region count) and z itself.
        se = np.sqrt((1 + z ** 2 / 2) / np.maximum(n_per_reg, 1))
        return z, sigma, se

    n_run = 0
    init_n = min(race_init, n_max)
    for i in range(init_n):
        update(draw_one(i))
    n_run = init_n

    active = (size >= min_vox).copy()
    leader_idx = -1
    warmup_hook_pending = on_warmup_done is not None

    while n_run < n_max:
        z, _, se = current_z_se(n_run)
        eligible = active & np.isfinite(z)

        if z_threshold is None:
            if eligible.sum() <= 1:
                break
            z_for_max = np.where(eligible, z, -np.inf)
            leader_idx = int(np.argmax(z_for_max))
            leader_lower = z[leader_idx] - race_k_sigma * se[leader_idx]
            upper = z + race_k_sigma * se
            cant_catch = eligible & (upper < leader_lower)
            cant_catch[leader_idx] = False
            if cant_catch.any():
                active &= ~cant_catch
                if (active & np.isfinite(z)).sum() <= 1:
                    break
        else:
            if eligible.sum() == 0:
                break
            upper = z + race_k_sigma * se
            lower = z - race_k_sigma * se
            decided = eligible & ((upper < z_threshold) |
                                  (lower > z_threshold))
            if decided.any():
                active &= ~decided
                if (active & np.isfinite(z)).sum() == 0:
                    break

        # First-time survivor freeze: hand the survivor mask to the
        # caller so it can swap in a kernel-based draw_one.  Fires
        # after the first post-warmup trim; from here on the active
        # set only ever shrinks.
        if warmup_hook_pending:
            replacement = on_warmup_done(active)
            if replacement is not None:
                draw_one = replacement
            warmup_hook_pending = False

        n_batch = min(race_batch, n_max - n_run)
        for i in range(n_run, n_run + n_batch):
            update(draw_one(i))
        n_run += n_batch

    # Finalize stats
    with np.errstate(invalid='ignore', divide='ignore'):
        var_final = np.where(n_per_reg >= 2, M2 / (n_per_reg - 1), np.nan)
    sigma_final = np.sqrt(var_final)

    out = {
        'mu': mean,
        'sigma': sigma_final,
        'n_per_reg': n_per_reg,
        'n_inner_used': n_run,
    }
    if z_threshold is None:
        # Recompute leader at exit on final stats (might not match the
        # last race-loop leader if no further drop checks happened).
        sigma_safe = np.where(sigma_final < 1e-12, 1.0, sigma_final)
        z_final = (llr_outer - mean) / sigma_safe
        eligible_final = (size >= min_vox) & np.isfinite(z_final)
        if eligible_final.any():
            out['leader'] = int(np.argmax(np.where(eligible_final, z_final,
                                                    -np.inf)))
        else:
            out['leader'] = -1
    else:
        sigma_safe = np.where(sigma_final < 1e-12, 1.0, sigma_final)
        z_final = (llr_outer - mean) / sigma_safe
        se_final = np.sqrt((1 + z_final ** 2 / 2)
                            / np.maximum(n_per_reg, 1))
        sig_mask = ((size >= min_vox) & np.isfinite(z_final)
                    & ((z_final - race_k_sigma * se_final) > z_threshold))
        out['sig_regions'] = np.where(sig_mask)[0]
    return out


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
                 n_jobs_perm=1, cloud_config=None, perm_dir=None,
                 cluster_mode="q1",
                 use_fast_path=True,
                 **kwargs):
        """
        Args:
            exp: Experiment to analyze
            n_perm_fwer: number of outer FL permutations for FWER control
            n_perm_inner: number of inner FL permutations used per
                outer-perm worker to estimate per-region (mu, std) of
                the H0 LLR distribution against that worker's tree.
            alpha_fwer: family-wise error rate
            min_vox: minimum region size (in voxels) admitted to the
                FWER comparison set.  Regions smaller than this are
                excluded from the max-z null and assigned NaN p-values.
                Defaults to 4 — the merged-graph profiling showed that
                regions with size < 5 are >99% tree-private and
                contribute disproportionately to the max-z tail without
                ever being plausible scientific findings.
            verbose: print progress
            n_jobs_perm: parallel jobs for outer perms (1=serial, -1=all)
            cloud_config: CloudConfig for AWS execution (None = local)
            perm_dir: directory for per-perm result pickles.  If None
                a temp dir is created and cleaned up; if given, files
                are kept and reused on resume.
            cluster_mode: Ward projection.  Must be one of the keys in
                ``glow.analysis.cluster._MODES``.  Default
                ``"q1"`` (Focus) projects onto the contrast
                subspace; ``"q0, q1"`` (GLM Error) keeps bias
                + contrast; ``"all"`` (Naive) clusters raw y.
            use_fast_path: when True (default), enable the
                intercept-only Phase-1-precompute fast path in the
                inner FL loop when ``is_intercept_only_nuisance`` holds
                for ``exp``.  Set False to force the original Phase-1-
                per-inner-perm slow path — used by the runtime
                benchmark to measure the speedup factor.
        """
        super().__init__(exp, **kwargs)
        self.verbose = verbose
        self.cluster_mode = cluster_mode
        self.min_vox = min_vox
        self.n_perm_fwer = n_perm_fwer
        self.n_perm_inner = n_perm_inner
        self.use_fast_path = use_fast_path

        if cloud_config is not None:
            self._run_on_cloud(exp, n_perm_fwer, n_perm_inner,
                              alpha_fwer, min_vox, verbose,
                              cloud_config,
                              perms_per_job=kwargs.pop('perms_per_job', None),
                              cluster_mode=cluster_mode,
                              **kwargs)
            return

        b, num_img, num_vox = exp.y.shape

        # decompose() depends only on (x, contrast), which permute()
        # leaves untouched — hoist outside the per-perm loop and reuse
        # for every outer + inner FL draw.
        q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)

        # Permutation index layout:
        #   0                = observed data (outer perm 0)
        #   1..n_perm_fwer   = outer FL nulls (FWER comparison set)
        # Inner perms run inside each outer-perm worker (against that
        # worker's tree) and are not written to disk.
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
            print(f'  [1/2] outer perms: clustering + inner perms for '
                  f'{len(todo)} permutations ({num_vox} voxels, '
                  f'{n_perm_fwer} FWER, {n_perm_inner} inner) ...')

        if n_jobs_perm not in (0, 1) and todo:
            # Pin BLAS threads to 1 inside each worker.  Without this,
            # numpy / scipy / sklearn each spawn O(N_cores) BLAS threads
            # inside every joblib worker, leading to thread-pool
            # explosion and occasional SIGSEGV on large WGN volumes.
            with parallel_config(backend='loky', inner_max_num_threads=1):
                results = Parallel(
                    n_jobs=n_jobs_perm,
                    verbose=10 if verbose else 0,
                )(delayed(self._process_permutation)(
                        exp, perm_idx, n_perm_inner, min_vox, q0, q1)
                  for perm_idx in todo)
            for r in results:
                p = r['perm_idx']
                with open(perm_dir / f'{p:06d}_result.pkl', 'wb') as fh:
                    pickle.dump(r, fh)
                del r
        else:
            for perm_idx in tqdm(todo, desc='outer perms',
                                 disable=not verbose):
                r = self._process_permutation(
                    exp, perm_idx, n_perm_inner, min_vox, q0, q1)
                with open(perm_dir / f'{r["perm_idx"]:06d}_result.pkl',
                          'wb') as fh:
                    pickle.dump(r, fh)
                del r

        if verbose:
            print(f'  [2/2] per_region_z finalization ...')

        self._finalize_per_region_z(
            exp, perm_dir, n_perm_fwer, alpha_fwer, min_vox)

        if _cleanup_dir:
            shutil.rmtree(perm_dir, ignore_errors=True)

    def _finalize_per_region_z(self, exp, perm_dir, n_perm_fwer,
                                alpha_fwer, min_vox):
        """Per-region permutation z-scoring with Westfall-Young FWER.

        Reads each outer-perm worker's pickle (which already contains
        the worker-local mu/sigma/z + max_z), assembles the max-z null,
        and runs the FWER + pruning pipeline on the observed tree.

        With per-worker inner perms, there is no merged graph: each
        outer perm's z-scores are computed against its own tree.  The
        ``min_vox`` cutoff defends FWER power against the long tail of
        small (mostly tree-private) regions whose noise dominates the
        max-z null.
        """
        verbose = getattr(self, 'verbose', False)

        results = []
        for k in range(n_perm_fwer + 1):
            with open(perm_dir / f'{k:06d}_result.pkl', 'rb') as fh:
                results.append(pickle.load(fh))

        r0 = results[0]
        stat_0 = np.asarray(r0['stat'], dtype=float)
        size_0 = np.asarray(r0['size'], dtype=float)
        children_0 = np.asarray(r0['children'])
        z_0 = np.asarray(r0['z'], dtype=float)

        # max-z null: each outer-perm worker has already computed its
        # own max-z over (size >= min_vox) — just gather and sort.
        max_z_list = sorted(float(r['max_z']) for r in results)

        if verbose:
            print(f'  per_region_z: assembled max-z null from '
                  f'{len(max_z_list)} outer perms (min_vox={min_vox})')

        self._finalize_analysis(
            exp, n_perm_fwer,
            stat_0, size_0, children_0,
            z_0, max_z_list,
            alpha_fwer, min_vox)

        # diagnostics for the viewer (and the diag scripts).  With the
        # per-worker pipeline these are the observed tree's per-region
        # mu/sigma; there is no merged graph any more.
        self._mu_per_region = np.asarray(r0['mu'], dtype=float)
        self._sigma_per_region = np.asarray(r0['sigma'], dtype=float)

    @classmethod
    def from_precomputed(cls, *, exp, get_stat=None, verbose=False,
                         cluster_mode="q1"):
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
            n_perm_inner: number of inner FL permutations to run
                against this tree for per-region (mu, std).
            min_vox: minimum region size for the max-z comparison set.
            q0, q1: pre-decomposed contrast subspaces.  May be None
                when called via ``rerun_permutation``; in that case
                they are recomputed from ``exp``.

        Returns:
            dict with keys
                ``perm_idx``, ``children``, ``stat`` (raw LLR),
                ``size``, ``mu``, ``sigma``, ``z``, ``max_z``.
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
        del _exp

        # inner FL perms against THIS tree.  Phase 2 of compute_llr_batched
        # is skipped for regions with size < min_vox (they're inactive
        # in the FWER set anyway, so their mu/sigma is unused).  This
        # roughly halves the per-walk cost on typical neuroimaging trees
        # at min_vox=4.
        # Seed scheme: each outer perm reserves a 100_000-wide block,
        # far above any realistic n_perm_inner, so seeds never collide
        # across outer perms.
        n_inner_used = 0
        race_extra = {}
        llr_inner = None  # populated only by the buffer (non-race) path
        if n_perm_inner > 0:
            num_reg = llr_outer.shape[0]
            base = (perm_idx + 1) * 100_000
            n_img = exp.y.shape[1]
            q0_proj = q0.T @ q0                       # (N, N) nuisance projector
            eye_n = np.eye(n_img, dtype=q0.dtype)

            # Warmup draw: general slow path (works for any Q0).  Used
            # for the first race_init perms, after which the race trims
            # decided regions and we swap in the kernel-based draw_one
            # for the surviving subset via `on_warmup_done` below.
            #
            # The intercept-only Phase-1-precompute fast path used to
            # live here as a separate branch; it's been collapsed in
            # favour of the unified kernel path.  See
            # `glow.graph.build_survivor_kernels` and
            # `glow.graph.compute_llr_inner_kernel`.
            def draw_one(i):
                _exp_inner = exp.permute(base + i)
                llr_i, _ = glow.graph.compute_llr_batched(
                    _exp_inner, children=children, q0=q0, q1=q1,
                    min_size=min_vox, layer=layer)
                return llr_i

            def make_kernel_draw_one(active):
                # Build M_{ij} kernels once for the survivor set, then
                # serve every subsequent inner perm via a cheap
                # survivor-only gather instead of a fresh full-tree
                # Phase 1.  See graph.compute_llr_inner_kernel for the
                # algebra.
                survivor_idx = np.where(active)[0]
                if survivor_idx.size == 0:
                    return None
                kernels = glow.graph.build_survivor_kernels(
                    exp.y, children, survivor_idx, q0)

                def draw_one_kernel(i):
                    rng = np.random.default_rng(base + i)
                    perm = np.argsort(rng.permutation(n_img))
                    freed_lane = (eye_n - q0_proj)[perm, :] + q0_proj
                    return glow.graph.compute_llr_inner_kernel(
                        kernels, q0, q1, freed_lane, perm, num_reg,
                        min_size=min_vox)

                return draw_one_kernel

            if getattr(self, 'race_inner_perm', False):
                race_out = _run_inner_race(
                    draw_one, n_perm_inner, llr_outer, size, min_vox,
                    race_init=getattr(self, 'race_init', 50),
                    race_batch=getattr(self, 'race_batch', 25),
                    race_k_sigma=getattr(self, 'race_k_sigma', 3.0),
                    z_threshold=getattr(self, 'race_z_threshold', None),
                    on_warmup_done=make_kernel_draw_one)
                mu = race_out['mu']
                sigma = race_out['sigma']
                n_per_reg = race_out['n_per_reg']
                n_inner_used = race_out['n_inner_used']
                for k in ('leader', 'sig_regions'):
                    if k in race_out:
                        race_extra[k] = race_out[k]
            else:
                llr_inner = np.empty((n_perm_inner, num_reg), dtype=float)
                for i in range(n_perm_inner):
                    llr_inner[i, :] = draw_one(i)
                n_inner_used = n_perm_inner
                # Regions excluded by min_vox have all-NaN slices;
                # nanmean / nanstd legitimately return NaN for them
                # but emit RuntimeWarnings.  Silence those — the NaN
                # is the answer.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    mu = np.nanmean(llr_inner, axis=0)
                    sigma = np.nanstd(llr_inner, axis=0, ddof=1)
                # In the buffer path every region got the full budget
                # (or zero, if size<min_vox kept it all-NaN).
                n_per_reg = np.where(np.isfinite(mu), n_inner_used, 0
                                      ).astype(np.int64)
        else:
            # rerun_permutation path: caller only wants children/stat/size.
            mu = np.full_like(llr_outer, fill_value=np.nan)
            sigma = np.full_like(llr_outer, fill_value=np.nan)
            n_per_reg = np.zeros_like(llr_outer, dtype=np.int64)

        # zero-std guard (constant inner draws → divide-by-zero z).
        sigma_safe = np.where(sigma < 1e-12, 1.0, sigma)
        z = (llr_outer - mu) / sigma_safe
        z = _sanitize_adjusted_stat(z)

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
            'n_per_reg': n_per_reg,
        }
        result.update(race_extra)
        if (getattr(self, 'store_llr_inner', False) and llr_inner is not None):
            result['llr_inner'] = llr_inner
        return result

    @classmethod
    def rerun_permutation(cls, exp, perm_idx, get_stat=get_llr,
                          cluster_mode="q1", n_perm_inner=0, min_vox=4):
        """Re-run a single outer permutation for inspection.

        Defaults to ``n_perm_inner=0`` (skips the inner FL loop), which
        reproduces the cheap "just give me children/stat/size" use
        case of the old method.

        Returns:
            dict — see ``_process_permutation``.  With
            ``n_perm_inner=0`` the ``mu``/``sigma`` arrays are NaN and
            ``z`` reduces to the (sanitised) raw LLR.
        """
        ana = cls.from_precomputed(exp=exp, get_stat=get_stat)
        ana.cluster_mode = cluster_mode
        return ana._process_permutation(
            exp, perm_idx, n_perm_inner, min_vox)

    def _finalize_analysis(self, exp, n_perm_fwer,
                          stat_0, size_0, children_0,
                          llr_z_0, stat_max_sorted,
                          alpha_fwer, min_size,
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
        """
        num_reg = stat_0.shape[0]

        llr_z_0 = _sanitize_adjusted_stat(np.asarray(llr_z_0, dtype=float))

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

        self.rethreshold(alpha_fwer)

    def rethreshold(self, alpha_fwer):
        """Re-apply the FWER cutoff using stored permutation stats.

        Recomputes the alpha-dependent tail of the analysis from
        ``self.pval``, ``self.stat``, ``self.children``, and
        ``self.stat_max_sorted``: ``adj_crit``, ``sig_reg_list``,
        the greedy prune, and ``effect_list``.  No permutation work
        is re-run.
        """
        verbose = getattr(self, 'verbose', False)
        self.alpha_fwer = alpha_fwer

        n_total = len(self.stat_max_sorted)
        if n_total > 0:
            crit_idx = min(int(np.ceil(n_total * (1 - alpha_fwer))),
                           n_total - 1)
            self.adj_crit = float(self.stat_max_sorted[crit_idx])
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
        stat_gain = np.nan_to_num(self.stat.astype(float), nan=0.0,
                                  posinf=0.0, neginf=0.0)

        reg_out_list, self.prune_info = prune_greedy(
            sig_reg_list=self.sig_reg_list,
            children=self.children,
            stat=stat_gain)

        self.effect_list = list()
        for reg_idx in reg_out_list:
            label_map = glow.graph.get_label_map(
                reg_idx_list=[reg_idx],
                mask_idx=self.exp.mask_idx,
                children=self.children)
            pval_fwer = self.pval[reg_idx]
            eff = glow.effect.EffectEstimate.from_exp_mask(
                mask=label_map > -1, exp=self.exp,
                reg_idx=reg_idx, pval_fwer=pval_fwer)
            self.effect_list.append(eff)

        if verbose:
            n_disc = len(self.effect_list)
            n_pruned = len(self.sig_reg_list) - n_disc
            print(f'  done: {n_disc} discovered, {n_pruned} pruned')

        return self

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
                     cluster_mode="q1",
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
            'get_stat': self.get_stat,
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

        print(f'cloud analysis complete: found {len(self.effect_list)} effects')

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
