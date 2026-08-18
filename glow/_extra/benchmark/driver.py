"""Drive the benchmark: data_factory -> effect_factory -> fnc, as a grid.

drive sweeps the cartesian product of two upstream kwargs grids -- the data
and effect stages -- and runs a leaf function fnc on each planted cell, once
per kwargs dict in kwargs_fnc_list, as nested loops:

    for kw_data  in kwargs_data_list:    exp        = data_factory(...)
      for kw_eff in kwargs_effect_list:  exp, mask  = effect_factory(exp, ...)
        for kw   in kwargs_fnc_list:     score      = fnc(exp, ...)

Nested rather than one flat product so each upstream stage's output feeds the
stage below: a clean exp is built once per data cell and a planted exp once
per (data, effect) cell, then shared by every fnc run under it. Every stage is
disk-memoised and recorded (see .data / .run), so a repeated cell is a cache
hit and a resumed sweep reuses the stored artifacts.

skip_recorded exists because that memoisation reaches only the joblib cache,
which is the half that gets archived or pruned for space (mv_cache) while the
records stay. It drops any (data, effect) cell whose whole leaf set is already
recorded (results.get_cell_complete), so a rerun fills only the gaps and never
builds an exp it has no work for.

fnc is the leaf measurement, swept over its own kwargs grid so one (data,
effect) cell can be measured several ways at once. It is called
fnc(exp, mask_target_list=..., **kwargs) and, to join the provenance DAG, must
be memoised + recorded with exp as a linked input -- the shape of run_ana. The
driver knows nothing fnc-specific; the config layer supplies it.

A None effect cell is the null / FWER-calibration path: it plants nothing and
runs fnc on the clean exp against an empty target, skipping effect_factory
(EffectSynthetic has no no-op), so its provenance chains straight to the build.

drive returns the innermost scores, but the richer output is the provenance DAG
every call writes to: RECORDER.flatten_to_df yields one row per fnc leaf
carrying the data / effect inputs that produced it. Which CONFIG cache a leaf
belongs to is recomputed at read time by walking the records forward (see
.results), so the driver keeps no per-sweep bookkeeping.

Parallelism (n_jobs != 1) splits the sweep by data cell: each whole
data_factory -> effect_factory* -> fnc* subtree is one joblib task, so a build
has exactly one owner and no two workers race to write its cache or record.
That is the right split while the data grids satiate the pool. A leaf may also
parallelise its own fit (run_ana's fit_params), which multiplies against
n_jobs rather than sharing it, so check_fit_params refuses the combinations
that would oversubscribe. The per-hash record files the workers write are
folded back in with RECORDER.load once the pool drains.

A verbose drive shows a tqdm bar over the total leaf count, known up front
from the grid sizes less whatever skip_recorded dropped. It cannot tell a
cache hit from a real compute, so the bar lurches, but it stays bounded. The
serial bar ticks per leaf; the parallel bar ticks per data cell, since a
worker cannot reach the caller's bar.
"""

import os

from tqdm import tqdm

from .data import (RECORDER, data_factory, data_recipe, effect_factory,
                   effect_recipe)

# How far a sweep may oversubscribe the CPU before check_fit_params stops it:
# drive's own n_jobs times the worst leaf's, against this many times the core
# count. Some slack is deliberate (a fit is not compute-bound end to end), but
# an order of magnitude is a typo, not a scheduling choice.
_CPU_OVERSUBSCRIBE_FACTOR = 2


def _leaf_n_jobs(kwargs) -> int:
    """Return the worker count one leaf-kwargs cell asks Analysis.fit for."""
    n_jobs = (kwargs.get('fit_params') or {}).get('n_jobs', 1)
    n_cpu = os.cpu_count() or 1
    return max(1, n_cpu + 1 + n_jobs) if n_jobs < 0 else max(1, n_jobs)


def _leaf_wants_device(kwargs) -> bool:
    """True if this leaf-kwargs cell would actually claim a CUDA device.

    gpu='auto' claims one only where one is visible, so it is read against
    this machine rather than treated as a request: an 'auto' grid must stay
    runnable at any n_jobs on the CPU-only boxes (CI, a machine without a
    card) that are the reason to write 'auto' in the first place.
    """
    gpu = (kwargs.get('fit_params') or {}).get('gpu', False)
    if not gpu:
        return False
    if gpu == 'auto':
        from glow.analysis import draws_gpu

        return draws_gpu.is_available()
    return True


def check_fit_params(kwargs_fnc_list, n_jobs: int) -> None:
    """Raise before a sweep that would oversubscribe the GPU or the CPU.

    drive's n_jobs and a leaf's fit_params multiply: -j8 over a grid whose
    GLOW leaf asks for 32 workers and a device is 8 concurrent fits, 256
    joblib workers and 8 CUDA contexts on one card. Nothing downstream
    notices -- joblib takes both counts literally and the device OOMs an
    hour in -- so the arithmetic is checked here, before a cell is built,
    and the sweep refuses rather than thrashes.

    Only the product is judged, not either factor: -j1 with a 32-worker
    device leaf is the intended way to run GLOW, and -j8 over CPU-only
    leaves is the intended way to run everything else.

    Args:
        kwargs_fnc_list (list[dict]): the leaf-fnc kwargs grid, whose cells
            may carry a fit_params dict (see run.run_ana).
        n_jobs (int): drive's own data-cell parallelism.

    Raises:
        ValueError: the sweep would run concurrent device fits, or would
            ask for more than _CPU_OVERSUBSCRIBE_FACTOR times the machine's
            cores.
    """
    n_cell = _leaf_n_jobs({'fit_params': {'n_jobs': n_jobs}})
    if n_cell == 1:
        return

    if any(_leaf_wants_device(kwargs) for kwargs in kwargs_fnc_list):
        raise ValueError(
            f'n_jobs={n_jobs} would run {n_cell} fits at once, and a leaf '
            f'recipe asks for a GPU: one CUDA context per concurrent fit on '
            f'one card. Sweep device leaves with n_jobs=1 (the device fit '
            f'already uses the whole machine), or drop the device for this '
            f'sweep (grid.strip_gpu, --no-gpu on the CLI).')

    n_leaf = max((_leaf_n_jobs(kwargs) for kwargs in kwargs_fnc_list),
                 default=1)
    n_cpu = os.cpu_count() or 1
    if n_cell * n_leaf > _CPU_OVERSUBSCRIBE_FACTOR * n_cpu:
        raise ValueError(
            f'n_jobs={n_jobs} ({n_cell} cells at once) times the leaf grid\'s '
            f'fit_params n_jobs={n_leaf} is {n_cell * n_leaf} workers on '
            f'{n_cpu} cores. Lower one of the two: a GLOW fit also holds a '
            f'copy of y per worker (~1 GB at full-brain num_vox).')


def _run_data_cell(kwargs_data, kwargs_effect_list, kwargs_fnc_list, fnc,
                   bar=None):
    """Build one data cell, run its effect x fnc subtree, return its scores.

    The per-data-cell unit of work, shared by the serial loop and the parallel
    tasks: build the clean exp once (then reuse it across the effect / fnc
    loops below), plant each effect (or skip it for a None cell), and run fnc
    over its kwargs grid on each planted cell.

    Args:
        kwargs_data (dict): kwargs for one data_factory call.
        kwargs_effect_list (list[dict | None]): the effect grid (a list).
        kwargs_fnc_list (list[dict]): the fnc kwargs grid (a list).
        fnc (Callable): the leaf measurement,
            fnc(exp, mask_target_list=..., **kwargs).
        bar (tqdm | None): progress bar to advance one step per fnc leaf, or
            None to advance nothing (the parallel path, where a worker cannot
            reach the caller's bar -- it ticks per returned cell instead).

    Returns:
        list[dict]: this cell's fnc scores, in (effect, fnc-kwargs) order.
    """
    # the cell's uid chain, named from the kwargs before anything is built:
    # each stage is told its parent's uid, so provenance is declared on the way
    # down rather than rediscovered afterwards from array content hashes (see
    # glow._extra.benchmark.recipe).
    uid_data = data_recipe(kwargs_data).uid
    exp = data_factory(**kwargs_data)
    score_list = []
    for kwargs_effect in kwargs_effect_list:
        if kwargs_effect is None:
            # null / FWER-calibration cell: no effect, empty target, so the
            # leaf hangs off the clean exp itself
            exp_eff, mask_target_list = exp, []
            uid_parent = uid_data
        else:
            # effect_factory returns the realized supports as a list (one
            # entry for a single effect, two for a split), threaded as-is
            exp_eff, mask_target_list = effect_factory(
                exp, parent_uid=uid_data, **kwargs_effect)
            uid_parent = effect_recipe(kwargs_effect, uid_data).uid
        for kwargs in kwargs_fnc_list:
            score = fnc(exp_eff, mask_target_list=mask_target_list,
                        parent_uid=uid_parent, **kwargs)
            score_list.append(score)
            if bar is not None:
                bar.update(1)
    return score_list


def drive(kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc, *,
          n_jobs=1, verbose=False, skip_recorded=False):
    """Sweep data x effect, run fnc per cell over its kwargs, return scores.

    The two upstream lists are kwargs grids for the data and effect stages;
    the driver runs their cartesian product, threading each stage's output
    into the next -- the clean exp into effect_factory, then the planted exp
    and the realized supports into fnc as mask_target_list (one entry for a
    single effect, two for a split). Every stage is memoised + recorded, so
    this only forwards kwargs.

    kwargs_effect_list and kwargs_fnc_list are pulled into lists up front,
    being re-iterated per cell, so one-shot generators are fine;
    kwargs_data_list is iterated once and stays lazy.

    skip_recorded drops the (data, effect) cells the records already hold in
    full, which also skips building their exp -- the expensive part. A cell
    reads as incomplete unless every leaf is present, so a partial cell reruns
    whole (see the module docstring for why the records, not the cache, are
    the source of truth).

    With n_jobs != 1 the sweep runs over joblib, one task per data cell. A
    leaf that parallelises its own fit multiplies against that n_jobs, so
    check_fit_params runs before any cell is built and refuses a sweep whose
    product would oversubscribe the CPU or put several fits on one GPU.

    A verbose sweep needs len(data) for its bar, so it pulls
    kwargs_data_list into a list up front; otherwise that stays lazy.

    Args:
        kwargs_data_list (iterable[dict]): one kwargs dict per data_factory
            call, e.g. {'source': 'wgn', 'shape': (5, 5, 5), 'seed': 0}.
        kwargs_effect_list (iterable[dict | None]): one kwargs dict per
            effect_factory call (exp is supplied by the driver), e.g.
            {'kind': 'single', 'effect_llr': 0.05, 'extenter_cls':
            ExtenterMinVar, 'n_vox_frac': 0.1, 'seed_from_exp': True}; a None
            cell
            plants no effect (the null / FWER-calibration path, run on the
            clean exp with an empty target).
        kwargs_fnc_list (iterable[dict]): one kwargs dict per fnc call on each
            planted cell (exp and mask_target_list are supplied by the driver),
            e.g. [{'ana': AnalysisGLOW(n_perm_fwer=250)}].
        fnc (Callable): the leaf measurement, called
            fnc(exp, mask_target_list=..., **kwargs) and memoised + recorded
            with exp linked (see module docstring), e.g. run_ana.
        n_jobs (int): 1 (default) runs serially in-process; otherwise the data
            cells run in parallel over joblib.Parallel(n_jobs=n_jobs).
        verbose (bool): False (default) runs silently; True shows a tqdm bar
            over the total leaf count, advanced per fnc call (see module
            docstring), and reports how many cells the records skipped.
        skip_recorded (bool): False (default) runs every cell of the grid;
            True drops the cells already complete in the records, which
            materialises kwargs_data_list (the walk needs it up front).

    Returns:
        list[dict]: the fnc score dicts, one per (data, effect, fnc-kwargs)
            cell in data-cell order (effect then fnc-kwargs within a cell),
            covering only the cells that ran under skip_recorded. See
            glow._extra.benchmark.score.score_effects for the schema; the
            per-cell provenance is on the shared recorder, not here.
    """
    # the effect / fnc grids are re-iterated per data (resp. data x effect)
    # cell, so pull them into lists once -- a one-shot generator (as config
    # passes) would otherwise be spent after the first data cell. data is
    # iterated once, so it stays lazy.
    kwargs_effect_list = list(kwargs_effect_list)
    kwargs_fnc_list = list(kwargs_fnc_list)

    check_fit_params(kwargs_fnc_list, n_jobs)

    # the sweep as (kwargs_data, that cell's effect grid) pairs: the whole
    # effect grid per data cell, or -- under skip_recorded -- only the effect
    # cells the records lack, dropping a data cell left with none so its exp is
    # never built.
    total = None
    if skip_recorded:
        # imported here: results pulls in the CONFIG catalogue, which a plain
        # drive never needs. RECORDER.load first, so the walk sees what other
        # writers (a parallel sweep, a run on another machine) left on disk.
        from .results import get_cell_complete

        RECORDER.load()
        cell_complete = get_cell_complete(kwargs_fnc_list, fnc)

        plan, n_skip = [], 0
        for kwargs_data in kwargs_data_list:
            todo = [kwargs_effect for kwargs_effect in kwargs_effect_list
                    if not cell_complete(kwargs_data, kwargs_effect)]
            n_skip += len(kwargs_effect_list) - len(todo)
            if todo:
                plan.append((kwargs_data, todo))

        n_run = sum(len(todo) for _, todo in plan)
        total = n_run * len(kwargs_fnc_list)
        if verbose and n_skip:
            print(f'[drive] {n_skip} cell(s) already complete in records, '
                  f'skipped; {n_run} to run')
    else:
        # the bar spans the total leaf count, known once data is a list;
        # materialise data when verbose (else keep it lazy, iterated once, as
        # documented above).
        if verbose:
            kwargs_data_list = list(kwargs_data_list)
            total = (len(kwargs_data_list) * len(kwargs_effect_list)
                     * len(kwargs_fnc_list))
        plan = ((kwargs_data, kwargs_effect_list)
                for kwargs_data in kwargs_data_list)

    if n_jobs == 1:
        score_list = []
        with tqdm(total=total, desc='drive', disable=not verbose) as bar:
            for kwargs_data, effect_list in plan:
                score_list.extend(_run_data_cell(
                    kwargs_data, effect_list, kwargs_fnc_list, fnc,
                    bar=bar))
        return score_list

    # parallel: one task per data cell, so each build has a single owner -- no
    # two workers compute / write the same cell (see module docstring).
    from joblib import Parallel, delayed

    # return_as='generator' streams results in submission order (so the
    # returned scores keep their order) as tasks drain, letting the bar advance
    # per cell.
    results = Parallel(n_jobs=n_jobs, return_as='generator')(
        delayed(_run_data_cell)(
            kwargs_data, effect_list, kwargs_fnc_list, fnc)
        for kwargs_data, effect_list in plan)

    cell_scores = []
    with tqdm(total=total, desc='drive', disable=not verbose) as bar:
        for scores in results:
            cell_scores.append(scores)
            bar.update(len(scores))

    # workers wrote their per-hash record files to the shared folder in their
    # own processes; fold them into this process's recorder so flatten_to_df
    # sees the whole sweep (a no-op merge when the backend shares this
    # process's memory).
    RECORDER.load()
    return [score for scores in cell_scores for score in scores]
