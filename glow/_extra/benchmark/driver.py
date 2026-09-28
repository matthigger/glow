"""Drive the benchmark: get_exp_effect -> fnc, as a grid.

drive sweeps the cartesian product of two kwargs grids -- the data and effect
halves of a cell -- and runs a leaf function fnc on each cell, once per
kwargs dict in kwargs_fnc_list, as nested loops:

    for kw_data  in kwargs_data_list:
      for kw_eff in kwargs_effect_list:  cell  = get_exp_effect(...)
        for kw   in kwargs_fnc_list:     score = fnc(cell, ...)

Nested rather than one flat product so a data cell's effect cells are
realized back to back, which is what lets them share one clean build in
memory (.cell). Both stages are disk-memoised and recorded (see .cell /
.run), so a repeated cell is a cache hit and a resumed sweep reuses the
stored artifacts.

skip_recorded exists because that memoisation reaches only the joblib cache,
which is the half that gets archived or pruned for space (mv_cache) while the
records stay. It drops any (data, effect) cell whose whole leaf set is already
recorded (results.get_cell_complete), so a rerun never realizes a cell it has
no work for, and within a cell it keeps only the leaves still missing
(results.get_leaf_todo) -- a cell short one leaf owes that leaf, not its whole
grid.

fnc is the leaf measurement, swept over its own kwargs grid so one cell can
be measured several ways at once. It is called fnc(cell, **kwargs) and, to
join the provenance DAG, must be memoised + recorded and take the cell's uid
as its parent_uid -- the shape of run_ana. The driver knows nothing
fnc-specific; the config layer supplies it.

A None effect cell is the null / FWER-calibration path: the cell plants
nothing and its leaves score against an empty target.

drive returns the innermost scores, but the richer output is the provenance DAG
every call writes to: RECORDER.flatten_to_df yields one row per fnc leaf
carrying the cell spec that produced it. Which CONFIG cache a leaf belongs to
is recomputed at read time from the declared uids (see .results), so the
driver keeps no per-sweep bookkeeping.

Parallelism (n_jobs != 1) splits the sweep by data cell: each whole
get_exp_effect* -> fnc* subtree is one joblib task, so a cell has exactly one
owner and no two workers race to write its cache or record.
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

from .cell import get_exp_effect
from .store import RECORDER

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


def _report_failed(fail_list) -> None:
    """Print the cells a sweep skipped, or nothing when none were.

    Loud on purpose: a skipped cell leaves a hole the seed accounting will
    later report as an unfinished seed, and the only other trace is its
    absence from the records.

    Args:
        fail_list (list[dict]): the failure records _run_data_cell returned.
    """
    if not fail_list:
        return
    print(f'\n[drive] {len(fail_list)} cell(s) failed and were skipped; '
          f'{sum(f["n_leaf"] for f in fail_list)} leaf/leaves unrun')
    for f in fail_list:
        print(f'  {f["error"]}')
        print(f'    data:   {f["kwargs_data"]}')
        print(f'    effect: {f["kwargs_effect"]}')


def _run_data_cell(kwargs_data, effect_plan, fnc, bar=None):
    """Realize one data cell's cells, run their fnc grid, return the scores.

    The per-data-cell unit of work, shared by the serial loop and the
    parallel tasks: realize each of the data cell's (data, effect) cells --
    which share one clean build in memory -- and run fnc over that cell's
    kwargs grid.

    Each effect carries its own fnc grid rather than sharing one, which is what
    lets the caller hand a cell only the leaves it still owes (see drive).

    Args:
        kwargs_data (dict): the data half of a cell (see .cell).
        effect_plan (list[tuple]): (kwargs_effect, kwargs_fnc_list) pairs --
            the effect cells to plant, each with the fnc kwargs grid to run on
            it. A None kwargs_effect plants nothing (the null path).
        fnc (Callable): the leaf measurement, fnc(cell, **kwargs).
        bar (tqdm | None): progress bar to advance one step per fnc leaf, or
            None to advance nothing (the parallel path, where a worker cannot
            reach the caller's bar -- it ticks per returned cell instead).

    Returns:
        tuple[list[dict], list[dict]]: this cell's fnc scores in (effect,
            fnc-kwargs) order, and one failure record per (data, effect) cell
            that raised (see drive for the schema).
    """
    score_list, fail_list = [], []
    for kwargs_effect, kwargs_fnc_list in effect_plan:
        n_done = 0
        try:
            # a cell carries its own declared uid, which its leaves are passed
            # as their parent: provenance is declared on the way down rather
            # than rediscovered afterwards from array content hashes
            # (see .recipe)
            cell = get_exp_effect(kwargs_data, kwargs_effect)
            for kwargs in kwargs_fnc_list:
                score = fnc(cell, parent_uid=cell.uid, **kwargs)
                score_list.append(score)
                n_done += 1
                if bar is not None:
                    bar.update(1)
        except Exception as exc:
            # One cell that will not build is one cell, not the sweep. A plant
            # that re-measures off its target (cell.LLR_RTOL) is deterministic,
            # so raising here would stop this grid at the same place on every
            # rerun and leave every later cell permanently unrun.
            fail = dict(kwargs_data=str(kwargs_data),
                        kwargs_effect=str(kwargs_effect),
                        n_leaf=len(kwargs_fnc_list) - n_done,
                        error=f'{type(exc).__name__}: {exc}')
            fail_list.append(fail)
            # at once, not only in the end-of-sweep tally: a cache takes
            # hours, and a silent hole is what the seed accounting trips
            # over long afterwards
            print(f'\n[drive] skipping cell: {fail["error"]}', flush=True)
            if bar is not None:
                bar.update(len(kwargs_fnc_list) - n_done)
    return score_list, fail_list


def drive(kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc, *,
          n_jobs=1, verbose=False, skip_recorded=False):
    """Sweep data x effect, run fnc per cell over its kwargs, return scores.

    The two upstream lists are kwargs grids for the data and effect halves
    of a cell; the driver runs their cartesian product, realizing each cell
    and handing it to fnc. Both stages are memoised + recorded, so this only
    forwards kwargs.

    kwargs_effect_list and kwargs_fnc_list are pulled into lists up front,
    being re-iterated per cell, so one-shot generators are fine;
    kwargs_data_list is iterated once and stays lazy.

    skip_recorded drops the (data, effect) cells the records already hold in
    full, which also skips realizing them -- the expensive part. A cell
    holding only some of its leaves is realized, but runs just the leaves it
    lacks: its recorded siblings are not recomputed, which matters wherever
    only their records were merged and the joblib cache cannot answer for them
    (see the module docstring for why the records, not the cache, are the
    source of truth).

    With n_jobs != 1 the sweep runs over joblib, one task per data cell. A
    leaf that parallelises its own fit multiplies against that n_jobs, so
    check_fit_params runs before any cell is realized and refuses a sweep whose
    product would oversubscribe the CPU or put several fits on one GPU.

    A verbose sweep needs len(data) for its bar, so it pulls
    kwargs_data_list into a list up front; otherwise that stays lazy.

    Args:
        kwargs_data_list (iterable[dict]): one kwargs dict per data
            call, e.g. {'source': 'wgn', 'shape': (5, 5, 5), 'seed': 0}.
        kwargs_effect_list (iterable[dict | None]): one kwargs dict per
            effect half, e.g. {'kind': 'single', 'effect_llr': 0.05,
            'extenter_cls': ExtenterMinVar, 'n_vox_frac': 0.1}; a None cell
            plants no effect (the null / FWER-calibration path, scored
            against an empty target).
        kwargs_fnc_list (iterable[dict]): one kwargs dict per fnc call on
            each cell (the cell and its parent_uid are supplied by the
            driver), e.g. [{'ana': AnalysisGLOW(n_perm_fwer=250)}].
        fnc (Callable): the leaf measurement, called fnc(cell, **kwargs) and
            memoised + recorded (see module docstring), e.g. run_ana.
        n_jobs (int): 1 (default) runs serially in-process; otherwise the data
            cells run in parallel over joblib.Parallel(n_jobs=n_jobs).
        verbose (bool): False (default) runs silently; True shows a tqdm bar
            over the total leaf count, advanced per fnc call (see module
            docstring), and reports how many cells the records skipped.
        skip_recorded (bool): False (default) runs every cell of the grid;
            True drops the cells already complete in the records and, within
            the rest, the individual leaves already recorded -- which
            materialises kwargs_data_list (the walk needs it up front).

    A (data, effect) cell that raises is skipped, not fatal: the sweep goes
    on and _report_failed prints what was dropped at the end. A cell that
    cannot build fails the same way on every rerun, so raising would stop
    the grid at the same index forever and leave every later cell unrun.

    Returns:
        list[dict]: the fnc score dicts, one per (data, effect, fnc-kwargs)
            cell in data-cell order (effect then fnc-kwargs within a cell),
            covering only the leaves that ran under skip_recorded, less any
            the skip above dropped. See
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

    # the sweep as (kwargs_data, [(kwargs_effect, its fnc grid), ...]) pairs:
    # the whole grid per data cell, or -- under skip_recorded -- only the
    # effect cells the records lack, each cut down to the leaves it lacks,
    # dropping a data cell left with none so its exp is never built.
    total = None
    if skip_recorded:
        # imported here: results pulls in the CONFIG catalogue, which a plain
        # drive never needs. RECORDER.load first, so the walk sees what other
        # writers (a parallel sweep, a run on another machine) left on disk.
        from .results import get_cell_complete, get_leaf_todo

        RECORDER.load()
        cell_complete = get_cell_complete(kwargs_fnc_list, fnc)
        leaf_todo = get_leaf_todo(kwargs_fnc_list, fnc)

        plan, n_skip, n_leaf_skip = [], 0, 0
        for kwargs_data in kwargs_data_list:
            todo = []
            for kwargs_effect in kwargs_effect_list:
                # the cell-level walk first: it also reaches cells recorded
                # before the recipe fields, which the uid check cannot
                if cell_complete(kwargs_data, kwargs_effect):
                    n_skip += 1
                    continue
                fnc_todo = leaf_todo(kwargs_data, kwargs_effect)
                if not fnc_todo:
                    n_skip += 1
                    continue
                n_leaf_skip += len(kwargs_fnc_list) - len(fnc_todo)
                todo.append((kwargs_effect, fnc_todo))
            if todo:
                plan.append((kwargs_data, todo))

        n_run = sum(len(todo) for _, todo in plan)
        total = sum(len(fnc_todo) for _, todo in plan for _, fnc_todo in todo)
        if verbose and (n_skip or n_leaf_skip):
            note = (f'[drive] {n_skip} cell(s) already complete in records, '
                    f'skipped; {n_run} to run')
            if n_leaf_skip:
                note += f' ({n_leaf_skip} recorded leaf/leaves within them)'
            print(note)
    else:
        # the bar spans the total leaf count, known once data is a list;
        # materialise data when verbose (else keep it lazy, iterated once, as
        # documented above).
        if verbose:
            kwargs_data_list = list(kwargs_data_list)
            total = (len(kwargs_data_list) * len(kwargs_effect_list)
                     * len(kwargs_fnc_list))
        plan = ((kwargs_data, [(kwargs_effect, kwargs_fnc_list)
                               for kwargs_effect in kwargs_effect_list])
                for kwargs_data in kwargs_data_list)

    if n_jobs == 1:
        score_list, fail_list = [], []
        with tqdm(total=total, desc='drive', disable=not verbose) as bar:
            for kwargs_data, effect_plan in plan:
                scores, fails = _run_data_cell(
                    kwargs_data, effect_plan, fnc, bar=bar)
                score_list.extend(scores)
                fail_list.extend(fails)
        _report_failed(fail_list)
        return score_list

    # parallel: one task per data cell, so each build has a single owner -- no
    # two workers compute / write the same cell (see module docstring).
    from joblib import Parallel, delayed

    # return_as='generator' streams results in submission order (so the
    # returned scores keep their order) as tasks drain, letting the bar advance
    # per cell.
    results = Parallel(n_jobs=n_jobs, return_as='generator')(
        delayed(_run_data_cell)(kwargs_data, effect_plan, fnc)
        for kwargs_data, effect_plan in plan)

    cell_scores, fail_list = [], []
    with tqdm(total=total, desc='drive', disable=not verbose) as bar:
        for scores, fails in results:
            cell_scores.append(scores)
            fail_list.extend(fails)
            bar.update(len(scores) + sum(f['n_leaf'] for f in fails))
    _report_failed(fail_list)

    # workers wrote their per-hash record files to the shared folder in their
    # own processes; fold them into this process's recorder so flatten_to_df
    # sees the whole sweep (a no-op merge when the backend shares this
    # process's memory).
    RECORDER.load()
    return [score for scores in cell_scores for score in scores]
