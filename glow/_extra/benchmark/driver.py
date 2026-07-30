"""Drive the benchmark: data_factory -> effect_factory -> fnc, as a grid.

drive sweeps the cartesian product of two upstream kwargs grids -- the data
and effect stages -- and runs a leaf function fnc on each planted cell, once
per kwargs dict in kwargs_fnc_list, as nested loops:

    for kw_data  in kwargs_data_list:    exp        = data_factory(...)
      for kw_eff in kwargs_effect_list:  exp, mask  = effect_factory(exp, ...)
        for kw   in kwargs_fnc_list:     score      = fnc(exp, ...)

The nesting (rather than one flat product) is deliberate: each upstream
stage's output feeds the stage below, so a clean exp is built once per data
cell and a planted exp once per (data, effect) cell, then shared across every
fnc run below it -- no redundant rebuild even on a cold cache. The stage
functions are each disk-memoised + recorded (see glow._extra.benchmark.data /
run), so a repeated cell is a cache hit and the driver never worries about
issuing the same call twice: a re-run, an overlapping grid, or a resumed
sweep all reuse the stored artifacts.

That memoisation reaches only the local joblib cache, which is why drive takes
skip_recorded: the records travel where the cache does not. An AWS run ships
its records home but not its cache entries (see glow._extra.aws), so its
finished cells are provably done yet miss locally and would recompute from
cold. skip_recorded drops any (data, effect) cell whose whole leaf set is
already recorded before the sweep starts -- the same records-as-source-of-truth
skip the AWS driver applies to its cells (results.get_cell_complete) -- so a
local rerun fills only the gaps, and a cell with nothing left to run never
builds its exp.

fnc is the leaf measurement, swept over its own kwargs grid so one (data,
effect) cell can be measured several ways at once (e.g. run_ana under several
Analysis recipes). It is called fnc(exp, mask_target_list=..., **kwargs) and,
to join the provenance DAG, must be memoised + recorded with exp as a linked
input -- exactly the shape of run_ana (see glow._extra.benchmark.run). The
driver knows nothing fnc-specific; the config layer supplies it.

A None effect cell is the null / FWER-calibration path: it plants nothing and
runs fnc on the clean exp against an empty target. effect_factory is skipped
entirely (EffectSynthetic has no no-op -- it imposes a real effect_llr), so a
null run is data_factory -> fnc directly, and its provenance row chains
straight to the build with no plant node (see glow._extra.benchmark.config's
'null' cache, whose effect grid is [None]).

The driver returns the innermost scores, but the richer output is the shared
provenance DAG every call writes to: RECORDER.flatten_to_df yields one row
per fnc leaf, each carrying the swept data / effect inputs that produced it
(the data -> plant -> score chain), so a sweep is analysed from the records
without the driver tracking anything itself (see
glow._extra.benchmark.recorder).

Which CONFIG cache a leaf belongs to is recomputed at read time by walking the
records forward from the cache's data cells (see glow._extra.benchmark.results),
so the driver records provenance and nothing else -- no per-sweep bookkeeping.

Parallelism (n_jobs != 1) splits the sweep by data cell: each whole
data_factory -> effect_factory* -> fnc* subtree is one joblib task, so the
clean exp (and each planted exp) has exactly one owner. No two workers ever
compute -- or race to write the cache / record of -- the same build, and the
exp it builds is reused in-process across its subtree, never pickled between
workers. This is the right split while the data grids satiate the pool (the
sweeps run 15-1000 seeds); a narrower grid would leave workers idle and want
a finer (two-phase) split, deferred until needed. The stage functions are
shared module-level singletons re-imported in each worker, so workers
transparently share the on-disk cache and records folder; the per-hash record
files they write are folded back into this process with RECORDER.load once
the pool drains. (The leaf fnc is pickled to the workers like any task
argument; the recorder snapshots its records as they are made, so the wrapped
fnc carries no live Experiments and stays light to ship -- see
glow._extra.benchmark.recorder.)

A verbose drive shows a tqdm bar over the total leaf count, known up front
from the grid sizes (n_data * n_effect * n_fnc, less whatever skip_recorded
dropped) and advanced one leaf per fnc call. It makes no attempt to tell a
real compute from a cache hit, so the bar lurches -- racing through cached
cells, crawling through the ones that actually run -- but it stays bounded and
honest about how far the sweep has left to go. The serial bar ticks per leaf;
the parallel bar ticks per data cell as each task returns its scores, since a
worker cannot reach the caller's bar.
"""

from tqdm import tqdm

from .data import RECORDER, data_factory, effect_factory


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
    exp = data_factory(**kwargs_data)
    score_list = []
    for kwargs_effect in kwargs_effect_list:
        if kwargs_effect is None:
            # null / FWER-calibration cell: no effect, empty target
            exp_eff, mask_target_list = exp, []
        else:
            # effect_factory returns the realized supports as a list (one
            # entry for a single effect, two for a split), threaded as-is
            exp_eff, mask_target_list = effect_factory(exp, **kwargs_effect)
        for kwargs in kwargs_fnc_list:
            score = fnc(exp_eff, mask_target_list=mask_target_list, **kwargs)
            score_list.append(score)
            if bar is not None:
                bar.update(1)
    return score_list


def drive(kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc, *,
          n_jobs=1, verbose=False, skip_recorded=False):
    """Sweep data x effect, run fnc per cell over its kwargs, return scores.

    The two upstream lists are kwargs grids for the data and effect stages;
    the driver runs their cartesian product, threading each stage's output
    into the next (the clean exp into effect_factory, the planted exp and the
    effect factory's realized supports into fnc as mask_target_list -- one
    entry for a single effect, two for a split).
    Each planted cell is then measured by fnc once per kwargs dict in
    kwargs_fnc_list. All stages and fnc are memoised + recorded, so this only
    forwards kwargs -- caching dedupes repeated cells and the recorder
    captures provenance (see module docstring).

    kwargs_effect_list and kwargs_fnc_list are pulled into lists up front
    (they are re-iterated once per data / per data x effect cell), so one-shot
    generators are fine -- as the config layer passes; kwargs_data_list is
    iterated once and stays lazy.

    skip_recorded drops the (data, effect) cells the records already hold in
    full before anything runs (results.get_cell_complete). The memoised stages
    already dedupe a repeat, but only against the local joblib cache, which
    holds nothing a run elsewhere computed -- the AWS path ships records, not
    cache entries, so its finished cells would otherwise recompute from cold
    here. Skipping a cell whose whole effect subtree is done also skips
    building its exp, the expensive part. The records are the source of truth
    either way (the same skip the AWS driver applies), and a cell reads as
    incomplete unless every leaf is present, so a partial cell reruns whole.

    With n_jobs != 1 the sweep runs in parallel over joblib, one task per data
    cell (the whole effect x fnc subtree); see the module docstring for why
    this split is race-free and how worker records are merged back. n_jobs is
    forwarded to joblib.Parallel (so -1 uses all cores); the default loky
    backend caps each worker's inner BLAS threads to avoid oversubscription.

    With verbose a tqdm bar tracks the sweep over its total leaf count,
    advanced as each fnc call passes (it lurches over cached cells; see the
    module docstring); without it the sweep is silent. The total needs
    len(data), so a verbose sweep pulls kwargs_data_list into a list up front
    -- otherwise it stays lazy.

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

    # the sweep as (kwargs_data, that cell's effect grid) pairs: the whole
    # effect grid per data cell, or -- under skip_recorded -- only the effect
    # cells the records lack, dropping a data cell left with none so its exp is
    # never built.
    total = None
    if skip_recorded:
        # imported here: results pulls in the CONFIG catalogue, which a plain
        # drive never needs. RECORDER.load first, so the walk sees what other
        # writers (an AWS run, a parallel sweep) left on disk.
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
