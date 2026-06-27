"""Drive the benchmark: data_factory -> effect_factory -> fnc, as a grid.

``drive`` sweeps the cartesian product of two upstream kwargs grids -- the data
and effect stages -- and runs a leaf function ``fnc`` on each planted cell, once
per kwargs dict in ``kwargs_fnc_list``, as nested loops:

    for kw_data   in kwargs_data_list:     exp        = data_factory(...)
      for kw_eff  in kwargs_effect_list:   exp, mask  = effect_factory(exp, ...)
        for kw    in kwargs_fnc_list:      score      = fnc(exp, ...)

The nesting (rather than one flat product) is deliberate: each upstream stage's
output feeds the stage below, so a clean ``exp`` is built once per data cell and
a planted ``exp`` once per (data, effect) cell, then shared across every fnc run
below it -- no redundant rebuild even on a cold cache. The stage functions are
each disk-memoised + recorded (see glow._extra.benchmark.data / run), so a
repeated cell is a cache hit and the driver never worries about issuing the same
call twice: a re-run, an overlapping grid, or a resumed sweep all reuse the
stored artifacts.

``fnc`` is the leaf measurement, swept over its own kwargs grid so one (data,
effect) cell can be measured several ways at once (e.g. run_ana under several
Analysis recipes). It is called ``fnc(exp, mask_target_list=..., **kwargs)`` and,
to join the provenance DAG, must be memoised + recorded with ``exp`` as a linked
input -- exactly the shape of run_ana (see glow._extra.benchmark.run). The driver
knows nothing fnc-specific; the config layer supplies it.

A ``None`` effect cell is the null / FWER-calibration path: it plants nothing and
runs ``fnc`` on the clean ``exp`` against an empty target. effect_factory is
skipped entirely (EffectSynthetic has no no-op -- it imposes a real effect_llr),
so a null run is data_factory -> fnc directly, and its provenance row chains
straight to the build with no plant node (see glow._extra.benchmark.config's
'null' cache, whose effect grid is ``[None]``).

The driver returns the innermost scores, but the richer output is the shared
provenance DAG every call writes to: ``RECORDER.flatten_to_df`` yields one row
per fnc leaf, each carrying the swept data / effect inputs that produced it (the
data -> plant -> score chain), so a sweep is analysed from the records without
the driver tracking anything itself (see glow._extra.benchmark.recorder).

Parallelism (``n_jobs != 1``) splits the sweep **by data cell**: each whole
``data_factory -> effect_factory* -> fnc*`` subtree is one joblib task, so the
clean ``exp`` (and each planted ``exp``) has exactly one owner. No two workers
ever compute -- or race to write the cache / record of -- the same build, and the
``exp`` it builds is reused in-process across its subtree, never pickled between
workers. This is the right split while the data grids satiate the pool (the
sweeps run 15-1000 seeds); a narrower grid would leave workers idle and want a
finer (two-phase) split, deferred until needed. The stage functions are shared
module-level singletons re-imported in each worker, so workers transparently
share the on-disk cache and records folder; the per-hash record files they write
are folded back into this process with ``RECORDER.load`` once the pool drains.
(The leaf ``fnc`` is pickled to the workers like any task argument; the recorder
snapshots its records as they are made, so the wrapped ``fnc`` carries no live
Experiments and stays light to ship -- see glow._extra.benchmark.recorder.)
"""

from .data import RECORDER, data_factory, effect_factory


def _run_data_cell(kwargs_data, kwargs_effect_list, kwargs_fnc_list, fnc):
    """Build one data cell and run its effect x fnc subtree; return its scores.

    The per-data-cell unit of work, shared by the serial loop and the parallel
    tasks: build the clean ``exp`` once (then reuse it across the effect /
    fnc loops below), plant each effect (or skip it for a ``None`` cell), and run
    ``fnc`` over its kwargs grid on each planted cell.

    Args:
        kwargs_data (dict): kwargs for one data_factory call.
        kwargs_effect_list (list[dict | None]): the effect grid (already a list).
        kwargs_fnc_list (list[dict]): the fnc kwargs grid (already a list).
        fnc (Callable): the leaf measurement, ``fnc(exp, mask_target_list=...,
            **kwargs)``.

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
            exp_eff, mask = effect_factory(exp, **kwargs_effect)
            mask_target_list = [mask]
        for kwargs in kwargs_fnc_list:
            score = fnc(exp_eff, mask_target_list=mask_target_list, **kwargs)
            score_list.append(score)
    return score_list


def drive(kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc, *,
          n_jobs=1):
    """Sweep data x effect, run ``fnc`` on each cell over its kwargs; return scores.

    The two upstream lists are kwargs grids for the data and effect stages; the
    driver runs their cartesian product, threading each stage's output into the
    next (the clean ``exp`` into effect_factory, the planted ``exp`` and its
    realized support ``mask`` into ``fnc`` as the single-element
    ``mask_target_list``). Each planted cell is then measured by ``fnc`` once per
    kwargs dict in ``kwargs_fnc_list``. All stages and ``fnc`` are memoised +
    recorded, so this only forwards kwargs -- caching dedupes repeated cells and
    the recorder captures provenance (see module docstring).

    ``kwargs_effect_list`` and ``kwargs_fnc_list`` are pulled into lists up front
    (they are re-iterated once per data / per data x effect cell), so one-shot
    generators are fine -- as the config layer passes; ``kwargs_data_list`` is
    iterated once and stays lazy.

    With ``n_jobs != 1`` the sweep runs in parallel over joblib, one task per
    data cell (the whole effect x fnc subtree); see the module docstring for why
    this split is race-free and how worker records are merged back. ``n_jobs``
    is forwarded to joblib.Parallel (so -1 uses all cores); the default loky
    backend caps each worker's inner BLAS threads to avoid oversubscription.

    Args:
        kwargs_data_list (iterable[dict]): one kwargs dict per data_factory call,
            e.g. ``{'source': 'wgn', 'shape': (5, 5, 5), 'seed': 0}``.
        kwargs_effect_list (iterable[dict | None]): one kwargs dict per
            effect_factory call (``exp`` is supplied by the driver), e.g.
            ``{'effect_llr': 0.05, 'extenter': ExtenterSphere(n_vox=20)}``; a
            ``None`` cell plants no effect (the null / FWER-calibration path, run
            on the clean exp with an empty target).
        kwargs_fnc_list (iterable[dict]): one kwargs dict per ``fnc`` call on each
            planted cell (``exp`` and ``mask_target_list`` are supplied by the
            driver), e.g. ``[{'ana': AnalysisGLOW(n_perm_fwer=250)}]``.
        fnc (Callable): the leaf measurement, called
            ``fnc(exp, mask_target_list=..., **kwargs)`` and memoised + recorded
            with ``exp`` linked (see module docstring), e.g. run_ana.
        n_jobs (int): 1 (default) runs serially in-process; otherwise the data
            cells run in parallel over joblib.Parallel(n_jobs=n_jobs).

    Returns:
        list[dict]: the ``fnc`` score dicts, one per (data, effect, fnc-kwargs)
            cell in data-cell order (effect then fnc-kwargs within a cell). See
            glow._extra.benchmark.score.score_effects for the schema; the per-cell
            provenance is on the shared recorder, not here.
    """
    # the effect / fnc grids are re-iterated per data (resp. data x effect) cell,
    # so pull them into lists once -- a one-shot generator (as config passes)
    # would otherwise be spent after the first data cell. data is iterated once,
    # so it stays lazy.
    kwargs_effect_list = list(kwargs_effect_list)
    kwargs_fnc_list = list(kwargs_fnc_list)

    if n_jobs == 1:
        score_list = []
        for kwargs_data in kwargs_data_list:
            score_list.extend(_run_data_cell(
                kwargs_data, kwargs_effect_list, kwargs_fnc_list, fnc))
        return score_list

    # parallel: one task per data cell, so each build has a single owner -- no
    # two workers compute / write the same cell (see module docstring).
    from joblib import Parallel, delayed

    cell_scores = Parallel(n_jobs=n_jobs)(
        delayed(_run_data_cell)(
            kwargs_data, kwargs_effect_list, kwargs_fnc_list, fnc)
        for kwargs_data in kwargs_data_list)

    # workers wrote their per-hash record files to the shared folder in their own
    # processes; fold them into this process's recorder so flatten_to_df sees the
    # whole sweep (a no-op merge when the backend shares this process's memory).
    RECORDER.load()
    return [score for scores in cell_scores for score in scores]
