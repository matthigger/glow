"""Drive the benchmark: data_factory -> effect_factory -> run_ana, as a grid.

``drive`` sweeps the cartesian product of three kwargs grids -- one per stage
of the benchmark pipeline -- as three nested loops:

    for kw_data   in kwargs_data_list:        exp        = data_factory(...)
      for kw_eff  in kwargs_effect_list:      exp, mask  = effect_factory(exp, ...)
        for kw_ana in kwargs_run_ana_list:    score      = run_ana(exp, ...)

The nesting (rather than one flat product) is deliberate: each stage's output
feeds the stage below, so a clean ``exp`` is built once per data cell and a
planted ``exp`` once per (data, effect) cell, then shared across the runs below
it -- no redundant rebuild even on a cold cache. The three stage functions are
each disk-memoised + recorded (see glow._extra.benchmark.data / run), so a
repeated cell is a cache hit and the driver never worries about issuing the same
call twice: a re-run, an overlapping grid, or a resumed sweep all reuse the
stored artifacts.

The driver returns the innermost scores, but the richer output is the shared
provenance DAG every call writes to: ``RECORDER.flatten_to_df`` yields one row
per run_ana leaf, each carrying the swept data / effect inputs that produced it
(the data -> plant -> score chain), so a sweep is analysed from the records
without the driver tracking anything itself (see glow._extra.benchmark.recorder).

Parallelism is deferred: the loops are serial for now; a joblib.Parallel variant
over the flattened cells will be wired in once this is solid (the memoised stages
already make the per-cell work independent and idempotent).
"""

from .data import data_factory, effect_factory
from .run import run_ana


def drive(kwargs_data_list, kwargs_effect_list, kwargs_run_ana_list):
    """Sweep data x effect x analysis as three nested loops; return the scores.

    Each list is a grid of keyword arguments for one pipeline stage; the
    driver runs their cartesian product, threading each stage's output into
    the next (the clean ``exp`` into effect_factory, the planted ``exp`` and
    its realized support ``mask`` into run_ana as the single-element
    ``mask_target_list``). All three stages are memoised + recorded, so this
    only forwards kwargs -- caching dedupes repeated cells and the recorder
    captures provenance (see module docstring).

    Args:
        kwargs_data_list (list[dict]): one kwargs dict per data_factory call,
            e.g. ``{'source': 'wgn', 'shape': (5, 5, 5), 'seed': 0}``.
        kwargs_effect_list (list[dict]): one kwargs dict per effect_factory
            call (``exp`` is supplied by the driver), e.g.
            ``{'effect_llr': 0.05, 'extenter': ExtenterSphere(n_vox=20)}``.
        kwargs_run_ana_list (list[dict]): one kwargs dict per run_ana call
            (``exp`` and ``mask_target_list`` are supplied by the driver),
            e.g. ``{'ana': AnalysisGLOW(n_perm_fwer=250)}``.

    Returns:
        list[dict]: the run_ana score dicts, one per (data, effect, analysis)
            cell in nested-loop order (data outermost, analysis innermost).
            See glow._extra.benchmark.score.score_effects for the schema; the
            per-cell provenance is on the shared recorder, not here.
    """
    score_list = []
    for kwargs_data in kwargs_data_list:
        exp = data_factory(**kwargs_data)
        for kwargs_effect in kwargs_effect_list:
            exp_eff, mask = effect_factory(exp, **kwargs_effect)
            for kwargs_run_ana in kwargs_run_ana_list:
                score = run_ana(exp_eff, mask_target_list=[mask],
                                **kwargs_run_ana)
                score_list.append(score)
    return score_list
