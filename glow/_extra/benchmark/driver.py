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

Parallelism is deferred: the loops are serial for now; a joblib.Parallel variant
over the flattened cells will be wired in once this is solid (the memoised stages
already make the per-cell work independent and idempotent).
"""

from .data import data_factory, effect_factory


def drive(kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc):
    """Sweep data x effect, run ``fnc`` on each cell over its kwargs; return scores.

    The two upstream lists are kwargs grids for the data and effect stages; the
    driver runs their cartesian product, threading each stage's output into the
    next (the clean ``exp`` into effect_factory, the planted ``exp`` and its
    realized support ``mask`` into ``fnc`` as the single-element
    ``mask_target_list``). Each planted cell is then measured by ``fnc`` once per
    kwargs dict in ``kwargs_fnc_list``. All stages and ``fnc`` are memoised +
    recorded, so this only forwards kwargs -- caching dedupes repeated cells and
    the recorder captures provenance (see module docstring).

    ``kwargs_effect_list`` and ``kwargs_fnc_list`` are re-iterated (once per data
    / per data x effect cell), so they must be re-iterable sequences (lists /
    tuples, as the config layer passes -- not one-shot generators);
    ``kwargs_data_list`` is iterated once, so any iterable works.

    Args:
        kwargs_data_list (iterable[dict]): one kwargs dict per data_factory call,
            e.g. ``{'source': 'wgn', 'shape': (5, 5, 5), 'seed': 0}``.
        kwargs_effect_list (Sequence[dict | None]): one kwargs dict per
            effect_factory call (``exp`` is supplied by the driver), e.g.
            ``{'effect_llr': 0.05, 'extenter_cls': ExtenterMinVar, 'n_vox': 20,
            'seed_from_exp': True}``; a ``None`` cell plants no effect (the null /
            FWER-calibration path, run on the clean exp with an empty target).
        kwargs_fnc_list (Sequence[dict]): one kwargs dict per ``fnc`` call on each
            planted cell (``exp`` and ``mask_target_list`` are supplied by the
            driver), e.g. ``[{'ana': AnalysisGLOW(n_perm_fwer=250)}]``.
        fnc (Callable): the leaf measurement, called
            ``fnc(exp, mask_target_list=..., **kwargs)`` and memoised + recorded
            with ``exp`` linked (see module docstring), e.g. run_ana.

    Returns:
        list[dict]: the ``fnc`` score dicts, one per (data, effect, fnc-kwargs)
            cell in nested-loop order (data outermost, fnc kwargs innermost). See
            glow._extra.benchmark.score.score_effects for the schema; the per-cell
            provenance is on the shared recorder, not here.
    """
    score_list = []
    for kwargs_data in kwargs_data_list:
        exp = data_factory(**kwargs_data)
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
