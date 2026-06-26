"""Assemble per-trial records into the tidy results frame the plots consume.

The recorder writes one json per trial under a cache's records/ dir, each a
list of step records (one per decorated call: function, optional method label,
inputs, outputs, the trial's scalar axes, and time_sec; see
glow._extra.benchmark.recorder). The plots want the opposite shape -- one tidy row per
(trial, method), the trial's scalar axes beside the method's detection score
flattened to columns -- the long-format results.csv the old driver wrote
directly. records_to_df is that pivot.

It keeps the score steps (a step whose outputs hold a 'score' dict: run_ana's
score_effects, run_segment's oracle counts), pairs each with the axes stamped on
the record (source / seed / effect_llr / b / num_img / n_vox_eff), and flattens
the score's confusion block(s) to tp/fp/tn/fn (suffixed tp0/.. for several
planted effects) plus num_vox / min_pval / n_pred. add_metric_cols then derives
dice / sens / ppv / spec from the counts, exactly as load_results_csv does for a
stored csv -- so a records-backed frame and a csv-backed one carry the same
columns. Steps with no score (a fit, the experiment setup) and the heavy per-
region 'pred' list are dropped; a trial that failed simply contributes no score
row.
"""
import pandas as pd

from glow._extra.benchmark.file import add_metric_cols


# scalar trial axes stamped on every record (recorder.flush); copied through to
# the tidy row so each row is self-describing by its axis columns. Only those
# present on a given record are emitted (caches sweep different subsets).
_AXES = ('source', 'seed', 'b', 'num_img', 'n_vox_eff',
         'effect_llr', 'effect_total_llr', 'angle')

_COUNTS = ('tp', 'fp', 'tn', 'fn')


def _flatten_score(score: dict) -> dict:
    """Flatten one score dict to scalar columns (confusion counts + scalars).

    Handles both score shapes the effect-discovery caches record:
      - score_effects (run_ana): scalar num_vox / min_pval / n_pred (and
        n_selected for the prune arms) plus a
        'target' confusion block (tp/fp/tn/fn vs the union of planted effects)
        and, with several effects, target0 / target1 / ... blocks scored
        against each effect in turn -> suffixed tp0/fp0/.. columns.
      - the oracle counts (run_segment): a flat {tp, fp, tn, fn} dict.
    vox_effect (tp + fn, the planted support inside the analysis) and vox_total
    (num_vox) are derived too, so the extent sweep's effect_perc is available.
    The heavy per-region 'pred' list is dropped.

    Args:
        score (dict): one score step's 'score' output.

    Returns:
        a flat dict of scalar columns for this (trial, method) row.
    """
    out = {}
    for k in ('num_vox', 'min_pval', 'n_pred', 'n_selected'):
        if k in score:
            out[k] = score[k]

    found_block = False
    for key, val in score.items():
        if not isinstance(val, dict):
            continue
        if key == 'target':
            out.update(val)
            found_block = True
        elif key.startswith('target'):
            # 'target0' -> tp0/fp0/.. (one block per planted effect)
            i = key[len('target'):]
            out.update({f'{k}{i}': v for k, v in val.items()})
            found_block = True

    # flat oracle counts (run_segment): tp/fp/tn/fn already at the top level
    if not found_block and set(_COUNTS).issubset(score):
        out.update({k: score[k] for k in _COUNTS})

    if set(_COUNTS).issubset(out):
        out['vox_effect'] = out['tp'] + out['fn']
        out['vox_total'] = out.get('num_vox', sum(out[k] for k in _COUNTS))
    return out


def records_to_df(records: list):
    """Pivot per-trial records into one tidy row per (trial, method).

    One row per recorded score step: its trial id (as trial_hash, matching the
    cache hash the plots filter on), its method label, the trial's scalar axes,
    and the flattened score (see _flatten_score). dice / sens / ppv / spec are
    derived from the tp/fp/tn/fn counts via add_metric_cols.

    Args:
        records (list): the recorder's step records (e.g. cache.recorder.load()
            or TrialCache.load_records()).

    Returns:
        the tidy results DataFrame (empty when no record carries a score).
    """
    rows = []
    for r in records:
        score = (r.get('outputs') or {}).get('score')
        if not isinstance(score, dict):
            continue
        row = {'trial_hash': r['trial_id'], 'label': r.get('label')}
        row.update({ax: r[ax] for ax in _AXES if ax in r})
        row.update(_flatten_score(score))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return add_metric_cols(df)


def load_config_df(cache):
    """Load one cache's records as tidy results, keeping the in-config trials.

    Reads the cache's per-trial record files (cache.recorder.load), pivots them
    to one tidy row per (trial, method) with derived dice/sens/ppv/spec
    (records_to_df), then drops rows whose trial_hash is not one
    cache.iter_trial() would produce -- i.e. trials left over from a different
    config. "Complete" here is per trial, not per config: a trial writes all its
    method rows under one trial_hash (its record file), so a present hash means
    that trial is done. The result is therefore however many in-config trials
    have finished so far -- a half-run cache yields a partial frame, enough to
    plot or to drive the redo / compare REPLs; it is empty only when no completed
    trial on disk belongs to the current config.

    The canonical "current json outputs" loader: plot, compare and redo_view all
    read a cache's results through it, so they agree on which trials count and
    on the derived metric columns.

    Args:
        cache (TrialCache): the config-catalogue cache whose iter_trial()
            defines the in-config trial set and whose recorder reads the records.

    Returns:
        the completed in-config results so far, one tidy row per (trial, method)
            with a trial_hash column (empty only when no completed trial on disk
            belongs to the current config).
    """
    df = records_to_df(cache.recorder.load())
    if df.empty or 'trial_hash' not in df.columns:
        return pd.DataFrame()
    expected = {cache.hash(trial)
                for trial in cache.iter_trial(include_completed=True)}
    return df[df['trial_hash'].astype(str).isin(expected)]


def find_caches(labels=None) -> list:
    """List (label, cache) for catalogue caches that have records on disk.

    Walks config.CACHE_BY_LABEL and keeps the caches whose recorder has at least
    one per-trial record file (so the result is the set of configs a reader can
    actually load), skipping the rest. Replaces the old results.csv folder scan:
    discovery now keys off the records/ dir the driver writes.

    Args:
        labels (list | None): restrict to these cache labels; None lists every
            catalogue cache with records.

    Returns:
        sorted list of (label, TrialCache) pairs, one per cache with records.
    """
    # deferred: config imports this package's run module, so importing it at
    # module load would be circular (config -> run -> ... ; results <- plot).
    from .config import CACHE_BY_LABEL

    out = []
    for label, (cache, _run_fnc) in CACHE_BY_LABEL.items():
        if labels is not None and label not in labels:
            continue
        if cache.recorder.completed_ids():
            out.append((label, cache))
    return sorted(out, key=lambda lc: lc[0])
