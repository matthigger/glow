"""Local driver capturing recorder provenance to records.json.

Mirror of glow.benchmark.driver.driver_local, but the recorder-wired trial fns
populate the shared recorder (glow.benchmark.recorder.recorder) instead of
returning a results DataFrame. The TrialCache owns the recorder and the
records.json IO: its iter_record scopes each trial (trial_id = the cache hash)
so every call the trial fn records -- its setup and each method -- shares that
hash and stays associated, then appends the serialised records to records.json.

Serial runs drive cache.iter_record directly. Parallel runs can't share one
generator scope across processes, so each worker opens the trial scope itself
(on its own imported recorder) and returns the serialised records, which the
main process appends through the cache.

Transitional: trial fns not yet converted to the recorder still return a
DataFrame, saved through trial_cache.save_result (results.csv) as before -- so
every cache keeps running during the migration.
"""
import traceback

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from glow.benchmark.recorder import recorder
from ..trial_cache import ERROR_LABEL, TrialCache


def _save_legacy(trial_cache, result, trial: dict) -> None:
    """Persist a not-yet-converted fn's DataFrame return to results.csv."""
    if isinstance(result, pd.DataFrame):
        trial_cache.save_result(result, trial)


def _run_one(run_fnc, trial: dict, trial_hash: str):
    """Run one trial in a worker; return (serialised_records, legacy_result).

    Opens the trial scope on the worker's imported recorder (trial_id = the
    cache hash) so the records carry it, serialises them in-worker (only compact
    recipe dicts cross the joblib boundary, never the heavy Experiment /
    Analysis objects), and hands back any legacy DataFrame for the csv path. A
    legacy fn that raises becomes an ERROR row, mirroring driver_local.
    """
    recorder.records.clear()
    try:
        with recorder.trial(trial_id=trial_hash):
            result = run_fnc(**trial)
    except Exception:
        result = pd.DataFrame([{'label': ERROR_LABEL,
                                'error': traceback.format_exc()}])
    return TrialCache.serialize_records(recorder, trial), result


def driver_paper(trial_cache, run_fnc, n_jobs: int = 1,
                 verbose: bool = True) -> None:
    """Run every uncompleted trial, capturing recorder records to records.json.

    Args:
        trial_cache (TrialCache): trial spec + cache (owns the recorder and the
            records.json IO).
        run_fnc (Callable): accepts **trial; populates the shared recorder and
            returns None, or (not yet converted) returns a DataFrame.
        n_jobs (int): worker count. 0 or 1 runs serially via iter_record; any
            other value spawns a joblib pool whose workers each scope their own
            trial (records.json / csv writes stay on the main thread).
        verbose (bool): show a tqdm progress bar.
    """
    if n_jobs in (0, 1):
        for trial in trial_cache.iter_record(verbose=verbose):
            try:
                result = run_fnc(**trial)
            except Exception:
                # recorder-wired fns swallow their own failures; a legacy fn may
                # raise -> ERROR row (iter_record still persists its records)
                result = pd.DataFrame([{'label': ERROR_LABEL,
                                        'error': traceback.format_exc()}])
            _save_legacy(trial_cache, result, trial)
        return

    trials = list(trial_cache.iter_uncompleted())
    if not trials:
        return
    hashes = [trial_cache.hash(t) for t in trials]

    results = Parallel(n_jobs=n_jobs, return_as='generator')(
        delayed(_run_one)(run_fnc, t, h) for t, h in zip(trials, hashes))

    bar = tqdm(total=len(trials), disable=not verbose, desc='trials')
    for trial, (records, result) in zip(trials, results):
        trial_cache.append_records(records)
        _save_legacy(trial_cache, result, trial)
        bar.update(1)
    bar.close()
