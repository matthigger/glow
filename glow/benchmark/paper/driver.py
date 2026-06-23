"""Local driver capturing recorder provenance to records.json.

Mirror of glow.benchmark.driver.driver_local, but the recorder-wired trial fns
populate the shared recorder (glow.benchmark.paper.run.recorder) instead of
returning a results DataFrame. driver_paper clears the recorder per trial, runs
the trial inside a recorder.trial scope keyed by the cache hash -- so every
record the trial fn captures (its setup and each method) shares that hash as
its trial_id and stays associated -- stamps the scalar trial axes onto each,
and appends to records.json. The cache hash is the trial's identity, so records
from independent joblib workers merge without collision.

Transitional: trial fns not yet converted to the recorder still return a
DataFrame, which is saved through trial_cache.save_result (results.csv) as
before -- so every cache keeps running during the migration. A trial is
considered done if its hash is in records.json or results.csv.
"""
import json
import traceback
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from glow.util import value_id
from ..trial_cache import ERROR_LABEL
from .run import recorder

RECORDS_NAME = 'records.json'


def _run_one(run_fnc, trial: dict, trial_hash: str):
    """Run one trial; return (records, legacy_result).

    Clears the shared recorder, runs run_fnc inside a trial scope keyed by the
    cache hash (so each captured record carries that hash as its trial_id), and
    serialises whatever it captured to JSON-friendly dicts in-worker (so only
    compact records cross a joblib boundary, never the heavy Experiment /
    Analysis objects). The scalar trial axes are stamped onto each record. A
    recorder-wired fn returns None and leaves its work in the records; a
    not-yet-converted fn returns a DataFrame, handed back untouched for the
    legacy csv path.

    Args:
        run_fnc (Callable): the trial fn; accepts **trial.
        trial (dict): the trial kwargs.
        trial_hash (str): the cache hash; the trial scope's trial_id.

    Returns:
        records (list): the trial's serialised, stamped records (possibly empty).
        result: run_fnc's DataFrame for a legacy fn, else None.
    """
    recorder.records.clear()
    try:
        # one trial scope per run_fnc call, keyed by the cache hash, so every
        # record the trial fn captures (its setup and each method) shares one
        # trial_id and stays associated across workers
        with recorder.trial(trial_id=trial_hash):
            result = run_fnc(**trial)
    except Exception:
        # recorder-wired fns swallow their own failures; reaching here means a
        # legacy fn raised (or a bug before any recorded call) -> a legacy
        # ERROR row, mirroring driver_local._run_one.
        result = pd.DataFrame([{'label': ERROR_LABEL,
                                'error': traceback.format_exc()}])
    records = json.loads(recorder.to_json()) if recorder.records else []
    axes = {k: value_id(v) for k, v in trial.items()}
    for r in records:
        r.update(axes)
    return records, result


def driver_paper(trial_cache, run_fnc, n_jobs: int = 1,
                 verbose: bool = True) -> None:
    """Run every uncached trial, capturing recorder records to records.json.

    Args:
        trial_cache (TrialCache): trial spec + cache.
        run_fnc (Callable): accepts **trial; populates the shared recorder and
            returns None, or (not yet converted) returns a DataFrame.
        n_jobs (int): worker count. 0 or 1 runs serially; any other value
            spawns a joblib pool (records.json / csv writes stay on the main
            thread).
        verbose (bool): show a tqdm progress bar.
    """
    path = Path(trial_cache.folder) / RECORDS_NAME
    all_recs = json.loads(path.read_text()) if path.exists() else []
    # records carry the cache hash as their trial_id (the driver's trial scope);
    # a trial is done if seen there or in the legacy results.csv
    done = {r['trial_id'] for r in all_recs} | trial_cache._cached_hashes()

    trials = [t for t in trial_cache.iter_trial()
              if trial_cache.hash(t) not in done]
    if not trials:
        return
    hashes = [trial_cache.hash(t) for t in trials]

    bar = tqdm(total=len(trials), disable=not verbose, desc='trials')

    if n_jobs in (0, 1):
        results = (_run_one(run_fnc, t, h) for t, h in zip(trials, hashes))
    else:
        results = Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(_run_one)(run_fnc, t, h) for t, h in zip(trials, hashes))

    for (records, result), trial in zip(results, trials):
        if records:
            all_recs.extend(records)
            path.write_text(json.dumps(all_recs, indent=2))
        elif isinstance(result, pd.DataFrame):
            trial_cache.save_result(result, trial)
        bar.update(1)
    bar.close()
