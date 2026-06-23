"""Local driver: run a TrialCache's trials, capturing recorder provenance.

The TrialCache owns the recorder and the record files (one json per trial under
its records/ dir). A recorder-wired trial fn takes the recorder as its first
argument and wraps its calls with it; a not-yet-converted fn returns a DataFrame
saved through the legacy results.csv path. driver_paper detects which by the
fn's signature.

Serial runs drive cache.iter_trial(record=True, flush=True): the cache scopes
each trial (trial_id = the cache hash), the fn records under it, and the cache
flushes the trial's records to records/<hash>.json. Parallel runs can't share
one generator scope across processes, so each worker scopes the trial on its own
recorder and writes its own per-trial file -- no aggregation, no contention.
"""
import functools
import inspect
import traceback
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from glow.util import value_id
from ..recorder import Recorder
from ..trial_cache import ERROR_LABEL


def _call(run_fnc, recorder, trial: dict):
    """Call run_fnc for one trial, passing the recorder iff it accepts one.

    A recorder-wired fn (run_ana) takes recorder as its first parameter; a
    legacy fn does not -- detected on the underlying fn (unwrapping a partial).
    """
    fn = run_fnc.func if isinstance(run_fnc, functools.partial) else run_fnc
    if 'recorder' in inspect.signature(fn).parameters:
        return run_fnc(recorder, **trial)
    return run_fnc(**trial)


def _run_one(run_fnc, trial: dict, trial_hash: str, records_dir: str):
    """Run one trial in a worker on its own recorder; write its own json file.

    Scopes the trial (trial_id = the cache hash) on a worker-local recorder, so
    a recorder-wired fn records under it, then flushes those records to
    records_dir/<hash>.json in-worker (only compact recipe dicts touch disk,
    never the heavy objects; per-trial files never contend). Returns any legacy
    DataFrame for the main process to persist to csv; a legacy fn that raises
    becomes an ERROR row.
    """
    recorder = Recorder()
    try:
        with recorder.trial(trial_id=trial_hash):
            result = _call(run_fnc, recorder, trial)
    except Exception:
        result = pd.DataFrame([{'label': ERROR_LABEL,
                                'error': traceback.format_exc()}])
    if recorder.records:
        Path(records_dir).mkdir(parents=True, exist_ok=True)
        axes = {k: value_id(v) for k, v in trial.items()}
        recorder.flush(Path(records_dir) / f'{trial_hash}.json', **axes)
    return result


def driver_paper(trial_cache, run_fnc, n_jobs: int = 1,
                 verbose: bool = True) -> None:
    """Run every not-yet-completed trial, capturing records to records/.

    Args:
        trial_cache (TrialCache): trial spec + cache (owns the recorder and the
            per-trial record files).
        run_fnc (Callable): a recorder-wired fn (takes the recorder first) or a
            legacy fn returning a DataFrame.
        n_jobs (int): worker count. 0 or 1 runs serially via iter_trial; any
            other value spawns a joblib pool whose workers each write their own
            per-trial file (legacy csv writes stay on the main thread).
        verbose (bool): show a tqdm progress bar.
    """
    if n_jobs in (0, 1):
        for trial in trial_cache.iter_trial(record=True, flush=True,
                                            verbose=verbose):
            try:
                result = _call(run_fnc, trial_cache.recorder, trial)
            except Exception:
                # recorder-wired fns swallow their own failures; a legacy fn may
                # raise -> ERROR row (the cache still flushes any records)
                result = pd.DataFrame([{'label': ERROR_LABEL,
                                        'error': traceback.format_exc()}])
            if isinstance(result, pd.DataFrame):
                trial_cache.save_result(result, trial)
        return

    trials = list(trial_cache.iter_trial())  # uncompleted, no scope/record
    if not trials:
        return
    records_dir = str(trial_cache._records_dir)
    args = [(run_fnc, t, trial_cache.hash(t), records_dir) for t in trials]

    results = Parallel(n_jobs=n_jobs, return_as='generator')(
        delayed(_run_one)(*a) for a in args)

    bar = tqdm(total=len(trials), disable=not verbose, desc='trials')
    for trial, result in zip(trials, results):
        if isinstance(result, pd.DataFrame):
            trial_cache.save_result(result, trial)
        bar.update(1)
    bar.close()
