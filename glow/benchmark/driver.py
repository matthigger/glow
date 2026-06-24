"""Local driver: run a TrialCache's trials, capturing recorder provenance.

The TrialCache owns the recorder and its per-trial record files (one json per
trial under records/). Every trial fn takes the recorder as its first argument
and wraps its calls with it, so the driver just passes the recorder in.

Serial runs drive cache.iter_trial(record=True, flush=True): the cache scopes
each trial (trial_id = the cache hash), the fn records under it, and the cache
flushes the trial's records to records/<hash>.json. Parallel runs can't share
one generator scope across processes, so each worker scopes the trial on its own
recorder and writes its own per-trial file -- no aggregation, no contention.
"""
from glow.util import value_id
from joblib import Parallel, delayed
from tqdm import tqdm

from .recorder import Recorder


def _run_one(run_fnc, trial: dict, trial_hash: str, folder: str) -> None:
    """Run one trial in a worker on its own recorder; write its own json file.

    Scopes the trial (trial_id = the cache hash) on a worker-local recorder
    rooted at the same experiment folder, so the recorder-wired fn records under
    it, then flushes those records to its records/<hash>.json in-worker (only
    compact recipe dicts touch disk; per-trial files never contend). The fn
    swallows its own failures; any other exception is swallowed here (the trial
    simply produces no records).

    Args:
        run_fnc (Callable): the trial fn (recorder first, bound to its recipe).
        trial (dict): the trial's scalar axes.
        trial_hash (str): the cache hash scoping this trial's records.
        folder (str): the experiment folder the worker's recorder roots at.
    """
    recorder = Recorder(folder=folder)
    try:
        with recorder.trial(trial_id=trial_hash):
            run_fnc(recorder, **trial)
    except Exception:
        pass
    recorder.flush(**{k: value_id(v) for k, v in trial.items()})


def driver_local(trial_cache, run_fnc, n_jobs: int = 1,
                 verbose: bool = True) -> None:
    """Run every not-yet-completed trial, capturing records to records/.

    Args:
        trial_cache (TrialCache): trial spec + cache (owns the recorder and the
            per-trial record files).
        run_fnc (Callable): the trial fn; it takes the recorder as its first
            argument, which the driver passes in.
        n_jobs (int): worker count. 0 or 1 runs serially via iter_trial; any
            other value spawns a joblib pool whose workers each write their own
            per-trial file.
        verbose (bool): show a tqdm progress bar.
    """
    if n_jobs in (0, 1):
        for trial in trial_cache.iter_trial(record=True, flush=True,
                                            verbose=verbose):
            try:
                run_fnc(trial_cache.recorder, **trial)
            except Exception:
                # recorder-wired fns swallow their own failures; anything else
                # is swallowed too (the trial just produces no records)
                pass
        return

    trials = list(trial_cache.iter_trial())  # uncompleted, no scope/record
    if not trials:
        return
    folder = str(trial_cache.recorder.folder)
    args = [(run_fnc, t, trial_cache.hash(t), folder) for t in trials]

    bar = tqdm(total=len(trials), disable=not verbose, desc='trials')
    for _ in Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(_run_one)(*a) for a in args):
        bar.update(1)
    bar.close()
