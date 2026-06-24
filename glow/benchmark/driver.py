"""Local driver: run a TrialCache's trials, capturing recorder provenance.

The TrialCache owns the recorder and its per-trial record files (one json per
trial under records/). A recorder-wired trial fn takes the recorder as its first
argument and wraps its calls with it; driver_local detects that by signature and
passes the recorder. (Trial fns not yet converted to take a recorder --
run_segment, run_min_size -- produce no records: their contract is broken until
they are converted.)

Serial runs drive cache.iter_trial(record=True, flush=True): the cache scopes
each trial (trial_id = the cache hash), the fn records under it, and the cache
flushes the trial's records to records/<hash>.json. Parallel runs can't share
one generator scope across processes, so each worker scopes the trial on its own
recorder and writes its own per-trial file -- no aggregation, no contention.
"""
import inspect

from glow.util import value_id
from joblib import Parallel, delayed
from tqdm import tqdm

from .recorder import Recorder


def _run_one(run_fnc, pass_recorder: bool, trial: dict, trial_hash: str,
             folder: str) -> None:
    """Run one trial in a worker on its own recorder; write its own json file.

    Scopes the trial (trial_id = the cache hash) on a worker-local recorder
    rooted at the same experiment folder, so a recorder-wired fn records under
    it, then flushes those records to its records/<hash>.json in-worker (only
    compact recipe dicts touch disk; per-trial files never contend). A
    recorder-wired fn swallows its own failures; any other exception is
    swallowed here (the trial simply produces no records).

    Args:
        run_fnc (Callable): the trial fn (bound to its analysis recipe).
        pass_recorder (bool): pass the recorder as run_fnc's first argument
            (a recorder-wired fn) vs call it with the trial kwargs alone.
        trial (dict): the trial's scalar axes.
        trial_hash (str): the cache hash scoping this trial's records.
        folder (str): the experiment folder the worker's recorder roots at.
    """
    recorder = Recorder(folder=folder)
    try:
        with recorder.trial(trial_id=trial_hash):
            if pass_recorder:
                run_fnc(recorder, **trial)
            else:
                run_fnc(**trial)
    except Exception:
        pass
    recorder.flush(**{k: value_id(v) for k, v in trial.items()})


def driver_local(trial_cache, run_fnc, n_jobs: int = 1,
                 verbose: bool = True) -> None:
    """Run every not-yet-completed trial, capturing records to records/.

    Args:
        trial_cache (TrialCache): trial spec + cache (owns the recorder and the
            per-trial record files).
        run_fnc (Callable): the trial fn; recorder-wired ones take the recorder
            first (detected by signature) and the driver passes it.
        n_jobs (int): worker count. 0 or 1 runs serially via iter_trial; any
            other value spawns a joblib pool whose workers each write their own
            per-trial file.
        verbose (bool): show a tqdm progress bar.
    """
    # run_fnc is fixed for the whole cache, so detect once. inspect.signature
    # sees a recorder parameter through a functools.partial (the bound analysis
    # kwargs drop out), so there is no need to unwrap to .func.
    pass_recorder = 'recorder' in inspect.signature(run_fnc).parameters

    if n_jobs in (0, 1):
        for trial in trial_cache.iter_trial(record=True, flush=True,
                                            verbose=verbose):
            try:
                if pass_recorder:
                    run_fnc(trial_cache.recorder, **trial)
                else:
                    run_fnc(**trial)
            except Exception:
                # recorder-wired fns swallow their own failures; anything else
                # is swallowed too (the trial just produces no records)
                pass
        return

    trials = list(trial_cache.iter_trial())  # uncompleted, no scope/record
    if not trials:
        return
    folder = str(trial_cache.recorder.folder)
    args = [(run_fnc, pass_recorder, t, trial_cache.hash(t), folder)
            for t in trials]

    bar = tqdm(total=len(trials), disable=not verbose, desc='trials')
    for _ in Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(_run_one)(*a) for a in args):
        bar.update(1)
    bar.close()
