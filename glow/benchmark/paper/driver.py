"""Local driver: run a TrialCache's trials, capturing recorder provenance.

The TrialCache owns the recorder and its per-trial record files (one json per
trial under records/). A recorder-wired trial fn takes the recorder as its first
argument and wraps its calls with it; driver_paper detects that by signature and
passes the recorder. (Trial fns not yet converted to take a recorder produce no
records -- their contract is broken until they are converted.)

Serial runs drive cache.iter_trial(record=True, flush=True): the cache scopes
each trial (trial_id = the cache hash), the fn records under it, and the cache
flushes the trial's records to records/<hash>.json. Parallel runs can't share
one generator scope across processes, so each worker scopes the trial on its own
recorder and writes its own per-trial file -- no aggregation, no contention.
"""
import functools
import inspect

from glow.util import value_id
from joblib import Parallel, delayed
from tqdm import tqdm

from ..recorder import Recorder


def _call(run_fnc, recorder, trial: dict):
    """Call run_fnc for one trial, passing the recorder iff it accepts one.

    A recorder-wired fn (run_ana) takes recorder as its first parameter --
    detected on the underlying fn, unwrapping a functools.partial.
    """
    fn = run_fnc.func if isinstance(run_fnc, functools.partial) else run_fnc
    if 'recorder' in inspect.signature(fn).parameters:
        return run_fnc(recorder, **trial)
    return run_fnc(**trial)


def _run_one(run_fnc, trial: dict, trial_hash: str, records_dir: str) -> None:
    """Run one trial in a worker on its own recorder; write its own json file.

    Scopes the trial (trial_id = the cache hash) on a worker-local recorder
    rooted at records_dir, so a recorder-wired fn records under it, then flushes
    those records to records_dir/<hash>.json in-worker (only compact recipe
    dicts touch disk; per-trial files never contend). A recorder-wired fn
    swallows its own failures; any other exception is swallowed here (the trial
    simply produces no records).
    """
    recorder = Recorder(folder=records_dir)
    try:
        with recorder.trial(trial_id=trial_hash):
            _call(run_fnc, recorder, trial)
    except Exception:
        pass
    recorder.flush(**{k: value_id(v) for k, v in trial.items()})


def driver_paper(trial_cache, run_fnc, n_jobs: int = 1,
                 verbose: bool = True) -> None:
    """Run every not-yet-completed trial, capturing records to records/.

    Args:
        trial_cache (TrialCache): trial spec + cache (owns the recorder and the
            per-trial record files).
        run_fnc (Callable): a recorder-wired fn (takes the recorder first).
        n_jobs (int): worker count. 0 or 1 runs serially via iter_trial; any
            other value spawns a joblib pool whose workers each write their own
            per-trial file.
        verbose (bool): show a tqdm progress bar.
    """
    if n_jobs in (0, 1):
        for trial in trial_cache.iter_trial(record=True, flush=True,
                                            verbose=verbose):
            try:
                _call(run_fnc, trial_cache.recorder, trial)
            except Exception:
                # recorder-wired fns swallow their own failures; anything else
                # is swallowed too (the trial just produces no records)
                pass
        return

    trials = list(trial_cache.iter_trial())  # uncompleted, no scope/record
    if not trials:
        return
    records_dir = str(trial_cache.recorder.folder)
    args = [(run_fnc, t, trial_cache.hash(t), records_dir) for t in trials]

    bar = tqdm(total=len(trials), disable=not verbose, desc='trials')
    for _ in Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(_run_one)(*a) for a in args):
        bar.update(1)
    bar.close()
