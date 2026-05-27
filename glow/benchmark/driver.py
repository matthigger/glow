"""Drive a TrialCache: iterate uncached trials, run them, save results.

The driver does no analysis bookkeeping — it just plumbs trials from
``TrialCache.iter_trial_no_repeat`` through ``run_fnc`` and back into
``TrialCache.save_result``.  ``run_fnc`` must accept ``**trial`` and
return a dict (one row) or a DataFrame (one row per emitted record).
"""
from joblib import Parallel, delayed


def driver_local(trial_cache, run_fnc, n_jobs=1, verbose=True):
    """Run every uncached trial in ``trial_cache``.

    Args:
        trial_cache (TrialCache): trial spec + cache.
        run_fnc: callable accepting ``**trial`` and returning a dict
            or DataFrame to be persisted by ``trial_cache.save_result``.
        n_jobs (int): worker count.  ``0`` or ``1`` runs serially; any
            other value spawns a joblib pool.  Parallel results are
            saved as they arrive (csv writes stay on the main thread).
        verbose (bool): print a one-line status as each trial saves.
    """
    if n_jobs in (0, 1):
        for trial in trial_cache.iter_trial_no_repeat():
            result = run_fnc(**trial)
            trial_cache.save_result(result, trial)
            if verbose:
                print(f'saved trial: {trial}')
        return

    trials = list(trial_cache.iter_trial_no_repeat())
    if not trials:
        return

    # joblib's return_as='generator_unordered' streams results back as
    # workers finish, so the csv stays current and Ctrl-C loses at most
    # the in-flight trials.
    pairs = Parallel(n_jobs=n_jobs, return_as='generator_unordered')(
        delayed(_run_one)(run_fnc, trial) for trial in trials)
    for trial, result in pairs:
        trial_cache.save_result(result, trial)
        if verbose:
            print(f'saved trial: {trial}')


def _run_one(run_fnc, trial):
    """Worker entry: returns (trial, result) so the main thread can save."""
    return trial, run_fnc(**trial)
