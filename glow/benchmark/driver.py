from joblib import Parallel, delayed
from tqdm import tqdm


def driver_local(trial_cache, run_fnc, n_jobs=1, verbose=True):
    """Run every uncached trial in ``trial_cache``, save results.

    Args:
        trial_cache (TrialCache): trial spec + cache.
        run_fnc: callable accepting ``**trial`` and returning a dict
            or DataFrame to be persisted by ``trial_cache.save_result``.
        n_jobs (int): worker count.  ``0`` or ``1`` runs serially; any
            other value spawns a joblib pool whose results stream back
            in submission order (csv writes stay on the main thread).
        verbose (bool): show a tqdm progress bar.
    """
    trials = list(trial_cache.iter_trial_no_repeat())
    if not trials:
        return

    n_total = len(trial_cache)
    bar = tqdm(total=n_total, initial=n_total - len(trials),
               disable=not verbose, desc='trials')

    if n_jobs in (0, 1):
        results = (run_fnc(**trial) for trial in trials)
    else:
        results = Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(run_fnc)(**trial) for trial in trials)

    for trial, result in zip(trials, results):
        trial_cache.save_result(result, trial)
        bar.update(1)
    bar.close()
