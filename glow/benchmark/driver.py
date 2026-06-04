"""driver_local: in-process runner for a TrialCache.

Pulls the uncached trials from a TrialCache, runs each through run_fnc
(serially or across a joblib pool), and saves every result back through
trial_cache.save_result. Mirror of glow.aws.driver.driver_aws, which
runs the same trials on AWS Batch.
"""

import traceback
from typing import Callable

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .trial_cache import ERROR_LABEL


def _run_one(run_fnc: Callable, trial: dict):
    """Run one trial, never raising: a failure becomes a recorded ERROR row.

    A single bad trial -- effect-imposition non-convergence, an analysis
    edge case, a degenerate draw -- must not abort the whole cache, least
    of all partway through a long run. On failure the trial is logged with
    its identity and a one-row ERROR DataFrame is returned, so save_result
    persists it (carrying the merged trial axes): the failure is auditable
    in results.csv, the trial is marked done rather than retried forever,
    and the driver moves on. Plotting drops ERROR rows from the curves.

    Args:
        run_fnc (Callable): the trial function; accepts **trial.
        trial (dict): the trial kwargs.

    Returns:
        run_fnc's result, or a one-row ERROR DataFrame if it raised.
    """
    try:
        return run_fnc(**trial)
    except Exception:
        tb = traceback.format_exc()
        ident = {k: trial[k] for k in
                 ('source', 'b', 'num_img', 'n_vox_eff', 'seed')
                 if k in trial}
        print(f'  [ERROR] trial failed {ident}:\n{tb}')
        return pd.DataFrame([{'label': ERROR_LABEL, 'error': tb}])


def driver_local(trial_cache, run_fnc: Callable, n_jobs: int = 1,
                 verbose: bool = True) -> None:
    """Run every uncached trial in trial_cache, save results.

    A trial that raises is recorded as an ERROR row and skipped (see
    _run_one) rather than aborting the cache.

    Args:
        trial_cache (TrialCache): trial spec + cache.
        run_fnc (Callable): accepts **trial and returns a dict or
            DataFrame to be persisted by trial_cache.save_result.
        n_jobs (int): worker count. 0 or 1 runs serially; any other
            value spawns a joblib pool whose results stream back in
            submission order (csv writes stay on the main thread).
        verbose (bool): show a tqdm progress bar.
    """
    trials = list(trial_cache.iter_trial_no_repeat())
    if not trials:
        return

    bar = tqdm(total=len(trials), disable=not verbose, desc='trials')

    if n_jobs in (0, 1):
        results = (_run_one(run_fnc, trial) for trial in trials)
    else:
        results = Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(_run_one)(run_fnc, trial) for trial in trials)

    for trial, result in zip(trials, results):
        trial_cache.save_result(result, trial)
        bar.update(1)
    bar.close()
