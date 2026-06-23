"""Trial iteration spec + result IO + cache lookup.

A TrialCache owns a results csv and answers three questions:

  1. What kwarg dicts should run (cartesian product over iter_kwargs,
     merged with constant kwargs)?
  2. Which of those are already on disk (is_cached,
     iter_trial_no_repeat)?
  3. How do I persist a result (save_result)?

The three answers all key off _hash(trial), which is stable_hash(trial)
optionally redirected through trial_alias_map. The alias map lets a second
cache whose trials differ only by a swapped-in stand-in (the AWS driver
ships every real-data DataSource as a DataSourceS3) share the original
cache's results rows: see glow.aws.driver._to_s3_cache.
"""

from itertools import product
from math import prod
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from tqdm import tqdm

from glow.util import stable_hash, value_id
from .file import get_path_result
from .recorder import Recorder


CSV_NAME = 'results.csv'
RECORDS_NAME = 'records.json'
RECORDS_DIR = 'records'
HASH_COL = 'trial_hash'

# Sentinel `label` values for trials that produced no scored result. They
# are still recorded (one row, carrying the merged trial axes) so the trial
# is auditable and marked done rather than silently missing or retried
# forever; plotting excludes them from the curves. ERROR: run_fnc raised
# (see driver._run_one). SKIP: an infeasible cell, e.g. an HCP feature
# count beyond the pool (see paper.run).
ERROR_LABEL = 'ERROR'
SKIP_LABEL = 'SKIP'
NON_RESULT_LABELS = (ERROR_LABEL, SKIP_LABEL)


class TrialCache:
    """Trial iteration spec + result IO + cache lookup.

    Construct with exactly one of name (resolves to
    <default_results_dir>/<name>/) or folder (used directly).

    Attributes:
        folder (Path): on-disk directory where results live. Created if
            missing.
        df (pd.DataFrame): all saved results so far. Loaded from
            results.csv at construction; save_result keeps it in sync as
            trials complete.
        iter_kwargs (dict | None): values are iterables; their cartesian
            product yields one kwarg dict per trial.
        kwargs (dict | None): constant kwargs merged into every yielded
            dict.
        trial_alias_map (dict | None): {raw_hash: alias_hash} redirecting a
            trial's stable_hash for all on-disk keying (lookup and save). An
            empty / None map is the identity. Used to make a swapped-trial
            cache (an AWS run shipping DataSourceS3 stand-ins) write into the
            same rows the original trials would, so the two runs share one
            results.csv.

    The trial function passed to a Driver must accept
    f(**trial) -> dict | pd.DataFrame; persistence is the cache's job,
    not the function's.
    """

    def __init__(self, *, name: Optional[str] = None,
                 folder: Optional[Path] = None,
                 iter_kwargs: Optional[dict] = None,
                 kwargs: Optional[dict] = None,
                 trial_alias_map: Optional[dict] = None):
        if (name is None) == (folder is None):
            raise ValueError('exactly one of `name` or `folder` required')

        if name is not None:
            folder = get_path_result() / name
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

        if iter_kwargs and kwargs:
            overlap = set(iter_kwargs) & set(kwargs)
            if overlap:
                raise ValueError(
                    f'iter_kwargs and kwargs share keys: {sorted(overlap)}')

        self.iter_kwargs = iter_kwargs
        self.kwargs = kwargs
        self.trial_alias_map = trial_alias_map
        self.df = self._load_results()
        # this cache's own recorder; iter_trial(record=True) scopes each trial on
        # it and the driver passes it to the trial fn, which wraps its calls
        self.recorder = Recorder()

    @property
    def _csv_path(self) -> Path:
        """Path to the results csv inside folder."""
        return self.folder / CSV_NAME

    def _load_results(self) -> pd.DataFrame:
        """Load results.csv into a DataFrame, or empty if absent."""
        if not self._csv_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self._csv_path, index_col=HASH_COL)

    def _cached_hashes(self) -> set:
        """Return the set of trial hashes already on disk (as str)."""
        return set(self.df.index.astype(str))

    def hash(self, trial: dict) -> str:
        """Hash a trial, redirected through trial_alias_map when one is set.

        stable_hash gives the trial's content identity; trial_alias_map then
        optionally maps that to a different hash, so a trial whose spec differs
        only by a swapped-in stand-in (an S3-shipped DataSource on AWS) keys
        the same results row its original spec would. An empty / None map is
        the identity, so an ordinary cache hashes trials unchanged.

        Public because the recorder uses it as the trial_id scoping each
        trial's recorded calls (see glow.benchmark.recorder), so local and
        S3-shipped runs of the same trial record under one id.

        Args:
            trial (dict): the trial kwargs.

        Returns:
            the (possibly aliased) trial hash used for lookup and result IO.
        """
        h = stable_hash(trial)
        if self.trial_alias_map:
            return self.trial_alias_map.get(h, h)
        return h

    def __len__(self) -> int:
        """Total trial count yielded by iter_trial (cached + uncached)."""
        return prod(len(list(v)) for v in (self.iter_kwargs or {}).values())

    def _iter_grid(self) -> Iterator[dict]:
        """Yield one merged kwarg dict per cell of the scalar-axis grid."""
        const = dict(self.kwargs or {})
        if not self.iter_kwargs:
            yield const
            return
        names = list(self.iter_kwargs)
        values = [list(self.iter_kwargs[n]) for n in names]
        for combo in product(*values):
            yield {**const, **dict(zip(names, combo))}

    @property
    def _records_dir(self) -> Path:
        """Folder of per-trial record json files (<trial_id>.json each)."""
        return self.folder / RECORDS_DIR

    def _completed_ids(self) -> set:
        """Trial ids already on disk: per-trial record files plus csv hashes.

        A trial is done if a record file is named for its hash (the recorder
        path) or its hash is on the results.csv index (the legacy DataFrame
        path), so a resumed run skips it either way.
        """
        return self.recorder.completed_ids(self._records_dir) | self._cached_hashes()

    def iter_trial(self, *, include_completed: bool = False,
                   record: bool = False, flush: bool = False,
                   verbose: bool = True) -> Iterator[dict]:
        """Yield each trial's kwargs, optionally recording and flushing it.

        Args:
            include_completed (bool): if False (default) skip trials already on
                disk (resume); if True yield every cell of the grid.
            record (bool): if True open self.recorder.trial(trial_id=hash) around
                each yield, so every call the consumer records (the trial fn,
                given this recorder) lands under that trial id. The recorder is
                cleared before each trial.
            flush (bool): if True (record only) write the trial's captured
                records to records/<hash>.json (axis-stamped) when the consumer
                advances; if False the records stay in self.recorder in memory.
            verbose (bool): show a tqdm bar (record only).

        Yields:
            the trial kwargs; with its recorder scope active when record=True.
        """
        done = set() if include_completed else self._completed_ids()
        trials = [t for t in self._iter_grid() if self.hash(t) not in done]

        if not record:
            yield from trials
            return

        bar = tqdm(total=len(trials), disable=not verbose, desc='trials')
        for trial in trials:
            self.recorder.records.clear()
            with self.recorder.trial(trial_id=self.hash(trial)):
                yield trial
            if flush and self.recorder.records:
                self._records_dir.mkdir(parents=True, exist_ok=True)
                axes = {k: value_id(v) for k, v in trial.items()}
                self.recorder.flush(
                    self._records_dir / f'{self.hash(trial)}.json', **axes)
            bar.update(1)
        bar.close()

    def iter_trial_no_repeat(self) -> Iterator[dict]:
        """Yield only not-yet-completed trials (alias of iter_trial())."""
        return self.iter_trial()

    def is_cached(self, trial: dict) -> bool:
        """Return True if this trial's result is already on disk."""
        return self.hash(trial) in self._completed_ids()

    def load_records(self, consolidate: bool = False) -> list:
        """Load every per-trial record file; optionally consolidate to one json.

        Delegates the disk reads to the recorder. With consolidate=True also
        writes the combined list to <folder>/records.json (outside the per-trial
        records dir, so it is not re-read).

        Args:
            consolidate (bool): also write the union to records.json.

        Returns:
            the combined records across all trials.
        """
        if consolidate:
            return self.recorder.consolidate(self._records_dir,
                                             self.folder / RECORDS_NAME)
        return self.recorder.load(self._records_dir)

    def save_result(self, result, trial: dict) -> None:
        """Append one trial's result to self.df and results.csv.

        result is a dict (one row) or a DataFrame (multiple rows).
        Per-trial kwarg columns are merged into every emitted row; the
        trial_hash lives on self.df's index (so multi-row results share
        an index value).

        Not safe for concurrent writers; the Driver layer is responsible
        for serializing csv writes (or batching).

        Args:
            result: dict (one row) or DataFrame (multiple rows) to
                persist.
            trial (dict): the trial kwargs, hashed for the row index and
                merged in as columns.

        Raises:
            TypeError: if result is neither a dict nor a DataFrame.
        """
        cols = {k: value_id(v) for k, v in trial.items()}
        th = self.hash(trial)

        if isinstance(result, pd.DataFrame):
            new = result.copy()
            for k, v in cols.items():
                new[k] = v
            new.index = pd.Index([th] * len(new), name=HASH_COL)
        elif isinstance(result, dict):
            new = pd.DataFrame(
                [{**cols, **result}],
                index=pd.Index([th], name=HASH_COL))
        else:
            raise TypeError(
                f'result must be dict or DataFrame, '
                f'got {type(result).__name__}')

        self.df = new if self.df.empty else pd.concat([self.df, new])
        self.df.to_csv(self._csv_path, index=True)
