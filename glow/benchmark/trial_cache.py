"""Trial iteration spec + result IO + cache lookup.

A TrialCache owns a results csv and answers three questions:

  1. What kwarg dicts should run (cartesian product over iter_kwargs,
     merged with constant kwargs)?
  2. Which of those are already on disk (is_cached,
     iter_trial_no_repeat)?
  3. How do I persist a result (save_result)?
"""

from itertools import product
from math import prod
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from glow.util import stable_hash, value_id
from .file import get_path_result


CSV_NAME = 'results.csv'
HASH_COL = 'trial_hash'


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

    The trial function passed to a Driver must accept
    f(**trial) -> dict | pd.DataFrame; persistence is the cache's job,
    not the function's.
    """

    def __init__(self, *, name: Optional[str] = None,
                 folder: Optional[Path] = None,
                 iter_kwargs: Optional[dict] = None,
                 kwargs: Optional[dict] = None):
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
        self.df = self._load_results()

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

    def __len__(self) -> int:
        """Total trial count yielded by iter_trial (cached + uncached)."""
        return prod(len(list(v)) for v in (self.iter_kwargs or {}).values())

    def iter_trial(self) -> Iterator[dict]:
        """Yield one merged kwarg dict per trial."""
        const = dict(self.kwargs or {})
        if not self.iter_kwargs:
            yield const
            return
        names = list(self.iter_kwargs)
        values = [list(self.iter_kwargs[n]) for n in names]
        for combo in product(*values):
            yield {**const, **dict(zip(names, combo))}

    def iter_trial_no_repeat(self) -> Iterator[dict]:
        """Yield only trials whose result is not already in self.df."""
        cached = self._cached_hashes()
        for trial in self.iter_trial():
            if stable_hash(trial) not in cached:
                yield trial

    def is_cached(self, trial: dict) -> bool:
        """Return True if this trial's result is already in self.df."""
        return stable_hash(trial) in self._cached_hashes()

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
        th = stable_hash(trial)

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
