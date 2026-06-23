"""Trial iteration spec + recorder ownership.

A TrialCache answers two questions and owns one object:

  1. What kwarg dicts should run (cartesian product over iter_kwargs,
     merged with constant kwargs)?
  2. Which of those are already done (their records are on disk)?

and it owns a Recorder, which captures and persists each trial's records (one
json per trial under the cache's records/ dir). The TrialCache itself never
touches disk -- all record reads/writes go through the recorder.

Both answers key off hash(trial), which is stable_hash(trial) optionally
redirected through trial_alias_map. The alias map lets a second cache whose
trials differ only by a swapped-in stand-in (the AWS driver ships every
real-data DataSource as a DataSourceS3) share the original cache's record ids:
see glow.aws.driver._to_s3_cache.
"""

from itertools import product
from math import prod
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

from glow.util import stable_hash, value_id
from .file import get_path_result
from .recorder import Recorder


RECORDS_NAME = 'records.json'
RECORDS_DIR = 'records'

# Sentinel `label` values for trials that produced no scored result, used by
# the trial fns / plotting (not by the cache). SKIP: an infeasible cell, e.g.
# an HCP feature count beyond the pool (see paper.run). ERROR: a failed trial.
ERROR_LABEL = 'ERROR'
SKIP_LABEL = 'SKIP'
NON_RESULT_LABELS = (ERROR_LABEL, SKIP_LABEL)


class TrialCache:
    """Trial iteration spec that owns a recorder; no disk IO of its own.

    Construct with exactly one of name (resolves to
    <default_results_dir>/<name>/) or folder (used directly).

    Attributes:
        folder (Path): on-disk directory for this cache. Created if missing.
        recorder (Recorder): this cache's recorder, rooted at folder/records;
            iter_trial(record=True) scopes each trial on it, and it is the only
            object that reads/writes the per-trial record files.
        iter_kwargs (dict | None): values are iterables; their cartesian
            product yields one kwarg dict per trial.
        kwargs (dict | None): constant kwargs merged into every yielded dict.
        trial_alias_map (dict | None): {raw_hash: alias_hash} redirecting a
            trial's stable_hash for all on-disk keying. An empty / None map is
            the identity. Used so a swapped-trial cache (an AWS run shipping
            DataSourceS3 stand-ins) shares the original trials' record ids.
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
        # this cache's recorder, rooted at the per-trial records dir;
        # iter_trial(record=True) scopes each trial on it and the driver passes
        # it to the trial fn, which wraps its calls
        self.recorder = Recorder(folder=self.folder / RECORDS_DIR)

    def hash(self, trial: dict) -> str:
        """Hash a trial, redirected through trial_alias_map when one is set.

        stable_hash gives the trial's content identity; trial_alias_map then
        optionally maps that to a different hash, so a trial whose spec differs
        only by a swapped-in stand-in (an S3-shipped DataSource on AWS) keys the
        same record file its original spec would. An empty / None map is the
        identity, so an ordinary cache hashes trials unchanged.

        Public because it is the trial_id scoping each trial's recorded calls
        (the per-trial record file is named for it), so local and S3-shipped
        runs of the same trial record under one id.

        Args:
            trial (dict): the trial kwargs.

        Returns:
            the (possibly aliased) trial hash used as the trial id.
        """
        h = stable_hash(trial)
        if self.trial_alias_map:
            return self.trial_alias_map.get(h, h)
        return h

    def __len__(self) -> int:
        """Total trial count (cached + uncached) the grid yields."""
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

    def iter_trial(self, *, include_completed: bool = False,
                   record: bool = False, flush: bool = False,
                   verbose: bool = True) -> Iterator[dict]:
        """Yield each trial's kwargs, optionally recording and flushing it.

        Args:
            include_completed (bool): if False (default) skip trials whose
                records are already on disk (resume); if True yield every cell.
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
        done = set() if include_completed else self.recorder.completed_ids()
        trials = [t for t in self._iter_grid() if self.hash(t) not in done]

        if not record:
            yield from trials
            return

        bar = tqdm(total=len(trials), disable=not verbose, desc='trials')
        for trial in trials:
            self.recorder.records.clear()
            with self.recorder.trial(trial_id=self.hash(trial)):
                yield trial
            if flush:
                axes = {k: value_id(v) for k, v in trial.items()}
                self.recorder.flush(**axes)
            bar.update(1)
        bar.close()

    def iter_trial_no_repeat(self) -> Iterator[dict]:
        """Yield only not-yet-completed trials (alias of iter_trial())."""
        return self.iter_trial()

    def is_cached(self, trial: dict) -> bool:
        """Return True if this trial's records are already on disk."""
        return self.hash(trial) in self.recorder.completed_ids()

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
            return self.recorder.consolidate(self.folder / RECORDS_NAME)
        return self.recorder.load()
