"""TrialCache-compatible AWS Batch runner.

Mirrors glow.benchmark.driver.driver_local: pulls uncached trials, runs
each through run_fnc, saves the result back through
trial_cache.save_result. Each run is one Batch array job (potentially
escalated through aws_config.memory_mb_tiers for OOM children) whose
worker downloads the per-trial pickle from S3, executes, and uploads the
result.

A worker has only the glow image, no datasets, so a real-data DataSource
(e.g. DataSourceHCP) cannot rebuild its .exp there. Before submitting,
_to_s3_cache builds an S3-shipped twin of each cache: every shippable
DataSource in the trial grid is uploaded once (content-addressed) and
replaced by a DataSourceS3 that downloads it on demand; DataSourceWGN is
deterministic from its seed and left untouched. The twin's distinct trial
hashes are redirected back to the originals via TrialCache.trial_alias_map,
so a trial run on AWS lands in the same results.csv row a local run would
write -- and is later seen as cached by the local driver.

driver_aws_multi runs several caches at once: it uploads and submits
every (twin) cache's trials up front, then polls all the array jobs
together (one tqdm bar per cache) so the Batch queue stays saturated
instead of draining one cache to completion before the next is submitted.
driver_aws is the single-cache wrapper over it.

Each trial's result is downloaded, saved, and deleted from S3 the moment
that trial reaches a terminal state in the poll loop — not after the
whole tier finishes — so results land as soon as they are ready and the
bucket only ever holds the still-in-flight trials.
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import boto3
import cloudpickle
from tqdm import tqdm

from glow.aws.config import s3_key
from glow.aws.datasource import DataSourceS3
from glow.benchmark.data import DataSource, DataSourceWGN
from glow.benchmark.trial_cache import TrialCache
from glow.util import stable_hash, value_id


UPLOAD_THREADS = 16
# AWS Batch arrayProperties.size bounds.
ARRAY_MIN, ARRAY_MAX = 2, 10_000

# Batch job-state lifecycle: a child moves down ACTIVE_STATES (in this
# order) before reaching one of TERMINAL_STATES. The poll bar advances on
# terminal children and tallies the rest by state, so Spot spin-up reads as
# live movement instead of a bar stuck at 0.
ACTIVE_STATES = ('SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING')
TERMINAL_STATES = frozenset({'SUCCEEDED', 'FAILED'})


def driver_aws(trial_cache, run_fnc: Callable, aws_config,
               verbose: bool = True) -> None:
    """Run every uncached trial of trial_cache on AWS Batch.

    Thin single-cache wrapper over driver_aws_multi; the cache's folder
    name labels its progress bar and status lines.

    Args:
        trial_cache (TrialCache): trial spec plus cache.
        run_fnc (Callable): accepts **trial and returns a dict or
            DataFrame, persisted by trial_cache.save_result.
        aws_config (AWSConfig): bucket, queue, definition, region.
        verbose (bool): tqdm progress bar plus status prints.
    """
    driver_aws_multi(
        [(trial_cache.recorder.folder.name, trial_cache, run_fnc)],
        aws_config, verbose=verbose)


def driver_aws_multi(jobs, aws_config, verbose: bool = True) -> None:
    """Run several caches' uncached trials concurrently on AWS Batch.

    Every cache is uploaded and submitted up front, then all the array
    jobs are polled together (one tqdm bar per cache) so the Batch queue
    stays saturated instead of draining one cache before the next is
    submitted. The OOM tier-escalation loop runs across all caches at
    once: each tier submits every cache that still has trials, waits for
    all of them, then escalates only the OOM children to the next tier.

    Args:
        jobs (list): (label, trial_cache, run_fnc) triples. label is a
            human-readable cache name used in prints and bar descriptions;
            run_fnc accepts **trial and returns a dict or DataFrame,
            persisted by that cache's save_result.
        aws_config (AWSConfig): bucket, queue, definition, region.
        verbose (bool): tqdm progress bars plus status prints.
    """
    # Pass 1: which caches have uncached trials? Decided on the original
    # caches, so a fully-cached run returns before any boto3 client is built.
    active_jobs = []
    for label, cache, run_fnc in jobs:
        if next(cache.iter_trial_no_repeat(), None) is not None:
            active_jobs.append((label, cache, run_fnc))
        elif verbose:
            print(f'[driver_aws] {label}: no uncached trials')

    if not active_jobs:
        if verbose:
            print('[driver_aws] no uncached trials; nothing to do.')
        return

    s3 = boto3.client('s3', region_name=aws_config.region)
    batch = boto3.client('batch', region_name=aws_config.region)

    # Pass 2: build each cache's S3-shipped twin (uploading its real-data
    # sources once, deduped across caches via swap_memo), then collect the
    # twin's uncached trials keyed by their worker-side hash. save_result on
    # the twin redirects that hash back to the original via trial_alias_map.
    pending_by_label: Dict[str, Dict[str, dict]] = {}
    fnc_by_label: Dict[str, Callable] = {}
    cache_by_label: dict = {}
    swap_memo: Dict[str, DataSourceS3] = {}
    for label, cache, run_fnc in active_jobs:
        aws_cache = _to_s3_cache(cache, aws_config=aws_config, s3=s3,
                                 swap_memo=swap_memo, verbose=verbose)
        pending = {stable_hash(t): t
                   for t in aws_cache.iter_trial_no_repeat()}
        pending_by_label[label] = pending
        fnc_by_label[label] = run_fnc
        cache_by_label[label] = aws_cache
        if verbose:
            print(f'[driver_aws] {label}: {len(pending)} uncached trials')

    # Upload one job.pkl per uncached trial, all caches under one bar.
    _upload_all_jobs(
        s3, aws_config, pending_by_label, fnc_by_label, verbose=verbose)

    # Global tier-escalation loop. failures/remaining are per label.
    failures_by_label: Dict[str, List[Tuple[str, str]]] = {
        label: [] for label in pending_by_label}
    remaining_by_label: Dict[str, List[str]] = {
        label: list(pending) for label, pending in pending_by_label.items()}

    # Invoked by _poll_attempts the first sweep a child reaches a terminal
    # state: download + save + drop each SUCCEEDED trial right away rather
    # than waiting for the whole tier. Failures fall through to _classify.
    def on_complete(attempt: '_Attempt', trial_hash: str, job: dict) -> None:
        if job.get('status') != 'SUCCEEDED':
            return
        result = _download_result(s3, aws_config, trial_hash)
        cache_by_label[attempt.label].save_result(
            result, pending_by_label[attempt.label][trial_hash])
        _delete_trial_objects(s3, aws_config, trial_hash)

    for tier_idx, mem_mb in enumerate(aws_config.memory_mb_tiers):
        active = {label: rem for label, rem in remaining_by_label.items()
                  if rem}
        if not active:
            break
        is_last = tier_idx == len(aws_config.memory_mb_tiers) - 1
        if verbose:
            total = sum(len(rem) for rem in active.values())
            print(f'[driver_aws] tier {tier_idx} ({mem_mb} MB): '
                  f'{total} trials across {len(active)} cache(s)')

        # Submit every active cache's array job(s), then poll them together.
        attempts: List[_Attempt] = []
        for label, remaining in active.items():
            attempts.extend(_submit_tier(
                s3=s3, batch=batch, aws_config=aws_config, label=label,
                manifest=remaining, mem_mb=mem_mb, verbose=verbose))

        statuses_per_attempt = _poll_attempts(
            batch=batch, attempts=attempts,
            poll_seconds=aws_config.poll_seconds, verbose=verbose,
            on_complete=on_complete)

        # Successes were already downloaded + saved by on_complete; here we
        # only route the failures. A not-OOM failure (timeout, crash, bad
        # image) is permanent: only OOM children retry at the next tier.
        oom_by_label: Dict[str, List[Tuple[str, str]]] = {
            label: [] for label in active}
        for attempt, statuses in zip(attempts, statuses_per_attempt):
            _, oom_failed, other_failed = _classify(statuses, attempt.manifest)
            failures_by_label[attempt.label].extend(other_failed)
            oom_by_label[attempt.label].extend(oom_failed)

        for label in active:
            if is_last:
                for h, _ in oom_by_label[label]:
                    failures_by_label[label].append(
                        (h, f'OOM at {mem_mb} MB (last tier)'))
                remaining_by_label[label] = []
            else:
                remaining_by_label[label] = [h for h, _ in oom_by_label[label]]

    if verbose:
        for label, pending in pending_by_label.items():
            _print_summary(label, pending, failures_by_label[label])


# ---------- S3-shipped twin cache -------------------------------------------


def _is_shippable(v) -> bool:
    """Whether v is a real-data DataSource that must be shipped to S3.

    A DataSourceWGN rebuilds bit-for-bit from its seed on the worker, so it
    is left in place; every other DataSource loads files the worker image
    does not carry, so its built exp is shipped as a DataSourceS3.
    """
    return isinstance(v, DataSource) and not isinstance(v, DataSourceWGN)


def _to_s3_cache(cache, *, aws_config, s3, swap_memo: Dict[str, DataSourceS3],
                 verbose: bool):
    """Build an S3-shipped twin of cache for worker execution.

    Every shippable DataSource (see _is_shippable) appearing in the trial
    grid is uploaded once -- content-addressed, so identical sources across
    trials and caches upload a single time -- and replaced by a DataSourceS3
    that downloads it on the worker. The twin keeps cache.folder, so it reads
    and writes the same results.csv; its trial_alias_map redirects each
    swapped trial's hash back to the original's, keeping the AWS and local
    runs row-for-row consistent.

    A cache with no shippable source (a pure-WGN or scalar-axis grid) is
    returned unchanged: there is nothing to swap and no alias is needed.

    Args:
        cache (TrialCache): the original cache to mirror.
        aws_config (AWSConfig): supplies the S3 bucket and prefix.
        s3: boto3 S3 client.
        swap_memo (dict): value_id(ds) -> DataSourceS3, shared across caches
            so an identical source uploads (and HEAD-probes) once per run.
        verbose (bool): show a tqdm upload bar.

    Returns:
        aws_cache (TrialCache): the twin, or cache itself if nothing to swap.
    """
    grid = list((cache.kwargs or {}).values())
    for vals in (cache.iter_kwargs or {}).values():
        grid.extend(vals)
    shippable = {value_id(v): v for v in grid if _is_shippable(v)}
    if not shippable:
        return cache

    todo = [(k, v) for k, v in shippable.items() if k not in swap_memo]
    bar = tqdm(total=len(todo), disable=not verbose or not todo,
               desc='upload ds.exp')
    for k, v in todo:
        swap_memo[k] = DataSourceS3.from_source(
            v, bucket=aws_config.s3_bucket,
            prefix=aws_config.s3_prefix, s3=s3)
        bar.update(1)
    bar.close()

    def _swap(v):
        return swap_memo[value_id(v)] if _is_shippable(v) else v

    new_kwargs = (None if cache.kwargs is None
                  else {k: _swap(v) for k, v in cache.kwargs.items()})
    new_iter = (None if cache.iter_kwargs is None
                else {k: [_swap(v) for v in vals]
                      for k, vals in cache.iter_kwargs.items()})

    aws_cache = TrialCache(folder=cache.recorder.folder, iter_kwargs=new_iter,
                           kwargs=new_kwargs)
    # Pair each original trial with its swapped twin (same grid order) and
    # alias the twin's hash back, so save_result writes the original's row.
    alias: Dict[str, str] = {}
    for orig, new in zip(cache.iter_trial(include_completed=True),
                         aws_cache.iter_trial(include_completed=True)):
        oh, nh = stable_hash(orig), stable_hash(new)
        if oh != nh:
            alias[nh] = oh
    aws_cache.trial_alias_map = alias
    return aws_cache


# ---------- per-trial job.pkl upload ----------------------------------------


def _job_key(prefix: str, trial_hash: str) -> str:
    """Build the S3 key holding one trial's input pickle."""
    return s3_key(prefix, 'jobs', trial_hash, 'job.pkl')


def _result_key(prefix: str, trial_hash: str) -> str:
    """Build the S3 key holding one trial's result pickle."""
    return s3_key(prefix, 'jobs', trial_hash, 'result.pkl')


def _manifest_key(prefix: str, run_id: str) -> str:
    """Build the S3 key holding one array job's trial-hash manifest."""
    return s3_key(prefix, 'jobs', run_id, 'manifest.pkl')


def _upload_all_jobs(s3, aws_config, pending_by_label, fnc_by_label, verbose):
    """Pickle (run_fnc, trial) per trial and put_object in parallel.

    Spans every cache under one tqdm bar so the pre-submit upload reads as
    a single step. Each trial is pickled with its own cache's run_fnc. The
    trials are already the S3-shipped twins built by _to_s3_cache, so the
    worker downloads any DataSourceS3.exp on demand.

    Args:
        s3: boto3 S3 client.
        aws_config (AWSConfig): supplies the S3 bucket and prefix.
        pending_by_label (dict): {label: {trial_hash: trial}} for the
            uncached trials of every cache.
        fnc_by_label (dict): {label: run_fnc} pickled alongside each of
            that cache's trials.
        verbose (bool): show a tqdm upload bar.
    """
    bucket = aws_config.s3_bucket
    prefix = aws_config.s3_prefix

    items = [(fnc_by_label[label], trial_hash, trial)
             for label, pending in pending_by_label.items()
             for trial_hash, trial in pending.items()]

    def _put_one(item):
        run_fnc, trial_hash, trial = item
        bundle = cloudpickle.dumps((run_fnc, trial))
        s3.put_object(
            Bucket=bucket, Key=_job_key(prefix, trial_hash), Body=bundle)

    with ThreadPoolExecutor(max_workers=UPLOAD_THREADS) as pool:
        bar = tqdm(total=len(items), disable=not verbose, desc='upload jobs')
        for _ in pool.map(_put_one, items):
            bar.update(1)
        bar.close()


# ---------- submit a tier; poll many attempts; classify ---------------------


@dataclass
class _Attempt:
    """One submitted array (or single) job awaiting its terminal status.

    Attributes:
        label (str): cache label this attempt belongs to; routes results
            back and labels the attempt's progress bar.
        manifest (list): trial-hash strings submitted, in child-index
            order; at most ARRAY_MAX long.
        parent_id (str): the submitted job's id.
        is_array (bool): True if submitted as an array job.
    """

    label: str
    manifest: List[str]
    parent_id: str
    is_array: bool

    @property
    def child_ids(self) -> List[str]:
        """Per-child job ids to describe (parent itself if not an array)."""
        if self.is_array:
            return [f'{self.parent_id}:{i}'
                    for i in range(len(self.manifest))]
        return [self.parent_id]


def _submit_tier(*, s3, batch, aws_config, label: str, manifest: List[str],
                 mem_mb: int, verbose: bool):
    """Upload manifest(s) and submit one cache's array job(s) for a tier.

    For trial counts above ARRAY_MAX the manifest is split across
    multiple array submissions, each its own _Attempt.

    Args:
        s3: boto3 S3 client.
        batch: boto3 Batch client.
        aws_config (AWSConfig): bucket, queue, definition, region.
        label (str): cache label, carried on each returned _Attempt.
        manifest (list): trial-hash strings to attempt this tier.
        mem_mb (int): memory ceiling for this tier, in MB.
        verbose (bool): print a per-submission status line.

    Returns:
        attempts (list): _Attempt, one per ARRAY_MAX-sized chunk.
    """
    bucket = aws_config.s3_bucket
    prefix = aws_config.s3_prefix

    attempts: List[_Attempt] = []
    chunks = [manifest[i:i + ARRAY_MAX]
              for i in range(0, len(manifest), ARRAY_MAX)]
    for chunk in chunks:
        run_id = f'run-{uuid.uuid4().hex[:8]}'
        manifest_key = _manifest_key(prefix, run_id)
        s3.put_object(
            Bucket=bucket, Key=manifest_key, Body=cloudpickle.dumps(chunk))
        manifest_uri = f's3://{bucket}/{manifest_key}'

        parent_id, is_array = _submit_job(
            batch=batch, aws_config=aws_config, run_id=run_id,
            manifest_uri=manifest_uri, n=len(chunk), mem_mb=mem_mb)
        if verbose:
            kind = 'array' if is_array else 'single'
            print(f'[driver_aws] {label}: submitted {kind} job {parent_id} '
                  f'({len(chunk)} trial{"s" if len(chunk) != 1 else ""}, '
                  f'{mem_mb} MB)')
        attempts.append(_Attempt(
            label=label, manifest=chunk, parent_id=parent_id,
            is_array=is_array))
    return attempts


def _submit_job(*, batch, aws_config, run_id: str, manifest_uri: str,
                n: int, mem_mb: int):
    """Submit one array (n >= 2) or single (n == 1) Batch job.

    Args:
        batch: boto3 Batch client.
        aws_config (AWSConfig): queue, definition, vcpus, retry, timeout.
        run_id (str): short run identifier used in the job name.
        manifest_uri (str): s3:// URI of the manifest passed to the worker.
        n (int): number of trials in the manifest.
        mem_mb (int): memory ceiling for this tier, in MB.

    Returns:
        parent_job_id (str): the submitted job's id.
        is_array (bool): True if submitted as an array job.
    """
    overrides = {
        # The image ENTRYPOINT is `python -m glow.aws.worker`; the command
        # is appended as its argv, so pass only the manifest URI here.
        'command': [manifest_uri],
        'resourceRequirements': [
            {'type': 'VCPU', 'value': str(aws_config.vcpus)},
            {'type': 'MEMORY', 'value': str(mem_mb)},
        ],
    }
    kwargs = dict(
        jobName=f'glow-{run_id}',
        jobQueue=aws_config.job_queue,
        jobDefinition=aws_config.job_definition,
        containerOverrides=overrides,
        retryStrategy={'attempts': aws_config.retry_attempts},
        timeout={'attemptDurationSeconds':
                 aws_config.timeout_minutes * 60},
    )
    is_array = n >= ARRAY_MIN
    if is_array:
        kwargs['arrayProperties'] = {'size': n}
    response = batch.submit_job(**kwargs)
    return response['jobId'], is_array


def _poll_attempts(*, batch, attempts: List['_Attempt'],
                   poll_seconds: float, verbose: bool,
                   on_complete: 'Callable | None' = None):
    """Block until every child of every attempt reaches a terminal state.

    All attempts are polled in one describe_jobs sweep per interval and
    each gets its own tqdm bar (stacked via position), so concurrently
    submitted array jobs show independent progress. Each bar's postfix
    tallies its still-in-flight children by Batch state (_inflight_postfix)
    so Spot spin-up shows as movement before any child finishes the bar.

    Args:
        batch: boto3 Batch client.
        attempts (list): _Attempt records to wait on.
        poll_seconds (float): sleep between describe_jobs sweeps.
        verbose (bool): show one tqdm bar per attempt.
        on_complete (Callable | None): if given, called once per child the
            first sweep it reaches a terminal state, as
            on_complete(attempt, trial_hash, job) and before that child's
            bar advances. Lets the caller download + drop each result as it
            lands instead of after the whole tier finishes.

    Returns:
        statuses_per_attempt (list): aligned with attempts; each element is
            that attempt's describe_jobs payloads in child-index order.
    """
    all_child_ids = [cid for a in attempts for cid in a.child_ids]
    statuses: Dict[str, dict] = {}

    bars = [tqdm(total=len(a.manifest), desc=a.label, position=i,
                 disable=not verbose)
            for i, a in enumerate(attempts)]
    # Child indices per attempt already handed to on_complete + counted.
    handled: List[set] = [set() for _ in attempts]

    while True:
        # describe_jobs takes max 100 ids per call
        for batch_ids in _chunked(all_child_ids, 100):
            payload = batch.describe_jobs(jobs=batch_ids)
            for job in payload.get('jobs', []):
                statuses[job['jobId']] = job

        all_done = True
        for i, a in enumerate(attempts):
            newly = 0
            for idx, cid in enumerate(a.child_ids):
                if idx in handled[i]:
                    continue
                job = statuses.get(cid, {})
                if job.get('status') not in TERMINAL_STATES:
                    continue
                handled[i].add(idx)
                newly += 1
                if on_complete is not None:
                    on_complete(a, a.manifest[idx], job)
            if newly:
                bars[i].update(newly)
            # Live breakdown of the still-in-flight children by Batch state,
            # so spin-up (SUBMITTED -> RUNNABLE -> STARTING -> RUNNING) reads
            # as movement before any child finishes and advances the bar.
            bars[i].set_postfix_str(_inflight_postfix(a, statuses))
            if len(handled[i]) < len(a.manifest):
                all_done = False

        if all_done:
            break
        time.sleep(poll_seconds)

    for bar in bars:
        bar.close()
    # Preserve child-index order within each attempt.
    return [[statuses[cid] for cid in a.child_ids] for a in attempts]


def _inflight_postfix(attempt: '_Attempt', statuses: Dict[str, dict]) -> str:
    """Summarize an attempt's still-in-flight children by Batch state.

    Reads the per-child status the poll loop already fetched, so it adds no
    API calls. Terminal children are omitted (the bar's n/total counts
    them); states with no children are dropped. A child not yet seen by
    describe_jobs is counted as SUBMITTED.

    Args:
        attempt (_Attempt): the attempt whose children to tally.
        statuses (dict): child job id -> latest describe_jobs payload.

    Returns:
        postfix (str): e.g. 'RUNNABLE=80 STARTING=12 RUNNING=18', or '' once
            every child has reached a terminal state.
    """
    counts: Dict[str, int] = {}
    for cid in attempt.child_ids:
        status = statuses.get(cid, {}).get('status', 'SUBMITTED')
        if status in TERMINAL_STATES:
            continue
        counts[status] = counts.get(status, 0) + 1
    return ' '.join(f'{s}={counts[s]}' for s in ACTIVE_STATES if counts.get(s))


def _chunked(seq, k: int):
    """Yield successive length-k slices of seq."""
    for i in range(0, len(seq), k):
        yield seq[i:i + k]


def _classify(statuses: List[dict], manifest: List[str]):
    """Sort terminal children into completed, oom-failed, other-failed.

    Args:
        statuses (list): describe_jobs payload dicts, in manifest order.
        manifest (list): trial-hash strings aligned with statuses.

    Returns:
        completed (list): trial-hash strings that SUCCEEDED.
        oom (list): (trial_hash, reason) pairs that failed on OOM.
        other (list): (trial_hash, reason) pairs that failed otherwise.
    """
    completed: List[str] = []
    oom: List[Tuple[str, str]] = []
    other: List[Tuple[str, str]] = []

    for trial_hash, job in zip(manifest, statuses):
        status = job.get('status')
        if status == 'SUCCEEDED':
            completed.append(trial_hash)
            continue
        reason = (job.get('statusReason') or '').strip()
        container = job.get('container') or {}
        container_reason = (container.get('reason') or '').strip()
        exit_code = container.get('exitCode')
        summary = (f'exit={exit_code} statusReason={reason!r} '
                   f'container.reason={container_reason!r}')
        if _is_oom(job):
            oom.append((trial_hash, summary))
        else:
            other.append((trial_hash, summary))
    return completed, oom, other


def _is_oom(job: dict) -> bool:
    """Decide whether a failed job was killed for running out of memory.

    Carry-over from the old aws_batch.py: three text checks plus an
    exit-code fallback. Timeouts also exit 137, so the duration/timeout
    phrasing is checked first to keep them out of the OOM retry loop.

    Args:
        job (dict): a describe_jobs payload for one child.

    Returns:
        is_oom (bool): True if the failure looks like an OOM kill.
    """
    status_reason = (job.get('statusReason') or '').lower()
    container = job.get('container') or {}
    container_reason = (container.get('reason') or '').lower()

    for text in (status_reason, container_reason):
        if 'duration' in text and 'timeout' in text:
            return False

    exit_code = container.get('exitCode')
    if exit_code in (137, 134):
        return True

    for text in (status_reason, container_reason):
        if any(key in text for key in ('outofmemory', 'oom', 'memory')):
            return True
    return False


# ---------- result download + summary ---------------------------------------


def _download_result(s3, aws_config, trial_hash: str):
    """Fetch and unpickle one trial's result from S3.

    Args:
        s3: boto3 S3 client.
        aws_config (AWSConfig): supplies the S3 bucket and prefix.
        trial_hash (str): the trial whose result to download.

    Returns:
        result: the unpickled worker result (dict or DataFrame).
    """
    body = s3.get_object(
        Bucket=aws_config.s3_bucket,
        Key=_result_key(aws_config.s3_prefix, trial_hash))['Body'].read()
    return cloudpickle.loads(body)


def _delete_trial_objects(s3, aws_config, trial_hash: str) -> None:
    """Delete one succeeded trial's job + result pickles from S3.

    Called once a trial's result is downloaded and saved, so the bucket
    only ever holds the still-in-flight trials. Safe because a SUCCEEDED
    trial is never resubmitted (only OOM children retry at the next tier).

    Args:
        s3: boto3 S3 client.
        aws_config (AWSConfig): supplies the S3 bucket and prefix.
        trial_hash (str): the trial whose input + result objects to remove.
    """
    prefix = aws_config.s3_prefix
    s3.delete_objects(
        Bucket=aws_config.s3_bucket,
        Delete={'Objects': [
            {'Key': _job_key(prefix, trial_hash)},
            {'Key': _result_key(prefix, trial_hash)},
        ]})


def _print_summary(label: str, pending: Dict[str, dict],
                   failures: Iterable[Tuple[str, str]]) -> None:
    """Print one cache's success count and up to 20 failures.

    Args:
        label (str): cache label, prefixed onto each summary line.
        pending (dict): {trial_hash: trial} for every attempted trial.
        failures (Iterable): (trial_hash, reason) pairs that failed.
    """
    failures = list(failures)
    n_ok = len(pending) - len(failures)
    print(f'[driver_aws] {label}: {n_ok}/{len(pending)} succeeded')
    if failures:
        print(f'[driver_aws] {label}: {len(failures)} failed:')
        for trial_hash, reason in failures[:20]:
            print(f'  {trial_hash}: {reason}')
        if len(failures) > 20:
            print(f'  ... and {len(failures) - 20} more')
        print('[driver_aws] failed trials will resurface on rerun '
              '(iter_trial_no_repeat).')
