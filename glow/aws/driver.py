"""TrialCache-compatible AWS Batch runner.

Mirrors glow.benchmark.driver.driver_local: pulls uncached trials, runs
each through run_fnc, saves the result back through
trial_cache.save_result. Each run is one Batch array job (potentially
escalated through aws_config.memory_mb_tiers for OOM children) whose
worker downloads the per-trial pickle from S3, executes, and uploads the
result.

driver_aws_multi runs several caches at once: it uploads and submits
every cache's trials up front, then polls all the array jobs together
(one tqdm bar per cache) so the Batch queue stays saturated instead of
draining one cache to completion before the next is submitted.
driver_aws is the single-cache wrapper over it.

Result-key keying uses stable_hash(trial) on the ORIGINAL trial, not the
S3-wrapped worker variant, so AWS and local runs write the same row hash
and share one results.csv.
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
from glow.benchmark.data import DataSourceWGN
from glow.util import stable_hash


UPLOAD_THREADS = 16
# AWS Batch arrayProperties.size bounds.
ARRAY_MIN, ARRAY_MAX = 2, 10_000


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
        [(trial_cache.folder.name, trial_cache, run_fnc)],
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
    # Per label: ORIGINAL trial dict by hash (for save_result), run_fnc,
    # and the cache itself. Caches with nothing uncached are dropped here.
    pending_by_label: Dict[str, Dict[str, dict]] = {}
    fnc_by_label: Dict[str, Callable] = {}
    cache_by_label: dict = {}
    all_trials: list = []
    for label, cache, run_fnc in jobs:
        trials = list(cache.iter_trial_no_repeat())
        if not trials:
            if verbose:
                print(f'[driver_aws] {label}: no uncached trials')
            continue
        pending = {stable_hash(t): t for t in trials}
        pending_by_label[label] = pending
        fnc_by_label[label] = run_fnc
        cache_by_label[label] = cache
        all_trials.extend(trials)
        if verbose:
            print(f'[driver_aws] {label}: {len(pending)} uncached trials')

    if not pending_by_label:
        if verbose:
            print('[driver_aws] no uncached trials; nothing to do.')
        return

    s3 = boto3.client('s3', region_name=aws_config.region)
    batch = boto3.client('batch', region_name=aws_config.region)

    # 1. Upload non-WGN data sources once each (deduped across all caches).
    ds_wrap = _upload_shared_datasources(
        all_trials, aws_config=aws_config, s3=s3, verbose=verbose)

    # 2. Upload one job.pkl per uncached trial, all caches under one bar.
    _upload_all_jobs(
        s3, aws_config, pending_by_label, fnc_by_label, ds_wrap,
        verbose=verbose)

    # 3. Global tier-escalation loop. failures/remaining are per label.
    failures_by_label: Dict[str, List[Tuple[str, str]]] = {
        label: [] for label in pending_by_label}
    remaining_by_label: Dict[str, List[str]] = {
        label: list(pending) for label, pending in pending_by_label.items()}

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
            poll_seconds=aws_config.poll_seconds, verbose=verbose)

        # Classify each attempt and route its results to the right cache.
        oom_by_label: Dict[str, List[Tuple[str, str]]] = {
            label: [] for label in active}
        for attempt, statuses in zip(attempts, statuses_per_attempt):
            completed, oom_failed, other_failed = _classify(
                statuses, attempt.manifest)
            label = attempt.label
            for h in completed:
                result = _download_result(s3, aws_config, h)
                cache_by_label[label].save_result(
                    result, pending_by_label[label][h])
            # A not-OOM failure (timeout, crash, bad image) is permanent:
            # only OOM children are eligible to retry at the next tier.
            failures_by_label[label].extend(other_failed)
            oom_by_label[label].extend(oom_failed)

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


# ---------- shared-datasource upload ----------------------------------------


def _upload_shared_datasources(trials, *, aws_config, s3, verbose):
    """Upload each non-WGN data source's built exp once.

    WGN sources skip the upload; workers rebuild them locally from seed.

    Args:
        trials (list): trial dicts, each possibly carrying a 'ds' key.
        aws_config (AWSConfig): supplies the S3 bucket and prefix.
        s3: boto3 S3 client.
        verbose (bool): show a tqdm upload bar.

    Returns:
        ds_wrap (dict): {id(ds): DataSourceS3} for callers to swap into
            worker_trial.
    """
    ds_wrap: Dict[int, DataSourceS3] = {}
    # id(ds) -> ds, deduping shared sources across trials.
    unique = {}
    for t in trials:
        ds = t.get('ds')
        if ds is None or isinstance(ds, DataSourceWGN):
            continue
        unique.setdefault(id(ds), ds)

    if not unique:
        return ds_wrap

    bar = tqdm(total=len(unique), disable=not verbose,
               desc='upload ds.exp')
    for ds_id, ds in unique.items():
        ds_wrap[ds_id] = DataSourceS3.from_source(
            ds, bucket=aws_config.s3_bucket,
            prefix=aws_config.s3_prefix, s3=s3)
        bar.update(1)
    bar.close()
    return ds_wrap


def _to_worker_trial(trial: dict, ds_wrap: Dict[int, DataSourceS3]) -> dict:
    """Replace ds with its S3 wrapper if one was uploaded for it."""
    ds = trial.get('ds')
    if ds is None or isinstance(ds, DataSourceWGN):
        return trial
    wrap = ds_wrap.get(id(ds))
    if wrap is None:
        return trial
    return {**trial, 'ds': wrap}


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


def _upload_all_jobs(s3, aws_config, pending_by_label, fnc_by_label,
                     ds_wrap, verbose):
    """Pickle (run_fnc, worker_trial) per trial and put_object in parallel.

    Spans every cache under one tqdm bar so the pre-submit upload reads as
    a single step. Each trial is pickled with its own cache's run_fnc.

    Args:
        s3: boto3 S3 client.
        aws_config (AWSConfig): supplies the S3 bucket and prefix.
        pending_by_label (dict): {label: {trial_hash: trial}} for the
            uncached trials of every cache.
        fnc_by_label (dict): {label: run_fnc} pickled alongside each of
            that cache's trials.
        ds_wrap (dict): {id(ds): DataSourceS3} from
            _upload_shared_datasources.
        verbose (bool): show a tqdm upload bar.
    """
    bucket = aws_config.s3_bucket
    prefix = aws_config.s3_prefix

    items = [(fnc_by_label[label], trial_hash, trial)
             for label, pending in pending_by_label.items()
             for trial_hash, trial in pending.items()]

    def _put_one(item):
        run_fnc, trial_hash, trial = item
        bundle = cloudpickle.dumps(
            (run_fnc, _to_worker_trial(trial, ds_wrap)))
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
                   poll_seconds: float, verbose: bool):
    """Block until every child of every attempt reaches a terminal state.

    All attempts are polled in one describe_jobs sweep per interval and
    each gets its own tqdm bar (stacked via position), so concurrently
    submitted array jobs show independent progress.

    Args:
        batch: boto3 Batch client.
        attempts (list): _Attempt records to wait on.
        poll_seconds (float): sleep between describe_jobs sweeps.
        verbose (bool): show one tqdm bar per attempt.

    Returns:
        statuses_per_attempt (list): aligned with attempts; each element is
            that attempt's describe_jobs payloads in child-index order.
    """
    terminal = {'SUCCEEDED', 'FAILED'}
    all_child_ids = [cid for a in attempts for cid in a.child_ids]
    statuses: Dict[str, dict] = {}

    bars = [tqdm(total=len(a.manifest), desc=a.label, position=i,
                 disable=not verbose)
            for i, a in enumerate(attempts)]
    last_done = [0] * len(attempts)

    while True:
        # describe_jobs takes max 100 ids per call
        for batch_ids in _chunked(all_child_ids, 100):
            payload = batch.describe_jobs(jobs=batch_ids)
            for job in payload.get('jobs', []):
                statuses[job['jobId']] = job

        all_done = True
        for i, a in enumerate(attempts):
            done = sum(1 for cid in a.child_ids
                       if statuses.get(cid, {}).get('status') in terminal)
            if done > last_done[i]:
                bars[i].update(done - last_done[i])
                last_done[i] = done
            if done < len(a.manifest):
                all_done = False

        if all_done:
            break
        time.sleep(poll_seconds)

    for bar in bars:
        bar.close()
    # Preserve child-index order within each attempt.
    return [[statuses[cid] for cid in a.child_ids] for a in attempts]


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
