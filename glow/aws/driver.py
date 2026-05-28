"""TrialCache-compatible AWS Batch runner.

Mirrors glow.benchmark.driver.driver_local: pulls uncached trials, runs
each through run_fnc, saves the result back through
trial_cache.save_result. Each run is one Batch array job (potentially
escalated through aws_config.memory_mb_tiers for OOM children) whose
worker downloads the per-trial pickle from S3, executes, and uploads the
result.

Result-key keying uses stable_hash(trial) on the ORIGINAL trial, not the
S3-wrapped worker variant, so AWS and local runs write the same row hash
and share one results.csv.
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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

    Args:
        trial_cache (TrialCache): trial spec plus cache.
        run_fnc (Callable): accepts **trial and returns a dict or
            DataFrame, persisted by trial_cache.save_result.
        aws_config (AWSConfig): bucket, queue, definition, region.
        verbose (bool): tqdm progress bar plus status prints.
    """
    trials = list(trial_cache.iter_trial_no_repeat())
    if not trials:
        if verbose:
            print('[driver_aws] no uncached trials; nothing to do.')
        return

    s3 = boto3.client('s3', region_name=aws_config.region)
    batch = boto3.client('batch', region_name=aws_config.region)

    # Map trial_hash -> ORIGINAL trial dict (used for save_result). One job
    # per uncached trial, deduped if hashes collide (rare).
    pending: Dict[str, dict] = {stable_hash(t): t for t in trials}
    if verbose:
        print(f'[driver_aws] {len(pending)} uncached trials')

    # 1. Upload non-WGN data sources once each, build DataSourceS3 wrappers.
    ds_wrap = _upload_shared_datasources(
        trials, aws_config=aws_config, s3=s3, verbose=verbose)

    # 2. Upload one job.pkl per uncached trial.
    _upload_jobs(
        s3, aws_config, run_fnc, pending, ds_wrap, verbose=verbose)

    # 3. Tier-escalation loop.
    # failures holds (trial_hash, reason) pairs.
    failures: List[Tuple[str, str]] = []
    remaining = list(pending)
    for tier_idx, mem_mb in enumerate(aws_config.memory_mb_tiers):
        if not remaining:
            break
        is_last = tier_idx == len(aws_config.memory_mb_tiers) - 1
        if verbose:
            print(f'[driver_aws] tier {tier_idx} ({mem_mb} MB): '
                  f'{len(remaining)} trials')

        completed, oom_failed, other_failed = _run_one_tier(
            s3=s3, batch=batch, aws_config=aws_config,
            trial_hashes=remaining, mem_mb=mem_mb, verbose=verbose)

        for h in completed:
            result = _download_result(s3, aws_config, h)
            trial_cache.save_result(result, pending[h])

        # A not-OOM failure (timeout, crash, bad image) is permanent: only
        # OOM children are eligible to retry at the next memory tier.
        for h, reason in other_failed:
            failures.append((h, reason))

        if is_last:
            for h, reason in oom_failed:
                failures.append((h, f'OOM at {mem_mb} MB (last tier)'))
            remaining = []
        else:
            remaining = [h for h, _ in oom_failed]

    if verbose:
        _print_summary(pending, failures)


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


def _upload_jobs(s3, aws_config, run_fnc, pending, ds_wrap, verbose):
    """Pickle (run_fnc, worker_trial) per trial and put_object in parallel.

    Args:
        s3: boto3 S3 client.
        aws_config (AWSConfig): supplies the S3 bucket and prefix.
        run_fnc (Callable): the per-trial run function pickled alongside
            each trial.
        pending (dict): {trial_hash: trial} for the uncached trials.
        ds_wrap (dict): {id(ds): DataSourceS3} from
            _upload_shared_datasources.
        verbose (bool): show a tqdm upload bar.
    """
    bucket = aws_config.s3_bucket
    prefix = aws_config.s3_prefix

    def _put_one(trial_hash):
        trial = pending[trial_hash]
        bundle = cloudpickle.dumps(
            (run_fnc, _to_worker_trial(trial, ds_wrap)))
        s3.put_object(
            Bucket=bucket, Key=_job_key(prefix, trial_hash), Body=bundle)

    with ThreadPoolExecutor(max_workers=UPLOAD_THREADS) as pool:
        bar = tqdm(total=len(pending), disable=not verbose,
                   desc='upload jobs')
        for _ in pool.map(_put_one, list(pending)):
            bar.update(1)
        bar.close()


# ---------- one tier-attempt: submit + poll + classify ----------------------


def _run_one_tier(*, s3, batch, aws_config, trial_hashes: List[str],
                  mem_mb: int, verbose: bool):
    """Submit and poll a single array-job attempt at one memory tier.

    For trial counts above ARRAY_MAX the manifest is split across
    multiple array submissions; the results merge before returning.

    Args:
        s3: boto3 S3 client.
        batch: boto3 Batch client.
        aws_config (AWSConfig): bucket, queue, definition, region.
        trial_hashes (list): trial-hash strings to attempt this tier.
        mem_mb (int): memory ceiling for this tier, in MB.
        verbose (bool): tqdm progress bar plus status prints.

    Returns:
        completed (list): trial-hash strings that SUCCEEDED.
        oom_failed (list): (trial_hash, reason) pairs that failed on OOM.
        other_failed (list): (trial_hash, reason) pairs that failed for
            any other reason.
    """
    completed: List[str] = []
    oom: List[Tuple[str, str]] = []
    other: List[Tuple[str, str]] = []

    chunks = [trial_hashes[i:i + ARRAY_MAX]
              for i in range(0, len(trial_hashes), ARRAY_MAX)]
    for chunk in chunks:
        c, o, x = _run_one_array(
            s3=s3, batch=batch, aws_config=aws_config,
            manifest=chunk, mem_mb=mem_mb, verbose=verbose)
        completed.extend(c)
        oom.extend(o)
        other.extend(x)
    return completed, oom, other


def _run_one_array(*, s3, batch, aws_config, manifest: List[str],
                   mem_mb: int, verbose: bool):
    """Upload the manifest, submit one array (or single) job, and poll.

    Args:
        s3: boto3 S3 client.
        batch: boto3 Batch client.
        aws_config (AWSConfig): bucket, queue, definition, region.
        manifest (list): trial-hash strings for this single submission,
            at most ARRAY_MAX long.
        mem_mb (int): memory ceiling for this tier, in MB.
        verbose (bool): tqdm progress bar plus status prints.

    Returns:
        completed (list): trial-hash strings that SUCCEEDED.
        oom_failed (list): (trial_hash, reason) pairs that failed on OOM.
        other_failed (list): (trial_hash, reason) pairs that failed for
            any other reason.
    """
    run_id = f'run-{uuid.uuid4().hex[:8]}'
    bucket = aws_config.s3_bucket
    prefix = aws_config.s3_prefix
    manifest_key = _manifest_key(prefix, run_id)

    s3.put_object(
        Bucket=bucket, Key=manifest_key, Body=cloudpickle.dumps(manifest))
    manifest_uri = f's3://{bucket}/{manifest_key}'

    parent_id, is_array = _submit_job(
        batch=batch, aws_config=aws_config, run_id=run_id,
        manifest_uri=manifest_uri, n=len(manifest), mem_mb=mem_mb)
    if verbose:
        kind = 'array' if is_array else 'single'
        print(f'[driver_aws] submitted {kind} job {parent_id} '
              f'({len(manifest)} trial{"s" if len(manifest) != 1 else ""}, '
              f'{mem_mb} MB)')

    statuses = _poll_until_terminal(
        batch=batch, parent_id=parent_id, n=len(manifest),
        is_array=is_array, poll_seconds=aws_config.poll_seconds,
        verbose=verbose)

    return _classify(statuses, manifest)


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
        'command': ['python', '-m', 'glow.aws.worker', manifest_uri],
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


def _poll_until_terminal(*, batch, parent_id: str, n: int, is_array: bool,
                         poll_seconds: float, verbose: bool):
    """Block until every child reaches a terminal state.

    Args:
        batch: boto3 Batch client.
        parent_id (str): the submitted job's id.
        n (int): number of children to wait on.
        is_array (bool): True if parent_id is an array job.
        poll_seconds (float): sleep between describe_jobs polls.
        verbose (bool): show a tqdm progress bar.

    Returns:
        statuses (list): describe_jobs payload dicts, one per child, in
            array-index order (a single-element list for non-array).
    """
    terminal = {'SUCCEEDED', 'FAILED'}
    child_ids = ([f'{parent_id}:{i}' for i in range(n)]
                 if is_array else [parent_id])

    bar = tqdm(total=n, disable=not verbose, desc='aws batch')
    last_done = 0
    statuses: Dict[str, dict] = {}

    while True:
        # describe_jobs takes max 100 ids per call
        for batch_ids in _chunked(child_ids, 100):
            payload = batch.describe_jobs(jobs=batch_ids)
            for job in payload.get('jobs', []):
                statuses[job['jobId']] = job

        done = sum(1 for j in statuses.values()
                   if j.get('status') in terminal)
        if done > last_done:
            bar.update(done - last_done)
            last_done = done

        if done >= n:
            break
        time.sleep(poll_seconds)

    bar.close()
    # Preserve array-index order.
    return [statuses[i] for i in child_ids]


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


def _print_summary(pending: Dict[str, dict],
                   failures: Iterable[Tuple[str, str]]) -> None:
    """Print the success count and up to 20 failures.

    Args:
        pending (dict): {trial_hash: trial} for every attempted trial.
        failures (Iterable): (trial_hash, reason) pairs that failed.
    """
    failures = list(failures)
    n_ok = len(pending) - len(failures)
    print(f'[driver_aws] done: {n_ok}/{len(pending)} succeeded')
    if failures:
        print(f'[driver_aws] {len(failures)} failed:')
        for trial_hash, reason in failures[:20]:
            print(f'  {trial_hash}: {reason}')
        if len(failures) > 20:
            print(f'  ... and {len(failures) - 20} more')
        print('[driver_aws] failed trials will resurface on rerun '
              '(iter_trial_no_repeat).')
