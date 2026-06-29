"""Run a CONFIG cache's data cells on AWS Batch, then build its CSVs.

drive_aws is the AWS counterpart of glow._extra.benchmark.run.run: instead of
sweeping the cells in local joblib workers, it submits them as a Batch array
job (one child per data cell) and lets each worker rebuild + run its cell from
CONFIG, writing its records and run_ana cache to a shared S3 prefix. When the
array drains, the driver pulls the records down and writes the per-config CSVs
with the unchanged results.write_config_csvs -- the AWS path produces the same
records a local run would, so the read side is identical.

The work split needs no per-cell shipping: the array index addresses a cell
(resolve_cells is a pure function of CONFIG, so child i is the same cell on the
driver and the worker; see glow._extra.aws.units). The manifest is a tiny JSON
naming the cache, the sources, and this attempt's cell indices.

OOM escalation is kept from the old driver: a cell killed for memory is
re-submitted at the next memory_mb_tier, and only at the last tier does it
count as a permanent failure. Other failures (timeout, crash) are permanent
and resurface on a rerun. Because the worker syncs its cache as it goes, a
retried cell resumes from the fits it already completed rather than from cold.

The whole sweep is correct to rerun: resubmitted cells whose fits are already
on S3 come back as cache hits (the worker pulls the warm cache first), so a
rerun re-tags membership and fills gaps without recomputing finished work.
"""

import json
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

import boto3
from tqdm import tqdm

from . import s3, sync
from .config import s3_key
from .units import resolve_cells

# AWS Batch arrayProperties.size bounds.
ARRAY_MIN, ARRAY_MAX = 2, 10_000

# Batch job-state lifecycle: a child moves down ACTIVE_STATES (in this order)
# before reaching one of TERMINAL_STATES. The poll bar advances on terminal
# children and tallies the rest by state, so Spot spin-up reads as movement.
ACTIVE_STATES = ('SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING')
TERMINAL_STATES = frozenset({'SUCCEEDED', 'FAILED'})


def drive_aws(names, aws_config, *, sources=('wgn',), write_csv: bool = True,
              verbose: bool = True, out_dir=None) -> dict:
    """Run the selected CONFIG caches on AWS Batch, then write their CSVs.

    Each cache's selected data cells (sources, e.g. WGN only) are submitted as
    a Batch array job and escalated through aws_config.memory_mb_tiers for any
    OOM-killed cells. When every cache's array has drained, the shared records
    are pulled from S3 and the per-config CSVs written.

    Args:
        names (str | list[str]): a CONFIG cache name or list of them.
        aws_config (AWSConfig): bucket, prefix, queue, definition, tiers.
        sources (tuple[str]): data sources to run ('wgn' and/or 'hcp');
            defaults to WGN only (HCP needs its data staged to S3 first).
        write_csv (bool): pull records and write the per-config CSVs at the
            end (results.write_config_csvs).
        verbose (bool): tqdm progress bars and status prints.
        out_dir (str | Path | None): CSV destination forwarded to
            write_config_csvs; None is glow's per-user results dir.

    Returns:
        written (dict): {cache name: csv path} for caches that had rows
            (empty when write_csv is False).
    """
    if isinstance(names, str):
        names = [names]

    s3_client = boto3.client('s3', region_name=aws_config.region)
    batch = boto3.client('batch', region_name=aws_config.region)

    # cell count per cache (the array size); a cache with no selected cell is
    # skipped (e.g. an hcp-only filter against a wgn-only sweep).
    remaining: Dict[str, List[int]] = {}
    for name in names:
        data_cells, *_ = resolve_cells(name, sources)
        if data_cells:
            remaining[name] = list(range(len(data_cells)))
            if verbose:
                print(f'[drive_aws] {name}: {len(data_cells)} cell(s) '
                      f'(sources={list(sources)})')
        elif verbose:
            print(f'[drive_aws] {name}: no cells for sources={list(sources)}')

    failures: Dict[str, List[Tuple[int, str]]] = {n: [] for n in remaining}

    for tier_idx, mem_mb in enumerate(aws_config.memory_mb_tiers):
        active = {n: cells for n, cells in remaining.items() if cells}
        if not active:
            break
        is_last = tier_idx == len(aws_config.memory_mb_tiers) - 1
        if verbose:
            total = sum(len(c) for c in active.values())
            print(f'[drive_aws] tier {tier_idx} ({mem_mb} MB): {total} '
                  f'cell(s) across {len(active)} cache(s)')

        attempts: List[_Attempt] = []
        for name, cells in active.items():
            attempts.extend(_submit_run(
                s3=s3_client, batch=batch, aws_config=aws_config, name=name,
                sources=sources, cell_indices=cells, mem_mb=mem_mb,
                verbose=verbose))

        statuses_per_attempt = _poll_attempts(
            batch=batch, attempts=attempts,
            poll_seconds=aws_config.poll_seconds, verbose=verbose)

        oom: Dict[str, List[int]] = {n: [] for n in active}
        for attempt, statuses in zip(attempts, statuses_per_attempt):
            _, oom_cells, other = _classify(statuses, attempt.cell_indices)
            failures[attempt.name].extend(other)
            oom[attempt.name].extend(oom_cells)

        for name in active:
            if is_last:
                failures[name].extend(
                    (c, f'OOM at {mem_mb} MB (last tier)') for c in oom[name])
                remaining[name] = []
            else:
                remaining[name] = oom[name]

    if verbose:
        for name in failures:
            _print_summary(name, failures[name])

    if not write_csv:
        return {}

    # pull the shared records and write the CSVs with the unchanged read path
    local_dir, key_prefix = sync.records_pair(aws_config.s3_prefix)
    s3.download_prefix(s3_client, aws_config.s3_bucket, key_prefix, local_dir)
    from glow._extra.benchmark.results import write_config_csvs
    written = write_config_csvs(out_dir=out_dir, names=list(remaining))
    if verbose:
        for name, path in written.items():
            print(f'[drive_aws] wrote {name}: {path}')
    return written


# ---------- submit a run; poll many attempts; classify ----------------------


@dataclass
class _Attempt:
    """One submitted array (or single) job awaiting its terminal status.

    Attributes:
        name (str): CONFIG cache this attempt belongs to; routes failures back
            and labels the progress bar.
        cell_indices (list[int]): the data-cell indices submitted, in child
            order; child i runs cell_indices[i]. At most ARRAY_MAX long.
        parent_id (str): the submitted job's id.
        is_array (bool): True if submitted as an array job.
    """

    name: str
    cell_indices: List[int]
    parent_id: str
    is_array: bool

    @property
    def child_ids(self) -> List[str]:
        """Per-child job ids to describe (parent itself if not an array)."""
        if self.is_array:
            return [f'{self.parent_id}:{i}'
                    for i in range(len(self.cell_indices))]
        return [self.parent_id]


def _manifest_key(prefix: str, run_id: str) -> str:
    """The S3 key holding one array job's manifest JSON."""
    return s3_key(prefix, 'runs', run_id, 'manifest.json')


def _submit_run(*, s3, batch, aws_config, name: str, sources, cell_indices,
                mem_mb: int, verbose: bool):
    """Write the manifest(s) and submit one cache's array job(s) for a tier.

    Cell counts above ARRAY_MAX are split across multiple array submissions,
    each its own manifest + _Attempt.

    Args:
        s3: boto3 S3 client.
        batch: boto3 Batch client.
        aws_config (AWSConfig): bucket, prefix, queue, definition, region.
        name (str): CONFIG cache name, carried on each returned _Attempt.
        sources (tuple[str]): the data sources, recorded in the manifest so
            the worker resolves the same cells.
        cell_indices (list[int]): the data-cell indices to attempt this tier.
        mem_mb (int): memory ceiling for this tier, in MB.
        verbose (bool): print a per-submission status line.

    Returns:
        attempts (list[_Attempt]): one per ARRAY_MAX-sized chunk.
    """
    bucket = aws_config.s3_bucket
    prefix = aws_config.s3_prefix

    attempts: List[_Attempt] = []
    chunks = [cell_indices[i:i + ARRAY_MAX]
              for i in range(0, len(cell_indices), ARRAY_MAX)]
    for chunk in chunks:
        run_id = f'{name}-{uuid.uuid4().hex[:8]}'
        manifest = {
            'config_name': name,
            'sources': list(sources),
            'cell_indices': chunk,
            's3_prefix': prefix,
            'region': aws_config.region,
        }
        manifest_key = _manifest_key(prefix, run_id)
        s3.put_object(Bucket=bucket, Key=manifest_key,
                      Body=json.dumps(manifest).encode())
        manifest_uri = f's3://{bucket}/{manifest_key}'

        parent_id, is_array = _submit_job(
            batch=batch, aws_config=aws_config, run_id=run_id,
            manifest_uri=manifest_uri, n=len(chunk), mem_mb=mem_mb)
        if verbose:
            kind = 'array' if is_array else 'single'
            print(f'[drive_aws] {name}: submitted {kind} job {parent_id} '
                  f'({len(chunk)} cell{"s" if len(chunk) != 1 else ""}, '
                  f'{mem_mb} MB)')
        attempts.append(_Attempt(name=name, cell_indices=chunk,
                                 parent_id=parent_id, is_array=is_array))
    return attempts


def _submit_job(*, batch, aws_config, run_id: str, manifest_uri: str,
                n: int, mem_mb: int):
    """Submit one array (n >= 2) or single (n == 1) Batch job.

    Args:
        batch: boto3 Batch client.
        aws_config (AWSConfig): queue, definition, vcpus, retry, timeout.
        run_id (str): short run identifier used in the job name.
        manifest_uri (str): s3:// URI of the manifest passed to the worker.
        n (int): number of cells in the manifest.
        mem_mb (int): memory ceiling for this tier, in MB.

    Returns:
        parent_job_id (str): the submitted job's id.
        is_array (bool): True if submitted as an array job.
    """
    overrides = {
        # the image ENTRYPOINT is `python -m glow._extra.aws.worker`; the
        # command is appended as its argv, so pass only the manifest URI.
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
        timeout={'attemptDurationSeconds': aws_config.timeout_minutes * 60},
    )
    is_array = n >= ARRAY_MIN
    if is_array:
        kwargs['arrayProperties'] = {'size': n}
    response = batch.submit_job(**kwargs)
    return response['jobId'], is_array


def _poll_attempts(*, batch, attempts: List['_Attempt'],
                   poll_seconds: float, verbose: bool):
    """Block until every child of every attempt reaches a terminal state.

    All attempts are polled in one describe_jobs sweep per interval and each
    gets its own tqdm bar (stacked via position), so concurrently submitted
    array jobs show independent progress; each bar's postfix tallies its
    still-in-flight children by Batch state (_inflight_postfix) so Spot spin-up
    shows as movement before any child finishes the bar.

    Args:
        batch: boto3 Batch client.
        attempts (list): _Attempt records to wait on.
        poll_seconds (float): sleep between describe_jobs sweeps.
        verbose (bool): show one tqdm bar per attempt.

    Returns:
        statuses_per_attempt (list): aligned with attempts; each element is
            that attempt's describe_jobs payloads in child order.
    """
    all_child_ids = [cid for a in attempts for cid in a.child_ids]
    statuses: Dict[str, dict] = {}

    bars = [tqdm(total=len(a.cell_indices), desc=a.name, position=i,
                 disable=not verbose)
            for i, a in enumerate(attempts)]
    handled: List[set] = [set() for _ in attempts]

    while True:
        for batch_ids in _chunked(all_child_ids, 100):  # describe max 100/call
            payload = batch.describe_jobs(jobs=batch_ids)
            for job in payload.get('jobs', []):
                statuses[job['jobId']] = job

        all_done = True
        for i, a in enumerate(attempts):
            newly = 0
            for idx, cid in enumerate(a.child_ids):
                if idx in handled[i]:
                    continue
                if statuses.get(cid, {}).get('status') not in TERMINAL_STATES:
                    continue
                handled[i].add(idx)
                newly += 1
            if newly:
                bars[i].update(newly)
            bars[i].set_postfix_str(_inflight_postfix(a, statuses))
            if len(handled[i]) < len(a.cell_indices):
                all_done = False

        if all_done:
            break
        time.sleep(poll_seconds)

    for bar in bars:
        bar.close()
    return [[statuses[cid] for cid in a.child_ids] for a in attempts]


def _inflight_postfix(attempt: '_Attempt', statuses: Dict[str, dict]) -> str:
    """Summarize an attempt's still-in-flight children by Batch state.

    Reads the per-child status the poll loop already fetched, so it adds no API
    calls. Terminal children are omitted (the bar's n/total counts them);
    states with no children are dropped; a child not yet seen counts as
    SUBMITTED.

    Returns:
        postfix (str): e.g. 'RUNNABLE=80 STARTING=12 RUNNING=18', or '' once
            every child is terminal.
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


def _classify(statuses: List[dict], cell_indices: List[int]):
    """Sort terminal children into completed, oom-failed, other-failed cells.

    Args:
        statuses (list): describe_jobs payload dicts, in child order.
        cell_indices (list[int]): data-cell indices aligned with statuses.

    Returns:
        completed (list[int]): cell indices that SUCCEEDED.
        oom (list[int]): cell indices that failed on OOM.
        other (list[tuple]): (cell_index, reason) pairs that failed otherwise.
    """
    completed: List[int] = []
    oom: List[int] = []
    other: List[Tuple[int, str]] = []

    for cell_idx, job in zip(cell_indices, statuses):
        if job.get('status') == 'SUCCEEDED':
            completed.append(cell_idx)
            continue
        reason = (job.get('statusReason') or '').strip()
        container = job.get('container') or {}
        creason = (container.get('reason') or '').strip()
        summary = (f'exit={container.get("exitCode")} '
                   f'statusReason={reason!r} container.reason={creason!r}')
        if _is_oom(job):
            oom.append(cell_idx)
        else:
            other.append((cell_idx, summary))
    return completed, oom, other


def _is_oom(job: dict) -> bool:
    """Decide whether a failed job was killed for running out of memory.

    Three text checks plus an exit-code fallback. Timeouts also exit 137, so
    the duration/timeout phrasing is checked first to keep them out of the OOM
    retry loop.

    Returns:
        is_oom (bool): True if the failure looks like an OOM kill.
    """
    status_reason = (job.get('statusReason') or '').lower()
    container = job.get('container') or {}
    container_reason = (container.get('reason') or '').lower()

    for text in (status_reason, container_reason):
        if 'duration' in text and 'timeout' in text:
            return False

    if container.get('exitCode') in (137, 134):
        return True

    for text in (status_reason, container_reason):
        if any(key in text for key in ('outofmemory', 'oom', 'memory')):
            return True
    return False


def _print_summary(name: str, failures: List[Tuple[int, str]]) -> None:
    """Print one cache's failure count and up to 20 failed cells.

    Args:
        name (str): CONFIG cache name, prefixed onto each summary line.
        failures (list): (cell_index, reason) pairs that failed.
    """
    if not failures:
        print(f'[drive_aws] {name}: all cells succeeded')
        return
    print(f'[drive_aws] {name}: {len(failures)} cell(s) failed:')
    for cell_idx, reason in failures[:20]:
        print(f'  cell {cell_idx}: {reason}')
    if len(failures) > 20:
        print(f'  ... and {len(failures) - 20} more')
    print('[drive_aws] failed cells resurface on a rerun (cache hits skip '
          'finished fits).')
