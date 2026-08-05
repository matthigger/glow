"""Run a CONFIG cache's planted cells on AWS Batch, then build its CSVs.

drive_aws is the AWS counterpart of glow._extra.benchmark.run.run: instead of
sweeping the cells in local joblib workers, it submits them as a Batch array
job (one child per planted cell -- a data cell crossed with one effect) and
lets each worker rebuild + run its cell from the shipped bundle, writing its
records to a shared S3 prefix. The driver pulls finished records down as the
workers ship them (so results land locally as they complete), and when the
array drains does a final pull and writes the per-config CSVs with the
unchanged results.write_config_csvs -- the AWS path produces the same records a
local run would, so the read side is identical.

The driver is the single source of truth for what runs: it resolves a cache's
cells locally (resolve_cells -- see glow._extra.aws.units), drops the cells
already complete in the local records (incomplete_cell_indices -- the local
machine is the source of truth, so a cell whose full leaf set is on disk is
never resubmitted to recompute from cold), and ships the resolved run bundle --
the selected planted cells plus the shared fnc grid (params) and the leaf fnc
(an import reference; see glow._extra.aws.bundle) -- as one pickle per
submission to S3. The worker downloads + unpickles it and runs its array-index
cell (build the data once, plant its one effect, fit each recipe), looking
nothing up in CONFIG (so a config edit ships at submit time, with no image
rebuild; only a code change to glow itself still needs one). The manifest is a
tiny JSON pointing at the bundle (its key, the shared prefix / region, the
cache label). Array child i runs the i-th cell of the shipped bundle -- a
position, not a re-derived index -- so the driver and worker can never disagree
on the cell list.

Failure handling is two-layered. At the Batch level a per-job retryStrategy
(RETRY_EVALUATE_ON_EXIT) retries a Spot reclaim -- a host loss or the graceful
SIGTERM -- in place on a fresh box, but lets an OOM exit rather than re-running
it at the same memory. At the driver level an OOM-killed cell is re-submitted
at the next memory_mb_tier (a permanent failure only at the last tier); other
failures (timeout, crash) are permanent and resurface on a rerun. A
Spot-reclaimed worker checkpoints its partial progress (on the SIGTERM ~2 min
ahead), so a Batch Spot retry resumes from the recipes it already finished
rather than from cold; a later resubmit (a tier escalation or a rerun) restores
that checkpoint too if it is still present.

The whole sweep is correct to rerun: the local records are the source of truth
for what is done, so a rerun resubmits only the cells not already complete on
disk and recomputes them, filling gaps without touching finished work. A rerun
of one method (drive_aws methods=...) narrows the shipped fnc grid, so both the
skip and the worker see only those recipes -- the siblings are not refit.
"""

import json
import pickle
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

import boto3
from tqdm import tqdm

from . import s3, sync
from .bundle import fnc_to_ref
from .config import s3_key
from .units import resolve_cells

# AWS Batch arrayProperties.size bounds.
ARRAY_MIN, ARRAY_MAX = 2, 10_000

# Batch job-state lifecycle: a child moves down ACTIVE_STATES (in this order)
# before reaching one of TERMINAL_STATES. The poll bar advances on terminal
# children and tallies the rest by state, so Spot spin-up reads as movement.
ACTIVE_STATES = ('SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING')
TERMINAL_STATES = frozenset({'SUCCEEDED', 'FAILED'})

# Per-job Batch retry policy (passed as retryStrategy.evaluateOnExit). Rules are
# evaluated in order, first match wins; an attempt matching no rule retries by
# default, so the trailing catch-all is explicit only for clarity. Batch caps
# this at 5 rules. The split: an OOM (137) or abort/bad_alloc (134) exits so it
# is NOT retried at the same memory -- the driver escalates it to the next
# memory_mb_tier instead (retrying in place just re-OOMs). A wall-clock timeout
# also exits 137, so it exits here too and resurfaces on a rerun (warm cache
# shortens it). A Spot reclaim -- a hard host loss ('Host EC2 ... terminated',
# no container exit) or the graceful SIGTERM Batch sends ~2 min ahead (worker
# exits 143) -- retries in place on a fresh box, warm-resuming from the cache
# synced so far. Anything else is treated as transient and retried.
RETRY_EVALUATE_ON_EXIT = [
    {'onExitCode': '137', 'action': 'EXIT'},
    {'onExitCode': '134', 'action': 'EXIT'},
    {'onStatusReason': 'Host EC2*', 'action': 'RETRY'},
    {'onExitCode': '143', 'action': 'RETRY'},
    {'onStatusReason': '*', 'action': 'RETRY'},
]


def drive_aws(names, aws_config, *, write_csv: bool = True,
              verbose: bool = True, out_dir=None, methods=None) -> dict:
    """Run the selected CONFIG caches on AWS Batch, then write their CSVs.

    Each cache's data cells -- whatever sources its CONFIG grid declares, minus
    the cells already complete in the local records -- are submitted as a Batch
    array job and escalated through aws_config.memory_mb_tiers for any
    OOM-killed cells. When every cache's array has drained, the shared records
    are pulled from S3 and the per-config CSVs written. (HCP cells need their
    reference data staged to S3 first; see glow._extra.aws stage_hcp.)

    methods narrows each cache's shipped leaf grid to the named analysis
    recipes (config.filter_ana_list), the path for rerunning one method after
    its recipe changed: the cells to submit are the ones missing those leaves
    (not the cache's whole grid), and a worker fits only the shipped recipes, so
    the siblings already computed are neither resubmitted nor refit. A selected
    cache whose leaf grid has no matching recipe is skipped.

    Args:
        names (str | list[str]): a CONFIG cache name or list of them.
        aws_config (AWSConfig): bucket, prefix, queue, definition, tiers.
        write_csv (bool): pull records and write the per-config CSVs at the
            end (results.write_config_csvs).
        verbose (bool): tqdm progress bars and status prints.
        out_dir (str | Path | None): CSV destination forwarded to
            write_config_csvs; None is glow's per-user results dir.
        methods (list[str] | None): analysis-recipe labels
            (config.ana_kwargs_dict keys, e.g. ['VBA', 'CET']) to run; None
            (default) ships each cache's whole leaf grid.

    Returns:
        written (dict): {cache name: csv path} for caches that had rows
            (empty when write_csv is False).
    """
    if isinstance(names, str):
        names = [names]

    s3_client = boto3.client('s3', region_name=aws_config.region)
    batch = boto3.client('batch', region_name=aws_config.region)

    # resolve each cache's run bundle once (its planted cells + shared fnc grid
    # + leaf fnc); remaining tracks the cell indices still to run,
    # resolved holds the bundle to ship them from. The local records are the
    # source of truth: a cell already complete on disk is dropped from
    # remaining (its full leaf set is present), so a rerun submits only the
    # gaps. RECORDER is loaded once up front for the incomplete_cell_indices
    # walk (which reads the in-memory records).
    from glow._extra.benchmark.config import filter_ana_list
    from glow._extra.benchmark.data import RECORDER
    from glow._extra.benchmark.results import incomplete_cell_indices
    RECORDER.load()

    remaining: Dict[str, List[int]] = {}
    resolved: Dict[str, tuple] = {}
    for name in names:
        cells, kwargs_fnc_list, fnc = resolve_cells(name)
        if methods:
            kwargs_fnc_list = filter_ana_list(kwargs_fnc_list, methods)
            if not kwargs_fnc_list:
                if verbose:
                    print(f'[drive_aws] {name}: no {methods} recipe in its '
                          f'leaf grid, skipped')
                continue
        remaining[name] = incomplete_cell_indices(
            name, kwargs_fnc_list=kwargs_fnc_list)
        resolved[name] = (cells, kwargs_fnc_list, fnc)
        if verbose:
            n_done = len(cells) - len(remaining[name])
            done_note = (f' ({n_done} already complete locally, skipped)'
                         if n_done else '')
            print(f'[drive_aws] {name}: {len(remaining[name])} cell(s)'
                  f'{done_note}')

    failures: Dict[str, List[Tuple[int, str]]] = {n: [] for n in remaining}

    # the records tree to pull down while the arrays run -- the same pair the
    # final pull + CSV write below uses (see _poll_attempts / _pull_finished)
    download_pairs = [sync.records_pair(aws_config.s3_prefix)]

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
                resolved=resolved[name], cell_indices=cells, mem_mb=mem_mb,
                verbose=verbose))

        statuses_per_attempt = _poll_attempts(
            batch=batch, attempts=attempts,
            poll_seconds=aws_config.poll_seconds, verbose=verbose,
            s3_client=s3_client, bucket=aws_config.s3_bucket,
            download_pairs=download_pairs)

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
        cell_indices (list[int]): the original planted-cell indices submitted,
            in child order -- child i ran the i-th cell of its shipped bundle,
            which is cell cell_indices[i]; kept for failure reporting and OOM
            retry. At most ARRAY_MAX long.
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


def _submit_run(*, s3, batch, aws_config, name: str, resolved: tuple,
                cell_indices, mem_mb: int, verbose: bool):
    """Ship the run bundle(s) and submit one cache's array job(s) for a tier.

    Each chunk's selected planted cells are sliced out of the resolved bundle,
    pickled with the shared fnc grid + leaf-fnc reference + cache label, and
    uploaded to S3; the manifest is a tiny JSON pointing at that pickle (see
    glow._extra.aws.bundle for the bundle layout). Cell counts above ARRAY_MAX
    are split across multiple submissions, each its own bundle + manifest +
    _Attempt. Array child i runs the i-th cell of its bundle, so
    _Attempt.cell_indices[i] (the original index, kept for failure reporting /
    OOM retry) is the cell child i ran.

    Args:
        s3: boto3 S3 client.
        batch: boto3 Batch client.
        aws_config (AWSConfig): bucket, prefix, queue, definition, region.
        name (str): CONFIG cache name; the bundle's label and each _Attempt's.
        resolved (tuple): the cache's full bundle
            (cells, kwargs_fnc_list, fnc); this tier's chunk slices its planted
            cells out of cells.
        cell_indices (list[int]): planted-cell indices to attempt this tier.
        mem_mb (int): memory ceiling for this tier, in MB.
        verbose (bool): print a per-submission status line.

    Returns:
        attempts (list[_Attempt]): one per ARRAY_MAX-sized chunk.
    """
    bucket = aws_config.s3_bucket
    prefix = aws_config.s3_prefix
    cells, kwargs_fnc_list, fnc = resolved

    attempts: List[_Attempt] = []
    chunks = [cell_indices[i:i + ARRAY_MAX]
              for i in range(0, len(cell_indices), ARRAY_MAX)]
    for chunk in chunks:
        run_id = f'{name}-{uuid.uuid4().hex[:8]}'
        # fnc rides as an import reference (code), the rest as pickled params;
        # see glow._extra.aws.bundle
        bundle = ([cells[c] for c in chunk], kwargs_fnc_list,
                  fnc_to_ref(fnc), name)
        bundle_key = s3_key(prefix, 'runs', run_id, 'bundle.pkl')
        s3.put_object(Bucket=bucket, Key=bundle_key,
                      Body=pickle.dumps(bundle))
        manifest = {
            'config_name': name,
            'bundle_key': bundle_key,
            's3_prefix': prefix,
            'region': aws_config.region,
            'n_cells': len(chunk),
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
        retryStrategy={'attempts': aws_config.retry_attempts,
                       'evaluateOnExit': RETRY_EVALUATE_ON_EXIT},
        timeout={'attemptDurationSeconds': aws_config.timeout_minutes * 60},
    )
    is_array = n >= ARRAY_MIN
    if is_array:
        kwargs['arrayProperties'] = {'size': n}
    response = batch.submit_job(**kwargs)
    return response['jobId'], is_array


def _poll_attempts(*, batch, attempts: List['_Attempt'],
                   poll_seconds: float, verbose: bool,
                   s3_client=None, bucket: str = None, download_pairs=None,
                   download_interval: float = s3.SYNC_INTERVAL_SEC):
    """Block until every child of every attempt reaches a terminal state.

    All attempts are polled in one describe_jobs sweep per interval and each
    gets its own tqdm bar (stacked via position), so concurrently submitted
    array jobs show independent progress; each bar's postfix tallies its
    still-in-flight children by Batch state (_inflight_postfix) so Spot spin-up
    shows as movement before any child finishes the bar.

    Between sweeps the finished records are pulled down (download_pairs), so
    results land locally as the workers ship them up rather than only at the
    end; the pull is incremental (download_prefix skips files already local)
    and gated to download_interval so a fast status poll does not re-LIST S3
    every sweep.

    Args:
        batch: boto3 Batch client.
        attempts (list): _Attempt records to wait on.
        poll_seconds (float): sleep between describe_jobs sweeps.
        verbose (bool): show one tqdm bar per attempt.
        s3_client: boto3 S3 client for the mid-run pull (None disables it).
        bucket (str): S3 bucket the records live in.
        download_pairs (list[tuple] | None): (local_dir, key_prefix) trees to
            pull down between sweeps; None skips the mid-run pull entirely.
        download_interval (float): minimum seconds between pulls; defaults to
            the worker's upload interval (s3.SYNC_INTERVAL_SEC).

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
    last_download = time.time()

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
        # pull finished records down between sweeps so results land locally as
        # workers ship them up; the drain-time pull in drive_aws is the final
        # catch-up
        if download_pairs and time.time() - last_download >= download_interval:
            _pull_finished(s3_client, bucket, download_pairs, verbose)
            last_download = time.time()
        time.sleep(poll_seconds)

    for bar in bars:
        bar.close()
    return [[statuses[cid] for cid in a.child_ids] for a in attempts]


def _pull_finished(s3_client, bucket: str, pairs, verbose: bool) -> int:
    """Pull records the workers have shipped since the last sweep.

    Incremental: download_prefix skips files already local, so only records
    finished since the previous pull move. Reports via tqdm.write so the live
    poll bars are not clobbered.

    Args:
        s3_client: boto3 S3 client.
        bucket (str): source bucket.
        pairs (list[tuple]): (local_dir, key_prefix) trees to pull down.
        verbose (bool): print a note when files were downloaded.

    Returns:
        n (int): number of files pulled this call.
    """
    n = 0
    for local_dir, key_prefix in pairs:
        n += s3.download_prefix(s3_client, bucket, key_prefix, local_dir)
    if verbose and n:
        tqdm.write(f'[drive_aws] pulled {n} new result file(s)')
    return n


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
        cell_indices (list[int]): planted-cell indices aligned with statuses.

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
