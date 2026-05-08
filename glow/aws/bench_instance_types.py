#!/usr/bin/env python3
"""Compare AWS Batch instance types on a small GLOW workload.

For each instance type listed, create a temporary on-demand AWS Batch
compute environment + queue, submit one identical GLOW benchmark job,
capture wall-time and the inner GLOW time from each result tarball,
then tear everything down.

Stays well under $5 by sizing the workload at ~10s GLOW per job (1k
vox, b=2, 100 perm, n_sa=12) on .large instances (2 vCPUs, 1 physical
core with SMT — matches the 1 vCPU AWS Batch jobs paper.py submits).

Output ranks instance types by inner GLOW time and shows $/run based on
on-demand us-east-1 prices.  Use these to prune
glow-compute-env-spot's instance type list (see setup_aws_batch.sh).

Usage:
    python -m glow.aws.bench_instance_types
    python -m glow.aws.bench_instance_types --instance-types c6i.large c7a.large
    python -m glow.aws.bench_instance_types --dry-run
"""

import argparse
import boto3
import io
import json
import sys
import tarfile
import time
import uuid
from typing import Dict, List, Optional, Tuple


# ---- workload --------------------------------------------------------------

# Sized so total cost is well under $1 across the default 12-type sweep,
# and so the per-job runtime (~10s on the fastest types, ~25s on the
# slowest) sits comfortably above the ~5s container-boot floor for a
# clean inter-type ratio.
BENCH_NUM_VOX = 1000
BENCH_B = 2
BENCH_NUM_IMG = 100
BENCH_N_PERM = 100
BENCH_N_SA = 12


DEFAULT_INSTANCE_TYPES = [
    'c5.large',     # Intel Skylake-SP (2017)
    'c5a.large',    # AMD Naples / Zen 1 (2017)
    'c6i.large',    # Intel Ice Lake (2021)
    'c6a.large',    # AMD Milan / Zen 3 (2021)
    'c7i.large',    # Intel Sapphire Rapids (2023)
    'c7a.large',    # AMD Genoa / Zen 4 (2023)
    'm5.large', 'm6i.large', 'm7i.large',
    'r5.large', 'r6i.large', 'r7i.large',
]

# Approximate on-demand $/hr in us-east-1 (April 2026).  Used for the
# cost estimate guard and per-run dollar comparison — meant for ranking,
# not exact billing.
OD_PRICE_USD_PER_HR = {
    'c5.large':  0.085,  'c5a.large': 0.077,
    'c6i.large': 0.085,  'c6a.large': 0.0765,
    'c7i.large': 0.0893, 'c7a.large': 0.0913,
    'm5.large':  0.096,  'm6i.large': 0.096,
    'm7i.large': 0.1008, 'r5.large':  0.126,
    'r6i.large': 0.126,  'r7i.large': 0.1323,
}

MAX_TOTAL_COST_USD = 5.00
EXPECTED_RUN_SEC = 60       # rough per-job inner runtime
BOOT_OVERHEAD_SEC = 240     # instance boot + image pull overhead

TEMPLATE_CE = 'glow-compute-env-spot'
JOB_DEFINITION = 'glow-job-definition'
REGION = 'us-east-1'

S3_BUCKET = 'glow-experiments'
BENCH_PREFIX = 'glow-bench'


# ---- helpers ---------------------------------------------------------------

def safe_name(s: str) -> str:
    return s.replace('.', '-')


def estimate_cost(instance_types: List[str]) -> float:
    total = 0.0
    for itype in instance_types:
        rate = OD_PRICE_USD_PER_HR.get(itype, 0.10)
        hours = (EXPECTED_RUN_SEC + BOOT_OVERHEAD_SEC) / 3600
        total += rate * hours
    return total


def stage_workload(s3, bench_run_id: str, n_copies: int) -> None:
    """Build a tiny RunAna(GLOW) Config locally, pickle it + n_copies of
    identical kwargs, upload to BENCH_PREFIX/<bench_run_id>/.  Each
    instance type's job uses a unique exp_idx so its result tarball
    lands at a unique S3 key (no overwrite race)."""
    import math
    import numpy as np
    import cloudpickle as pickle  # match glow.benchmark.config.py serialization
    import glow
    from glow.benchmark.config import Config
    from glow.benchmark.runner import RunAna
    from glow.analysis.mancova import get_llr

    side = max(2, math.ceil(BENCH_NUM_VOX ** (1 / 3)))
    config = Config(
        label=f'instbench_{bench_run_id}',
        source='wgn',
        runner=RunAna({'GLOW': (
            glow.analysis.AnalysisGLOW,
            dict(n_perm_fwer=BENCH_N_PERM,
                 alpha_fwer=0.05, min_vox=1, get_stat=get_llr),
        )}),
        n_seed=1,
        effect_llr_all=np.array([0.05]),
        effect_perc=0.2,
        wgn_shape=(side, side, side),
        wgn_a=2,
        wgn_b=BENCH_B,
        wgn_num_img=BENCH_NUM_IMG,
        crop_n_vox=BENCH_NUM_VOX,
        n_jobs=1,
        error_save=False,   # AWS instance benchmarking — fail loudly
    )

    # Worker indexes all_kwargs[exp_idx]; identical kwargs give all
    # instance types the same workload but distinct result keys.
    kw = {'seed': np.int64(0), 'effect_llr': np.float64(0.05)}
    all_kwargs = [dict(kw) for _ in range(n_copies)]

    s3.put_object(Bucket=S3_BUCKET,
                  Key=f'{BENCH_PREFIX}/{bench_run_id}/config.pkl',
                  Body=pickle.dumps(config))
    s3.put_object(Bucket=S3_BUCKET,
                  Key=f'{BENCH_PREFIX}/{bench_run_id}/all_kwargs.pkl',
                  Body=pickle.dumps(all_kwargs))
    return bench_run_id


def get_template_ce(batch) -> dict:
    resp = batch.describe_compute_environments(computeEnvironments=[TEMPLATE_CE])
    ces = resp.get('computeEnvironments', [])
    if not ces:
        sys.exit(f'template compute env {TEMPLATE_CE!r} not found — '
                 f'script borrows its subnets/SG/IAM-role config')
    return ces[0]




def create_compute_env(batch, template, instance_type: str, run_id: str) -> str:
    name = f'glow-bench-{run_id}-{safe_name(instance_type)}'
    cr = template['computeResources']
    batch.create_compute_environment(
        computeEnvironmentName=name,
        type='MANAGED',
        state='ENABLED',
        computeResources={
            'type': 'EC2',                                 # on-demand
            'minvCpus': 0,
            'maxvCpus': 2,
            'desiredvCpus': 0,
            'instanceTypes': [instance_type],
            'subnets': cr['subnets'],
            'securityGroupIds': cr['securityGroupIds'],
            'instanceRole': cr['instanceRole'],
            'allocationStrategy': 'BEST_FIT_PROGRESSIVE',
            'tags': {'glow-bench': run_id},
        },
        serviceRole=template.get('serviceRole', ''),
    )
    return name


def create_job_queue(batch, ce_name: str, instance_type: str, run_id: str) -> str:
    name = f'glow-bench-{run_id}-{safe_name(instance_type)}-q'
    batch.create_job_queue(
        jobQueueName=name,
        state='ENABLED',
        priority=1,
        computeEnvironmentOrder=[
            {'order': 1, 'computeEnvironment': ce_name}
        ],
    )
    return name


def wait_until(check, names: List[str], kind: str, timeout: int = 600,
               interval: int = 5) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if check(names):
            return
        time.sleep(interval)
    raise TimeoutError(f'{kind} did not become ready within {timeout}s')


def all_ce_valid(batch, names):
    resp = batch.describe_compute_environments(computeEnvironments=names)
    return all(ce['state'] == 'ENABLED' and ce.get('status') == 'VALID'
               for ce in resp['computeEnvironments'])


def all_queues_valid(batch, names):
    resp = batch.describe_job_queues(jobQueues=names)
    return all(q['state'] == 'ENABLED' and q.get('status') == 'VALID'
               for q in resp['jobQueues'])


def submit_bench_job(batch, queue: str, instance_type: str,
                     bench_run_id: str, exp_idx: int) -> str:
    cmd = [
        '--s3-bucket', S3_BUCKET,
        '--s3-prefix', BENCH_PREFIX,
        '--run-id', bench_run_id,
        '--exp-idx', str(exp_idx),
    ]
    resp = batch.submit_job(
        jobName=f'gbench_{safe_name(instance_type)}',
        jobQueue=queue,
        jobDefinition=JOB_DEFINITION,
        containerOverrides={'command': cmd},
    )
    return resp['jobId']


def wait_for_jobs(batch, job_ids: List[str],
                  poll_sec: int = 15, timeout: int = 2400) -> Dict[str, dict]:
    """Poll until all jobs reach SUCCEEDED/FAILED, return final descriptions."""
    pending = set(job_ids)
    final: Dict[str, dict] = {}
    deadline = time.time() + timeout
    last_print = 0
    while pending and time.time() < deadline:
        ids = list(pending)
        for i in range(0, len(ids), 100):
            chunk = ids[i:i + 100]
            resp = batch.describe_jobs(jobs=chunk)
            for j in resp['jobs']:
                if j['status'] in ('SUCCEEDED', 'FAILED'):
                    final[j['jobId']] = j
                    pending.discard(j['jobId'])
        now = time.time()
        if now - last_print > 30:
            elapsed = int(now - (deadline - timeout))
            print(f'  [{elapsed:4d}s]  done={len(final)}/{len(job_ids)}')
            last_print = now
        if pending:
            time.sleep(poll_sec)
    if pending:
        print(f'  WARNING: {len(pending)} jobs did not finish before timeout')
    return final


def fetch_inner_time(s3, bench_run_id: str, exp_idx: int) -> Optional[float]:
    """Pull the result tarball, extract time_sec from the GLOW result.json."""
    import io
    import tarfile
    key = f'{BENCH_PREFIX}/{bench_run_id}/results/{exp_idx:06d}.tar.gz'
    try:
        body = s3.get_object(Bucket=S3_BUCKET, Key=key)['Body'].read()
    except Exception:
        return None
    try:
        with tarfile.open(fileobj=io.BytesIO(body), mode='r:gz') as tar:
            for m in tar.getmembers():
                if m.name.endswith('_result.json'):
                    f = tar.extractfile(m)
                    d = json.load(f)
                    if 'time_sec' in d:
                        return float(d['time_sec'])
    except Exception:
        return None
    return None


def cleanup(batch, queue_names: List[str], ce_names: List[str]) -> None:
    print(f'\ncleanup: {len(queue_names)} queue(s), {len(ce_names)} CE(s)')

    # Disable queues, then wait for them to be DISABLED+VALID before delete.
    for q in queue_names:
        try:
            batch.update_job_queue(jobQueue=q, state='DISABLED')
        except Exception as e:
            print(f'  {q} disable: {e}')

    deadline = time.time() + 180
    while queue_names and time.time() < deadline:
        try:
            resp = batch.describe_job_queues(jobQueues=queue_names)
        except Exception:
            break
        ok = all(q['state'] == 'DISABLED' and q.get('status') == 'VALID'
                 for q in resp['jobQueues'])
        if ok:
            break
        time.sleep(5)

    for q in queue_names:
        try:
            batch.delete_job_queue(jobQueue=q)
        except Exception as e:
            print(f'  {q} delete: {e}')

    # Wait for queues to actually be gone — describing a deleted queue returns
    # an empty list.  Until then, the CE has a JobQueue relationship and can't
    # be deleted.
    deadline = time.time() + 240
    while queue_names and time.time() < deadline:
        try:
            resp = batch.describe_job_queues(jobQueues=queue_names)
            remaining = [q['jobQueueName'] for q in resp.get('jobQueues', [])]
        except Exception:
            remaining = []
        if not remaining:
            break
        time.sleep(5)

    for ce in ce_names:
        try:
            batch.update_compute_environment(computeEnvironment=ce,
                                             state='DISABLED')
        except Exception as e:
            print(f'  {ce} disable: {e}')

    deadline = time.time() + 240
    while ce_names and time.time() < deadline:
        try:
            resp = batch.describe_compute_environments(computeEnvironments=ce_names)
        except Exception:
            break
        ok = all(ce['state'] == 'DISABLED' and ce.get('status') == 'VALID'
                 for ce in resp['computeEnvironments'])
        if ok:
            break
        time.sleep(5)

    for ce in ce_names:
        try:
            batch.delete_compute_environment(computeEnvironment=ce)
        except Exception as e:
            print(f'  {ce} delete: {e}')

    print('cleanup done')


# ---- main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--instance-types', nargs='+',
                        default=DEFAULT_INSTANCE_TYPES)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    cost = estimate_cost(args.instance_types)
    print(f'instance types: {len(args.instance_types)}')
    for it in args.instance_types:
        print(f'  - {it}')
    print(f'workload: {BENCH_NUM_VOX} vox, b={BENCH_B}, '
          f'{BENCH_NUM_IMG} img, n_perm={BENCH_N_PERM}+{BENCH_N_SA}')
    print(f'cost estimate (on-demand, ~{EXPECTED_RUN_SEC + BOOT_OVERHEAD_SEC}s/inst): ${cost:.2f}')
    if cost > MAX_TOTAL_COST_USD:
        sys.exit(f'ABORT: estimate ${cost:.2f} > budget ${MAX_TOTAL_COST_USD:.2f}')
    if args.dry_run:
        print('\n--dry-run: not launching')
        return

    batch = boto3.client('batch', region_name=REGION)
    s3 = boto3.client('s3', region_name=REGION)

    template = get_template_ce(batch)
    run_id = uuid.uuid4().hex[:8]
    bench_run_id = f'instbench_{run_id}'
    print(f'\nrun id: {run_id}')

    ce_names: List[str] = []
    queue_names: List[str] = []
    job_ids: Dict[str, str] = {}     # itype -> jobId
    type_exp_idx: Dict[str, int] = {it: i for i, it in enumerate(args.instance_types)}

    try:
        print(f'\n[1/5] stage workload (build Config locally → s3://{S3_BUCKET}/{BENCH_PREFIX}/{bench_run_id}/)')
        stage_workload(s3, bench_run_id, len(args.instance_types))

        print('\n[2/5] create compute envs')
        for itype in args.instance_types:
            ce = create_compute_env(batch, template, itype, run_id)
            ce_names.append(ce)
            print(f'  + {ce}')

        wait_until(lambda n: all_ce_valid(batch, n), ce_names,
                   'compute envs', timeout=600)

        print('\n[3/5] create job queues')
        type_for_ce = dict(zip(args.instance_types, ce_names))
        for itype, ce in type_for_ce.items():
            q = create_job_queue(batch, ce, itype, run_id)
            queue_names.append(q)

        wait_until(lambda n: all_queues_valid(batch, n), queue_names,
                   'job queues', timeout=300)

        print('\n[4/5] submit jobs')
        type_for_q = dict(zip(args.instance_types, queue_names))
        for itype, q in type_for_q.items():
            jid = submit_bench_job(batch, q, itype, bench_run_id,
                                   type_exp_idx[itype])
            job_ids[itype] = jid
            print(f'  + {itype:14s}  job={jid[:8]}  exp_idx={type_exp_idx[itype]}')

        print('\n[5/5] wait (boot + run, expect ~5-8 min)')
        results = wait_for_jobs(batch, list(job_ids.values()),
                                poll_sec=15, timeout=2400)

        print('\ncollect results')
        rows = []
        for itype, jid in job_ids.items():
            j = results.get(jid)
            if j is None:
                rows.append((itype, None, None, None, 'incomplete'))
                continue
            status = j['status']
            started = j.get('startedAt')
            stopped = j.get('stoppedAt')
            wall = (stopped - started) / 1000 if started and stopped else None
            inner = fetch_inner_time(s3, bench_run_id, type_exp_idx[itype])
            rate = OD_PRICE_USD_PER_HR.get(itype, 0.10)
            cost_run = wall / 3600 * rate if wall else None
            rows.append((itype, wall, inner, cost_run, status))

        # sort by inner glow time (None last)
        rows.sort(key=lambda r: (r[2] is None, r[2] or 0))

        print(f'\n{"instance":>14}  {"wall_s":>7}  {"glow_s":>7}  '
              f'{"$/run":>9}  status')
        print('  ' + '-' * 60)
        best_inner = next((r[2] for r in rows if r[2] is not None), None)
        for itype, wall, inner, cost_run, note in rows:
            wall_s = f'{wall:7.1f}' if wall is not None else '     —'
            inner_s = f'{inner:7.2f}' if inner is not None else '     —'
            cost_s = f'${cost_run:7.4f}' if cost_run is not None else '       —'
            rel = ''
            if inner is not None and best_inner:
                rel = f'  ({inner / best_inner:.2f}x)'
            print(f'  {itype:>14}  {wall_s}  {inner_s}  {cost_s}  {note}{rel}')

    finally:
        cleanup(batch, queue_names, ce_names)


if __name__ == '__main__':
    main()
