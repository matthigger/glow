"""AWS Batch integration for parallel permutation processing."""

import json
import re
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager

import boto3
import cloudpickle as pickle
from botocore.exceptions import ClientError
from tqdm import tqdm

MONITOR_COLS = 79


@dataclass
class CloudConfig:
    """configuration for AWS cloud execution."""
    s3_bucket: str
    s3_prefix: str
    job_queue: str
    job_definition: str
    region: str = 'us-east-1'
    max_concurrent_jobs: int = 100
    timeout_minutes: int = 60
    memory_mb: int = 2000
    vcpus: int = 1
    retry_attempts: int = 3
    shared_exp_sources: List[str] = field(default_factory=lambda: ['hcp'])
    oom_memory_mb_tiers: List[int] = field(
        default_factory=lambda: [4000, 8000, 16000]
    )  # 4 -> 8 -> 16 GB on OOM; hard error above 16 GB
    max_spot_retries: int = 3  # max resubmissions per job due to spot reclamation
    
    def to_dict(self):
        return asdict(self)



class AWSBatchRunner:
    """manages AWS Batch execution for permutation processing."""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.s3 = boto3.client('s3', region_name=config.region)
        self.batch = boto3.client('batch', region_name=config.region)
        self.ecs = boto3.client('ecs', region_name=config.region)
        self.ec2 = boto3.client('ec2', region_name=config.region)
        self._job_memory_tier_index = {}
        self._job_spot_retry_count = {}
        self._experiment_shapes = {}

    @staticmethod
    def _base_job_name(job_name: str) -> str:
        # strip array child index suffix (e.g. ":42")
        name = job_name.split(':')[0]
        return name.rsplit('_retry', 1)[0]

    @staticmethod
    def _is_spot_termination(job_failure: Dict[str, Any]) -> bool:
        status_reason = (job_failure.get('statusReason') or '').lower()
        return 'host ec2' in status_reason and 'terminated' in status_reason

    @staticmethod
    def _is_timeout_failure(job_failure: Dict[str, Any]) -> bool:
        status_reason = (job_failure.get('statusReason') or '').lower()
        return 'duration' in status_reason and 'timeout' in status_reason

    @staticmethod
    def _is_oom_failure(job_failure: Dict[str, Any]) -> bool:
        status_reason = (job_failure.get('statusReason') or '').lower()
        container = job_failure.get('container', {}) or {}
        container_reason = (container.get('reason') or '').lower()
        # timeout kills also produce exit code 137 — check text first
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

    def _resubmit_failed_jobs(self, failed_jobs, job_info_map):
        """resubmit OOM and spot-terminated jobs.

        OOM jobs advance to the next memory tier.  Spot-terminated jobs are
        resubmitted with the same resource requirements.  Jobs that exceed
        the max memory tier are recorded as permanently failed (not raised).

        Returns:
            resubmitted (list): new job IDs
            resubmit_reasons (dict): maps source job ID -> 'oom' | 'spot'
            messages (list): status messages to display
            permanently_failed (list): jobs that exceeded max memory tier
        """
        tiers = self.config.oom_memory_mb_tiers or []

        resubmitted = []
        resubmit_reasons = {}
        messages = []
        permanently_failed = []
        for job in failed_jobs:
            # determine failure type — check timeout first (exit 137 overlap)
            if self._is_timeout_failure(job):
                permanently_failed.append({
                    'jobId': job['jobId'],
                    'jobName': job['jobName'],
                    'reason': 'timeout: job exceeded attemptDurationSeconds',
                })
                messages.append(
                    f'  ✗ {job["jobName"]}: timed out — permanent failure')
                continue
            is_oom = self._is_oom_failure(job)
            is_spot = (not is_oom) and self._is_spot_termination(job)
            if not (is_oom or is_spot):
                continue

            base_name = self._base_job_name(job['jobName'])
            container = job.get('container', {}) or {}
            command = container.get('command')
            if not command:
                messages.append(f'  ⚠ Cannot resubmit {job["jobName"]}: missing command')
                continue

            # For array child jobs, replace --index-map with explicit index arg
            job_id = job['jobId']
            if ':' in job_id and job_id in job_info_map:
                info = job_info_map[job_id]
                command = list(command)  # copy
                # Remove --index-map and its value from command
                new_cmd = []
                skip_next = False
                for tok in command:
                    if skip_next:
                        skip_next = False
                        continue
                    if tok == '--index-map':
                        skip_next = True
                        continue
                    new_cmd.append(tok)
                command = new_cmd
                # Add explicit index arg
                if 'exp_idx' in info:
                    command.extend(['--exp-idx', str(info['exp_idx'])])
                elif 'perm_indices' in info:
                    command.extend([
                        '--perm-indices',
                        ','.join(str(x) for x in info['perm_indices'])])
                elif 'perm_idx' in info:
                    command.extend(['--perm-idx', str(info['perm_idx'])])

            if is_oom:
                current_idx = self._job_memory_tier_index.get(base_name, -1)
                next_idx = current_idx + 1
                if next_idx >= len(tiers) or len(tiers) < 2:
                    max_mb = tiers[-1] if tiers else self.config.memory_mb
                    permanently_failed.append({
                        'jobId': job['jobId'],
                        'jobName': job['jobName'],
                        'reason': (
                            f'OOM: exceeded max memory tier ({max_mb} MB)'),
                    })
                    messages.append(
                        f'  ✗ {job["jobName"]}: OOM at max tier '
                        f'({max_mb} MB) — permanent failure')
                    continue
                memory_mb = tiers[next_idx]
                retry_name = f'{base_name}_retry{next_idx}'
                reason_tag = 'oom'
            else:
                # spot: resubmit with same memory
                count = self._job_spot_retry_count.get(base_name, 0)
                if count >= self.config.max_spot_retries:
                    continue
                current_idx = self._job_memory_tier_index.get(base_name, -1)
                memory_mb = (tiers[current_idx] if current_idx >= 0 and tiers
                             else self.config.memory_mb)
                retry_name = f'{base_name}_spot{count + 1}'
                reason_tag = 'spot'

            overrides: Dict[str, Any] = {
                'command': command,
                'resourceRequirements': [
                    {'type': 'VCPU', 'value': str(self.config.vcpus)},
                    {'type': 'MEMORY', 'value': str(memory_mb)},
                ],
            }

            try:
                response = self.batch.submit_job(
                    jobName=retry_name,
                    jobQueue=self.config.job_queue,
                    jobDefinition=self.config.job_definition,
                    containerOverrides=overrides,
                    retryStrategy={'attempts': self.config.retry_attempts},
                    timeout={'attemptDurationSeconds': self.config.timeout_minutes * 60}
                )
                new_job_id = response['jobId']
                resubmitted.append(new_job_id)
                resubmit_reasons[job['jobId']] = reason_tag
                if job['jobId'] in job_info_map:
                    job_info_map[new_job_id] = job_info_map[job['jobId']]

                if is_oom:
                    self._job_memory_tier_index[base_name] = next_idx
                    messages.append(f'  ↻ {job["jobName"]} → {memory_mb} MB ({retry_name})')
                else:
                    self._job_spot_retry_count[base_name] = \
                        self._job_spot_retry_count.get(base_name, 0) + 1
                    messages.append(f'  ↻ {job["jobName"]} → resubmitted ({retry_name})')
            except ClientError as e:
                messages.append(f'  ✗ Failed to resubmit {job["jobName"]}: {e}')

        return resubmitted, resubmit_reasons, messages, permanently_failed
    
    def estimate_memory_mb(self, exp_shape):
        """Estimate minimum memory for a worker (perm or streaming synthesis).

        Uses a Lasso regression model fitted by
        ``glow.benchmark.memory`` if available, otherwise falls back to
        a 3x heuristic on the data tensor size.

        Args:
            exp_shape: (b, num_img, num_vox) tuple, or an experiment object
                       with a ``.y`` attribute.

        Returns:
            memory_mb (int or None): OOM tier to request, or None when
            the default ``config.memory_mb`` is sufficient.

        Raises:
            MemoryError: if the estimate exceeds the highest OOM tier.
        """
        if hasattr(exp_shape, 'y'):
            exp_shape = exp_shape.y.shape
        b, num_img, num_vox = exp_shape

        est_mb = self._predict_memory_mb(b, num_img, num_vox)

        tiers = self.config.oom_memory_mb_tiers or []
        default_mb = self.config.memory_mb
        max_mb = tiers[-1] if tiers else default_mb

        if est_mb > max_mb:
            raise MemoryError(
                f'Experiment {exp_shape} needs ~{est_mb:.0f} MB, '
                f'exceeding max tier ({max_mb} MB). '
                f'Reduce num_vox or increase oom_memory_mb_tiers.')
        if est_mb <= default_mb:
            return None
        for t in tiers:
            if t >= est_mb:
                return t

    @staticmethod
    def _predict_memory_mb(b, num_img, num_vox):
        """Predict peak memory in MB from experiment dimensions.

        Tries to load a fitted model from ``memory_permutation.json`` (produced
        by ``python -m glow.benchmark.memory``).  Falls back to a 3x
        heuristic on the raw data tensor size.
        """
        try:
            from glow.benchmark.memory import load_model, predict
            model = load_model()
            if model is not None:
                return predict(model, num_vox, b, num_img)
        except (ImportError, FileNotFoundError):
            pass
        return 3.0 * b * num_img * num_vox * 8 / (1024 * 1024)

    def estimate_experiment_memory_mb(self, b: int, num_img: int, num_vox: int, n_perm: int):
        """Estimate memory for an experiment worker (full run_ana in one process).

        Uses the Lasso regression in memory_experiment.json (from
        ``python -m glow.benchmark.memory --profile experiment``) if present.
        Returns None if the model is missing or if the estimate is <= default
        (so the job definition default memory is used).
        """
        try:
            from glow.benchmark.memory import load_experiment_model, predict_experiment_memory
        except ImportError:
            return None
        model = load_experiment_model()
        if model is None:
            return None
        est_mb = predict_experiment_memory(model, num_vox, b, num_img, n_perm)

        default_mb = self.config.memory_mb
        tiers = self.config.oom_memory_mb_tiers or []
        max_mb = tiers[-1] if tiers else default_mb

        if est_mb > max_mb:
            raise MemoryError(
                f'Experiment worker estimate {est_mb:.0f} MB exceeds max tier '
                f'({max_mb} MB). Reduce size or increase oom_memory_mb_tiers.')
        if est_mb <= default_mb:
            return None
        for t in tiers:
            if t >= est_mb:
                return t
        return tiers[-1] if tiers else None

    def upload_experiment(self, exp, ana_kwargs: Dict[str, Any], 
                         experiment_id: str) -> str:
        """upload experiment data to S3
        
        Args:
            exp: experiment object
            ana_kwargs: kwargs for analysis (without n_perm)
            experiment_id: unique identifier for this experiment
        
        Returns:
            s3_path: path to uploaded experiment data
        """
        self._experiment_shapes[experiment_id] = exp.y.shape

        # prepare data bundle
        data = {
            'exp': exp,
            'ana_kwargs': ana_kwargs,
            'experiment_id': experiment_id
        }
        
        # serialize
        data_bytes = pickle.dumps(data)
        
        # upload to S3
        s3_key = f'{self.config.s3_prefix}/experiments/{experiment_id}/data.pkl'
        
        try:
            self.s3.put_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Body=data_bytes
            )
            return f's3://{self.config.s3_bucket}/{s3_key}'
        except ClientError as e:
            raise RuntimeError(f'failed to upload experiment: {e}')
    
    def check_existing_results(self, experiment_id, n_perm):
        """return the set of permutation indices that already have results."""
        completed = set()
        prefix = f'{self.config.s3_prefix}/results/{experiment_id}/'
        
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.config.s3_bucket,
                                           Prefix=prefix):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    # extract perm_idx from filename
                    key = obj['Key']
                    if key.endswith('_result.pkl'):
                        filename = Path(key).name
                        perm_idx_str = filename.split('_')[0]
                        try:
                            perm_idx = int(perm_idx_str)
                            if 0 <= perm_idx <= n_perm:
                                completed.add(perm_idx)
                        except ValueError:
                            continue
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket' or error_code == 'NoSuchKey':
                pass  # bucket/prefix doesn't exist yet
            else:
                raise
        
        return completed
    
    @staticmethod
    def _fmt_duration(minutes):
        """Format minutes into human-readable duration."""
        if minutes < 60:
            return f'{minutes:.1f} min'
        if minutes < 1440:
            return f'{minutes / 60:.1f} hr'
        return f'{minutes / 1440:.1f} days'

    @staticmethod
    def estimate_batch_table(n_perm, perm_sec, overhead_sec=180,
                             cost_per_vcpu_hr=0.02):
        """Print a table of perms-per-job vs cost/time tradeoffs.

        Args:
            n_perm: total permutations to run
            perm_sec: estimated seconds per permutation
            overhead_sec: fixed per-job overhead — instance scheduling,
                image pull, container start, data transfer (default 180s)
            cost_per_vcpu_hr: cost per vCPU-hour (default $0.02)

        Returns:
            list of dicts with table rows
        """
        fixed_sec = overhead_sec

        candidates = [1, 2, 5, 10, 25, 50]
        # also include n_perm itself (single job) if not already covered
        candidates = [c for c in candidates if c <= n_perm]
        if not candidates or candidates[-1] < n_perm:
            candidates.append(n_perm)

        rows = []
        for ppj in candidates:
            n_jobs = -(-n_perm // ppj)  # ceil division
            compute_sec = ppj * perm_sec
            total_sec = compute_sec + fixed_sec
            total_min = total_sec / 60
            overhead_pct = fixed_sec / total_sec * 100
            total_vcpu_hr = n_jobs * total_sec / 3600
            cost = total_vcpu_hr * cost_per_vcpu_hr
            rows.append({
                'perms_per_job': ppj,
                'n_jobs': n_jobs,
                'wall_min': total_min,
                'overhead_pct': overhead_pct,
                'vcpu_hr': total_vcpu_hr,
                'cost': cost,
            })

        fmt = AWSBatchRunner._fmt_duration
        perm_str = fmt(perm_sec / 60)
        fixed_str = fmt(fixed_sec / 60)

        print(f'\n  ~{fixed_str} per-job overhead budgeted '
              f'(instance scheduling, image pull, container start, '
              f'data transfer)')
        print(f'  Estimated compute: ~{perm_str}/perm, '
              f'{n_perm} permutations total\n')

        hdr = (f'  {"perms/job":>10s}  {"jobs":>6s}  {"total time":>10s}'
               f'  {"overhead":>8s}  {"vCPU-hrs":>9s}  {"cost":>8s}')
        sep = '  ' + '-' * 62
        print(hdr)
        print(sep)
        for r in rows:
            print(f'  {r["perms_per_job"]:>10d}  {r["n_jobs"]:>6d}'
                  f'  {fmt(r["wall_min"]):>10s}'
                  f'  {r["overhead_pct"]:>7.1f}%'
                  f'  {r["vcpu_hr"]:>9.1f}'
                  f'  ${r["cost"]:>7.2f}')
        print(sep)
        return rows

    def submit_jobs(self, experiment_id, n_perm, skip_completed=True,
                    memory_mb=None, perms_per_job=1):
        """submit permutation jobs to AWS Batch.

        Args:
            experiment_id: experiment identifier
            n_perm: number of permutations (0 to n_perm inclusive)
            skip_completed: skip permutations that already have results
            memory_mb: override memory per job (None = auto-estimate from
                       experiment dimensions, or job definition default)
            perms_per_job: number of permutations per job (>1 = batch mode)

        Returns:
            submission_info dict with job_ids and metadata
        """
        if memory_mb is None and experiment_id in self._experiment_shapes:
            memory_mb = self.estimate_memory_mb(
                self._experiment_shapes[experiment_id])

        # check which permutations are already done
        completed = set()
        if skip_completed:
            completed = self.check_existing_results(experiment_id, n_perm)
            print(f'found {len(completed)} completed permutations, will skip')

        # determine which permutations to run
        perm_indices = [i for i in range(n_perm + 1) if i not in completed]

        if len(perm_indices) == 0:
            print('all permutations already completed')
            return {'n_jobs': 0, 'job_ids': [], 'skipped': list(completed)}

        # batch permutations into jobs
        if perms_per_job > 1:
            batches = [perm_indices[i:i + perms_per_job]
                       for i in range(0, len(perm_indices), perms_per_job)]
            n_jobs = len(batches)
            index_arg = '--perm-indices'
            indices = batches
        else:
            n_jobs = len(perm_indices)
            index_arg = '--perm-idx'
            indices = perm_indices

        print(f'\n{"="*60}')
        print('AWS BATCH JOB SUBMISSION')
        print(f'{"="*60}')
        print(f'experiment: {experiment_id}')
        print(f'permutations: {len(perm_indices)}'
              f' ({perms_per_job} per job, {n_jobs} jobs)')
        print(f'skipped (completed): {len(completed)}')
        print(f'timeout: {self.config.timeout_minutes} min per job')
        print(f'{"="*60}\n')

        # submit jobs via array job
        data_path = f's3://{self.config.s3_bucket}/{self.config.s3_prefix}/experiments/{experiment_id}/data.pkl'

        command_template = [
            '--data-path', data_path,
            '--s3-bucket', self.config.s3_bucket,
            '--s3-prefix', self.config.s3_prefix,
            '--experiment-id', experiment_id,
        ]

        # scale timeout for batch mode
        timeout = None
        if perms_per_job > 1:
            timeout = self.config.timeout_minutes * perms_per_job

        array_info = self.submit_array_job(
            job_name=f'{experiment_id}_perm',
            command_template=command_template,
            indices=indices,
            index_arg=index_arg,
            memory_mb=memory_mb,
            timeout_minutes=timeout,
        )
        job_ids = array_info['child_job_ids']

        if memory_mb is not None:
            print(f'submitted {len(job_ids)} jobs (memory: {memory_mb} MB)')
        else:
            print(f'submitted {len(job_ids)} jobs')

        # build job_info_map for monitor_jobs resubmit support
        job_info_map = {}
        for child_id, actual_idx in array_info['index_map'].items():
            if isinstance(actual_idx, list):
                job_info_map[child_id] = {'perm_indices': actual_idx}
            else:
                job_info_map[child_id] = {'perm_idx': actual_idx}

        return {
            'n_jobs': n_jobs,
            'job_ids': job_ids,
            'perm_indices': perm_indices,
            'skipped': list(completed),
            'index_map': array_info['index_map'],
            'job_info_map': job_info_map,
        }
    
    def submit_synthesis_job(self, experiment_id, n_perm, memory_mb=None):
        """Submit a synthesis job that polls S3 for permutation results.

        The synthesis worker waits until all permutation result files appear
        in S3 before running ``_finalize_analysis``, so AWS Batch
        ``dependsOn`` is not needed (works for any n_perm).

        Returns:
            synthesis job ID (str)
        """
        if memory_mb is None and experiment_id in self._experiment_shapes:
            memory_mb = self.estimate_memory_mb(
                self._experiment_shapes[experiment_id])
        job_name = f'{experiment_id}_synthesis'
        overrides = {
            'command': [
                '--synthesize',
                '--experiment-id', experiment_id,
                '--n-perm', str(n_perm),
                '--s3-bucket', self.config.s3_bucket,
                '--s3-prefix', self.config.s3_prefix,
            ]
        }
        if memory_mb is not None:
            overrides['resourceRequirements'] = [
                {'type': 'VCPU', 'value': str(self.config.vcpus)},
                {'type': 'MEMORY', 'value': str(memory_mb)},
            ]

        try:
            response = self.batch.submit_job(
                jobName=job_name,
                jobQueue=self.config.job_queue,
                jobDefinition=self.config.job_definition,
                containerOverrides=overrides,
                retryStrategy={'attempts': 1},
                timeout={'attemptDurationSeconds': self.config.timeout_minutes * 60},
            )
            mem_str = f' (memory: {memory_mb} MB)' if memory_mb else ''
            print(f'submitted synthesis job: {response["jobId"][:12]}...{mem_str}')
            if memory_mb is not None:
                tiers = self.config.oom_memory_mb_tiers or []
                for idx, t in enumerate(tiers):
                    if t >= memory_mb:
                        self._job_memory_tier_index[job_name] = idx
                        break
            return response['jobId']
        except ClientError as e:
            raise RuntimeError(f'failed to submit synthesis job: {e}')

    def download_final_analysis(self, experiment_id):
        """Download the final analysis pickle produced by the synthesis worker.

        Returns:
            unpickled AnalysisGLOW object
        """
        final_key = (f'{self.config.s3_prefix}/results/'
                     f'{experiment_id}/analysis_final.pkl')
        try:
            response = self.s3.get_object(
                Bucket=self.config.s3_bucket, Key=final_key)
            return pickle.loads(response['Body'].read())
        except ClientError as e:
            raise RuntimeError(
                f'failed to download final analysis '
                f'(s3://{self.config.s3_bucket}/{final_key}): {e}')

    def monitor_jobs(self, job_ids, poll_interval=30,
                    job_info_map=None, cancel_on_error=True):
        """poll AWS Batch until all jobs finish, downloading results as they complete.

        Returns:
            list of permanently failed job dicts (OOM at max tier), or
            empty list if all jobs succeeded or were retried.
        """
        if not job_ids:
            print('no jobs to monitor')
            return
        
        print(f'monitoring {len(job_ids)} jobs...\n')
        
        # build job_info_map from job names if not provided
        if job_info_map is None:
            job_info_map = {}
            for i in range(0, len(job_ids), 100):
                chunk = job_ids[i:i+100]
                try:
                    response = self.batch.describe_jobs(jobs=chunk)
                    for job in response['jobs']:
                        # extract run_id and exp_idx from job name: glow_{run_id}_exp{exp_idx:06d}
                        job_name = job['jobName']
                        if job_name.startswith('glow_') and '_exp' in job_name:
                            parts = job_name.replace('glow_', '').split('_exp')
                            if len(parts) == 2:
                                run_id = parts[0]
                                try:
                                    exp_idx = int(parts[1])
                                    job_info_map[job['jobId']] = {
                                        'run_id': run_id,
                                        'exp_idx': exp_idx,
                                        'output_folder': None  # will be set by caller
                                    }
                                except ValueError:
                                    pass
                except ClientError:
                    pass
        
        failed_jobs = []  # track failed jobs for detailed reporting
        resubmitted_job_ids = set()  # jobs that were successfully resubmitted
        permanently_failed_jobs = []  # jobs that exceeded max memory tier
        first_check = True  # track if this is the first instance type check
        had_running_jobs = False  # track if we've seen running jobs (for immediate check)
        downloaded_jobs = set()  # track which jobs have been downloaded
        start_time = time.time()
        last_done_count = 0
        last_done_time = start_time
        resubmitted_total = 0
        last_status_print = 0  # track when we last printed status
        
        # track job timestamps for vCPU-hours calculation
        # maps job_id -> {'started_at': timestamp, 'stopped_at': timestamp or None}
        job_timestamps = {}
        
        # track instance types currently in use
        # maps instance_type -> count of running jobs on that type
        instance_check_interval = 30  # check instance types every 30 seconds
        instance_type_counts = {}
        instance_type_error = None
        last_instance_check = -instance_check_interval  # initialize to allow immediate first check
        
        # group jobs by config (run_id) for per-config progress bars
        config_jobs = {}  # run_id -> set of job_ids
        for jid in job_ids:
            if jid in job_info_map and 'run_id' in job_info_map[jid]:
                rid = job_info_map[jid]['run_id']
            else:
                rid = 'jobs'
            config_jobs.setdefault(rid, set()).add(jid)

        # create per-config progress bars
        config_names = sorted(config_jobs.keys())
        config_pbars = {}
        bar_fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        # strip trailing _<hex> hash from run_id for display
        _strip_hash = lambda s: re.sub(r'_[0-9a-f]{8,12}$', '', s)
        for i, name in enumerate(config_names):
            config_pbars[name] = tqdm(
                total=len(config_jobs[name]),
                desc=_strip_hash(name), unit='job',
                bar_format=bar_fmt, ncols=MONITOR_COLS,
                position=i + 1, leave=True)
        config_done = {name: 0 for name in config_names}
        config_resubmitted = {name: 0 for name in config_names}
        first_pbar = config_pbars[config_names[0]]
        job_statuses_map = {}  # job_id -> latest status string

        def _clear_heartbeat():
            print('\r' + ' ' * MONITOR_COLS + '\r', end='', flush=True)

        def _write(msg):
            """write message above progress bars, clearing heartbeat first."""
            _clear_heartbeat()
            first_pbar.write(msg)

        try:
            while True:
                # get job statuses and timestamps
                statuses = {'SUBMITTED': 0, 'PENDING': 0, 'RUNNABLE': 0,
                           'STARTING': 0, 'RUNNING': 0, 'SUCCEEDED': 0,
                           'FAILED': 0}
                
                jobs_found = 0
                completed_jobs = []  # jobs that just completed
                newly_failed_jobs = []  # jobs that just failed
                current_time = time.time()
                
                # batch describe in chunks of 100
                for i in range(0, len(job_ids), 100):
                    chunk = job_ids[i:i+100]
                    try:
                        response = self.batch.describe_jobs(jobs=chunk)
                        jobs_found += len(response['jobs'])
                        for job in response['jobs']:
                            job_id = job['jobId']
                            status = job['status']
                            statuses[status] = statuses.get(status, 0) + 1
                            job_statuses_map[job_id] = status

                            # track timestamps for vCPU-hours calculation
                            if job_id not in job_timestamps:
                                job_timestamps[job_id] = {}
                            
                            # track when job started
                            if 'startedAt' in job and job['startedAt']:
                                started_at_val = job['startedAt']
                                # Handle both datetime objects and Unix timestamps (int/float)
                                if hasattr(started_at_val, 'timestamp'):
                                    started_at = started_at_val.timestamp()
                                elif isinstance(started_at_val, (int, float)):
                                    started_at = float(started_at_val) / 1000 if started_at_val > 1e10 else float(started_at_val)
                                else:
                                    started_at = None
                                
                                if started_at is not None and 'started_at' not in job_timestamps[job_id]:
                                    job_timestamps[job_id]['started_at'] = started_at
                            
                            # track when job stopped
                            if 'stoppedAt' in job and job['stoppedAt']:
                                stopped_at_val = job['stoppedAt']
                                # Handle both datetime objects and Unix timestamps (int/float)
                                if hasattr(stopped_at_val, 'timestamp'):
                                    stopped_at = stopped_at_val.timestamp()
                                elif isinstance(stopped_at_val, (int, float)):
                                    stopped_at = float(stopped_at_val) / 1000 if stopped_at_val > 1e10 else float(stopped_at_val)
                                else:
                                    stopped_at = None
                                
                                if stopped_at is not None and 'stopped_at' not in job_timestamps[job_id]:
                                    job_timestamps[job_id]['stopped_at'] = stopped_at
                            
                            # track newly completed jobs
                            if status == 'SUCCEEDED' and job_id not in downloaded_jobs:
                                completed_jobs.append(job)
                            
                            # track failed jobs with details and print immediately
                            if status == 'FAILED' and job_id not in [f['jobId'] for f in failed_jobs]:
                                job_failure = {
                                    'jobId': job_id,
                                    'jobName': job['jobName'],
                                    'statusReason': job.get('statusReason', 'Unknown'),
                                    'container': job.get('container', {}),
                                    'jobDefinition': job.get('jobDefinition', '')
                                }
                                failed_jobs.append(job_failure)
                                newly_failed_jobs.append(job_failure)
                    except ClientError as e:
                        _write(f'error checking jobs: {e}')
                        continue
                
                # get instance types for running jobs (periodically, to avoid too many API calls)
                # Check immediately on first iteration if there are running jobs, then every interval
                # Also check if we just transitioned from 0 to >0 running jobs
                just_started_running = (statuses['RUNNING'] > 0 and not had_running_jobs)
                should_check_instances = ((current_time - last_instance_check >= instance_check_interval or just_started_running) and 
                                         statuses['RUNNING'] > 0)
                
                # Track if we've seen running jobs
                if statuses['RUNNING'] > 0:
                    had_running_jobs = True
                
                if should_check_instances:
                    try:
                        # get compute environment name from job queue
                        queue_info = self.batch.describe_job_queues(
                            jobQueues=[self.config.job_queue]
                        )
                        if not queue_info.get('jobQueues'):
                            if first_check:
                                _write(f'  Note: Could not find job queue: {self.config.job_queue}')
                            raise ValueError("Job queue not found")
                            
                        compute_envs = queue_info['jobQueues'][0].get('computeEnvironmentOrder', [])
                        if not compute_envs:
                            if first_check:
                                _write(f'  Note: Job queue has no compute environments')
                            raise ValueError("No compute environments")
                            
                        compute_env_name = compute_envs[0].get('computeEnvironment')
                        
                        # get ECS cluster name from compute environment
                        env_info = self.batch.describe_compute_environments(
                            computeEnvironments=[compute_env_name]
                        )
                        if not env_info.get('computeEnvironments'):
                            if first_check:
                                _write(f'  Note: Could not find compute environment: {compute_env_name}')
                            raise ValueError("Compute environment not found")
                            
                        ecs_cluster_arn = env_info['computeEnvironments'][0].get('ecsClusterArn', '')
                        if not ecs_cluster_arn:
                            if first_check:
                                _write(f'  Note: Compute environment has no ECS cluster')
                            raise ValueError("No ECS cluster")
                            
                        ecs_cluster = ecs_cluster_arn.split('/')[-1]
                        
                        # get container instances from ECS cluster
                        new_instance_type_counts = {}
                        paginator = self.ecs.get_paginator('list_container_instances')
                        found_instances = False
                        for page in paginator.paginate(cluster=ecs_cluster):
                            if not page.get('containerInstanceArns'):
                                continue
                            
                            found_instances = True
                            # describe container instances to get EC2 instance IDs
                            container_instances = self.ecs.describe_container_instances(
                                cluster=ecs_cluster,
                                containerInstances=page['containerInstanceArns']
                            )
                            
                            # get EC2 instance IDs
                            ec2_instance_ids = []
                            for ci in container_instances.get('containerInstances', []):
                                ec2_instance_id = ci.get('ec2InstanceId')
                                if ec2_instance_id:
                                    ec2_instance_ids.append(ec2_instance_id)
                            
                            # describe EC2 instances to get instance types
                            if ec2_instance_ids:
                                ec2_instances = self.ec2.describe_instances(
                                    InstanceIds=ec2_instance_ids
                                )
                                for reservation in ec2_instances.get('Reservations', []):
                                    for instance in reservation.get('Instances', []):
                                        if instance.get('State', {}).get('Name') == 'running':
                                            instance_type = instance.get('InstanceType', 'unknown')
                                            # count instances of each type
                                            new_instance_type_counts[instance_type] = new_instance_type_counts.get(instance_type, 0) + 1
                        
                        if not found_instances and first_check:
                            _write(f'  Note: No container instances found in ECS cluster (jobs may not have started yet)')
                        
                        # update outer scope variable
                        instance_type_counts = new_instance_type_counts
                    except Exception as e:
                        # Log error but don't fail - instance type display is optional
                        if instance_type_error is None:
                            if isinstance(e, ClientError):
                                error_code = e.response.get('Error', {}).get('Code', '')
                                if error_code in ['UnauthorizedOperation', 'AccessDenied', 'AccessDeniedException']:
                                    instance_type_error = (
                                        'missing permission: ec2:DescribeInstances '
                                        '(ask to add this to the IAM user/role)'
                                    )
                            if instance_type_error is None:
                                instance_type_error = f'{type(e).__name__}: {e}'
                        if first_check:  # Only print on first failure to avoid spam
                            _write(f'  Note: Could not fetch instance types: {instance_type_error}')
                            _write('  (This is optional - jobs will still run normally)')
                    
                    last_instance_check = current_time
                    first_check = False  # Mark that we've attempted the first check
                
                # calculate vCPU-hours
                vcpu_hours = 0.0
                vcpus = self.config.vcpus
                for job_id, timestamps in job_timestamps.items():
                    if 'started_at' in timestamps:
                        if 'stopped_at' in timestamps:
                            # completed job: use stopped_at - started_at
                            runtime_hours = (timestamps['stopped_at'] - timestamps['started_at']) / 3600
                        else:
                            # running job: use current_time - started_at
                            runtime_hours = (current_time - timestamps['started_at']) / 3600
                        vcpu_hours += runtime_hours * vcpus
                
                # calculate status counts (don't count resubmitted OOM jobs
                # as done — they are being retried, not finished)
                total = len(job_ids) - resubmitted_total
                done = statuses['SUCCEEDED']
                all_terminal = statuses['SUCCEEDED'] + statuses['FAILED'] - resubmitted_total
                pending = (statuses['SUBMITTED'] + statuses['PENDING'] + 
                          statuses['RUNNABLE'] + statuses['STARTING'])
                running = statuses['RUNNING']
                completed = statuses['SUCCEEDED']
                failed = statuses['FAILED']
                submitted = statuses['SUBMITTED']
                starting = statuses['STARTING']
                
                # estimate remaining time
                eta_str = '?'
                if done > last_done_count:
                    elapsed = current_time - last_done_time
                    rate = (done - last_done_count) / elapsed if elapsed > 0 else 0
                    remaining = total - done
                    if rate > 0:
                        eta_seconds = remaining / rate
                        eta_str = f'{eta_seconds/60:.1f}m' if eta_seconds > 60 else f'{eta_seconds:.0f}s'
                    
                    last_done_count = done
                    last_done_time = current_time
                elif done > 0:
                    # reuse previous ETA calculation if no new completions
                    elapsed = current_time - last_done_time
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = total - done
                    if rate > 0:
                        eta_seconds = remaining / rate
                        eta_str = f'{eta_seconds/60:.1f}m' if eta_seconds > 60 else f'{eta_seconds:.0f}s'
                
                # print status update (scrolls above progress bars)
                should_print = (current_time - last_status_print >= poll_interval or
                               done > last_done_count or
                               len(newly_failed_jobs) > 0 or
                               len(completed_jobs) > 0)

                if should_print:
                    state_parts = []
                    if submitted > 0:
                        state_parts.append(f'Submitted: {submitted}')
                    if pending > 0:
                        state_parts.append(f'Pending: {pending}')
                    if statuses["RUNNABLE"] > 0:
                        state_parts.append(f'Runnable: {statuses["RUNNABLE"]}')
                    if starting > 0:
                        state_parts.append(f'Starting: {starting}')
                    if running > 0:
                        state_parts.append(f'Running: {running}')
                    if completed > 0:
                        state_parts.append(f'Succeeded: {completed}')
                    n_perm_failed = len(permanently_failed_jobs)
                    n_resubmitted = resubmitted_total - n_perm_failed
                    actual_failed = failed - resubmitted_total
                    if n_resubmitted > 0:
                        state_parts.append(f'Resubmitted: {n_resubmitted}')
                    if n_perm_failed > 0:
                        state_parts.append(f'Perm. failed: {n_perm_failed}')
                    if actual_failed > 0:
                        state_parts.append(f'Failed: {actual_failed}')

                    state_str = ', '.join(state_parts) if state_parts else 'All done'

                    timestamp_str = datetime.now().strftime('%H:%M:%S')
                    parts = [f'[{timestamp_str}]']
                    parts.append(f'vCPU-hours: {vcpu_hours:.3f}')
                    parts.append(f'Jobs: {state_str}')
                    if instance_type_counts:
                        instance_str = ', '.join([f'{itype}: {count}' for itype, count in sorted(instance_type_counts.items())])
                        parts.append(f'instances: {instance_str}')
                    _write('')  # blank line between query cycles
                    _write('\n'.join(parts))
                    last_status_print = current_time

                # update per-config progress bars
                for name in config_names:
                    succeeded = sum(1 for jid in config_jobs[name]
                                    if job_statuses_map.get(jid) == 'SUCCEEDED')
                    delta = succeeded - config_done[name]
                    if delta > 0:
                        _clear_heartbeat()
                        config_pbars[name].update(delta)
                        config_done[name] = succeeded
                
                # immediately resubmit OOM / spot failures
                resubmit_reasons = {}
                if newly_failed_jobs:
                    resubmitted, resubmit_reasons, resubmit_msgs, perm_failed = \
                        self._resubmit_failed_jobs(
                            newly_failed_jobs, job_info_map)
                    for msg in resubmit_msgs:
                        _write(msg)
                    permanently_failed_jobs.extend(perm_failed)
                    resubmitted_total += len(resubmit_reasons)
                    resubmitted_job_ids.update(resubmit_reasons.keys())
                    # also count permanently failed OOM jobs as "resubmitted"
                    # for progress bar accounting (they won't succeed)
                    perm_failed_ids = {j['jobId'] for j in perm_failed}
                    resubmitted_job_ids.update(perm_failed_ids)
                    resubmitted_total += len(perm_failed)
                    if resubmitted:
                        job_ids.extend(resubmitted)
                        # add resubmitted jobs to their config group
                        for new_jid in resubmitted:
                            if new_jid in job_info_map and 'run_id' in job_info_map[new_jid]:
                                rid = job_info_map[new_jid]['run_id']
                            else:
                                rid = 'jobs'
                            config_jobs.setdefault(rid, set()).add(new_jid)
                            config_resubmitted[rid] = config_resubmitted.get(rid, 0) + 1
                            if rid in config_pbars:
                                config_pbars[rid].total = len(config_jobs[rid]) - config_resubmitted[rid]
                                _clear_heartbeat()
                                config_pbars[rid].refresh()

                # print failed jobs (resubmitted jobs get a softer message)
                for job in newly_failed_jobs:
                    job_id = job['jobId']
                    reason = resubmit_reasons.get(job_id)
                    if reason == 'oom':
                        _write(f'⟳ OOM: {job["jobName"]} — resubmitting with more memory')
                    elif reason == 'spot':
                        _write(f'⟳ SPOT: {job["jobName"]} — resubmitting (instance reclaimed)')
                    else:
                        _write(f'✗ FAILED: {job["jobName"]} ({job_id[:8]}...)')
                        _write(f'  Reason: {job["statusReason"]}')
                        container = job.get('container', {})
                        if 'reason' in container:
                            _write(f'  Container: {container["reason"]}')
                        if 'exitCode' in container:
                            _write(f'  Exit Code: {container["exitCode"]}')
                        if 'logStreamName' in container:
                            log_stream = container["logStreamName"]
                            _write(f'  Logs: aws logs get-log-events --log-group-name /aws/batch/job --log-stream-name {log_stream} --limit 50 --output text | tail -30')

                # verify we got responses for all requested jobs
                if first_check and jobs_found != len(job_ids):
                    _write(f'⚠ Warning: Requested {len(job_ids)} jobs, AWS returned {jobs_found}')
                    first_check = False
                
                # download results for newly completed jobs (parallel)
                from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
                download_targets = []
                for job in completed_jobs:
                    job_id = job['jobId']
                    if job_id not in job_info_map:
                        continue
                    info = job_info_map[job_id]
                    if info.get('output_folder'):
                        download_targets.append((job_id, job, info))
                    elif info.get('on_complete'):
                        try:
                            info['on_complete'](job)
                            downloaded_jobs.add(job_id)
                        except Exception as e:
                            _write(f'  ⚠ on_complete failed for {job["jobName"]}: {e}')

                if download_targets:
                    def _dl(job_id, job, info):
                        return job_id, job['jobName'], \
                            self._download_single_experiment_result(
                                info['run_id'], info['exp_idx'],
                                info['output_folder'])

                    with ThreadPoolExecutor(max_workers=16) as dl_pool:
                        futs = {dl_pool.submit(_dl, *t): t[0]
                                for t in download_targets}
                        for fut in _as_completed(futs):
                            try:
                                jid, jname, n_files = fut.result()
                                downloaded_jobs.add(jid)
                                _write(f'  ✓ Downloaded {jname} ({n_files} file(s))')
                            except Exception as e:
                                jid = futs[fut]
                                _write(f'⚠ Error downloading {jid[:8]}...: {e}')
            
                # check if all jobs reached a terminal state
                # (skip if we just resubmitted -- counts are stale)
                if all_terminal == total and not resubmit_reasons:
                    _clear_heartbeat()
                    for pb in config_pbars.values():
                        pb.close()
                    n_perm_failed = len(permanently_failed_jobs)
                    print(f'\nall jobs complete!')
                    print(f'  succeeded: {statuses["SUCCEEDED"]}')
                    print(f'  failed: {statuses["FAILED"]}')
                    if n_perm_failed:
                        print(f'  permanently failed (max OOM): {n_perm_failed}')
                    print(f'  results downloaded: {len(downloaded_jobs)}/{statuses["SUCCEEDED"]}')
                    print(f'  total vCPU-hours: {vcpu_hours:.2f}')

                    # print permanently failed OOM jobs
                    if permanently_failed_jobs:
                        print(f'\n{"="*60}')
                        print('PERMANENTLY FAILED (exceeded max memory tier):')
                        print(f'{"="*60}')
                        for i, job in enumerate(permanently_failed_jobs, 1):
                            print(f'  {i}. {job["jobName"]}: {job["reason"]}')

                    # print detailed failure reasons (exclude resubmitted
                    # and permanently-failed jobs)
                    perm_failed_ids = {j['jobId']
                                       for j in permanently_failed_jobs}
                    unresolved = [j for j in failed_jobs
                                  if j['jobId'] not in resubmitted_job_ids
                                  and j['jobId'] not in perm_failed_ids]
                    if unresolved:
                        print(f'\n{"="*60}')
                        print('FAILURE DETAILS:')
                        print(f'{"="*60}')
                        for i, job in enumerate(unresolved, 1):
                            print(f'\n{i}. Job: {job["jobName"]} ({job["jobId"]})')
                            print(f'   Reason: {job["statusReason"]}')
                            container = job.get('container', {})
                            if 'reason' in container:
                                print(f'   Container: {container["reason"]}')
                            if 'exitCode' in container:
                                print(f'   Exit Code: {container["exitCode"]}')
                            if 'logStreamName' in container:
                                log_stream = container["logStreamName"]
                                print(f'   Logs: aws logs get-log-events --log-group-name /aws/batch/job --log-stream-name {log_stream} --limit 50 --output text | tail -30')

                    break

                # heartbeat: tick every second during poll sleep
                heartbeat_interval = 1
                slept = 0
                while slept < poll_interval:
                    nap = min(heartbeat_interval, poll_interval - slept)
                    time.sleep(nap)
                    slept += nap
                    ts = datetime.now().strftime('%H:%M:%S')
                    print(f'\r[{ts}]', end='', flush=True)
        except KeyboardInterrupt:
            _clear_heartbeat()
            for pb in config_pbars.values():
                pb.close()
            print('\n\nMonitoring interrupted by user')
            print('cancelling AWS jobs')
            cancel_summary = self.cancel_jobs(job_ids, reason='Monitoring interrupted by user')
            print(f'✓ Canceled: {cancel_summary["canceled"]}')
            if cancel_summary['failed']:
                print(f'✗ Failed to cancel: {cancel_summary["failed"]}')
            raise
        except Exception as e:
            _clear_heartbeat()
            for pb in config_pbars.values():
                pb.close()
            if cancel_on_error:
                print('\n\nMonitoring failed; cancelling AWS jobs')
                cancel_summary = self.cancel_jobs(job_ids, reason='Monitoring error')
                print(f'✓ Canceled: {cancel_summary["canceled"]}')
                if cancel_summary['failed']:
                    print(f'✗ Failed to cancel: {cancel_summary["failed"]}')
            raise

        return permanently_failed_jobs

    def cancel_jobs(self, job_ids, reason='Canceled by user'):
        """cancel AWS Batch jobs, returning counts of canceled/failed/skipped."""
        if not job_ids:
            return {'canceled': 0, 'failed': 0, 'skipped': 0}

        canceled = 0
        failed = 0
        skipped = 0

        # describe in chunks of 100 (AWS limit)
        for i in range(0, len(job_ids), 100):
            chunk = job_ids[i:i+100]
            try:
                response = self.batch.describe_jobs(jobs=chunk)
                jobs = response.get('jobs', [])
            except Exception:
                jobs = []

            # map jobId -> status if we can
            status_map = {job.get('jobId'): job.get('status') for job in jobs}

            for job_id in chunk:
                status = status_map.get(job_id)
                if status in ['SUCCEEDED', 'FAILED']:
                    skipped += 1
                    continue
                try:
                    self.batch.terminate_job(jobId=job_id, reason=reason)
                    canceled += 1
                except Exception:
                    failed += 1

        return {'canceled': canceled, 'failed': failed, 'skipped': skipped}

    @contextmanager
    def cancel_on_exit(self, job_ids: list, reason: str = 'Canceled on exit'):
        """context manager to cancel jobs on exit"""
        try:
            yield
        except KeyboardInterrupt:
            print('\n\nMonitoring interrupted by user')
            print('cancelling AWS jobs')
            summary = self.cancel_jobs(job_ids, reason=reason)
            print(f'✓ Canceled: {summary["canceled"]}')
            if summary['failed']:
                print(f'✗ Failed to cancel: {summary["failed"]}')
            raise
        except Exception:
            summary = self.cancel_jobs(job_ids, reason=reason)
            print(f'✓ Canceled: {summary["canceled"]}')
            if summary['failed']:
                print(f'✗ Failed to cancel: {summary["failed"]}')
            raise
    
    def get_failure_details(self, job_ids):
        """return detailed failure information for failed jobs.
        
        Args:
            job_ids: list of AWS Batch job IDs to check
        
        Returns:
            list of dicts with failure details
        """
        failed_jobs = []
        
        # batch describe in chunks of 100
        for i in range(0, len(job_ids), 100):
            chunk = job_ids[i:i+100]
            try:
                response = self.batch.describe_jobs(jobs=chunk)
                for job in response['jobs']:
                    if job['status'] == 'FAILED':
                        failed_jobs.append({
                            'jobId': job['jobId'],
                            'jobName': job['jobName'],
                            'statusReason': job.get('statusReason', 'Unknown'),
                            'container': job.get('container', {}),
                            'createdAt': job.get('createdAt'),
                            'stoppedAt': job.get('stoppedAt')
                        })
            except ClientError as e:
                print(f'error checking jobs: {e}')
                continue
        
        # print detailed failure reasons
        if failed_jobs:
            print(f'\n{"="*60}')
            print('FAILURE DETAILS:')
            print(f'{"="*60}')
            for i, job in enumerate(failed_jobs, 1):
                print(f'\n{i}. Job: {job["jobName"]} ({job["jobId"]})')
                print(f'   Reason: {job["statusReason"]}')
                container = job.get('container', {})
                if 'reason' in container:
                    print(f'   Container: {container["reason"]}')
                if 'exitCode' in container:
                    print(f'   Exit Code: {container["exitCode"]}')
                if 'logStreamName' in container:
                    log_stream = container["logStreamName"]
                    print(f'   Logs: aws logs get-log-events --log-group-name /aws/batch/job --log-stream-name {log_stream} --limit 50 --output text | tail -30')
        else:
            print('no failed jobs found')
        
        return failed_jobs
    
    def download_results(self, experiment_id, n_perm, output_dir):
        """download permutation results from S3 to a local directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        prefix = f'{self.config.s3_prefix}/results/{experiment_id}/'
        
        print(f'downloading results from s3://{self.config.s3_bucket}/{prefix}')
        
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.config.s3_bucket,
                                           Prefix=prefix):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    if not key.endswith('_result.pkl'):
                        continue
                    
                    # download and deserialize
                    try:
                        response = self.s3.get_object(
                            Bucket=self.config.s3_bucket,
                            Key=key
                        )
                        data = pickle.loads(response['Body'].read())
                        perm_idx = data['perm_idx']
                        results[perm_idx] = data
                        
                        # also save locally
                        local_path = output_dir / Path(key).name
                        with open(local_path, 'wb') as f:
                            pickle.dump(data, f)
                    except Exception as e:
                        print(f'error downloading {key}: {e}')
        except ClientError as e:
            raise RuntimeError(f'failed to download results: {e}')
        
        print(f'downloaded {len(results)} results to {output_dir}')
        
        # check for missing results
        missing = set(range(n_perm + 1)) - set(results.keys())
        if missing:
            print(f'warning: missing results for permutations: {sorted(missing)}')
        
        return results
    
    # experiment-level execution methods (new architecture)
    
    def upload_config(self, config, run_id: str):
        """upload config object to S3 for experiment-level execution
        
        Args:
            config: Config object to upload
            run_id: unique run identifier
        """
        config_key = f'{self.config.s3_prefix}/{run_id}/config.pkl'
        config_bytes = pickle.dumps(config)
        
        try:
            self.s3.put_object(
                Bucket=self.config.s3_bucket,
                Key=config_key,
                Body=config_bytes
            )
            print(f'  uploaded config to s3://{self.config.s3_bucket}/{config_key}')
        except ClientError as e:
            raise RuntimeError(f'failed to upload config: {e}')
    
    def upload_all_kwargs(self, run_id: str, kwargs_list: List[Tuple[int, Dict[str, Any]]]):
        """Upload all experiment kwargs as a single S3 object.

        Args:
            run_id: unique run identifier
            kwargs_list: list of (exp_idx, kwargs) pairs
        """
        kwargs_dict = {exp_idx: kw for exp_idx, kw in kwargs_list}
        key = f'{self.config.s3_prefix}/{run_id}/all_kwargs.pkl'
        try:
            self.s3.put_object(
                Bucket=self.config.s3_bucket,
                Key=key,
                Body=pickle.dumps(kwargs_dict),
            )
        except ClientError as e:
            raise RuntimeError(f'failed to upload kwargs: {e}')

    def submit_experiment_job(self, run_id: str, exp_idx: int,
                             memory_mb: Optional[int] = None,
                             timeout_minutes: Optional[int] = None) -> str:
        """Submit a single experiment job to AWS Batch.

        Kwargs are read by the worker from the bulk ``all_kwargs.pkl`` file
        uploaded via :meth:`upload_all_kwargs`.

        Args:
            run_id: unique run identifier
            exp_idx: experiment index
            memory_mb: optional memory override for OOM tier escalation
            timeout_minutes: per-job timeout override (falls back to
                ``self.config.timeout_minutes``)

        Returns:
            job_id: AWS Batch job ID
        """
        if timeout_minutes is None:
            timeout_minutes = self.config.timeout_minutes

        job_name = f'glow_{run_id}_exp{exp_idx:06d}'
        overrides = {
            'command': [
                '--s3-bucket', self.config.s3_bucket,
                '--s3-prefix', self.config.s3_prefix,
                '--run-id', run_id,
                '--exp-idx', str(exp_idx)
            ]
        }
        if memory_mb is not None:
            overrides['resourceRequirements'] = [
                {'type': 'VCPU', 'value': str(self.config.vcpus)},
                {'type': 'MEMORY', 'value': str(memory_mb)},
            ]
            self._job_memory_tier_index[job_name] = 0

        max_retries = 8
        for attempt in range(max_retries):
            try:
                response = self.batch.submit_job(
                    jobName=job_name,
                    jobQueue=self.config.job_queue,
                    jobDefinition=self.config.job_definition,
                    timeout={'attemptDurationSeconds': timeout_minutes * 60},
                    retryStrategy={'attempts': self.config.retry_attempts},
                    containerOverrides=overrides
                )
                return response['jobId']
            except ClientError as e:
                if 'TooManyRequestsException' in str(e) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt * 0.5)
                    continue
                raise RuntimeError(f'failed to submit job: {e}')
    
    def submit_array_job(self, job_name, command_template, indices, index_arg,
                         memory_mb=None, timeout_minutes=None):
        """Submit an AWS Batch array job (one API call creates N child jobs).

        Args:
            job_name: base name for the job
            command_template: list of CLI args common to all children
            indices: list of actual indices to run (may have gaps)
            index_arg: CLI flag name (e.g. '--exp-idx' or '--perm-idx')
            memory_mb: optional memory override
            timeout_minutes: optional per-job timeout override

        Returns:
            dict with 'parent_job_ids', 'child_job_ids', 'index_map'
        """
        if timeout_minutes is None:
            timeout_minutes = self.config.timeout_minutes

        if len(indices) == 0:
            return {'parent_job_ids': [], 'child_job_ids': [], 'index_map': {}}

        # single index: fall back to regular submit (array size must be >= 2)
        if len(indices) == 1:
            idx = indices[0]
            if isinstance(idx, list):
                # batch mode: comma-separated perm indices
                command = list(command_template) + [
                    index_arg, ','.join(str(x) for x in idx)]
            else:
                command = list(command_template) + [index_arg, str(idx)]
            overrides = {'command': command}
            if memory_mb is not None:
                overrides['resourceRequirements'] = [
                    {'type': 'VCPU', 'value': str(self.config.vcpus)},
                    {'type': 'MEMORY', 'value': str(memory_mb)},
                ]
            max_retries = 8
            for attempt in range(max_retries):
                try:
                    response = self.batch.submit_job(
                        jobName=job_name,
                        jobQueue=self.config.job_queue,
                        jobDefinition=self.config.job_definition,
                        containerOverrides=overrides,
                        retryStrategy={'attempts': self.config.retry_attempts},
                        timeout={'attemptDurationSeconds': timeout_minutes * 60},
                    )
                    job_id = response['jobId']
                    if memory_mb is not None:
                        base = self._base_job_name(job_name)
                        tiers = self.config.oom_memory_mb_tiers or []
                        for idx, t in enumerate(tiers):
                            if t >= memory_mb:
                                self._job_memory_tier_index[base] = idx
                                break
                    return {
                        'parent_job_ids': [job_id],
                        'child_job_ids': [job_id],
                        'index_map': {job_id: indices[0]},
                    }
                except ClientError as e:
                    if 'TooManyRequestsException' in str(e) and attempt < max_retries - 1:
                        time.sleep(2 ** attempt * 0.5)
                        continue
                    raise RuntimeError(f'failed to submit job: {e}')

        # chunk into array jobs of at most 10000
        max_array_size = 10000
        chunks = [indices[i:i + max_array_size]
                  for i in range(0, len(indices), max_array_size)]

        parent_job_ids = []
        child_job_ids = []
        index_map = {}

        for chunk_idx, chunk in enumerate(chunks):
            # upload index map to S3
            map_data = {'index_arg': index_arg, 'indices': chunk}
            chunk_suffix = f'_chunk{chunk_idx}' if len(chunks) > 1 else ''
            map_key = (f'{self.config.s3_prefix}/{job_name}/'
                       f'array_index_map{chunk_suffix}.json')
            self.s3.put_object(
                Bucket=self.config.s3_bucket,
                Key=map_key,
                Body=json.dumps(map_data),
            )

            command = list(command_template) + ['--index-map', map_key]
            overrides = {'command': command}
            if memory_mb is not None:
                overrides['resourceRequirements'] = [
                    {'type': 'VCPU', 'value': str(self.config.vcpus)},
                    {'type': 'MEMORY', 'value': str(memory_mb)},
                ]

            array_job_name = (f'{job_name}{chunk_suffix}'
                              if len(chunks) > 1 else job_name)

            max_retries = 8
            for attempt in range(max_retries):
                try:
                    response = self.batch.submit_job(
                        jobName=array_job_name,
                        jobQueue=self.config.job_queue,
                        jobDefinition=self.config.job_definition,
                        containerOverrides=overrides,
                        arrayProperties={'size': len(chunk)},
                        retryStrategy={'attempts': self.config.retry_attempts},
                        timeout={'attemptDurationSeconds': timeout_minutes * 60},
                    )
                    parent_id = response['jobId']
                    parent_job_ids.append(parent_id)

                    if memory_mb is not None:
                        base = self._base_job_name(array_job_name)
                        tiers = self.config.oom_memory_mb_tiers or []
                        for idx, t in enumerate(tiers):
                            if t >= memory_mb:
                                self._job_memory_tier_index[base] = idx
                                break

                    for pos, actual_idx in enumerate(chunk):
                        child_id = f'{parent_id}:{pos}'
                        child_job_ids.append(child_id)
                        index_map[child_id] = actual_idx
                    break
                except ClientError as e:
                    if 'TooManyRequestsException' in str(e) and attempt < max_retries - 1:
                        time.sleep(2 ** attempt * 0.5)
                        continue
                    raise RuntimeError(f'failed to submit array job: {e}')

        return {
            'parent_job_ids': parent_job_ids,
            'child_job_ids': child_job_ids,
            'index_map': index_map,
        }

    def _download_single_experiment_result(self, run_id: str, exp_idx: int, output_folder: Path):
        """Download the tar.gz archive for one experiment and extract locally.

        Args:
            run_id: unique run identifier
            exp_idx: experiment index
            output_folder: local folder to extract results into

        Returns:
            file_count: number of files extracted
        """
        import io
        import tarfile

        s3_key = f'{self.config.s3_prefix}/{run_id}/results/{exp_idx:06d}.tar.gz'

        try:
            response = self.s3.get_object(
                Bucket=self.config.s3_bucket, Key=s3_key)
            buf = io.BytesIO(response['Body'].read())

            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=buf, mode='r:gz') as tar:
                tar.extractall(path=str(output_folder))
                file_count = len(tar.getmembers())

            # delete archive from S3
            try:
                self.s3.delete_object(
                    Bucket=self.config.s3_bucket, Key=s3_key)
            except ClientError:
                pass

        except ClientError as e:
            raise RuntimeError(f'failed to download results: {e}')

        return file_count
    
    def download_experiment_results(self, run_id: str, output_folder: Path):
        """Download all experiment tar.gz archives for a run in parallel.

        Args:
            run_id: unique run identifier
            output_folder: local folder to extract results into
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        result_prefix = f'{self.config.s3_prefix}/{run_id}/results/'
        print(f'downloading results from s3://{self.config.s3_bucket}/{result_prefix}')

        # list all .tar.gz archives
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.config.s3_bucket, Prefix=result_prefix)

            tar_keys = []
            for page in pages:
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.tar.gz'):
                        tar_keys.append(obj['Key'])
        except ClientError as e:
            raise RuntimeError(f'failed to list results: {e}')

        if not tar_keys:
            print('  no results found')
            return

        output_folder = Path(output_folder)
        file_count = 0

        def _download_one(s3_key):
            exp_tag = s3_key.rsplit('/', 1)[-1].replace('.tar.gz', '')
            exp_idx = int(exp_tag)
            return self._download_single_experiment_result(
                run_id, exp_idx, output_folder)

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(_download_one, k) for k in tar_keys]
            for fut in as_completed(futures):
                file_count += fut.result()

        print(f'  downloaded {file_count} files from {len(tar_keys)} archives to {output_folder}')
