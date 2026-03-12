"""AWS Batch integration for parallel permutation processing."""

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
        return job_name.rsplit('_retry', 1)[0]

    @staticmethod
    def _is_spot_termination(job_failure: Dict[str, Any]) -> bool:
        status_reason = (job_failure.get('statusReason') or '').lower()
        return 'host ec2' in status_reason and 'terminated' in status_reason

    @staticmethod
    def _is_oom_failure(job_failure: Dict[str, Any]) -> bool:
        status_reason = (job_failure.get('statusReason') or '').lower()
        container = job_failure.get('container', {}) or {}
        container_reason = (container.get('reason') or '').lower()
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
        resubmitted with the same resource requirements.

        Returns:
            resubmitted (list): new job IDs
            resubmit_reasons (dict): maps source job ID -> 'oom' | 'spot'
        """
        tiers = self.config.oom_memory_mb_tiers or []

        resubmitted = []
        resubmit_reasons = {}
        for job in failed_jobs:
            # determine failure type
            is_oom = self._is_oom_failure(job)
            is_spot = (not is_oom) and self._is_spot_termination(job)
            if not (is_oom or is_spot):
                continue

            base_name = self._base_job_name(job['jobName'])
            container = job.get('container', {}) or {}
            command = container.get('command')
            if not command:
                print(f'  ⚠ Cannot resubmit {job["jobName"]}: missing command')
                continue

            if is_oom:
                current_idx = self._job_memory_tier_index.get(base_name, -1)
                next_idx = current_idx + 1
                if next_idx >= len(tiers) or len(tiers) < 2:
                    max_mb = tiers[-1] if tiers else self.config.memory_mb
                    raise MemoryError(
                        f'FATAL OOM: {job["jobName"]} exceeded max memory '
                        f'tier ({max_mb} MB). Reduce experiment size or '
                        f'increase oom_memory_mb_tiers.')
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
                    print(f'  ↻ {job["jobName"]} → {memory_mb} MB ({retry_name})')
                else:
                    self._job_spot_retry_count[base_name] = \
                        self._job_spot_retry_count.get(base_name, 0) + 1
                    print(f'  ↻ {job["jobName"]} → resubmitted ({retry_name})')
            except ClientError as e:
                print(f'  ✗ Failed to resubmit {job["jobName"]}: {e}')

        return resubmitted, resubmit_reasons
    
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
    
    def submit_jobs(self, experiment_id, n_perm, skip_completed=True,
                    memory_mb=None):
        """submit permutation jobs to AWS Batch.
        
        Args:
            experiment_id: experiment identifier
            n_perm: number of permutations (0 to n_perm inclusive)
            skip_completed: skip permutations that already have results
            memory_mb: override memory per job (None = auto-estimate from
                       experiment dimensions, or job definition default)
        
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
        n_jobs = len(perm_indices)
        
        if n_jobs == 0:
            print('all permutations already completed')
            return {'n_jobs': 0, 'job_ids': [], 'skipped': list(completed)}
        
        print(f'\n{"="*60}')
        print('AWS BATCH JOB SUBMISSION')
        print(f'{"="*60}')
        print(f'experiment: {experiment_id}')
        print(f'jobs to submit: {n_jobs}')
        print(f'skipped (completed): {len(completed)}')
        print(f'timeout: {self.config.timeout_minutes} min per job')
        print(f'{"="*60}\n')
        
        # submit jobs
        job_ids = []
        data_path = f's3://{self.config.s3_bucket}/{self.config.s3_prefix}/experiments/{experiment_id}/data.pkl'
        
        for perm_idx in perm_indices:
            job_name = f'{experiment_id}_perm_{perm_idx}'
            
            try:
                overrides = {
                    'command': [
                        '--data-path', data_path,
                        '--perm-idx', str(perm_idx),
                        '--s3-bucket', self.config.s3_bucket,
                        '--s3-prefix', self.config.s3_prefix,
                        '--experiment-id', experiment_id
                    ]
                }
                if memory_mb is not None:
                    overrides['resourceRequirements'] = [
                        {'type': 'VCPU', 'value': str(self.config.vcpus)},
                        {'type': 'MEMORY', 'value': str(memory_mb)},
                    ]
                
                response = self.batch.submit_job(
                    jobName=job_name,
                    jobQueue=self.config.job_queue,
                    jobDefinition=self.config.job_definition,
                    containerOverrides=overrides,
                    retryStrategy={'attempts': self.config.retry_attempts},
                    timeout={'attemptDurationSeconds': self.config.timeout_minutes * 60}
                )
                job_ids.append(response['jobId'])
                if memory_mb is not None:
                    tiers = self.config.oom_memory_mb_tiers or []
                    for idx, t in enumerate(tiers):
                        if t >= memory_mb:
                            self._job_memory_tier_index[job_name] = idx
                            break
            except ClientError as e:
                print(f'error submitting job for perm {perm_idx}: {e}')
        
        if memory_mb is not None:
            print(f'submitted {len(job_ids)} jobs (memory: {memory_mb} MB)')
        else:
            print(f'submitted {len(job_ids)} jobs')
        
        return {
            'n_jobs': n_jobs,
            'job_ids': job_ids,
            'perm_indices': perm_indices,
            'skipped': list(completed)
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
        """poll AWS Batch until all jobs finish, downloading results as they complete."""
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
        first_check = True  # track if this is the first instance type check
        had_running_jobs = False  # track if we've seen running jobs (for immediate check)
        downloaded_jobs = set()  # track which jobs have been downloaded
        start_time = time.time()
        last_done_count = 0
        last_done_time = start_time
        previous_done = 0
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
        
        # create progress bar
        pbar = tqdm(total=len(job_ids), desc='Jobs', unit='job', 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                   leave=True)
        
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
                        print(f'error checking jobs: {e}')
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
                                print(f'  Note: Could not find job queue: {self.config.job_queue}')
                            raise ValueError("Job queue not found")
                            
                        compute_envs = queue_info['jobQueues'][0].get('computeEnvironmentOrder', [])
                        if not compute_envs:
                            if first_check:
                                print(f'  Note: Job queue has no compute environments')
                            raise ValueError("No compute environments")
                            
                        compute_env_name = compute_envs[0].get('computeEnvironment')
                        
                        # get ECS cluster name from compute environment
                        env_info = self.batch.describe_compute_environments(
                            computeEnvironments=[compute_env_name]
                        )
                        if not env_info.get('computeEnvironments'):
                            if first_check:
                                print(f'  Note: Could not find compute environment: {compute_env_name}')
                            raise ValueError("Compute environment not found")
                            
                        ecs_cluster_arn = env_info['computeEnvironments'][0].get('ecsClusterArn', '')
                        if not ecs_cluster_arn:
                            if first_check:
                                print(f'  Note: Compute environment has no ECS cluster')
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
                            print(f'  Note: No container instances found in ECS cluster (jobs may not have started yet)')
                        
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
                            print(f'  Note: Could not fetch instance types: {instance_type_error}')
                            print('  (This is optional - jobs will still run normally)')
                    
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
                
                # print status update in log style (simple, no overwriting)
                # only print if significant change or enough time has passed
                should_print = (current_time - last_status_print >= poll_interval or 
                               done > last_done_count or 
                               len(newly_failed_jobs) > 0 or
                               len(completed_jobs) > 0)
                
                if should_print:
                    # build state string on one line
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
                    actual_failed = failed - resubmitted_total
                    if resubmitted_total > 0:
                        state_parts.append(f'Resubmitted: {resubmitted_total}')
                    if actual_failed > 0:
                        state_parts.append(f'Failed: {actual_failed}')
                    
                    state_str = ', '.join(state_parts) if state_parts else 'All done'
                    
                    timestamp_str = datetime.now().strftime('%H:%M:%S')
                    parts = [f'  [{timestamp_str}] {state_str}']
                    parts.append(f'  vCPU-hours: {vcpu_hours:.3f}')
                    if instance_type_counts:
                        instance_str = ', '.join([f'{itype}: {count}' for itype, count in sorted(instance_type_counts.items())])
                        parts.append(f'  instances: {instance_str}')
                    pbar.write('\n'.join(parts))
                    last_status_print = current_time
                
                # update progress bar
                if done > previous_done:
                    pbar.update(done - previous_done)
                    previous_done = done
                
                # update progress bar postfix
                postfix_parts = []
                if pending > 0:
                    postfix_parts.append(f'pending={pending}')
                if running > 0:
                    postfix_parts.append(f'running={running}')
                if completed > 0:
                    postfix_parts.append(f'completed={completed}')
                actual_failed = failed - resubmitted_total
                if resubmitted_total > 0:
                    postfix_parts.append(f'resubmitted={resubmitted_total}')
                if actual_failed > 0:
                    postfix_parts.append(f'failed={actual_failed}')
                postfix_parts.append(f'eta={eta_str}')
                
                postfix_str = ', '.join(postfix_parts)
                pbar.set_postfix_str(postfix_str)
                
                # immediately resubmit OOM / spot failures
                resubmit_reasons = {}
                if newly_failed_jobs:
                    resubmitted, resubmit_reasons = \
                        self._resubmit_failed_jobs(
                            newly_failed_jobs, job_info_map)
                    resubmitted_total += len(resubmit_reasons)
                    resubmitted_job_ids.update(resubmit_reasons.keys())
                    if resubmitted:
                        job_ids.extend(resubmitted)

                # print failed jobs (resubmitted jobs get a softer message)
                for job in newly_failed_jobs:
                    job_id = job['jobId']
                    reason = resubmit_reasons.get(job_id)
                    if reason == 'oom':
                        print(f'\n⟳ OOM: {job["jobName"]} — resubmitting with more memory')
                    elif reason == 'spot':
                        print(f'\n⟳ SPOT: {job["jobName"]} — resubmitting (instance reclaimed)')
                    else:
                        print(f'\n✗ FAILED: {job["jobName"]} ({job_id[:8]}...)')
                        print(f'  Reason: {job["statusReason"]}')
                        container = job.get('container', {})
                        if 'reason' in container:
                            print(f'  Container: {container["reason"]}')
                        if 'exitCode' in container:
                            print(f'  Exit Code: {container["exitCode"]}')
                        if 'logStreamName' in container:
                            log_stream = container["logStreamName"]
                            print(f'  Logs: aws logs get-log-events --log-group-name /aws/batch/job --log-stream-name {log_stream} --limit 50 --output text | tail -30')

                # verify we got responses for all requested jobs
                if first_check and jobs_found != len(job_ids):
                    print(f'\n⚠ Warning: Requested {len(job_ids)} jobs, AWS returned {jobs_found}')
                    first_check = False
                
                # download results for newly completed jobs and print immediately
                for job in completed_jobs:
                    job_id = job['jobId']
                    if job_id in job_info_map:
                        info = job_info_map[job_id]
                        if info.get('output_folder'):
                            try:
                                output_folder = Path(info['output_folder'])
                                print(f'\n[Downloading] {job["jobName"]} → {output_folder}')
                                
                                n_files = self._download_single_experiment_result(
                                    info['run_id'], info['exp_idx'], info['output_folder']
                                )
                                downloaded_jobs.add(job_id)
                                print(f'  ✓ Downloaded {n_files} file(s)')
                            except Exception as e:
                                print(f'\n⚠ Error downloading results for {job["jobName"]}: {e}')
                        elif info.get('on_complete'):
                            try:
                                info['on_complete'](job)
                                downloaded_jobs.add(job_id)
                            except Exception as e:
                                print(f'\n  ⚠ on_complete failed for {job["jobName"]}: {e}')
            
                # check if all jobs reached a terminal state
                # (skip if we just resubmitted -- counts are stale)
                if all_terminal == total and not resubmit_reasons:
                    pbar.close()
                    print(f'\nall jobs complete!')
                    print(f'  succeeded: {statuses["SUCCEEDED"]}')
                    print(f'  failed: {statuses["FAILED"]}')
                    print(f'  results downloaded: {len(downloaded_jobs)}/{statuses["SUCCEEDED"]}')
                    print(f'  total vCPU-hours: {vcpu_hours:.2f}')
                    
                    # print detailed failure reasons (exclude resubmitted jobs)
                    unresolved = [j for j in failed_jobs
                                  if j['jobId'] not in resubmitted_job_ids]
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
            
                heartbeat_interval = 1
                slept = 0
                while slept < poll_interval:
                    nap = min(heartbeat_interval, poll_interval - slept)
                    time.sleep(nap)
                    slept += nap
                    ts = datetime.now().strftime('%H:%M:%S')
                    pbar.set_description(f'Jobs [{ts}]')
                    pbar.refresh()
        except KeyboardInterrupt:
            pbar.close()
            print('\n\nMonitoring interrupted by user')
            print('cancelling AWS jobs')
            cancel_summary = self.cancel_jobs(job_ids, reason='Monitoring interrupted by user')
            print(f'✓ Canceled: {cancel_summary["canceled"]}')
            if cancel_summary['failed']:
                print(f'✗ Failed to cancel: {cancel_summary["failed"]}')
            raise
        except Exception as e:
            pbar.close()
            if cancel_on_error:
                print('\n\nMonitoring failed; cancelling AWS jobs')
                cancel_summary = self.cancel_jobs(job_ids, reason='Monitoring error')
                print(f'✓ Canceled: {cancel_summary["canceled"]}')
                if cancel_summary['failed']:
                    print(f'✗ Failed to cancel: {cancel_summary["failed"]}')
            raise

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
    
    def submit_experiment_job(self, run_id: str, exp_idx: int, kwargs: Dict[str, Any],
                             memory_mb: Optional[int] = None) -> str:
        """submit a single experiment job to AWS Batch
        
        Each job runs a full experiment with all permutations serially.
        This amortizes container overhead across all permutations.
        
        Args:
            run_id: unique run identifier
            exp_idx: experiment index
            kwargs: experiment kwargs dict (seed, effect_llr, etc.)
            memory_mb: optional memory in MB; if set, overrides job definition default
                       and is recorded for OOM tier escalation.
        
        Returns:
            job_id: AWS Batch job ID
        """
        # upload kwargs
        kwargs_key = f'{self.config.s3_prefix}/{run_id}/kwargs/{exp_idx:06d}.pkl'
        kwargs_bytes = pickle.dumps(kwargs)
        
        try:
            self.s3.put_object(
                Bucket=self.config.s3_bucket,
                Key=kwargs_key,
                Body=kwargs_bytes
            )
        except ClientError as e:
            raise RuntimeError(f'failed to upload kwargs: {e}')
        
        # submit job
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

        try:
            response = self.batch.submit_job(
                jobName=job_name,
                jobQueue=self.config.job_queue,
                jobDefinition=self.config.job_definition,
                timeout={'attemptDurationSeconds': self.config.timeout_minutes * 60},
                retryStrategy={'attempts': self.config.retry_attempts},
                containerOverrides=overrides
            )
            
            return response['jobId']
        except ClientError as e:
            raise RuntimeError(f'failed to submit job: {e}')
    
    def _download_single_experiment_result(self, run_id: str, exp_idx: int, output_folder: Path):
        """download results for a single experiment and delete from S3
        
        Args:
            run_id: unique run identifier
            exp_idx: experiment index
            output_folder: local folder to save results
        """
        result_prefix = f'{self.config.s3_prefix}/{run_id}/results/{exp_idx:06d}/'
        
        # list objects for this experiment
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.config.s3_bucket,
                Prefix=result_prefix
            )
            
            s3_keys_to_delete = []
            file_count = 0
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    relative_path = s3_key[len(result_prefix):]
                    local_path = output_folder / relative_path
                    
                    # create parent dirs
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # download file
                    self.s3.download_file(
                        self.config.s3_bucket,
                        s3_key,
                        str(local_path)
                    )
                    file_count += 1
                    s3_keys_to_delete.append(s3_key)
            
            # delete from S3 after successful download
            if s3_keys_to_delete:
                # delete in batches of 1000 (S3 limit)
                for i in range(0, len(s3_keys_to_delete), 1000):
                    batch = s3_keys_to_delete[i:i+1000]
                    delete_objects = [{'Key': key} for key in batch]
                    try:
                        self.s3.delete_objects(
                            Bucket=self.config.s3_bucket,
                            Delete={'Objects': delete_objects}
                        )
                    except ClientError as e:
                        # log but don't fail - results are already downloaded
                        print(f'  ⚠ Warning: Could not delete some S3 objects: {e}')
            
        except ClientError as e:
            raise RuntimeError(f'failed to download results: {e}')

        return file_count
    
    def download_experiment_results(self, run_id: str, output_folder: Path):
        """download all experiment results from S3 to local folder
        
        Args:
            run_id: unique run identifier
            output_folder: local folder to save results
        """
        result_prefix = f'{self.config.s3_prefix}/{run_id}/results/'
        
        print(f'downloading results from s3://{self.config.s3_bucket}/{result_prefix}')
        
        # list all objects
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.config.s3_bucket,
                Prefix=result_prefix
            )
            
            file_count = 0
            s3_keys_to_delete = []
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    relative_path = s3_key[len(result_prefix):]
                    local_path = output_folder / relative_path
                    
                    # create parent dirs
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # download file
                    self.s3.download_file(
                        self.config.s3_bucket,
                        s3_key,
                        str(local_path)
                    )
                    file_count += 1
                    s3_keys_to_delete.append(s3_key)
            
            # delete from S3 after successful download
            if s3_keys_to_delete:
                # delete in batches of 1000 (S3 limit)
                for i in range(0, len(s3_keys_to_delete), 1000):
                    batch = s3_keys_to_delete[i:i+1000]
                    delete_objects = [{'Key': key} for key in batch]
                    try:
                        self.s3.delete_objects(
                            Bucket=self.config.s3_bucket,
                            Delete={'Objects': delete_objects}
                        )
                    except ClientError as e:
                        # log but don't fail - results are already downloaded
                        print(f'  ⚠ Warning: Could not delete some S3 objects: {e}')
            
            print(f'  downloaded {file_count} files to {output_folder}')
        except ClientError as e:
            raise RuntimeError(f'failed to download results: {e}')
