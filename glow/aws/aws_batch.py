"""AWS Batch integration for parallel permutation processing"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

import boto3
import cloudpickle as pickle
import numpy as np
from botocore.exceptions import ClientError


@dataclass
class CloudConfig:
    """configuration for AWS cloud computing
    
    Attributes:
        s3_bucket: S3 bucket for storing data and results
        s3_prefix: prefix path within bucket (e.g., 'glow-experiments/run1')
        job_queue: AWS Batch job queue name
        job_definition: AWS Batch job definition ARN
        region: AWS region (e.g., 'us-east-1')
        max_concurrent_jobs: maximum number of jobs to run simultaneously (cost control)
        max_cost_per_hour: maximum estimated cost per hour (USD)
        use_spot: use spot instances for cost savings
        timeout_minutes: timeout per job in minutes
        memory_mb: memory allocation per job in MB
        vcpus: number of vCPUs per job
    """
    s3_bucket: str
    s3_prefix: str
    job_queue: str
    job_definition: str
    region: str = 'us-east-1'
    max_concurrent_jobs: int = 100
    max_cost_per_hour: float = 10.0  # USD
    use_spot: bool = True
    timeout_minutes: int = 60
    memory_mb: int = 4096
    vcpus: int = 2
    
    def to_dict(self):
        return asdict(self)


def estimate_cost(n_jobs: int, 
                  runtime_minutes: float,
                  memory_mb: int = 4096,
                  vcpus: int = 2,
                  use_spot: bool = True) -> Dict[str, float]:
    """estimate cost for running jobs on AWS
    
    Args:
        n_jobs: number of jobs to run
        runtime_minutes: estimated runtime per job in minutes
        memory_mb: memory per job in MB
        vcpus: number of vCPUs per job
        use_spot: use spot instances (cheaper but can be interrupted)
    
    Returns:
        dict with cost estimates:
            - cost_per_job: estimated cost per job (USD)
            - total_cost: total estimated cost (USD)
            - total_runtime_hours: total compute hours
            - cost_per_hour: average cost per hour
    """
    # rough AWS pricing (as of 2024, subject to change)
    # Fargate pricing: $0.04048 per vCPU per hour, $0.004445 per GB per hour
    
    # spot instances are ~70% cheaper
    spot_discount = 0.7 if use_spot else 1.0
    
    vcpu_cost_per_hour = 0.04048 * vcpus * spot_discount
    memory_gb = memory_mb / 1024
    memory_cost_per_hour = 0.004445 * memory_gb * spot_discount
    
    cost_per_hour = vcpu_cost_per_hour + memory_cost_per_hour
    cost_per_job = cost_per_hour * (runtime_minutes / 60)
    total_cost = cost_per_job * n_jobs
    total_runtime_hours = (runtime_minutes / 60) * n_jobs
    
    return {
        'cost_per_job': cost_per_job,
        'total_cost': total_cost,
        'total_runtime_hours': total_runtime_hours,
        'cost_per_hour': cost_per_hour,
        'spot_discount': spot_discount,
        'vcpus': vcpus,
        'memory_gb': memory_gb
    }


class AWSBatchRunner:
    """manages AWS Batch execution for permutation processing
    
    Key features:
    - uploads experiment data to S3 once
    - submits independent jobs for each permutation
    - handles retries automatically via AWS Batch
    - downloads results when complete
    - supports spot instances for cost savings
    - idempotent: can re-run interrupted computations
    """
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.s3 = boto3.client('s3', region_name=config.region)
        self.batch = boto3.client('batch', region_name=config.region)
        self._uploaded_data = set()  # track uploaded files
    
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
            self._uploaded_data.add(experiment_id)
            return f's3://{self.config.s3_bucket}/{s3_key}'
        except ClientError as e:
            raise RuntimeError(f'failed to upload experiment: {e}')
    
    def check_existing_results(self, experiment_id: str, 
                               n_perm: int) -> set:
        """check which permutations already have results
        
        Args:
            experiment_id: experiment identifier
            n_perm: total number of permutations
        
        Returns:
            set of completed permutation indices
        """
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
        except ClientError:
            # bucket/prefix doesn't exist yet
            pass
        
        return completed
    
    def submit_jobs(self, experiment_id: str, n_perm: int,
                   skip_completed: bool = True,
                   dry_run: bool = False) -> Dict[str, Any]:
        """submit permutation jobs to AWS Batch
        
        Args:
            experiment_id: experiment identifier
            n_perm: number of permutations (0 to n_perm inclusive)
            skip_completed: skip permutations that already have results
            dry_run: if True, only estimate costs without submitting
        
        Returns:
            submission_info dict with job_ids, costs, etc.
        """
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
        
        # estimate costs
        cost_est = estimate_cost(
            n_jobs=n_jobs,
            runtime_minutes=self.config.timeout_minutes,
            memory_mb=self.config.memory_mb,
            vcpus=self.config.vcpus,
            use_spot=self.config.use_spot
        )
        
        print(f'\n{"="*60}')
        print('AWS BATCH JOB SUBMISSION')
        print(f'{"="*60}')
        print(f'experiment: {experiment_id}')
        print(f'jobs to submit: {n_jobs}')
        print(f'skipped (completed): {len(completed)}')
        print(f'\nCOST ESTIMATE:')
        print(f'  per job: ${cost_est["cost_per_job"]:.4f}')
        print(f'  total: ${cost_est["total_cost"]:.2f}')
        print(f'  compute hours: {cost_est["total_runtime_hours"]:.1f}')
        print(f'  using spot instances: {self.config.use_spot}')
        print(f'\nRESOURCES PER JOB:')
        print(f'  vCPUs: {self.config.vcpus}')
        print(f'  memory: {self.config.memory_mb} MB')
        print(f'  timeout: {self.config.timeout_minutes} min')
        print(f'{"="*60}\n')
        
        # check cost limits
        if cost_est['total_cost'] > self.config.max_cost_per_hour:
            warnings.warn(
                f'estimated cost ${cost_est["total_cost"]:.2f} exceeds '
                f'max ${self.config.max_cost_per_hour:.2f}. '
                f'set dry_run=False to proceed anyway.'
            )
            if not dry_run:
                response = input('proceed anyway? (yes/no): ')
                if response.lower() != 'yes':
                    return {'cancelled': True, 'cost_estimate': cost_est}
        
        if dry_run:
            print('DRY RUN - not submitting jobs')
            return {
                'dry_run': True,
                'n_jobs': n_jobs,
                'perm_indices': perm_indices,
                'cost_estimate': cost_est
            }
        
        # submit jobs
        job_ids = []
        data_path = f's3://{self.config.s3_bucket}/{self.config.s3_prefix}/experiments/{experiment_id}/data.pkl'
        
        for perm_idx in perm_indices:
            job_name = f'{experiment_id}_perm_{perm_idx}'
            
            try:
                response = self.batch.submit_job(
                    jobName=job_name,
                    jobQueue=self.config.job_queue,
                    jobDefinition=self.config.job_definition,
                    containerOverrides={
                        'vcpus': self.config.vcpus,
                        'memory': self.config.memory_mb,
                        'command': [
                            '--data-path', data_path,
                            '--perm-idx', str(perm_idx),
                            '--s3-bucket', self.config.s3_bucket,
                            '--s3-prefix', self.config.s3_prefix,
                            '--experiment-id', experiment_id
                        ]
                    },
                    timeout={'attemptDurationSeconds': self.config.timeout_minutes * 60}
                )
                job_ids.append(response['jobId'])
            except ClientError as e:
                print(f'error submitting job for perm {perm_idx}: {e}')
        
        print(f'submitted {len(job_ids)} jobs')
        
        return {
            'n_jobs': n_jobs,
            'job_ids': job_ids,
            'perm_indices': perm_indices,
            'cost_estimate': cost_est,
            'skipped': list(completed)
        }
    
    def monitor_jobs(self, job_ids: list, poll_interval: int = 30):
        """monitor job progress
        
        Args:
            job_ids: list of AWS Batch job IDs
            poll_interval: seconds between status checks
        """
        if not job_ids:
            print('no jobs to monitor')
            return
        
        print(f'monitoring {len(job_ids)} jobs...')
        
        failed_jobs = []  # track failed jobs for detailed reporting
        
        while True:
            # get job statuses
            statuses = {'SUBMITTED': 0, 'PENDING': 0, 'RUNNABLE': 0,
                       'STARTING': 0, 'RUNNING': 0, 'SUCCEEDED': 0,
                       'FAILED': 0}
            
            # batch describe in chunks of 100
            for i in range(0, len(job_ids), 100):
                chunk = job_ids[i:i+100]
                try:
                    response = self.batch.describe_jobs(jobs=chunk)
                    for job in response['jobs']:
                        status = job['status']
                        statuses[status] = statuses.get(status, 0) + 1
                        
                        # track failed jobs with details
                        if status == 'FAILED' and job['jobId'] not in [f['jobId'] for f in failed_jobs]:
                            failed_jobs.append({
                                'jobId': job['jobId'],
                                'jobName': job['jobName'],
                                'statusReason': job.get('statusReason', 'Unknown'),
                                'container': job.get('container', {})
                            })
                except ClientError as e:
                    print(f'error checking jobs: {e}')
                    continue
            
            # print status
            total = len(job_ids)
            done = statuses['SUCCEEDED'] + statuses['FAILED']
            print(f'\n[{time.strftime("%H:%M:%S")}] Job Status:')
            print(f'  completed: {statuses["SUCCEEDED"]}/{total}')
            print(f'  failed: {statuses["FAILED"]}')
            print(f'  running: {statuses["RUNNING"]}')
            print(f'  pending: {statuses["PENDING"] + statuses["RUNNABLE"]}')
            print(f'  progress: {done/total*100:.1f}%')
            
            # check if all done
            if done == total:
                print(f'\nall jobs complete!')
                print(f'  succeeded: {statuses["SUCCEEDED"]}')
                print(f'  failed: {statuses["FAILED"]}')
                
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
                
                break
            
            time.sleep(poll_interval)
    
    def get_failure_details(self, job_ids: list):
        """get detailed failure information for failed jobs
        
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
    
    def download_results(self, experiment_id: str, n_perm: int,
                        output_dir: Path) -> Dict[int, Any]:
        """download results from S3
        
        Args:
            experiment_id: experiment identifier
            n_perm: total number of permutations
            output_dir: local directory to save results
        
        Returns:
            dict mapping perm_idx to result data
        """
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
