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
        retry_attempts: number of retry attempts for failed jobs (1 = no retries)
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
    retry_attempts: int = 3
    
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


def upload_hcp_data(s3_bucket: str, s3_prefix: str, hcp_path: str, 
                    region: str = 'us-east-1') -> str:
    """upload HCP imaging data to S3 for cloud experiments
    
    Args:
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix (e.g., 'glow-paper-benchmarks')
        hcp_path: local path to HCP data directory
        region: AWS region
    
    Returns:
        S3 URI where data was uploaded (s3://bucket/prefix/hcp_data/)
    """
    import os
    from pathlib import Path
    from tqdm import tqdm
    
    s3 = boto3.client('s3', region_name=region)
    hcp_path = Path(hcp_path)
    s3_hcp_prefix = f'{s3_prefix}/hcp_data'
    
    if not hcp_path.exists():
        raise FileNotFoundError(f'HCP data not found at {hcp_path}')
    
    # list all NIfTI files
    nii_files = list(hcp_path.glob('*.nii.gz'))
    if not nii_files:
        raise FileNotFoundError(f'No .nii.gz files found in {hcp_path}')
    
    print(f'Uploading {len(nii_files)} HCP files to s3://{s3_bucket}/{s3_hcp_prefix}/')
    
    # check if already uploaded (quick check for first file)
    first_key = f'{s3_hcp_prefix}/{nii_files[0].name}'
    try:
        s3.head_object(Bucket=s3_bucket, Key=first_key)
        # check file count matches
        paginator = s3.get_paginator('list_objects_v2')
        existing_count = 0
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_hcp_prefix):
            if 'Contents' in page:
                existing_count += len(page['Contents'])
        
        if existing_count == len(nii_files):
            print(f'  ✓ HCP data already uploaded ({existing_count} files)')
            return f's3://{s3_bucket}/{s3_hcp_prefix}'
        else:
            print(f'  Partial upload detected ({existing_count}/{len(nii_files)}), re-uploading...')
    except ClientError:
        pass  # not uploaded yet
    
    # upload files
    uploaded = 0
    for nii_file in tqdm(nii_files, desc='  Uploading'):
        s3_key = f'{s3_hcp_prefix}/{nii_file.name}'
        try:
            s3.upload_file(str(nii_file), s3_bucket, s3_key)
            uploaded += 1
        except ClientError as e:
            print(f'\n  ✗ Error uploading {nii_file.name}: {e}')
            raise
    
    print(f'  ✓ Uploaded {uploaded} files')
    return f's3://{s3_bucket}/{s3_hcp_prefix}'


def check_hcp_data_exists(s3_bucket: str, s3_prefix: str, 
                          region: str = 'us-east-1') -> bool:
    """check if HCP data exists in S3
    
    Args:
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix (e.g., 'glow-paper-benchmarks')
        region: AWS region
    
    Returns:
        True if HCP data exists in S3
    """
    s3 = boto3.client('s3', region_name=region)
    s3_hcp_prefix = f'{s3_prefix}/hcp_data'
    
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_hcp_prefix, MaxKeys=1):
            if 'Contents' in page and len(page['Contents']) > 0:
                return True
        return False
    except ClientError:
        return False


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
        
        print(f'\n{"="*60}')
        print('AWS BATCH JOB SUBMISSION')
        print(f'{"="*60}')
        print(f'experiment: {experiment_id}')
        print(f'jobs to submit: {n_jobs}')
        print(f'skipped (completed): {len(completed)}')
        print(f'timeout: {self.config.timeout_minutes} min per job')
        print(f'{"="*60}\n')
        
        if dry_run:
            print('DRY RUN - not submitting jobs')
            return {
                'dry_run': True,
                'n_jobs': n_jobs,
                'perm_indices': perm_indices
            }
        
        # submit jobs
        job_ids = []
        data_path = f's3://{self.config.s3_bucket}/{self.config.s3_prefix}/experiments/{experiment_id}/data.pkl'
        
        for perm_idx in perm_indices:
            job_name = f'{experiment_id}_perm_{perm_idx}'
            
            try:
                # build container overrides
                # note: vcpus/memory are NOT included for EC2 job definitions
                # they are set in the job definition itself
                # for fargate, use resourceRequirements instead
                overrides = {
                    'command': [
                        '--data-path', data_path,
                        '--perm-idx', str(perm_idx),
                        '--s3-bucket', self.config.s3_bucket,
                        '--s3-prefix', self.config.s3_prefix,
                        '--experiment-id', experiment_id
                    ]
                }
                
                response = self.batch.submit_job(
                    jobName=job_name,
                    jobQueue=self.config.job_queue,
                    jobDefinition=self.config.job_definition,
                    containerOverrides=overrides,
                    retryStrategy={'attempts': self.config.retry_attempts},
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
        
        print(f'monitoring {len(job_ids)} jobs from this run (not all queue jobs)...')
        
        failed_jobs = []  # track failed jobs for detailed reporting
        first_check = True
        
        while True:
            # get job statuses
            statuses = {'SUBMITTED': 0, 'PENDING': 0, 'RUNNABLE': 0,
                       'STARTING': 0, 'RUNNING': 0, 'SUCCEEDED': 0,
                       'FAILED': 0}
            
            jobs_found = 0
            
            # batch describe in chunks of 100
            for i in range(0, len(job_ids), 100):
                chunk = job_ids[i:i+100]
                try:
                    response = self.batch.describe_jobs(jobs=chunk)
                    jobs_found += len(response['jobs'])
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
            
            # verify we got responses for all requested jobs
            if first_check and jobs_found != len(job_ids):
                print(f'\n⚠ Warning: Requested {len(job_ids)} jobs, AWS returned {jobs_found}')
                first_check = False
            
            # print status (only counts jobs from this run's job_ids)
            total = len(job_ids)
            done = statuses['SUCCEEDED'] + statuses['FAILED']
            print(f'\n[{time.strftime("%H:%M:%S")}] Job Status (THIS RUN ONLY):')
            print(f'  completed: {statuses["SUCCEEDED"]}/{total}')
            print(f'  failed: {statuses["FAILED"]} (from this run only)')
            print(f'  running: {statuses["RUNNING"]}')
            print(f'  pending: {statuses["PENDING"] + statuses["RUNNABLE"] + statuses["STARTING"]}')
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
    
    def submit_experiment_job(self, run_id: str, exp_idx: int, kwargs: Dict[str, Any]) -> str:
        """submit a single experiment job to AWS Batch
        
        Each job runs a full experiment with all permutations serially.
        This amortizes container overhead across all permutations.
        
        Args:
            run_id: unique run identifier
            exp_idx: experiment index
            kwargs: experiment kwargs dict (seed, hotel_tr, etc.)
        
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
        
        try:
            response = self.batch.submit_job(
                jobName=job_name,
                jobQueue=self.config.job_queue,
                jobDefinition=self.config.job_definition,
                timeout={'attemptDurationSeconds': self.config.timeout_minutes * 60},
                retryStrategy={'attempts': self.config.retry_attempts},
                containerOverrides={
                    'command': [
                        '--s3-bucket', self.config.s3_bucket,
                        '--s3-prefix', self.config.s3_prefix,
                        '--run-id', run_id,
                        '--exp-idx', str(exp_idx)
                    ]
                }
            )
            
            return response['jobId']
        except ClientError as e:
            raise RuntimeError(f'failed to submit job: {e}')
    
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
            
            print(f'  downloaded {file_count} files to {output_folder}')
        except ClientError as e:
            raise RuntimeError(f'failed to download results: {e}')
