import argparse

from glow.benchmark.paper_config import CONFIG_BY_LABEL


def load_cloud_config():
    import configparser
    from pathlib import Path
    from glow.aws.aws_batch import CloudConfig

    config_file = Path(__file__).parent.parent.parent / '.glow_aws_config'

    parser = configparser.ConfigParser()
    parser.read(config_file)

    s3_bucket = parser['aws']['s3_bucket']
    s3_prefix = 'glow-paper-benchmarks'
    region = parser['aws']['region']

    return CloudConfig(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        job_queue=parser['aws']['job_queue'],
        job_definition=parser['aws']['job_definition'],
        region=region,
        timeout_minutes=180,  # 3 hours per experiment
        retry_attempts=1
    )


def submit_all_jobs(configs):
    all_job_info = []
    print('\n[PHASE 1] Submitting all jobs...\n')
    for config in configs:
        job_info = config.submit_cloud_jobs(verbose=True)
        all_job_info.append(job_info)
    return all_job_info


def build_job_info_map(all_job_info):
    all_job_ids = []
    job_info_map = {}
    for job_info in all_job_info:
        for job_id in job_info['job_ids']:
            all_job_ids.append(job_id)

    print(f'\nTotal jobs submitted in this run: {len(all_job_ids)}')
    if all_job_ids:
        print(f'Job IDs tracked: {all_job_ids[0][:8]}... through {all_job_ids[-1][:8]}...')

    if not all_job_ids:
        return all_job_ids, job_info_map

    runner = all_job_info[0]['runner']
    run_id_to_folder = {info['run_id']: info['folder'] for info in all_job_info}

    for i in range(0, len(all_job_ids), 100):
        chunk = all_job_ids[i:i + 100]
        try:
            response = runner.batch.describe_jobs(jobs=chunk)
            for job in response['jobs']:
                job_name = job['jobName']
                if job_name.startswith('glow_') and '_exp' in job_name:
                    parts = job_name.replace('glow_', '').split('_exp')
                    if len(parts) == 2:
                        run_id = parts[0]
                        try:
                            exp_idx = int(parts[1])
                            if run_id in run_id_to_folder:
                                job_info_map[job['jobId']] = {
                                    'run_id': run_id,
                                    'exp_idx': exp_idx,
                                    'output_folder': run_id_to_folder[run_id]
                                }
                        except ValueError:
                            pass
        except Exception:
            pass

    return all_job_ids, job_info_map


def monitor_all_jobs(all_job_info, all_job_ids, job_info_map):
    print('\n' + '=' * 60)
    print(f'[PHASE 2] Monitoring all jobs from {len(all_job_info)} configs')
    print('=' * 60)

    if all_job_ids:
        runner = all_job_info[0]['runner']
        runner.monitor_jobs(all_job_ids, job_info_map=job_info_map)


def download_remaining_results(all_job_info):
    print('\n' + '=' * 60)
    print('[PHASE 3] Downloading any remaining results...')
    print('=' * 60)

    for job_info in all_job_info:
        runner = job_info['runner']
        result_prefix = f'{runner.config.s3_prefix}/{job_info["run_id"]}/results/'
        try:
            paginator = runner.s3.get_paginator('list_objects_v2')
            has_results = False
            for page in paginator.paginate(Bucket=runner.config.s3_bucket, Prefix=result_prefix, MaxKeys=1):
                if 'Contents' in page and len(page['Contents']) > 0:
                    has_results = True
                    break

            if has_results:
                print(f'\nDownloading remaining results for {job_info["label"]}...')
                runner.download_experiment_results(job_info['run_id'], job_info['folder'])
                print(f'  ✓ {job_info["folder"]}')
        except Exception as e:
            print(f'  ⚠ Could not check/download remaining results: {e}')


def run_cloud(configs):
    cloud_config = load_cloud_config()
    for config in configs:
        config.cloud_config = cloud_config

    print('=' * 60)
    print('Cloud execution enabled - parallel submission mode')
    print('=' * 60)

    all_job_info = submit_all_jobs(configs)
    all_job_ids, job_info_map = build_job_info_map(all_job_info)
    monitor_all_jobs(all_job_info, all_job_ids, job_info_map)
    download_remaining_results(all_job_info)

    print('\n' + '=' * 60)
    print('✓ All benchmarks complete')
    print('=' * 60)


def run_local(configs):
    for config in configs:
        print(f'begin: {config.label}')
        config.run_all(verbose=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Run paper benchmarks.')
    parser.add_argument('configs', nargs='*', help='config labels to run')
    parser.add_argument('--local', action='store_true', help='run locally')
    return parser.parse_args()


def resolve_configs(args):
    labels = list(args.configs)
    if not labels:
        labels = list(CONFIG_BY_LABEL.keys())

    configs = []
    for label in labels:
        if label in CONFIG_BY_LABEL:
            configs.append(CONFIG_BY_LABEL[label])
        else:
            raise ValueError(f'unknown config label: {label}')
    return configs


if __name__ == '__main__':
    args = parse_args()
    configs = resolve_configs(args)
    if args.local:
        run_local(configs)
    else:
        run_cloud(configs)
