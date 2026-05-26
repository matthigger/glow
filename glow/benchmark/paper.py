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
    vcpus = int(parser['aws'].get('vcpus_per_job', 1))

    return CloudConfig(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        job_queue=parser['aws']['job_queue'],
        job_definition=parser['aws']['job_definition'],
        region=region,
        vcpus=vcpus,
        timeout_minutes=180,  # 3 hours per experiment
        retry_attempts=5,
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
        run_id = job_info['run_id']
        folder = job_info['folder']
        job_exp_idx = job_info.get('job_exp_idx', {})
        for job_id in job_info['job_ids']:
            all_job_ids.append(job_id)
            if job_id in job_exp_idx:
                job_info_map[job_id] = {
                    'run_id': run_id,
                    'exp_idx': job_exp_idx[job_id],
                    'output_folder': folder,
                }

    print(f'\nTotal jobs submitted in this run: {len(all_job_ids)}')
    if all_job_ids:
        print(f'Job IDs tracked: {all_job_ids[0][:8]}... through {all_job_ids[-1][:8]}...')

    return all_job_ids, job_info_map


def monitor_all_jobs(all_job_info, all_job_ids, job_info_map):
    print('\n' + '=' * 60)
    print(f'[PHASE 2] Monitoring all jobs from {len(all_job_info)} configs')
    print('=' * 60)

    permanently_failed = []
    if all_job_ids:
        runner = next((info['runner'] for info in all_job_info
                       if info['runner'] is not None), None)
        if runner is not None:
            permanently_failed = runner.monitor_jobs(
                all_job_ids, job_info_map=job_info_map) or []

    if permanently_failed:
        print(f'\n⚠ {len(permanently_failed)} job(s) permanently failed '
              f'(exceeded max memory tier)')
    return permanently_failed


def download_remaining_results(all_job_info):
    print('\n' + '=' * 60)
    print('[PHASE 3] Downloading any remaining results...')
    print('=' * 60)

    for job_info in all_job_info:
        runner = job_info['runner']
        if runner is None:
            continue  # fully cached config, nothing to download
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


def _count_uncached_jobs(config):
    """Return number of experiments that still need to run."""
    kwargs_list = list(config.iter_kwargs())
    uncached = config._filter_uncached(kwargs_list, verbose=False)
    return len(uncached)


def _print_cost_summary(configs, cloud_config):
    """Print uncached job counts and return True if user confirms."""
    rows = [(c.label, _count_uncached_jobs(c)) for c in configs]
    total_jobs = sum(n for _, n in rows)
    default_timeout = cloud_config.timeout_minutes

    print('\n' + '=' * 60)
    print('  JOB SUMMARY')
    print('=' * 60)
    print(f'  Default per-job timeout: {default_timeout} min\n')
    print(f'  {"Config":<32} {"Jobs":>6}')
    print('  ' + '-' * 56)
    for label, n_jobs in rows:
        print(f'  {label:<32} {n_jobs:>6}')
    print('  ' + '-' * 56)
    print(f'  {"Total":<32} {total_jobs:>6}')
    print('=' * 60)

    resp = input('\n  Proceed? [y/N] ').strip().lower()
    return resp == 'y'


def run_cloud(configs):
    cloud_config = load_cloud_config()
    for config in configs:
        config.cloud_config = cloud_config

    print('=' * 60)
    print('Cloud execution enabled - parallel submission mode')
    print('=' * 60)

    if not _print_cost_summary(configs, cloud_config):
        print('\nAborted.')
        return

    all_job_info = submit_all_jobs(configs)

    # persist job info so results can be downloaded later
    from glow.aws.download import save_job_info
    saved = save_job_info(all_job_info, cloud_config)

    all_job_ids, job_info_map = build_job_info_map(all_job_info)

    if not all_job_ids:
        print('\nAll experiments cached — nothing to monitor.')
        return

    print(f'\nJob info saved to disk ({len(saved)} run(s)).')
    print('You can download results later with:')
    print('  python -m glow.aws.download')
    resp = input('\nWait for results now? [Y/n] ').strip().lower()
    if resp == 'n':
        print('\nJobs are running on AWS Batch. Download when ready with:')
        print('  python -m glow.aws.download')
        return

    permanently_failed = monitor_all_jobs(all_job_info, all_job_ids,
                                          job_info_map)
    download_remaining_results(all_job_info)

    # clean up job info files for completed runs
    from glow.aws.download import JOBS_DIR
    for info in all_job_info:
        run_id = info.get('run_id')
        if run_id:
            p = JOBS_DIR / f'{run_id}.json'
            if p.exists():
                p.unlink()

    print('\n' + '=' * 60)
    if permanently_failed:
        print(f'⚠ Benchmarks finished with {len(permanently_failed)} '
              f'permanent failure(s)')
    else:
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
    from fnmatch import fnmatch

    patterns = list(args.configs)
    if not patterns:
        return list(CONFIG_BY_LABEL.values())

    configs = []
    for pattern in patterns:
        if pattern in CONFIG_BY_LABEL:
            configs.append(CONFIG_BY_LABEL[pattern])
        else:
            matched = [c for k, c in CONFIG_BY_LABEL.items()
                       if fnmatch(k, pattern)]
            if not matched:
                raise ValueError(f'no config labels match: {pattern}')
            configs.extend(matched)
    return configs


if __name__ == '__main__':
    args = parse_args()
    configs = resolve_configs(args)
    if args.local:
        run_local(configs)
    else:
        run_cloud(configs)
