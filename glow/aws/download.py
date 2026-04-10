"""Download deferred AWS Batch results from S3.

Usage:
    python -m glow.aws.download                    # download all pending runs
    python -m glow.aws.download <run_id> ...       # download specific runs
    python -m glow.aws.download --list              # list saved runs
"""

import argparse
import json
from pathlib import Path

from platformdirs import user_data_dir

JOBS_DIR = Path(user_data_dir('glow', 'glow_author')) / 'jobs'


def save_job_info(all_job_info, cloud_config):
    """Persist job info after submission so results can be downloaded later.

    Writes one JSON file per run_id into JOBS_DIR.
    """
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for info in all_job_info:
        run_id = info.get('run_id')
        if run_id is None:
            continue  # fully cached config
        path = JOBS_DIR / f'{run_id}.json'
        data = {
            'run_id': run_id,
            'label': info['label'],
            'folder': str(info['folder']),
            'job_ids': info['job_ids'],
            's3_bucket': cloud_config.s3_bucket,
            's3_prefix': cloud_config.s3_prefix,
            'region': cloud_config.region,
        }
        path.write_text(json.dumps(data, indent=2))
        paths.append(path)
    return paths


def list_saved_runs():
    """Print all saved run info files."""
    if not JOBS_DIR.exists():
        print('No saved runs.')
        return
    files = sorted(JOBS_DIR.glob('*.json'))
    if not files:
        print('No saved runs.')
        return
    print(f'{"Run ID":<40} {"Label":<28} {"Jobs":>5}')
    print('-' * 75)
    for f in files:
        data = json.loads(f.read_text())
        print(f'{data["run_id"]:<40} {data["label"]:<28} '
              f'{len(data["job_ids"]):>5}')


def download_run(run_id):
    """Download results for a single run_id from S3."""
    from glow.aws.aws_batch import AWSBatchRunner, CloudConfig

    path = JOBS_DIR / f'{run_id}.json'
    if not path.exists():
        raise FileNotFoundError(f'No saved job info for run_id: {run_id}')

    data = json.loads(path.read_text())
    config = CloudConfig(
        s3_bucket=data['s3_bucket'],
        s3_prefix=data['s3_prefix'],
        region=data['region'],
        # remaining fields unused for download
        job_queue='', job_definition='',
    )
    runner = AWSBatchRunner(config)
    folder = Path(data['folder'])

    print(f'\n[{data["label"]}] run_id={run_id}')
    runner.download_experiment_results(run_id, folder)
    print(f'  results saved to {folder}')

    # remove job info file after successful download
    path.unlink()


def download_all():
    """Download results for all saved runs."""
    if not JOBS_DIR.exists():
        print('No saved runs.')
        return
    files = sorted(JOBS_DIR.glob('*.json'))
    if not files:
        print('No saved runs.')
        return
    for f in files:
        data = json.loads(f.read_text())
        try:
            download_run(data['run_id'])
        except Exception as e:
            print(f'  error downloading {data["run_id"]}: {e}')


def main():
    parser = argparse.ArgumentParser(
        description='Download deferred AWS Batch results from S3.')
    parser.add_argument('run_ids', nargs='*',
                        help='run IDs to download (all if omitted)')
    parser.add_argument('--list', action='store_true',
                        help='list saved runs')
    args = parser.parse_args()

    if args.list:
        list_saved_runs()
    elif args.run_ids:
        for run_id in args.run_ids:
            download_run(run_id)
    else:
        download_all()


if __name__ == '__main__':
    main()
