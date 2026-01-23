"""AWS Batch worker for GLOW experiments

Supports two modes:
1. Permutation mode: Process a single permutation of one experiment
   - For large single experiments (e.g. 1M voxels, 100+ permutations)
   - Parallelize across permutations
   
2. Experiment mode: Process all permutations of one experiment
   - For many small experiments (e.g. paper benchmarks with 163 configs)
   - Parallelize across experiments, run permutations serially within each
   - ~34x cheaper, ~48x faster by amortizing container overhead

Mode is determined by which argument is provided (--perm-idx or --exp-idx)
"""

import argparse
import sys
from pathlib import Path

import boto3
import cloudpickle as pickle
from botocore.exceptions import ClientError


def process_permutation(exp, ana_kwargs, perm_idx):
    """process a single permutation
    
    Args:
        exp: experiment object
        ana_kwargs: analysis kwargs
        perm_idx: permutation index to process
    
    Returns:
        result dict with children and stat
    """
    from glow.experiment.cluster import cluster
    from glow.experiment.mancova import get_hotel_tr
    
    # get stat function (default to hotel_tr)
    get_stat = ana_kwargs.get('get_stat', get_hotel_tr)
    
    # permute experiment
    _exp = exp.permute(perm_idx)
    
    # cluster
    children = cluster(exp=_exp)
    
    # compute stats
    b, num_img, num_vox = _exp.y.shape
    num_reg = num_vox + children.shape[0]
    
    stat = []
    import glow.graph
    for reg_idx, e, h in glow.graph.iter_stat(exp=_exp, children=children, n_perm=None):
        stat_val = get_stat(e=e[:, :, 0], h=h[:, :, 0])
        stat.append(stat_val)
    
    return {
        'perm_idx': perm_idx,
        'children': children,
        'stat': stat
    }


def run_permutation_mode(args):
    """run single permutation (for large experiments)"""
    print(f'=' * 60)
    print(f'GLOW Worker - PERMUTATION MODE')
    print(f'Permutation: {args.perm_idx}')
    print(f'=' * 60)
    
    # parse S3 path for experiment data
    if not args.data_path.startswith('s3://'):
        print('error: data-path must start with s3://')
        sys.exit(1)
    
    s3_path = args.data_path[5:]
    bucket, key = s3_path.split('/', 1)
    
    # download experiment data
    s3 = boto3.client('s3')
    print(f'\nDownloading experiment from s3://{bucket}/{key}')
    
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = pickle.loads(response['Body'].read())
        exp = data['exp']
        ana_kwargs = data['ana_kwargs']
        experiment_id = data['experiment_id']
        print(f'  ✓ Loaded experiment: {exp.y.shape}')
    except ClientError as e:
        print(f'  ✗ Error: {e}')
        sys.exit(1)
    
    # check if result already exists (idempotency)
    result_key = f'{args.s3_prefix}/results/{experiment_id}/{args.perm_idx:06d}_result.pkl'
    try:
        s3.head_object(Bucket=args.s3_bucket, Key=result_key)
        print(f'\n✓ Result already exists, skipping: {result_key}')
        print(f'{"=" * 60}')
        return
    except ClientError:
        pass
    
    # process permutation
    print(f'\nProcessing permutation {args.perm_idx}...')
    try:
        result = process_permutation(exp, ana_kwargs, args.perm_idx)
        print(f'  ✓ Completed')
    except Exception as e:
        print(f'  ✗ Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # upload result
    print(f'\nUploading result to s3://{args.s3_bucket}/{result_key}')
    try:
        s3.put_object(
            Bucket=args.s3_bucket,
            Key=result_key,
            Body=pickle.dumps(result)
        )
        print(f'  ✓ Uploaded')
    except ClientError as e:
        print(f'  ✗ Error: {e}')
        sys.exit(1)


def run_experiment_mode(args):
    """run full experiment with all permutations (for many small experiments)"""
    print(f'=' * 60)
    print(f'GLOW Worker - EXPERIMENT MODE')
    print(f'Run ID: {args.run_id}')
    print(f'Experiment Index: {args.exp_idx}')
    print(f'=' * 60)
    
    # download config and kwargs
    s3 = boto3.client('s3')
    config_key = f'{args.s3_prefix}/{args.run_id}/config.pkl'
    kwargs_key = f'{args.s3_prefix}/{args.run_id}/kwargs/{args.exp_idx:06d}.pkl'
    
    print(f'\nDownloading config from s3://{args.s3_bucket}/{config_key}')
    try:
        response = s3.get_object(Bucket=args.s3_bucket, Key=config_key)
        config = pickle.loads(response['Body'].read())
        print(f'  ✓ Config: {config.label}, source: {config.source}')
        
        response = s3.get_object(Bucket=args.s3_bucket, Key=kwargs_key)
        kwargs = pickle.loads(response['Body'].read())
        print(f'  ✓ Kwargs loaded')
    except ClientError as e:
        print(f'  ✗ Error: {e}')
        sys.exit(1)
    
    # Download shared experiment data if referenced
    if hasattr(config, '_shared_exp_s3_key') and config._shared_exp_s3_key:
        print(f'\nDownloading shared experiment data from s3://{args.s3_bucket}/{config._shared_exp_s3_key}')
        try:
            response = s3.get_object(Bucket=args.s3_bucket, Key=config._shared_exp_s3_key)
            config.exp_orig = pickle.loads(response['Body'].read())
            print(f'  ✓ Loaded shared experiment data')
        except ClientError as e:
            print(f'  ✗ Error loading shared experiment data: {e}')
            sys.exit(1)
    elif config.source == 'hcp' and config.exp_orig is None:
        # HCP should have exp_orig loaded (either from shared cache or included in config)
        print(f'\n✗ Error: exp_orig is None for HCP source')
        print(f'  Expected either _shared_exp_s3_key or exp_orig in config')
        sys.exit(1)
    
    # set up temp folder
    temp_folder = Path('/tmp/glow_output') / args.run_id / str(args.exp_idx)
    temp_folder.mkdir(parents=True, exist_ok=True)
    print(f'\nTemp output folder: {temp_folder}')
    
    original_folder = config.folder
    config.folder = temp_folder
    
    # prevent nested cloud calls
    config.cloud_config = None
    
    # force serial execution (we parallelize at job level)
    print(f'\nRunning experiment...')
    print(f'  Config: {config.label}')
    print(f'  Source: {config.source}')
    print(f'  Analyses: {len(config.ana_kwargs_dict) if hasattr(config, "ana_kwargs_dict") and config.ana_kwargs_dict else "N/A"}')
    
    if hasattr(config, 'ana_kwargs_dict') and config.ana_kwargs_dict:
        for label, (Ana, ana_kwargs) in config.ana_kwargs_dict.items():
            if 'n_jobs_perm' in ana_kwargs:
                ana_kwargs['n_jobs_perm'] = 1
                print(f'  Setting n_jobs_perm=1 for {label}')
    
    # run experiment
    try:
        config.run_fnc(config=config, **kwargs)
        print(f'  ✓ Completed')
    except Exception as e:
        print(f'  ✗ Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # upload results
    result_prefix = f'{args.s3_prefix}/{args.run_id}/results/{args.exp_idx:06d}'
    print(f'\nUploading results to s3://{args.s3_bucket}/{result_prefix}/')
    
    try:
        uploaded = 0
        for local_file in temp_folder.rglob('*'):
            if local_file.is_file():
                rel_path = local_file.relative_to(temp_folder)
                s3_key = f'{result_prefix}/{rel_path}'
                
                with open(local_file, 'rb') as f:
                    s3.put_object(
                        Bucket=args.s3_bucket,
                        Key=s3_key,
                        Body=f.read()
                    )
                print(f'  uploaded {rel_path}')
                uploaded += 1
        
        print(f'  ✓ Uploaded {uploaded} files')
    except ClientError as e:
        print(f'  ✗ Error: {e}')
        sys.exit(1)
    
    config.folder = original_folder


def main():
    parser = argparse.ArgumentParser(
        description='AWS Batch worker (supports permutation and experiment modes)'
    )
    
    # common arguments
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket')
    parser.add_argument('--s3-prefix', required=True, help='S3 prefix')
    
    # permutation mode arguments
    parser.add_argument('--data-path', help='S3 path to experiment data (permutation mode)')
    parser.add_argument('--perm-idx', type=int, help='permutation index (permutation mode)')
    parser.add_argument('--experiment-id', help='experiment ID (permutation mode)')
    
    # experiment mode arguments
    parser.add_argument('--run-id', help='run ID (experiment mode)')
    parser.add_argument('--exp-idx', type=int, help='experiment index (experiment mode)')
    
    args = parser.parse_args()
    
    # determine mode
    if args.perm_idx is not None:
        # permutation mode
        if not all([args.data_path, args.experiment_id]):
            parser.error('permutation mode requires: --data-path, --perm-idx, --experiment-id')
        run_permutation_mode(args)
        
    elif args.exp_idx is not None:
        # experiment mode
        if not args.run_id:
            parser.error('experiment mode requires: --run-id, --exp-idx')
        run_experiment_mode(args)
        
    else:
        parser.error('must specify either --perm-idx (permutation mode) or --exp-idx (experiment mode)')
    
    print(f'\n{"=" * 60}')
    print(f'✓ Worker completed successfully')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
