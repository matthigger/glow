"""AWS Batch worker for processing single permutations

This script runs on AWS and processes a single permutation, then uploads
the result to S3. It's designed to be idempotent and fault-tolerant.
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


def main():
    parser = argparse.ArgumentParser(description='AWS Batch permutation worker')
    parser.add_argument('--data-path', required=True, help='S3 path to experiment data')
    parser.add_argument('--perm-idx', required=True, type=int, help='permutation index')
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket for results')
    parser.add_argument('--s3-prefix', required=True, help='S3 prefix for results')
    parser.add_argument('--experiment-id', required=True, help='experiment ID')
    
    args = parser.parse_args()
    
    print(f'starting worker for permutation {args.perm_idx}')
    print(f'data path: {args.data_path}')
    
    # parse S3 path
    if args.data_path.startswith('s3://'):
        s3_path = args.data_path[5:]  # remove 's3://'
        bucket, key = s3_path.split('/', 1)
    else:
        print('error: data-path must start with s3://')
        sys.exit(1)
    
    # download experiment data from S3
    s3 = boto3.client('s3')
    
    print(f'downloading experiment data from s3://{bucket}/{key}')
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = pickle.loads(response['Body'].read())
        exp = data['exp']
        ana_kwargs = data['ana_kwargs']
        experiment_id = data['experiment_id']
    except ClientError as e:
        print(f'error downloading data: {e}')
        sys.exit(1)
    
    print(f'loaded experiment: {exp.y.shape}')
    
    # check if result already exists (idempotency)
    result_key = f'{args.s3_prefix}/results/{args.experiment_id}/{args.perm_idx:06d}_result.pkl'
    try:
        s3.head_object(Bucket=args.s3_bucket, Key=result_key)
        print(f'result already exists at s3://{args.s3_bucket}/{result_key}')
        print('skipping computation (idempotent)')
        sys.exit(0)
    except ClientError:
        # object doesn't exist, proceed with computation
        pass
    
    # process permutation
    print(f'processing permutation {args.perm_idx}...')
    try:
        result = process_permutation(exp, ana_kwargs, args.perm_idx)
        print(f'completed permutation {args.perm_idx}')
    except Exception as e:
        print(f'error processing permutation: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # upload result to S3
    print(f'uploading result to s3://{args.s3_bucket}/{result_key}')
    try:
        result_bytes = pickle.dumps(result)
        s3.put_object(
            Bucket=args.s3_bucket,
            Key=result_key,
            Body=result_bytes
        )
        print(f'successfully uploaded result')
    except ClientError as e:
        print(f'error uploading result: {e}')
        sys.exit(1)
    
    print(f'worker completed successfully')
    sys.exit(0)


if __name__ == '__main__':
    main()
