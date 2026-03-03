"""AWS Batch worker for GLOW experiments."""

import argparse
import sys
import os
from pathlib import Path

import boto3
import cloudpickle as pickle
from botocore.exceptions import ClientError
import numpy as np
import psutil
import tracemalloc


class S3Checkpoint:
    """Persist and resume partial AnalysisGLOW state via S3."""

    def __init__(self, s3_client, bucket, key):
        self._s3 = s3_client
        self._bucket = bucket
        self._key = key

    def load(self):
        """Return saved state dict or None if no checkpoint exists."""
        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=self._key)
            state = pickle.loads(response['Body'].read())
            print(f'  checkpoint loaded: s3://{self._bucket}/{self._key}')
            return state
        except ClientError:
            return None

    def save(self, child_dict, stat, perm_idx):
        """Upload current permutation state to S3."""
        state = {
            'child_dict': child_dict,
            'stat': stat,
            'last_perm_idx': perm_idx,
        }
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._key,
            Body=pickle.dumps(state),
        )
        print(f'  checkpoint saved (perm {perm_idx})')

    def delete(self):
        """Remove checkpoint from S3 after successful completion."""
        try:
            self._s3.delete_object(Bucket=self._bucket, Key=self._key)
        except ClientError:
            pass


def get_array_info(obj, prefix='', visited=None, max_depth=5, depth=0):
    """recursively find all numpy arrays in an object and return their info."""
    if visited is None:
        visited = set()
    
    if depth > max_depth:
        return []
    
    # Prevent cycles
    obj_id = id(obj)
    if obj_id in visited:
        return []
    visited.add(obj_id)
    
    arrays = []
    
    try:
        if isinstance(obj, np.ndarray):
            size_mb = obj.nbytes / 1024 / 1024
            arrays.append({
                'name': prefix or 'root',
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'size_mb': size_mb
            })
        elif isinstance(obj, (dict, list, tuple)):
            # Iterate through containers
            if isinstance(obj, dict):
                items = obj.items()
            else:
                items = enumerate(obj)
            
            for key, value in items:
                new_prefix = f"{prefix}.{key}" if prefix else str(key)
                arrays.extend(get_array_info(value, new_prefix, visited, max_depth, depth + 1))
        elif hasattr(obj, '__dict__'):
            # Inspect object attributes
            try:
                for attr_name in dir(obj):
                    if attr_name.startswith('__') and attr_name != '__dict__':
                        continue
                    try:
                        attr = getattr(obj, attr_name, None)
                        if attr is obj:  # Skip self-references
                            continue
                        new_prefix = f"{prefix}.{attr_name}" if prefix else attr_name
                        arrays.extend(get_array_info(attr, new_prefix, visited, max_depth, depth + 1))
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    finally:
        visited.discard(obj_id)  # Remove from visited when done with this branch
    
    return arrays


def get_memory_profile(config=None, exp=None, ana=None):
    """return a detailed memory profile dict.
    """
    # Get process memory
    process = psutil.Process(os.getpid())
    process_memory_mb = process.memory_info().rss / 1024 / 1024
    
    # Get tracemalloc stats if available
    tracemalloc_stats = None
    if tracemalloc.is_tracing():
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        tracemalloc_stats = []
        for stat in top_stats[:20]:  # Top 20
            tracemalloc_stats.append({
                'filename': stat.traceback[0].filename if stat.traceback else 'unknown',
                'lineno': stat.traceback[0].lineno if stat.traceback else 0,
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })
    
    # Find numpy arrays in key objects
    arrays_info = []
    
    if config is not None:
        arrays_info.extend(get_array_info(config, 'config'))
    
    if exp is not None:
        arrays_info.extend(get_array_info(exp, 'exp'))
    
    if ana is not None:
        arrays_info.extend(get_array_info(ana, 'ana'))
    
    # Sort arrays by size
    arrays_info.sort(key=lambda x: x['size_mb'], reverse=True)
    total_array_memory = sum(a['size_mb'] for a in arrays_info)
    
    return {
        'process_memory_mb': process_memory_mb,
        'total_array_memory_mb': total_array_memory,
        'arrays': arrays_info[:50],  # Top 50 arrays
        'tracemalloc_stats': tracemalloc_stats
    }


def print_memory_profile(config=None, exp=None, ana=None):
    """print a detailed memory profile to stdout."""
    print('\n' + '=' * 60)
    print('MEMORY PROFILE (on error)')
    print('=' * 60)
    
    profile = get_memory_profile(config=config, exp=exp, ana=ana)
    
    print(f'\nProcess Memory: {profile["process_memory_mb"]:.1f} MB')
    print(f'Total NumPy Array Memory (in inspected objects): {profile["total_array_memory_mb"]:.1f} MB')
    
    if profile['arrays']:
        print(f'\nTop NumPy Arrays (by size):')
        print(f'{"Name":<40} {"Shape":<30} {"Dtype":<12} {"Size (MB)":<12}')
        print('-' * 100)
        for arr in profile['arrays']:
            shape_str = str(arr['shape'])[:29]
            name_str = arr['name'][:39]
            print(f"{name_str:<40} {shape_str:<30} {arr['dtype']:<12} {arr['size_mb']:<12.2f}")
    
    if profile['tracemalloc_stats']:
        print(f'\nTop Memory Allocations (tracemalloc):')
        print(f'{"File":<40} {"Line":<8} {"Size (MB)":<12} {"Count":<10}')
        print('-' * 75)
        for stat in profile['tracemalloc_stats']:
            filename = stat['filename'].split('/')[-1][:39]
            print(f"{filename:<40} {stat['lineno']:<8} {stat['size_mb']:<12.2f} {stat['count']:<10}")
    
    print('=' * 60)


def process_permutation(exp, ana_kwargs, perm_idx):
    """process a single permutation and return children + stat.
    """
    from glow.experiment.cluster import cluster
    from glow.experiment.mancova import get_llr
    
    get_stat = ana_kwargs.get('get_stat', get_llr)
    
    # permute experiment
    _exp = exp.permute(perm_idx)
    
    # cluster
    children = cluster(exp=_exp)
    
    # compute stats
    b, num_img, num_vox = _exp.y.shape
    num_reg = num_vox + children.shape[0]
    
    stat = []
    import glow.graph
    for reg_idx, size, e, h in glow.graph.iter_stat(exp=_exp, children=children, n_perm=None):
        stat_val = get_stat(e=e[:, :, 0], h=h[:, :, 0], n=size)
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
        import glow.experiment
        for label, (Ana, ana_kwargs) in config.ana_kwargs_dict.items():
            if 'n_jobs_perm' in ana_kwargs:
                ana_kwargs['n_jobs_perm'] = 1
                print(f'  Setting n_jobs_perm=1 for {label}')
            # inject S3 checkpoint for AnalysisGLOW analyses
            if Ana is glow.experiment.AnalysisGLOW:
                ckpt_key = (f'{args.s3_prefix}/{args.run_id}/checkpoints/'
                            f'{args.exp_idx:06d}_{label}.pkl')
                ana_kwargs['checkpoint'] = S3Checkpoint(s3, args.s3_bucket,
                                                       ckpt_key)
                print(f'  Checkpoint enabled for {label}')
    
    # run experiment with memory profiling on error
    try:
        # Start tracemalloc for memory tracking
        tracemalloc.start()
        config.run_fnc(config=config, **kwargs)
        tracemalloc.stop()
        print(f'  ✓ Completed')
    except MemoryError as e:
        print(f'  ✗ Memory Error: {e}')
        # Try to get exp from config if available
        exp_captured = None
        if hasattr(config, 'exp_orig'):
            exp_captured = config.exp_orig
        print_memory_profile(config=config, exp=exp_captured, ana=None)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f'  ✗ Error: {e}')
        # Check if it might be memory-related
        if 'memory' in str(e).lower() or 'MemoryError' in str(type(e)):
            exp_captured = None
            if hasattr(config, 'exp_orig'):
                exp_captured = config.exp_orig
            print_memory_profile(config=config, exp=exp_captured, ana=None)
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
                
                s3.upload_file(
                    Filename=str(local_file),
                    Bucket=args.s3_bucket,
                    Key=s3_key
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
