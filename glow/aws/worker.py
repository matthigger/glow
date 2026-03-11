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
    """process a single permutation and return children + stat + size."""
    from glow.experiment.cluster import cluster
    from glow.experiment.mancova import get_llr

    get_stat = ana_kwargs.get('get_stat', get_llr)

    _exp = exp.permute(perm_idx)
    children = cluster(exp=_exp)

    b, num_img, num_vox = _exp.y.shape

    stat = []
    import glow.graph
    for reg_idx, size, e, h in glow.graph.iter_stat(exp=_exp, children=children, n_perm=None):
        stat_val = get_stat(e=e[:, :, 0], h=h[:, :, 0], n=size)
        stat.append(stat_val)

    size = glow.graph.node_sum(x=np.ones(num_vox, dtype=int),
                               children=children)

    return {
        'perm_idx': perm_idx,
        'children': children,
        'stat': stat,
        'size': size,
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
        import time as _time
        _t0 = _time.time()
        result = process_permutation(exp, ana_kwargs, args.perm_idx)
        result['elapsed_sec'] = _time.time() - _t0
        print(f'  ✓ Completed ({result["elapsed_sec"]:.1f}s)')
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


def run_synthesis_mode(args):
    """Collect permutation results from S3 and run _finalize_analysis."""
    import time
    import glow.graph
    from glow.experiment.analysis import AnalysisGLOW
    from glow.experiment.mancova import get_llr, stat_sign

    print('=' * 60)
    print('GLOW Worker - SYNTHESIS MODE')
    print(f'Experiment: {args.experiment_id}')
    print(f'Permutations: {args.n_perm + 1}')
    print('=' * 60)

    s3 = boto3.client('s3')
    data_key = (f'{args.s3_prefix}/experiments/'
                f'{args.experiment_id}/data.pkl')

    print(f'\nDownloading experiment data...')
    try:
        response = s3.get_object(Bucket=args.s3_bucket, Key=data_key)
        data = pickle.loads(response['Body'].read())
        exp = data['exp']
        ana_kwargs = data['ana_kwargs']
        print(f'  ✓ Loaded experiment: {exp.y.shape}')
    except ClientError as e:
        print(f'  ✗ Error: {e}')
        sys.exit(1)

    # poll S3 until all permutation results are available
    n_perm_fit = ana_kwargs.get('n_perm_fit', 25)
    n_perm_fwer = args.n_perm - n_perm_fit
    fit_start = n_perm_fwer + 1
    n_expected = args.n_perm + 1
    result_prefix = (f'{args.s3_prefix}/results/'
                     f'{args.experiment_id}/')

    print(f'\nWaiting for {n_expected} permutation results...')
    while True:
        completed = set()
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=args.s3_bucket,
                                       Prefix=result_prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('_result.pkl'):
                    fname = Path(key).name
                    try:
                        perm_idx = int(fname.split('_')[0])
                        if 0 <= perm_idx <= args.n_perm:
                            completed.add(perm_idx)
                    except ValueError:
                        continue
        print(f'  {len(completed)}/{n_expected} results available')
        if len(completed) >= n_expected:
            break
        time.sleep(30)

    # --- streaming synthesis: two passes over S3 results ---
    from glow.experiment.analysis import get_best_model

    b, num_img, num_vox = exp.y.shape
    get_stat = ana_kwargs.get('get_stat', get_llr)
    model = ana_kwargs.get('size_adjust_model') or get_best_model(get_stat)
    n_perm_prune = ana_kwargs.get('n_perm_prune', 100)
    alpha_fwer = ana_kwargs.get('alpha_fwer', 0.05)
    alpha_prune = ana_kwargs.get('alpha_prune', 0.05)
    min_size = ana_kwargs.get('min_size', 1)
    prune_geom_exp_eff = ana_kwargs.get('prune_geom_exp_eff', None)

    def _load_result(perm_idx):
        key = f'{result_prefix}{perm_idx:06d}_result.pkl'
        response = s3.get_object(Bucket=args.s3_bucket, Key=key)
        return pickle.loads(response['Body'].read())

    # load observed (perm 0) — kept permanently
    print(f'\nPass 1: fitting size regression ({model}) ...')
    r0 = _load_result(0)
    stat_0 = np.asarray(r0['stat'], dtype=float)
    children_0 = r0['children']
    size_0 = (np.asarray(r0['size'], dtype=float) if 'size' in r0
              else glow.graph.node_sum(np.ones(num_vox, dtype=int),
                                       children_0).astype(float))
    perm_elapsed = [r0.get('elapsed_sec')]

    XtX, Xty = None, None
    for perm_idx in range(1, n_expected):
        r = _load_result(perm_idx)
        stat_p = np.asarray(r['stat'], dtype=float)
        size_p = (np.asarray(r['size'], dtype=float) if 'size' in r
                  else glow.graph.node_sum(np.ones(num_vox, dtype=int),
                                           r['children']).astype(float))
        perm_elapsed.append(r.get('elapsed_sec'))
        if perm_idx >= fit_start:
            XtX, Xty = AnalysisGLOW.accumulate_regression(
                size_p, stat_p, model, XtX, Xty)
        del r, stat_p, size_p

    mu_fn, _, beta = AnalysisGLOW.fit_size_regression_online(
        XtX, Xty, model, get_stat)
    print(f'  model={model}  beta={beta}')

    # pass 2: compute adjusted max-stat per permutation
    print(f'Pass 2: computing FWER max-stats ...')
    sign = stat_sign.get(get_stat, 1)
    reg_active = size_0 >= min_size
    stat_max_list = []

    adj_0 = sign * (stat_0 - mu_fn(size_0))
    adj_0 = np.nan_to_num(adj_0, nan=0.0, posinf=0.0, neginf=-30.0)
    if reg_active.any():
        stat_max_list.append(float(np.nanmax(adj_0[reg_active])))
    else:
        stat_max_list.append(float('-inf'))

    for perm_idx in range(1, n_perm_fwer + 1):
        r = _load_result(perm_idx)
        stat_p = np.asarray(r['stat'], dtype=float)
        size_p = (np.asarray(r['size'], dtype=float) if 'size' in r
                  else glow.graph.node_sum(np.ones(num_vox, dtype=int),
                                           r['children']).astype(float))
        adj_p = sign * (stat_p - mu_fn(size_p))
        adj_p = np.nan_to_num(adj_p, nan=0.0, posinf=0.0, neginf=-30.0)
        if reg_active.any():
            stat_max_list.append(float(np.nanmax(adj_p[reg_active])))
        else:
            stat_max_list.append(float('-inf'))
        del r, stat_p, size_p, adj_p

    stat_max_sorted = np.sort(stat_max_list)
    n_fwer = n_perm_fwer + 1
    print(f'  ✓ {n_fwer} max-stats collected ({n_perm_fit} fit perms held out)')

    # build analysis shell and run streaming finalization
    ana = object.__new__(AnalysisGLOW)
    ana.exp = exp
    ana.get_stat = get_stat
    ana.n_jobs_perm = 1
    ana.verbose = True
    ana.adj_model = model
    ana.adj_beta = beta

    print(f'\nRunning _finalize_analysis ...')
    _t0 = time.time()
    ana._finalize_analysis(
        exp, n_perm_fwer,
        stat_0, size_0, children_0,
        mu_fn, stat_max_sorted,
        n_perm_prune, alpha_fwer, alpha_prune, min_size,
        prune_geom_exp_eff=prune_geom_exp_eff)
    ana.synthesis_elapsed_sec = time.time() - _t0
    ana.perm_elapsed_sec = perm_elapsed
    print(f'  ✓ {len(ana.effect_list)} effects discovered '
          f'(synthesis: {ana.synthesis_elapsed_sec:.1f}s)')

    # upload final analysis
    final_key = f'{result_prefix}analysis_final.pkl'
    print(f'\nUploading final analysis to s3://{args.s3_bucket}/{final_key}')
    try:
        s3.put_object(
            Bucket=args.s3_bucket,
            Key=final_key,
            Body=pickle.dumps(ana))
        print(f'  ✓ Uploaded')
    except ClientError as e:
        print(f'  ✗ Error: {e}')
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='AWS Batch worker (supports permutation, experiment, and synthesis modes)'
    )
    
    # common arguments
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket')
    parser.add_argument('--s3-prefix', required=True, help='S3 prefix')
    
    # permutation mode arguments
    parser.add_argument('--data-path', help='S3 path to experiment data (permutation mode)')
    parser.add_argument('--perm-idx', type=int, help='permutation index (permutation mode)')
    parser.add_argument('--experiment-id', help='experiment ID (permutation/synthesis mode)')
    
    # experiment mode arguments
    parser.add_argument('--run-id', help='run ID (experiment mode)')
    parser.add_argument('--exp-idx', type=int, help='experiment index (experiment mode)')

    # synthesis mode arguments
    parser.add_argument('--synthesize', action='store_true',
                        help='synthesis mode: collect perm results and finalize')
    parser.add_argument('--n-perm', type=int, help='number of permutations (synthesis mode)')
    
    args = parser.parse_args()
    
    # determine mode
    if args.synthesize:
        if not all([args.experiment_id, args.n_perm is not None]):
            parser.error('synthesis mode requires: --experiment-id, --n-perm')
        run_synthesis_mode(args)

    elif args.perm_idx is not None:
        if not all([args.data_path, args.experiment_id]):
            parser.error('permutation mode requires: --data-path, --perm-idx, --experiment-id')
        run_permutation_mode(args)
        
    elif args.exp_idx is not None:
        if not args.run_id:
            parser.error('experiment mode requires: --run-id, --exp-idx')
        run_experiment_mode(args)
        
    else:
        parser.error('must specify --perm-idx, --exp-idx, or --synthesize')
    
    print(f'\n{"=" * 60}')
    print(f'✓ Worker completed successfully')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
