import numpy as np

import glow
from glow.benchmark.config import Config
from glow.benchmark.run import run_ana, run_segment

# common params
# run experiments serially (n_jobs=1) but parallelize permutations
# within each experiment (n_jobs_perm=-1) to avoid memory explosion
common_dict = dict(n_seed=2, 
                   hotel_tr_all=np.logspace(np.log10(0.03), np.log10(1.0), 2),
                   effect_perc=.2,
                   n_jobs=1,
                   detail_save=False,
                   error_save=False)

# wgn params
wgn_dict = dict(wgn_shape=(8, 8, 8),
                wgn_a=2,
                wgn_b=2,
                wgn_num_img=100,
                exp_seed=0)

# hcp params
hcp_dict = dict(hcp_feats=['fa', 'md'],
                radius=8)

# Analysis params
n_perm = 100
alpha_fwer = .05
n_jobs_perm=-1

kwargs_glow = dict(n_perm=n_perm,
                   n_perm_adj=50,
                   n_perm_prune=200,
                   min_size=1,
                   alpha_prune=.05,
                   alpha_fwer=alpha_fwer,
                   n_jobs_perm=n_jobs_perm) 
kwargs_vba = dict(n_perm=n_perm,
                  tfce_flag=False,
                  alpha_fwer=alpha_fwer,
                  n_jobs_perm=n_jobs_perm)
kwargs_tfce = kwargs_vba | dict(tfce_flag=True)

# build configs of experiments
config_list = list()

# vba experiment: compare which method performs best
ana_kwargs_dict_vba = {'GLOW': (glow.experiment.AnalysisGLOW, kwargs_glow),
                        'VBA': (glow.experiment.AnalysisVBA, kwargs_vba),
                        'VBA-TFCE': (glow.experiment.AnalysisVBA, kwargs_tfce)}

config_list.append(Config(label='vba_hcp',
                          source='hcp',
                          run_fnc=run_ana,
                          ana_kwargs_dict=ana_kwargs_dict_vba,
                          **(common_dict | hcp_dict)))
config_list.append(Config(label='vba_wgn',
                          source='wgn',
                          run_fnc=run_ana,
                          ana_kwargs_dict=ana_kwargs_dict_vba,
                          **(common_dict | wgn_dict)))

# mancova stat experiment: compare which statistic performs best
ana_kwargs_dict_mancova_stat = dict()
for label, get_stat in glow.experiment.mancova.stat_dict.items():
    kwargs = kwargs_glow | dict(get_stat=get_stat)
    ana_kwargs_dict_mancova_stat[label] = glow.experiment.AnalysisGLOW, kwargs

config_list.append(Config(label='mancova_stat_hcp',
                          source='hcp',
                          run_fnc=run_ana,
                          ana_kwargs_dict=ana_kwargs_dict_mancova_stat,
                          **(common_dict | hcp_dict)))
config_list.append(Config(label='mancova_stat_wgn',
                          source='wgn',
                          run_fnc=run_ana,
                          ana_kwargs_dict=ana_kwargs_dict_mancova_stat,
                          **(common_dict | wgn_dict)))

# alpha_prune experiment: vary alpha_prune, which offers reasonable
# performance / speed tradeoff point?
ana_kwargs_dict_alpha_prune = dict()
for _alpha_prune in [.05, .15, .5]:
    _kwargs_glow = kwargs_glow | dict(alpha_prune=_alpha_prune)
    ana_kwargs_dict_alpha_prune[f'alpha_prune={_alpha_prune}'] = (
        glow.experiment.AnalysisGLOW, _kwargs_glow)
config_list.append(Config(label='alpha_prune',
                          source='hcp',
                          run_fnc=run_ana,
                          ana_kwargs_dict=ana_kwargs_dict_alpha_prune,
                          **(common_dict | hcp_dict)))

# runtime experiment, how does it vary with region size?
# reduced from 10 seeds × 6 radii (60 jobs) to 5 seeds × 4 radii (20 jobs)
radius_all = [3, 5]
config_list.append(Config(label='runtime_radius',
                          source='hcp',
                          run_fnc=run_ana,
                          ana_kwargs_dict=ana_kwargs_dict_vba,
                          iter_params={'seed': np.arange(5), 'radius': radius_all},
                          fixed_params={'hotel_tr': 0.1},
                          **(common_dict | hcp_dict)))

# how does it vary with b? (wgn with b=1, b=2)
# WGN: vary b
config_list.append(Config(label='dataset_wgn_b',
                          source='wgn',
                          run_fnc=run_ana,
                          ana_kwargs_dict=ana_kwargs_dict_vba,
                          iter_params={'seed': np.arange(5), 'wgn_b': [1, 2]},  # reduced from 10 to 5
                          fixed_params={'hotel_tr': 0.1},
                          **(common_dict | wgn_dict)))

# HCP: vary features
config_list.append(Config(label='dataset_hcp_feats',
                          source='hcp',
                          run_fnc=run_ana,
                          ana_kwargs_dict=ana_kwargs_dict_vba,
                          iter_params={'seed': np.arange(5), 'hcp_feats': [['fa'], ['fa', 'md']]},  # reduced from 10 to 5
                          fixed_params={'hotel_tr': 0.1},
                          **(common_dict | hcp_dict)))

# segmentation configs
config_list.append(Config(label='segment_hcp',
                          source='hcp',
                          run_fnc=run_segment,
                          **(common_dict | hcp_dict)))
config_list.append(Config(label='segment_wgn',
                          source='wgn',
                          run_fnc=run_segment,
                          **(common_dict | wgn_dict)))


if __name__ == '__main__':
    USE_CLOUD = True
    
    if USE_CLOUD:
        import configparser
        from pathlib import Path
        from glow.aws.aws_batch import CloudConfig
        
        # load AWS config
        config_file = Path.home() / '.glow_aws_config'
        if not config_file.exists():
            config_file = Path(__file__).parent.parent.parent / '.glow_aws_config'
        
        parser = configparser.ConfigParser()
        parser.read(config_file)
        
        s3_bucket = parser['aws']['s3_bucket']
        s3_prefix = 'glow-paper-benchmarks'
        region = parser['aws']['region']
        
        # create cloud config
        cloud_config = CloudConfig(
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            job_queue=parser['aws']['job_queue'],
            job_definition=parser['aws']['job_definition'],
            region=region,
            timeout_minutes=180,  # 3 hours per experiment
            retry_attempts=1
        )
        
        # add cloud_config to each config
        for config in config_list:
            config.cloud_config = cloud_config
        
        print('=' * 60)
        print('Cloud execution enabled - parallel submission mode')
        print('=' * 60)
        
        # submit all jobs from all configs first
        all_job_info = []
        print('\n[PHASE 1] Submitting all jobs...\n')
        for config in config_list:
            job_info = config.submit_cloud_jobs(verbose=True)
            all_job_info.append(job_info)
        
        # monitor ALL jobs together (not sequentially)
        print('\n' + '=' * 60)
        print(f'[PHASE 2] Monitoring all jobs from {len(all_job_info)} configs')
        print('=' * 60)
        
        # collect all job IDs and build job_info_map for incremental downloads
        all_job_ids = []
        job_info_map = {}  # maps job_id -> {'run_id': str, 'exp_idx': int, 'output_folder': Path}
        
        for job_info in all_job_info:
            run_id = job_info['run_id']
            output_folder = job_info['folder']
            
            # get exp_idx from job names (we need to query AWS to get job names)
            # but we can build a partial map from the job submission pattern
            # For now, we'll let monitor_jobs extract from job names
            for job_id in job_info['job_ids']:
                all_job_ids.append(job_id)
                # We'll populate exp_idx when we query jobs in monitor_jobs
        
        print(f'\nTotal jobs submitted in this run: {len(all_job_ids)}')
        if all_job_ids:
            print(f'Job IDs tracked: {all_job_ids[0][:8]}... through {all_job_ids[-1][:8]}...')
        
        # build job_info_map by querying job names and matching to run_ids
        if all_job_ids:
            runner = all_job_info[0]['runner']
            
            # create mapping from run_id to output_folder
            run_id_to_folder = {info['run_id']: info['folder'] for info in all_job_info}
            
            # query jobs to get names and build map
            for i in range(0, len(all_job_ids), 100):
                chunk = all_job_ids[i:i+100]
                try:
                    response = runner.batch.describe_jobs(jobs=chunk)
                    for job in response['jobs']:
                        job_name = job['jobName']
                        # extract run_id and exp_idx from job name: glow_{run_id}_exp{exp_idx:06d}
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
        
        # monitor all jobs together and download incrementally
        if all_job_ids:
            runner = all_job_info[0]['runner']  # any runner works (same cloud_config)
            runner.monitor_jobs(all_job_ids, job_info_map=job_info_map)
        
        # download any remaining results (in case some weren't downloaded incrementally)
        print('\n' + '=' * 60)
        print('[PHASE 3] Downloading any remaining results...')
        print('=' * 60)
        for config, job_info in zip(config_list, all_job_info):
            runner = job_info['runner']
            # check if results still exist in S3
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
        
        print('\n' + '=' * 60)
        print('✓ All benchmarks complete')
        print('=' * 60)
    else:
        # run all benchmarks locally (serial)
        for config in config_list:
            print(f'begin: {config.label}')
            config.run_all(verbose=True)
