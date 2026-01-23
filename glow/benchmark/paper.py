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

# config_list.append(Config(label='vba_hcp',
#                           source='hcp',
#                           run_fnc=run_ana,
#                           ana_kwargs_dict=ana_kwargs_dict_vba,
#                           **(common_dict | hcp_dict)))
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

# config_list.append(Config(label='mancova_stat_hcp',
#                           source='hcp',
#                           run_fnc=run_ana,
#                           ana_kwargs_dict=ana_kwargs_dict_mancova_stat,
#                           **(common_dict | hcp_dict)))
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
# config_list.append(Config(label='alpha_prune',
#                           source='hcp',
#                           run_fnc=run_ana,
#                           ana_kwargs_dict=ana_kwargs_dict_alpha_prune,
#                           **(common_dict | hcp_dict)))

# runtime experiment, how does it vary with region size?
# reduced from 10 seeds × 6 radii (60 jobs) to 5 seeds × 4 radii (20 jobs)
# radius_all = [3, 5]
# config_list.append(Config(label='runtime_radius',
#                           source='hcp',
#                           run_fnc=run_ana,
#                           ana_kwargs_dict=ana_kwargs_dict_vba,
#                           iter_params={'seed': np.arange(5), 'radius': radius_all},
#                           fixed_params={'hotel_tr': 0.1},
#                           **(common_dict | hcp_dict)))

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
# config_list.append(Config(label='dataset_hcp_feats',
#                           source='hcp',
#                           run_fnc=run_ana,
#                           ana_kwargs_dict=ana_kwargs_dict_vba,
#                           iter_params={'seed': np.arange(5), 'hcp_feats': [['fa'], ['fa', 'md']]},  # reduced from 10 to 5
#                           fixed_params={'hotel_tr': 0.1},
#                           **(common_dict | hcp_dict)))

# segmentation configs
# config_list.append(Config(label='segment_hcp',
#                           source='hcp',
#                           run_fnc=run_segment,
#                           **(common_dict | hcp_dict)))
config_list.append(Config(label='segment_wgn',
                          source='wgn',
                          run_fnc=run_segment,
                          **(common_dict | wgn_dict)))


if __name__ == '__main__':
    USE_CLOUD = True
    
    if USE_CLOUD:
        import configparser
        from pathlib import Path
        from glow.aws.aws_batch import CloudConfig, upload_hcp_data, check_hcp_data_exists
        
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
        
        # check if any HCP configs are in the list
        has_hcp_configs = any('hcp' in c.source for c in config_list)
        
        # upload HCP data if needed
        if has_hcp_configs:
            print('\n[PRE-FLIGHT] Checking HCP data...')
            hcp_path = hcp_dict.get('hcp_path', '/home/matt/data/hcp100_aug25_registered')
            # use the first hcp config's path if available
            for c in config_list:
                if 'hcp' in c.source:
                    hcp_path = c.hcp_path
                    break
            
            if Path(hcp_path).exists():
                print(f'  Local HCP data found at {hcp_path}')
                if not check_hcp_data_exists(s3_bucket, s3_prefix, region):
                    print('  Uploading HCP data to S3 (one-time operation)...')
                    upload_hcp_data(s3_bucket, s3_prefix, hcp_path, region)
                else:
                    print('  ✓ HCP data already in S3')
            else:
                if check_hcp_data_exists(s3_bucket, s3_prefix, region):
                    print(f'  ✓ HCP data already in S3 (local not found at {hcp_path})')
                else:
                    print(f'  ⚠ WARNING: HCP data not found locally ({hcp_path})')
                    print(f'  ⚠ HCP data not found in S3 either')
                    print(f'  ⚠ HCP experiments will FAIL until data is uploaded')
                    print(f'  ⚠ Upload with: upload_hcp_data("{s3_bucket}", "{s3_prefix}", "/path/to/hcp_data")')
        
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
        
        # collect all job IDs
        all_job_ids = []
        for job_info in all_job_info:
            all_job_ids.extend(job_info['job_ids'])
        
        print(f'\nTotal jobs submitted in this run: {len(all_job_ids)}')
        if all_job_ids:
            print(f'Job IDs tracked: {all_job_ids[0][:8]}... through {all_job_ids[-1][:8]}...')
        print('Note: Monitoring ONLY jobs from this run (ignoring other queue jobs)')
        
        # monitor all jobs together
        if all_job_ids:
            runner = all_job_info[0]['runner']  # any runner works (same cloud_config)
            runner.monitor_jobs(all_job_ids)
        
        # download results for each config
        print('\n' + '=' * 60)
        print('[PHASE 3] Downloading results...')
        print('=' * 60)
        for config, job_info in zip(config_list, all_job_info):
            print(f'\nDownloading {job_info["label"]}...')
            runner = job_info['runner']
            runner.download_experiment_results(job_info['run_id'], job_info['folder'])
            print(f'  ✓ {job_info["folder"]}')
        
        print('\n' + '=' * 60)
        print('✓ All benchmarks complete')
        print('=' * 60)
    else:
        # run all benchmarks locally (serial)
        for config in config_list:
            print(f'begin: {config.label}')
            config.run_all(verbose=True)
