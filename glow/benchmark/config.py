from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from itertools import product
from pathlib import Path
from typing import Literal, Optional, Tuple, List
import hashlib
import json

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from platformdirs import user_data_dir
from tqdm import tqdm
from botocore.exceptions import ClientError
import cloudpickle as pickle

import glow
from glow.benchmark.hcp_data import DEFAULT_HCP_PATH, get_hcp_path

base = Path(user_data_dir('glow', 'glow_author'))
path_result = base / 'results'
path_result.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    label: str

    # -------- run function --------
    # function to run for this config (e.g., run_ana or run_segment)
    run_fnc: callable = None

    # -------- cloud execution --------
    # if set, runs experiments on AWS cloud (one job per experiment)
    cloud_config: Optional['CloudConfig'] = None

    # -------- analysis kwargs --------
    # a dictionary, keys are labels of each analysis, values are tuples of
    # Analysis objects (AnalysisGLOW or AnalysisVBA) and kwargs to be sent
    # to their constructor
    ana_kwargs_dict: dict = None

    # -------- image set selection --------
    source: Literal['wgn', 'hcp'] = 'wgn'

    # -------- sweeps / repetitions --------
    # an experiment is run per seed / effect_llr pair
    n_seed: int = 100

    # controls severity of effects via size-normalized LLR
    # (large values = easier to find)
    effect_llr_all: np.ndarray = field(
        default_factory=lambda: np.logspace(np.log10(0.01), np.log10(0.3), 15)
    )

    # -------- experiment knobs --------
    # the percentage of total volume which the effect occupies
    effect_perc: float = 0.2

    # number of jobs (each runs another effect).  1 is serial, -1 runs as many
    # as the computer has threads
    n_jobs: int = 1

    # -------- persistence / debugging --------
    detail_save: bool = True
    error_save: bool = False

    # -------- source-specific --------
    # HCP
    hcp_feats: List[str] = field(default_factory=lambda: ['fa', 'md'])
    hcp_sbj_regex: str = r'[\d]{6}'

    # WGN
    wgn_shape: Tuple = (5, 5, 5)
    wgn_a: int = 2
    wgn_b: int = 2
    wgn_num_img: int = 100
    exp_seed: int = 0

    # region restriction (sphere crop applied per-experiment in get_exp_eff)
    radius: Optional[int] = None
    crop_n_vox: Optional[int] = None

    # -------- experiment iteration specification --------
    # Declarative way to specify what to iterate over
    # iter_params: dict mapping parameter names to lists of values to iterate over
    #   Default: {'seed': range(n_seed), 'effect_llr': effect_llr_all}
    #   Example: {'seed': range(10), 'radius': [2, 5, 10, None]}
    #   Example: {'seed': range(10), 'wgn_b': [1, 2]}
    iter_params: Optional[dict] = None
    
    # fixed_params: dict mapping parameter names to fixed values
    #   These override default values when iterating
    #   Example: {'effect_llr': 0.1}  # Use fixed effect_llr when iterating over other params
    fixed_params: Optional[dict] = None

    # -------- plotting --------
    # which iterated parameter to use as the x-axis when plotting
    x_param: str = 'effect_llr'

    def __post_init__(self):
        self.exp_orig = None
        self.folder = None
        self._shared_exp_s3_key = None  # S3 key for shared experiment data

    def _config_hash(self):
        """hash of all config parameters that affect results.

        Combines: exp_orig data hash + effect params + analysis config +
        run function.  Used for result-level cache invalidation.
        """
        if self.exp_orig is None:
            self.prep_exp_orig()

        d = {
            'exp_hash': self.exp_orig._hash(),
            'effect_perc': self.effect_perc,
            'radius': self.radius,
            'run_fnc': self.run_fnc.__name__ if self.run_fnc else None,
        }

        if self.ana_kwargs_dict:
            # keys injected at runtime (don't affect results)
            _runtime_keys = {'perm_dir', 'n_jobs_perm'}
            ana = {}
            for label, (cls, kw) in self.ana_kwargs_dict.items():
                entry = {'class': cls.__name__}
                for k, v in sorted(kw.items()):
                    if k in _runtime_keys:
                        continue
                    entry[k] = v.__name__ if callable(v) else v
                ana[label] = entry
            d['ana'] = ana

        sig = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(sig.encode()).hexdigest()[:12]

    def prep_exp_orig(self, hcp_feats=None, wgn_b=None, wgn_num_img=None):
        """prepare the base experiment (HCP or WGN)."""
        if self.source == 'hcp':
            path = get_hcp_path()
            feats = hcp_feats if hcp_feats is not None else self.hcp_feats
            img_glob_dict = {feat: f'*_{feat}.nii.gz' for feat in feats}
            exp = glow.experiment.ExperimentImageOnly.from_search(
                folder=path,
                sbj_regex=self.hcp_sbj_regex,
                img_glob_dict=img_glob_dict)
            self.exp_orig = exp.sample_x(a=2, seed=self.exp_seed,
                                         add_bias=True)
        elif self.source == 'wgn':
            b = wgn_b if wgn_b is not None else self.wgn_b
            num_img = wgn_num_img if wgn_num_img is not None else self.wgn_num_img
            self.exp_orig = glow.experiment.Experiment.from_gauss(
                seed=self.exp_seed,
                shape=self.wgn_shape,
                a=self.wgn_a,
                b=b,
                num_img=num_img)

    def get_exp_eff(self, seed, effect_llr, radius=None, hcp_feats=None,
                    wgn_b=None, wgn_num_img=None, effect_perc=None,
                    roughness=None):
        """return an experiment with a synthetic effect imposed."""
        # Re-prepare exp_orig if dataset parameters changed or if not yet created
        # Note: On cloud workers, exp_orig should already be loaded from shared cache
        needs_recreate = (
            self.exp_orig is None or
            (hcp_feats is not None and hcp_feats != getattr(self, '_last_hcp_feats', None)) or
            (wgn_b is not None and wgn_b != getattr(self, '_last_wgn_b', None)) or
            (wgn_num_img is not None and wgn_num_img != getattr(self, '_last_wgn_num_img', None))
        )
        
        if needs_recreate:
            if hasattr(self, '_shared_exp_s3_key') and self._shared_exp_s3_key and self.exp_orig is None:
                raise RuntimeError(
                    f'exp_orig is None but shared cache reference exists. '
                    f'Worker should have loaded from: {self._shared_exp_s3_key}'
                )
            self.prep_exp_orig(hcp_feats=hcp_feats, wgn_b=wgn_b,
                               wgn_num_img=wgn_num_img)
            if hcp_feats is not None:
                self._last_hcp_feats = hcp_feats
            if wgn_b is not None:
                self._last_wgn_b = wgn_b
            if wgn_num_img is not None:
                self._last_wgn_num_img = wgn_num_img

        # trim experiment to reasonable size (for speedup)
        radius_to_use = radius if radius is not None else self.radius
        if self.crop_n_vox is not None:
            extenter = glow.effect.ExtenterSphere(n_vox=self.crop_n_vox)
            mask = extenter(mask_idx=self.exp_orig.mask_idx, seed=seed,
                            contiguous=True)
            exp = self.exp_orig.apply_mask(mask)
        elif radius_to_use is not None:
            extenter = glow.effect.ExtenterSphere(radius=radius_to_use)
            mask = extenter(mask_idx=self.exp_orig.mask_idx, seed=seed,
                            contiguous=True)
            exp = self.exp_orig.apply_mask(mask)
        else:
            exp = self.exp_orig

        # scale normalize before sampling minimum variance (each feature given
        # equal weight in sampling extent)
        exp = glow.experiment.ExperimentScaled.from_exp(exp)

        # sample effect space
        perc = effect_perc if effect_perc is not None else self.effect_perc
        n = exp.y.shape[2] * perc
        extenter = glow.effect.ExtenterMinVar(n=n)

        # impose effect
        return exp.impose_effect(extenter=extenter,
                                 seed=seed,
                                 effect_llr=effect_llr,
                                 roughness=roughness)

    def iter_kwargs(self):
        """yield kwarg dicts for each experiment (product of iter_params)."""
        # build iteration specification
        if self.iter_params is None:
            # default: iterate over seed and effect_llr
            iter_spec = {
                'seed': np.arange(self.n_seed),
                'effect_llr': self.effect_llr_all
            }
        else:
            iter_spec = self.iter_params.copy()
            # ensure seed is always included if not specified
            if 'seed' not in iter_spec:
                iter_spec['seed'] = np.arange(self.n_seed)
        
        # build fixed parameters
        fixed = self.fixed_params.copy() if self.fixed_params is not None else {}
        
        # get parameter names and value lists for iteration
        param_names = list(iter_spec.keys())
        param_values = [iter_spec[name] for name in param_names]
        
        # iterate over all combinations
        for values in product(*param_values):
            kwargs = dict(zip(param_names, values))
            # add fixed parameters (these override any iterated values if there's a conflict)
            kwargs.update(fixed)
            
            yield kwargs

    def _get_expected_labels(self):
        """return the set of result 'label' strings one experiment produces."""
        from glow.benchmark.run import run_ana, run_segment
        if self.run_fnc is run_ana:
            return set(self.ana_kwargs_dict.keys())
        if self.run_fnc is run_segment:
            from glow.experiment.cluster import _MODES
            return set(_MODES)
        return set()

    def _is_experiment_cached(self, kwargs, df, expected_labels, config_hash):
        """check if all expected labels already have results for these kwargs."""
        if df.empty or not expected_labels:
            return False
        mask = pd.Series(True, index=df.index)
        # match on config_hash
        if 'config_hash' in df.columns:
            mask &= df['config_hash'] == config_hash
        else:
            return False  # no hash column means legacy data
        for key, val in kwargs.items():
            if key not in df.columns:
                continue
            if isinstance(val, (float, np.floating)):
                mask &= df[key].round(14) == round(float(val), 14)
            else:
                mask &= df[key] == val
        cached_labels = set(df.loc[mask, 'label'].unique())
        return expected_labels.issubset(cached_labels)

    def _filter_uncached(self, kwargs_list, verbose=True):
        """return list of (exp_idx, kwargs) for experiments not yet cached."""
        from glow.benchmark.file import load_update_all
        df, _folder, _n_new = load_update_all(self.label, verbose=False)
        expected = self._get_expected_labels()
        config_hash = self._config_hash()

        uncached = []
        for exp_idx, kwargs in enumerate(kwargs_list):
            if not self._is_experiment_cached(kwargs, df, expected, config_hash):
                uncached.append((exp_idx, kwargs))

        n_cached = len(kwargs_list) - len(uncached)
        if verbose and n_cached > 0:
            print(f'  {n_cached} cached, {len(uncached)} to run')
        return uncached

    def prep_folder(self):
        self.folder = path_result / self.label
        self.folder.mkdir(exist_ok=True, parents=True)

    def _as_serializable(self):
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            if hasattr(obj, '__name__'):
                return obj.__name__
            return obj

        return convert(asdict(self))

    def save_config(self, path):
        path = Path(path).with_suffix('.yaml')
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('w') as f:
            yaml.safe_dump(self._as_serializable(), f, sort_keys=True)
        return path

    def run_all(self, verbose=True):
        """run all experiments (local or cloud)."""
        # if cloud_config is set, submit experiments to AWS
        if self.cloud_config is not None:
            self._run_all_on_cloud(verbose=verbose)
            return

        # prep folder and save config
        self.prep_folder()
        path_config = self.folder / 'config.yaml'
        self.save_config(path=path_config)

        if verbose:
            print(f'outputs stored in: {self.folder}')

        # check cache: skip experiments that already have results
        kwargs_list = list(self.iter_kwargs())
        uncached = self._filter_uncached(kwargs_list, verbose=verbose)

        if not uncached:
            if verbose:
                print(f'  all {len(kwargs_list)} experiments cached')
            return

        if verbose:
            print(f'  running {len(uncached)} experiments')

        n_jobs = self.n_jobs if self.n_jobs not in (0, 1) else 1
        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self.run_fnc)(config=self, **kwargs)
            for _, kwargs in tqdm(uncached)
        )
    
    def submit_cloud_jobs(self, verbose=True):
        """submit experiments to AWS Batch (returns job info dict)."""
        from glow.aws.aws_batch import AWSBatchRunner
        import uuid

        if verbose:
            print(f'Submitting jobs for: {self.label}')

        # prep folder locally (results will be downloaded here)
        self.prep_folder()

        # save config locally for reference
        path_config = self.folder / 'config.yaml'
        self.save_config(path=path_config)

        # check cache: skip experiments that already have results
        kwargs_list = list(self.iter_kwargs())
        uncached = self._filter_uncached(kwargs_list, verbose=verbose)

        if not uncached:
            if verbose:
                print(f'  all {len(kwargs_list)} experiments cached, nothing to submit')
            return {
                'runner': None,
                'run_id': None,
                'job_ids': [],
                'folder': self.folder,
                'label': self.label
            }

        # initialize runner
        runner = AWSBatchRunner(self.cloud_config)

        # Load experiment data and compute its hash (used as S3 cache key)
        if self.exp_orig is None:
            if verbose:
                print('  Preparing experiment data...')
            self.prep_exp_orig()
        exp_sig = self.exp_orig._hash()

        # Estimate memory for experiment workers (before clearing exp_orig for HCP)
        memory_mb = None
        b, num_img, num_vox_full = self.exp_orig.y.shape[0], self.exp_orig.y.shape[1], self.exp_orig.y.shape[2]
        n_perm = 100
        if self.ana_kwargs_dict:
            first_kw = next(iter(self.ana_kwargs_dict.values()))[1]
            n_perm = first_kw.get('n_perm_fwer', n_perm)
        if self.source == 'hcp':
            num_vox_cropped = self.crop_n_vox if self.crop_n_vox is not None else num_vox_full
            full_dataset_mb = b * num_img * num_vox_full * 8 / (1024 * 1024)
            regression_mb = runner.estimate_experiment_memory_mb(b, num_img, num_vox_cropped, n_perm)
            if regression_mb is not None:
                memory_mb = int(full_dataset_mb + regression_mb)
            else:
                memory_mb = int(full_dataset_mb)
            if runner.config.oom_memory_mb_tiers:
                for t in runner.config.oom_memory_mb_tiers:
                    if t >= memory_mb:
                        memory_mb = t
                        break
                else:
                    memory_mb = runner.config.oom_memory_mb_tiers[-1]
            if memory_mb <= runner.config.memory_mb:
                memory_mb = None
        else:
            memory_mb = runner.estimate_experiment_memory_mb(b, num_img, num_vox_full, n_perm)

        # For shared cache sources, upload to S3 if not already there
        if (hasattr(self.cloud_config, 'shared_exp_sources') and
            self.source in self.cloud_config.shared_exp_sources):

            shared_data_key = f'{self.cloud_config.s3_prefix}/shared_exp_data/{exp_sig}.pkl'

            try:
                runner.s3.head_object(
                    Bucket=self.cloud_config.s3_bucket,
                    Key=shared_data_key
                )
                if verbose:
                    print(f'  ✓ Found shared experiment data on S3: {exp_sig[:8]}...')
            except ClientError:
                # Upload to S3
                if verbose:
                    print(f'  Uploading shared experiment data: {exp_sig[:8]}...')
                exp_bytes = pickle.dumps(self.exp_orig)
                runner.s3.put_object(
                    Bucket=self.cloud_config.s3_bucket,
                    Key=shared_data_key,
                    Body=exp_bytes
                )
                if verbose:
                    print(f'  ✓ Uploaded ({len(exp_bytes) / 1024**2:.1f} MB)')

            # Store S3 reference; clear local copy so it's not in pickled config
            self._shared_exp_s3_key = shared_data_key
            self.exp_orig = None

        # generate unique run ID
        run_id = f'{self.label}_{uuid.uuid4().hex[:8]}'
        if verbose:
            print(f'  Run ID: {run_id}')

        # upload config + all kwargs to S3 (workers will download these)
        runner.upload_config(self, run_id)
        runner.upload_all_kwargs(run_id, uncached)

        # submit jobs in parallel (API calls are I/O-bound)
        job_ids = []
        job_exp_idx = {}

        if verbose:
            print(f'  Submitting {len(uncached)} jobs to AWS Batch...')

        def _submit(exp_idx):
            return exp_idx, runner.submit_experiment_job(
                run_id=run_id, exp_idx=exp_idx, memory_mb=memory_mb)

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(_submit, idx): idx
                       for idx, _kw in uncached}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f'  {self.label}', disable=not verbose):
                exp_idx, job_id = fut.result()
                job_ids.append(job_id)
                job_exp_idx[job_id] = exp_idx

        if verbose:
            print(f'  ✓ Submitted {len(job_ids)} jobs')

        # return info needed for monitoring/downloading
        return {
            'runner': runner,
            'run_id': run_id,
            'job_ids': job_ids,
            'job_exp_idx': job_exp_idx,
            'folder': self.folder,
            'label': self.label
        }
    
    def wait_and_download_results(self, job_info, verbose=True):
        """monitor jobs and download results as they complete."""
        runner = job_info['runner']
        run_id = job_info['run_id']
        job_ids = job_info['job_ids']
        folder = job_info['folder']
        label = job_info['label']
        job_exp_idx = job_info.get('job_exp_idx', {})

        job_info_map = {
            jid: {
                'run_id': run_id,
                'exp_idx': job_exp_idx[jid],
                'output_folder': folder,
            }
            for jid in job_ids if jid in job_exp_idx
        }

        if verbose:
            print(f'\nMonitoring {label} ({len(job_ids)} jobs)...')

        runner.monitor_jobs(job_ids, job_info_map=job_info_map)

        if verbose:
            print(f'✓ {label} complete: {folder}')
    
    def _run_all_on_cloud(self, verbose=True):
        """submit all experiments to AWS Batch and wait for results."""
        # submit jobs
        job_info = self.submit_cloud_jobs(verbose=verbose)
        
        # wait and download
        self.wait_and_download_results(job_info, verbose=verbose)