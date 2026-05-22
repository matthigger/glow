from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Literal, Optional, Tuple, List
import hashlib
import json
import math

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from platformdirs import user_data_dir
from tqdm import tqdm
from botocore.exceptions import ClientError
import cloudpickle as pickle

import glow
from glow.benchmark.hcp_data import get_hcp_path

base = Path(user_data_dir('glow', 'glow_author'))
path_result = base / 'results'
path_result.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    label: str

    # -------- runner --------
    # Runner instance that owns this config's result emission + hashing.
    # See glow.benchmark.runner for concrete subclasses.
    runner: 'Runner' = None

    # -------- cloud execution --------
    # if set, runs experiments on AWS cloud (one job per experiment)
    cloud_config: Optional['CloudConfig'] = None

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

    # how the effect's spatial extent is sampled:
    #   'minvar' — ExtenterMinVar (greedy variance-minimising growth)
    #   'sphere' — ExtenterSphere (random-centre dilation)
    effect_extenter: Literal['minvar', 'sphere'] = 'minvar'

    # number of jobs (each runs another effect).  1 is serial, -1 runs as many
    # as the computer has threads
    n_jobs: int = 1

    # -------- persistence / debugging --------
    # error_save=True wraps Ana(...) in try/except so a single failed
    # experiment doesn't abort the whole batch (important for AWS jobs);
    # the trace is logged to ERROR/<uuid>.json for diagnosis.
    error_save: bool = True

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

    # override default result directory (None → standard results/)
    result_dir: Optional[Path] = None

    def __post_init__(self):
        self.exp_orig = None
        self.folder = None
        self._shared_exp_s3_key = None  # S3 key for shared experiment data
        self._exp_img_only = None  # cached HCP image-only exp (pre-sample_x)
        self._hcp_subjects = None  # cached sorted subject id list

    def _hcp_subject_list(self):
        """Sorted list of HCP subject ids picked up by hcp_sbj_regex.

        Canonical across regex rewrites that select the same files.  Cached
        on the Config after the first directory scan.
        """
        if self._hcp_subjects is None:
            feats = tuple(sorted(self.hcp_feats))
            img_glob_dict = {feat: f'*_{feat}.nii.gz' for feat in feats}
            self._hcp_subjects = tuple(
                glow.experiment.ExperimentImageOnly.list_subjects(
                    folder=get_hcp_path(),
                    sbj_regex=self.hcp_sbj_regex,
                    img_glob_dict=img_glob_dict))
        return self._hcp_subjects

    def base_recipe(self):
        """Shared per-row recipe components (merged with Runner.label_recipe).

        Hashes the declared config knobs that define the dataset — not the
        realized noise/design draw, which now varies per trial.  Source-
        specific identity:
          * WGN: shape, a, b, num_img, exp_seed
          * HCP: sorted feats + sorted subject ids (stable under regex rewrites)
        """
        if self.source == 'wgn':
            source_id = {
                'wgn_shape': tuple(self.wgn_shape),
                'wgn_a': self.wgn_a,
                'wgn_b': self.wgn_b,
                'wgn_num_img': self.wgn_num_img,
                'exp_seed': self.exp_seed,
            }
        elif self.source == 'hcp':
            source_id = {
                'hcp_feats': tuple(sorted(self.hcp_feats)),
                'hcp_subjects': self._hcp_subject_list(),
            }
        else:
            raise ValueError(f'unknown source: {self.source}')

        return {
            'source': self.source,
            **source_id,
            'crop_n_vox': self.crop_n_vox,
            'effect_perc': self.effect_perc,
            'effect_extenter': self.effect_extenter,
            'radius': self.radius,
            'runner': type(self.runner).__name__ if self.runner else None,
        }

    def _job_hash(self, labels=None):
        """Combined hash for a set of labels (for job identification).

        Hashes the sorted per-label hashes together. If labels is None,
        uses all labels the runner emits.
        """
        if labels is None:
            labels = sorted(self.runner.labels)
        else:
            labels = sorted(labels)
        per_label = [self.runner.hash(self, lab) for lab in labels]
        sig = json.dumps(per_label, sort_keys=True)
        return hashlib.sha256(sig.encode()).hexdigest()[:12]

    def prep_exp_orig(self, hcp_feats=None, wgn_b=None, wgn_num_img=None,
                      seed=None):
        """prepare the base experiment (HCP or WGN)."""
        if self.source == 'hcp':
            if self._exp_img_only is None:
                path = get_hcp_path()
                feats = hcp_feats if hcp_feats is not None else self.hcp_feats
                img_glob_dict = {feat: f'*_{feat}.nii.gz' for feat in feats}
                self._exp_img_only = (
                    glow.experiment.ExperimentImageOnly.from_search(
                        folder=path,
                        sbj_regex=self.hcp_sbj_regex,
                        img_glob_dict=img_glob_dict))
            # Per-trial X draw: combine exp_seed with trial seed so each
            # trial gets an independent design matrix.
            if seed is None:
                x_seed = self.exp_seed
            else:
                x_seed = int(np.random.SeedSequence(
                    [int(self.exp_seed), int(seed)]).generate_state(1)[0])
            self.exp_orig = self._exp_img_only.sample_x(a=2, seed=x_seed,
                                                        add_bias=True)
        elif self.source == 'wgn':
            b = wgn_b if wgn_b is not None else self.wgn_b
            num_img = wgn_num_img if wgn_num_img is not None else self.wgn_num_img
            # Per-trial noise draw: combine exp_seed with trial seed so each
            # trial is an independent WGN realization.
            if seed is None:
                noise_seed = self.exp_seed
            else:
                noise_seed = int(np.random.SeedSequence(
                    [int(self.exp_seed), int(seed)]).generate_state(1)[0])
            self.exp_orig = glow.experiment.Experiment.from_gauss(
                seed=noise_seed,
                shape=self.wgn_shape,
                a=self.wgn_a,
                b=b,
                num_img=num_img)

    def get_exp_eff(self, seed, effect_llr, radius=None, hcp_feats=None,
                    wgn_b=None, wgn_num_img=None, effect_perc=None):
        """return an experiment with a synthetic effect imposed."""
        # Re-prepare exp_orig if dataset parameters changed or if not yet created
        # Note: On cloud workers, exp_orig should already be loaded from shared cache
        wgn_seed_changed = (
            self.source == 'wgn' and
            seed != getattr(self, '_last_wgn_seed', object())
        )
        hcp_seed_changed = (
            self.source == 'hcp' and
            seed != getattr(self, '_last_hcp_seed', object())
        )
        needs_recreate = (
            self.exp_orig is None or
            (hcp_feats is not None and hcp_feats != getattr(self, '_last_hcp_feats', None)) or
            (wgn_b is not None and wgn_b != getattr(self, '_last_wgn_b', None)) or
            (wgn_num_img is not None and wgn_num_img != getattr(self, '_last_wgn_num_img', None)) or
            wgn_seed_changed or
            hcp_seed_changed
        )

        if needs_recreate:
            if (hasattr(self, '_shared_exp_s3_key') and self._shared_exp_s3_key
                    and self.exp_orig is None
                    and getattr(self, '_exp_img_only', None) is None):
                raise RuntimeError(
                    f'exp_orig is None but shared cache reference exists. '
                    f'Worker should have loaded from: {self._shared_exp_s3_key}'
                )
            self.prep_exp_orig(hcp_feats=hcp_feats, wgn_b=wgn_b,
                               wgn_num_img=wgn_num_img, seed=seed)
            if hcp_feats is not None:
                self._last_hcp_feats = hcp_feats
            if wgn_b is not None:
                self._last_wgn_b = wgn_b
            if wgn_num_img is not None:
                self._last_wgn_num_img = wgn_num_img
            if self.source == 'wgn':
                self._last_wgn_seed = seed
            if self.source == 'hcp':
                self._last_hcp_seed = seed

        # trim experiment to reasonable size (for speedup)
        radius_to_use = radius if radius is not None else self.radius
        if self.crop_n_vox is not None:
            extenter = glow.effect.ExtenterSphere(n_vox=self.crop_n_vox)
            mask = extenter(mask_idx=self.exp_orig.mask_idx, seed=seed,
                            contiguous=True)
            recipe_step = {'op': 'apply_mask', 'args': {'mask': mask}}
            exp = self.exp_orig.apply_mask(mask, recipe_step=recipe_step)
        elif radius_to_use is not None:
            extenter = glow.effect.ExtenterSphere(radius=radius_to_use)
            mask = extenter(mask_idx=self.exp_orig.mask_idx, seed=seed,
                            contiguous=True)
            recipe_step = {'op': 'apply_mask', 'args': {'mask': mask}}
            exp = self.exp_orig.apply_mask(mask, recipe_step=recipe_step)
        else:
            exp = self.exp_orig

        # scale normalize before sampling minimum variance (each feature given
        # equal weight in sampling extent)
        exp = glow.experiment.ExperimentScaled.from_exp(exp)

        # sample effect space
        perc = effect_perc if effect_perc is not None else self.effect_perc
        n = exp.y.shape[2] * perc
        if self.effect_extenter == 'minvar':
            extenter = glow.effect.ExtenterMinVar(n_vox=n)
        elif self.effect_extenter == 'sphere':
            extenter = glow.effect.ExtenterSphere(n_vox=int(n),
                                                   connected=True)
        else:
            raise ValueError(
                f'unknown effect_extenter: {self.effect_extenter!r}')

        # impose effect
        return glow.effect.EffectSynthetic.impose(
            exp, effect_llr=effect_llr, extenter=extenter,
            seed=seed)

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

    def _cached_labels(self, kwargs, df, label_hashes):
        """return the set of labels already cached for these kwargs.

        Parameters
        ----------
        label_hashes : dict[str, str]
            Mapping from label -> per-label config hash.
        """
        if df.empty:
            return set()
        if 'config_hash' not in df.columns:
            return set()

        # build kwargs mask once
        kw_mask = pd.Series(True, index=df.index)
        for key, val in kwargs.items():
            if key not in df.columns:
                continue
            if isinstance(val, (float, np.floating)):
                kw_mask &= df[key].round(14) == round(float(val), 14)
            else:
                kw_mask &= df[key] == val

        cached = set()
        for label, lhash in label_hashes.items():
            label_mask = kw_mask & (df['label'] == label)
            hash_mask = df['config_hash'] == lhash
            if (label_mask & hash_mask).any():
                cached.add(label)
        return cached

    def _is_experiment_cached(self, kwargs, df, expected_labels, label_hashes):
        """check if all expected labels already have results for these kwargs."""
        if not expected_labels:
            return False
        return expected_labels.issubset(
            self._cached_labels(kwargs, df, label_hashes))

    def _filter_uncached(self, kwargs_list, verbose=True):
        """return list of (exp_idx, kwargs, missing_labels) for experiments
        that have at least one missing label.

        Each entry is (exp_idx, kwargs, missing_labels) where missing_labels
        is the set of labels still needed.  Fully cached experiments are
        omitted entirely.
        """
        from glow.benchmark.file import load_update_all
        df, _folder, _n_new = load_update_all(
            self.label, verbose=False, result_dir=self.result_dir)
        expected = set(self.runner.labels)
        label_hashes = {lab: self.runner.hash(self, lab)
                        for lab in expected}

        uncached = []
        n_fully_cached = 0
        n_partial = 0
        for exp_idx, kwargs in enumerate(kwargs_list):
            cached = self._cached_labels(kwargs, df, label_hashes)
            missing = expected - cached
            if not missing:
                n_fully_cached += 1
            else:
                if cached:
                    n_partial += 1
                uncached.append((exp_idx, kwargs, missing))

        if verbose and (n_fully_cached > 0 or n_partial > 0):
            parts = []
            if n_fully_cached:
                parts.append(f'{n_fully_cached} fully cached')
            if n_partial:
                parts.append(f'{n_partial} partially cached')
            parts.append(f'{len(uncached)} to run')
            print(f'  {", ".join(parts)}')
        return uncached

    def prep_folder(self):
        base = self.result_dir if self.result_dir is not None else path_result
        self.folder = base / self.label
        self.folder.mkdir(exist_ok=True, parents=True)

    def _as_serializable(self):
        from glow.benchmark.runner import Runner
        from dataclasses import is_dataclass, fields

        def convert(obj):
            if isinstance(obj, Runner):
                return convert(obj.to_dict())
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            if is_dataclass(obj) and not isinstance(obj, type):
                return {f.name: convert(getattr(obj, f.name))
                        for f in fields(obj)}
            if hasattr(obj, '__name__'):
                return obj.__name__
            return obj

        # asdict() is recursive on dataclasses but passes non-dataclass
        # attributes through unchanged; convert() handles the rest.
        d = {k: convert(v) for k, v in self.__dict__.items()}
        for k in ('exp_orig', 'folder', '_shared_exp_s3_key',
                  '_exp_img_only',
                  '_last_hcp_feats', '_last_wgn_b', '_last_wgn_num_img',
                  '_last_wgn_seed', '_last_hcp_seed'):
            d.pop(k, None)
        return d

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
        expected = set(self.runner.labels)

        # _skip_labels is only honored by RunAna (per-label analyses); other
        # runners ignore unknown kwargs via **iter_kw signature.
        from glow.benchmark.runner import RunAna
        if isinstance(self.runner, RunAna):
            Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(self.runner.run)(
                    config=self,
                    _skip_labels=expected - missing,
                    **kwargs)
                for _, kwargs, missing in tqdm(uncached)
            )
        else:
            Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(self.runner.run)(config=self, **kwargs)
                for _, kwargs, _missing in tqdm(uncached)
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
        # For shared-cache sources use the pre-sample_x hash so the S3 key
        # is stable across per-trial X redraws.
        is_shared = (hasattr(self.cloud_config, 'shared_exp_sources') and
                     self.source in self.cloud_config.shared_exp_sources)
        if is_shared and self._exp_img_only is not None:
            exp_sig = self._exp_img_only._hash()
        else:
            exp_sig = self.exp_orig._hash()

        # Estimate memory for experiment workers (before clearing exp_orig for HCP)
        memory_mb = None
        b, num_img, num_vox_full = self.exp_orig.y.shape[0], self.exp_orig.y.shape[1], self.exp_orig.y.shape[2]
        n_perm = 100
        rep = self.runner.representative_ana_kw() if self.runner else None
        if rep is not None:
            _Ana, first_kw = rep
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
        if is_shared:
            # Upload the pre-sample_x image-only experiment so workers can
            # redraw X per trial from the same cached data.
            shared_obj = (self._exp_img_only
                          if self._exp_img_only is not None
                          else self.exp_orig)
            shared_data_key = f'{self.cloud_config.s3_prefix}/shared_exp_data/{exp_sig}.pkl'

            try:
                runner.s3.head_object(
                    Bucket=self.cloud_config.s3_bucket,
                    Key=shared_data_key
                )
                if verbose:
                    print(f'  ✓ Found shared experiment data on S3: {exp_sig[:8]}...')
            except ClientError:
                # Upload to S3 (cross-machine: force full pickle so the
                # worker has y inline; the recipe — with the laptop's
                # paths — is preserved and travels through to result
                # pickles for rehydration back on the laptop).
                if verbose:
                    print(f'  Uploading shared experiment data: {exp_sig[:8]}...')
                from glow.experiment.regen import force_full_pickle
                with force_full_pickle(shared_obj):
                    exp_bytes = pickle.dumps(shared_obj)
                runner.s3.put_object(
                    Bucket=self.cloud_config.s3_bucket,
                    Key=shared_data_key,
                    Body=exp_bytes
                )
                if verbose:
                    print(f'  ✓ Uploaded ({len(exp_bytes) / 1024**2:.1f} MB)')

            # Store S3 reference; clear local copies so they aren't pickled.
            self._shared_exp_s3_key = shared_data_key
            self.exp_orig = None
            self._exp_img_only = None

        # generate unique run ID
        run_id = f'{self.label}_{uuid.uuid4().hex[:8]}'
        if verbose:
            print(f'  Run ID: {run_id}')

        # upload config + all kwargs to S3 (workers will download these)
        runner.upload_config(self, run_id)
        runner.upload_all_kwargs(
            run_id, [(idx, kw) for idx, kw, _missing in uncached])

        # estimate per-job timeout from runtime models
        timeout_minutes = None
        try:
            from glow.benchmark.runtime import estimate_timeout_minutes
            est = estimate_timeout_minutes(self, platform='aws')
            if est is not None:
                timeout_minutes = int(math.ceil(est[0]))
                est_min = est[1]
                tag = ' (upper bound)' if est[2] else ''
                if verbose:
                    print(f'  Estimated runtime: {est_min:.1f} min/job{tag}'
                          f'  ->  timeout: {timeout_minutes} min')
        except (ImportError, FileNotFoundError, json.JSONDecodeError,
                KeyError, ValueError, TypeError):
            pass

        # submit jobs as array job (single API call)
        if verbose:
            print(f'  Submitting {len(uncached)} jobs to AWS Batch...')

        indices = [idx for idx, _kw, _missing in uncached]
        command_template = [
            '--s3-bucket', runner.config.s3_bucket,
            '--s3-prefix', runner.config.s3_prefix,
            '--run-id', run_id,
        ]
        array_info = runner.submit_array_job(
            job_name=f'glow_{run_id}',
            command_template=command_template,
            indices=indices,
            index_arg='--exp-idx',
            memory_mb=memory_mb,
            timeout_minutes=timeout_minutes,
        )
        job_ids = array_info['child_job_ids']
        job_exp_idx = array_info['index_map']

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