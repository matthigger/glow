from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Literal, Optional, Tuple, List
import json

import numpy as np
import yaml
from joblib import Parallel, delayed
from platformdirs import user_data_dir
from tqdm import tqdm

import glow

base = Path(user_data_dir('glow', 'glow_author'))
path_result = base / 'results'
path_result.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    label: str

    # -------- analysis kwargs --------
    # a dictionary, keys are labels of each analysis, values are tuples of
    # Analysis objects (AnalysisGLOW or AnalysisVBA) and kwargs to be sent
    # to their constructor
    ana_kwargs_dict: dict = None

    # -------- image set selection --------
    source: Literal['wgn', 'hcp'] = 'wgn'

    # -------- sweeps / repetitions --------
    # an experiment is run per seed / hotel_tr pair
    n_seed: int = 100

    # controls severity of effects (large values = easier to find)
    hotel_tr_all: np.ndarray = field(
        default_factory=lambda: np.logspace(np.log10(0.03), np.log10(1.0), 15)
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
    hcp_path: str = '/home/matt/data/hcp100_aug25_registered'
    hcp_old_path: str = '/home/matt/data/hcp100_lowres_old/image'
    hcp_feats: List[str] = field(default_factory=lambda: ['fa', 'md'])
    hcp_sbj_regex: str = r'[\d]{6}'

    # WGN
    wgn_shape: Tuple = (5, 5, 5)
    wgn_a: int = 2
    wgn_b: int = 2
    wgn_num_img: int = 100
    exp_seed: int = 0

    # region restriction
    radius: Optional[int] = None

    # -------- experiment iteration specification --------
    # Declarative way to specify what to iterate over
    # iter_params: dict mapping parameter names to lists of values to iterate over
    #   Default: {'seed': range(n_seed), 'hotel_tr': hotel_tr_all}
    #   Example: {'seed': range(10), 'radius': [2, 5, 10, None]}
    #   Example: {'seed': range(10), 'wgn_b': [1, 2]}
    iter_params: Optional[dict] = None
    
    # fixed_params: dict mapping parameter names to fixed values
    #   These override default values when iterating
    #   Example: {'hotel_tr': 0.1}  # Use fixed hotel_tr when iterating over other params
    fixed_params: Optional[dict] = None

    def __post_init__(self):
        self.exp_orig = None
        self.folder = None

    def prep_exp_orig(self, hcp_feats=None, wgn_b=None):
        """Prepare the original experiment.
        
        Args:
            hcp_feats: Override hcp_feats if provided (for dataset experiment)
            wgn_b: Override wgn_b if provided (for dataset experiment)
        """
        if 'hcp' in self.source:
            if self.source == 'hcp':
                path = self.hcp_path
            elif self.source == 'hcp_old':
                path = self.hcp_old_path
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
            self.exp_orig = glow.experiment.Experiment.from_gauss(
                seed=self.exp_seed,
                shape=self.wgn_shape,
                a=self.wgn_a,
                b=b,
                num_img=self.wgn_num_img)

    def get_exp_eff(self, seed, hotel_tr, radius=None, hcp_feats=None, wgn_b=None):
        """Get experiment with effect imposed.
        
        Args:
            seed: Random seed
            hotel_tr: Hotelling's trace (effect strength)
            radius: Override radius if provided (for runtime experiment)
            hcp_feats: Override hcp_feats if provided (for dataset experiment)
            wgn_b: Override wgn_b if provided (for dataset experiment)
        """
        # Re-prepare exp_orig if dataset parameters changed or if not yet created
        # For dataset experiments, we need to recreate exp_orig each time
        needs_recreate = (
            self.exp_orig is None or
            (hcp_feats is not None and hcp_feats != getattr(self, '_last_hcp_feats', None)) or
            (wgn_b is not None and wgn_b != getattr(self, '_last_wgn_b', None))
        )
        
        if needs_recreate:
            self.prep_exp_orig(hcp_feats=hcp_feats, wgn_b=wgn_b)
            # Cache the parameters used
            if hcp_feats is not None:
                self._last_hcp_feats = hcp_feats
            if wgn_b is not None:
                self._last_wgn_b = wgn_b

        # trim experiment to reasonable size (for speedup)
        # Use provided radius or fall back to self.radius
        radius_to_use = radius if radius is not None else self.radius
        if radius_to_use is None:
            exp = self.exp_orig
        else:
            extenter = glow.effect.ExtenterSphere(radius=radius_to_use)
            mask = extenter(mask_idx=self.exp_orig.mask_idx, seed=seed,
                            contiguous=True)
            exp = self.exp_orig.apply_mask(mask)

        # scale normalize before sampling minimum variance (each feature given
        # equal weight in sampling extent)
        exp = glow.experiment.ExperimentScaled.from_exp(exp)

        # sample effect space
        n = exp.y.shape[2] * self.effect_perc
        extenter = glow.effect.ExtenterMinVar(n=n)

        # impose effect
        return exp.impose_effect(extenter=extenter,
                                 seed=seed,
                                 hotel_tr=hotel_tr)

    def iter_kwargs(self):
        """Iterates through all inputs to get_exp_eff.
        
        Uses a declarative approach:
        - iter_params: dict of {param_name: [values]} to iterate over
        - fixed_params: dict of {param_name: value} for fixed values
        
        Default behavior (if iter_params is None):
        - Iterate over seed (0 to n_seed-1) and hotel_tr_all
        
        Example custom iterations:
        - {'iter_params': {'seed': range(10), 'radius': [2, 5, 10]}, 
           'fixed_params': {'hotel_tr': 0.1}}
        - {'iter_params': {'seed': range(10), 'wgn_b': [1, 2]}, 
           'fixed_params': {'hotel_tr': 0.1}}
        """
        # build iteration specification
        if self.iter_params is None:
            # default: iterate over seed and hotel_tr
            iter_spec = {
                'seed': np.arange(self.n_seed),
                'hotel_tr': self.hotel_tr_all
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

    def prep_folder(self):
        ts = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
        self.folder = path_result / self.label / ts

        if self.folder.exists():
            raise RuntimeError(f'quitting, folder exists: {self.folder}')

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

    def run_all(self, run_fnc, verbose=True):
        """run all experiments
        
        Args:
            run_fnc: function to run for each experiment
            verbose: print progress
        """
        # prep folder and save config
        self.prep_folder()
        path_config = self.folder / 'config.yaml'
        self.save_config(path=path_config)

        if verbose:
            print(f'outputs stored in: {self.folder}')
            
            path_config = self.folder / 'config.yaml'
            if path_config.exists():
                with open(path_config, 'r') as f:
                    print(f.read())

        kwargs_list = list(self.iter_kwargs())
        
        if verbose:
            print(f'running {len(kwargs_list)} experiments')

        if self.n_jobs not in (0, 1):
            Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(run_fnc)(config=self, **kwargs)
                for kwargs in tqdm(kwargs_list))
        else:
            for kwargs in tqdm(kwargs_list):
                run_fnc(config=self, **kwargs)