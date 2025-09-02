from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Literal, Optional, Tuple, List

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
    hcp_path: str = '/home/matt/Dropbox/pnl_hglm/data/hcp100_lowres/image'
    hcp_feats: List[str] = field(default_factory=lambda: ['FA', 'MD'])

    # WGN
    wgn_shape: Tuple = (5, 5, 5)
    wgn_a: int = 2
    wgn_b: int = 2
    wgn_num_img: int = 100
    exp_seed: int = 0

    # region restriction
    radius: Optional[int] = None

    def __post_init__(self):
        self.exp_orig = None
        self.folder = None

    def prep_exp_orig(self):
        if self.source == 'hcp':
            img_glob_dict = {feat: f'*_{feat}.nii.gz' for feat in
                             self.hcp_feats}
            exp = glow.experiment.ExperimentImageOnly.from_search(
                folder=self.hcp_path,
                sbj_regex=r'[\d]{6}',
                img_glob_dict=img_glob_dict)
            self.exp_orig = exp.sample_x(a=2, seed=self.exp_seed,
                                         add_bias=True)
        elif self.source == 'wgn':
            self.exp_orig = glow.experiment.Experiment.from_gauss(
                seed=self.exp_seed,
                shape=self.wgn_shape,
                a=self.wgn_a,
                b=self.wgn_b,
                num_img=self.wgn_num_img)

    def get_exp_eff(self, seed, hotel_tr):
        if self.exp_orig is None:
            self.prep_exp_orig()

        # trim experiment to reasonable size (for speedup)
        if self.radius is None:
            exp = self.exp_orig
        else:
            extenter = glow.effect.ExtenterSphere(radius=self.radius)
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
        """ iterates through all inputs to get_exp_eff"""
        for s, h in product(np.arange(self.n_seed), self.hotel_tr_all):
            yield dict(seed=s, hotel_tr=h)

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
        self.prep_folder()
        path_config = self.folder / 'config.yaml'
        self.save_config(path=path_config)

        if verbose:
            print(f'outputs stored in: {self.folder}')
            with open(path_config, 'r') as f:
                print(f.read())

        kwargs_list = list(self.iter_kwargs())

        if self.n_jobs not in (0, 1):
            Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(run_fnc)(config=self, **kwargs)
                for kwargs in tqdm(kwargs_list))
        else:
            for kwargs in tqdm(kwargs_list):
                run_fnc(config=self, **kwargs)
