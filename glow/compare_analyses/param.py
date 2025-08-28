from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import yaml
from joblib import Parallel, delayed
from platformdirs import user_data_dir
from tqdm import tqdm

import glow
from glow.compare_analyses.run import run

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
    ana_kwargs_dict: dict

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

    # fwer bound
    alpha_fwer: float = 0.05

    # number of jobs (each runs another effect).  1 is serial, -1 runs as many
    # as the computer has threads
    n_jobs: int = 1

    # -------- persistence / debugging --------
    detail_save: bool = True
    error_save: bool = False

    # -------- source-specific --------
    # HCP
    hcp_path: str = '/home/matt/Dropbox/pnl_hglm/data/hcp100_lowres/image'

    # WGN
    wgn_shape: Tuple = (5, 5, 5)
    wgn_a: int = 2
    wgn_b: int = 2
    wgn_num_img: int = 100
    exp_seed: int = 0

    # region restriction
    radius: Optional[int] = None

    def build_experiment(self):
        if self.source == 'hcp':
            exp = glow.experiment.ExperimentImageOnly.from_search(
                folder=self.hcp_path,
                sbj_regex=r'[\d]{6}',
                img_glob_dict={'FA': '*_FA.nii.gz', 'MD': '*_MD.nii.gz'})
            exp.sample_x(a=2, seed=self.exp_seed, add_bias=True)
        elif self.source == 'wgn':
            exp = glow.experiment.Experiment.from_gauss(
                seed=self.exp_seed,
                shape=self.wgn_shape,
                a=self.wgn_a,
                b=self.wgn_b,
                num_img=self.wgn_num_img)

        return exp

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
            if isinstance(obj, type):
                # classes like AnalysisGLOW
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
        self.prep_folder()
        path_config = self.folder / 'config.yaml'
        self.save_config(path=path_config)

        if verbose:
            print(f'outputs stored in: {self.folder}')
            with open(path_config, 'r') as f:
                print(f.read())

        self.exp = self.build_experiment()

        kwargs_list = [dict(seed=s, hotel_tr=h) for s, h in
                       product(np.arange(self.n_seed), self.hotel_tr_all)]

        if self.n_jobs not in (0, 1):
            Parallel(n_jobs=self.n_jobs, verbose=10)(
                delayed(run)(config=self, **kwargs)
                for kwargs in tqdm(kwargs_list))
        else:
            for kwargs in tqdm(kwargs_list):
                run(config=self, **kwargs)
