"""Time a GLOW fit at each inner-draw count, on one sweep_n_perm_inner cell.

The cost side of the n_perm_inner question. sweep_n_perm_inner records what
each count buys (Dice, and which region the max-z came from) but not what it
costs: a cell's counts share one capture, so a leaf's time_sec is the whole
walk's, not that count's. This fits AnalysisGLOW once per count and times it.

    python scripts/time_n_perm_inner.py --n-perm-fwer 250

Worth measuring rather than deriving from the draw count. A fit is
n_perm_fwer x (n_perm_inner + 1) inner draws plus one Ward tree per outer
perm, and only the first term moves with the count, so the two ends of the
grid are far closer in wall time than in draws. Time is linear in
n_perm_fwer, so one outer count rescales to any other.

The cell is the sweep's own first cell, built through the raw builders
(inspect.unwrap) so an ad-hoc timing cannot land on a benchmark cell's key
and rewrite its record. A time is the machine that recorded it; compare only
within one run on one box.
"""

import argparse
import inspect
import json
import time

import numpy as np

from glow._extra.benchmark import config
from glow._extra.benchmark.config import CONFIG
from glow._extra.benchmark.data import (DATA_FACTORY, EFFECT_FACTORY,
                                        data_recipe)
from glow.analysis import AnalysisGLOW


def build_cell(idx_data: int = 0, idx_effect: int = 1):
    """Build one sweep_n_perm_inner cell, touching no cache or record.

    Args:
        idx_data (int): which data cell of the cache's grid to build.
        idx_effect (int): which effect cell (its llr axis is ascending).

    Returns:
        exp (Experiment): the planted experiment to fit.
        mask_target (np.array): (X, Y, Z) bool, the realized support.
    """
    kwargs_data_list, kwargs_effect_list, _, _ = CONFIG['sweep_n_perm_inner']
    kwargs_data = list(kwargs_data_list)[idx_data]
    kwargs_effect = dict(list(kwargs_effect_list)[idx_effect])
    print(f'cell data:   {kwargs_data}')
    print(f'cell effect: {kwargs_effect}')

    kwargs_build = {k: v for k, v in kwargs_data.items() if k != 'source'}
    exp = inspect.unwrap(DATA_FACTORY[kwargs_data['source']])(**kwargs_build)
    kind = kwargs_effect.pop('kind', 'single')
    exp, mask_target_list = inspect.unwrap(EFFECT_FACTORY[kind])(
        exp, parent_uid=data_recipe(kwargs_data).uid, **kwargs_effect)
    return exp, mask_target_list[0]


def time_counts(exp, count_list, n_perm_fwer: int) -> list:
    """Fit and time AnalysisGLOW once per inner-draw count.

    Args:
        exp (Experiment): the planted experiment to fit.
        count_list (list[int]): the n_perm_inner values to time.
        n_perm_fwer (int): outer FL perms, held fixed across the counts.

    Returns:
        list[dict]: one record per count, {n_perm_inner, n_perm_fwer,
            time_sec, sec_per_outer, n_draw, z_ceiling, max_z}.
    """
    ana_kwargs = config.ana_kwargs_dict[config.REPORTED_GLOW_LABEL]
    fit_params = dict(config.GLOW_FIT_PARAMS)
    print(f'fit_params={fit_params}  n_perm_fwer={n_perm_fwer}')

    rows = []
    for n_perm_inner in count_list:
        ana = AnalysisGLOW(n_perm_fwer=n_perm_fwer,
                           n_perm_inner=n_perm_inner,
                           alpha_fwer=config.ALPHA_FWER,
                           cluster_mode=ana_kwargs.cluster_mode,
                           prune_rule=ana_kwargs.prune_rule)
        tic = time.time()
        ana.fit(exp, **fit_params)
        sec = time.time() - tic
        rows.append(dict(n_perm_inner=n_perm_inner, n_perm_fwer=n_perm_fwer,
                         time_sec=sec, sec_per_outer=sec / n_perm_fwer,
                         n_draw=n_perm_fwer * (n_perm_inner + 1),
                         z_ceiling=n_perm_inner / np.sqrt(n_perm_inner + 1),
                         max_z=float(np.nanmax(ana.fwer.stat_obs))))
        print(f'n_perm_inner={n_perm_inner:>4}  {sec:8.1f} s  '
              f'({sec / n_perm_fwer * 1e3:6.1f} ms/outer perm, '
              f'{rows[-1]["n_draw"]:>7} draws)  '
              f'max_z={rows[-1]["max_z"]:.2f} '
              f'(ceiling {rows[-1]["z_ceiling"]:.2f})', flush=True)
    return rows


def main(argv=None) -> None:
    """Time every count on the grid and optionally write the records out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n-perm-fwer', type=int,
                        default=config.INNER_N_PERM_FWER,
                        help='outer perms (the cache\'s, so a time pairs '
                             'with the Dice read off it)')
    parser.add_argument('--counts', type=int, nargs='+',
                        default=list(config.N_PERM_INNER_GRID))
    parser.add_argument('--out', default=None, help='write the records here')
    args = parser.parse_args(argv)

    exp, mask_target = build_cell()
    print(f'exp: b={exp.y.shape[0]} num_img={exp.y.shape[1]} '
          f'num_vox={exp.y.shape[2]}  plant={int(mask_target.sum())} vox')

    rows = time_counts(exp, args.counts, args.n_perm_fwer)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=2)
        print(f'saved: {args.out}')


if __name__ == '__main__':
    main()
