"""Measure what float32 does to a GLOW fit, against the float64 reference.

The device backend is documented as a pure speedup: same seeds, same
estimator, agreeing with the CPU fit to round-off. float64 holds that
promise; float32 historically did not, and this is the harness that says by
how much. Three signatures, because a dtype can fail at different depths:

  - max_stat: the FWER null itself, as a relative deviation.
  - pval: how many DECISIONS move, which is the only thing a reader sees.
    Compared with equal_nan, since regions below min_vox are NaN in both
    and a bare != counts every one of them as a flip.
  - effect_list: the discovered supports, the end of the pipeline.

Run it on a real benchmark cell rather than a toy: the poisoning is driven
by near-constant voxels, which synthetic Gaussian data does not have.

    python scripts/fp32_drift.py --n-perm-fwer 100 --n-perm-inner 250
"""
import argparse
import inspect
import time

import numpy as np

from glow._extra.benchmark import config
from glow._extra.benchmark.config import CONFIG
from glow._extra.benchmark.data import (DATA_FACTORY, EFFECT_FACTORY,
                                        data_recipe)
from glow.analysis import AnalysisGLOW
from glow.analysis._fit_gpu import GpuConfig


def build_cell(idx_data: int = 0, idx_effect: int = 1):
    """Build one sweep_n_perm_inner cell without touching cache or records.

    Args:
        idx_data (int): index into the cache's data grid.
        idx_effect (int): index into its effect grid.

    Returns:
        exp (Experiment): the planted experiment.
    """
    kd, ke, _, _ = CONFIG['sweep_n_perm_inner']
    kwargs_data = list(kd)[idx_data]
    kwargs_effect = dict(list(ke)[idx_effect])
    kb = {k: v for k, v in kwargs_data.items() if k != 'source'}
    exp = inspect.unwrap(DATA_FACTORY[kwargs_data['source']])(**kb)
    kind = kwargs_effect.pop('kind', 'single')
    exp, _ = inspect.unwrap(EFFECT_FACTORY[kind])(
        exp, parent_uid=data_recipe(kwargs_data).uid, **kwargs_effect)
    return exp


def fit_dtype(exp, acc_dtype, *, n_perm_fwer: int, n_perm_inner: int,
              n_jobs: int = 4):
    """Fit on device at one accumulator dtype and time it.

    Args:
        exp (Experiment): the planted experiment.
        acc_dtype (type): np.float64 or np.float32.
        n_perm_fwer (int): outer FL perms.
        n_perm_inner (int): inner FL draws per outer perm.
        n_jobs (int): Ward-pool workers feeding the device.

    Returns:
        ana (AnalysisGLOW): the fitted analysis.
        sec (float): wall seconds.
    """
    ana_kwargs = config.ana_kwargs_dict[config.REPORTED_GLOW_LABEL]
    ana = AnalysisGLOW(n_perm_fwer=n_perm_fwer, n_perm_inner=n_perm_inner,
                       alpha_fwer=config.ALPHA_FWER,
                       cluster_mode=ana_kwargs.cluster_mode,
                       prune_rule=ana_kwargs.prune_rule)
    tic = time.time()
    ana.fit(exp, n_jobs=n_jobs, gpu=GpuConfig(acc_dtype=acc_dtype))
    return ana, time.time() - tic


def compare(ref, got):
    """Report how far a fit has drifted from the float64 reference.

    Args:
        ref (AnalysisGLOW): the float64 fit.
        got (AnalysisGLOW): the fit under test.

    Returns:
        dict: {max_stat_rel, z_rel, n_pval_flip, n_pval, max_z_ref,
            max_z_got, finite_frac, disc_ref, disc_got}
    """
    a, b = ref.fwer.max_stat, got.fwer.max_stat
    denom = np.maximum(np.abs(a), 1e-12)
    max_stat_rel = float(np.nanmax(np.abs(a - b) / denom))

    za, zb = ref.fwer.stat_obs, got.fwer.stat_obs
    z_rel = float(np.nanmax(np.abs(za - zb) / np.maximum(np.abs(za), 1e-12)))

    pa, pb = ref.fwer.pval, got.fwer.pval
    same = np.isclose(pa, pb, rtol=0, atol=0, equal_nan=True)
    return dict(max_stat_rel=max_stat_rel, z_rel=z_rel,
                n_pval_flip=int((~same).sum()), n_pval=int(pa.size),
                max_z_ref=float(np.nanmax(za)),
                max_z_got=float(np.nanmax(zb)),
                finite_frac=float(np.isfinite(zb).mean()),
                disc_ref=[int(e.mask.sum()) for e in ref.effect_list],
                disc_got=[int(e.mask.sum()) for e in got.effect_list])


def main(argv=None) -> None:
    """Fit the cell in both dtypes and print the drift table."""
    p = argparse.ArgumentParser()
    p.add_argument('--n-perm-fwer', type=int, default=100)
    p.add_argument('--n-perm-inner', type=int, default=250)
    p.add_argument('--n-jobs', type=int, default=4)
    p.add_argument('--idx-data', type=int, default=0)
    args = p.parse_args(argv)

    exp = build_cell(idx_data=args.idx_data)
    print(f'cell {args.idx_data}: b={exp.y.shape[0]} '
          f'num_img={exp.y.shape[1]} num_vox={exp.y.shape[2]}  '
          f'n_perm_fwer={args.n_perm_fwer} n_perm_inner={args.n_perm_inner}')

    kw = dict(n_perm_fwer=args.n_perm_fwer,
              n_perm_inner=args.n_perm_inner, n_jobs=args.n_jobs)
    ref, sec64 = fit_dtype(exp, np.float64, **kw)
    got, sec32 = fit_dtype(exp, np.float32, **kw)
    out = compare(ref, got)

    print(f'\nfp64 {sec64:6.1f} s     fp32 {sec32:6.1f} s     '
          f'speedup {sec64 / sec32:.2f}x')
    print(f'max_stat rel drift : {out["max_stat_rel"]:.3e}')
    print(f'z_obs    rel drift : {out["z_rel"]:.3e}')
    print(f'pval flips         : {out["n_pval_flip"]} of {out["n_pval"]}')
    print(f'max z  fp64 / fp32 : {out["max_z_ref"]:.4f} / '
          f'{out["max_z_got"]:.4f}')
    print(f'finite z fraction  : {out["finite_frac"]:.4f}')
    print(f'discovered  fp64   : {out["disc_ref"]}')
    print(f'discovered  fp32   : {out["disc_got"]}')


if __name__ == '__main__':
    main()
