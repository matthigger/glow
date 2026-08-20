"""Time a shipped-shape GLOW fit on this box, float64 against float32.

The configuration the local-vs-cloud comparison is quoted at:
n_perm_fwer = 500, n_perm_inner = 250, one 25k HCP cell. Repeated, since
GPU boost clocks move single runs by several percent.

    python scripts/fit_time_500.py --rep 3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp32_drift import build_cell, fit_dtype


def main(argv=None) -> None:
    """Time both dtypes at the shipped outer count and print medians."""
    p = argparse.ArgumentParser()
    p.add_argument('--rep', type=int, default=3)
    p.add_argument('--n-perm-fwer', type=int, default=500)
    p.add_argument('--n-perm-inner', type=int, default=250)
    p.add_argument('--n-jobs', type=int, default=4)
    args = p.parse_args(argv)

    exp = build_cell()
    n_total = args.n_perm_fwer + 1
    print(f'num_vox={exp.y.shape[2]} b={exp.y.shape[0]} '
          f'n_perm_fwer={args.n_perm_fwer} '
          f'n_perm_inner={args.n_perm_inner} n_jobs={args.n_jobs}')

    for label, dtype in (('float64', np.float64), ('float32', np.float32)):
        sec = [fit_dtype(exp, dtype, n_perm_fwer=args.n_perm_fwer,
                         n_perm_inner=args.n_perm_inner,
                         n_jobs=args.n_jobs)[1] for _ in range(args.rep)]
        med = float(np.median(sec))
        print(f'{label}: median {med:6.1f} s  '
              f'({med / n_total * 1e3:5.1f} ms/outer perm)  '
              f'runs {[round(s, 1) for s in sec]}')


if __name__ == '__main__':
    main()
