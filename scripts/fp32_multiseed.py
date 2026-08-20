"""Run the float32 drift check over several cells, not one.

A single cell can flatter a dtype: whether the log difference cancels
badly depends on how close H sits to T, which varies with the plant and
the crop. This walks the first few cells of the inner-draw sweep's own
grid and reports the worst case, which is what a decision to ship float32
has to rest on.

    python scripts/fp32_multiseed.py --n-cell 6 --out out.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp32_drift import build_cell, compare, fit_dtype


def main(argv=None) -> None:
    """Fit each cell in both dtypes and print a per-cell drift table."""
    p = argparse.ArgumentParser()
    p.add_argument('--n-cell', type=int, default=6)
    p.add_argument('--n-perm-fwer', type=int, default=60)
    p.add_argument('--n-perm-inner', type=int, default=250)
    p.add_argument('--out', default=None)
    p.add_argument('--fp64-scan', action='store_true',
                   help='accumulate the per-draw region scan in float64')
    args = p.parse_args(argv)

    if args.fp64_scan:
        import torch

        import glow.analysis.draws_gpu as dg
        pass  # scan is float64 unconditionally now
        print('per-draw region scan accumulating in float64')

    rows = []
    print(f'{"cell":>5}{"max_stat rel":>14}{"pval flips":>12}'
          f'{"disc same":>11}{"fp64 s":>9}{"fp32 s":>9}{"x":>7}')
    for idx in range(args.n_cell):
        exp = build_cell(idx_data=idx)
        kw = dict(n_perm_fwer=args.n_perm_fwer,
                  n_perm_inner=args.n_perm_inner, n_jobs=4)
        ref, s64 = fit_dtype(exp, np.float64, **kw)
        got, s32 = fit_dtype(exp, np.float32, **kw)
        out = compare(ref, got)
        out.update(cell=idx, sec64=s64, sec32=s32)
        rows.append(out)
        same = out['disc_ref'] == out['disc_got']
        print(f'{idx:>5}{out["max_stat_rel"]:>14.3e}'
              f'{out["n_pval_flip"]:>12}{str(same):>11}'
              f'{s64:>9.1f}{s32:>9.1f}{s64 / s32:>6.2f}x', flush=True)
        if args.out:
            with open(args.out, 'w') as f:
                json.dump(rows, f, indent=2)

    worst = max(r['max_stat_rel'] for r in rows)
    flips = sum(r['n_pval_flip'] for r in rows)
    n_same = sum(r['disc_ref'] == r['disc_got'] for r in rows)
    print(f'\nworst max_stat rel drift : {worst:.3e}')
    print(f'total pval flips         : {flips}')
    print(f'cells with same supports : {n_same} / {len(rows)}')
    print(f'median speedup           : '
          f'{np.median([r["sec64"] / r["sec32"] for r in rows]):.2f}x')


if __name__ == '__main__':
    sys.exit(main())
