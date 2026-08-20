"""Price a float64 region scan inside an otherwise float32 hot loop.

_reg_sum_cumsum differences two prefix sums, and in float32 the prefixes
grow along the DFS axis while the answer does not, so a small region keeps
only the low digits of a large gap. Accumulating just that scan in float64
and casting the region sums back leaves every other per-draw tensor
float32. This asks what that buys and what it costs.

    python scripts/scan_acc_ab.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp32_drift import build_cell, compare, fit_dtype

import glow.analysis.draws_gpu as dg

exp = build_cell()
kw = dict(n_perm_fwer=100, n_perm_inner=250, n_jobs=4)

dg._SCAN_ACC = None
ref, sec64 = fit_dtype(exp, np.float64, **kw)
print(f'fp64 reference                     : {sec64:.1f} s')

for label, scan_acc in (('fp32, fp32 scan', None),
                        ('fp32, fp64 scan', torch.float64)):
    dg._SCAN_ACC = scan_acc
    got, sec = fit_dtype(exp, np.float32, **kw)
    out = compare(ref, got)
    print(f'{label:<35}: {sec:5.1f} s  '
          f'({sec64 / sec:.2f}x)  drift {out["max_stat_rel"]:.3e}  '
          f'flips {out["n_pval_flip"]}  '
          f'supports {"same" if out["disc_ref"] == out["disc_got"] else "DIFF"}')
dg._SCAN_ACC = None
