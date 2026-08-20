"""Dump a fitted GLOW cell's reported arrays, for cross-revision compare.

Companion to fp32_drift: that one asks what a dtype costs, this one asks
whether a code change moved the float64 answer at all.

    python scripts/dump_cell_fit.py out.npz float64
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp32_drift import build_cell, fit_dtype

out_path = sys.argv[1]
acc_dtype = np.float64 if sys.argv[2] == 'float64' else np.float32
n_perm_fwer = int(sys.argv[3]) if len(sys.argv) > 3 else 60

import glow.analysis.draws_gpu as dg
print(f'draws_gpu: {dg.__file__}')
print(f'has _quad_form_inv: {hasattr(dg, "_quad_form_inv")}')

exp = build_cell()
ana, sec = fit_dtype(exp, acc_dtype, n_perm_fwer=n_perm_fwer,
                     n_perm_inner=250, n_jobs=4)
np.savez(out_path, llr=ana.llr, mu=ana.mu, std=ana.std,
         max_stat=ana.fwer.max_stat, stat_obs=ana.fwer.stat_obs,
         pval=ana.fwer.pval, size=ana.size,
         disc=np.array([e.mask.sum() for e in ana.effect_list]))
print(f'saved {out_path}  ({sec:.1f} s)')
