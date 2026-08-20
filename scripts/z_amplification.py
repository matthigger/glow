"""Measure the amplification the z-standardization applies to LLR error.

The inner null standardizes each region as z = (llr - mu) / std, and the
LLR distribution has var << mean^2 (which is why the CPU path accumulates
Chan-parallel moments at all). So a RELATIVE error eps in llr lands on z
as an ABSOLUTE error of about eps * mu / std: the ratio mu / std is a
second error amplifier, downstream of the LLR itself and independent of
how the LLR is formed.

That ratio bounds what any narrow accumulator can deliver, so it is the
number to look at before deciding float32 is safe.

    python scripts/z_amplification.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp32_drift import build_cell

import glow.graph
from glow._extra.benchmark import config
from glow.analysis import draws
from glow.analysis._fit_gpu import GpuConfig, gpu_summary, resolve_gpu
from glow.analysis.cluster import cluster
from glow.analysis.mancova import decompose
from glow.experiment.exper import ExperimentScaled

exp = ExperimentScaled.from_exp(build_cell())
mode = config.ana_kwargs_dict[config.REPORTED_GLOW_LABEL].cluster_mode
q0, q1, _ = decompose(x=exp.x, contrast=exp.contrast)
exp_k = exp.permute(1)
children = cluster(exp_k, mode=mode)
_, rl, rh = glow.graph.build_dfs_preorder(children=children,
                                          num_vox=exp_k.y.shape[2])
size = rh - rl
reg_active = size >= 1

dk = dict(exp=exp_k, base_seed=0, n_perm=251, q0=q0, q1=q1,
          children=children, min_vox=1)
s64 = gpu_summary(GpuConfig(acc_dtype=np.float64), reg_active=reg_active,
                  **dk)
s32 = gpu_summary(GpuConfig(acc_dtype=np.float32), reg_active=reg_active,
                  **dk)

ok = np.isfinite(s64.mu) & np.isfinite(s64.std) & (s64.std > 0)
ratio = np.abs(s64.mu[ok]) / s64.std[ok]
print(f'regions with usable moments : {ok.sum()} of {ok.size}')
print(f'|mu| / std   median         : {np.median(ratio):.1f}')
print(f'             90th pct       : {np.percentile(ratio, 90):.1f}')
print(f'             max            : {ratio.max():.1f}')

llr_rel = np.abs(s32.llr[ok] - s64.llr[ok]) / np.maximum(
    np.abs(s64.llr[ok]), 1e-30)
z_abs = np.abs(s32.z_obs[ok] - s64.z_obs[ok])
print(f'\nllr relative error (fp32)   : median {np.median(llr_rel):.2e}'
      f'  max {llr_rel.max():.2e}')
print(f'z absolute error   (fp32)   : median {np.median(z_abs):.2e}'
      f'  max {z_abs.max():.2e}')
print(f'predicted z err = llr_rel * mu/std (median): '
      f'{np.median(llr_rel) * np.median(ratio):.2e}')
