"""End-to-end A/B: the pipelined GPU driver against AnalysisGLOW.fit.

This is the gate that justifies keeping device / acc_dtype out of the
recipe hash. The unit tests in test_inner_perm_gpu.py pin the backend's
draws and moments; these pin the whole fit -- z, the FWER null, the
p-values and the discovered effect list -- so a divergence anywhere in
the phase split (seeds, the outer-perm reduction, which tree pairs with
which observed LLR) surfaces here rather than in a sweep.

Run at acc_dtype=float64 so what is under test is the pipeline, not the
dtype; a separate case checks that float32 leaves the discoveries alone.

Run:
    ~/venv_glow/bin/pytest test/analysis/test_driver_gpu_local.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

import glow.mask
from glow.analysis import inner_perm_gpu
from glow.analysis._glow import AnalysisGLOW
from glow.analysis.cluster import ClusterMode
from glow.analysis.driver_gpu_local import driver_gpu_local
from glow.experiment.exper import Experiment


requires_cuda = pytest.mark.skipif(
    not inner_perm_gpu.is_available(),
    reason='no CUDA device visible')


def _exp_with_effect(b=2, n_img=30, shape=(6, 6, 6), beta=1.5, seed=1):
    """Build a small exp with a planted block effect on the first contrast.

    A real effect makes the comparison meaningful: without one the
    discovered-effect lists would both be empty and agree vacuously.
    """
    rng = np.random.default_rng(seed)
    num_vox = int(np.prod(shape))
    y = rng.standard_normal((b, n_img, num_vox))
    x = np.vstack([
        np.ones(n_img),
        rng.standard_normal(n_img),
    ])
    contrast = np.array([False, True])

    mask = np.zeros(shape, dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    mask_idx_full = glow.mask.get_mask_idx(np.ones(shape, dtype=bool))
    planted = mask_idx_full[mask]
    y[:, :, planted] += beta * x[1][None, :, None]
    return Experiment(x=x, y=y, contrast=contrast, mask_idx=mask_idx_full)


def _fit_kwargs(**over):
    kw = dict(n_perm_fwer=6, n_perm_inner=16, alpha_fwer=0.05, min_vox=2,
              cluster_mode=ClusterMode.FOCUS)
    kw.update(over)
    return kw


@requires_cuda
def test_driver_matches_analysis_fit():
    """The driver reproduces AnalysisGLOW.fit on every synthesis output."""
    exp = _exp_with_effect()
    kw = _fit_kwargs()

    ref = AnalysisGLOW(**kw).fit(exp, n_jobs=1)
    got = driver_gpu_local(exp, n_jobs_cpu=1, acc_dtype=np.float64, **kw)

    np.testing.assert_allclose(got.max_z_null, ref.max_z_null,
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(got.z, ref.z, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(got.pval, ref.pval, rtol=0, atol=0)
    np.testing.assert_array_equal(got.size, ref.size)
    np.testing.assert_allclose(got.llr, ref.llr, rtol=1e-10, atol=1e-12)

    assert [e.reg_idx for e in got.effect_list] == \
           [e.reg_idx for e in ref.effect_list]


@requires_cuda
def test_driver_observed_tree_is_the_unpermuted_one():
    """k=0 stores the observed tree, not some permuted perm's."""
    exp = _exp_with_effect()
    kw = _fit_kwargs()
    ref = AnalysisGLOW(**kw).fit(exp, n_jobs=1)
    got = driver_gpu_local(exp, n_jobs_cpu=1, acc_dtype=np.float64, **kw)
    np.testing.assert_array_equal(got.children, ref.children)


@requires_cuda
def test_driver_float32_preserves_discoveries():
    """float32 changes nothing that reaches a conclusion.

    z may shift by round-off, but the FWER p-values and the pruned effect
    list -- the outputs a result is read off -- must be identical.
    """
    exp = _exp_with_effect()
    kw = _fit_kwargs()
    ref = driver_gpu_local(exp, n_jobs_cpu=1, acc_dtype=np.float64, **kw)
    got = driver_gpu_local(exp, n_jobs_cpu=1, acc_dtype=np.float32, **kw)

    np.testing.assert_allclose(got.max_z_null, ref.max_z_null,
                               rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(got.pval, ref.pval, rtol=0, atol=0)
    assert [e.reg_idx for e in got.effect_list] == \
           [e.reg_idx for e in ref.effect_list]


@requires_cuda
def test_driver_parallel_phase_a_is_deterministic():
    """A parallel Phase A gives the same answer as a serial one.

    The generator is unordered, so this pins that results are keyed by
    their own k rather than by arrival order.
    """
    exp = _exp_with_effect()
    kw = _fit_kwargs()
    serial = driver_gpu_local(exp, n_jobs_cpu=1, acc_dtype=np.float64, **kw)
    par = driver_gpu_local(exp, n_jobs_cpu=4, acc_dtype=np.float64, **kw)
    np.testing.assert_allclose(par.max_z_null, serial.max_z_null,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(par.z, serial.z, rtol=1e-12, atol=1e-12)
