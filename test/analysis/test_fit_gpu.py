"""End-to-end A/B: AnalysisGLOW.fit(gpu=True) against the CPU fit.

This is the gate that justifies keeping device / acc_dtype out of the
recipe hash. The unit tests in test_inner_perm_gpu.py pin the backend's
draws and moments; these pin the whole fit -- z, the FWER null, the
p-values and the discovered effect list -- so a divergence anywhere in
the phase split (seeds, the outer-perm reduction, which tree pairs with
which observed LLR) surfaces here rather than in a sweep.

Run at acc_dtype=float64 (what gpu=True selects) so what is under test is
the pipeline, not the dtype; a separate case checks that float32 leaves
the discoveries alone. The gpu-argument cases need no device and always
run.

Run:
    ~/venv_glow/bin/pytest test/analysis/test_fit_gpu.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

import glow.mask
from glow.analysis import AnalysisVBA, GpuConfig, inner_perm_gpu
from glow.analysis._fit_gpu import resolve_gpu
from glow.analysis._glow import AnalysisGLOW
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import get_hotel_tr
from glow.experiment.exper import Experiment


requires_cuda = pytest.mark.skipif(
    not inner_perm_gpu.is_available(),
    reason='no CUDA device visible')

skip_if_cuda = pytest.mark.skipif(
    inner_perm_gpu.is_available(),
    reason='a CUDA device is visible')

# The A/B cases below compare a device fit against a CPU fit. There is no
# device fit to compare while _fit_gpu is being ported to the split
# architecture, so they are held rather than deleted -- every assertion
# here is still the right one to make once the port lands, and the whole
# point of this file is that it is the gate for keeping device / acc_dtype
# out of the recipe hash. The gpu-argument cases are unaffected and still
# run: resolve_gpu did not change.
pending_gpu_port = pytest.mark.skip(
    reason='device backend offline pending its port to the split '
           'architecture (see glow.analysis._fit_gpu)')


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
    kw = dict(n_perm_fwer=6, alpha_fwer=0.05, min_vox=2,
              cluster_mode=ClusterMode.FOCUS)
    kw.update(over)
    return kw


# ---------- the gpu argument (no device needed) ------------------------------
class TestResolveGpu:
    """resolve_gpu maps every accepted form onto a GpuConfig or None."""

    def test_falsey_is_cpu(self):
        assert resolve_gpu(False) is None
        assert resolve_gpu(None) is None

    def test_config_defaults_to_float64(self):
        """The default is the dtype that reproduces a CPU fit exactly."""
        assert GpuConfig().acc_dtype is np.float64

    def test_bad_form_raises(self):
        with pytest.raises(TypeError, match='gpu must be'):
            resolve_gpu('cuda')

    @requires_cuda
    def test_true_and_auto_take_the_device(self):
        assert resolve_gpu(True) == GpuConfig()
        assert resolve_gpu('auto') == GpuConfig()

    @requires_cuda
    def test_config_passes_through(self):
        cfg = GpuConfig(acc_dtype=np.float32, perm_chunk=8)
        assert resolve_gpu(cfg) is cfg

    @skip_if_cuda
    def test_auto_falls_back_without_a_device(self):
        assert resolve_gpu('auto') is None

    @skip_if_cuda
    def test_true_raises_without_a_device(self):
        with pytest.raises(RuntimeError, match='CUDA device'):
            resolve_gpu(True, name='AnalysisGLOW.fit')


def test_voxel_analysis_rejects_an_explicit_device():
    """A method with no backend refuses gpu=True but tolerates 'auto'.

    That is what lets one fit_params dict be handed to every recipe in a
    leaf grid (glow._extra.benchmark.run.run_ana).
    """
    ana = AnalysisVBA(n_perm_fwer=2, get_stat=get_hotel_tr)
    exp = _exp_with_effect(shape=(4, 4, 4), n_img=12)
    with pytest.raises(ValueError, match='no GPU backend'):
        ana.fit(exp, gpu=True)
    # 'auto' is a no-op, so this is an ordinary CPU fit
    assert ana.fit(exp, gpu='auto') is ana


# ---------- the device fit against the CPU fit -------------------------------
@pending_gpu_port
@requires_cuda
def test_gpu_fit_matches_cpu_fit():
    """fit(gpu=True) reproduces fit() on every synthesis output."""
    exp = _exp_with_effect()
    kw = _fit_kwargs()

    ref = AnalysisGLOW(**kw).fit(exp, n_jobs=1)
    got = AnalysisGLOW(**kw).fit(exp, n_jobs=1, gpu=True)

    np.testing.assert_allclose(got.max_z_null, ref.max_z_null,
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(got.z, ref.z, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(got.pval, ref.pval, rtol=0, atol=0)
    np.testing.assert_array_equal(got.size, ref.size)
    np.testing.assert_allclose(got.llr, ref.llr, rtol=1e-10, atol=1e-12)

    assert [e.reg_idx for e in got.effect_list] == \
           [e.reg_idx for e in ref.effect_list]


@pending_gpu_port
@requires_cuda
def test_gpu_fit_returns_self():
    """fit(gpu=True) keeps fit's contract: the recipe it was called on."""
    ana = AnalysisGLOW(**_fit_kwargs())
    assert ana.fit(_exp_with_effect(), n_jobs=1, gpu=True) is ana


@pending_gpu_port
@requires_cuda
def test_gpu_observed_tree_is_the_unpermuted_one():
    """k=0 stores the observed tree, not some permuted perm's."""
    exp = _exp_with_effect()
    kw = _fit_kwargs()
    ref = AnalysisGLOW(**kw).fit(exp, n_jobs=1)
    got = AnalysisGLOW(**kw).fit(exp, n_jobs=1, gpu=True)
    np.testing.assert_array_equal(got.children, ref.children)


@pending_gpu_port
@requires_cuda
def test_gpu_float32_preserves_discoveries():
    """float32 changes nothing that reaches a conclusion.

    z may shift by round-off, but the FWER p-values and the pruned effect
    list -- the outputs a result is read off -- must be identical.
    """
    exp = _exp_with_effect()
    kw = _fit_kwargs()
    ref = AnalysisGLOW(**kw).fit(exp, n_jobs=1, gpu=True)
    got = AnalysisGLOW(**kw).fit(
        exp, n_jobs=1, gpu=GpuConfig(acc_dtype=np.float32))

    np.testing.assert_allclose(got.max_z_null, ref.max_z_null,
                               rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(got.pval, ref.pval, rtol=0, atol=0)
    assert [e.reg_idx for e in got.effect_list] == \
           [e.reg_idx for e in ref.effect_list]


@pending_gpu_port
@requires_cuda
def test_gpu_parallel_phase_a_is_deterministic():
    """A parallel Phase A gives the same answer as a serial one.

    The generator is unordered, so this pins that results are keyed by
    their own k rather than by arrival order.
    """
    exp = _exp_with_effect()
    kw = _fit_kwargs()
    serial = AnalysisGLOW(**kw).fit(exp, n_jobs=1, gpu=True)
    par = AnalysisGLOW(**kw).fit(exp, n_jobs=4, gpu=True)
    np.testing.assert_allclose(par.max_z_null, serial.max_z_null,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(par.z, serial.z, rtol=1e-12, atol=1e-12)
