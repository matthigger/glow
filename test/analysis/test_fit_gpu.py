"""End-to-end A/B: AnalysisGLOW.fit(gpu=True) against the CPU fit.

This is the gate that justifies keeping device / acc_dtype out of the
recipe hash. The unit tests in test_draws_gpu.py pin the backend's
draws; these pin the whole fit -- z, the FWER null, the p-values and the
discovered effect list -- so a divergence anywhere downstream of the
draws (the column moments, the max-z null, pruning) surfaces here rather
than in a sweep.

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
from glow.analysis import AnalysisVBA, GpuConfig, draws_gpu
from glow.analysis._fit_gpu import resolve_gpu, resolve_perm_chunk
from glow.analysis._glow import AnalysisGLOW
from glow.analysis.cluster import ClusterMode
from glow.analysis.mancova import get_hotel_tr
from glow.experiment.exper import Experiment


requires_cuda = pytest.mark.skipif(
    not draws_gpu.is_available(),
    reason='no CUDA device visible')

skip_if_cuda = pytest.mark.skipif(
    draws_gpu.is_available(),
    reason='a CUDA device is visible')


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
@requires_cuda
def test_gpu_fit_matches_cpu_fit():
    """fit(gpu=True) reproduces fit() on every synthesis output."""
    exp = _exp_with_effect()
    kw = _fit_kwargs()

    ref = AnalysisGLOW(**kw).fit(exp, n_jobs=1)
    got = AnalysisGLOW(**kw).fit(exp, n_jobs=1, gpu=True)

    np.testing.assert_allclose(got.fwer.max_stat, ref.fwer.max_stat,
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(got.fwer.stat_obs, ref.fwer.stat_obs,
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(got.fwer.pval, ref.fwer.pval, rtol=0, atol=0)
    np.testing.assert_array_equal(got.size, ref.size)
    np.testing.assert_allclose(got.llr, ref.llr, rtol=1e-10, atol=1e-12)

    assert [e.reg_idx for e in got.effect_list] == \
           [e.reg_idx for e in ref.effect_list]


@requires_cuda
def test_gpu_fit_returns_self():
    """fit(gpu=True) keeps fit's contract: the recipe it was called on."""
    ana = AnalysisGLOW(**_fit_kwargs())
    assert ana.fit(_exp_with_effect(), n_jobs=1, gpu=True) is ana


@requires_cuda
def test_gpu_tree_is_the_cpu_tree():
    """The device changes the draws only, never the hypothesis family.

    The tree comes from the segmentation fold on the CPU either way, so a
    device fit that differed here would be testing a different family.
    """
    exp = _exp_with_effect()
    kw = _fit_kwargs()
    ref = AnalysisGLOW(**kw).fit(exp, n_jobs=1)
    got = AnalysisGLOW(**kw).fit(exp, n_jobs=1, gpu=True)
    np.testing.assert_array_equal(got.children, ref.children)


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

    np.testing.assert_allclose(got.fwer.max_stat, ref.fwer.max_stat,
                               rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(got.fwer.pval, ref.fwer.pval, rtol=0, atol=0)
    assert [e.reg_idx for e in got.effect_list] == \
           [e.reg_idx for e in ref.effect_list]


@requires_cuda
def test_gpu_keep_stat_matches_the_cpu_matrix():
    """keep_stat gives up the streaming reduction, not the numbers.

    The device path has no matrix to keep -- gpu_summarize folds each
    chunk away -- so keep_stat routes it through gpu_perm instead. That
    is a different code path from the streamed one, hence this: the
    matrix it hands back is the CPU's, and the summary read off it is
    still the summary the streaming fit would have produced.
    """
    exp = _exp_with_effect()
    kw = _fit_kwargs()

    cpu = AnalysisGLOW(**kw, keep_stat=True).fit(exp, n_jobs=1)
    got = AnalysisGLOW(**kw, keep_stat=True).fit(exp, n_jobs=1, gpu=True)
    streamed = AnalysisGLOW(**kw).fit(exp, n_jobs=1, gpu=True)

    np.testing.assert_allclose(got.stat, cpu.stat, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(got.llr, got.stat[0], rtol=0, atol=0,
                               equal_nan=True)
    # the kept matrix does not perturb what the fit concludes
    np.testing.assert_allclose(got.fwer.pval, streamed.fwer.pval,
                               rtol=0, atol=0)
    np.testing.assert_allclose(got.fwer.max_stat, streamed.fwer.max_stat,
                               rtol=1e-7, atol=1e-9)


@requires_cuda
def test_gpu_fit_ignores_n_jobs():
    """n_jobs cannot move a device fit, which is why it is not in the hash.

    Analysis.fit promises a fit is identical at any n_jobs; the draws are
    seeded by index and the device loop is serial, so this pins that the
    promise still holds on the device path.
    """
    exp = _exp_with_effect()
    kw = _fit_kwargs()
    serial = AnalysisGLOW(**kw).fit(exp, n_jobs=1, gpu=True)
    par = AnalysisGLOW(**kw).fit(exp, n_jobs=4, gpu=True)
    np.testing.assert_allclose(par.fwer.max_stat, serial.fwer.max_stat,
                               rtol=0, atol=0)
    np.testing.assert_allclose(par.fwer.stat_obs, serial.fwer.stat_obs,
                               rtol=0, atol=0)


# ---------- chunk sizing -----------------------------------------------------
class TestResolvePermChunk:
    """perm_chunk is sized from free device memory unless pinned."""

    def test_an_explicit_int_passes_through(self):
        """No device needed: an int short-circuits the memory query."""
        cfg = GpuConfig(perm_chunk=3)
        assert resolve_perm_chunk(
            cfg, b=6, num_img=100, num_vox=224_619, a0=2) == 3

    @requires_cuda
    def test_small_b_takes_the_maximum(self):
        """A cheap chunk is capped by throughput, not by memory."""
        got = resolve_perm_chunk(GpuConfig(), b=1, num_img=30,
                                 num_vox=1_000, a0=2)
        assert got == 16

    @requires_cuda
    def test_full_brain_b6_is_clamped_below_the_maximum(self):
        """The case that OOMed at a fixed 16 on an 8 GiB card.

        b = 6 is the paper's full DKI + NODDI panel, and 224,619 is the HCP
        support, so this is a configuration the benchmark actually fits --
        not a synthetic corner.
        """
        got = resolve_perm_chunk(GpuConfig(), b=6, num_img=100,
                                 num_vox=224_619, a0=2)
        assert 1 <= got < 16

    @requires_cuda
    def test_never_returns_zero(self):
        """An absurd problem still yields a runnable chunk of 1."""
        got = resolve_perm_chunk(GpuConfig(), b=64, num_img=1_000,
                                 num_vox=10_000_000, a0=8)
        assert got == 1
