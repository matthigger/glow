import numpy as np
import pytest

from glow.effect.impose import (compute_offset, impose_effect,
                                sample_beta_direction)
from glow.experiment import Experiment
from glow.analysis.mancova import decompose
from glow.analysis.mancova import get_mancova
from glow.analysis.mancova import get_llr


@pytest.mark.parametrize('effect_llr_exp', [0, 0.5, 1, 3])
def test_compute_offset(effect_llr_exp):
    """Offset achieves target LLR (round-trip through get_mancova/get_llr)."""
    exp = Experiment.from_gauss(seed=0)
    x, y, contrast = exp.x, exp.y, exp.contrast

    offset, sigma_scale = compute_offset(
        x=x, y=y, contrast=contrast, effect_llr=effect_llr_exp)

    assert sigma_scale is None

    _y = y + offset[..., np.newaxis]
    e, h, _ = get_mancova(x=x, y=_y, contrast=contrast)
    effect_llr_obs = get_llr(e, h, n=1)

    assert np.isclose(effect_llr_exp, effect_llr_obs)


def _fro_angle_deg(a, b):
    """Angle (degrees) between two arrays under the Frobenius inner product."""
    a, b = a.ravel(), b.ravel()
    c = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _make_region(b=3, num_img=80, k=200, seed=0, n_interest=1, a=4):
    """Build (x, y, contrast) for a region: bias + nuisance + interest cols."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((a, num_img))
    x[0] = 1.0
    contrast = np.zeros(a, dtype=bool)
    contrast[a - n_interest:] = True
    y = rng.standard_normal((b, num_img, k)) + 0.3
    return x, y, contrast


class TestSampleBetaDirection:
    @pytest.mark.parametrize('a1, b', [(1, 2), (1, 5), (2, 3)])
    def test_unit_norm_and_shape(self, a1, b):
        beta = sample_beta_direction(a1=a1, b=b, angle=37.0, seed=0)
        assert beta.shape == (a1, b)
        assert np.isclose(np.linalg.norm(beta), 1.0)

    @pytest.mark.parametrize('t1, t2', [(0, 60), (-30, 30), (0, 90),
                                        (10, 100), (45, 45)])
    def test_angle_is_difference(self, t1, t2):
        # the whole point: sharing a seed, two draws sit exactly |t1-t2| apart
        beta1 = sample_beta_direction(a1=1, b=4, angle=t1, seed=7)
        beta2 = sample_beta_direction(a1=1, b=4, angle=t2, seed=7)
        assert np.isclose(_fro_angle_deg(beta1, beta2), abs(t1 - t2),
                          atol=1e-6)

    def test_same_args_identical(self):
        a = sample_beta_direction(a1=1, b=4, angle=20.0, seed=3)
        b = sample_beta_direction(a1=1, b=4, angle=20.0, seed=3)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_different_plane(self):
        a = sample_beta_direction(a1=1, b=4, angle=0.0, seed=0)
        b = sample_beta_direction(a1=1, b=4, angle=0.0, seed=1)
        assert not np.allclose(a, b)

    @pytest.mark.parametrize('angle', [0.0, 30.0])
    def test_dim_lt_2_raises(self, angle):
        # a 1-D coefficient space (a1*b < 2) has only one direction, so any
        # angle request raises
        with pytest.raises(ValueError, match='direction'):
            sample_beta_direction(a1=1, b=1, angle=angle, seed=0)


class TestImposeEffect:
    @pytest.mark.parametrize('effect_llr', [0.0, 0.02, 0.1, 0.5])
    def test_purge_hits_target_llr(self, effect_llr):
        x, y, contrast = _make_region(seed=1)
        beta = sample_beta_direction(a1=1, b=y.shape[0], angle=0.0, seed=2)
        offset = impose_effect(x=x, y=y, contrast=contrast,
                               beta_direction=beta, effect_llr=effect_llr,
                               purge_interest=True)
        e, h, _ = get_mancova(x=x, y=y + offset[..., np.newaxis],
                              contrast=contrast)
        assert np.isclose(get_llr(e, h, n=1), effect_llr, atol=1e-6)

    def test_purge_recovers_direction(self):
        # purge ⇒ the least-squares interest coef points exactly along beta
        x, y, contrast = _make_region(seed=3)
        q1 = decompose(x, contrast)[1]
        beta = sample_beta_direction(a1=1, b=y.shape[0], angle=12.0, seed=4)
        offset = impose_effect(x=x, y=y, contrast=contrast,
                               beta_direction=beta, effect_llr=0.1,
                               purge_interest=True)
        beta_new = (y + offset[..., np.newaxis]).mean(2) @ q1.T
        # atol loose: arccos is ill-conditioned near 0 deg (parallel)
        assert np.isclose(_fro_angle_deg(beta_new, beta), 0.0, atol=1e-3)

    def test_nopurge_change_is_along_beta(self):
        # without purge the recovered coef is baseline + alpha*beta, so the
        # CHANGE (not the coef itself) is along beta
        x, y, contrast = _make_region(seed=5)
        q1 = decompose(x, contrast)[1]
        beta = sample_beta_direction(a1=1, b=y.shape[0], angle=40.0, seed=6)
        beta_orig = y.mean(2) @ q1.T
        offset = impose_effect(x=x, y=y, contrast=contrast,
                               beta_direction=beta, effect_llr=0.1,
                               purge_interest=False)
        beta_new = (y + offset[..., np.newaxis]).mean(2) @ q1.T
        # atol loose: arccos is ill-conditioned near 0 deg (parallel)
        assert np.isclose(_fro_angle_deg(beta_new - beta_orig, beta), 0.0,
                          atol=1e-3)

    @pytest.mark.parametrize('purge', [True, False])
    def test_error_matrix_unchanged(self, purge):
        # offset lives in span(q1), so it touches H but not E or sigma
        x, y, contrast = _make_region(seed=7)
        beta = sample_beta_direction(a1=1, b=y.shape[0], angle=33.0, seed=8)
        offset = impose_effect(x=x, y=y, contrast=contrast,
                               beta_direction=beta, effect_llr=0.2,
                               purge_interest=purge)
        e0, _, sig0 = get_mancova(x=x, y=y, contrast=contrast)
        e1, _, sig1 = get_mancova(x=x, y=y + offset[..., np.newaxis],
                                  contrast=contrast)
        assert np.allclose(e0, e1)
        assert np.allclose(sig0, sig1)

    def test_offset_in_span_q1(self):
        x, y, contrast = _make_region(seed=9)
        q0, q1, q2 = decompose(x, contrast)
        beta = sample_beta_direction(a1=1, b=y.shape[0], angle=55.0, seed=10)
        offset = impose_effect(x=x, y=y, contrast=contrast,
                               beta_direction=beta, effect_llr=0.15)
        assert np.allclose(offset @ q0.T, 0.0, atol=1e-10)
        assert np.allclose(offset @ q2.T, 0.0, atol=1e-10)

    @pytest.mark.parametrize('n_interest', [1, 2, 3])
    def test_general_a1_hits_llr_and_direction(self, n_interest):
        # a1 > 1 is supported: alpha solves the monotone LLR equation and,
        # with purge, the recovered coefficient lands exactly along beta
        x, y, contrast = _make_region(seed=21, n_interest=n_interest, a=5)
        a1 = int(contrast.sum())
        beta = np.random.default_rng(99).standard_normal((a1, y.shape[0]))
        offset = impose_effect(x=x, y=y, contrast=contrast,
                               beta_direction=beta, effect_llr=0.2,
                               purge_interest=True)
        e, h, _ = get_mancova(x=x, y=y + offset[..., np.newaxis],
                              contrast=contrast)
        assert np.isclose(get_llr(e, h, n=1), 0.2, atol=1e-6)
        q1 = decompose(x, contrast)[1]
        beta_new = (y + offset[..., np.newaxis]).mean(2) @ q1.T   # (b, a1)
        assert np.isclose(_fro_angle_deg(beta_new, beta.T), 0.0, atol=1e-3)

    def test_beta_shape_mismatch_raises(self):
        # a (b,) vector is only valid when a1 == 1
        x, y, contrast = _make_region(seed=11, n_interest=2)
        with pytest.raises(AssertionError, match='beta_direction'):
            impose_effect(x=x, y=y, contrast=contrast,
                          beta_direction=np.ones(y.shape[0]), effect_llr=0.1)

    def test_vector_beta_accepted(self):
        x, y, contrast = _make_region(seed=12)
        b = y.shape[0]
        vec = np.arange(1, b + 1, dtype=float)
        offset = impose_effect(x=x, y=y, contrast=contrast,
                               beta_direction=vec, effect_llr=0.1)
        assert offset.shape == (b, y.shape[1])
