"""Tests for EffectSynthetic (frozen spec, (exp_eff, mask) output).

The effect seed lives in the extenter (it carries its own seed), so a
seeded extenter makes the sampled extent reproducible. EffectSynthetic's
own seed drives only the imposed direction (angle). fit(exp) returns an
(exp_eff, mask) pair and leaves the frozen spec untouched.
"""

import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from glow.effect import EffectSynthetic, ExtenterMinVar, ExtenterSphere
from glow.effect.impose import sample_beta_direction
from glow.experiment.exper import Experiment
from glow.analysis.mancova import decompose, get_mancova, get_llr


@pytest.fixture
def exp():
    """small WGN experiment with X sampled (so x/contrast available)."""
    return Experiment.from_gauss(seed=0, shape=(5, 5), num_img=20, b=2)


@pytest.fixture
def extenter():
    # seed lives in the extenter, so its sampled extent is reproducible
    return ExtenterSphere(radius=2, seed=0)


class TestInit:
    def test_extenter_only_ok(self, extenter):
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5)
        assert synth.extenter is extenter
        assert synth.mask is None

    def test_mask_only_ok(self, exp):
        mask = exp.mask_idx >= 0
        synth = EffectSynthetic(mask=mask, effect_llr=0.5)
        assert synth.extenter is None
        assert synth.mask is not None
        assert synth.mask.dtype == bool

    def test_neither_raises(self):
        with pytest.raises(ValueError, match='xor'):
            EffectSynthetic(effect_llr=0.5)

    def test_both_raises(self, extenter, exp):
        mask = exp.mask_idx >= 0
        with pytest.raises(ValueError, match='xor'):
            EffectSynthetic(extenter=extenter, mask=mask, effect_llr=0.5)

    def test_unknown_kwarg_raises(self, extenter):
        # regression guard: unknown kwargs must raise, not be swallowed
        with pytest.raises(TypeError):
            EffectSynthetic(extenter=extenter, effect_llr=0.5,
                            noise_scale=0.5)

    def test_init_mask_is_frozen(self, exp):
        mask = exp.mask_idx >= 0
        synth = EffectSynthetic(mask=mask, effect_llr=0.5)
        assert synth.mask.flags.writeable is False
        with pytest.raises(ValueError):
            synth.mask[0, 0] = False

    def test_spec_is_frozen(self, extenter):
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5)
        with pytest.raises(FrozenInstanceError):
            synth.effect_llr = 0.9


class TestFitExtenterPath:
    def test_returns_exp_and_mask(self, exp, extenter):
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5)
        exp_eff, mask = synth.fit(exp)
        assert mask is not None
        assert exp_eff.y.shape == exp.y.shape

    def test_mask_matches_extenter(self, exp, extenter):
        # the extenter (seed baked in) produces the same mask directly
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5)
        _, mask = synth.fit(exp)
        direct = extenter(y=exp.y, mask_idx=exp.mask_idx)
        np.testing.assert_array_equal(mask, direct)

    def test_effect_llr_preserved(self, exp):
        synth = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=0),
                                effect_llr=0.7)
        synth.fit(exp)
        assert synth.effect_llr == 0.7


class TestFitMaskPath:
    def test_mask_is_used_verbatim(self, exp):
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[0:3, 0:3] = True
        mask &= (exp.mask_idx >= 0)
        synth = EffectSynthetic(mask=mask, effect_llr=0.5)
        _, fit_mask = synth.fit(exp)
        np.testing.assert_array_equal(fit_mask, mask)


class TestReproducibility:
    def test_same_spec_same_fit(self, exp):
        a = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=0),
                            effect_llr=0.5)
        b = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=0),
                            effect_llr=0.5)
        (exp_a, mask_a), (exp_b, mask_b) = a.fit(exp), b.fit(exp)
        np.testing.assert_array_equal(mask_a, mask_b)
        np.testing.assert_array_equal(exp_a.y, exp_b.y)

    def test_different_seed_different_mask(self, exp):
        # seeds 0 and 1 place distinct sphere centres for this fixture
        # (8 vs 12 voxels), avoiding small-grid saturation flakiness.
        a = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=0),
                            effect_llr=0.5)
        b = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=1),
                            effect_llr=0.5)
        assert not np.array_equal(a.fit(exp)[1], b.fit(exp)[1])

    def test_effect_llr_no_leak_into_mask(self, exp):
        # same extenter (same seed) -> same mask regardless of llr
        a = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=0),
                            effect_llr=0.3)
        b = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=0),
                            effect_llr=0.9)
        (exp_a, mask_a), (exp_b, mask_b) = a.fit(exp), b.fit(exp)
        np.testing.assert_array_equal(mask_a, mask_b)
        # different llr -> different imposed offset -> different applied y
        assert not np.allclose(exp_a.y, exp_b.y)


class TestIdentity:
    """EffectSynthetic carries an ndarray and is never a cache key, so it
    uses identity equality (eq=False) rather than a value hash."""

    def test_identity_equality(self, extenter):
        a = EffectSynthetic(extenter=extenter, effect_llr=0.5)
        b = EffectSynthetic(extenter=extenter, effect_llr=0.5)
        assert a != b           # distinct objects, identity equality
        assert a == a
        assert hash(a) == hash(a)   # hashable by identity

    def test_neq_other_type(self, extenter):
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5)
        assert synth != {'extenter': extenter}
        assert synth != 42


class TestToRecord:
    """to_record gives a JSON-friendly recipe without the realized mask."""

    def test_extenter_recipe_nested_no_mask(self, extenter):
        rec = EffectSynthetic(extenter=extenter, effect_llr=0.5,
                              seed=3).to_record()
        assert rec['kind'] == 'EffectSynthetic'
        assert rec['effect_llr'] == 0.5
        assert rec['seed'] == 3
        assert rec['extenter'] == extenter.to_record()
        assert 'mask' not in rec

    def test_mask_path_records_no_mask(self, exp):
        # the explicit-mask path is not in the pipeline; to_record omits the
        # array (extenter is None) rather than dumping it
        mask = exp.mask_idx >= 0
        rec = EffectSynthetic(mask=mask, effect_llr=0.5).to_record()
        assert rec['extenter'] is None
        assert 'mask' not in rec


class TestFrozenInputsInterop:
    def test_fit_does_not_mutate_exp(self, exp, extenter):
        # freeze the exp arrays — fit() must not write to them
        for arr in (exp.y, exp.x, exp.contrast, exp.mask_idx):
            arr.flags.writeable = False
        y_snap = exp.y.copy()
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5)
        synth.fit(exp)
        # exp arrays still frozen, unchanged
        assert exp.y.flags.writeable is False
        np.testing.assert_array_equal(exp.y, y_snap)


class TestPickle:
    def test_round_trip_reproduces_fit(self, exp):
        synth = EffectSynthetic(extenter=ExtenterSphere(radius=2, seed=0),
                                effect_llr=0.5)
        exp_eff, _ = synth.fit(exp)
        clone = pickle.loads(pickle.dumps(synth))
        # a pickled clone re-fits to the same applied experiment
        np.testing.assert_array_equal(clone.fit(exp)[0].y, exp_eff.y)


def _region_coef(applied_exp, base_exp, mask):
    """Least-squares interest coefficient (b, a1) for a region after fit."""
    idx = base_exp.mask_idx[mask]
    q1 = decompose(base_exp.x, base_exp.contrast)[1]
    return applied_exp.y[:, :, idx].mean(2) @ q1.T


def _region_llr(applied_exp, base_exp, mask):
    """Size-normalized (n=1) LLR of a region after fit."""
    idx = base_exp.mask_idx[mask]
    e, h, _ = get_mancova(x=base_exp.x, y=applied_exp.y[:, :, idx],
                          contrast=base_exp.contrast)
    return get_llr(e, h, n=1)


def _fro_angle_deg(a, b):
    """Angle (degrees) between two arrays under the Frobenius inner product."""
    a, b = a.ravel(), b.ravel()
    c = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


class TestDirectionInit:
    def test_angle_without_seed_raises(self, exp):
        mask = exp.mask_idx >= 0
        with pytest.raises(ValueError, match='seed'):
            EffectSynthetic(mask=mask, effect_llr=0.5, angle=30)

    def test_default_purge_true(self, exp):
        mask = exp.mask_idx >= 0
        synth = EffectSynthetic(mask=mask, effect_llr=0.5)
        assert synth.purge_interest is True


class TestDirectionFit:
    def _mask(self, exp, rows):
        m = np.zeros(exp.mask_idx.shape, dtype=bool)
        m[rows, :] = True
        return m & (exp.mask_idx >= 0)

    def test_angle_path(self, exp):
        mask = self._mask(exp, slice(0, 3))
        synth = EffectSynthetic(mask=mask, effect_llr=0.1, angle=25, seed=4)
        exp_eff, _ = synth.fit(exp)
        assert np.isclose(_region_llr(exp_eff, exp, mask), 0.1, atol=1e-6)
        # recovered coef points along the seed-sampled direction (recompute,
        # not stored): same a1/b/angle/seed -> same beta_direction
        expected = sample_beta_direction(a1=1, b=exp.y.shape[0], angle=25,
                                         seed=4)
        coef = _region_coef(exp_eff, exp, mask)
        assert np.isclose(_fro_angle_deg(coef, expected.T), 0.0, atol=1e-3)

    def test_two_adjacent_effects_at_angle(self, exp):
        # two disjoint supports sharing a seed, angles 0 and 60: recovered
        # effect directions sit 60 deg apart, each at the target LLR
        mask_a = self._mask(exp, slice(0, 2))
        mask_b = self._mask(exp, slice(3, 5))
        e1 = EffectSynthetic(mask=mask_a, effect_llr=0.1, angle=0, seed=5)
        e2 = EffectSynthetic(mask=mask_b, effect_llr=0.1, angle=60, seed=5)
        out, _ = e2.fit(e1.fit(exp)[0])
        assert np.isclose(_region_llr(out, exp, mask_a), 0.1, atol=1e-6)
        assert np.isclose(_region_llr(out, exp, mask_b), 0.1, atol=1e-6)
        coef_a = _region_coef(out, exp, mask_a)
        coef_b = _region_coef(out, exp, mask_b)
        assert np.isclose(_fro_angle_deg(coef_a, coef_b), 60.0, atol=1e-6)
