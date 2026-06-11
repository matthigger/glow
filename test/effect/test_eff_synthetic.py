"""Tests for the refactored EffectSynthetic (sklearn-shape)."""

import pickle

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
    return ExtenterSphere(radius=2)


class TestInit:
    def test_extenter_only_ok(self, extenter):
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        assert synth.extenter is extenter
        assert synth.mask is None
        assert synth.mask_ is synth.offset_ is synth.sigma_scale_ is None

    def test_mask_only_ok(self, exp):
        mask = exp.mask_idx >= 0
        synth = EffectSynthetic(mask=mask, effect_llr=0.5, seed=0)
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


class TestFitExtenterPath:
    def test_outputs_populated(self, exp, extenter):
        # sigma_scale_ is documented as always None — only mask_ and
        # offset_ must be populated.
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        out = synth.fit(exp)
        assert synth.mask_ is not None
        assert synth.offset_ is not None
        assert out.y.shape == exp.y.shape

    def test_mask_size_matches_extenter(self, exp, extenter):
        # extenter would produce the same mask given the same exp+seed
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        synth.fit(exp)
        direct = extenter(y=exp.y, mask_idx=exp.mask_idx, seed=0)
        np.testing.assert_array_equal(synth.mask_, direct)

    def test_effect_llr_preserved(self, exp, extenter):
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.7, seed=0)
        synth.fit(exp)
        assert synth.effect_llr == 0.7


class TestFitMaskPath:
    def test_mask_is_used_verbatim(self, exp):
        mask = np.zeros(exp.mask_idx.shape, dtype=bool)
        mask[0:3, 0:3] = True
        mask &= (exp.mask_idx >= 0)
        synth = EffectSynthetic(mask=mask, effect_llr=0.5)
        synth.fit(exp)
        np.testing.assert_array_equal(synth.mask_, mask)
        # seed is irrelevant on the mask path: same offset for any seed
        other = EffectSynthetic(mask=mask, effect_llr=0.5, seed=42)
        other.fit(exp)
        np.testing.assert_array_equal(synth.offset_, other.offset_)


class TestReproducibility:
    def test_same_recipe_same_fit(self, exp, extenter):
        a = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        b = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        out_a = a.fit(exp)
        out_b = b.fit(exp)
        np.testing.assert_array_equal(a.mask_, b.mask_)
        np.testing.assert_array_equal(a.offset_, b.offset_)
        np.testing.assert_array_equal(out_a.y, out_b.y)

    def test_different_seed_different_mask(self, exp, extenter):
        # On this 5x5 grid a radius-2 sphere can saturate the whole grid,
        # so different seeds do NOT always differ. Seeds 0 and 1 are pinned
        # here because they are verified to place distinct sphere centres
        # (8 vs 12 voxels) for this fixture, avoiding small-grid flakiness.
        a = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        b = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=1)
        a.fit(exp)
        b.fit(exp)
        assert not np.array_equal(a.mask_, b.mask_)

    def test_effect_llr_no_leak_into_mask(self, exp, extenter):
        # same seed + extenter must produce same mask regardless of llr
        a = EffectSynthetic(extenter=extenter, effect_llr=0.3, seed=0)
        b = EffectSynthetic(extenter=extenter, effect_llr=0.9, seed=0)
        a.fit(exp)
        b.fit(exp)
        np.testing.assert_array_equal(a.mask_, b.mask_)
        # offset should differ (different llr targets)
        assert not np.allclose(a.offset_, b.offset_)


class TestApplyReplay:
    def test_apply_idempotent(self, exp, extenter):
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        first = synth.fit(exp)
        replay = synth.apply(exp)
        np.testing.assert_array_equal(first.y, replay.y)


class TestHash:
    def test_equal_params_equal_hash_extenter(self, extenter):
        a = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        b = EffectSynthetic(extenter=ExtenterSphere(radius=2),
                            effect_llr=0.5, seed=0)
        assert a == b
        assert hash(a) == hash(b)

    def test_equal_params_equal_hash_mask(self):
        m1 = np.array([[True, False], [False, True]])
        m2 = m1.copy()
        a = EffectSynthetic(mask=m1, effect_llr=0.5, seed=0)
        b = EffectSynthetic(mask=m2, effect_llr=0.5, seed=0)
        assert a == b
        assert hash(a) == hash(b)

    @pytest.mark.parametrize('a, b', [
        # different extenter
        (EffectSynthetic(extenter=ExtenterSphere(radius=2),
                         effect_llr=0.5, seed=0),
         EffectSynthetic(extenter=ExtenterSphere(radius=3),
                         effect_llr=0.5, seed=0)),
        # different effect_llr
        (EffectSynthetic(extenter=ExtenterSphere(radius=2),
                         effect_llr=0.5, seed=0),
         EffectSynthetic(extenter=ExtenterSphere(radius=2),
                         effect_llr=0.7, seed=0)),
        # different seed
        (EffectSynthetic(extenter=ExtenterSphere(radius=2),
                         effect_llr=0.5, seed=0),
         EffectSynthetic(extenter=ExtenterSphere(radius=2),
                         effect_llr=0.5, seed=1)),
    ])
    def test_each_init_param_changes_hash(self, a, b):
        assert a != b
        assert hash(a) != hash(b)

    def test_mask_vs_extenter_distinct(self):
        # even when masks happen to match
        mask = np.ones((4, 4), dtype=bool)
        a = EffectSynthetic(mask=mask, effect_llr=0.5, seed=0)
        b = EffectSynthetic(extenter=ExtenterSphere(radius=2),
                            effect_llr=0.5, seed=0)
        assert a != b
        assert hash(a) != hash(b)

    def test_neq_other_type(self, extenter):
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        assert synth != {'extenter': extenter}
        assert synth != 42


class TestFrozenInputsInterop:
    def test_fit_does_not_mutate_exp(self, exp, extenter):
        # freeze the exp arrays — fit() must not write to them
        for arr in (exp.y, exp.x, exp.contrast, exp.mask_idx):
            arr.flags.writeable = False
        y_snap = exp.y.copy()
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        synth.fit(exp)
        # exp arrays still frozen, unchanged
        assert exp.y.flags.writeable is False
        np.testing.assert_array_equal(exp.y, y_snap)


class TestPickle:
    def test_round_trip_pre_fit(self, extenter):
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        clone = pickle.loads(pickle.dumps(synth))
        assert clone == synth
        assert hash(clone) == hash(synth)

    def test_round_trip_post_fit(self, exp, extenter):
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        out = synth.fit(exp)
        clone = pickle.loads(pickle.dumps(synth))
        assert clone == synth
        assert hash(clone) == hash(synth)
        # clone reproduces the same applied exp on the original input
        replay = clone.apply(exp)
        np.testing.assert_array_equal(replay.y, out.y)


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
        out = synth.fit(exp)
        assert synth.offset_ is not None
        assert synth.sigma_scale_ is None
        assert np.isclose(_region_llr(out, exp, mask), 0.1, atol=1e-6)
        # recovered coef points along the seed-sampled direction (recompute,
        # not stored): same a1/b/angle/seed -> same beta_direction
        expected = sample_beta_direction(a1=1, b=exp.y.shape[0], angle=25,
                                         seed=4)
        coef = _region_coef(out, exp, mask)
        assert np.isclose(_fro_angle_deg(coef, expected.T), 0.0, atol=1e-3)

    def test_two_adjacent_effects_at_angle(self, exp):
        # two disjoint supports sharing a seed, angles 0 and 60: recovered
        # effect directions sit 60 deg apart, each at the target LLR
        mask_a = self._mask(exp, slice(0, 2))
        mask_b = self._mask(exp, slice(3, 5))
        e1 = EffectSynthetic(mask=mask_a, effect_llr=0.1, angle=0, seed=5)
        e2 = EffectSynthetic(mask=mask_b, effect_llr=0.1, angle=60, seed=5)
        out = e2.fit(e1.fit(exp))
        assert np.isclose(_region_llr(out, exp, mask_a), 0.1, atol=1e-6)
        assert np.isclose(_region_llr(out, exp, mask_b), 0.1, atol=1e-6)
        coef_a = _region_coef(out, exp, mask_a)
        coef_b = _region_coef(out, exp, mask_b)
        assert np.isclose(_fro_angle_deg(coef_a, coef_b), 60.0, atol=1e-6)


class TestDirectionHash:
    def _base(self, **kw):
        mask = np.ones((4, 4), dtype=bool)
        return EffectSynthetic(mask=mask, effect_llr=0.5, **kw)

    def test_angle_changes_hash(self):
        a = self._base(angle=0, seed=0)
        b = self._base(angle=60, seed=0)
        assert a != b
        assert hash(a) != hash(b)

    def test_purge_interest_changes_hash(self):
        a = self._base(angle=30, seed=0, purge_interest=True)
        b = self._base(angle=30, seed=0, purge_interest=False)
        assert a != b
        assert hash(a) != hash(b)
