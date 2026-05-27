"""Tests for the refactored EffectSynthetic (sklearn-shape)."""

import pickle

import numpy as np
import pytest

from glow.effect import EffectSynthetic, ExtenterMinVar, ExtenterSphere
from glow.experiment.exper import Experiment


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
        # the old impose() swallowed **kwargs silently — the new
        # __init__ does not.
        with pytest.raises(TypeError):
            EffectSynthetic(extenter=extenter, effect_llr=0.5,
                            noise_scale=0.5)

    def test_init_mask_is_frozen(self, exp):
        mask = exp.mask_idx >= 0
        synth = EffectSynthetic(mask=mask, effect_llr=0.5)
        assert synth.mask.flags.writeable is False
        with pytest.raises(ValueError):
            synth.mask[0, 0] = False

    def test_seed_optional(self, extenter):
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5)
        assert synth.seed is None


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

    def test_no_seed_needed(self, exp):
        # seed should be irrelevant when mask is provided
        mask = (exp.mask_idx >= 0) & (exp.mask_idx < 5)
        a = EffectSynthetic(mask=mask, effect_llr=0.5, seed=0)
        b = EffectSynthetic(mask=mask, effect_llr=0.5, seed=42)
        a.fit(exp)
        b.fit(exp)
        np.testing.assert_array_equal(a.offset_, b.offset_)


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
        a = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        b = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=1)
        a.fit(exp)
        b.fit(exp)
        # at least one differs (overwhelmingly likely for radius=2 sphere)
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
    def test_hashable_at_construction(self, extenter):
        # no .fit() needed — hash works pre-fit
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        assert isinstance(hash(synth), int)

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

    def test_dict_key(self, extenter):
        synth = EffectSynthetic(extenter=extenter, effect_llr=0.5, seed=0)
        d = {synth: 'a'}
        same = EffectSynthetic(extenter=ExtenterSphere(radius=2),
                               effect_llr=0.5, seed=0)
        assert d[same] == 'a'

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
