import pickle
import warnings

import nibabel as nib
import numpy as np
import pytest

from glow.experiment.exper import (Experiment, ExperimentImageOnly,
                                    ExperimentScaled)
from glow.experiment.regen import (FullPickleNotice, REGEN_REGISTRY,
                                    PickleStatus, compute_pickle_status)


def _build_test_tree(tmp_path, shape=(4, 4)):
    """create two-subject, two-feature nifti tree under tmp_path."""
    affine = np.eye(4)
    paths = {}
    intensities = {}
    for sbj_idx, sbj in enumerate(['s1', 's2']):
        paths[sbj] = {}
        intensities[sbj] = {}
        for feat_idx, feat in enumerate(['f0', 'f1']):
            intensity = sbj_idx * 10 + feat_idx + 1
            arr = np.full(shape, intensity, dtype=float)
            file = tmp_path / f'{sbj}_{feat}.nii.gz'
            nib.Nifti1Image(arr, affine).to_filename(file)
            paths[sbj][feat] = str(file)
            intensities[sbj][feat] = intensity
    return paths, intensities


class TestSlimPickleGauss:
    def test_round_trip(self):
        exp = Experiment.from_gauss(seed=0, shape=(5, 5), a=2, b=1, num_img=20)
        y_orig = exp.y.copy()
        x_orig = exp.x.copy()

        data = pickle.dumps(exp)
        exp2 = pickle.loads(data)

        assert exp2.y is None
        assert exp2.x is not None
        assert np.allclose(exp2.x, x_orig)
        assert exp2.meta['recipe']['source'] == 'gauss'

        exp2.rehydrate()
        assert np.allclose(exp2.y, y_orig)
        assert np.allclose(exp2.x, x_orig)

    def test_round_trip_image_only(self):
        exp = ExperimentImageOnly.from_gauss(seed=42, shape=(4, 4), b=1,
                                              num_img=10)
        y_orig = exp.y.copy()

        data = pickle.dumps(exp)
        exp2 = pickle.loads(data)
        assert exp2.y is None

        exp2.rehydrate()
        assert np.allclose(exp2.y, y_orig)

    def test_round_trip_scaled(self):
        exp = Experiment.from_gauss(seed=7, shape=(5, 5), a=2, b=2, num_img=30)
        exps = ExperimentScaled.from_exp(exp)
        y_prep = exps.y.copy()
        x_orig = exps.x.copy()

        data = pickle.dumps(exps)
        exps2 = pickle.loads(data)
        assert exps2.y is None
        assert exps2.x is not None
        assert np.allclose(exps2.x, x_orig)

        exps2.rehydrate()
        assert np.allclose(exps2.y, y_prep)
        assert np.allclose(exps2.x, x_orig)


class TestSlimPickleImagePaths:
    def test_round_trip(self, tmp_path):
        paths, _ = _build_test_tree(tmp_path)

        exp = ExperimentImageOnly.from_paths(paths)
        y_orig = exp.y.copy()
        assert exp.meta['recipe']['source'] == 'image_paths'

        data = pickle.dumps(exp)
        exp2 = pickle.loads(data)
        assert exp2.y is None

        exp2.rehydrate()
        assert np.array_equal(exp2.y, y_orig)

    def test_rehydrate_missing_file_fails(self, tmp_path):
        paths, _ = _build_test_tree(tmp_path)
        exp = ExperimentImageOnly.from_paths(paths)
        data = pickle.dumps(exp)

        # delete one of the files
        first_sbj = next(iter(paths))
        first_feat = next(iter(paths[first_sbj]))
        import os
        os.remove(paths[first_sbj][first_feat])

        exp2 = pickle.loads(data)
        with pytest.raises(Exception):
            exp2.rehydrate()


class TestFromSearchPathsEquivalence:
    def test_equal_y(self, tmp_path):
        paths, _ = _build_test_tree(tmp_path)

        # rebuild paths as a DataFrame; from_search produces the same df
        exp_paths = ExperimentImageOnly.from_paths(paths)
        exp_search = ExperimentImageOnly.from_search(
            folder=tmp_path,
            sbj_regex=r's\d',
            img_glob_dict={'f0': '*_f0.nii.gz', 'f1': '*_f1.nii.gz'})

        assert np.array_equal(exp_paths.y, exp_search.y)


class TestFullPickleFallback:
    def test_warn_and_fallback_no_recipe(self):
        exp = Experiment.from_gauss(seed=0, shape=(4, 4), a=2, b=1, num_img=10)
        # remove the recipe to force the fallback
        exp.meta.pop('recipe', None)

        # reset 'once'-dedup so the warning fires for this category here
        warnings.simplefilter('always', FullPickleNotice)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            data = pickle.dumps(exp)
        full_pickle_warns = [w for w in caught
                             if issubclass(w.category, FullPickleNotice)]
        assert len(full_pickle_warns) == 1
        assert 'without recipe' in str(full_pickle_warns[0].message)

        exp2 = pickle.loads(data)
        # full pickle: y survives
        assert exp2.y is not None
        assert np.allclose(exp2.y, exp.y)

    def test_warn_unregistered_source(self):
        exp = Experiment.from_gauss(seed=0, shape=(4, 4), a=2, b=1, num_img=10)
        exp.meta['recipe'] = {'source': 'nonexistent', 'args': {}}

        warnings.simplefilter('always', FullPickleNotice)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            data = pickle.dumps(exp)
        full_pickle_warns = [w for w in caught
                             if issubclass(w.category, FullPickleNotice)]
        assert len(full_pickle_warns) == 1
        assert 'not registered' in str(full_pickle_warns[0].message)

        exp2 = pickle.loads(data)
        assert exp2.y is not None

    def test_apply_mask_no_step_clears_recipe(self):
        # Legacy behavior: when no recipe_step is supplied, the recipe
        # is cleared so __getstate__ falls back to a full pickle.
        exp = Experiment.from_gauss(seed=0, shape=(5, 5), a=2, b=1, num_img=10)
        assert 'recipe' in exp.meta
        mask = np.zeros((5, 5), dtype=bool)
        mask[:3, :3] = True
        exp_masked = exp.apply_mask(mask)
        assert 'recipe' not in exp_masked.meta

    def test_bootstrap_self_recipes(self):
        # New behavior: bootstrap_img is self-describing and extends
        # the recipe with a 'bootstrap_img' step.
        exp = Experiment.from_gauss(seed=0, shape=(4, 4), a=2, b=1, num_img=20)
        assert 'recipe' in exp.meta
        exp_b = exp.bootstrap_img(n=5, seed=0)
        assert 'recipe' in exp_b.meta
        steps = exp_b.meta['recipe']['steps']
        assert len(steps) == 1
        assert steps[0]['op'] == 'bootstrap_img'
        assert steps[0]['args']['n'] == 5


class TestPickleStatus:
    def test_slim_eligible(self):
        exp = Experiment.from_gauss(seed=0, shape=(5, 5), a=2, b=1, num_img=20)
        st = exp.pickle_status()
        assert isinstance(st, PickleStatus)
        assert st.is_loaded is True
        assert st.recipe_source == 'gauss'
        assert st.regen_registered is True
        assert st.will_slim_on_pickle is True
        assert st.y_mb > 0
        assert st.recipe_kb > 0

    def test_full_pickling(self):
        exp = Experiment.from_gauss(seed=0, shape=(5, 5), a=2, b=1, num_img=20)
        exp.meta.pop('recipe', None)
        st = exp.pickle_status()
        assert st.recipe_source is None
        assert st.regen_registered is False
        assert st.will_slim_on_pickle is False

    def test_str(self):
        exp = Experiment.from_gauss(seed=0, shape=(4, 4), a=2, b=1, num_img=10)
        s = str(exp.pickle_status())
        assert 'PickleStatus' in s
        assert 'gauss' in s


class TestAnalysisGLOWProvenance:
    def test_attrs_survive_pickle(self):
        from glow.analysis import AnalysisGLOW
        exp = Experiment.from_gauss(seed=0, shape=(5, 5), a=2, b=1, num_img=20)
        ana = AnalysisGLOW(exp, n_perm_fwer=5, n_perm_inner=5, min_vox=2).fit()

        assert ana.n_perm_fwer == 5
        assert ana.n_perm_inner == 5
        assert ana.max_z_null.shape == (6,)

        data = pickle.dumps(ana)
        ana2 = pickle.loads(data)

        assert ana2.n_perm_fwer == 5
        assert ana2.n_perm_inner == 5
        assert np.array_equal(ana2.max_z_null, ana.max_z_null)
        # exp slimmed
        assert ana2.exp.y is None
        ana2.exp.rehydrate()
        assert ana2.exp.y is not None


class TestSlimAnalysisGLOWSize:
    def test_slim_smaller_than_full(self):
        from glow.analysis import AnalysisGLOW
        exp = Experiment.from_gauss(seed=0, shape=(8, 8), a=2, b=1, num_img=30)
        ana = AnalysisGLOW(exp, n_perm_fwer=5, n_perm_inner=5, min_vox=2).fit()

        slim_bytes = len(pickle.dumps(ana))

        # forge a full pickle by removing the recipe
        ana.exp.meta.pop('recipe', None)
        warnings.simplefilter('always', FullPickleNotice)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FullPickleNotice)
            full_bytes = len(pickle.dumps(ana))

        # the y array dominates: a (1, 30, 64) float array is ~15 KB,
        # plus the analysis arrays.  Slim should be meaningfully smaller.
        assert slim_bytes < full_bytes

class TestRecipeStepReplay:
    """Recipe-step composition: each mutating op should be replayable
    on slim rehydrate."""

    def test_permute_self_recipe_round_trip(self):
        exp = Experiment.from_gauss(seed=0, shape=(5, 5), a=2, b=1, num_img=20)
        exp_p = exp.permute(perm_idx=3)
        y_orig = exp_p.y.copy()
        x_orig = exp_p.x.copy()

        data = pickle.dumps(exp_p)
        exp_p2 = pickle.loads(data)
        assert exp_p2.y is None
        assert exp_p2.meta['recipe']['steps'][-1]['op'] == 'permute'

        exp_p2.rehydrate()
        assert np.allclose(exp_p2.y, y_orig)
        assert np.allclose(exp_p2.x, x_orig)

    def test_bootstrap_self_recipe_round_trip(self):
        exp = ExperimentImageOnly.from_gauss(seed=0, shape=(4, 4), b=1,
                                              num_img=15)
        exp_b = exp.bootstrap_img(n=5, seed=42)
        y_orig = exp_b.y.copy()

        data = pickle.dumps(exp_b)
        exp_b2 = pickle.loads(data)
        assert exp_b2.y is None
        assert exp_b2.meta['recipe']['steps'][-1]['op'] == 'bootstrap_img'

        exp_b2.rehydrate()
        assert np.allclose(exp_b2.y, y_orig)

    def test_apply_mask_with_step_round_trip(self):
        exp = Experiment.from_gauss(seed=0, shape=(5, 5), a=2, b=1, num_img=20)
        mask = np.zeros((5, 5), dtype=bool)
        mask[:3, :3] = True
        recipe_step = {'op': 'apply_mask', 'args': {'mask': mask}}
        exp_m = exp.apply_mask(mask, recipe_step=recipe_step)
        y_orig = exp_m.y.copy()
        x_orig = exp_m.x.copy()

        data = pickle.dumps(exp_m)
        exp_m2 = pickle.loads(data)
        assert exp_m2.y is None
        assert exp_m2.meta['recipe']['steps'][-1]['op'] == 'apply_mask'

        exp_m2.rehydrate()
        assert np.allclose(exp_m2.y, y_orig)
        assert np.allclose(exp_m2.x, x_orig)

    def test_effect_synthetic_apply_round_trip(self):
        from glow.effect import EffectSynthetic, ExtenterSphere
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                     num_img=30, seed=0)
        extenter = ExtenterSphere(radius=2)
        exp_eff, _eff = EffectSynthetic.impose(
            exp, effect_llr=0.3, extenter=extenter, seed=0)
        y_orig = exp_eff.y.copy()
        x_orig = exp_eff.x.copy()

        data = pickle.dumps(exp_eff)
        exp_eff2 = pickle.loads(data)
        assert exp_eff2.y is None
        steps = exp_eff2.meta['recipe']['steps']
        assert steps[-1]['op'] == 'add_offset'

        exp_eff2.rehydrate()
        assert np.allclose(exp_eff2.y, y_orig)
        assert np.allclose(exp_eff2.x, x_orig)

    def test_effect_synthetic_returns_tuple(self):
        # The new API mirrors the old (Experiment, Effect-like) tuple shape.
        from glow.effect import EffectSynthetic, ExtenterSphere
        exp = Experiment.from_gauss(a=2, b=1, shape=(5, 5),
                                     num_img=20, seed=0)
        result = EffectSynthetic.impose(
            exp, effect_llr=0.2,
            extenter=ExtenterSphere(radius=2), seed=0)
        assert isinstance(result, tuple) and len(result) == 2
        new_exp, synth = result
        assert isinstance(new_exp, Experiment)
        assert isinstance(synth, EffectSynthetic)
        assert synth.mask.sum() > 0

    def test_paper_pipeline_round_trip(self):
        # Full paper_config-style chain: gauss -> apply_mask -> scale ->
        # impose effect. After slim pickle + rehydrate, the final y should
        # match the original.
        from glow.effect import EffectSynthetic, ExtenterSphere
        from glow.experiment.exper import ExperimentScaled

        exp = Experiment.from_gauss(a=2, b=1, shape=(6, 6),
                                     num_img=30, seed=0)
        # crop
        crop_mask = np.zeros((6, 6), dtype=bool)
        crop_mask[1:5, 1:5] = True
        exp_crop = exp.apply_mask(
            crop_mask,
            recipe_step={'op': 'apply_mask', 'args': {'mask': crop_mask}},
        )
        # scale
        exp_scaled = ExperimentScaled.from_exp(exp_crop)
        # impose
        n_eff = max(1, int(0.3 * exp_scaled.y.shape[2]))
        extenter = ExtenterSphere(n_vox=n_eff)
        exp_eff, _ = EffectSynthetic.impose(
            exp_scaled, effect_llr=0.2, extenter=extenter, seed=1)

        y_orig = exp_eff.y.copy()
        x_orig = exp_eff.x.copy()

        data = pickle.dumps(exp_eff)
        exp_eff2 = pickle.loads(data)
        assert exp_eff2.y is None
        ops = [s['op'] for s in exp_eff2.meta['recipe']['steps']]
        assert ops == ['apply_mask', 'scale', 'add_offset']

        exp_eff2.rehydrate()
        assert np.allclose(exp_eff2.y, y_orig)
        assert np.allclose(exp_eff2.x, x_orig)

    def test_cleared_recipe_full_pickle_fallback(self):
        # If the recipe is somehow cleared, the full-pickle warning fires
        # and round-trip still produces correct y.
        exp = Experiment.from_gauss(seed=0, shape=(4, 4), a=2, b=1, num_img=10)
        exp.meta.pop('recipe', None)

        warnings.simplefilter('always', FullPickleNotice)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FullPickleNotice)
            data = pickle.dumps(exp)
        exp2 = pickle.loads(data)
        assert exp2.y is not None
        assert np.allclose(exp2.y, exp.y)


class TestForceFullPickle:
    """Cross-machine transfers must inline y/x even when a recipe is
    present, otherwise the receiver can't rehydrate from the sender's
    local paths.  force_full_pickle() is the escape hatch."""

    def test_inlines_y_during_context(self):
        from glow.experiment.regen import force_full_pickle
        exp = Experiment.from_gauss(seed=0, shape=(4, 4), a=2, b=1, num_img=10)
        # default behaviour: slim
        assert pickle.loads(pickle.dumps(exp)).y is None
        # forced full: y survives
        with force_full_pickle(exp):
            data = pickle.dumps(exp)
        assert pickle.loads(data).y is not None

    def test_recipe_restored_after_context(self):
        from glow.experiment.regen import force_full_pickle
        exp = Experiment.from_gauss(seed=0, shape=(4, 4), a=2, b=1, num_img=10)
        with force_full_pickle(exp):
            pickle.dumps(exp)
        # recipe still present after exiting the context
        assert exp.meta.get('recipe') is not None
        # default slim behaviour still works after the context
        assert pickle.loads(pickle.dumps(exp)).y is None

    def test_recipe_restored_on_exception(self):
        from glow.experiment.regen import force_full_pickle
        exp = Experiment.from_gauss(seed=0, shape=(4, 4), a=2, b=1, num_img=10)
        try:
            with force_full_pickle(exp):
                raise RuntimeError('boom')
        except RuntimeError:
            pass
        assert exp.meta.get('recipe') is not None

    def test_silences_warning(self):
        from glow.experiment.regen import force_full_pickle
        exp = Experiment.from_gauss(seed=0, shape=(4, 4), a=2, b=1, num_img=10)
        warnings.simplefilter('always', FullPickleNotice)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with force_full_pickle(exp):
                pickle.dumps(exp)
        full = [w for w in caught if issubclass(w.category, FullPickleNotice)]
        assert full == []

