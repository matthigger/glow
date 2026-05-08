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
        assert exp2.x is None
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
        assert exps2.x is None

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

    def test_apply_mask_clears_recipe(self):
        exp = Experiment.from_gauss(seed=0, shape=(5, 5), a=2, b=1, num_img=10)
        assert 'recipe' in exp.meta
        mask = np.zeros((5, 5), dtype=bool)
        mask[:3, :3] = True
        exp_masked = exp.apply_mask(mask)
        assert 'recipe' not in exp_masked.meta

    def test_bootstrap_clears_recipe(self):
        exp = Experiment.from_gauss(seed=0, shape=(4, 4), a=2, b=1, num_img=20)
        assert 'recipe' in exp.meta
        exp_b = exp.bootstrap_img(n=5, seed=0)
        assert 'recipe' not in exp_b.meta


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

