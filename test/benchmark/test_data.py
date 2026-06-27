"""Tests for glow._extra.benchmark.data: the data_factory builders.

Exercises the WGN builder end to end and the HCP builder with its two heavy
dependencies (the dataset download and the NIfTI search) mocked, then checks
the two decorators wrapping both builders: the joblib.Memory disk cache and
the Recorder. No test touches the network.
"""
import random

import numpy as np
import pytest

from glow._extra.benchmark import data, hcp
from glow.effect import ExtenterMinVar, ExtenterSphere
from glow.experiment import ExperimentImageOnly
from glow.experiment.exper import NoBiasTermWarning


def _fresh_seed() -> int:
    """A seed unlikely to already be in the on-disk cache, so a call misses."""
    return random.randrange(2 ** 31)


@pytest.fixture(autouse=True)
def _records_to_tmp(monkeypatch, tmp_path):
    """Mirror the module recorder's per-hash files to a tmp dir, not the real one."""
    monkeypatch.setattr(data.RECORDER, 'folder', tmp_path)


# ---------------------------------------------------------------------------
# WGN builder
# ---------------------------------------------------------------------------

class TestDataFactoryWGN:
    def test_shapes_and_bias(self):
        exp = data.data_factory_wgn(shape=(6, 6, 6), b=3, num_img=40, a=2,
                                    has_bias=True, seed=_fresh_seed())
        # b channels, num_img images, 6*6*6 voxels; x is the a rows + 1 bias
        assert exp.y.shape == (3, 40, 216)
        assert exp.x.shape == (3, 40)
        # bias row prepended -> a leading False then the a interest columns
        assert exp.contrast.tolist() == [False, True, True]

    def test_contrast_arg_sets_a(self):
        # has_bias=False -> no all-ones row, which warns (regression at origin)
        with pytest.warns(NoBiasTermWarning):
            exp = data.data_factory_wgn(contrast=np.array([False, True]),
                                        has_bias=False, b=2, num_img=20,
                                        seed=_fresh_seed())
        assert exp.x.shape == (2, 20)
        assert exp.contrast.tolist() == [False, True]

    def test_extenter_crops_to_support(self):
        ext = ExtenterSphere(n_vox=20, connected=True, seed=0, contiguous=True)
        exp = data.data_factory_wgn(shape=(8, 8, 8), b=2, num_img=30,
                                    extenter=ext, seed=_fresh_seed())
        assert (exp.mask_idx > -1).sum() == 20
        assert exp.y.shape == (2, 30, 20)

    def test_deterministic_for_a_seed(self):
        kw = dict(shape=(5, 5, 5), b=2, num_img=20, a=1, seed=_fresh_seed())
        e0 = data.data_factory_wgn(**kw)
        e1 = data.data_factory_wgn(**kw)
        assert np.array_equal(e0.y, e1.y)
        assert np.array_equal(e0.x, e1.x)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

class TestDataFactoryDispatch:
    def test_wgn_routes_to_builder(self):
        kw = dict(shape=(4, 4, 4), b=2, num_img=10, a=1, seed=_fresh_seed())
        assert np.array_equal(data.data_factory('wgn', **kw).y,
                              data.data_factory_wgn(**kw).y)

    def test_bad_source_raises(self):
        with pytest.raises(ValueError, match="'wgn' or 'hcp'"):
            data.data_factory('nope')


# ---------------------------------------------------------------------------
# joblib.Memory cache decorator
# ---------------------------------------------------------------------------

class TestCacheDecorator:
    def test_miss_then_hit(self):
        # cache is outermost, so data_factory_wgn is itself the MemorizedFunc
        mf = data.data_factory_wgn
        assert hasattr(mf, 'check_call_in_cache')

        kw = dict(shape=(4, 4, 4), b=2, num_img=10, a=1, seed=_fresh_seed())
        assert not mf.check_call_in_cache(**kw)  # fresh seed -> not cached
        data.data_factory_wgn(**kw)              # compute + store
        assert mf.check_call_in_cache(**kw)      # now cached

    def test_cached_result_matches_compute(self):
        kw = dict(shape=(4, 4, 4), b=2, num_img=10, a=1, seed=_fresh_seed())
        first = data.data_factory_wgn(**kw)      # computed
        second = data.data_factory_wgn(**kw)     # served from cache
        assert np.array_equal(first.y, second.y)
        assert np.array_equal(first.x, second.x)


# ---------------------------------------------------------------------------
# Recorder decorator
# ---------------------------------------------------------------------------

class TestRecorderDecorator:
    def test_records_under_joblib_hash_on_miss(self):
        kw = dict(shape=(4, 4, 4), b=2, num_img=10, a=1, seed=_fresh_seed())
        # the args hash joblib keys the cache entry by
        args_id = data.data_factory_wgn._get_args_id(**kw)

        data.RECORDER.records.clear()
        data.data_factory_wgn(**kw)              # fresh seed -> miss -> records

        assert len(data.RECORDER.records) == 1
        # the record is keyed by joblib's args hash, so it lines up one-to-one
        # with the cached artifact on disk
        rec = data.RECORDER.records[args_id]
        assert rec['function'] == 'data_factory_wgn'
        assert set(rec['outputs']) == {'exp'}
        # inputs carry every (defaulted) build axis, including the seed key
        assert rec['inputs']['seed'] == kw['seed']
        assert rec['inputs']['b'] == 2
        assert 'time_sec' in rec
        assert rec['hash'] == args_id

    def test_cache_hit_does_not_record(self):
        # recorder nested inside the cache -> a hit returns without recording
        kw = dict(shape=(4, 4, 4), b=2, num_img=10, a=1, seed=_fresh_seed())
        data.data_factory_wgn(**kw)              # miss -> builds + records
        data.RECORDER.records.clear()
        data.data_factory_wgn(**kw)              # hit -> no build, no record
        assert data.RECORDER.records == {}

    def test_args_hash_reproduces_cache_key(self):
        # the recorder's key re-derives joblib's own cache key from the raw fn
        from glow._extra.benchmark.recorder import Recorder
        kw = dict(shape=(4, 4, 4), b=2, num_img=10, a=1, seed=_fresh_seed())
        mf = data.data_factory_wgn               # MemorizedFunc
        raw = mf.func.__wrapped__                # raw build fn the recorder wraps
        assert Recorder._args_hash(raw, (), kw) == mf._get_args_id(**kw)


# ---------------------------------------------------------------------------
# HCP builder (download + NIfTI search mocked)
# ---------------------------------------------------------------------------

class TestDataFactoryHCP:
    def test_builds_and_records_via_mocked_loader(self, monkeypatch, tmp_path):
        feats = ('fa', 'md')
        # the image-only experiment the mocked NIfTI search stands in for
        img = ExperimentImageOnly.from_gauss(shape=(4, 4, 4), b=len(feats),
                                             num_img=10, seed=0)

        seen = {}

        def fake_from_search(**kwargs):
            seen.update(kwargs)
            return img

        # mock the two heavy dependencies: the dataset dir and the NIfTI load
        monkeypatch.setattr(ExperimentImageOnly, 'from_search', fake_from_search)
        (tmp_path / 'brain_mask_space-MNI152NLin2009cAsym.nii.gz').write_bytes(b'')
        monkeypatch.setattr(hcp, 'ensure_hcp_data', lambda: tmp_path)

        data.RECORDER.records.clear()
        exp = data.data_factory_hcp(hcp_feats=feats, a=1, seed=_fresh_seed())

        # the design matrix was sampled onto the loaded images (a=1 + bias)
        assert exp.y.shape[0] == len(feats)
        assert exp.x.shape == (2, 10)

        # the builder selected the requested features and passed the brain mask
        assert tuple(seen['img_glob_dict']) == feats
        assert seen['sbj_regex'] == hcp.SBJ_REGEX
        assert seen['mask'].name.startswith('brain_mask_space-')

        # the call was recorded under the hcp builder's name
        assert len(data.RECORDER.records) == 1
        (rec,) = data.RECORDER.records.values()
        assert rec['function'] == 'data_factory_hcp'


# ---------------------------------------------------------------------------
# effect planter (shares data_factory's MEMORY / RECORDER, tested above)
# ---------------------------------------------------------------------------

class TestEffectFactory:
    def _clean_exp(self):
        return data.data_factory_wgn(shape=(6, 6, 6), b=2, num_img=20, a=1,
                                     seed=_fresh_seed())

    def test_plants_effect_on_support(self):
        exp = self._clean_exp()
        exp_eff, mask = data.effect_factory(
            exp, effect_llr=0.05, extenter_cls=ExtenterMinVar, n_vox=12, seed=0)
        # effect added in place: same shapes, mask over the spatial grid, y
        # changed, and the extenter grew exactly n_vox voxels
        assert exp_eff.y.shape == exp.y.shape
        assert mask.shape == exp.mask_idx.shape
        assert int(mask.sum()) == 12
        assert not np.array_equal(exp.y, exp_eff.y)

    def test_seed_from_exp_places_per_realization(self):
        # seed_from_exp derives the support seed from a hash of exp, so a fixed
        # config plants in a different place on different data...
        kw = dict(effect_llr=0.05, extenter_cls=ExtenterMinVar, n_vox=12,
                  seed_from_exp=True)
        _, mask0 = data.effect_factory(self._clean_exp(), **kw)
        _, mask1 = data.effect_factory(self._clean_exp(), **kw)
        assert not np.array_equal(mask0, mask1)
        # ...but it's a pure function of the data: identical across effect
        # strengths for one experiment (only the imposed offset changes)
        exp = self._clean_exp()
        _, m_weak = data.effect_factory(exp, **kw)
        _, m_strong = data.effect_factory(exp, **{**kw, 'effect_llr': 0.3})
        np.testing.assert_array_equal(m_weak, m_strong)

    def test_requires_exactly_one_seed_spec(self):
        # seed XOR seed_from_exp: neither and both are errors
        exp = self._clean_exp()
        base = dict(effect_llr=0.05, extenter_cls=ExtenterMinVar, n_vox=10)
        with pytest.raises(ValueError):
            data.effect_factory(exp, **base)                       # neither
        with pytest.raises(ValueError):
            data.effect_factory(exp, seed=0, seed_from_exp=True, **base)  # both

    def test_records_outputs_under_joblib_hash(self):
        exp = self._clean_exp()
        kw = dict(effect_llr=0.05, extenter_cls=ExtenterMinVar, n_vox=10, seed=0)
        args_id = data.effect_factory._get_args_id(exp, **kw)

        data.RECORDER.records.clear()   # drop the clean-build record above
        data.effect_factory(exp, **kw)  # fresh exp -> miss -> records

        assert len(data.RECORDER.records) == 1
        rec = data.RECORDER.records[args_id]
        assert rec['function'] == 'effect_factory'
        # output_name_list unpacks the (exp, mask) return into two outputs
        assert set(rec['outputs']) == {'exp', 'mask'}

    def test_miss_then_hit(self):
        exp = self._clean_exp()
        kw = dict(effect_llr=0.05, extenter_cls=ExtenterMinVar, n_vox=10, seed=0)
        assert not data.effect_factory.check_call_in_cache(exp, **kw)
        data.effect_factory(exp, **kw)
        assert data.effect_factory.check_call_in_cache(exp, **kw)
